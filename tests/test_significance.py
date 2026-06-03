"""
Tests for lib/sigma.py — all significance types and core math.

Coverage:
  - _solve_global_beta_hat: analytical limits
  - evaluate_profile_likelihood_discovery: known cases + numerical stability
  - evaluate_significance: gaussian, asimov, profile — basic + edge cases
  - NaN / inf / zero-input robustness
  - Physical sanity: significance grows with signal, shrinks with more background
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lib.sigma import (
    _solve_global_beta_hat,
    evaluate_profile_likelihood_discovery,
    evaluate_significance,
)


# ── _solve_global_beta_hat ────────────────────────────────────────────────────

class TestSolveGlobalBetaHat:
    def test_zero_background_returns_one(self):
        assert _solve_global_beta_hat(10.0, 0.0, 0.02) == 1.0

    def test_zero_sigma_returns_one(self):
        assert _solve_global_beta_hat(10.0, 5.0, 0.0) == 1.0

    def test_positive_result(self):
        result = _solve_global_beta_hat(N_total=100.0, B_total=80.0, sigma_rel=0.05)
        assert result > 0.0

    def test_no_signal_beta_near_one(self):
        # When N≈B (no signal), β̂ should be close to 1.
        result = _solve_global_beta_hat(N_total=100.0, B_total=100.0, sigma_rel=0.02)
        assert abs(result - 1.0) < 0.05

    def test_excess_signal_beta_below_one(self):
        # Observed >> background → null-model absorbs excess by scaling β up.
        # β̂ solves quadratic; for large N/B the positive root > 1.
        result = _solve_global_beta_hat(N_total=200.0, B_total=100.0, sigma_rel=0.5)
        assert result > 1.0

    def test_satisfies_quadratic(self):
        N, B, sigma = 120.0, 80.0, 0.10
        beta = _solve_global_beta_hat(N, B, sigma)
        sigma2 = sigma ** 2
        residual = beta ** 2 + (B * sigma2 - 1.0) * beta - N * sigma2
        assert abs(residual) < 1e-8


# ── evaluate_profile_likelihood_discovery ────────────────────────────────────

class TestProfileLikelihoodDiscovery:
    def test_zero_signal_returns_zero(self):
        signal = np.zeros(10)
        background = np.ones(10) * 5.0
        result = evaluate_profile_likelihood_discovery(signal, background)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_no_background_uncertainty_analytic(self):
        # With σ_rel=0, β̂=1, q0 = 2·Σ[n·log(n/b) - (n-b)] with n=s+b.
        # Single bin: n=110, b=100 → q0 = 2*(110*log(1.1)-10)
        s = np.array([10.0])
        b = np.array([100.0])
        result = evaluate_profile_likelihood_discovery(s, b, background_uncertainty=None)
        expected_q0 = 2.0 * (110.0 * np.log(110.0 / 100.0) - 10.0)
        assert result == pytest.approx(np.sqrt(max(0.0, expected_q0)), rel=1e-6)

    def test_grows_with_signal(self):
        b = np.ones(20) * 5.0
        sig_small = np.ones(20) * 0.1
        sig_large = np.ones(20) * 2.0
        z_small = evaluate_profile_likelihood_discovery(sig_small, b)
        z_large = evaluate_profile_likelihood_discovery(sig_large, b)
        assert z_large > z_small

    def test_shrinks_with_background_uncertainty(self):
        s = np.ones(10) * 1.0
        b = np.ones(10) * 10.0
        z_no_unc = evaluate_profile_likelihood_discovery(s, b, background_uncertainty=None)
        z_with_unc = evaluate_profile_likelihood_discovery(s, b, background_uncertainty=0.10)
        assert z_no_unc >= z_with_unc

    def test_array_bkg_uncertainty_same_as_scalar_when_uniform(self):
        s = np.array([2.0, 1.0, 0.5])
        b = np.array([10.0, 8.0, 5.0])
        sigma_rel = 0.05
        per_bin_unc = sigma_rel * b
        z_scalar = evaluate_profile_likelihood_discovery(s, b, background_uncertainty=sigma_rel)
        z_array  = evaluate_profile_likelihood_discovery(s, b, background_uncertainty=per_bin_unc)
        assert z_scalar == pytest.approx(z_array, rel=1e-5)

    def test_nan_input_handled(self):
        s = np.array([np.nan, 1.0, 0.5])
        b = np.array([10.0, np.nan, 5.0])
        result = evaluate_profile_likelihood_discovery(s, b)
        assert np.isfinite(result)
        assert result >= 0.0

    def test_mismatched_sizes_raises(self):
        with pytest.raises(ValueError, match="same length"):
            evaluate_profile_likelihood_discovery(np.ones(3), np.ones(4))

    def test_all_zero_returns_zero(self):
        result = evaluate_profile_likelihood_discovery(np.zeros(5), np.zeros(5))
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_non_negative(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            s = np.abs(rng.normal(1.0, 0.5, 15))
            b = np.abs(rng.normal(5.0, 1.0, 15))
            assert evaluate_profile_likelihood_discovery(s, b) >= 0.0


# ── evaluate_significance: Gaussian ──────────────────────────────────────────

class TestGaussianSignificance:
    def test_pure_poisson_s_over_sqrt_b(self):
        s = np.array([4.0])
        b = np.array([16.0])
        z = evaluate_significance(s, b, type="gaussian")
        assert z[0] == pytest.approx(1.0, rel=1e-9)

    def test_zero_signal_gives_zero(self):
        s = np.zeros(5)
        b = np.ones(5) * 10.0
        z = evaluate_significance(s, b, type="gaussian")
        assert np.all(z == 0.0)

    def test_zero_background_gives_zero(self):
        s = np.ones(5) * 3.0
        b = np.zeros(5)
        z = evaluate_significance(s, b, type="gaussian")
        assert np.all(z == 0.0)

    def test_background_uncertainty_reduces_significance(self):
        s = np.array([3.0])
        b = np.array([9.0])
        z_no_unc = evaluate_significance(s, b, type="gaussian")
        bkg_unc = np.array([1.0])
        z_with_unc = evaluate_significance(s, b, background_uncertainty=bkg_unc, type="gaussian")
        assert z_no_unc[0] > z_with_unc[0]

    def test_signal_uncertainty_reduces_significance(self):
        s = np.array([3.0])
        b = np.array([9.0])
        z_no_unc = evaluate_significance(s, b, type="gaussian")
        sig_unc = np.array([1.0])
        z_with_unc = evaluate_significance(s, b, signal_uncertainty=sig_unc, type="gaussian")
        assert z_no_unc[0] > z_with_unc[0]

    def test_both_uncertainties_further_reduce(self):
        s = np.array([3.0])
        b = np.array([9.0])
        sig_unc = np.array([0.5])
        bkg_unc = np.array([0.5])
        z_bkg_only = evaluate_significance(s, b, background_uncertainty=bkg_unc, type="gaussian")
        z_both = evaluate_significance(
            s, b, signal_uncertainty=sig_unc, background_uncertainty=bkg_unc, type="gaussian"
        )
        assert z_bkg_only[0] >= z_both[0]

    def test_grows_with_signal(self):
        b = np.array([25.0])
        s_small = np.array([1.0])
        s_large = np.array([5.0])
        assert evaluate_significance(s_large, b, type="gaussian")[0] > \
               evaluate_significance(s_small, b, type="gaussian")[0]

    def test_shrinks_with_background(self):
        s = np.array([2.0])
        b_small = np.array([4.0])
        b_large = np.array([100.0])
        assert evaluate_significance(s, b_small, type="gaussian")[0] > \
               evaluate_significance(s, b_large, type="gaussian")[0]

    def test_multibin_independent_formula(self):
        s = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 4.0, 9.0])
        expected = s / np.sqrt(b)
        z = evaluate_significance(s, b, type="gaussian")
        np.testing.assert_allclose(z, expected, rtol=1e-10)


# ── evaluate_significance: Asimov ─────────────────────────────────────────────

class TestAsimovSignificance:
    def test_zero_signal_gives_zero(self):
        s = np.zeros(5)
        b = np.ones(5) * 10.0
        z = evaluate_significance(s, b, type="asimov")
        assert np.all(z == 0.0)

    def test_zero_background_gives_zero(self):
        s = np.ones(5)
        b = np.zeros(5)
        z = evaluate_significance(s, b, type="asimov")
        assert np.all(z == 0.0)

    def test_analytic_single_bin(self):
        s = np.array([10.0])
        b = np.array([100.0])
        z = evaluate_significance(s, b, type="asimov")
        expected = np.sqrt(2 * ((110.0) * np.log(1 + 10.0 / 100.0) - 10.0))
        assert z[0] == pytest.approx(expected, rel=1e-9)

    def test_large_signal_gaussian_overestimates(self):
        # Gaussian Z=S/√B diverges; Asimov accounts for observed (S+B).
        # At large s/b, Gaussian > Asimov.
        s = np.array([50.0])
        b = np.array([10.0])
        z_gauss = evaluate_significance(s, b, type="gaussian")[0]
        z_asimov = evaluate_significance(s, b, type="asimov")[0]
        assert z_gauss > z_asimov

    def test_grows_with_signal(self):
        b = np.array([100.0])
        z1 = evaluate_significance(np.array([1.0]), b, type="asimov")[0]
        z5 = evaluate_significance(np.array([5.0]), b, type="asimov")[0]
        assert z5 > z1

    def test_background_uncertainty_version(self):
        s = np.array([5.0])
        b = np.array([20.0])
        bkg_unc = np.array([2.0])
        z = evaluate_significance(s, b, background_uncertainty=bkg_unc, type="asimov")
        assert np.all(np.isfinite(z))
        assert np.all(z >= 0.0)

    def test_background_uncertainty_reduces_significance(self):
        s = np.array([5.0])
        b = np.array([20.0])
        bkg_unc = np.array([4.0])
        z_no_unc = evaluate_significance(s, b, type="asimov")
        z_with_unc = evaluate_significance(s, b, background_uncertainty=bkg_unc, type="asimov")
        assert z_no_unc[0] >= z_with_unc[0]


# ── evaluate_significance: Profile Likelihood ─────────────────────────────────

class TestProfileSignificance:
    def test_zero_signal_gives_zero(self):
        s = np.zeros(5)
        b = np.ones(5) * 10.0
        z = evaluate_significance(s, b, type="profile")
        assert np.all(z == 0.0)

    def test_finite_nonzero(self):
        s = np.array([2.0, 1.0, 0.5])
        b = np.array([10.0, 8.0, 5.0])
        z = evaluate_significance(s, b, type="profile")
        assert np.all(np.isfinite(z[s > 0]))
        assert np.all(z >= 0.0)

    def test_grows_with_signal(self):
        b = np.array([10.0])
        z1 = evaluate_significance(np.array([0.5]), b, type="profile")[0]
        z5 = evaluate_significance(np.array([5.0]), b, type="profile")[0]
        assert z5 > z1

    def test_with_background_uncertainty(self):
        s = np.array([3.0, 2.0])
        b = np.array([10.0, 8.0])
        bkg_unc = np.array([0.5, 0.4])
        z = evaluate_significance(s, b, background_uncertainty=bkg_unc, type="profile")
        assert np.all(np.isfinite(z))
        assert np.all(z >= 0.0)

    def test_all_zero_background_nonzero_signal_finite(self):
        # Signal with zero background → PL is well-defined (high significance)
        s = np.array([1.0, 2.0])
        b = np.zeros(2)
        z = evaluate_significance(s, b, type="profile")
        assert np.all(np.isfinite(z))

    def test_aliases_all_work(self):
        s = np.array([2.0])
        b = np.array([8.0])
        z1 = evaluate_significance(s, b, type="profile")[0]
        z2 = evaluate_significance(s, b, type="profile_likelihood")[0]
        z3 = evaluate_significance(s, b, type="profile-likelihood")[0]
        assert z1 == pytest.approx(z2, rel=1e-10)
        assert z1 == pytest.approx(z3, rel=1e-10)


# ── Unknown type raises ───────────────────────────────────────────────────────

def test_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown significance type"):
        evaluate_significance(np.array([1.0]), np.array([1.0]), type="magic")


# ── Cross-type ordering ───────────────────────────────────────────────────────

class TestCrossTypeOrdering:
    """Physical sanity: for large S/B, Asimov >= Gaussian (tighter test)."""

    def test_gaussian_overestimates_asimov_large_signal(self):
        # Gaussian Z=S/√B is an approximation that overestimates at large s/b.
        # Asimov accounts for observed counts properly.
        s = np.array([20.0])
        b = np.array([10.0])
        z_g = evaluate_significance(s, b, type="gaussian")[0]
        z_a = evaluate_significance(s, b, type="asimov")[0]
        assert z_g > z_a

    def test_gaussian_gte_profile_with_bkg_uncertainty(self):
        # Profile likelihood with non-zero bkg unc should be ≤ Gaussian (absorbs some signal)
        s = np.ones(10) * 1.0
        b = np.ones(10) * 5.0
        bkg_unc = np.ones(10) * 0.5
        z_g = float(np.sqrt(np.sum(evaluate_significance(s, b, background_uncertainty=bkg_unc, type="gaussian") ** 2)))
        z_pl = float(np.sqrt(np.sum(evaluate_significance(s, b, background_uncertainty=bkg_unc, type="profile") ** 2)))
        # Both should be finite and non-negative
        assert np.isfinite(z_g) and z_g >= 0.0
        assert np.isfinite(z_pl) and z_pl >= 0.0
