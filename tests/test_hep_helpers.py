"""
Tests for HEP significance helper math.

_isotonic_monotone and _hep_profile_step cannot be imported from 01_hep.py
(module-level argparse + pipeline execution). Their logic is self-contained
and re-implemented here for unit testing, keeping them in sync with the source.
The underlying math (evaluate_profile_likelihood_discovery, gaussian_filter1d,
IsotonicRegression) is imported from the same libraries used by the real code.

Coverage:
  - _isotonic_monotone: non-decreasing, clips negatives, preserves rough magnitude
  - _hep_significance_step: return keys, non-negative values, monotonicity enforcement
  - _hep_profile_step: finite result, perm_mask zeroes bins, fluctuation scaling
  - _rebin_from_starts: sum consistency
"""
import os
import sys

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lib.sigma import evaluate_profile_likelihood_discovery, evaluate_significance
from lib.smoothing import apply_adaptive_tail_rebin, rebin_with_starts


# ── Local re-implementation of helpers (mirrors 01_hep.py exactly) ──────────

_PL_SMOOTH_SIGMA = 6.0
_isotonic_regressor = IsotonicRegression(increasing=True)


def _isotonic_monotone(values: list) -> list:
    arr = np.clip(np.asarray(values, dtype=float), 0.0, None)
    arr = gaussian_filter1d(arr, sigma=_PL_SMOOTH_SIGMA, mode="nearest")
    arr = np.clip(arr, 0.0, None)
    return list(_isotonic_regressor.fit_transform(np.arange(len(arr)), arr))


def _hep_profile_step(
    signal_rate,
    background_rate,
    background_uncertainty_frac,
    factor,
    fluctuation_sigma=0.0,
    signal_norm_frac=0.0,
    perm_mask=None,
):
    signal_events = factor * signal_rate * (1.0 + fluctuation_sigma * signal_norm_frac)
    background_events = factor * background_rate
    if perm_mask is not None:
        background_events = np.where(perm_mask, background_events, 0.0)
        signal_events = np.where(perm_mask, signal_events, 0.0)
    return float(evaluate_profile_likelihood_discovery(
        signal_events, background_events, background_uncertainty=background_uncertainty_frac,
    ))


def _global(arr):
    return float(np.sqrt(np.sum(np.power(
        np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), 2,
    ))))


def _hep_significance_step(
    signal_rate,
    background_rate,
    bkg_error_rate,
    factor,
    detection_requirement,
    signal_uncertainty,
    detection_threshold,
    prev_starts,
    last_asimov,
    adaptive_rebin_config,
):
    signal_events = factor * signal_rate
    background_events = factor * background_rate
    bkg_error_events = factor * bkg_error_rate
    detection_signal = signal_events * (1.0 - detection_requirement * signal_uncertainty)

    detectable = detection_signal >= detection_threshold
    s_nr = np.where(detectable, signal_events, 0.0)
    b_nr = np.where(detectable, background_events, 0.0)
    u_nr = np.where(detectable, bkg_error_events, 0.0)

    gaussian_no_rebin = _global(evaluate_significance(s_nr, b_nr, background_uncertainty=u_nr, type="gaussian"))
    asimov_no_rebin   = _global(evaluate_significance(s_nr, b_nr, background_uncertainty=u_nr, type="asimov"))

    s_cand, b_cand, u_cand, starts_cand = apply_adaptive_tail_rebin(
        signal_events, background_events, bkg_error_events, detection_signal, adaptive_rebin_config,
    )

    s_in, b_in, u_in, starts_out = s_cand, b_cand, u_cand, starts_cand
    if prev_starts.size > 0 and last_asimov is not None:
        starts_prev = prev_starts
        s_prev = rebin_with_starts(signal_events, starts_prev)
        b_prev = rebin_with_starts(background_events, starts_prev)
        u_prev = rebin_with_starts(bkg_error_events, starts_prev)
        asimov_cand_g = _global(evaluate_significance(
            s_cand, b_cand, background_uncertainty=u_cand, type="asimov",
        ))
        if asimov_cand_g < last_asimov:
            s_in, b_in, u_in, starts_out = s_prev, b_prev, u_prev, starts_prev

    gaussian_rebinned = _global(evaluate_significance(s_in, b_in, background_uncertainty=u_in, type="gaussian"))
    asimov_rebinned   = _global(evaluate_significance(s_in, b_in, background_uncertainty=u_in, type="asimov"))

    return {
        "gaussian_no_rebin": gaussian_no_rebin,
        "asimov_no_rebin":   asimov_no_rebin,
        "gaussian_rebinned": gaussian_rebinned,
        "asimov_rebinned":   asimov_rebinned,
        "starts": np.asarray(starts_out, dtype=int).copy(),
        "n_bins": int(len(starts_out)),
    }


# ── _DISABLED_adaptive_config helper ─────────────────────────────────────────

def _no_rebin_config():
    return {"enabled": False}


# ── _isotonic_monotone tests ─────────────────────────────────────────────────

class TestIsotonicMonotone:
    def test_output_non_decreasing(self):
        values = [3.0, 1.0, 5.0, 2.0, 4.0, 6.0]
        result = _isotonic_monotone(values)
        for i in range(len(result) - 1):
            assert result[i] <= result[i + 1] + 1e-10, f"non-monotone at index {i}"

    def test_already_monotone_unchanged(self):
        values = [0.0, 1.0, 2.0, 3.0, 4.0]
        result = _isotonic_monotone(values)
        assert all(r >= 0.0 for r in result)
        # After Gaussian smoothing of already-monotone, still non-decreasing
        for i in range(len(result) - 1):
            assert result[i] <= result[i + 1] + 1e-10

    def test_negative_values_clamped_to_zero(self):
        values = [-5.0, -1.0, 2.0, 3.0]
        result = _isotonic_monotone(values)
        assert all(r >= 0.0 for r in result)

    def test_all_zeros_returns_zeros(self):
        result = _isotonic_monotone([0.0] * 10)
        assert all(abs(r) < 1e-10 for r in result)

    def test_single_element(self):
        result = _isotonic_monotone([3.7])
        assert len(result) == 1
        assert result[0] >= 0.0

    def test_length_preserved(self):
        for n in [5, 10, 30, 100]:
            values = list(np.random.default_rng(n).normal(2.0, 1.0, n))
            result = _isotonic_monotone(values)
            assert len(result) == n

    def test_monotone_after_random_input(self):
        rng = np.random.default_rng(42)
        for _ in range(10):
            values = list(rng.uniform(0, 5, 50))
            result = _isotonic_monotone(values)
            arr = np.asarray(result)
            assert np.all(np.diff(arr) >= -1e-10), "non-monotone result"

    def test_exposure_curve_shape(self):
        # Simulate a significance-vs-exposure curve with oscillations at low exposure.
        n = 40
        t = np.linspace(0, 3, n)
        values = list(np.sqrt(t) + 0.3 * np.sin(10 * t))
        result = _isotonic_monotone(values)
        assert result[-1] >= result[0]


# ── _hep_profile_step tests ───────────────────────────────────────────────────

class TestHepProfileStep:
    def _rates(self, n=20, seed=0):
        rng = np.random.default_rng(seed)
        sig = np.abs(rng.normal(0.5, 0.1, n))
        bkg = np.abs(rng.normal(5.0, 0.5, n))
        return sig, bkg

    def test_returns_finite_positive(self):
        s, b = self._rates()
        result = _hep_profile_step(s, b, 0.02, factor=10.0)
        assert np.isfinite(result)
        assert result >= 0.0

    def test_grows_with_factor(self):
        s, b = self._rates()
        z1 = _hep_profile_step(s, b, 0.02, factor=1.0)
        z10 = _hep_profile_step(s, b, 0.02, factor=10.0)
        assert z10 > z1

    def test_perm_mask_all_false_gives_zero(self):
        s, b = self._rates()
        mask = np.zeros(len(s), dtype=bool)
        result = _hep_profile_step(s, b, 0.02, factor=10.0, perm_mask=mask)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_perm_mask_all_true_same_as_no_mask(self):
        s, b = self._rates()
        mask = np.ones(len(s), dtype=bool)
        z_no_mask = _hep_profile_step(s, b, 0.02, factor=10.0)
        z_masked  = _hep_profile_step(s, b, 0.02, factor=10.0, perm_mask=mask)
        assert z_no_mask == pytest.approx(z_masked, rel=1e-10)

    def test_perm_mask_partial_reduces_significance(self):
        s, b = self._rates(n=20)
        mask_full = np.ones(20, dtype=bool)
        mask_half = np.zeros(20, dtype=bool)
        mask_half[:10] = True
        z_full = _hep_profile_step(s, b, 0.02, factor=10.0, perm_mask=mask_full)
        z_half = _hep_profile_step(s, b, 0.02, factor=10.0, perm_mask=mask_half)
        assert z_full >= z_half

    def test_positive_fluctuation_increases_signal(self):
        s, b = self._rates()
        z_nominal  = _hep_profile_step(s, b, 0.02, factor=10.0, fluctuation_sigma=0.0, signal_norm_frac=0.10)
        z_plus_one = _hep_profile_step(s, b, 0.02, factor=10.0, fluctuation_sigma=1.0, signal_norm_frac=0.10)
        assert z_plus_one > z_nominal

    def test_negative_fluctuation_decreases_signal(self):
        s, b = self._rates()
        z_nominal   = _hep_profile_step(s, b, 0.02, factor=10.0, fluctuation_sigma=0.0,  signal_norm_frac=0.10)
        z_minus_one = _hep_profile_step(s, b, 0.02, factor=10.0, fluctuation_sigma=-1.0, signal_norm_frac=0.10)
        assert z_minus_one < z_nominal

    def test_zero_signal_returns_zero(self):
        s = np.zeros(10)
        b = np.ones(10) * 5.0
        result = _hep_profile_step(s, b, 0.02, factor=10.0)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_bkg_uncertainty_reduces_significance(self):
        s, b = self._rates()
        z_no_unc   = _hep_profile_step(s, b, 0.0,  factor=10.0)
        z_with_unc = _hep_profile_step(s, b, 0.10, factor=10.0)
        assert z_no_unc >= z_with_unc


# ── _hep_significance_step tests ─────────────────────────────────────────────

class TestHepSignificanceStep:
    def _setup(self, n=25, seed=7):
        rng = np.random.default_rng(seed)
        sig = np.abs(rng.normal(0.5, 0.1, n))
        bkg = np.abs(rng.normal(5.0, 0.5, n))
        unc = bkg * 0.02
        return sig, bkg, unc

    def test_returns_expected_keys(self):
        s, b, u = self._setup()
        result = _hep_significance_step(
            s, b, u,
            factor=10.0, detection_requirement=0.0, signal_uncertainty=0.0,
            detection_threshold=0.0, prev_starts=np.array([]), last_asimov=None,
            adaptive_rebin_config=_no_rebin_config(),
        )
        assert set(result.keys()) == {
            "gaussian_no_rebin", "asimov_no_rebin",
            "gaussian_rebinned", "asimov_rebinned",
            "starts", "n_bins",
        }

    def test_all_values_non_negative(self):
        s, b, u = self._setup()
        result = _hep_significance_step(
            s, b, u,
            factor=10.0, detection_requirement=0.0, signal_uncertainty=0.0,
            detection_threshold=0.0, prev_starts=np.array([]), last_asimov=None,
            adaptive_rebin_config=_no_rebin_config(),
        )
        assert result["gaussian_no_rebin"] >= 0.0
        assert result["asimov_no_rebin"] >= 0.0
        assert result["gaussian_rebinned"] >= 0.0
        assert result["asimov_rebinned"] >= 0.0

    def test_significance_grows_with_factor(self):
        s, b, u = self._setup()
        kwargs = dict(
            detection_requirement=0.0, signal_uncertainty=0.0,
            detection_threshold=0.0, prev_starts=np.array([]), last_asimov=None,
            adaptive_rebin_config=_no_rebin_config(),
        )
        r1 = _hep_significance_step(s, b, u, factor=1.0, **kwargs)
        r10 = _hep_significance_step(s, b, u, factor=10.0, **kwargs)
        assert r10["gaussian_no_rebin"] > r1["gaussian_no_rebin"]

    def test_monotonicity_enforcement_keeps_prev_starts_when_lower(self):
        # Give a prev_starts with a good existing rebin, and a config that produces
        # a worse candidate. The step should fall back to prev_starts.
        s, b, u = self._setup()
        # prev_starts merges all into one bin → high significance at low exposure
        prev_starts = np.array([0])
        # Compute what that gives at factor=1 to set last_asimov high.
        s_prev = rebin_with_starts(s * 1.0, prev_starts)
        b_prev = rebin_with_starts(b * 1.0, prev_starts)
        u_prev = rebin_with_starts(u * 1.0, prev_starts)
        very_high_asimov = _global(evaluate_significance(s_prev, b_prev, background_uncertainty=u_prev, type="asimov")) * 1000.0

        result = _hep_significance_step(
            s, b, u,
            factor=1.0, detection_requirement=0.0, signal_uncertainty=0.0,
            detection_threshold=0.0,
            prev_starts=prev_starts, last_asimov=very_high_asimov,
            adaptive_rebin_config=_no_rebin_config(),
        )
        # Since last_asimov is artificially very high, step should use prev_starts
        np.testing.assert_array_equal(result["starts"], prev_starts)

    def test_detection_threshold_masks_low_signal_bins(self):
        n = 10
        s = np.ones(n) * 0.1
        b = np.ones(n) * 5.0
        u = np.ones(n) * 0.1
        # High detection threshold → all bins masked
        r_high = _hep_significance_step(
            s, b, u,
            factor=1.0, detection_requirement=0.0, signal_uncertainty=0.0,
            detection_threshold=100.0, prev_starts=np.array([]), last_asimov=None,
            adaptive_rebin_config=_no_rebin_config(),
        )
        assert r_high["gaussian_no_rebin"] == pytest.approx(0.0, abs=1e-9)


# ── rebin_with_starts integrity ───────────────────────────────────────────────

class TestRebinWithStarts:
    def test_sum_preserved(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        starts = np.array([0, 2, 4])
        result = rebin_with_starts(values, starts)
        assert np.sum(result) == pytest.approx(np.sum(values), rel=1e-10)
        assert len(result) == 3

    def test_identity_starts(self):
        values = np.array([1.0, 2.0, 3.0])
        starts = np.arange(3)
        result = rebin_with_starts(values, starts)
        np.testing.assert_array_equal(result, values)

    def test_single_bin(self):
        values = np.array([1.0, 2.0, 3.0, 4.0])
        result = rebin_with_starts(values, np.array([0]))
        assert result[0] == pytest.approx(10.0)

    def test_nan_replaced_with_zero(self):
        values = np.array([np.nan, 2.0, np.nan])
        result = rebin_with_starts(values, np.array([0, 1]))
        assert np.all(np.isfinite(result))
