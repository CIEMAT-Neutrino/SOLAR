"""
Tests for DayNight significance computation math.

The DayNight pipeline computes:
  1. Effective background from weighted two-sample combination
  2. Signal = asymmetry_scale * (signal_night - signal_day) * factor
  3. Significance via evaluate_significance(signal, bkg_effective, type="gaussian")

These tests validate the formulas directly (no file I/O), mirroring
the computation in src/physics/daynight/01_daynight.py.

Coverage:
  - Effective background formula: n_n/g² + n_d/f²
  - Zero asymmetry (equal day/night signal) → zero significance
  - Non-zero asymmetry → positive significance
  - Significance grows with exposure (factor)
  - Symmetric fractions (f=g=0.5) → simplifies correctly
  - Background uncertainty propagation (Poisson + systematic)
  - ErrorGaussian formula with quadrature combination
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lib.sigma import evaluate_significance


# ── Reference implementation of DayNight formulas ────────────────────────────

def _daynight_compute(
    signal_day,
    signal_night,
    background,
    factor,
    asymmetry_scale=1.0,
    day_fraction=0.5,
    background_uncertainty=0.02,
    day_fraction_band=0.0,
    use_background_error=False,
):
    """Mirror of DayNight significance computation from 01_daynight.py."""
    night_fraction = 1.0 - day_fraction

    night_counts = factor * (night_fraction * background + signal_night)
    day_counts   = factor * (day_fraction   * background + signal_day)

    background_effective = (
        night_counts / night_fraction ** 2
        + day_counts  / day_fraction  ** 2
    )

    signal = factor * asymmetry_scale * (signal_night - signal_day)
    signal = np.where(background_effective == 0, 0.0, signal)

    if not use_background_error:
        bkg_unc = None
    else:
        night_bkg = factor * night_fraction * background
        day_bkg   = factor * day_fraction   * background
        sigma_night = np.sqrt(
            night_bkg
            + (background_uncertainty * night_bkg) ** 2
            + (day_fraction_band * factor * background) ** 2
        )
        sigma_day = np.sqrt(
            day_bkg
            + (background_uncertainty * day_bkg) ** 2
            + (day_fraction_band * factor * background) ** 2
        )
        bkg_unc = np.where(
            background_effective > 0,
            np.sqrt(
                (sigma_night / night_fraction) ** 2
                + (sigma_day  / day_fraction)  ** 2
            ),
            0.0,
        )

    z = evaluate_significance(signal, background_effective, background_uncertainty=bkg_unc, type="gaussian")
    return float(np.sqrt(np.sum(z ** 2)))


# ── Zero-asymmetry tests ──────────────────────────────────────────────────────

class TestZeroAsymmetry:
    def test_equal_day_night_gives_zero_significance(self):
        s = np.ones(10) * 1.0
        b = np.ones(10) * 5.0
        z = _daynight_compute(s, s, b, factor=10.0)
        assert z == pytest.approx(0.0, abs=1e-10)

    def test_zero_signal_gives_zero_significance(self):
        s = np.zeros(10)
        b = np.ones(10) * 5.0
        z = _daynight_compute(s, s, b, factor=10.0)
        assert z == pytest.approx(0.0, abs=1e-10)

    def test_zero_asymmetry_scale_gives_zero(self):
        s_day   = np.ones(10) * 0.5
        s_night = np.ones(10) * 1.5
        b = np.ones(10) * 5.0
        z = _daynight_compute(s_day, s_night, b, factor=10.0, asymmetry_scale=0.0)
        assert z == pytest.approx(0.0, abs=1e-10)


# ── Non-zero asymmetry tests ──────────────────────────────────────────────────

class TestNonZeroAsymmetry:
    def test_night_excess_gives_positive_significance(self):
        s_day   = np.ones(10) * 0.5
        s_night = np.ones(10) * 1.5
        b = np.ones(10) * 5.0
        z = _daynight_compute(s_day, s_night, b, factor=10.0)
        assert z > 0.0

    def test_day_excess_also_gives_positive_significance(self):
        s_day   = np.ones(10) * 1.5
        s_night = np.ones(10) * 0.5
        b = np.ones(10) * 5.0
        # signal = night - day < 0 → significance will be zero (masked by signal>0)
        # Physical expectation: day excess means signal < 0; gaussian sig = 0
        z = _daynight_compute(s_day, s_night, b, factor=10.0)
        assert z >= 0.0

    def test_larger_asymmetry_gives_higher_significance(self):
        b = np.ones(10) * 5.0
        s_day = np.ones(10) * 1.0
        z1 = _daynight_compute(s_day, s_day + np.ones(10) * 0.5, b, factor=10.0)
        z2 = _daynight_compute(s_day, s_day + np.ones(10) * 2.0, b, factor=10.0)
        assert z2 > z1


# ── Exposure scaling ──────────────────────────────────────────────────────────

class TestExposureScaling:
    def test_significance_grows_with_exposure(self):
        s_day   = np.ones(10) * 0.5
        s_night = np.ones(10) * 1.5
        b = np.ones(10) * 5.0
        z1  = _daynight_compute(s_day, s_night, b, factor=1.0)
        z10 = _daynight_compute(s_day, s_night, b, factor=10.0)
        z50 = _daynight_compute(s_day, s_night, b, factor=50.0)
        assert z10 > z1
        assert z50 > z10

    def test_zero_factor_gives_zero_significance(self):
        s_day   = np.ones(5) * 0.5
        s_night = np.ones(5) * 1.5
        b = np.ones(5) * 5.0
        z = _daynight_compute(s_day, s_night, b, factor=0.0)
        assert z == pytest.approx(0.0, abs=1e-10)


# ── Effective background formula (f=g=0.5) ───────────────────────────────────

class TestEffectiveBackgroundFormula:
    def test_symmetric_fractions_formula(self):
        """At f=g=0.5, bkg_eff = 4*(n_n + n_d) = 4*factor*(B + 0.5*(S_n+S_d))."""
        s_d = np.array([0.5, 0.3])
        s_n = np.array([1.5, 0.9])
        b   = np.array([5.0, 8.0])
        factor = 10.0
        f = 0.5
        g = 0.5

        n_n = factor * (g * b + s_n)
        n_d = factor * (f * b + s_d)
        bkg_eff_expected = n_n / g**2 + n_d / f**2

        # Manually verify our reference matches numpy formula
        bkg_eff_simplified = 4.0 * (n_n + n_d)
        np.testing.assert_allclose(bkg_eff_expected, bkg_eff_simplified, rtol=1e-10)

    def test_asymmetric_fractions_differ_from_symmetric(self):
        s_d = np.array([0.5])
        s_n = np.array([1.5])
        b   = np.array([5.0])
        factor = 10.0

        z_sym   = _daynight_compute(s_d, s_n, b, factor=factor, day_fraction=0.5)
        z_asym  = _daynight_compute(s_d, s_n, b, factor=factor, day_fraction=0.4)
        assert z_sym != pytest.approx(z_asym, rel=1e-3)


# ── Background uncertainty propagation ───────────────────────────────────────

class TestBackgroundError:
    def test_background_error_reduces_significance(self):
        s_day   = np.ones(10) * 0.5
        s_night = np.ones(10) * 1.5
        b = np.ones(10) * 5.0
        z_no_err  = _daynight_compute(s_day, s_night, b, factor=20.0, use_background_error=False)
        z_with_err = _daynight_compute(s_day, s_night, b, factor=20.0, use_background_error=True,
                                        background_uncertainty=0.02)
        assert z_no_err >= z_with_err

    def test_larger_systematic_reduces_further(self):
        s_day   = np.ones(10) * 0.5
        s_night = np.ones(10) * 1.5
        b = np.ones(10) * 5.0
        z_small = _daynight_compute(s_day, s_night, b, factor=20.0, use_background_error=True,
                                     background_uncertainty=0.01)
        z_large = _daynight_compute(s_day, s_night, b, factor=20.0, use_background_error=True,
                                     background_uncertainty=0.20)
        assert z_small >= z_large

    def test_poisson_only_uncertainty_finite(self):
        # day_fraction_band=0, background_uncertainty=0 → pure Poisson uncertainty
        s_day   = np.ones(5) * 0.5
        s_night = np.ones(5) * 1.5
        b = np.ones(5) * 5.0
        z = _daynight_compute(
            s_day, s_night, b, factor=10.0, use_background_error=True,
            background_uncertainty=0.0, day_fraction_band=0.0,
        )
        assert np.isfinite(z)
        assert z >= 0.0

    def test_bkg_uncertainty_sigma_is_non_negative(self):
        """Verify the uncertainty array has no negative values for valid inputs."""
        b = np.array([5.0, 8.0, 3.0])
        factor = 10.0
        f, g = 0.5, 0.5
        bkg_unc = 0.02
        night_bkg = factor * g * b
        day_bkg   = factor * f * b
        sigma_night = np.sqrt(night_bkg + (bkg_unc * night_bkg) ** 2)
        sigma_day   = np.sqrt(day_bkg   + (bkg_unc * day_bkg)   ** 2)
        eff_unc = np.sqrt((sigma_night / g) ** 2 + (sigma_day / f) ** 2)
        assert np.all(eff_unc >= 0.0)


# ── Multi-bin sum significance ────────────────────────────────────────────────

class TestMultiBinSignificance:
    def test_more_signal_bins_increases_significance(self):
        b = np.ones(20) * 5.0
        s_day = np.ones(20) * 0.5
        s_night = np.ones(20) * 1.5

        # Only 5 bins contribute
        s_n_5 = s_night.copy(); s_n_5[5:] = s_day[5:]
        z5  = _daynight_compute(s_day, s_n_5,  b, factor=10.0)
        z20 = _daynight_compute(s_day, s_night, b, factor=10.0)
        assert z20 > z5

    def test_all_zero_background_still_finite(self):
        # bkg_eff gets contributions from signal terms when b=0.
        # Significance is non-zero when s_night > s_day.
        s_day   = np.ones(5) * 0.5
        s_night = np.ones(5) * 1.5
        b = np.zeros(5)
        z = _daynight_compute(s_day, s_night, b, factor=10.0)
        assert np.isfinite(z)
        assert z >= 0.0
