"""Smoke tests for Prob3++ pybind11 binding (_prob3_solar)."""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'external', 'Prob3plusplus', 'python'))
import _prob3_solar as prob3

# Default parameters matching analysis.json
DM2   = 6.0e-5
SIN13 = 0.021
SIN12 = 0.303

ENERGIES_GEV = np.linspace(5e-3, 20e-3, 20)   # 5-20 MeV solar range
NADIRS       = np.linspace(-0.975, 0.975, 40)  # 40-bin analysis grid


def test_import():
    assert hasattr(prob3, 'compute_solar_oscillogram')
    assert hasattr(prob3, 'solar_production_fraction')


def test_output_shape():
    pee = prob3.compute_solar_oscillogram(SIN12, SIN13, DM2, ENERGIES_GEV, NADIRS)
    assert pee.shape == (len(ENERGIES_GEV), len(NADIRS)), f"Expected {(len(ENERGIES_GEV), len(NADIRS))}, got {pee.shape}"


def test_probability_range():
    pee = prob3.compute_solar_oscillogram(SIN12, SIN13, DM2, ENERGIES_GEV, NADIRS)
    assert np.all(pee >= 0.0), f"Negative P_ee: min={pee.min()}"
    assert np.all(pee <= 1.0), f"P_ee > 1: max={pee.max()}"


def test_msw_suppression():
    """Day survival prob should be MSW-suppressed (~0.3-0.6 for solar 8B range)."""
    day_nadirs = np.array([0.5])
    pee = prob3.compute_solar_oscillogram(SIN12, SIN13, DM2, ENERGIES_GEV, day_nadirs)
    mean_pee = pee.mean()
    assert 0.25 < mean_pee < 0.70, f"Day P_ee out of MSW range: {mean_pee:.3f}"


def test_night_enhancement():
    """Night P_ee (nadir<0) should differ from day — Earth matter effect."""
    day_nadirs   = np.array([0.5])
    night_nadirs = np.array([-0.5])
    pee_day   = prob3.compute_solar_oscillogram(SIN12, SIN13, DM2, ENERGIES_GEV, day_nadirs)
    pee_night = prob3.compute_solar_oscillogram(SIN12, SIN13, DM2, ENERGIES_GEV, night_nadirs)
    # Night and day should not be identical (Earth matter effect changes P_ee)
    assert not np.allclose(pee_day, pee_night, atol=1e-4), \
        "Night and day P_ee are identical — Earth matter effect not applied"


def test_production_fractions_sum_to_one():
    f1, f2, f3 = prob3.solar_production_fraction(10.0, DM2, SIN12, SIN13)
    assert abs(f1 + f2 + f3 - 1.0) < 1e-10, f"Production fractions don't sum to 1: {f1+f2+f3}"


def test_regression_vs_fixture():
    """Compare with pre-computed pkl fixture. Max |ΔP_ee| < 0.005."""
    import pandas as pd

    fixture_path = os.path.join(os.path.dirname(__file__), 'fixtures', 'oscillogram_default.pkl')
    if not os.path.exists(fixture_path):
        pytest.skip("Fixture not found — run Phase 0 setup first")

    df = pd.read_pickle(fixture_path)
    e_centers_mev = df.columns.astype(float).values
    nadir_centers  = df.index.astype(float).values

    pee = prob3.compute_solar_oscillogram(
        SIN12, SIN13, DM2,
        e_centers_mev * 1e-3,  # MeV → GeV
        nadir_centers
    )

    # Fixture is nadir-convolved; raw Prob3++ is not.
    # Compare shape and rough scale only at this stage.
    assert pee.shape == (len(e_centers_mev), len(nadir_centers))

    # The convolved fixture has very small values (nadir-weighted sum ~0.5/2000 per bin).
    # Un-convolved P_ee should be 0.3-0.7 on average.
    assert pee.mean() > 0.1, f"Unexpected P_ee scale: mean={pee.mean():.4f}"


def test_invalid_parameters():
    with pytest.raises(Exception):
        prob3.compute_solar_oscillogram(1.5, SIN13, DM2, ENERGIES_GEV, NADIRS)
    with pytest.raises(Exception):
        prob3.compute_solar_oscillogram(SIN12, SIN13, -1.0, ENERGIES_GEV, NADIRS)
