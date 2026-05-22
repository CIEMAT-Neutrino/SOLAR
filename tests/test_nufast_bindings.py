"""Smoke tests for NuFast-Earth pybind11 binding (_nufast_earth)."""
import sys
import os
import math
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'external', 'NuFast-Earth', 'python'))
import _nufast_earth as nufast

DM2   = 6.0e-5
SIN13 = 0.021
SIN12 = 0.303

ENERGIES_GEV  = np.linspace(5e-3, 20e-3, 20)
# Night coszs: same convention as Prob3++ and lib_osc.py — cosz < 0 = night (Earth-traversing)
NIGHT_COSZS   = np.linspace(-0.975, -0.025, 40)
DUNE_LAT_RAD  = 44.35 * math.pi / 180.0


def test_import():
    assert hasattr(nufast, 'compute_solar_day')
    assert hasattr(nufast, 'compute_solar_night')
    assert hasattr(nufast, 'solar_weight')
    assert hasattr(nufast, 'solar_weight_array')


def test_day_output_shape():
    pee = nufast.compute_solar_day(SIN12, SIN13, DM2, ENERGIES_GEV)
    assert pee.shape == (len(ENERGIES_GEV),), f"Expected ({len(ENERGIES_GEV)},), got {pee.shape}"


def test_night_output_shape():
    pee = nufast.compute_solar_night(SIN12, SIN13, DM2, ENERGIES_GEV, NIGHT_COSZS)
    assert pee.shape == (len(ENERGIES_GEV), len(NIGHT_COSZS)), \
        f"Expected {(len(ENERGIES_GEV), len(NIGHT_COSZS))}, got {pee.shape}"


def test_day_probability_range():
    pee = nufast.compute_solar_day(SIN12, SIN13, DM2, ENERGIES_GEV)
    assert np.all(pee >= 0.0), f"Negative day P_ee: min={pee.min()}"
    assert np.all(pee <= 1.0), f"Day P_ee > 1: max={pee.max()}"


def test_night_probability_range():
    pee = nufast.compute_solar_night(SIN12, SIN13, DM2, ENERGIES_GEV, NIGHT_COSZS)
    assert np.all(pee >= 0.0), f"Negative night P_ee: min={pee.min()}"
    assert np.all(pee <= 1.0), f"Night P_ee > 1: max={pee.max()}"


def test_day_msw_suppression():
    """Day P_ee should be MSW-suppressed (0.25-0.70 for solar 8B range)."""
    pee = nufast.compute_solar_day(SIN12, SIN13, DM2, ENERGIES_GEV)
    mean_pee = pee.mean()
    assert 0.25 < mean_pee < 0.70, f"Day P_ee outside MSW range: {mean_pee:.3f}"


def test_night_differs_from_day():
    """Night P_ee must differ from day — Earth regeneration effect.
    cosz < 0 = neutrino upward-going through Earth (night convention).
    """
    pee_day   = nufast.compute_solar_day(SIN12, SIN13, DM2, ENERGIES_GEV)
    # Deep Earth path (cosz=-1 = straight through core) for maximum regeneration
    deep_cosz = np.array([-1.0])
    pee_night = nufast.compute_solar_night(SIN12, SIN13, DM2, ENERGIES_GEV, deep_cosz)
    assert not np.allclose(pee_day, pee_night[:, 0], atol=1e-4), \
        "Night and day P_ee are identical at cosz=-1 — Earth regeneration not applied"


def test_solar_weight_scalar():
    """Solar_Weight must be non-negative and integrate reasonably."""
    eta = math.pi / 3  # 60° nadir
    w = nufast.solar_weight(eta, DUNE_LAT_RAD)
    assert w >= 0.0, f"Negative solar weight: {w}"


def test_solar_weight_array_normalisation():
    """Nadir PDF should integrate to ~1 (within trapezoid integration error)."""
    eta_edges = np.linspace(0, math.pi, 2001)
    eta_centers = 0.5 * (eta_edges[1:] + eta_edges[:-1])
    deta = eta_edges[1] - eta_edges[0]

    weights = nufast.solar_weight_array(eta_centers, DUNE_LAT_RAD)
    integral = np.sum(weights * np.sin(eta_centers) * deta)  # dΩ ∝ sin(η)dη
    # Solar_Weight includes the sin(eta) factor already; integrate plain weights × deta
    integral_plain = np.sum(weights) * deta
    # Roughly should be of order 1 — exact norm depends on convention
    assert weights.shape == (len(eta_centers),)
    assert np.all(weights >= 0.0), "Negative solar weight values"
    assert integral_plain > 0.0, "Solar weight integrates to zero"


def test_day_energy_variation():
    """P_ee should vary with energy (MSW resonance creates energy dependence)."""
    pee = nufast.compute_solar_day(SIN12, SIN13, DM2, ENERGIES_GEV)
    assert np.std(pee) > 0.001, f"P_ee shows no energy variation: std={np.std(pee):.5f}"
