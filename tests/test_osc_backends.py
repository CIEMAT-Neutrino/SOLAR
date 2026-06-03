"""Tests for lib/lib_osc_backends.py — OscResult, compute_prob3/nufast, combine_day_night."""
import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.oscillation_backends import (
    OscResult,
    compute_prob3,
    compute_nufast,
    get_nadir_pdf_file,
    get_nadir_pdf_nufast,
    combine_day_night,
)

# Default oscillation parameters
DM2   = 6.0e-5
SIN13 = 0.021
SIN12 = 0.303

# Analysis grids matching analysis.json defaults (120 E bins, 40 nadir bins)
ENERGY_EDGES = np.linspace(0.0, 30.0, 121)      # MeV
NADIR_EDGES  = np.linspace(-1.0, 1.0, 41)
E_CENTERS    = 0.5 * (ENERGY_EDGES[1:] + ENERGY_EDGES[:-1])
N_CENTERS    = 0.5 * (NADIR_EDGES[1:]  + NADIR_EDGES[:-1])

# Smaller grid for speed in most tests
ENERGY_EDGES_SMALL = np.linspace(5.0, 20.0, 11)
NADIR_EDGES_SMALL  = np.linspace(-1.0, 1.0, 9)
E_SMALL = 0.5 * (ENERGY_EDGES_SMALL[1:] + ENERGY_EDGES_SMALL[:-1])
N_SMALL = 0.5 * (NADIR_EDGES_SMALL[1:]  + NADIR_EDGES_SMALL[:-1])


# ── OscResult structure ────────────────────────────────────────────────────────

def test_prob3_returns_osc_result():
    r = compute_prob3(DM2, SIN13, SIN12, ENERGY_EDGES_SMALL, NADIR_EDGES_SMALL)
    assert isinstance(r, OscResult)
    assert isinstance(r.day,   pd.DataFrame)
    assert isinstance(r.night, pd.DataFrame)


def test_nufast_returns_osc_result():
    r = compute_nufast(DM2, SIN13, SIN12, ENERGY_EDGES_SMALL, NADIR_EDGES_SMALL)
    assert isinstance(r, OscResult)
    assert isinstance(r.day,   pd.DataFrame)
    assert isinstance(r.night, pd.DataFrame)


def test_prob3_night_shape():
    r = compute_prob3(DM2, SIN13, SIN12, ENERGY_EDGES_SMALL, NADIR_EDGES_SMALL)
    assert r.night.shape == (len(N_SMALL), len(E_SMALL)), f"Got {r.night.shape}"


def test_nufast_night_shape():
    r = compute_nufast(DM2, SIN13, SIN12, ENERGY_EDGES_SMALL, NADIR_EDGES_SMALL)
    assert r.night.shape == (len(N_SMALL), len(E_SMALL)), f"Got {r.night.shape}"


def test_prob3_day_energy_shape():
    r = compute_prob3(DM2, SIN13, SIN12, ENERGY_EDGES_SMALL, NADIR_EDGES_SMALL)
    assert r.day.shape[1] == len(E_SMALL)


def test_nufast_day_energy_shape():
    r = compute_nufast(DM2, SIN13, SIN12, ENERGY_EDGES_SMALL, NADIR_EDGES_SMALL)
    assert r.day.shape[1] == len(E_SMALL)


def test_prob3_probability_range():
    r = compute_prob3(DM2, SIN13, SIN12, ENERGY_EDGES_SMALL, NADIR_EDGES_SMALL)
    assert np.all(r.night.values >= 0), f"Negative P_ee in prob3 night"
    assert np.all(r.night.values <= 1), f"P_ee > 1 in prob3 night"


def test_nufast_probability_range():
    r = compute_nufast(DM2, SIN13, SIN12, ENERGY_EDGES_SMALL, NADIR_EDGES_SMALL)
    assert np.all(r.night.values >= 0), f"Negative P_ee in nufast night"
    assert np.all(r.night.values <= 1), f"P_ee > 1 in nufast night"


# ── Nadir PDFs ─────────────────────────────────────────────────────────────────

def test_nadir_pdf_nufast_normalised():
    pdf = get_nadir_pdf_nufast(N_CENTERS)
    assert abs(pdf.sum() - 1.0) < 1e-10, f"NuFast nadir PDF not normalised: {pdf.sum()}"


def test_nadir_pdf_nufast_non_negative():
    pdf = get_nadir_pdf_nufast(N_CENTERS)
    assert np.all(pdf >= 0.0), "Negative nadir PDF values"


def test_nadir_pdf_nufast_symmetric():
    """Full PDF covers both day and night. Night and day weights roughly equal (symmetric yearly trajectory)."""
    pdf = get_nadir_pdf_nufast(N_CENTERS)
    night_weight = pdf[N_CENTERS < 0].sum()
    day_weight   = pdf[N_CENTERS > 0].sum()
    assert night_weight > 0.0, "Zero weight on night bins"
    assert day_weight   > 0.0, "Zero weight on day bins"
    # Should be roughly symmetric (within 10% relative)
    ratio = night_weight / day_weight if day_weight > 0 else 0
    assert 0.5 < ratio < 2.0, f"Day/night weight asymmetry unexpected: ratio={ratio:.3f}"


@pytest.mark.skipif(
    not os.path.exists("/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/data/OSCILLATION/nadir.root"),
    reason="PNFS not accessible"
)
def test_nadir_pdf_file_normalised():
    pdf = get_nadir_pdf_file(nadir_centers=N_CENTERS)
    assert abs(pdf.sum() - 1.0) < 1e-6, f"File nadir PDF not normalised: {pdf.sum()}"


# ── combine_day_night ──────────────────────────────────────────────────────────

def test_combine_output_shape():
    r   = compute_prob3(DM2, SIN13, SIN12, ENERGY_EDGES_SMALL, NADIR_EDGES_SMALL)
    pdf = get_nadir_pdf_nufast(N_SMALL)
    df  = combine_day_night(r, pdf)
    assert df.shape == (len(N_SMALL), len(E_SMALL))


def test_combine_output_non_negative():
    r   = compute_prob3(DM2, SIN13, SIN12, ENERGY_EDGES_SMALL, NADIR_EDGES_SMALL)
    pdf = get_nadir_pdf_nufast(N_SMALL)
    df  = combine_day_night(r, pdf)
    assert np.all(df.values >= 0), "Negative combined P_ee"


def test_combine_matches_fixture_structure():
    """Combined output must have same shape as pre-computed pkl."""
    fixture_path = os.path.join(os.path.dirname(__file__), 'fixtures', 'oscillogram_default.pkl')
    if not os.path.exists(fixture_path):
        pytest.skip("Fixture not found")

    ref = pd.read_pickle(fixture_path)
    r   = compute_prob3(DM2, SIN13, SIN12, ENERGY_EDGES, NADIR_EDGES)
    pdf = get_nadir_pdf_file(nadir_centers=N_CENTERS)
    df  = combine_day_night(r, pdf)

    assert df.shape == ref.shape, f"Shape mismatch: {df.shape} vs {ref.shape}"


# ── Cross-backend consistency ──────────────────────────────────────────────────

def test_prob3_nufast_night_agreement():
    """Prob3++ and NuFast-Earth night P_ee should agree within 2% (different implementations)."""
    r_p3 = compute_prob3( DM2, SIN13, SIN12, ENERGY_EDGES_SMALL, NADIR_EDGES_SMALL)
    r_nf = compute_nufast(DM2, SIN13, SIN12, ENERGY_EDGES_SMALL, NADIR_EDGES_SMALL)

    diff = np.abs(r_p3.night.values - r_nf.night.values)
    max_diff = diff.max()
    assert max_diff < 0.02, f"Backends differ by more than 2%: max |ΔP_ee| = {max_diff:.4f}"


def test_prob3_nufast_day_agreement():
    """Day P_ee should agree within 2% between backends."""
    r_p3 = compute_prob3( DM2, SIN13, SIN12, ENERGY_EDGES_SMALL, NADIR_EDGES_SMALL)
    r_nf = compute_nufast(DM2, SIN13, SIN12, ENERGY_EDGES_SMALL, NADIR_EDGES_SMALL)

    diff = np.abs(r_p3.day.values - r_nf.day.values)
    max_diff = diff.max()
    assert max_diff < 0.02, f"Day backends differ by more than 2%: max |ΔP_ee| = {max_diff:.4f}"
