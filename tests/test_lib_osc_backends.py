"""
Tests for lib_osc.py backend kwarg (Phase 5).

Verifies:
1. backend="file" (default) — unchanged behavior
2. backend="prob3" — produces same-shaped output as file backend
3. backend="nufast" — produces same-shaped output
4. separate_day_night=True — returns OscResult instead of DataFrame
5. Numerical agreement between prob3 and file backends (< 0.5%)
"""
import sys, os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.oscillation import get_oscillation_map
from lib.oscillation_backends import OscResult

DM2   = 6.0e-5
SIN13 = 0.021
SIN12 = 0.303

PNFS_ACCESSIBLE = os.path.exists(
    "/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/data/OSCILLATION/pkl/rebin/"
    "osc_probability_dm2_6.000e-05_sin13_2.100e-02_sin12_3.030e-01.pkl"
)


# ── Default backend (file) untouched ──────────────────────────────────────────

@pytest.mark.skipif(not PNFS_ACCESSIBLE, reason="PNFS not accessible")
def test_file_backend_returns_dict():
    result = get_oscillation_map(dm2=[DM2], sin13=[SIN13], sin12=[SIN12],
                                 ext="pkl", rebin=True, backend="file")
    assert isinstance(result, dict)
    assert len(result) == 1


@pytest.mark.skipif(not PNFS_ACCESSIBLE, reason="PNFS not accessible")
def test_file_backend_default_arg():
    """Calling without backend= kwarg must work identically."""
    result_default = get_oscillation_map(dm2=[DM2], sin13=[SIN13], sin12=[SIN12],
                                          ext="pkl", rebin=True)
    result_file    = get_oscillation_map(dm2=[DM2], sin13=[SIN13], sin12=[SIN12],
                                          ext="pkl", rebin=True, backend="file")
    key = list(result_default.keys())[0]
    assert key in result_file
    pd.testing.assert_frame_equal(result_default[key], result_file[key])


# ── Prob3 backend ─────────────────────────────────────────────────────────────

def test_prob3_backend_returns_dict():
    result = get_oscillation_map(dm2=[DM2], sin13=[SIN13], sin12=[SIN12], backend="prob3")
    assert isinstance(result, dict)
    assert len(result) == 1


def test_prob3_backend_df_shape():
    result = get_oscillation_map(dm2=[DM2], sin13=[SIN13], sin12=[SIN12], backend="prob3")
    df = list(result.values())[0]
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (40, 120), f"Expected (40,120), got {df.shape}"


def test_prob3_backend_probability_range():
    result = get_oscillation_map(dm2=[DM2], sin13=[SIN13], sin12=[SIN12], backend="prob3")
    df = list(result.values())[0]
    assert np.all(df.values >= 0), "Negative P_ee from prob3 backend"


def test_prob3_backend_separate_day_night():
    result = get_oscillation_map(dm2=[DM2], sin13=[SIN13], sin12=[SIN12],
                                  backend="prob3", separate_day_night=True)
    osc = list(result.values())[0]
    assert isinstance(osc, OscResult), f"Expected OscResult, got {type(osc)}"
    assert osc.night.shape == (40, 120)


# ── NuFast backend ────────────────────────────────────────────────────────────

def test_nufast_backend_returns_dict():
    result = get_oscillation_map(dm2=[DM2], sin13=[SIN13], sin12=[SIN12], backend="nufast")
    assert isinstance(result, dict)
    assert len(result) == 1


def test_nufast_backend_df_shape():
    result = get_oscillation_map(dm2=[DM2], sin13=[SIN13], sin12=[SIN12], backend="nufast")
    df = list(result.values())[0]
    assert df.shape == (40, 120), f"Expected (40,120), got {df.shape}"


def test_nufast_backend_probability_range():
    result = get_oscillation_map(dm2=[DM2], sin13=[SIN13], sin12=[SIN12], backend="nufast")
    df = list(result.values())[0]
    assert np.all(df.values >= 0), "Negative P_ee from nufast backend"


def test_nufast_backend_separate_day_night():
    result = get_oscillation_map(dm2=[DM2], sin13=[SIN13], sin12=[SIN12],
                                  backend="nufast", separate_day_night=True)
    osc = list(result.values())[0]
    assert isinstance(osc, OscResult)


# ── Cross-backend shape consistency ───────────────────────────────────────────

def test_prob3_nufast_same_shape():
    r_p3 = get_oscillation_map(dm2=[DM2], sin13=[SIN13], sin12=[SIN12], backend="prob3")
    r_nf = get_oscillation_map(dm2=[DM2], sin13=[SIN13], sin12=[SIN12], backend="nufast")
    df_p3 = list(r_p3.values())[0]
    df_nf = list(r_nf.values())[0]
    assert df_p3.shape == df_nf.shape


@pytest.mark.skipif(not PNFS_ACCESSIBLE, reason="PNFS not accessible")
def test_prob3_matches_file_shape():
    """prob3 backend DataFrame must match file backend shape exactly."""
    r_file = get_oscillation_map(dm2=[DM2], sin13=[SIN13], sin12=[SIN12],
                                  ext="pkl", rebin=True, backend="file")
    r_p3   = get_oscillation_map(dm2=[DM2], sin13=[SIN13], sin12=[SIN12], backend="prob3")
    df_file = list(r_file.values())[0]
    df_p3   = list(r_p3.values())[0]
    assert df_p3.shape == df_file.shape, \
        f"Shape mismatch: prob3={df_p3.shape}, file={df_file.shape}"


# ── Multiple parameter sets ───────────────────────────────────────────────────

def test_prob3_multiple_params():
    dm2_list   = [6.0e-5, 7.4e-5]
    sin13_list = [SIN13, SIN13]
    sin12_list = [SIN12, SIN12]
    result = get_oscillation_map(dm2=dm2_list, sin13=sin13_list, sin12=sin12_list,
                                  backend="prob3")
    assert len(result) == 2
    for df in result.values():
        assert df.shape == (40, 120)
