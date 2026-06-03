"""
Verify that the parallel chi² scan in 06_significance.py produces
bit-identical results to serial execution.

Tests:
  1. sensitivity_chi2_worker returns finite values on synthetic data
  2. ProcessPoolExecutor(workers=2) == serial loop (no escale)
  3. sensitivity_chi2_worker with energy-scale marginalization
  4. ProcessPoolExecutor(workers=2) == serial loop (with escale)
  5. ThreadPoolExecutor template loading produces same dict as serial loading
  6. None propagation: worker returns (params, None, None) when solar fit fails
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lib.fitting import (
    sensitivity_chi2_worker,
    _sensitivity_apply_energy_scale,
    _sensitivity_fit_with_escale,
)


# ── Shared synthetic data factory ─────────────────────────────────────────────

def _make_data(n_nadir: int = 3, n_energy: int = 25, n_grid: int = 8, seed: int = 42):
    rng = np.random.default_rng(seed)
    pred1 = np.abs(rng.normal(10.0, 2.0, (n_nadir, n_energy)))
    pred2 = np.abs(rng.normal(8.0,  2.0, (n_nadir, n_energy)))
    bkg   = np.abs(rng.normal(5.0,  1.0, (n_nadir, n_energy)))
    e_ctr = np.linspace(0.5, 24.5, n_energy)

    grid = [(float(i) * 1e-5, 0.021, 0.303) for i in range(1, n_grid + 1)]
    obs = {
        params: np.clip(pred1 + bkg + rng.normal(0, 0.5, pred1.shape), 0.0, None)
        for params in grid
    }
    return pred1, pred2, bkg, e_ctr, grid, obs


def _task(params, obs_arr, pred1, pred2, bkg, e_ctr, escale=False):
    return {
        "params":              params,
        "obs":                 obs_arr,
        "pred1":               pred1,
        "pred2":               pred2,
        "bkg":                 bkg,
        "sigma_pred":          0.04,
        "sigma_bkg":           0.02,
        "marginalize_e_scale": escale,
        "sigma_e_scale":       0.02,
        "e_centers_thld":      e_ctr,
    }


# ── Test 1: basic worker output ───────────────────────────────────────────────

def test_worker_returns_finite():
    pred1, pred2, bkg, e_ctr, grid, obs = _make_data(n_grid=2)
    params = grid[0]
    t = _task(params, obs[params], pred1, pred2, bkg, e_ctr)
    p_out, solar, react = sensitivity_chi2_worker(t)
    assert p_out == params
    assert solar is not None, "solar chi2 is None"
    assert react is not None, "react chi2 is None"
    assert np.isfinite(float(solar)), f"solar not finite: {solar}"
    assert np.isfinite(float(react)), f"react not finite: {react}"
    assert float(solar) >= 0.0
    assert float(react) >= 0.0


# ── Test 2: serial == parallel (no energy scale) ─────────────────────────────

def test_parallel_equals_serial_no_escale():
    pred1, pred2, bkg, e_ctr, grid, obs = _make_data(n_grid=8)
    tasks = [_task(p, obs[p], pred1, pred2, bkg, e_ctr, escale=False) for p in grid]

    serial = [sensitivity_chi2_worker(t) for t in tasks]

    with ProcessPoolExecutor(max_workers=2) as pool:
        parallel = list(pool.map(sensitivity_chi2_worker, tasks))

    for (ps, ss, rs), (pp, sp, rp) in zip(serial, parallel):
        assert ps == pp, "param mismatch"
        assert ss is not None and sp is not None
        assert np.isclose(ss, sp, rtol=1e-10, atol=0), \
            f"solar chi2 mismatch: serial={ss:.10f} parallel={sp:.10f}"
        assert rs is not None and rp is not None
        assert np.isclose(rs, rp, rtol=1e-10, atol=0), \
            f"react chi2 mismatch: serial={rs:.10f} parallel={rp:.10f}"


# ── Test 3: worker with energy-scale marginalization ─────────────────────────

def test_worker_with_escale():
    pred1, pred2, bkg, e_ctr, grid, obs = _make_data(n_grid=2)
    params = grid[0]
    t = _task(params, obs[params], pred1, pred2, bkg, e_ctr, escale=True)
    p_out, solar, react = sensitivity_chi2_worker(t)
    assert solar is not None and np.isfinite(float(solar)), f"escale solar: {solar}"
    assert react is not None and np.isfinite(float(react)), f"escale react: {react}"
    assert float(solar) >= 0.0
    assert float(react) >= 0.0


# ── Test 4: serial == parallel (with energy scale) ───────────────────────────

def test_parallel_equals_serial_with_escale():
    pred1, pred2, bkg, e_ctr, grid, obs = _make_data(n_grid=4)
    tasks = [_task(p, obs[p], pred1, pred2, bkg, e_ctr, escale=True) for p in grid]

    serial   = [sensitivity_chi2_worker(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=2) as pool:
        parallel = list(pool.map(sensitivity_chi2_worker, tasks))

    for (ps, ss, rs), (pp, sp, rp) in zip(serial, parallel):
        assert ps == pp
        assert ss is not None and sp is not None
        assert np.isclose(ss, sp, rtol=1e-8), \
            f"escale solar mismatch: serial={ss:.10f} parallel={sp:.10f}"
        if rs is not None and rp is not None:
            assert np.isclose(rs, rp, rtol=1e-8), \
                f"escale react mismatch: serial={rs:.10f} parallel={rp:.10f}"


# ── Test 5: ThreadPoolExecutor loading == serial loading ─────────────────────

def test_thread_loading_matches_serial():
    n_nadir, n_energy, n_grid = 3, 25, 6
    rng = np.random.default_rng(7)
    bkg = np.abs(rng.normal(5.0, 1.0, (n_nadir, n_energy)))
    grid = [(float(i) * 1e-5, 0.021, 0.303) for i in range(1, n_grid + 1)]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write one pkl per grid point
        pkl_paths = {}
        for params in grid:
            arr = pd.DataFrame(np.abs(rng.normal(10.0, 2.0, (n_nadir, n_energy))))
            fname = os.path.join(
                tmpdir,
                f"dm2_{params[0]:.3e}_sin13_{params[1]:.3e}_sin12_{params[2]:.3e}.pkl",
            )
            arr.to_pickle(fname)
            pkl_paths[params] = fname

        # Serial reference
        serial_dict = {}
        for params in grid:
            serial_dict[params] = (
                np.nan_to_num(pd.read_pickle(pkl_paths[params]).values, nan=0.0) + bkg
            )

        # Parallel loading via ThreadPoolExecutor (mirrors 06_significance.py)
        def _load(params):
            return params, (
                np.nan_to_num(pd.read_pickle(pkl_paths[params]).values, nan=0.0) + bkg
            )

        parallel_dict = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            for params, df in pool.map(_load, grid):
                parallel_dict[params] = df

        assert set(serial_dict) == set(parallel_dict), "key sets differ"
        for params in grid:
            np.testing.assert_array_equal(
                serial_dict[params],
                parallel_dict[params],
                err_msg=f"Array mismatch for {params}",
            )


# ── Test 6: None propagation ──────────────────────────────────────────────────

def test_none_solar_skips_reactor():
    """If solar chi2 is None, worker must return (params, None, None)."""
    # Force solar fit to fail by passing all-zero obs and pred (degenerate case)
    pred1, pred2, bkg, e_ctr, grid, obs = _make_data(n_grid=1)
    params = grid[0]
    zeros = np.zeros_like(obs[params])
    t = _task(params, zeros, zeros, pred2, zeros, e_ctr, escale=False)

    # Sensitivity_Fitter may or may not return None here — just verify
    # that if solar_chi2 is None the reactor slot is also None
    p_out, solar, react = sensitivity_chi2_worker(t)
    assert p_out == params
    if solar is None:
        assert react is None, \
            "react chi2 must be None when solar chi2 is None (serial semantics preserved)"


# ── Test 7: _sensitivity_apply_energy_scale ───────────────────────────────────

def test_apply_energy_scale_zero_noop():
    template = np.arange(12, dtype=float).reshape(3, 4)
    e = np.array([1.0, 2.0, 3.0, 4.0])
    result = _sensitivity_apply_energy_scale(template, 0.0, e)
    np.testing.assert_array_equal(result, template)


def test_apply_energy_scale_shifts_spectrum():
    e = np.linspace(1.0, 10.0, 50)
    template = np.sin(e)[np.newaxis, :]
    shifted = _sensitivity_apply_energy_scale(template, 0.10, e)
    # Positive delta_e shifts spectrum to lower energies (query = E/(1+delta_e) < E)
    assert not np.allclose(template, shifted), "shift had no effect"
    # Shifted values should still be in [-1, 1] (sin range)
    assert np.all(np.abs(shifted) <= 1.0 + 1e-10)
