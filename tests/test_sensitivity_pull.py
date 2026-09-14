"""
Tests for the pull-method sensitivity fit (lib/fitting.py, --fit_method pull) and the
method-independent validation gates used by 06_significance.py.

  1. Asimov data at the reference point give chi2 == 0 for every nuisance set
  2. Profiled chi2 never exceeds chi2 at nominal nuisances; Newton converges
  3. Linear (normalisation) nuisances: Newton result equals a brute-force minimisation
  4. Background-dominated regime (1e11 events/bin): chi2 is smooth and matches the
     Gaussian closed form, where the legacy nested minimiser produces O(100) noise
  5. Energy-scale nuisance absorbs a small energy shift of the data
  6. Worker dispatch: fit_method switch and optional diagnostics tuple
  7. Gates flag sentinels, isolated spikes, partial sin13 grid profiling and Asimov offsets
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lib.fitting import (
    SENSITIVITY_CHI2_SENTINEL,
    _poisson_deviance_terms,
    _sensitivity_apply_energy_scale,
    sensitivity_chi2_worker,
    sensitivity_pull_chi2,
    sensitivity_pull_jacobian,
    sensitivity_pull_profile,
    sensitivity_validation_gates,
)


def _templates(n_nadir=4, n_energy=20, bkg_level=50.0, seed=3):
    rng = np.random.default_rng(seed)
    e_ctr = np.linspace(1.0, 20.0, n_energy)
    shape = np.exp(-0.5 * ((e_ctr - 9.0) / 3.0) ** 2)
    pred = 200.0 * shape[None, :] * rng.uniform(0.8, 1.2, (n_nadir, 1))
    bkg = bkg_level * np.exp(-e_ctr / 6.0)[None, :] * np.ones((n_nadir, 1))
    return pred, bkg, e_ctr


# ── 1 ────────────────────────────────────────────────────────────────────────
def test_asimov_is_zero_for_all_nuisance_sets():
    pred, bkg, e_ctr = _templates()
    d13 = 0.05 * pred
    for kwargs in (
        {},
        {"sigma_pred": 0.04, "sigma_bkg": 0.02},
        {"sigma_pred": 0.04, "sigma_bkg": 0.02, "e_centers": e_ctr, "sigma_e_scale": 0.02},
        {"sigma_pred": 0.04, "sigma_bkg": 0.02, "e_centers": e_ctr, "sigma_e_scale": 0.02,
         "dpred_dsin13": d13, "sigma_sin13": 0.00056},
    ):
        res = sensitivity_pull_chi2(pred + bkg, pred, bkg, **kwargs)
        assert abs(res["chi2"]) < 1e-10, (kwargs, res["chi2"])
        assert res["converged"]


# ── 2 ────────────────────────────────────────────────────────────────────────
def test_profile_is_bounded_by_nominal_and_converges():
    pred, bkg, e_ctr = _templates()
    rng = np.random.default_rng(11)
    for _ in range(20):
        obs = rng.poisson(pred * rng.uniform(0.7, 1.3) + bkg).astype(float)
        res = sensitivity_pull_chi2(obs, pred, bkg, 0.04, 0.02, e_ctr, 0.02)
        assert res["chi2"] <= res["chi2_zero"] + 1e-9
        assert res["converged"], res
        assert np.isfinite(res["chi2"]) and res["chi2"] >= 0


# ── 3 ────────────────────────────────────────────────────────────────────────
def test_linear_nuisances_match_brute_force():
    pred, bkg, _ = _templates()
    obs = 1.08 * pred + 0.97 * bkg          # smooth, non-Asimov data
    mask = bkg > 0
    res = sensitivity_pull_chi2(obs, pred, bkg, sigma_pred=0.04, sigma_bkg=0.02)

    def objective(x):
        mu = (1 + x[0]) * pred + (1 + x[1]) * bkg
        return _poisson_deviance_terms(obs[mask], mu[mask]).sum() + (x[0] / 0.04) ** 2 + (x[1] / 0.02) ** 2

    brute = minimize(objective, x0=[0.0, 0.0], method="Nelder-Mead",
                     options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000})
    assert abs(res["chi2"] - brute.fun) < 1e-6 * max(1.0, brute.fun), (res["chi2"], brute.fun)
    assert np.allclose(
        [res["nuisances"]["signal_norm"], res["nuisances"]["background_norm"]], brute.x, atol=1e-5
    )


# ── 4 ────────────────────────────────────────────────────────────────────────
def test_background_dominated_scan_is_smooth():
    pred, bkg, e_ctr = _templates(bkg_level=1e11)
    tilt = np.linspace(-1.0, 1.0, pred.shape[1])[None, :]
    ts = np.linspace(-0.3, 0.3, 61)
    chi2 = []
    for t in ts:
        obs = pred * (1.0 + t * tilt) + bkg
        res = sensitivity_pull_chi2(obs, pred, bkg, 0.04, 0.02, e_ctr, 0.02)
        assert res["converged"]
        assert abs(res["chi2"] - res["chi2_gauss"]) <= 1e-6 * max(1.0, res["chi2"]) + 1e-9
        chi2.append(res["chi2"])
    chi2 = np.asarray(chi2)
    # quadratic in t to high accuracy: residual of a parabola fit is tiny vs the range
    coeff = np.polyfit(ts, chi2, 2)
    assert np.max(np.abs(np.polyval(coeff, ts) - chi2)) < 1e-6 * max(1.0, chi2.max())
    assert chi2[len(ts) // 2] < 1e-12


# ── 5 ────────────────────────────────────────────────────────────────────────
def test_energy_scale_nuisance_absorbs_small_shift():
    pred, bkg, e_ctr = _templates()
    obs = _sensitivity_apply_energy_scale(pred, 0.01, e_ctr) + bkg
    without = sensitivity_pull_chi2(obs, pred, bkg, 0.04, 0.02)
    with_e = sensitivity_pull_chi2(obs, pred, bkg, 0.04, 0.02, e_ctr, 0.02)
    assert with_e["chi2"] < 0.2 * without["chi2"]
    assert 0.005 < with_e["nuisances"]["energy_scale"] < 0.015


def test_jacobian_drops_zero_width_nuisances():
    pred, bkg, e_ctr = _templates()
    jac, sigma, names = sensitivity_pull_jacobian(pred, bkg, e_ctr, 0.0, 0.02, 0.0)
    assert names == ["background_norm"] and jac.shape == (1,) + pred.shape and sigma.tolist() == [0.02]


# ── 6 ────────────────────────────────────────────────────────────────────────
def _task(obs, pred, bkg, e_ctr, **extra):
    return {"params": (6e-5, 0.022, 0.304), "obs": obs, "pred1": pred, "pred2": 0.9 * pred,
            "bkg": bkg, "sigma_pred": 0.04, "sigma_bkg": 0.02, "marginalize_e_scale": True,
            "sigma_e_scale": 0.02, "e_centers_thld": e_ctr, **extra}


def test_worker_dispatch_and_diagnostics():
    pred, bkg, e_ctr = _templates()
    obs = pred + bkg
    out = sensitivity_chi2_worker(_task(obs, pred, bkg, e_ctr, fit_method="pull"))
    assert len(out) == 3 and abs(out[1]) < 1e-10 and out[2] > 0

    out = sensitivity_chi2_worker(_task(obs, pred, bkg, e_ctr, fit_method="pull", return_diagnostics=True))
    assert len(out) == 4 and set(out[3]) == {"solar", "react"}
    assert out[2] <= out[3]["react"]["chi2_zero"] + 1e-9

    legacy = sensitivity_chi2_worker(_task(obs, pred, bkg, e_ctr, marginalize_e_scale=False, return_diagnostics=True))
    assert len(legacy) == 4 and "chi2_zero" in legacy[3]["solar"]
    assert len(sensitivity_chi2_worker(_task(obs, pred, bkg, e_ctr, marginalize_e_scale=False))) == 3


def test_worker_precomputed_jacobians_match():
    pred, bkg, e_ctr = _templates()
    obs = 1.05 * pred + bkg
    base = _task(obs, pred, bkg, e_ctr, fit_method="pull")
    jacs = {
        "solar": sensitivity_pull_jacobian(pred, bkg, e_ctr, 0.04, 0.02, 0.02),
        "react": sensitivity_pull_jacobian(0.9 * pred, bkg, e_ctr, 0.04, 0.02, 0.02),
    }
    a = sensitivity_chi2_worker(base)
    b = sensitivity_chi2_worker({**base, "pull_jacobians": jacs})
    assert np.allclose(a[1:], b[1:], rtol=1e-12, atol=1e-12)


# ── 7 ────────────────────────────────────────────────────────────────────────
def _bowl_scan(sin13_values=(0.022,)):
    dm2 = np.linspace(3e-5, 1e-4, 30)
    sin12 = np.linspace(0.15, 0.45, 31)
    rows = []
    for d in dm2:
        for s12 in sin12:
            for s13 in sin13_values:
                rows.append([d, s13, s12, 4.0 * ((d - 6e-5) / 1e-5) ** 2 + 30.0 * (s12 - 0.304) ** 2 / 0.01])
    df = pd.DataFrame(rows, columns=["dm2", "sin13", "sin12", "chi2"])
    grid = df[np.isclose(df.sin13, 0.022)].pivot_table(index="dm2", columns="sin12", values="chi2")
    return df, grid


def test_gates_pass_on_smooth_bowl():
    df, grid = _bowl_scan()
    point = (float(grid.index[np.argmin(np.abs(grid.index - 6e-5))]), 0.022,
             float(grid.columns[np.argmin(np.abs(grid.columns - 0.304))]))
    df.loc[np.isclose(df.dm2, point[0]) & np.isclose(df.sin12, point[2]), "chi2"] = 0.0
    grid.loc[point[0], point[2]] = 0.0
    report = sensitivity_validation_gates(df, df, {"solar_sin12": grid}, solar_point=point, react_point=point)
    assert report["passed"], report


def test_gates_flag_spike_sentinel_and_offset():
    df, grid = _bowl_scan()
    spiky = grid.copy()
    spiky.iloc[10, 12] += 50.0
    report = sensitivity_validation_gates(df, df, {"solar_sin12": spiky})
    assert report["gates"]["smoothness"]["passed"] is False
    assert report["gates"]["smoothness"]["grids"]["solar_sin12"]["n_spikes"] == 1

    bad = df.copy()
    bad.loc[5, "chi2"] = SENSITIVITY_CHI2_SENTINEL
    report = sensitivity_validation_gates(bad, df, {})
    assert report["gates"]["finite"]["passed"] is False and not report["passed"]

    point = tuple(df.iloc[0][["dm2", "sin13", "sin12"]].astype(float))
    report = sensitivity_validation_gates(df, df, {}, solar_point=point)
    assert report["gates"]["asimov"]["passed"] is False


def test_gates_ignore_percent_level_bump_far_from_minimum():
    df, grid = _bowl_scan()
    bumpy = grid.copy()
    far = np.unravel_index(np.argmax(bumpy.to_numpy()), bumpy.shape)
    i, j = min(max(far[0], 2), bumpy.shape[0] - 3), min(max(far[1], 2), bumpy.shape[1] - 3)
    bumpy.iloc[i, j] += 0.02 * float(bumpy.iloc[i, j])     # 2% bump at large Delta chi2
    assert sensitivity_validation_gates(df, df, {"g": bumpy})["gates"]["smoothness"]["passed"]
    lo = np.unravel_index(np.argmin(bumpy.to_numpy()), bumpy.shape)
    bumpy.iloc[lo[0] + 1, lo[1]] += 60.0                    # large spike next to the bowl floor
    assert not sensitivity_validation_gates(df, df, {"g": bumpy})["gates"]["smoothness"]["passed"]


def test_gates_profile_bound_and_partial_sin13_grid():
    df, _ = _bowl_scan()
    diagnostics = {
        tuple(map(float, r[:3])): {"solar": {"chi2_zero": r[3] - 1.0}, "react": {"chi2_zero": r[3] + 1.0}}
        for r in df.itertuples(index=False)
    }
    report = sensitivity_validation_gates(df, df, {}, diagnostics=diagnostics)
    assert report["gates"]["profile_bound"]["n_violations"] == len(df)

    partial = pd.concat([df, df[np.isclose(df.sin12, df.sin12.iloc[0])].assign(sin13=0.03)])
    report = sensitivity_validation_gates(partial, partial, {}, sin13_profile="grid")
    assert report["gates"]["sin13_profile"]["passed"] is False
