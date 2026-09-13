"""
Validation script: cross-check Cutflow, Fiducial, and Counts pkl consistency.

Checks:
  1. Cutflow vs Counts — weighted sums match at best-cut stage (all components).
  2. Fiducial pkls    — schema completeness: required columns present in all 32 files.
  3. CountsError      — CountsError column present and finite (non-negative) in both
                        Cutflow and Fiducial pkls.
  4. Day/Night decomposition — Solar == f_day*(Solar Day) + f_night*(Solar Night)
                        in both Cutflow and Counts pkls.

The Test-4 identity: counts must be real events for the stated exposure
-----------------------------------------------------------------------
Cutflow and Counts pkls report events observed in `Exposure` detector-years
(CountsUnit "events / MeV / N yr"). For those products to be read honestly,
every component must be the actual number of events in that exposure:

    Solar Day + Solar Night == Solar          (each roughly half of Solar)

That does NOT hold for the raw weights. lib/weights.py::_compute_osc_kde_and_exposure
normalises each nadir slice by its own weight sum:

    pee_slice = sum(pee_2d[mask] * w[mask]) / sum(w[mask])

so "Solar Day" and "Solar Night" come out as *conditional* rates — each is the
rate the detector would see if it were day (or night) for the whole exposure.
Left unscaled they each sit at ~1.0x Solar and sum to ~2x it. The producing
scripts therefore scale each slice by the fraction of the exposure it occupies:

    Solar Day   *= DAY_FRACTION
    Solar Night *= 1 - DAY_FRACTION

with DAY_FRACTION from config/analysis/physics.json (0.493 at SURF), the same key
daynight/01_daynight.py uses for its day/night count split. This test asserts the
post-scaling picture: the components sum to Solar, and each sits at its exposure
fraction. A sum ratio near 2.0 means the scaling was dropped somewhere.

Run:
  python3 tests/validate_pkl_consistency.py
  python3 tests/validate_pkl_consistency.py --tol 0.5     # looser sum tolerance (%)
  python3 tests/validate_pkl_consistency.py --test 4      # day/night closure only
  python3 tests/validate_pkl_consistency.py --dn-tol 1e-3 # looser closure tolerance
"""

import argparse
import os
import sys
import pickle

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lib import load_analysis_info

# ── Config ────────────────────────────────────────────────────────────────────

ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
info    = load_analysis_info(ROOT)
BASE    = f"{ROOT}/output/data"
CFBASE  = f"{BASE}/solar/cutflow"
CNTBASE = f"{BASE}/analysis"
FIDBASE = f"{BASE}/solar/nhits"

CONFIGS  = [
    "hd_1x2x6_centralAPA",
    "hd_1x2x6_lateralAPA",
    "vd_1x8x14_3view_30deg_nominal",
    "vd_1x8x14_3view_30deg_shielded",
]
SIGNALS   = ["marley", "gamma", "neutron", "radiological"]
ANALYSES  = ["HEP", "DayNight"]
FOLDER    = "truncated"
ENERGY    = "SolarEnergy"

REQUIRED_FID_COLS = {
    "#AdjCls", "#Hits", "#OpHits", "AdjCl", "Component", "Config", "Counts",
    "CountsUnit", "CountsError", "Exposure", "Folder", "NEvents", "NHits",
    "Name", "OpHits", "Oscillation", "Stage", "Type", "Weight",
}
# Columns added after some pkls on disk were written. Their absence means the file
# predates the change and should be regenerated; it is reported as a warning rather
# than a failure so a stale tree does not bury the substantive checks.
STALE_TOLERATED_FID_COLS = {"Oscillation"}
REQUIRED_CF_COLS = {
    "Config", "Name", "Component", "Stage", "NHits", "OpHits", "AdjCl",
    "Energy", "Counts", "CountsError", "SmoothedCounts", "SmoothedCountsError",
    "Exposure", "EnergyUnit", "CountsUnit",
}

HEP_PAIRS = [
    ("marley", "8B",           "8B"),
    ("marley", "hep",          "hep"),
    ("gamma",  "gamma",        "gamma"),
    ("neutron","neutron",      "neutron"),
    ("radiological","radiological","radiological"),
]
DN_PAIRS = [
    ("marley", "Solar Day",    "Solar Day"),
    ("marley", "Solar Night",  "Solar Night"),
    ("gamma",  "gamma",        "Gamma"),
    ("neutron","neutron",      "Neutron"),
    ("radiological","radiological","Radiological"),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load(p):
    return pickle.load(open(p, "rb")) if os.path.exists(p) else None

def _cf_path(cfg, sig, a):
    return f"{CFBASE}/{cfg}/{sig}/{FOLDER}/{a.lower()}/{cfg}_{sig}_{ENERGY}_{a}_Cutflow.pkl"

def _cnt_path(cfg, a):
    adir  = "hep"       if a == "HEP" else "day-night"
    aname = "HEP"       if a == "HEP" else "DayNight"
    return f"{CNTBASE}/{adir}/{cfg}/marley/{FOLDER}/default/{cfg}_marley_{aname}_Counts.pkl"

def _fid_path(cfg, sig, a):
    an_cap = "HEP" if a == "HEP" else "DayNight"
    return f"{FIDBASE}/{cfg}/{sig}/{FOLDER}/{a.lower()}/{cfg}_{sig}_Weighted_Distributions_Fiducial_{an_cap}.pkl"

def _cf_sum(cf, comp, stage="NHits+OpHits+AdjCl"):
    if cf is None or "Stage" not in cf.columns:
        return None
    m = (cf["Stage"] == stage) & (cf["Component"] == comp)
    r = cf[m]
    if r.empty:
        return None
    return float(np.nansum(np.asarray(r.iloc[0]["Counts"], dtype=float)))

def _cnt_sum(cnt, comp):
    if cnt is None:
        return None
    m = cnt["Component"] == comp
    if "SpectrumType" in cnt.columns:
        m = m & (cnt["SpectrumType"] == "Raw")
    r = cnt[m]
    if r.empty:
        return None
    return float(np.nansum(np.asarray(r.iloc[0]["Counts"], dtype=float)))

def _array_ok(arr, label):
    """Return list of issues for a numeric array stored in a pkl cell."""
    issues = []
    try:
        a = np.asarray(arr, dtype=float)
    except Exception as e:
        return [f"{label}: cannot cast to float — {e}"]
    if np.any(np.isnan(a)):
        issues.append(f"{label}: contains NaN")
    if np.any(np.isinf(a)):
        issues.append(f"{label}: contains Inf")
    if np.any(a < 0):
        issues.append(f"{label}: contains negative values")
    return issues

# ── Test 1: Cutflow vs Counts sum agreement ───────────────────────────────────

def test_cutflow_counts_sums(tol_pct: float = 0.1):
    print(f"\n{'='*70}")
    print(f"TEST 1: Cutflow vs Counts sum agreement  (tolerance={tol_pct}%)")
    print(f"{'='*70}")
    header = f"{'Config':<36} {'An':9} {'Sig':14} {'Comp':14} {'CF':>12} {'CNT':>12} {'diff':>9}"
    print(header)
    print("-" * len(header))

    failures, warnings = [], []

    for cfg in CONFIGS:
        for analysis in ANALYSES:
            pairs = HEP_PAIRS if analysis == "HEP" else DN_PAIRS
            cnt   = _load(_cnt_path(cfg, analysis))
            cf_m  = _load(_cf_path(cfg, "marley", analysis))

            if cf_m is None:
                warnings.append(f"MISSING Cutflow: {_cf_path(cfg, 'marley', analysis)}")
                continue
            if cnt is None:
                warnings.append(f"MISSING Counts: {_cnt_path(cfg, analysis)}")
                continue

            cf_cut = cf_m[cf_m["Stage"] == "NHits+OpHits+AdjCl"]
            cuts = (
                f"({int(cf_cut.iloc[0]['NHits'])},{int(cf_cut.iloc[0]['OpHits'])},{int(cf_cut.iloc[0]['AdjCl'])})"
                if not cf_cut.empty else "(?)"
            )

            for cf_sig, cf_comp, cnt_comp in pairs:
                cf   = _load(_cf_path(cfg, cf_sig, analysis))
                cf_s = _cf_sum(cf, cf_comp)
                cnt_s= _cnt_sum(cnt, cnt_comp)

                if cf_s is None or cnt_s is None or cnt_s == 0:
                    diff_s = "N/A"
                    diff   = None
                else:
                    diff   = 100 * (cf_s - cnt_s) / cnt_s
                    diff_s = f"{diff:+.2f}%"

                flag = "  ***" if (diff is not None and abs(diff) >= tol_pct) else ""
                print(f"{cfg:<36} {analysis:9} {cf_sig:14} {cf_comp:14} "
                      f"{cf_s if cf_s is not None else float('nan'):>12.3f} "
                      f"{cnt_s if cnt_s is not None else float('nan'):>12.3f} "
                      f"{diff_s:>9}{flag}  cuts={cuts}")
                if flag:
                    failures.append(f"{cfg} {analysis} {cf_sig}/{cf_comp}: {diff_s}")

    return failures, warnings


# ── Test 2: Fiducial pkl schema ───────────────────────────────────────────────

def test_fiducial_schema():
    print(f"\n{'='*70}")
    print("TEST 2: Fiducial pkl schema (columns + stages)")
    print(f"{'='*70}")

    failures, warnings = [], []
    n_ok = 0

    for cfg in CONFIGS:
        for sig in SIGNALS:
            for analysis in ANALYSES:
                p   = _fid_path(cfg, sig, analysis)
                rel = p.replace(FIDBASE + "/", "")
                df  = _load(p)

                if df is None:
                    failures.append(f"MISSING: {rel}")
                    print(f"MISSING  {rel}")
                    continue

                cols    = set(df.columns.tolist())
                missing = REQUIRED_FID_COLS - cols
                extra   = cols - REQUIRED_FID_COLS
                bad_stg = "NHits+OpHits+AdjCl" not in df["Stage"].unique() if "Stage" in df.columns else True
                old_nm  = any(c in cols for c in ["NHits_cut", "OpHits_cut", "AdjCl_cut"])

                stale_missing = missing & STALE_TOLERATED_FID_COLS
                missing       = missing - STALE_TOLERATED_FID_COLS

                issues = []
                if missing:  issues.append(f"missing cols: {sorted(missing)}")
                if extra:    issues.append(f"extra cols: {sorted(extra)}")
                if bad_stg:  issues.append("stage 'NHits+OpHits+AdjCl' absent")
                if old_nm:   issues.append("old column names (NHits_cut etc.)")

                if stale_missing:
                    warnings.append(
                        f"{rel}: predates {sorted(stale_missing)} — regenerate with 04_weighted.py"
                    )
                    print(f"STALE {rel}  -> missing {sorted(stale_missing)} (regenerate)")

                if issues:
                    for iss in issues:
                        failures.append(f"{rel}: {iss}")
                    print(f"FAIL  {rel}")
                    for iss in issues:
                        print(f"       -> {iss}")
                else:
                    n_ok += 1
                    print(f"OK    {rel}")

    print(f"\n{n_ok}/{len(CONFIGS)*len(SIGNALS)*len(ANALYSES)} files OK")
    return failures, warnings


# ── Test 3: CountsError validity ──────────────────────────────────────────────

def test_counts_error():
    print(f"\n{'='*70}")
    print("TEST 3: CountsError column — present, finite, non-negative")
    print(f"{'='*70}")

    failures, warnings = [], []
    n_ok = 0

    # 3a — Fiducial pkls
    print("\n-- Fiducial pkls --")
    for cfg in CONFIGS:
        for sig in SIGNALS:
            for analysis in ANALYSES:
                p   = _fid_path(cfg, sig, analysis)
                rel = p.replace(FIDBASE + "/", "")
                df  = _load(p)

                if df is None:
                    warnings.append(f"SKIP (missing): {rel}")
                    continue

                if "CountsError" not in df.columns:
                    failures.append(f"{rel}: CountsError column absent")
                    print(f"FAIL  {rel}  -> CountsError absent")
                    continue

                issues = []
                for _, row in df.iterrows():
                    stage = row.get("Stage", "?")
                    issues += _array_ok(row["CountsError"], f"Stage={stage}")

                if issues:
                    for iss in issues:
                        failures.append(f"{rel}: {iss}")
                    print(f"FAIL  {rel}")
                    for iss in set(issues):
                        print(f"       -> {iss}")
                else:
                    n_ok += 1
                    print(f"OK    {rel}")

    # 3b — Cutflow pkls
    print("\n-- Cutflow pkls --")
    for cfg in CONFIGS:
        for sig in SIGNALS:
            for analysis in ANALYSES:
                p   = _cf_path(cfg, sig, analysis)
                rel = p.replace(CFBASE + "/", "")
                df  = _load(p)

                if df is None:
                    warnings.append(f"SKIP (missing): {rel}")
                    continue

                missing_cols = REQUIRED_CF_COLS & {"CountsError", "SmoothedCountsError"} - set(df.columns)
                if missing_cols:
                    failures.append(f"{rel}: missing {sorted(missing_cols)}")
                    print(f"FAIL  {rel}  -> missing {sorted(missing_cols)}")
                    continue

                issues = []
                for _, row in df.iterrows():
                    stage = row.get("Stage", "?")
                    issues += _array_ok(row["CountsError"],         f"Stage={stage}/CountsError")
                    issues += _array_ok(row["SmoothedCountsError"], f"Stage={stage}/SmoothedCountsError")

                if issues:
                    for iss in issues:
                        failures.append(f"{rel}: {iss}")
                    print(f"FAIL  {rel}")
                    for iss in set(issues):
                        print(f"       -> {iss}")
                else:
                    n_ok += 1
                    print(f"OK    {rel}")

    total = len(CONFIGS) * len(SIGNALS) * len(ANALYSES)
    print(f"\n{n_ok}/{total*2} files OK (Fiducial + Cutflow)")
    return failures, warnings


# ── Test 4: Day/Night decomposition closure ───────────────────────────────────
# Solar must equal the exposure-weighted mean of the Day and Night slices, not
# their sum. See the module docstring for the derivation.

DN_SOLAR_COMPONENTS = ("Solar", "Solar Day", "Solar Night")


def _closure(solar, day, night, f_day):
    """
    Return (rel_err, sum_ratio, day_frac, night_frac).

    rel_err   : bin-wise |Solar - (Day + Night)| / |Solar|
    sum_ratio : sum(Day + Night) / sum(Solar)   -> 1.0 when scaled, ~2.0 when not
    day_frac  : sum(Day) / sum(Solar)           -> DAY_FRACTION when scaled
    """
    combined = day + night
    denom = float(np.nansum(np.abs(solar)))
    rel = float(np.nansum(np.abs(solar - combined))) / denom if denom > 0 else float("nan")
    tot = float(np.nansum(solar))
    if tot:
        sum_ratio  = (float(np.nansum(day)) + float(np.nansum(night))) / tot
        day_frac   = float(np.nansum(day)) / tot
        night_frac = float(np.nansum(night)) / tot
    else:
        sum_ratio = day_frac = night_frac = float("nan")
    return rel, sum_ratio, day_frac, night_frac


def _components_from(df, selector):
    """Pull the three Solar arrays out of one slice of a pkl; None if any absent."""
    out = []
    for comp in DN_SOLAR_COMPONENTS:
        rows = selector(df, comp)
        if rows is None or rows.empty:
            return None, comp
        out.append(np.asarray(rows.iloc[0]["Counts"], dtype=float))
    lengths = {a.shape for a in out}
    if len(lengths) != 1:
        return None, f"length mismatch {sorted(lengths)}"
    return out, None


def test_daynight_solar_decomposition(rel_tol: float = 5e-3, f_day: float = None,
                                      frac_tol: float = 0.05):
    if f_day is None:
        f_day = float(info.get("DAY_FRACTION", 0.493))
    print(f"\n{'='*70}")
    print(f"TEST 4: Day/Night closure  Solar == Day + Night   "
          f"(rel_tol={rel_tol:g}, f_day={f_day:g}+/-{frac_tol:g})")
    print(f"{'='*70}")

    failures, warnings = [], []
    n_ok = 0

    # 4a — Cutflow pkls, per cut stage
    print("\n-- Cutflow (DayNight), per stage --")
    for cfg in CONFIGS:
        p = _cf_path(cfg, "marley", "DayNight")
        df = _load(p)
        rel_p = p.replace(CFBASE + "/", "")
        if df is None:
            warnings.append(f"SKIP (missing): {rel_p}")
            print(f"SKIP  {rel_p}")
            continue

        present = set(df["Component"].unique()) if "Component" in df.columns else set()
        absent = [c for c in DN_SOLAR_COMPONENTS if c not in present]
        if absent:
            failures.append(f"{rel_p}: missing component(s) {absent} — regenerate with cutflow_plot.py")
            print(f"FAIL  {rel_p}  -> missing component(s) {absent}")
            continue

        for stage in df["Stage"].unique():
            sl = df[df["Stage"] == stage]
            arrays, bad = _components_from(
                sl, lambda d, c: d[d["Component"] == c]
            )
            if arrays is None:
                failures.append(f"{rel_p} [{stage}]: {bad}")
                print(f"FAIL  {rel_p} [{stage}] -> {bad}")
                continue
            solar, day, night = arrays
            rel, sum_ratio, dfrac, nfrac = _closure(solar, day, night, f_day)
            frac_ok = (abs(dfrac - f_day) <= frac_tol) and (abs(nfrac - (1 - f_day)) <= frac_tol)
            ok = np.isfinite(rel) and rel <= rel_tol and frac_ok
            n_ok += int(ok)
            flag = "" if ok else "  ***"
            print(f"{'OK  ' if ok else 'FAIL'}  {cfg:<32} {stage:<22} "
                  f"rel={rel:.2e}  (D+N)/S={sum_ratio:.4f}  D/S={dfrac:.3f} N/S={nfrac:.3f}{flag}")
            if not ok:
                hint = ""
                if np.isfinite(sum_ratio) and abs(sum_ratio - 2.0) < 0.1:
                    hint = "  [exposure-fraction scaling missing: Day/Night are conditional rates]"
                failures.append(
                    f"{rel_p} [{stage}]: closure rel={rel:.3e} (tol {rel_tol:g}), "
                    f"(D+N)/Solar={sum_ratio:.4f}, D/S={dfrac:.3f}, N/S={nfrac:.3f}{hint}"
                )

    # 4b — Counts pkls, per spectrum type
    print("\n-- DayNight Counts, per SpectrumType --")
    for cfg in CONFIGS:
        p = _cnt_path(cfg, "DayNight")
        df = _load(p)
        rel_p = p.replace(CNTBASE + "/", "")
        if df is None:
            warnings.append(f"SKIP (missing): {rel_p}")
            print(f"SKIP  {rel_p}")
            continue

        present = set(df["Component"].unique()) if "Component" in df.columns else set()
        absent = [c for c in DN_SOLAR_COMPONENTS if c not in present]
        if absent:
            failures.append(f"{rel_p}: missing component(s) {absent}")
            print(f"FAIL  {rel_p}  -> missing component(s) {absent}")
            continue

        spectra = sorted(df["SpectrumType"].unique()) if "SpectrumType" in df.columns else [None]
        for spec in spectra:
            sl = df if spec is None else df[df["SpectrumType"] == spec]
            arrays, bad = _components_from(
                sl, lambda d, c: d[d["Component"] == c]
            )
            if arrays is None:
                failures.append(f"{rel_p} [{spec}]: {bad}")
                print(f"FAIL  {rel_p} [{spec}] -> {bad}")
                continue
            solar, day, night = arrays
            rel, sum_ratio, dfrac, nfrac = _closure(solar, day, night, f_day)
            frac_ok = (abs(dfrac - f_day) <= frac_tol) and (abs(nfrac - (1 - f_day)) <= frac_tol)
            ok = np.isfinite(rel) and rel <= rel_tol and frac_ok
            n_ok += int(ok)
            flag = "" if ok else "  ***"
            print(f"{'OK  ' if ok else 'FAIL'}  {cfg:<32} {str(spec):<22} "
                  f"rel={rel:.2e}  (D+N)/S={sum_ratio:.4f}  D/S={dfrac:.3f} N/S={nfrac:.3f}{flag}")
            if not ok:
                hint = ""
                if np.isfinite(sum_ratio) and abs(sum_ratio - 2.0) < 0.1:
                    hint = "  [exposure-fraction scaling missing: Day/Night are conditional rates]"
                failures.append(
                    f"{rel_p} [{spec}]: closure rel={rel:.3e} (tol {rel_tol:g}), "
                    f"(D+N)/Solar={sum_ratio:.4f}, D/S={dfrac:.3f}, N/S={nfrac:.3f}{hint}"
                )

    print(f"\n{n_ok} slice(s) satisfied the closure")
    return failures, warnings


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Validate Cutflow / Fiducial / Counts pkl consistency.")
    parser.add_argument("--tol", type=float, default=0.1, help="Sum diff tolerance in %% (default 0.1)")
    parser.add_argument("--dn-tol", type=float, default=5e-3,
                        help="Relative tolerance for the Day/Night closure in test 4 (default 5e-3; the\n                             floor is ~7e-4 because DAY_FRACTION 0.493 differs from the nadir\n                             PDF day fraction 0.49999 used to build the Solar mean)")
    parser.add_argument("--f-day", type=float, default=None,
                        help="Daytime exposure fraction for the test-4 closure "
                             "(default: DAY_FRACTION from physics.json)")
    parser.add_argument("--frac-tol", type=float, default=0.05,
                        help="Absolute tolerance on Day/Solar and Night/Solar (default 0.05; the\n                             day-night asymmetry itself shifts these off f_day)")
    parser.add_argument("--test", choices=["1", "2", "3", "4", "all"], default="all",
                        help="Which test(s) to run")
    args = parser.parse_args()

    all_failures, all_warnings = [], []

    if args.test in ("1", "all"):
        f, w = test_cutflow_counts_sums(tol_pct=args.tol)
        all_failures += f; all_warnings += w

    if args.test in ("2", "all"):
        f, w = test_fiducial_schema()
        all_failures += f; all_warnings += w

    if args.test in ("3", "all"):
        f, w = test_counts_error()
        all_failures += f; all_warnings += w

    if args.test in ("4", "all"):
        f, w = test_daynight_solar_decomposition(rel_tol=args.dn_tol, f_day=args.f_day,
                                                 frac_tol=args.frac_tol)
        all_failures += f; all_warnings += w

    print(f"\n{'='*70}")
    if all_warnings:
        print(f"WARNINGS ({len(all_warnings)}):")
        for w in all_warnings:
            print(f"  {w}")
    if all_failures:
        print(f"\nFAILED ({len(all_failures)}):")
        for f in all_failures:
            print(f"  {f}")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
