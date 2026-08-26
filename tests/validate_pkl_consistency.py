"""
Validation script: cross-check Cutflow, Fiducial, and Counts pkl consistency.

Checks:
  1. Cutflow vs Counts — weighted sums match at best-cut stage (all components).
  2. Fiducial pkls    — schema completeness: required columns present in all 32 files.
  3. CountsError      — CountsError column present and finite (non-negative) in both
                        Cutflow and Fiducial pkls.

Run:
  python3 tests/validate_pkl_consistency.py
  python3 tests/validate_pkl_consistency.py --tol 0.5   # looser sum tolerance (%)
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
    "Name", "OpHits", "Stage", "Type", "Weight",
}
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

                issues = []
                if missing:  issues.append(f"missing cols: {sorted(missing)}")
                if extra:    issues.append(f"extra cols: {sorted(extra)}")
                if bad_stg:  issues.append("stage 'NHits+OpHits+AdjCl' absent")
                if old_nm:   issues.append("old column names (NHits_cut etc.)")

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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Validate Cutflow / Fiducial / Counts pkl consistency.")
    parser.add_argument("--tol", type=float, default=0.1, help="Sum diff tolerance in %% (default 0.1)")
    parser.add_argument("--test", choices=["1", "2", "3", "all"], default="all",
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
