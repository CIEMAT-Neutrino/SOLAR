"""
compare_truth_position_exports.py — acceptance checks for export_repo/ against the old export/
===========================================================================================
1. structure: equal-length arrays per row, no all-NaN columns, Name == "all", key columns present
2. reference numbers (also stored in meta.json["checks"])
3. every number of every Kind equals the old pickles (arrays are expanded back to the old long format)
4. FoM / BestFoM equal the old best-FoM numbers (DayNight, HD central, reco, X and Y: ~369)

Exit status 0 only if everything passes.   Usage: python3 src/tools/compare_truth_position_exports.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "output/data/solar/truth_position"
OLD, NEW = BASE / "export_v2", BASE / "export_repo"
AN = {"DayNight": "DAYNIGHT", "HEP": "HEP", "Sensitivity": "SENSITIVITY"}
fails, n_checks = [], 0


def check(ok, msg):
    global n_checks
    n_checks += 1
    if not ok:
        fails.append(msg)
        print("FAIL", msg)


def load(kind):
    files = sorted(NEW.glob(f"*_all_{kind}.pkl"))
    return pd.concat([pd.read_pickle(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def old(name):
    return pd.read_pickle(OLD / f"{name}.pkl")


def expand(df, arrays):
    """One row per array element, other columns repeated."""
    rows = []
    for r in df.to_dict("records"):
        n = len(r[arrays[0]])
        for i in range(n):
            rows.append({**{k: v for k, v in r.items() if k not in arrays}, **{a: r[a][i] for a in arrays}})
    return pd.DataFrame(rows)


def same(new, oldd, keys, pairs, label):
    """pairs: {new column: old column}. Inner-merge on keys, require the full old table to be matched and all values equal."""
    a = new.rename(columns={k: v for k, v in pairs.items()})
    cols = list(dict.fromkeys(keys + list(pairs.values())))
    m = oldd[cols].merge(a[cols], on=keys, how="left", suffixes=("_old", "_new"), indicator=True)
    check((m["_merge"] == "both").all(), f"{label}: {int((m['_merge'] != 'both').sum())} old rows without a match")
    check(not a.duplicated(keys).any(), f"{label}: duplicate keys in the new table")
    for v in pairs.values():
        if v in keys:
            continue
        x, y = m[f"{v}_old"], m[f"{v}_new"]
        if x.dtype.kind in "fiu" and y.dtype.kind in "fiu":
            check(np.allclose(x.astype(float), y.astype(float), rtol=1e-12, atol=0, equal_nan=True), f"{label}.{v}: numeric mismatch")
        else:
            check((x.astype(str) == y.astype(str)).all(), f"{label}.{v}: string mismatch")


# ---- 1. structure --------------------------------------------------------------------------------
kinds = sorted({f.name.split("_all_")[1][:-4] for f in NEW.glob("*_all_*.pkl")})
for kind in kinds:
    for f in sorted(NEW.glob(f"*_all_{kind}.pkl")):
        df = pd.read_pickle(f)
        check(f.name == f"{df.Config.iloc[0]}_all_{kind}.pkl" and df.Config.nunique() == 1, f"{f.name}: file name / Config")
        check((df.Name == "all").all() and (df.Study == "default").all(), f"{f.name}: Name/Study")
        check(set(df.Analysis) <= {"DayNight", "HEP", "Sensitivity"}, f"{f.name}: Analysis labels")
        check(set(df.Geometry) <= {"hd", "vd"}, f"{f.name}: Geometry")
        check(not df.isna().all().any(), f"{f.name}: all-NaN column")
        lists = [c for c in df.columns if df[c].map(lambda v: isinstance(v, (list, np.ndarray))).all()]
        for r in df.itertuples():
            lens = {len(getattr(r, c)) for c in lists}
            if len(lens) > 1:
                check(False, f"{f.name}: unequal array lengths {lens}")
                break
        else:
            check(True, "")
print(f"structure checked on {len(kinds)} kinds")

# ---- 2. reference numbers ------------------------------------------------------------------------
meta = json.loads((NEW / "meta.json").read_text())
ck = meta["checks"]
wc = ck["WallCdf_HD_central_truth_20cm"]
check(abs(wc["gamma"] - 0.46) < 0.01 and abs(wc["marley"] - 0.10) < 0.01 and abs(wc["neutron"] - 0.08) < 0.01, f"WallCdf reference {wc}")
sg = ck["Significance_HD_lateral_DayNight"]   # values after the 2026-09-23 DayNight/HEP background-label fix
check(abs(sg["default"] - 1.537) < 5e-4 and abs(sg["fiduc_truth"] - 2.057) < 5e-4 and abs(sg["fiduc_truth_refvol"] - 1.701) < 5e-4, f"Significance reference {sg}")
pf = ck["PassFractions_DayNight_HD_central_neutron_disagree"]
check(np.allclose(pf["PassFraction"], [0.011, 0.995, 0.0066, 0.0066], atol=6e-4) and pf["NMC"] == 12, f"PassFractions reference {pf}")
print("reference numbers:", json.dumps(ck))

# ---- 3. number for number ------------------------------------------------------------------------
K5 = ["Analysis", "Config"]
w = expand(load("WallCdf"), ["Distance", "CDF"]).rename(columns={"Sample": "species", "Position": "position", "Distance": "distance_cm", "CDF": "cdf", "Config": "config"})
w = w.rename(columns={"Selection": "selection"})
same(w[w.Analysis == "DayNight"], old("wall_cdf"), ["config", "species", "selection", "position", "distance_cm"], {"cdf": "cdf", "NMC": "n_mc"}, "WallCdf")
for kind, src in (("Residuals", "residual_hist"), ("ResidualsWide", "residual_hist_wide")):
    r = expand(load(kind), ["Residual", "WeightFraction", "NMCPerBin"])
    r = r.rename(columns={"Sample": "species", "Config": "config", "Residual": "bin_lo_cm", "WeightFraction": "weight_fraction", "NMCPerBin": "n_mc"})
    r["axis"] = r.Variable.str.lower()
    same(r[r.Analysis == "DayNight"], old(src), ["config", "species", "axis", "bin_lo_cm"], {"weight_fraction": "weight_fraction", "n_mc": "n_mc"}, kind)
rs = load("ResidualSummary").rename(columns={"Sample": "species", "Config": "config"})
rs["axis"] = rs.Variable.str.lower()
same(rs[rs.Analysis == "DayNight"], old("residual_summary"), ["config", "species", "axis"], {"Median": "median_cm", "FractionWithin30cm": "frac_within_agree_cm", "FractionBeyond150cm": "frac_beyond_150cm", "NMC": "n_mc_window"}, "ResidualSummary")
sr = load("ShellRatio").rename(columns={"Sample": "species", "Config": "config", "ShellLow": "shell_lo_cm", "ShellHigh": "shell_hi_cm"})
same(sr[sr.Analysis == "DayNight"], old("shell_ratio"), ["config", "species", "shell_lo_cm", "shell_hi_cm"], {"Ratio": "ratio_vs_100_200cm", "RatioError": "ratio_err"}, "ShellRatio")
for kind in ("WallCdf", "ShellRatio", "ResidualSummary"):     # replicas across analyses are identical
    d = load(kind)
    cols = [c for c in d.columns if c != "Analysis" and not d[c].map(lambda v: isinstance(v, list)).any()]
    g = {a: d[d.Analysis == a][cols].sort_values(cols).reset_index(drop=True) for a in ("DayNight", "HEP", "Sensitivity")}
    check(g["DayNight"].equals(g["HEP"]) and g["DayNight"].equals(g["Sensitivity"]), f"{kind}: analysis replicas differ")

d = load("PassFractions").assign(analysis=lambda x: x.Analysis.map(AN)).rename(columns={"Sample": "group", "Config": "config"})
d = d.rename(columns={"VariantIndex": "variant_index"})
same(d, old("pass_fractions"), ["analysis", "config", "group", "variant_index"], {"Variant": "variant", "PassFraction": "pass_fraction", "NMC": "n_mc"}, "PassFractions")
d = load("Significance").assign(analysis=lambda x: x.Analysis).rename(columns={"Config": "config", "Variant": "variant"})
o = old("significance").dropna()
same(d, o, ["analysis", "config", "variant"], {"Significance": "significance_sigma"}, "Significance")
check(set(load("Significance").query("Analysis == 'Sensitivity'").SignificanceUnit) == {r"\Delta\chi^2"} and set(load("Significance").query("Analysis != 'Sensitivity'").SignificanceUnit) == {r"\sigma"}, "Significance units")

fc = old("face_composition")
d = load("FaceComposition").assign(analysis=lambda x: x.Analysis.map(AN)).rename(columns={"Sample": "species", "Config": "config", "Stage": "stage", "Face": "face"})
same(d, fc[fc.kind == "face"], ["analysis", "config", "species", "stage", "face"], {"WeightFraction": "weight_fraction", "NMC": "n_mc", "SumW": "sum_w"}, "FaceComposition")
out = fc[fc.face == "outside"].rename(columns={"weight_fraction": "outside"})
same(d.drop_duplicates(["analysis", "config", "species", "stage"]), out, ["analysis", "config", "species", "stage"], {"OutsideFraction": "outside"}, "FaceComposition.Outside")
check(np.allclose(d.groupby(["analysis", "config", "species", "stage"]).WeightFraction.sum(), 1.0), "FaceComposition: faces do not sum to 1")

xh = expand(load("XEntryHist"), ["AbsDX", "WeightFraction", "NMCPerBin"])
xh["after_cut"] = xh.Selection == "after_cut"
xh = xh.assign(analysis=xh.Analysis.map(AN)).rename(columns={"Sample": "species", "Config": "config", "AbsDX": "bin_lo_cm", "WeightFraction": "weight_fraction", "NMCPerBin": "n_mc"})
same(xh, old("x_entry_hist"), ["analysis", "config", "species", "after_cut", "bin_lo_cm"], {"weight_fraction": "weight_fraction", "n_mc": "n_mc"}, "XEntryHist")
xs = load("XEntrySummary").assign(analysis=lambda x: x.Analysis.map(AN), after_cut=lambda x: x.Selection == "after_cut").rename(columns={"Sample": "species", "Config": "config"})
same(xs, old("x_entry_summary"), ["analysis", "config", "species", "after_cut"], {"NMC": "n_mc", "SumW": "sum_w", "MedianAbsDX": "median_abs_dx_cm", "FractionDXAbove30cm": "frac_dx_gt_30cm",
                                                                                "FractionDXAbove100cm": "frac_dx_gt_100cm", "FractionRecoXAtEdge": "frac_recox_at_edge"}, "XEntrySummary")

og = old("xy_scan_grid")
sgd = load("ScanGrid").assign(analysis=lambda x: x.Analysis.map(AN)).rename(columns={"Config": "config", "Mode": "mode", "FiducialX": "fiducial_x_cm", "FiducialY": "fiducial_y_cm"})
same(sgd, og, ["analysis", "config", "mode", "fiducial_x_cm", "fiducial_y_cm"], {"SignalEfficiency": "signal_efficiency", "SumWGammaNeutron": "sum_w_gamma_neutron", "SumWRadiological": "sum_w_radiological",
                                                                                 "NMCGammaNeutron": "n_mc_gamma_neutron", "SumWSignalBefore": "sum_w_signal_before"}, "ScanGrid")
og = og.assign(fom_old=np.where(og.n_mc_gamma_neutron >= 5, og.signal_efficiency * og.sum_w_signal_before / np.sqrt(np.maximum(og.sum_w_gamma_neutron, 1e-12)), np.nan))
same(sgd, og, ["analysis", "config", "mode", "fiducial_x_cm", "fiducial_y_cm"], {"FoM": "fom_old"}, "ScanGrid.FoM")
sc = load("ScanCurves")
xo = expand(sc[sc.Variable == "X only"], ["FiducialCut", "SignalEfficiency", "SumWGammaNeutron", "FoM", "NMCGammaNeutron"]).assign(fiducial_x_cm=lambda x: x.FiducialCut, fiducial_y_cm=0.0)
yo = expand(sc[sc.Variable == "Y only"], ["FiducialCut", "SignalEfficiency", "SumWGammaNeutron", "FoM", "NMCGammaNeutron"]).assign(fiducial_y_cm=lambda x: x.FiducialCut, fiducial_x_cm=0.0)
cur = pd.concat([xo, yo]).assign(analysis=lambda x: x.Analysis.map(AN)).rename(columns={"Config": "config", "Mode": "mode"})
sub = og[(og.fiducial_y_cm == 0) | (og.fiducial_x_cm == 0)]
n_cur = len(cur)
cur = cur.drop_duplicates(["analysis", "config", "mode", "fiducial_x_cm", "fiducial_y_cm"])   # the origin belongs to both the X-only and the Y-only curve
same(cur, sub, ["analysis", "config", "mode", "fiducial_x_cm", "fiducial_y_cm"], {"SignalEfficiency": "signal_efficiency", "SumWGammaNeutron": "sum_w_gamma_neutron", "NMCGammaNeutron": "n_mc_gamma_neutron"}, "ScanCurves")
check(n_cur == len(sub) + int(((og.fiducial_x_cm == 0) & (og.fiducial_y_cm == 0)).sum()), "ScanCurves: row count (the origin is in both curves)")

se = old("signal_efficiency_x")
sr_ = load("SignalRecovered").assign(analysis=lambda x: x.Analysis.map(AN)).rename(columns={"Config": "config"})
for xd_old, xd_new, pos, col_e, col_g in (("default volume X", "default volume X", "reco X", "signal_eff_reco_x", "sum_w_gn_reco_x"), ("default volume X", "default volume X", "truth X", "signal_eff_truth_x", "sum_w_gn_truth_x"),
                                          ("truth volume X", "truth volume X", "reco X", "signal_eff_reco_x", "sum_w_gn_reco_x"), ("truth volume X", "truth volume X", "truth X", "signal_eff_truth_x", "sum_w_gn_truth_x"),
                                          ("FoM-best X, reco", "FoM-best X", "reco X", "signal_eff_reco_x", "sum_w_gn_reco_x"), ("FoM-best X, truth X", "FoM-best X", "truth X", "signal_eff_truth_x", "sum_w_gn_truth_x")):
    o = se[se.x_definition == xd_old][["analysis", "config", "fiducial_x_cm", col_e, col_g]].rename(columns={col_e: "e", col_g: "g"})
    n = sr_[(sr_.XDefinition == xd_new) & (sr_.Position == pos)].rename(columns={"SignalEfficiency": "e", "SumWGammaNeutron": "g", "FiducialX": "fiducial_x_cm"})
    same(n, o, ["analysis", "config"], {"e": "e", "g": "g", "fiducial_x_cm": "fiducial_x_cm"}, f"SignalRecovered[{xd_old}/{pos}]")
dd = sr_.groupby(["analysis", "config", "XDefinition"]).SignalEfficiencyDiffPP.nunique()
check((dd == 1).all(), "SignalRecovered: DiffPP not repeated on both rows")

bs = load("BackgroundSplit").assign(analysis=lambda x: x.Analysis.map(AN)).rename(columns={"Config": "config", "Selection": "selection"})
same(bs, old("background_split"), ["analysis", "config", "selection"], {"SumWGammaNeutron": "sum_w_gamma_neutron", "SumWRadiological": "sum_w_radiological", "RadiologicalFraction": "radiological_fraction",
                                                                         "NMCGammaNeutron": "n_mc_gamma_neutron", "NMCRadiological": "n_mc_radiological"}, "BackgroundSplit")
ss = load("SurvivorStatistics").assign(analysis=lambda x: x.Analysis.map(AN)).rename(columns={"Config": "config", "Sample": "species"})
same(ss, old("survivor_statistics"), ["analysis", "config", "species"], {"NMCWindowCut": "n_mc_window_cut", "NMCRecoFiducial": "n_mc_reco_fiducial", "NMCTruthPipeline": "n_mc_truth_pipeline",
                                                                         "SumWRecoFiducial": "sum_w_reco_fiducial", "MeanWPerEvent": "mean_w_per_event", "NEff": "n_eff", "LargestEventShare": "largest_event_share",
                                                                         "ShareOfBackgroundReco": "share_of_background_reco", "ShareOfBackgroundTruthPipeline": "share_of_background_truth_pipeline"}, "SurvivorStatistics")
ev = expand(load("SurvivorEvents"), ["Dist3D", "RecoX", "RecoY", "RecoZ", "TruthX", "TruthY", "TruthZ", "Weight"])
oe = old("survivors_events")
check(len(ev) == len(oe), "SurvivorEvents: event count")
for col_new, col_old in (("Dist3D", "dist3d_cm"), ("RecoX", "reco_x"), ("TruthX", "truth_x"), ("Weight", "weight")):
    a = np.sort(ev[col_new].to_numpy(float)); b = np.sort(oe[col_old].to_numpy(float))
    check(np.array_equal(a, b), f"SurvivorEvents.{col_new}")
check(int((load("SurvivorEvents").query("Agree == 'agree'").NMC.sum())) == int(oe.agree.sum()), "SurvivorEvents: agree count")

fm = load("FlashMismatch").assign(analysis=lambda x: x.Analysis.map(AN)).rename(columns={"Config": "config", "Sample": "species", "Selection": "selection"})
same(fm, old("flash_mismatch"), ["analysis", "config", "species", "selection"], {"NMC": "n_mc", "FractionRecoXAtEdge": "frac_recox_at_edge", "FractionDXAboveDriftTol": "frac_dx_gt_drift_tol",
                                                                                "FractionAnyTolerance": "frac_any_tolerance", "FractionPurityLow": "frac_purity_below", "FractionPurityLowOrDX": "frac_purity_below_or_dx"}, "FlashMismatch")
tk = load("TruthKeyCheck").assign(analysis=lambda x: x.Analysis.map(AN)).rename(columns={"Config": "config", "Sample": "species", "Selection": "selection", "CandidateKey": "candidate_key"})
same(tk, old("truth_key_check"), ["analysis", "config", "species", "selection", "candidate_key"], {"NMC": "n_mc", "FractionYZWithin30cm": "frac_yz_within_30cm", "FractionInActiveBox": "frac_in_active_box"}, "TruthKeyCheck")
check(set(tk.groupby(["analysis", "config", "species", "selection"]).IsConfiguredKey.apply(lambda v: (v == "yes").sum())) <= {0, 1}, "TruthKeyCheck: more than one configured key")

# ---- 4. FoM and BestFoM against the old best-FoM numbers -------------------------------------------
bo = old("xy_scan_best")
bf = load("BestFoM").assign(analysis=lambda x: x.Analysis.map(AN)).rename(columns={"Config": "config", "Mode": "mode"})
bf["scan"] = bf.Scan.map({"X and Y": "X and Y", "X only": "X only (Y=0)", "Y only": "Y only (X=0)"})
bo2 = bo[np.isfinite(bo.sn_over_sqrt_b)]
same(bf, bo2, ["analysis", "config", "mode", "scan"], {"FoM": "sn_over_sqrt_b", "FiducialX": "best_x_cm", "FiducialY": "best_y_cm", "SignalEfficiency": "signal_efficiency",
                                                       "SumWGammaNeutron": "sum_w_gamma_neutron", "NMC": "n_mc_gamma_neutron"}, "BestFoM")
ref = float(bf[(bf.analysis == "DAYNIGHT") & (bf.config == "hd_1x2x6_centralAPA") & (bf["mode"] == "reco") & (bf.Scan == "X and Y")].FoM.iloc[0])
check(abs(ref - 369.3) < 0.1, f"BestFoM DayNight HD central reco X and Y = {ref} (expected ~369)")
print(f"BestFoM DayNight HD central reco X and Y: {ref:.2f}")

print(f"\n{n_checks} checks, {len(fails)} failed")
sys.exit(1 if fails else 0)
