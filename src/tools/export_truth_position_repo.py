"""
export_truth_position_repo.py — repack the truth-position export for the LOWE_RECONSTRUCTION_PUBLICATION macros
=================================================================================================================
Reads the pickles written by `truth_position_study.py --stage export` (output/data/solar/truth_position/export/)
and writes them in the layout the plot repo's macros expect, so no loader adapter is needed there:

  export_repo/{config}_all_{Kind}.pkl       one file per config and Kind, all samples and analyses inside
  export_repo/README.md, export_repo/meta.json

Nothing is recomputed: every number comes from the old pickles (src/tools/compare_truth_position_exports.py
checks that number for number). Derived columns are pure arithmetic on them (FoM, differences, string relabelling).

Key columns of every row: Analysis (DayNight | HEP | Sensitivity), Geometry (hd | vd), Config, Name ("all"),
Study ("default"), and Sample wherever a sample or group exists. Curves are one row with equal-length lists;
categorical values are strings; all-NaN columns are dropped.

Analysis-independent kinds (WallCdf, ShellRatio, Residuals*, ResidualSummary, FlashMismatch) are written once per
analysis with identical rows so that any `--analysis` filter in the macros finds them.

Usage:  python3 src/tools/export_truth_position_repo.py
"""

import argparse
import json
import os
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "output/data/solar/truth_position"

parser = argparse.ArgumentParser()
parser.add_argument("--src", default=str(BASE / "export_v2"))
parser.add_argument("--dst", default=str(BASE / "export_repo"))
parser.add_argument("--sync_root", default=str(ROOT / "output/data/analysis"),
                    help="Also write every kind into the synced analysis tree (sync_solar_data.sh): "
                         "{sync_root}/{day-night|hep|sensitivity}/{config}/marley/truncated/fiduc_truth/{config}_marley_{Analysis}_TruthPosition{Kind}.pkl")
parser.add_argument("--sync", action=argparse.BooleanOptionalAction, default=True)
args = parser.parse_args()
SRC, DST = Path(args.src), Path(args.dst)

ANALYSES = {"DAYNIGHT": "DayNight", "HEP": "HEP", "SENSITIVITY": "Sensitivity"}
KEYS = ["Analysis", "Geometry", "Config", "Name", "Study"]
MIN_MC = 5                                                    # same support rule as the old best-FoM selection


def old(name):
    return pd.read_pickle(SRC / f"{name}.pkl")


def key(config, analysis, sample=None):
    d = {"Analysis": analysis, "Geometry": config.split("_")[0], "Config": config, "Name": "all", "Study": "default"}
    if sample is not None:
        d["Sample"] = sample
    return d


ROWS = {}


def add(kind, row):
    ROWS.setdefault(kind, []).append(row)


def add_all_analyses(kind, config, sample, fields):
    """Analysis-independent data, repeated for every analysis label."""
    for an in ANALYSES.values():
        add(kind, {**key(config, an, sample), **fields})


def fl(a):
    return [float(x) for x in a]


# ---- analysis-independent -------------------------------------------------------------------------
for (c, sp, pos), g in old("wall_cdf").groupby(["config", "species", "position"]):
    g = g.sort_values("distance_cm")
    add_all_analyses("WallCdf", c, sp, {"Position": pos, "Distance": fl(g.distance_cm), "CDF": fl(g.cdf), "NMC": int(g.n_mc.iloc[0]), "DistanceUnit": "cm"})

for r in old("shell_ratio").itertuples():
    add_all_analyses("ShellRatio", r.config, r.species, {"Shell": f"{r.shell_lo_cm:g}-{r.shell_hi_cm:g} cm", "ShellLow": float(r.shell_lo_cm), "ShellHigh": float(r.shell_hi_cm),
                                                         "Ratio": float(r.ratio_vs_100_200cm), "RatioError": float(r.ratio_err)})

for kind, src in (("Residuals", "residual_hist"), ("ResidualsWide", "residual_hist_wide")):
    for (c, sp, ax), g in old(src).groupby(["config", "species", "axis"]):
        g = g.sort_values("bin_lo_cm")
        add_all_analyses(kind, c, sp, {"Variable": ax.upper(), "Residual": fl(g.bin_lo_cm), "BinWidth": float(g.bin_hi_cm.iloc[0] - g.bin_lo_cm.iloc[0]),
                                       "WeightFraction": fl(g.weight_fraction), "NMCPerBin": [int(x) for x in g.n_mc], "NMC": int(g.n_mc.sum()), "ResidualUnit": "cm"})

for r in old("residual_summary").itertuples():
    add_all_analyses("ResidualSummary", r.config, r.species, {"Variable": r.axis.upper(), "Median": float(r.median_cm), "FractionWithin30cm": float(r.frac_within_agree_cm),
                                                              "FractionBeyond150cm": float(r.frac_beyond_150cm), "NMC": int(r.n_mc_window)})

for r in old("flash_mismatch").itertuples():
    add("FlashMismatch", {**key(r.config, ANALYSES[r.analysis], r.species), "Selection": r.selection, "NMC": int(r.n_mc), "FractionRecoXAtEdge": float(r.frac_recox_at_edge),
                          "FractionDXAboveDriftTol": float(r.frac_dx_gt_drift_tol), "FractionDXAbove30cm": float(r.frac_dx_gt_30cm),
                          "FractionDYZAboveTransverseTol": float(r.frac_dyz_gt_transverse_tol), "FractionAnyTolerance": float(r.frac_any_tolerance),
                          "FractionPurityZero": float(r.frac_purity_zero), "FractionPurityLow": float(r.frac_purity_below), "FractionPurityLowAndDX": float(r.frac_purity_below_and_dx),
                          "FractionPurityLowOrDX": float(r.frac_purity_below_or_dx), "PDXGivenPurityLow": float(r.p_dx_given_purity_low),
                          "PPurityLowGivenDX": float(r.p_purity_low_given_dx), "FractionDXAmongPure": float(r.frac_dx_among_pure)})

for r in old("truth_key_check").itertuples():
    add("TruthKeyCheck", {**key(r.config, ANALYSES[r.analysis], r.species), "Selection": r.selection, "CandidateKey": r.candidate_key, "IsConfiguredKey": "yes" if r.is_configured_key else "no",
                          "NMC": int(r.n_mc), "FractionYZWithin30cm": float(r.frac_yz_within_30cm), "FractionXWithin30cm": float(r.frac_x_within_30cm),
                          "FractionXYZWithin30cm": float(r.frac_xyz_within_30cm), "FractionInActiveBox": float(r.frac_in_active_box)})

# ---- analysis-dependent ---------------------------------------------------------------------------
for r in old("cuts").itertuples():
    add("Cuts", {**key(r.config, ANALYSES[r.analysis]), "NHits": int(r.nhits_min), "OpHits": int(r.ophits_min), "AdjCl": int(r.adjcl_max)})

for r in old("fiducial_volumes").itertuples():
    add("FiducialVolumes", {**key(r.config, ANALYSES[r.analysis]), "Variant": r.variant, "FiducialX": r.fiducial_x_cm, "FiducialY": r.fiducial_y_cm, "FiducialZ": r.fiducial_z_cm})

for r in old("pass_fractions").itertuples():
    add("PassFractions", {**key(r.config, ANALYSES[r.analysis], r.group), "VariantIndex": int(r.variant_index), "Variant": r.variant, "PassFraction": float(r.pass_fraction), "NMC": int(r.n_mc)})

for r in old("significance").itertuples():
    if not np.isfinite(r.significance_sigma):
        continue
    an = ANALYSES[r.analysis.upper()]
    add("Significance", {**key(r.config, an), "Variant": r.variant, "Significance": float(r.significance_sigma),
                         "SignificanceUnit": r"\Delta\chi^2" if an == "Sensitivity" else r"\sigma"})

fc = old("face_composition")
outside = fc[fc.face == "outside"].set_index(["analysis", "config", "species", "stage"]).weight_fraction
for r in fc[fc.kind == "face"].itertuples():
    add("FaceComposition", {**key(r.config, ANALYSES[r.analysis], r.species), "Stage": r.stage, "Face": r.face, "WeightFraction": float(r.weight_fraction), "NMC": int(r.n_mc),
                            "SumW": float(r.sum_w), "OutsideFraction": float(outside[(r.analysis, r.config, r.species, r.stage)])})

for (an, c, sp, ac), g in old("x_entry_hist").groupby(["analysis", "config", "species", "after_cut"]):
    g = g.sort_values("bin_lo_cm")
    add("XEntryHist", {**key(c, ANALYSES[an], sp), "Selection": "after_cut" if ac else "window", "AbsDX": fl(g.bin_lo_cm), "BinWidth": float(g.bin_hi_cm.iloc[0] - g.bin_lo_cm.iloc[0]),
                       "WeightFraction": fl(g.weight_fraction), "NMCPerBin": [int(x) for x in g.n_mc], "NMC": int(g.n_mc.sum()), "AbsDXUnit": "cm"})

for r in old("x_entry_summary").itertuples():
    add("XEntrySummary", {**key(r.config, ANALYSES[r.analysis], r.species), "Selection": "after_cut" if r.after_cut else "window", "NMC": int(r.n_mc), "SumW": float(r.sum_w),
                          "MedianAbsDX": float(r.median_abs_dx_cm), "FractionDXAbove30cm": float(r.frac_dx_gt_30cm), "FractionDXAbove100cm": float(r.frac_dx_gt_100cm),
                          "FractionRecoXAtEdge": float(r.frac_recox_at_edge)})

# scans: FoM = S/sqrt(B), NaN where the gamma+neutron MC support is below MIN_MC or the weight is zero
sg = old("xy_scan_grid").copy()
sg["fom"] = np.where((sg.n_mc_gamma_neutron >= MIN_MC) & (sg.sum_w_gamma_neutron > 0),
                     sg.signal_efficiency * sg.sum_w_signal_before / np.sqrt(np.where(sg.sum_w_gamma_neutron > 0, sg.sum_w_gamma_neutron, 1.0)), np.nan)
for r in sg.itertuples():
    add("ScanGrid", {**key(r.config, ANALYSES[r.analysis]), "Mode": r.mode, "ModeLabel": r.mode_label, "FiducialX": float(r.fiducial_x_cm), "FiducialY": float(r.fiducial_y_cm),
                     "SignalEfficiency": float(r.signal_efficiency), "SumWGammaNeutron": float(r.sum_w_gamma_neutron), "SumWRadiological": float(r.sum_w_radiological),
                     "NMCGammaNeutron": int(r.n_mc_gamma_neutron), "SumWSignalBefore": float(r.sum_w_signal_before), "FoM": float(r.fom)})

for (an, c, mode, ml), g in sg.groupby(["analysis", "config", "mode", "mode_label"]):
    for var, sel, xcol in (("X only", g[g.fiducial_y_cm == 0], "fiducial_x_cm"), ("Y only", g[g.fiducial_x_cm == 0], "fiducial_y_cm")):
        sel = sel.sort_values(xcol)
        add("ScanCurves", {**key(c, ANALYSES[an]), "Variable": var, "Mode": mode, "ModeLabel": ml, "FiducialCut": fl(sel[xcol]), "SignalEfficiency": fl(sel.signal_efficiency),
                           "SumWGammaNeutron": fl(sel.sum_w_gamma_neutron), "FoM": fl(sel.fom), "NMCGammaNeutron": [int(x) for x in sel.n_mc_gamma_neutron], "FiducialCutUnit": "cm"})
    for scan, sel in (("X and Y", g), ("X only", g[g.fiducial_y_cm == 0]), ("Y only", g[g.fiducial_x_cm == 0])):
        sel = sel.sort_values(["fiducial_x_cm", "fiducial_y_cm"])
        if sel.fom.notna().any():
            b = sel.loc[sel.fom.idxmax()]                     # first maximum in (X, Y) order, as the old best-volume search
            add("BestFoM", {**key(c, ANALYSES[an]), "Mode": mode, "ModeLabel": ml, "Scan": scan, "FoM": float(b.fom), "FiducialX": float(b.fiducial_x_cm), "FiducialY": float(b.fiducial_y_cm),
                            "SignalEfficiency": float(b.signal_efficiency), "SumWGammaNeutron": float(b.sum_w_gamma_neutron), "NMC": int(b.n_mc_gamma_neutron)})

# identical-volume note carried over from the old best table (strings; empty when the volume is unique)
best_old = old("xy_scan_best")
note = {(r.analysis, r.config, r.mode, r.scan): r.identical_volume_as for r in best_old.itertuples()}
for row in ROWS["BestFoM"]:
    an = {v: k for k, v in ANALYSES.items()}[row["Analysis"]]
    scan_old = {"X and Y": "X and Y", "X only": "X only (Y=0)", "Y only": "Y only (X=0)"}[row["Scan"]]
    row["IdenticalVolumeAs"] = str(note.get((an, row["Config"], row["Mode"], scan_old), ""))

# signal efficiency with truth X vs reco X (Y = Z = 0)
se = old("signal_efficiency_x")
for (an, c), g in se.groupby(["analysis", "config"], sort=False):
    g = g.set_index("x_definition")
    for xd in ("default volume X", "truth volume X"):
        if xd in g.index:
            r = g.loc[xd]
            diff = 100.0 * float(r.signal_eff_truth_x - r.signal_eff_reco_x)
            for pos, eff, gn in (("reco X", r.signal_eff_reco_x, r.sum_w_gn_reco_x), ("truth X", r.signal_eff_truth_x, r.sum_w_gn_truth_x)):
                add("SignalRecovered", {**key(c, ANALYSES[an]), "XDefinition": xd, "Position": pos, "FiducialX": float(r.fiducial_x_cm), "SignalEfficiency": float(eff),
                                        "SumWGammaNeutron": float(gn), "SignalEfficiencyDiffPP": diff})
    if "FoM-best X, reco" in g.index and "FoM-best X, truth X" in g.index:
        rr, rt = g.loc["FoM-best X, reco"], g.loc["FoM-best X, truth X"]
        diff = 100.0 * float(rt.signal_eff_truth_x - rr.signal_eff_reco_x)
        add("SignalRecovered", {**key(c, ANALYSES[an]), "XDefinition": "FoM-best X", "Position": "reco X", "FiducialX": float(rr.fiducial_x_cm), "SignalEfficiency": float(rr.signal_eff_reco_x),
                                "SumWGammaNeutron": float(rr.sum_w_gn_reco_x), "SignalEfficiencyDiffPP": diff})
        add("SignalRecovered", {**key(c, ANALYSES[an]), "XDefinition": "FoM-best X", "Position": "truth X", "FiducialX": float(rt.fiducial_x_cm), "SignalEfficiency": float(rt.signal_eff_truth_x),
                                "SumWGammaNeutron": float(rt.sum_w_gn_truth_x), "SignalEfficiencyDiffPP": diff})

for r in old("background_split").itertuples():
    add("BackgroundSplit", {**key(r.config, ANALYSES[r.analysis]), "Selection": r.selection, "SumWGammaNeutron": float(r.sum_w_gamma_neutron), "SumWRadiological": float(r.sum_w_radiological),
                            "RadiologicalFraction": float(r.radiological_fraction), "NMCGammaNeutron": int(r.n_mc_gamma_neutron), "NMCRadiological": int(r.n_mc_radiological),
                            "NMCRadiologicalNoFiducial": int(r.n_mc_radiological_no_fiducial), "RadiologicalAllRejected": "yes" if r.radiological_all_rejected else "no"})

for r in old("survivor_statistics").itertuples():
    add("SurvivorStatistics", {**key(r.config, ANALYSES[r.analysis], r.species), "NMCWindowCut": int(r.n_mc_window_cut), "NMCRecoFiducial": int(r.n_mc_reco_fiducial),
                               "NMCTruthPipeline": int(r.n_mc_truth_pipeline), "SumWRecoFiducial": float(r.sum_w_reco_fiducial), "MeanWPerEvent": float(r.mean_w_per_event),
                               "NEff": float(r.n_eff), "LargestEventShare": float(r.largest_event_share), "ShareOfBackgroundReco": float(r.share_of_background_reco),
                               "ShareOfBackgroundTruthPipeline": float(r.share_of_background_truth_pipeline)})

for (an, c, sp, ag), g in old("survivors_events").groupby(["analysis", "config", "species", "agree"]):
    add("SurvivorEvents", {**key(c, ANALYSES[an], sp), "Agree": "agree" if ag else "disagree", "NMC": int(len(g)), "Dist3D": fl(g.dist3d_cm),
                           "RecoX": fl(g.reco_x), "RecoY": fl(g.reco_y), "RecoZ": fl(g.reco_z), "TruthX": fl(g.truth_x), "TruthY": fl(g.truth_y), "TruthZ": fl(g.truth_z), "Weight": fl(g.weight)})

# ---- write ---------------------------------------------------------------------------------------
FEEDS = {
    "WallCdf": "fig 1 wall proximity (cumulative fraction vs distance; Position = truth/reco, one panel per config)",
    "ShellRatio": "fig 2 wall enhancement per shell (Shell x Ratio, error RatioError; a table column pivot works too)",
    "Residuals": "fig 3 top row: reco - truth histograms per axis (panel = Variable); bin lower edges in Residual, 5 cm bins, outer bins hold overflow",
    "ResidualsWide": "fig 3 for panels whose median is off scale in Residuals (VD gamma / neutron X); 20 cm bins over +-700 cm",
    "ResidualSummary": "fig 3 medians and fractions (table); Median is the number to print",
    "FlashMismatch": "table: position- and purity-based flash-mismatch rates (Selection = window / after_cut; Purity* need MatchedOpFlashPur, threshold in meta.json)",
    "TruthKeyCheck": "table: which truth key (Main / MainParent / End / SignalParticle) lies within 30 cm of the reconstructed cluster; IsConfiguredKey marks the key the study uses",
    "Cuts": "table: default best cut per analysis",
    "FiducialVolumes": "table 3: best fiducial volumes (Variant = default / fiduc_truth)",
    "PassFractions": "fig 4 pass fractions (Sample includes neutron_agree / neutron_disagree, VariantIndex 0-3)",
    "Significance": "fig 5 and table 3: significance per Variant (default / fiduc_truth / fiduc_truth_refvol)",
    "FaceComposition": "fig 6 entry-face composition (table, or stacked via a pivot); OutsideFraction is a flag, not a fifth face",
    "XEntryHist": "fig 7 |RecoX - truth X| histograms near the entry face (Selection = window / after_cut); bins 0-700 cm, last bin overflow",
    "XEntrySummary": "fig 7 table (median, fractions above 30 / 100 cm, RecoX at edge)",
    "ScanCurves": "fig 8 signal efficiency vs surviving gamma+neutron weight, X only / Y only (Variable), one line per Mode",
    "ScanGrid": "full (FiducialX, FiducialY) scan grid, all modes",
    "BestFoM": "fig 9 best S/sqrt(B) per Mode and Scan; IdenticalVolumeAs flags scans that landed on the same volume",
    "SignalRecovered": "fig 10 signal efficiency and gamma+neutron weight under reco X vs truth X",
    "BackgroundSplit": "table 11 radiological share of the background; RadiologicalAllRejected = yes when the selection removed every radiological MC event",
    "SurvivorStatistics": "table 12 surviving MC statistics",
    "SurvivorEvents": "fig 3 bottom row: per-event neutron (VD: gamma) survivors split by Agree; arrays, one entry per MC event",
}

DST.mkdir(parents=True, exist_ok=True)
for f in DST.glob("*_all_*.pkl"):
    f.unlink()
schema = {}
for kind, rows in ROWS.items():
    df = pd.DataFrame(rows)
    df = df.dropna(axis=1, how="all")
    lead = [c for c in KEYS + ["Sample"] if c in df.columns]
    df = df[lead + [c for c in df.columns if c not in lead]]
    for config, g in df.groupby("Config"):
        g = g.reset_index(drop=True)
        g.to_pickle(DST / f"{config}_all_{kind}.pkl", protocol=4)
    schema[kind] = list(df.columns)
    print(f"[repo-export] {kind:20s} {len(df):6d} rows  {list(df.columns)}")
    if args.sync:
        # Synced copy: the plot repo's sync pulls analysis/{analysis}/{config}/marley/{folder}/{label}/*.pkl for registered
        # labels and its macros load {config}_{name}_{datafile}.pkl filtering on Name, so Name must be "marley" here.
        sync_dirs = {"DayNight": "day-night", "HEP": "hep", "Sensitivity": "sensitivity"}
        for (config, analysis), g in df.groupby(["Config", "Analysis"]):
            out = Path(args.sync_root) / sync_dirs[analysis] / config / "marley" / "truncated" / "fiduc_truth"
            out.mkdir(parents=True, exist_ok=True)
            g = g.assign(Name="marley", Study="fiduc_truth").reset_index(drop=True)
            g.to_pickle(out / f"{config}_marley_{analysis}_TruthPosition{kind}.pkl", protocol=4)

# reference numbers (acceptance checks)
def one(kind, **kw):
    df = pd.concat([pd.read_pickle(f) for f in DST.glob(f"*_all_{kind}.pkl")])
    for k, v in kw.items():
        df = df[df[k] == v]
    return df

wc = one("WallCdf", Config="hd_1x2x6_centralAPA", Position="truth", Analysis="DayNight")
checks = {"WallCdf_HD_central_truth_20cm": {r.Sample: float(np.interp(20.0, r.Distance, r.CDF)) for r in wc.itertuples() if r.Sample in ("gamma", "marley", "neutron")}}
sg_ = one("Significance", Config="hd_1x2x6_lateralAPA", Analysis="DayNight")
checks["Significance_HD_lateral_DayNight"] = {r.Variant: float(r.Significance) for r in sg_.itertuples()}
pf = one("PassFractions", Config="hd_1x2x6_centralAPA", Analysis="DayNight", Sample="neutron_disagree").sort_values("VariantIndex")
checks["PassFractions_DayNight_HD_central_neutron_disagree"] = {"PassFraction": [float(x) for x in pf.PassFraction], "NMC": int(pf.NMC.iloc[0])}
bf = one("BestFoM", Config="hd_1x2x6_centralAPA", Analysis="DayNight", Mode="reco", Scan="X and Y")
checks["BestFoM_DayNight_HD_central_reco_XandY"] = float(bf.FoM.iloc[0])

meta = json.loads((SRC / "meta.json").read_text())
meta.update({"schema": "LOWE_RECONSTRUCTION_PUBLICATION macros", "source": str(SRC), "min_mc_for_fom": MIN_MC, "purity_threshold": 0.5, "kinds": schema, "checks": checks,
             "notes_repo": ["Analysis-independent kinds are repeated for all three analysis labels.",
                            "Sensitivity 'Score' = 0.5*(chi2_solar(theta_react) + chi2_react(theta_solar)) (04_best_cuts.py, 06_significance.py): a wrong-hypothesis Delta chi^2, so SignificanceUnit \\Delta\\chi^2 is correct; the sigma-equivalent is sqrt(Score). DayNight and HEP are sigma.",
                            "NMC of Residuals* and XEntryHist is the total; NMCPerBin holds the per-bin MC counts."]})
(DST / "meta.json").write_text(json.dumps(meta, indent=2))

lines = ["# Truth-position export for the plot repo", "",
         "Generated by `src/tools/export_truth_position_repo.py` from `export/` (SOLAR). File name `{config}_all_{Kind}.pkl`; load with `--configs C --names all --datafile Kind`.",
         "Key columns of every row: `Analysis` (DayNight|HEP|Sensitivity), `Geometry` (hd|vd), `Config`, `Name` (= `all`), `Study` (= `default`), and `Sample` wherever a sample or group exists.",
         "Curves are single rows holding equal-length lists. Categorical values are strings. `meta.json` holds the selection, the git revision and a `checks` block with reference numbers.",
         "Synced copies (picked up by `sync_solar_data.sh`, label `fiduc_truth`): `output/data/analysis/{day-night|hep|sensitivity}/{config}/marley/truncated/fiduc_truth/"
         "{config}_marley_{Analysis}_TruthPosition{Kind}.pkl`, same rows with `Name` = `marley` and `Study` = `fiduc_truth`; load with `--configs C --names marley --datafile {Analysis}_TruthPosition{Kind}`.", "",
         "| Kind | columns | feeds |", "|---|---|---|"]
for kind, cols in schema.items():
    lines.append(f"| `{kind}` | {', '.join(f'`{c}`' for c in cols)} | {FEEDS.get(kind, '')} |")
lines += ["", "## Notes", "- " + "\n- ".join(meta["notes_repo"]),
          "- `FoM` = SignalEfficiency * SumWSignalBefore / sqrt(SumWGammaNeutron), NaN where NMCGammaNeutron < 5 or the weight is 0 (ScanGrid, ScanCurves, BestFoM).",
          "- Radiological is excluded from every FoM (2-4 MC events with weight ~1e5); `SumWRadiological` is listed for reference.",
          "- Not expressible with the macros (no data change helps): grouped/stacked bars, text annotations (medians, MC counts, xN labels), highlighted table rows."]
(DST / "README.md").write_text("\n".join(lines) + "\n")

tar = BASE / "truth_position_export_repo.tar.gz"
with tarfile.open(tar, "w:gz") as t:
    t.add(DST, arcname=DST.name)
print(f"[repo-export] {DST}  ({len(list(DST.glob('*.pkl')))} pickles)  {tar}")
