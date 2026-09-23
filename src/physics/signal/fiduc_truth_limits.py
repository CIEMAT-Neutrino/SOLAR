"""
fiduc_truth_limits.py — what limits the gain of truth-position fiducialisation (fiduc_truth, Ch. 9.1)
====================================================================================================
Figures for the fiduc_truth discussion. Reads only existing outputs (no analysis inputs):

  composition  HEP background composition per energy bin, default vs fiduc_truth, from the analysis's own
               HEP_Counts.pkl (significance_plot.py; Raw spectra, the ones the ProfileLikelihood fits).
               Shows what truth fiducialisation removes and what it cannot (8B is spatially identical to hep).
  migration    Reconstructed SolarEnergy vs MainK of the gamma and neutron events that survive the default HEP
               cut and volume, from the truth_position_study.py caches. MainK is the true kinetic energy of the
               main contributing particle (a background gamma; one capture line for neutrons, which also capture
               outside the argon: lines at ~7.6-10.8 MeV). SolarEnergy is a neutrino-energy estimator and sits
               ~+5 MeV above MainK for the signal too, so the offset is the estimator, not a mismeasurement: the
               in-window gammas have true energies of 11-14 MeV (the generator endpoint), i.e. signal-like.
  summary      DayNight / HEP significance: default, fiduc_truth_refvol (truth position at the reco volume) and
               fiduc_truth (truth position, truth volume), read from the highest_* JSONs.

  export       Pickles for the plot repo in the synced analysis tree (see stage_export):
               output/data/analysis/{day-night|hep}/{config}/marley/truncated/fiduc_truth/
                 {config}_marley_{DayNight|HEP}_FiducTruthSummary.pkl      default / refvol / truth significance, cut, volume
                 {config}_marley_{DayNight|HEP}_FiducTruthComposition.pkl  the three variants' Counts spectra (Variant column)
                 {config}_marley_HEP_FiducTruthMigration.pkl              per-event SolarEnergy vs MainK, gamma and neutron

Output: output/images/solar/truth_position/limits_{stage}.{png,pdf}

Usage
-----
  python3 src/physics/signal/fiduc_truth_limits.py                     # all stages
  python3 src/physics/signal/fiduc_truth_limits.py --stage export
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lib import root  # noqa: E402
from lib.fiducial import accepted_flash_planes, build_fiducial_spatial_mask, get_best_fiducial  # noqa: E402
from lib.background import is_surface_background  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
parser.add_argument("--stage", nargs="+", choices=["composition", "migration", "summary", "export"], default=["composition", "migration", "summary", "export"])
parser.add_argument("--config", nargs="+", default=[
    "hd_1x2x6_centralAPA", "hd_1x2x6_lateralAPA", "vd_1x8x14_3view_30deg_nominal", "vd_1x8x14_3view_30deg_shielded"])
parser.add_argument("--folder", default="Truncated")
parser.add_argument("--energy", default="SolarEnergy")
parser.add_argument("--emin", type=float, default=12.0, help="Lower edge of the plotted HEP region [MeV]")
parser.add_argument("--emax", type=float, default=24.0, help="Upper edge of the plotted HEP region [MeV]")
parser.add_argument("--hep_window", nargs=2, type=float, default=[14.0, 30.0])
parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
args = parser.parse_args()

ROOT = str(root)
DATA = "/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR"
CACHE = Path(ROOT) / "output/data/solar/truth_position"
IMAGES = Path(ROOT) / "output/images/solar/truth_position"

# Same palette and style as truth_position_study.py, so the thesis figures read as one set.
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8f8e89", "#e3e2dc"
COLORS = {"hep": "#2a78d6", "gamma": "#eb6834", "neutron": "#1baf7a", "radiological": "#4a3aa7", "8B": "#8f8e89"}
LABELS = {"hep": "hep (signal)", "gamma": "Gamma", "neutron": "Neutron", "radiological": "Radiological", "8B": "⁸B (solar)"}
NICE = {"hd_1x2x6_centralAPA": "HD central APA", "hd_1x2x6_lateralAPA": "HD lateral APA",
        "vd_1x8x14_3view_30deg_nominal": "VD nominal", "vd_1x8x14_3view_30deg_shielded": "VD shielded"}
BKG_ORDER = ["8B", "radiological", "neutron", "gamma"]

plt.rcdefaults()
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8, "legend.frameon": False,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 2.0, "figure.dpi": 100, "pdf.fonttype": 42,
})


def save(fig, name):
    IMAGES.mkdir(parents=True, exist_ok=True)
    for fmt in args.formats:
        fig.savefig(IMAGES / f"{name}.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {IMAGES / name}.{{{','.join(args.formats)}}}")


def hep_counts(config, study):
    """Raw per-bin counts (events / MeV / 20 yr) by component at the study's HEP working point."""
    p = f"{DATA}/HEP/{config}/marley/{args.folder.lower()}/{study}/{config}_marley_HEP_Counts.pkl"
    df = pd.read_pickle(p)
    df = df[df.SpectrumType == "Raw"]
    energy = np.asarray(df.iloc[0]["Energy"], float)
    counts = {r["Component"]: np.asarray(r["Counts"], float) for _, r in df.iterrows()}
    cut = f"NHits {int(df.iloc[0].NHits)} · AdjCl {int(df.iloc[0].AdjCl)} · OpHits {int(df.iloc[0].OpHits)}"
    return energy, counts, cut


# ==========================================================================================
def stage_composition():
    fig, axes = plt.subplots(len(args.config), 2, figsize=(9.0, 2.35 * len(args.config)), sharex=True, squeeze=False)
    rows = []
    for i, config in enumerate(args.config):
        spectra = {st: hep_counts(config, st) for st in ("default", "fiduc_truth")}
        ymax = max(max(sum(c[k] for k in c if k != "hep").max(), c["hep"].max()) for _, c, _ in spectra.values())
        for j, (study, (energy, counts, cut)) in enumerate(spectra.items()):
            ax = axes[i, j]
            m = (energy >= args.emin) & (energy <= args.emax)
            e, width = energy[m], 1.0
            bottom = np.zeros(m.sum())
            for comp in BKG_ORDER:
                if comp not in counts:
                    continue
                v = counts[comp][m]
                ax.bar(e, v, width=width * 0.92, bottom=bottom, color=COLORS[comp], label=LABELS[comp] if (i, j) == (0, 0) else None,
                       linewidth=0)
                bottom += v
            ax.step(e, counts["hep"][m], where="mid", color=COLORS["hep"], label=LABELS["hep"] if (i, j) == (0, 0) else None)
            ax.set_yscale("log")
            ax.set_ylim(1e-1, ymax * 3)
            ax.axvspan(args.emin, args.hep_window[0], color=GRID, alpha=0.5, lw=0)
            title = f"{NICE[config]} — {'default (reco fiducial)' if study == 'default' else 'fiduc_truth (truth fiducial)'}"
            ax.set_title(title, loc="left", fontsize=9)
            ax.text(0.99, 0.95, cut, transform=ax.transAxes, ha="right", va="top", fontsize=7, color=INK2)
            if j == 0:
                ax.set_ylabel("events / MeV / 20 yr")
            if i == len(args.config) - 1:
                ax.set_xlabel("Reconstructed energy [MeV]")
            w = (energy >= args.hep_window[0]) & (energy <= args.hep_window[1])
            b = {c: float(counts[c][w].sum()) for c in counts if c != "hep"}
            rows.append({"config": config, "study": study, "hep": float(counts["hep"][w].sum()), **b, "B_total": sum(b.values())})
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    save(fig, "limits_composition_hep")
    table = pd.DataFrame(rows)
    out = CACHE / "tables" / "limits_composition_hep.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3g}"))


# ==========================================================================================
MIGRATION_COMPONENTS = ["gamma", "neutron"]


def migration_events(config):
    """Gamma/neutron events passing quality, the default HEP cut and the default (reco) HEP volume, all energies."""
    info = json.load(open(f"{ROOT}/config/{config}/{config}_config.json"))
    cut = json.load(open(f"{ROOT}/config/{config}/hep-json/{args.folder.lower()}/{config}_highest_HEP.json"))[config][args.energy]
    fids = json.load(open(f"{ROOT}/config/analysis/fiducial/{args.folder.lower()}/BestFiducials.json"))
    fid = get_best_fiducial(fids, config, args.energy, "HEP")
    dx = info["DETECTOR_SIZE_X"] + 2 * info["DETECTOR_GAP_X"]
    dy = info["DETECTOR_SIZE_Y"] + 2 * info["DETECTOR_GAP_Y"]
    events = {}
    for comp in MIGRATION_COMPONENTS:
        d = dict(np.load(CACHE / config / f"{config}_{comp}.npz", allow_pickle=True))
        a = dict(np.load(CACHE / config / f"{config}_{comp}_aux.npz", allow_pickle=True))
        q = accepted_flash_planes(d["plane"], ROOT, True) & (d["pe"] > 0)
        if is_surface_background(ROOT, comp):
            q &= (d["surface"] >= 0) & (d["surface"] < 3)
        sel = q & (d["NHits"] >= cut["NHits"]) & (d["AdjClNum"] < cut["AdjCl"]) & (d["OpHits"] >= cut["OpHits"])
        run = {"Reco": {"RecoX": d["reco_x"], "RecoY": d["reco_y"], "RecoZ": d["reco_z"]}}
        sel &= build_fiducial_spatial_mask(run, config, dx, dy, info, args.folder, fid)
        events[comp] = (d["energy"][sel], a["MainK"][sel], d["w"][sel])
    return {"NHits": int(cut["NHits"]), "OpHits": int(cut["OpHits"]), "AdjCl": int(cut["AdjCl"])}, fid, events


def stage_migration():
    comps = MIGRATION_COMPONENTS
    fig, axes = plt.subplots(len(comps), len(args.config), figsize=(2.6 * len(args.config), 2.6 * len(comps)), sharex=True, sharey=True, squeeze=False)
    lo, hi = args.hep_window
    rows = []
    for j, config in enumerate(args.config):
        _, _, events = migration_events(config)
        for i, comp in enumerate(comps):
            reco, true, w = events[comp]
            ax = axes[i, j]
            inwin = (reco >= lo) & (reco <= hi)
            ax.axhspan(lo, 25, color=COLORS["hep"], alpha=0.08, lw=0)
            ax.plot([0, 25], [0, 25], color=MUTED, lw=1.0, ls="--")
            size = 8 + 40 * (w / w.max()) if w.size and w.max() > 0 else 8
            ax.scatter(true[~inwin], reco[~inwin], s=8, color=MUTED, alpha=0.25, lw=0)
            ax.scatter(true[inwin], reco[inwin], s=size[inwin] if np.ndim(size) else size, color=COLORS[comp], alpha=0.7,
                       edgecolor="white", linewidth=0.5)
            f_below = float(w[inwin & (true < lo)].sum() / w[inwin].sum()) if w[inwin].sum() > 0 else np.nan
            neff = float(w[inwin].sum() ** 2 / (w[inwin] ** 2).sum()) if w[inwin].sum() > 0 else 0.0
            ax.text(0.03, 0.97, f"in window: {int(inwin.sum())} MC (N$_{{eff}}$ {neff:.0f})\nweight with MainK < {lo:g} MeV: {100 * f_below:.0f}%",
                    transform=ax.transAxes, ha="left", va="top", fontsize=7, color=INK)
            ax.set_xlim(0, 25)
            ax.set_ylim(0, 25)
            if i == 0:
                ax.set_title(NICE[config], fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{LABELS[comp]}\nreconstructed {args.energy} [MeV]")
            if i == len(comps) - 1:
                ax.set_xlabel("MainK (true) [MeV]")
            rows.append({"config": config, "component": comp, "n_mc_in_window": int(inwin.sum()), "neff": neff,
                         "weight_frac_MainK_below_window": f_below,
                         "median_reco_minus_MainK": float(np.median(reco[inwin] - true[inwin])) if inwin.any() else np.nan})
    fig.tight_layout()
    save(fig, "limits_migration_hep")
    table = pd.DataFrame(rows)
    table.to_csv(CACHE / "tables" / "limits_migration_hep.csv", index=False)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3g}"))


# ==========================================================================================
VARIANTS = {"default": "", "fiduc_truth_refvol": "_fiduc_truth_refvol", "fiduc_truth": "_fiduc_truth"}


def highest_entry(config, analysis, label):
    """Best-cut entry (Values = significance at the analysis exposure, NHits/OpHits/AdjCl) of one study."""
    sub = {"HEP": "hep-json", "DayNight": "daynight-json"}[analysis]
    p = f"{ROOT}/config/{config}/{sub}/{args.folder.lower()}/{config}_highest{label}_{analysis}.json"
    return json.load(open(p))[config][args.energy]


def stage_summary():
    def value(config, analysis, label):
        return float(highest_entry(config, analysis, label)["Values"])

    analyses = ["DayNight", "HEP"]
    fig, axes = plt.subplots(1, len(analyses), figsize=(9.0, 3.0), squeeze=False)
    rows = []
    x = np.arange(len(args.config))
    for k, analysis in enumerate(analyses):
        ax = axes[0, k]
        dflt = np.array([value(c, analysis, "") for c in args.config])
        truth = np.array([value(c, analysis, "_fiduc_truth") for c in args.config])
        refvol = np.array([value(c, analysis, "_fiduc_truth_refvol") for c in args.config])
        bw = 0.26
        ax.bar(x - bw, dflt, width=bw * 0.92, color=MUTED, label="default: reco position, reco volume")
        # Same hue as fiduc_truth (same truth-position family); the hatch is the secondary encoding.
        ax.bar(x, refvol, width=bw * 0.92, color="#c9c4ea", edgecolor=COLORS["radiological"], hatch="////", linewidth=0.8,
               label="fiduc_truth_refvol: truth position, reco volume")
        ax.bar(x + bw, truth, width=bw * 0.92, color=COLORS["radiological"], label="fiduc_truth: truth position, truth volume")
        for xi, (a, r, b) in enumerate(zip(dflt, refvol, truth)):
            ax.text(xi + bw, b, f"{b / a:.1f}×", ha="center", va="bottom", fontsize=7, color=INK)
            rows.append({"analysis": analysis, "config": args.config[xi], "default": a, "fiduc_truth_refvol": r,
                         "fiduc_truth": b, "ratio_truth": b / a, "ratio_refvol": r / a})
        ax.set_xticks(x, [NICE[c].replace(" APA", "") for c in args.config], fontsize=8)
        ax.set_ylabel(f"{analysis} significance [σ, 30 yr]")
        ax.set_title(analysis, loc="left")
        ax.grid(axis="x", visible=False)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    save(fig, "limits_summary")
    table = pd.DataFrame(rows)
    table.to_csv(CACHE / "tables" / "limits_summary.csv", index=False)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


# ==========================================================================================
# export: pickles in the synced analysis tree, one file per (analysis, config, kind), for the plot repo
# (LOWE_RECONSTRUCTION_PUBLICATION: sync_solar_data.sh pulls analysis/{analysis}/{config}/marley/{folder}/{label}/*.pkl
#  for registered labels; its macros load {config}_{name}_{datafile}.pkl and filter on Name).
SYNC_ROOT = Path(ROOT) / "output/data/analysis"
ANALYSIS_DIR = {"DayNight": "day-night", "HEP": "hep"}
SYNC_LABEL = "fiduc_truth"


def sync_path(analysis, config, kind):
    d = SYNC_ROOT / ANALYSIS_DIR[analysis] / config / "marley" / args.folder.lower() / SYNC_LABEL
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{config}_marley_{analysis}_{kind}.pkl"


def write_sync(df, analysis, config, kind):
    lead = ["Config", "Name", "EnergyLabel", "Variable", "Analysis", "Geometry", "Variant", "Component"]
    df = df[[c for c in lead if c in df.columns] + [c for c in df.columns if c not in lead]]
    p = sync_path(analysis, config, kind)
    df.reset_index(drop=True).to_pickle(p, protocol=4)
    print(f"[sync] {p.relative_to(ROOT)}  ({len(df)} rows)")


def base_row(config, analysis, variable):
    return {"Config": config, "Name": "marley", "EnergyLabel": args.energy, "Variable": variable, "Analysis": analysis,
            "Geometry": config.split("_")[0], "Study": SYNC_LABEL}


def stage_export():
    from lib import get_analysis_exposure
    fids = {"default": json.load(open(f"{ROOT}/config/analysis/fiducial/{args.folder.lower()}/BestFiducials.json")),
            "fiduc_truth": json.load(open(f"{ROOT}/config/analysis/fiducial/{args.folder.lower()}/BestFiducials_fiduc_truth.json"))}
    for config in args.config:
        for analysis in ("DayNight", "HEP"):
            # FiducTruthSummary: one row per variant; refvol holds the reco (default) volume by construction.
            exposure = float(get_analysis_exposure(ROOT, analysis.upper()))
            rows, reference = [], None
            for variant, label in VARIANTS.items():
                e = highest_entry(config, analysis, label)
                vol = get_best_fiducial(fids["fiduc_truth" if variant == "fiduc_truth" else "default"], config, args.energy, analysis.upper())
                reference = reference or float(e["Values"])
                rows.append({**base_row(config, analysis, "Significance"), "Variant": variant,
                             "Position": "reco" if variant == "default" else "truth", "Volume": "truth" if variant == "fiduc_truth" else "reco",
                             "Significance": float(e["Values"]), "SignificanceUnit": r"\sigma", "RatioToDefault": float(e["Values"]) / reference,
                             "NHits": int(e["NHits"]), "OpHits": int(e["OpHits"]), "AdjCl": int(e["AdjCl"]),
                             "FiducialX": vol["FiducialX"], "FiducialY": vol["FiducialY"], "FiducialZ": vol["FiducialZ"], "FiducialUnit": "cm",
                             "Exposure": exposure, "ExposureUnit": "year"})
            write_sync(pd.DataFrame(rows), analysis, config, "FiducTruthSummary")

            # FiducTruthComposition: the synced {Analysis}_Counts spectra of the three variants in one file.
            frames = []
            for variant in VARIANTS:
                p = SYNC_ROOT / ANALYSIS_DIR[analysis] / config / "marley" / args.folder.lower() / variant / f"{config}_marley_{analysis}_Counts.pkl"
                df = pd.read_pickle(p).copy()
                df["Variant"] = variant
                df["Study"] = SYNC_LABEL
                frames.append(df.drop(columns=[c for c in ("Significance", "SignificanceUnit", "SignificanceLabel", "Metric", "MetricError", "MetricUnit", "MetricLabel") if c in df.columns]))
            write_sync(pd.concat(frames, ignore_index=True), analysis, config, "FiducTruthComposition")

        # FiducTruthMigration (HEP): per-event reco energy vs MainK of the gamma/neutron that pass the default cut and volume.
        cut, fid, events = migration_events(config)
        lo, hi = args.hep_window
        rows = []
        for comp, (reco, true, w) in events.items():
            inwin = (reco >= lo) & (reco <= hi)
            wi = w[inwin]
            rows.append({**base_row(config, "HEP", "MainK"), "Component": comp, "Variant": "default",
                         "TrueEnergy": [float(x) for x in true], "RecoEnergy": [float(x) for x in reco], "Weight": [float(x) for x in w],
                         "InWindow": [int(x) for x in inwin], "NMC": int(len(w)), "NMCInWindow": int(inwin.sum()),
                         "NEffInWindow": float(wi.sum() ** 2 / (wi ** 2).sum()) if wi.sum() > 0 else 0.0,
                         "WeightFractionTrueBelowWindow": float(w[inwin & (true < lo)].sum() / wi.sum()) if wi.sum() > 0 else np.nan,
                         "MedianRecoMinusTrue": float(np.median(reco[inwin] - true[inwin])) if inwin.any() else np.nan,
                         "WindowLow": lo, "WindowHigh": hi, "EnergyUnit": "MeV", "WeightLabel": "SignalParticleWeight",
                         **cut, "FiducialX": fid["FiducialX"], "FiducialY": fid["FiducialY"], "FiducialZ": fid["FiducialZ"], "FiducialUnit": "cm"})
        write_sync(pd.DataFrame(rows), "HEP", config, "FiducTruthMigration")


for stage in args.stage:
    {"composition": stage_composition, "migration": stage_migration, "summary": stage_summary, "export": stage_export}[stage]()
