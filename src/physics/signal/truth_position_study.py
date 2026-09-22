"""
truth_position_study.py — Why does the truth-position fiducial study (fiduc_truth) barely move the result?
========================================================================================================
Event-level diagnostics behind the fiduc_truth discussion of Chapter 9.2.2-9.2.3.

Two stages
----------
  extract   Read ONE (config, sample) from the analysis inputs and cache the few per-event arrays the
            diagnostics need (reco and truth position, weight, energy, quality and topology branches).
            Slow (~3 min per sample); run it through src/tools/run_truth_position.sh, which loops over
            configs and samples under the container.
                output/data/solar/truth_position/{config}/{config}_{sample}.npz
  faces     Entry-face composition, X residuals near the entry faces, mixed truth/reco X-Y scans and
            flash-mismatch rates (output/docs/truth_position_faces_{analysis}.md, figures faces_*, x_residuals_*, xy_*).
  export    Tidy numeric DataFrames (pickle) of everything the figures draw, for plotting in another repository
            (output/data/solar/truth_position/export/*.pkl + meta.json; both analyses in one run).
  extract_aux  Per-event alternative truth positions (Main, MainParent, End, SignalParticle, TruthX) and MatchedOpFlashPur, aligned with the
            extract cache (output/data/solar/truth_position/{config}/{config}_{sample}_aux.npz). Feeds the truth-key check and the purity rates.
  plot      Read the caches and BestFiducials / highest-cut / significance JSONs and write the figures
            and tables (fast, no analysis inputs needed):
                output/images/solar/truth_position/*.{png,pdf}
                output/docs/truth_position_tables_{analysis}.md  (+ CSV in output/data/solar/truth_position/tables/)

Truth position keys are those of lib.fiducial.get_truth_pos_keys: SignalParticleX/Y/Z for marley,
EndX/Y/Z for gamma, MainX/Y/Z for neutron and radiological.

Selection used by the diagnostics (mirrors 01_fiducialize.py / 03_analysis.py)
------------------------------------------------------------------------------
  quality   matched flash on an accepted plane, MatchedOpFlashPE > 0, and the surface cut for
            surface-filtered backgrounds (Truncated: 0 <= SignalParticleSurface < 3)
  window    emin <= SolarEnergy <= emax  (default 10-20 MeV, the region where the solar signal is analysed)
  cut       NHits >= N, AdjClNum < A, MatchedOpFlashNHits >= O from the default DayNight best cut
            (config/{config}/daynight-json/{folder}/{config}_highest_DayNight.json); the wall-proximity
            figures skip it unless --apply_cuts_walls, because the survivors are only a handful of MC events
  fiducial  lib.fiducial.build_fiducial_spatial_mask at the default (reco) or truth best volume of --analysis
  truth     for the truth pipeline the fiducial is evaluated on the truth position, and additionally
            requires the truth containment cut and, for backgrounds, the position-consistency cut
            (lib.fiducial.truth_match_purity_mask), exactly as `--truth_fiducial` does.

Caveats worth repeating next to the figures
-------------------------------------------
  * "Distance to the nearest face" uses the walls of the active box: HD central x = -360 and +360 (the plane x = 0 is
    interior); HD lateral only x = 0 (background piles up there, and the pipeline's own X cut is made from it), not the far
    x = 360; VD both X faces and the Y and Z faces. See x_faces().
  * VD background truth X runs past the active volume while RecoX is clipped to it, so truth and reco
    distances are not comparable there in X.
  * weights are SignalParticleWeight only (no oscillation, no MC-support gate, no smoothing): these are
    diagnostics of position information, not replacement significances.

Usage
-----
  python3 src/physics/signal/truth_position_study.py --stage extract --config hd_1x2x6_centralAPA --signals gamma
  python3 src/physics/signal/truth_position_study.py --stage plot
  python3 src/physics/signal/truth_position_study.py --stage plot --config hd_1x2x6_centralAPA --folder Truncated
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import numpy as np

parser = argparse.ArgumentParser(
    description="Truth-vs-reco position diagnostics for the fiduc_truth study",
    formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=36, width=120),
)
parser.add_argument("--stage", choices=["extract", "extract_aux", "plot", "faces", "export"], required=True)
parser.add_argument("--config", nargs="+", default=[
    "hd_1x2x6_centralAPA", "hd_1x2x6_lateralAPA",
    "vd_1x8x14_3view_30deg_nominal", "vd_1x8x14_3view_30deg_shielded",
], help="extract: exactly one config. plot: configs to draw (first one is the main panel).")
parser.add_argument("--signals", nargs="+", default=["marley", "gamma", "neutron", "radiological"])
parser.add_argument("--folder", default="Truncated", choices=["Truncated", "Reduced", "Nominal"])
parser.add_argument("--analysis", default="DAYNIGHT", choices=["DAYNIGHT", "HEP", "SENSITIVITY"],
                    help="Analysis whose best fiducial volume and cut define the pass fractions (default DAYNIGHT)")
parser.add_argument("--energy", default="SolarEnergy")
parser.add_argument("--emin", type=float, default=10.0, help="Lower energy edge [MeV] (the DayNight analysis threshold is 0, which lets the sub-10 MeV radiological/neutron tail dominate)")
parser.add_argument("--emax", type=float, default=20.0, help="Upper energy edge [MeV]")
parser.add_argument("--apply_cuts_walls", action=argparse.BooleanOptionalAction, default=False,
                    help="Apply the NHits/OpHits/AdjCl cut in the wall-proximity and shell-ratio figures")
parser.add_argument("--purity_threshold", type=float, default=0.5, help="MatchedOpFlashPur below this counts as a low-purity flash match (needs the extract_aux caches)")
parser.add_argument("--export_dir", default="export", help="export stage: sub-folder of output/data/solar/truth_position to write the pickles to")
parser.add_argument("--agree_cm", type=float, default=30.0, help="|reco - truth| below which a position 'agrees'")
parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=False, help="extract: overwrite an existing cache")
args = parser.parse_args()

from lib import root, load_analysis_info                       # noqa: E402

ROOT = str(root)
CACHE = Path(ROOT) / "output/data/solar/truth_position"
IMAGES = Path(ROOT) / "output/images/solar/truth_position"
TABLES = CACHE / "tables"
DOC = Path(ROOT) / f"output/docs/truth_position_tables_{args.analysis.lower()}.md"
BKG = ["gamma", "neutron", "radiological"]


def config_info(config):
    return json.loads(open(f"{ROOT}/config/{config}/{config}_config.json").read())


# ==========================================================================================
# extract
# ==========================================================================================

def extract_one(config, name):
    from lib import load_multi, compute_reco_workflow          # heavy: only the extract stage needs it
    from lib.fiducial import get_truth_pos_keys, truth_containment_mask

    out = CACHE / config / f"{config}_{name}.npz"
    if out.exists() and not args.rewrite:
        print(f"[skip] {out} exists (use --rewrite)")
        return
    out.parent.mkdir(parents=True, exist_ok=True)

    info = config_info(config)
    configs = {config: [name]}
    run, _ = load_multi(configs, preset="SIGNIFICANCE", branches={"Config": ["Geometry"]}, debug=False)
    run = compute_reco_workflow(
        run, configs,
        params=({"DEFAULT_SIGNAL_WEIGHT": ["truth", "osc"], "DEFAULT_SIGNAL_NADIR": ["mean", "day", "night"],
                 "PARTICLE_TYPE": "signal", "PARTICLE_WEIGHTING": "volume", "OSCILLATION_BACKEND": "nufast"}
                if name == "marley" else {"PARTICLE_TYPE": "background", "PARTICLE_WEIGHTING": "histogram"}),
        rm_branches=False, workflow="SIGNIFICANCE", debug=False,
    )
    R = run["Reco"]
    truth_keys = get_truth_pos_keys(ROOT, name)
    d = {}
    for axis, key in zip("xyz", ("RecoX", "RecoY", "RecoZ")):
        d[f"reco_{axis}"] = np.asarray(R[key], float)
    for axis, key in zip("xyz", truth_keys):
        d[f"truth_{axis}"] = np.asarray(R[key], float)
    d["w"] = np.asarray(R["SignalParticleWeight"], float)
    d["energy"] = np.asarray(R[args.energy], float)
    d["surface"] = np.asarray(R["SignalParticleSurface"])
    d["plane"] = np.asarray(R["MatchedOpFlashPlane"])
    d["pe"] = np.asarray(R["MatchedOpFlashPE"])
    d["contained"] = truth_containment_mask(run, info, truth_keys)
    d["NHits"] = np.asarray(R["NHits"], float)
    d["AdjClNum"] = np.asarray(R["AdjClNum"], float)
    d["OpHits"] = np.asarray(R["MatchedOpFlashNHits"], float)
    d["truth_keys"] = np.array(truth_keys)
    np.savez_compressed(out, **d)
    print(f"[ok] {out}  ({len(d['w'])} events, truth keys {truth_keys})")


def extract_aux_one(config, name):
    from lib import load_multi, compute_reco_workflow

    out = CACHE / config / f"{config}_{name}_aux.npz"
    if out.exists() and not args.rewrite:
        print(f"[skip] {out} exists (use --rewrite)")
        return
    ref = CACHE / config / f"{config}_{name}.npz"
    n_ref = len(np.load(ref)["w"]) if ref.exists() else None
    configs = {config: [name]}
    run, _ = load_multi(configs, preset="SIGNIFICANCE", branches={"Config": ["Geometry"]}, debug=False)
    run = compute_reco_workflow(
        run, configs,
        params=({"DEFAULT_SIGNAL_WEIGHT": ["truth", "osc"], "DEFAULT_SIGNAL_NADIR": ["mean", "day", "night"],
                 "PARTICLE_TYPE": "signal", "PARTICLE_WEIGHTING": "volume", "OSCILLATION_BACKEND": "nufast"}
                if name == "marley" else {"PARTICLE_TYPE": "background", "PARTICLE_WEIGHTING": "histogram"}),
        rm_branches=False, workflow="SIGNIFICANCE", debug=False,
    )
    R = run["Reco"]
    n = len(R["RecoX"])
    if n_ref is not None and n != n_ref:
        sys.exit(f"[error] {config}/{name}: {n} events, cache has {n_ref}; the aux arrays would not align")
    d = {}
    for prefix in ("Main", "MainParent", "End", "SignalParticle"):
        for a in "XYZ":
            if f"{prefix}{a}" in R:
                d[f"{prefix.lower()}_{a.lower()}"] = np.asarray(R[f"{prefix}{a}"], float)
    for extra in ("TruthX", "MatchedOpFlashPur", "MatchedOpFlashTime", "MatchedOpFlashPE", "MainPDG", "MainK", "SignalParticleSurface"):
        if extra in R:
            d[extra] = np.asarray(R[extra], float)
    d["n"] = np.array(n)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **d)
    print(f"[ok] {out}  ({n} events, keys {sorted(d)})")


if args.stage in ("extract", "extract_aux"):
    if len(args.config) != 1:
        parser.error("--stage extract takes exactly one --config (loop in src/tools/run_truth_position.sh)")
    for sample in args.signals:
        (extract_one if args.stage == "extract" else extract_aux_one)(args.config[0], sample)
    sys.exit(0)


# ==========================================================================================
# plot: helpers
# ==========================================================================================

import matplotlib                                                # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.colors import to_rgba                            # noqa: E402
from matplotlib.patches import Patch                             # noqa: E402
from matplotlib.lines import Line2D                              # noqa: E402

from lib.fiducial import (                                       # noqa: E402
    truth_match_purity_config, accepted_flash_planes, build_fiducial_spatial_mask, get_best_fiducial, truth_match_purity_mask,
)
from lib.background import is_surface_background                 # noqa: E402

# Reference palette (dataviz skill): first three categorical slots validate on the all-pairs list.
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8f8e89", "#e3e2dc"
COLORS = {"marley": "#2a78d6", "gamma": "#eb6834", "neutron": "#1baf7a", "radiological": "#4a3aa7",
          "neutron_agree": "#1baf7a", "neutron_disagree": "#1baf7a"}
LABELS = {"marley": "Signal (marley)", "gamma": "Gamma", "neutron": "Neutron", "radiological": "Radiological"}
GROUPS = ("marley", "gamma", "neutron", "neutron_agree", "neutron_disagree", "radiological")
GROUP_LABEL = {"marley": "Signal", "gamma": "Gamma", "neutron": "Neutron", "neutron_agree": "Neutron\nreco≈truth",
               "neutron_disagree": "Neutron\nreco≠truth", "radiological": "Radiol."}
DEFAULT_C, TRUTH_C = MUTED, "#4a3aa7"
NICE = {
    "hd_1x2x6_centralAPA": "HD central APA", "hd_1x2x6_lateralAPA": "HD lateral APA",
    "vd_1x8x14_3view_30deg_nominal": "VD nominal", "vd_1x8x14_3view_30deg_shielded": "VD shielded",
}

plt.rcdefaults()
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8, "legend.frameon": False,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 1.6, "figure.dpi": 100, "pdf.fonttype": 42,
})


def save(fig, name):
    IMAGES.mkdir(parents=True, exist_ok=True)
    for fmt in args.formats:
        fig.savefig(IMAGES / f"{name}.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {IMAGES / name}.{{{','.join(args.formats)}}}")


def neff(w):
    return float(w.sum() ** 2 / (w ** 2).sum()) if len(w) and (w ** 2).sum() > 0 else 0.0


def wmedian(x, w):
    if len(x) == 0:
        return np.nan
    order = np.argsort(x)
    cum = np.cumsum(w[order])
    return float(x[order][np.searchsorted(cum, 0.5 * cum[-1])])


def wsum(w, m):
    return float(w[m].sum())


LATERAL = "hd_1x2x6_lateralAPA"


def x_faces(config):
    """(X low active, X high active). The faces are the box walls: HD central x = -360 and +360 (the plane x = 0 is interior); HD lateral only x = 0
    (the far x = 360 is not counted as a wall); VD both. Y and Z faces are always active."""
    return (True, False) if config == LATERAL else (True, True)


def wall_faces_distance(info, config, p):
    """Signed distance to the nearest active face per event (negative beyond it) and the index (0 X low, 1 X high, 2 Y, 3 Z) of that face."""
    lo = np.array([info[f"DETECTOR_MIN_{a}"] for a in "XYZ"], float)
    hi = np.array([info[f"DETECTOR_MAX_{a}"] for a in "XYZ"], float)
    dlo, dhi = p - lo, hi - p
    xl, xh = x_faces(config)
    d = np.stack([dlo[:, 0] if xl else np.full(len(p), np.inf), dhi[:, 0] if xh else np.full(len(p), np.inf),
                  np.minimum(dlo[:, 1], dhi[:, 1]), np.minimum(dlo[:, 2], dhi[:, 2])], axis=1)
    return d.argmin(axis=1), d.min(axis=1)


def max_wall_distance(info, config):
    """Largest possible distance from any point of the box to its nearest active face."""
    ext = [info[f"DETECTOR_MAX_{a}"] - info[f"DETECTOR_MIN_{a}"] for a in "XYZ"]
    xl, xh = x_faces(config)
    return float(min(ext[0] / (2.0 if (xl and xh) else 1.0), ext[1] / 2.0, ext[2] / 2.0))


class Sample:
    """Cached arrays of one (config, sample) plus the selections built on them."""

    def __init__(self, config, name, info, path):
        self.config, self.name, self.info = config, name, info
        self.d = dict(np.load(path, allow_pickle=True))
        d = self.d
        self.w = d["w"]
        aux_path = Path(str(path)[:-4] + "_aux.npz")
        self.aux = None
        if aux_path.exists():
            aux = dict(np.load(aux_path, allow_pickle=True))
            if int(aux["n"]) == len(self.w):
                self.aux = aux
            else:
                print(f"[warn] {aux_path.name}: {int(aux['n'])} events, cache has {len(self.w)}; aux ignored")
        quality = accepted_flash_planes(d["plane"], ROOT, True) & (d["pe"] > 0)
        if is_surface_background(ROOT, name):
            quality &= d["surface"] >= 0
            if args.folder in ("Reduced", "Truncated"):
                quality &= d["surface"] < 3
        self.quality = quality
        emin = args.emin
        self.window = quality & (d["energy"] >= emin) & (d["energy"] <= args.emax)
        self.emin = emin
        tk = [str(k) for k in d["truth_keys"]]
        fake = {"Reco": {"NHits": d["NHits"], "RecoX": d["reco_x"], "RecoY": d["reco_y"], "RecoZ": d["reco_z"],
                         tk[0]: d["truth_x"], tk[1]: d["truth_y"], tk[2]: d["truth_z"]}}
        self.truth_ok = d["contained"] & truth_match_purity_mask(fake, ROOT, name)   # what --truth_fiducial adds
        self.truth_keys = tk

    def cut(self, c):
        d = self.d
        return (d["NHits"] >= c["NHits"]) & (d["AdjClNum"] < c["AdjCl"]) & (d["OpHits"] >= c["OpHits"])

    def pos(self, kind):
        return np.stack([self.d[f"{kind}_{a}"] for a in "xyz"], 1)

    def fiducial(self, fid, kind, pipeline=True):
        """pipeline=False evaluates the truth position without the containment/consistency cuts (position knowledge only)."""
        info = self.info
        dx = info["DETECTOR_SIZE_X"] + 2 * info["DETECTOR_GAP_X"]
        dy = info["DETECTOR_SIZE_Y"] + 2 * info["DETECTOR_GAP_Y"]
        run = {"Reco": {"RecoX": self.d[f"{kind}_x"], "RecoY": self.d[f"{kind}_y"], "RecoZ": self.d[f"{kind}_z"]}}
        m = build_fiducial_spatial_mask(run, self.config, dx, dy, info, args.folder, fid)
        return m & self.truth_ok if (kind == "truth" and pipeline) else m

    def wall_distance(self, kind):
        """Distance to the nearest active wall (see x_faces), clipped at 0 for positions beyond it."""
        return np.clip(wall_faces_distance(self.info, self.config, self.pos(kind))[1], 0.0, None)

    def outside_box(self, kind):
        """True where the position lies outside the whole active box (all six faces)."""
        lo = np.array([self.info[f"DETECTOR_MIN_{a}"] for a in "XYZ"], float)
        hi = np.array([self.info[f"DETECTOR_MAX_{a}"] for a in "XYZ"], float)
        p = self.pos(kind)
        return ((p < lo) | (p > hi)).any(axis=1)


def load_config(config):
    info = config_info(config)
    samples = {}
    for name in args.signals:
        path = CACHE / config / f"{config}_{name}.npz"
        if path.exists():
            samples[name] = Sample(config, name, info, path)
    if "marley" not in samples:
        return None
    return {"info": info, "s": samples}


def best_cut(config):
    """Default best cut of --analysis (the Sensitivity map carries NHits/OpHits/AdjCl under the same keys)."""
    sub, tag = {"DAYNIGHT": ("daynight-json", "highest_DayNight"), "HEP": ("hep-json", "highest_HEP"),
                "SENSITIVITY": ("sensitivity-json", "highest_Sensitivity")}[args.analysis]
    p = f"{ROOT}/config/{config}/{sub}/{args.folder.lower()}/{config}_{tag}.json"
    e = json.load(open(p))[config][args.energy]
    return {"NHits": int(e["NHits"]), "OpHits": int(e["OpHits"]), "AdjCl": int(e["AdjCl"])}


def fiducials(label=""):
    p = f"{ROOT}/config/analysis/fiducial/{args.folder.lower()}/BestFiducials{label}.json"
    return json.load(open(p)) if os.path.exists(p) else {}


def volume(fids, config, analysis=None):
    try:
        e = get_best_fiducial(fids, config, args.energy, analysis or args.analysis)
    except KeyError:
        return None
    return {k: e[k] for k in ("FiducialX", "FiducialY", "FiducialZ")}


def vol_txt(v):
    return "–" if v is None else f"{v['FiducialX']:g}/{v['FiducialY']:g}/{v['FiducialZ']:g}"


def legend_handles(names, extra=()):
    h = [Line2D([0], [0], color=COLORS[n], lw=2, label=LABELS[n]) for n in names]
    return h + list(extra)


# ==========================================================================================
# 1. wall proximity
# ==========================================================================================

def fig_wall_proximity(data):
    configs = list(data)
    widths = [1.9] + [1.0] * (len(configs) - 1)
    fig, axes = plt.subplots(1, len(configs), figsize=(3.4 * (sum(widths)), 3.6), gridspec_kw={"width_ratios": widths}, squeeze=False)
    grid = np.arange(0, 301, 1.0)
    for ax, config in zip(axes[0], configs):
        S = data[config]["s"]
        cut = best_cut(config) if args.apply_cuts_walls else None
        for name in ("marley", "neutron", "gamma"):
            if name not in S:
                continue
            s = S[name]
            m = s.window & (s.cut(cut) if cut else True)
            if not m.any():
                continue
            for kind, ls, lw in (("truth", "-", 1.8), ("reco", (0, (3, 2)), 1.1)):
                dist, w = s.wall_distance(kind)[m], s.w[m]
                cdf = np.array([w[dist <= g].sum() for g in grid]) / w.sum()
                ax.plot(grid, 100 * cdf, color=COLORS[name], ls=ls, lw=lw)
        for x in (20, 100):
            ax.axvline(x, color=INK2, lw=0.8, ls=":")
            ax.text(x + 3, 3, f"{x} cm", color=INK2, fontsize=7.5, rotation=90, va="bottom")
        dmax = max_wall_distance(data[config]["info"], config)     # largest possible distance to the nearest wall
        if dmax < 300:
            ax.axvline(dmax, color=MUTED, lw=0.9, ls="-.")
            ax.text(dmax - 4, 97, f"box centre {dmax:g} cm", color=INK2, fontsize=7, rotation=90, va="top", ha="right")
        ax.set_xlim(0, 300)
        ax.set_ylim(0, 100)
        ax.set_xlabel("distance to nearest active-volume face [cm]")
        ax.set_title(NICE.get(config, config), loc="left")
        if ax is axes[0][0]:
            ax.set_ylabel("cumulative weighted fraction of events [%]")
            ax.legend(handles=legend_handles(("marley", "neutron", "gamma"), [
                Line2D([0], [0], color=INK2, lw=1.8, label="truth position"),
                Line2D([0], [0], color=INK2, lw=1.1, ls=(0, (3, 2)), label="reco position")]), loc="lower right")
    cutnote = "with analysis cut" if args.apply_cuts_walls else "no topological cut"
    fig.suptitle(f"Gamma piles up at the walls; neutrons and signal do not  ({cutnote}, {S['marley'].emin:g}–{args.emax:g} MeV)",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    save(fig, "wall_proximity_cdf")


# ==========================================================================================
# 2. background-to-signal ratio per shell
# ==========================================================================================

SHELLS = [(0, 20), (20, 100), (100, 200)]


def shell_ratio(data, config, bkg):
    S = data[config]["s"]
    cut = best_cut(config) if args.apply_cuts_walls else None

    def per_shell(s):
        m = s.window & (s.cut(cut) if cut else True)
        dist, w = s.wall_distance("truth")[m], s.w[m]
        return [(w[(dist >= lo) & (dist < hi)].sum(), neff(w[(dist >= lo) & (dist < hi)])) for lo, hi in SHELLS]

    if bkg not in S:
        return None
    sig, b = per_shell(S["marley"]), per_shell(S[bkg])
    ref_sig, ref_b = sig[-1][0], b[-1][0]
    if ref_sig <= 0 or ref_b <= 0:
        return None
    out = []
    for (bs, bn), (ss, sn) in zip(b, sig):
        if ss <= 0 or bs <= 0:
            out.append((np.nan, np.nan))
            continue
        ratio = (bs / ss) / (ref_b / ref_sig)
        rel = np.sqrt(1 / max(bn, 1) + 1 / max(sn, 1) + 1 / max(b[-1][1], 1) + 1 / max(sig[-1][1], 1))
        out.append((ratio, ratio * rel))
    return out


def fig_shell_ratio(data):
    configs = list(data)
    fig, ax = plt.subplots(figsize=(1.9 * len(configs) + 1.6, 3.8))
    width, gap = 0.34, 0.06
    rows = []
    for ci, config in enumerate(configs):
        for bi, bkg in enumerate(("gamma", "neutron")):
            res = shell_ratio(data, config, bkg)
            if res is None:
                continue
            xs = ci * 4 + np.arange(len(SHELLS)) + (bi - 0.5) * (width + gap)
            vals = np.array([r[0] for r in res])
            errs = np.array([r[1] for r in res])
            ax.bar(xs, np.nan_to_num(vals), width, color=COLORS[bkg], alpha=1.0 if bi == 0 else 0.85,
                   yerr=np.where(np.isfinite(errs), errs, 0), error_kw={"lw": 0.8, "ecolor": INK2, "capsize": 1.5})
            for k, (x, v) in enumerate(zip(xs, vals)):
                if np.isfinite(v) and bkg == "gamma" and k < len(SHELLS) - 1:
                    ax.text(x, v * 1.25, f"×{v:.0f}" if v >= 10 else f"×{v:.1f}", ha="center", va="bottom", fontsize=7.5, color=INK)
            rows.append((config, bkg, vals))
        ax.text(ci * 4 + 1, -0.17, NICE.get(config, config), ha="center", va="top", fontsize=9, color=INK,
                transform=ax.get_xaxis_transform(), clip_on=False)
    ax.set_yscale("log")
    ax.set_ylim(0.5, 3000)
    ax.axhline(1, color=INK2, lw=0.8, ls=":")
    ax.set_xticks([ci * 4 + k for ci in range(len(configs)) for k in range(len(SHELLS))])
    ax.set_xticklabels([f"{lo}–{hi}" for _ in configs for lo, hi in SHELLS], fontsize=7.5)
    ax.set_xlabel("shell: distance to nearest face [cm]", labelpad=26)
    ax.set_ylabel("(B/S in shell) / (B/S in 100–200 cm)")
    ax.grid(axis="x", visible=False)
    ax.legend(handles=legend_handles(("gamma", "neutron")), loc="upper right")
    ax.set_title("Wall pile-up as one number per detector: outer-shell background-to-signal enhancement", loc="left", fontsize=10)
    fig.tight_layout()
    save(fig, "shell_ratio")
    return rows


# ==========================================================================================
# 3. reco − truth residuals
# ==========================================================================================

def fig_residuals(data, config):
    """Row 1: reco - truth per axis for signal and gamma (window sample, no topological cut, for statistics).
    Row 2: neutron survivors at the analysis cut, reco against truth."""
    S = data[config]["s"]
    cut = best_cut(config)
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.4))
    lim, step = 150.0, 5.0
    edges = np.arange(-lim, lim + step, step)
    for j, axis in enumerate("xyz"):
        ax = axes[0][j]
        for k, name in enumerate(("marley", "gamma")):
            if name not in S:
                continue
            s = S[name]
            m = s.window
            if not m.any():
                continue
            delta = (s.d[f"reco_{axis}"] - s.d[f"truth_{axis}"])[m]
            w = s.w[m]
            h, _ = np.histogram(np.clip(delta, -lim + 1e-6, lim - 1e-6), bins=edges, weights=w)
            h = h / w.sum()
            ax.stairs(np.where(h > 0, h, np.nan), edges, color=COLORS[name], lw=1.3)
            med = wmedian(delta, w)
            ax.axvline(med, color=COLORS[name], lw=1.0, ls="--")
            out = 100 * w[np.abs(delta) >= lim].sum() / w.sum()
            within = 100 * w[np.abs(delta) < args.agree_cm].sum() / w.sum()
            ax.text(0.98, 0.97 - 0.085 * k, f"{LABELS[name].split(' ')[0]}: median {med:+.1f} cm, within {args.agree_cm:g} cm {within:.0f}%, beyond {lim:g}: {out:.1f}%",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7, color=INK)
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 1e3)
        ax.set_xlim(-lim, lim)
        ax.set_xlabel(f"reco − truth  {axis.upper()} [cm]  (outer bins include overflow)")
        if j == 0:
            ax.set_ylabel("weighted fraction / 5 cm")
        ax.set_title(axis.upper(), loc="left", fontsize=9)

    s = S.get("neutron")
    if s is None:
        for ax in axes[1]:
            ax.axis("off")
    else:
        m = s.window & s.cut(cut)
        w = s.w[m]
        delta = s.pos("reco") - s.pos("truth")
        dist = np.sqrt((delta ** 2).sum(axis=1))[m]
        agree = (np.abs(delta) < args.agree_cm).all(axis=1)[m]
        cagree, cdis = COLORS["neutron"], INK
        # (a) 3D |reco - truth| split at agree_cm
        ax = axes[1][0]
        bins = np.logspace(-1, 3.2, 40)
        for sel_, style, lab in ((agree, dict(color=to_rgba(cagree, 0.9)), "agree"), (~agree, dict(color=to_rgba(cdis, 0.25), edgecolor=cdis, lw=1.0), "disagree")):
            if sel_.any():
                ax.hist(np.clip(dist[sel_], bins[0], bins[-1] - 1e-6), bins=bins, histtype="stepfilled", **style,
                        label=f"{lab}: N_MC = {int(sel_.sum())} ({100 * w[sel_].sum() / w.sum():.0f}% of weight)")
        ax.axvline(args.agree_cm, color=INK2, lw=0.9, ls=":")
        ax.text(args.agree_cm * 1.08, 0.02, f"{args.agree_cm:g} cm", fontsize=7.5, color=INK2, va="bottom", transform=ax.get_xaxis_transform())
        ax.set_xscale("log")
        ax.set_xlabel("3D |reco − truth| [cm]")
        ax.set_ylabel("MC events")
        ax.set_title(f"Neutron survivors at the cut, N_MC = {int(m.sum())}", loc="left", fontsize=9)
        ax.legend(loc="upper left", fontsize=7)
        # (b) RecoX against truth X with the ±edge clipping marked
        xlim = float(s.info["DETECTOR_MAX_X"])
        ax = axes[1][1]
        tx, rx = s.d["truth_x"][m], s.d["reco_x"][m]
        ax.axhspan(xlim, xlim + 200, color=to_rgba(cdis, 0.08))
        ax.axhspan(-xlim - 200, -xlim, color=to_rgba(cdis, 0.08))
        ax.axhline(xlim, color=INK2, lw=0.9, ls="--")
        ax.axhline(-xlim, color=INK2, lw=0.9, ls="--")
        ax.scatter(tx[agree], rx[agree], s=16, color=cagree, zorder=3)
        ax.scatter(tx[~agree], rx[~agree], s=18, facecolor="none", edgecolor=cdis, lw=1.0, zorder=3)
        lo_, hi_ = min(tx.min(), rx.min()), max(tx.max(), rx.max())
        ax.plot([lo_, hi_], [lo_, hi_], color=INK2, lw=0.8, ls=":")
        ax.text(0.02, 0.97, f"RecoX = ±{xlim:g} cm (clipped)", transform=ax.transAxes, fontsize=7.5, color=INK2, va="top")
        ax.set_xlabel(f"truth {s.truth_keys[0]} [cm]")
        ax.set_ylabel("RecoX [cm]")
        # (c) what the disagreeing events look like in RecoX
        ax = axes[1][2]
        clipped = np.abs(rx) >= xlim - 0.1
        ebins = np.linspace(-xlim - 10, xlim + 10, 37)
        for sel_, style, lab in ((agree, dict(color=to_rgba(cagree, 0.9)), "agree"), (~agree, dict(color=to_rgba(cdis, 0.25), edgecolor=cdis, lw=1.0), "disagree")):
            if sel_.any():
                ax.hist(rx[sel_], bins=ebins, histtype="stepfilled", **style, label=lab)
        if (~agree).any():
            dis = ~agree
            yz = (np.abs(delta[m][:, 1]) >= args.agree_cm) | (np.abs(delta[m][:, 2]) >= args.agree_cm)
            ax.text(0.98, 0.72, f"disagreeing MC events ({int(dis.sum())}):\n|RecoX| at the edge: {int((dis & clipped).sum())}\n"
                                f"|ΔY| or |ΔZ| > {args.agree_cm:g} cm: {int((dis & yz).sum())}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=INK)
        ax.axvline(xlim, color=INK2, lw=0.9, ls="--")
        ax.axvline(-xlim, color=INK2, lw=0.9, ls="--")
        ax.set_xlabel("RecoX [cm]")
        ax.set_ylabel("MC events")
        ax.legend(loc="upper left", fontsize=7)
    fig.legend(handles=legend_handles(("marley", "gamma")), loc="upper right", ncol=2, bbox_to_anchor=(0.99, 0.94))
    fig.suptitle(f"{NICE.get(config, config)}: reco − truth position (top, {args.emin:g}–{args.emax:g} MeV window: N_MC signal "
                 f"{int(S['marley'].window.sum())}, gamma {int(S['gamma'].window.sum()) if 'gamma' in S else 0}) and neutron survivors "
                 f"(bottom, {args.analysis} cut N{cut['NHits']}/O{cut['OpHits']}/A{cut['AdjCl']})",
                 x=0.01, ha="left", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, f"residuals_{config}_{args.analysis.lower()}")


# ==========================================================================================
# 4. pass fractions
# ==========================================================================================

PF_LABELS = ["reco position, reco volume",
             "truth position, reco volume (fiducial only)",
             "truth position, reco volume (+ containment, consistency)",
             "truth position, truth volume (+ containment, consistency)"]


def pass_fractions(data, config):
    """Weighted pass fraction of the window+cut sample for four position treatments (see PF_LABELS)."""
    S = data[config]["s"]
    cut = best_cut(config)
    fr = volume(fiducials(), config)
    ft = volume(fiducials("_fiduc_truth"), config)
    out = {}
    for name in ("marley",) + tuple(BKG):
        if name not in S:
            continue
        s = S[name]
        base = s.window & s.cut(cut)
        groups = {name: base}
        if name == "neutron":
            agree = (np.abs(s.pos("reco") - s.pos("truth")) < args.agree_cm).all(axis=1)
            groups["neutron_agree"], groups["neutron_disagree"] = base & agree, base & ~agree
        for gname, denom in groups.items():
            tot = wsum(s.w, denom)
            if tot <= 0 or fr is None:
                continue
            out[gname] = ([wsum(s.w, denom & s.fiducial(fr, "reco")) / tot,
                           wsum(s.w, denom & s.fiducial(fr, "truth", pipeline=False)) / tot,
                           wsum(s.w, denom & s.fiducial(fr, "truth")) / tot,
                           (wsum(s.w, denom & s.fiducial(ft, "truth")) / tot) if ft else np.nan], int(denom.sum()))
    return out, fr, ft


def fig_pass_fractions(data):
    configs = list(data)
    fig, axes = plt.subplots(1, len(configs), figsize=(6.4 * len(configs), 4.4), squeeze=False, sharey=True)
    names = GROUPS
    nb = len(PF_LABELS)
    bw = 0.2
    for ax, config in zip(axes[0], configs):
        res, fr, ft = pass_fractions(data, config)
        for gi, name in enumerate(names):
            if name not in res:
                continue
            vals, n = res[name]
            c = COLORS[name]
            styles = [dict(color=c), dict(color=to_rgba(c, 0.35), edgecolor=c, hatch="////", lw=0.8),
                      dict(color=to_rgba(c, 0.35), edgecolor=c, hatch="\\\\", lw=0.8), dict(color="white", edgecolor=c, lw=1.4)]
            for k, v in enumerate(vals):
                if not np.isfinite(v):
                    continue
                x = gi + (k - (nb - 1) / 2) * (bw + 0.02)
                ax.bar(x, 100 * v, bw, **styles[k])
                ax.text(x, 100 * v + 1.5, f"{100 * v:.0f}", ha="center", va="bottom", fontsize=6, color=INK)
            ax.text(gi, -17, f"N_MC={n}", ha="center", fontsize=7, color=INK2)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([GROUP_LABEL[n] for n in names], fontsize=7.5)
        ax.set_ylim(0, 115)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{NICE.get(config, config)}\nreco vol {vol_txt(fr)}, truth vol {vol_txt(ft)}", loc="left", fontsize=8.5)
        if ax is axes[0][0]:
            ax.set_ylabel("pass fraction after fiducial [%]")
    fig.legend(handles=[Patch(facecolor=to_rgba(INK2, a), edgecolor=INK2, hatch=h, lw=l, label=t)
                        for (a, h, l), t in zip([(1, None, 0), (0.35, "////", 0.8), (0.35, "\\\\", 0.8), (0, None, 1.4)], PF_LABELS)],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), fontsize=7.5)
    fig.suptitle(f"Fiducial pass fractions with reco and truth positions ({args.analysis} cut, {args.emin:g}–{args.emax:g} MeV)",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    save(fig, f"pass_fractions_{args.analysis.lower()}")
    return {c: pass_fractions(data, c) for c in configs}


# ==========================================================================================
# 5/6. significance summary and volume table
# ==========================================================================================

ANALYSES = [("DayNight", "daynight-json", "DayNight", "Values"), ("HEP", "hep-json", "HEP", "Values"),
            ("Sensitivity", "sensitivity-json", "Sensitivity", "Score")]
VARIANTS = [("", "default"), ("_fiduc_truth", "fiduc_truth"), ("_fiduc_truth_refvol", "fiduc_truth_refvol")]


def significance(config, analysis, variant):
    _, sub, tag, key = next(a for a in ANALYSES if a[0] == analysis)
    p = f"{ROOT}/config/{config}/{sub}/{args.folder.lower()}/{config}_highest{variant}_{tag}.json" if analysis != "Sensitivity" \
        else f"{ROOT}/config/{config}/{sub}/{args.folder.lower()}/{config}_highest_Sensitivity{variant}.json"
    try:
        return float(json.load(open(p))[config][args.energy][key])
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return np.nan


def sig_table(configs):
    return {(c, a[0], v[1]): significance(c, a[0], v[0]) for c in configs for a in ANALYSES for v in VARIANTS}


def fig_summary(configs, sig):
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.4), gridspec_kw={"height_ratios": [3, 1.15]}, sharex="col")
    x = np.arange(len(configs))
    for j, (analysis, *_rest) in enumerate(ANALYSES):
        top, bot = axes[0][j], axes[1][j]
        d = np.array([sig[(c, analysis, "default")] for c in configs])
        t = np.array([sig[(c, analysis, "fiduc_truth")] for c in configs])
        for xs, vals, color, lab in ((x - 0.19, d, DEFAULT_C, "default (reco)"), (x + 0.19, t, TRUTH_C, "fiduc_truth")):
            top.bar(xs, np.nan_to_num(vals), 0.34, color=color, label=lab)
            for xi, v in zip(xs, vals):
                if np.isfinite(v):
                    top.text(xi, v * 1.02, f"{v:.2f}", ha="center", va="bottom", fontsize=6.8, color=INK)
        top.set_title(f"{analysis}", loc="left")
        top.grid(axis="x", visible=False)
        ratio = t / d
        bot.axhline(1, color=INK2, lw=0.8, ls=":")
        bot.plot(x, ratio, "o", color=TRUTH_C, ms=6, mec="white", mew=1.2)
        for xi, r in zip(x, ratio):
            if np.isfinite(r):
                bot.text(xi, r + 0.08 * (1 if r >= 1 else -1) + (0.02 if r >= 1 else -0.02), f"{r:.2f}", ha="center", va="bottom" if r >= 1 else "top", fontsize=7, color=INK)
        bot.set_ylim(0.8, max(1.45, np.nanmax(ratio) * 1.12 if np.isfinite(ratio).any() else 1.45))
        bot.set_xticks(x)
        bot.set_xticklabels([NICE.get(c, c).replace(" ", "\n", 1) for c in configs], fontsize=7.5)
        bot.grid(axis="x", visible=False)
        top.set_ylabel("significance [σ]" if analysis != "Sensitivity" else "Score = ½[χ²☉(θ_react) + χ²_react(θ☉)]  [Δχ²]", fontsize=8)
        if j == 0:
            bot.set_ylabel("truth / default")
            top.legend(loc="upper right")
    fig.suptitle("Default vs fiduc_truth significance per analysis (Truncated, SolarEnergy)", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    save(fig, "default_vs_fiduc_truth")


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def write_csv(name, headers, rows):
    TABLES.mkdir(parents=True, exist_ok=True)
    with open(TABLES / f"{name}.csv", "w") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join(str(c).replace(",", ";") for c in r) + "\n")


def fmt(v, n=3):
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.{n}f}"


def table_volumes(configs, sig):
    headers = ["Config", "Analysis", "Default volume X/Y/Z [cm]", "significance default (σ; Sensitivity: Δχ²)", "Truth volume X/Y/Z [cm]", "significance fiduc_truth",
               "significance fiduc_truth_refvol (truth pos., default volume)", "Δ fiduc_truth", "Δ refvol"]
    rows = []
    for c in configs:
        for a, *_ in ANALYSES:
            fa = a.upper()
            d, t, r = (sig[(c, a, v)] for v in ("default", "fiduc_truth", "fiduc_truth_refvol"))
            rows.append([NICE.get(c, c), a, vol_txt(volume(fiducials(), c, fa)), fmt(d), vol_txt(volume(fiducials("_fiduc_truth"), c, fa)),
                         fmt(t), fmt(r), f"{t - d:+.3f}" if np.isfinite(t - d) else "n/a", f"{r - d:+.3f}" if np.isfinite(r - d) else "n/a"])
    write_csv("volumes_significance", headers, rows)
    return headers, rows


def table_survivors(data):
    headers = ["Config", "Component", "N_MC (window+cut)", "N_MC + reco fiducial", "N_MC + truth pipeline",
               "Σw (reco fiducial)", "mean w / event", "N_eff", "largest single-event share of Σw", "rel. stat. unc. 1/√N_eff",
               "share of background Σw, reco", "share of background Σw, truth pipeline"]
    rows = []
    for c, D in data.items():
        cut = best_cut(c)
        fr = volume(fiducials(), c)
        ft = volume(fiducials("_fiduc_truth"), c)
        sel = {}
        for name in ("marley",) + tuple(BKG):
            s = D["s"].get(name)
            if s is None:
                continue
            base = s.window & s.cut(cut)
            sel[name] = (s, base, base & s.fiducial(fr, "reco") if fr else base, base & s.fiducial(ft, "truth") if ft else base)
        tot_reco = sum(wsum(v[0].w, v[2]) for n, v in sel.items() if n in BKG)
        tot_truth = sum(wsum(v[0].w, v[3]) for n, v in sel.items() if n in BKG)
        for name, (s, base, reco, truth) in sel.items():
            w = s.w[reco]
            share = ("–", "–") if name == "marley" else (
                f"{100 * wsum(s.w, reco) / tot_reco:.0f}%" if tot_reco > 0 else "n/a",
                f"{100 * wsum(s.w, truth) / tot_truth:.0f}%" if tot_truth > 0 else "n/a")
            if len(w) == 0:
                rows.append([NICE.get(c, c), name, int(base.sum()), 0, int(truth.sum()), "0", "–", "0", "–", "–", *share])
                continue
            ne = neff(w)
            rows.append([NICE.get(c, c), name, int(base.sum()), int(reco.sum()), int(truth.sum()), f"{w.sum():.4g}", f"{w.mean():.4g}",
                         f"{ne:.1f}", f"{100 * w.max() / w.sum():.0f}%", f"{100 / np.sqrt(ne):.0f}%", *share])
    write_csv(f"survivor_statistics_{args.analysis.lower()}", headers, rows)
    return headers, rows


def table_shells(shell_rows):
    headers = ["Config", "Background"] + [f"{lo}–{hi} cm" for lo, hi in SHELLS]
    rows = [[NICE.get(c, c), b] + [("n/a" if not np.isfinite(v) else f"×{v:.1f}") for v in vals] for c, b, vals in shell_rows]
    write_csv("shell_ratio", headers, rows)
    return headers, rows


def table_passfrac(pf):
    headers = ["Config", "Component", "N_MC", "reco pos, reco vol [%]", "truth pos, reco vol, fiducial only [%]",
               "truth pos, reco vol, + containment/consistency [%]", "truth pos, truth vol, + containment/consistency [%]"]
    rows = []
    for c, (res, fr, ft) in pf.items():
        for name, (vals, n) in res.items():
            rows.append([NICE.get(c, c), name, n] + [("n/a" if not np.isfinite(v) else f"{100 * v:.1f}") for v in vals])
    write_csv(f"pass_fractions_{args.analysis.lower()}", headers, rows)
    return headers, rows


# ==========================================================================================
# faces stage: entry-face composition, X residuals near the entry faces, mixed X/YZ scans,
#              signal loss vs background rejection, flash-mismatch rates
# ==========================================================================================

FACE_NAMES = ["X low", "X high", "Y", "Z"]
FACE_COLORS = ["#e34948", "#eda100", "#e87ba4", "#008300"]
FACE_CONFIGS = ["hd_1x2x6_lateralAPA", "vd_1x8x14_3view_30deg_nominal", "vd_1x8x14_3view_30deg_shielded", "hd_1x2x6_centralAPA"]
MODES = [("reco", "reco X, reco Y/Z"), ("truth", "truth X, truth Y/Z"), ("xT", "truth X, reco Y/Z"),
         ("xTnc", "truth X (no containment), reco Y/Z"), ("xR", "reco X, truth Y/Z")]


def bounds(info):
    lo = np.array([info[f"DETECTOR_MIN_{a}"] for a in "XYZ"], float)
    hi = np.array([info[f"DETECTOR_MAX_{a}"] for a in "XYZ"], float)
    return lo, hi


def face_split(s, kind="truth"):
    """(face index 0..3, signed distance to that face): X low, X high, Y (either), Z (either), among the active walls (x_faces);
    negative = beyond it. Use Sample.outside_box for the 'outside the box' flag."""
    return wall_faces_distance(s.info, s.config, s.pos(kind))


def entry_x_faces(config):
    """Faces counted as 'entry through X': both walls for HD central, only x = 0 (X low) for HD lateral, only the top (X high) for VD."""
    return [1] if config.startswith("vd") else ([0] if config == LATERAL else [0, 1])


def stage_masks(s, cut, fr):
    c = s.window & s.cut(cut)
    return [("window", s.window), ("+ analysis cut", c), ("+ cut + reco fiducial", c & s.fiducial(fr, "reco"))]


def fig_face_composition(data):
    configs = [c for c in FACE_CONFIGS if c in data]
    rows, table = ("gamma", "neutron", "radiological"), []
    fig, axes = plt.subplots(len(rows), len(configs), figsize=(3.9 * len(configs), 2.4 * len(rows)), squeeze=False, sharex=True)
    for ci, config in enumerate(configs):
        S, cut, fr = data[config]["s"], best_cut(config), volume(fiducials(), config)
        for ri, name in enumerate(rows):
            ax = axes[ri][ci]
            if name not in S:
                ax.axis("off")
                continue
            s = S[name]
            idx, dist = face_split(s)
            for yi, (label, m) in enumerate(stage_masks(s, cut, fr)):
                w = s.w[m]
                tot = w.sum()
                fr_face = [w[idx[m] == k].sum() / tot if tot > 0 else 0.0 for k in range(4)]
                outside = w[s.outside_box('truth')[m]].sum() / tot if tot > 0 else np.nan
                left = 0.0
                for k, f in enumerate(fr_face):
                    ax.barh(yi, 100 * f, left=left, height=0.62, color=FACE_COLORS[k], edgecolor="white", linewidth=1.5)
                    if f > 0.09:
                        ax.text(left + 50 * f, yi, f"{100 * f:.0f}", ha="center", va="center", fontsize=7, color="white" if k in (0, 3) else INK)
                    left += 100 * f
                ax.text(101, yi, f"N={int(m.sum())}", va="center", fontsize=6.8, color=INK2)
                table.append([NICE.get(config, config), name, label, int(m.sum()), f"{tot:.4g}"] + [f"{100 * f:.1f}" for f in fr_face]
                             + [("n/a" if not np.isfinite(outside) else f"{100 * outside:.0f}"), f"{w[np.isin(idx[m], entry_x_faces(config))].sum():.4g}"])
            ax.set_yticks(range(3))
            ax.set_yticklabels(["window", "+ cut", "+ cut + reco fid."] if ci == 0 else [], fontsize=7.5)
            ax.invert_yaxis()
            ax.set_xlim(0, 118)
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.grid(axis="y", visible=False)
            if ri == 0:
                ax.set_title(NICE.get(config, config), loc="left", fontsize=9)
            if ci == 0:
                ax.set_ylabel(name.capitalize(), fontsize=9, color=INK)
            if ri == len(rows) - 1:
                ax.set_xlabel("share of weight by nearest face [%] (truth position)")
    fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(FACE_COLORS, FACE_NAMES)], loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Where the background enters: nearest-face composition before and after the {args.analysis} cut", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.965))
    save(fig, f"faces_composition_{args.analysis.lower()}")
    headers = ["Config", "Component", "Stage", "N_MC", "Σw", "X low [%]", "X high [%]", "Y [%]", "Z [%]", "outside active box [%]", "Σw through entry X face(s)"]
    write_csv(f"faces_composition_{args.analysis.lower()}", headers, table)
    return headers, table


def fig_x_entry_residuals(data):
    configs = [c for c in FACE_CONFIGS if c in data]
    fig, axes = plt.subplots(2, len(configs), figsize=(3.9 * len(configs), 5.6), squeeze=False)
    table = []
    edges = np.arange(0, 710, 20.0)
    for ci, config in enumerate(configs):
        S, cut = data[config]["s"], best_cut(config)
        lo, hi = bounds(data[config]["info"])
        for ri, use_cut in enumerate((False, True)):
            ax = axes[ri][ci]
            for name in ("gamma", "neutron"):
                if name not in S:
                    continue
                s = S[name]
                idx, dist = face_split(s)
                m = s.window & (s.cut(cut) if use_cut else True) & np.isin(idx, entry_x_faces(config)) & (dist < 100)
                if not m.any():
                    table.append([NICE.get(config, config), name, "cut" if use_cut else "window", 0, "–", "–", "–", "–", "–"])
                    ax.text(0.5, 0.55 - 0.1 * (name == "neutron"), f"{name}: N_MC = 0", transform=ax.transAxes, ha="center", fontsize=8, color=COLORS[name])
                    continue
                if m.sum() < 5:      # a histogram of a handful of events is a spike, not a distribution
                    ax.text(0.5, 0.55 - 0.1 * (name == "neutron"), f"{name}: N_MC = {int(m.sum())}, not drawn", transform=ax.transAxes, ha="center", fontsize=8, color=COLORS[name])
                dx = np.abs(s.d["reco_x"] - s.d["truth_x"])[m]
                w = s.w[m]
                rx = s.d["reco_x"][m]
                clipped = (rx <= lo[0] + 0.5) | (rx >= hi[0] - 0.5)
                h, _ = np.histogram(np.clip(dx, 0, edges[-1] - 1e-6), bins=edges, weights=w)
                if m.sum() >= 5:
                    ax.stairs(np.where(h > 0, h / w.sum(), np.nan), edges, color=COLORS[name], lw=1.4)
                    ax.axvline(wmedian(dx, w), color=COLORS[name], lw=0.9, ls="--")
                table.append([NICE.get(config, config), name, "cut" if use_cut else "window", int(m.sum()), f"{w.sum():.4g}", f"{wmedian(dx, w):.1f}",
                              f"{100 * w[dx > 30].sum() / w.sum():.0f}", f"{100 * w[dx > 100].sum() / w.sum():.0f}", f"{100 * w[clipped].sum() / w.sum():.0f}"])
            ax.axvline(30, color=INK2, lw=0.8, ls=":")
            ax.set_yscale("log")
            ax.set_xlim(0, edges[-1])
            ax.set_ylim(1e-4, 2)
            ax.set_xlabel("|RecoX − truth X| [cm]  (last bin = overflow)")
            faces = "X high (top)" if config.startswith("vd") else ("X low (x = 0)" if config == LATERAL else "X low + X high")
            ax.set_title(f"{NICE.get(config, config)}, {faces}, < 100 cm from face\n{'after analysis cut' if use_cut else '10–20 MeV window'}", loc="left", fontsize=8)
            if ci == 0:
                ax.set_ylabel("weighted fraction / 20 cm")
    fig.legend(handles=legend_handles(("gamma", "neutron")), loc="upper right", ncol=2, bbox_to_anchor=(0.99, 0.985))
    fig.suptitle("Drift-coordinate residuals of the background that enters through the X face(s)", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, f"x_residuals_entry_{args.analysis.lower()}")
    headers = ["Config", "Component", "Sample", "N_MC", "Σw", "median |ΔX| [cm]", "|ΔX| > 30 cm [%]", "|ΔX| > 100 cm [%]", "RecoX at the volume edge [%]"]
    write_csv(f"x_residuals_entry_{args.analysis.lower()}", headers, table)
    return headers, table


def mode_positions(s, mode):
    """(x, y, z) arrays and the axes taken from truth for a scan mode."""
    src = {"reco": "rrr", "truth": "ttt", "xT": "trr", "xTnc": "trr", "xR": "rtt"}[mode]
    pos = [s.d[f"{'truth' if c == 't' else 'reco'}_{a}"] for c, a in zip(src, "xyz")]
    return pos, [a for a, c in zip("xyz", src) if c == "t"]


_margin = float(load_analysis_info(ROOT).get("BACKGROUND_SAMPLES", {}).get("truth_containment_margin_cm", 10.0))


def scan_mask(s, fid, mode):
    pos, truth_axes = mode_positions(s, mode)
    info = s.info
    dx = info["DETECTOR_SIZE_X"] + 2 * info["DETECTOR_GAP_X"]
    dy = info["DETECTOR_SIZE_Y"] + 2 * info["DETECTOR_GAP_Y"]
    m = build_fiducial_spatial_mask({"Reco": {"RecoX": pos[0], "RecoY": pos[1], "RecoZ": pos[2]}}, s.config, dx, dy, info, args.folder, fid)
    lo, hi = bounds(info)
    for i, a in enumerate("xyz"):
        if a in truth_axes and mode != "xTnc":
            m &= (pos[i] >= lo[i] - _margin) & (pos[i] <= hi[i] + _margin)
    return m


def scan_grid(data, config, mode, fxs, fys):
    """S efficiency and weighted background (gamma+neutron, radiological separately) on a (FiducialX, FiducialY) grid."""
    S, cut = data[config]["s"], best_cut(config)
    base = {n: S[n].window & S[n].cut(cut) for n in S}
    out = {k: np.zeros((len(fxs), len(fys))) for k in ("eff", "b", "rad", "nb")}
    tot_s = wsum(S["marley"].w, base["marley"])
    for i, fx in enumerate(fxs):
        for j, fy in enumerate(fys):
            fid = {"FiducialX": fx, "FiducialY": fy, "FiducialZ": 0}
            out["eff"][i, j] = wsum(S["marley"].w, base["marley"] & scan_mask(S["marley"], fid, mode)) / tot_s
            for n in ("gamma", "neutron"):
                if n in S:
                    m = base[n] & scan_mask(S[n], fid, mode)
                    out["b"][i, j] += wsum(S[n].w, m)
                    out["nb"][i, j] += m.sum()
            if "radiological" in S:
                out["rad"][i, j] = wsum(S["radiological"].w, base["radiological"] & scan_mask(S["radiological"], fid, mode))
    out["tot_s"] = tot_s
    return out


def best_of(g, fxs, fys, i_sel, j_sel, min_nb=5):
    """Volume maximising S/sqrt(B_gamma+neutron) among grid points keeping >= min_nb background MC events."""
    fom = np.where(g["nb"] >= min_nb, g["eff"] * g["tot_s"] / np.sqrt(np.maximum(g["b"], 1e-12)), -np.inf)
    sub = np.full_like(fom, -np.inf)
    sub[np.ix_(i_sel, j_sel)] = fom[np.ix_(i_sel, j_sel)]
    k = np.unravel_index(np.argmax(sub), sub.shape)
    return k, sub[k]


def fig_xy_scans(data):
    """Best S/sqrt(B) proxy per mode for the 2D, X-only and Y-only scans, plus the signal-vs-background curves."""
    configs = [c for c in FACE_CONFIGS if c in data]
    fxs = fys = np.arange(0, 341, 20.0)
    table, curves = [], {}
    for config in configs:
        for mode, mlabel in MODES:
            g = scan_grid(data, config, mode, fxs, fys)
            curves[(config, mode)] = g
            for scan, isel, jsel in (("X and Y", range(len(fxs)), range(len(fys))), ("X only (Y=0)", range(len(fxs)), [0]), ("Y only (X=0)", [0], range(len(fys)))):
                (i, j), fom = best_of(g, fxs, fys, list(isel), list(jsel))
                table.append([NICE.get(config, config), scan, mlabel, f"{fxs[i]:g}/{fys[j]:g}", f"{100 * g['eff'][i, j]:.1f}", f"{g['b'][i, j]:.4g}", int(g["nb"][i, j]),
                              f"{g['rad'][i, j]:.4g}", ("n/a" if not np.isfinite(fom) else f"{fom:.2f}")])
    headers = ["Config", "Scan", "Position source", "Best X/Y [cm]", "Signal eff. [%]", "Σw γ+n", "N_MC γ+n", "Σw radiological (not in FoM)", "S/√B (γ+n)"]
    write_csv(f"xy_scans_{args.analysis.lower()}", headers, table)

    # signal efficiency vs background surviving: X-only and Y-only curves, reco vs truth vs mixed
    fig, axes = plt.subplots(2, len(configs), figsize=(3.9 * len(configs), 6.2), squeeze=False)
    style = {"reco": (INK2, "-"), "truth": (TRUTH_C, "-"), "xT": (COLORS["gamma"], "--"), "xTnc": ("#eda100", ":"), "xR": (COLORS["neutron"], "--")}
    for ci, config in enumerate(configs):
        for ri, scan in enumerate(("X", "Y")):
            ax = axes[ri][ci]
            for mode, mlabel in MODES:
                g = curves[(config, mode)]
                eff = g["eff"][:, 0] if scan == "X" else g["eff"][0, :]
                b = g["b"][:, 0] if scan == "X" else g["b"][0, :]
                keep = b > 0
                c, ls = style[mode]
                ax.plot(100 * eff[keep], b[keep], color=c, ls=ls, lw=1.4, marker="o", ms=2.5, label=mlabel)
            ax.set_yscale("log")
            ax.set_xlabel("signal efficiency [%]")
            ax.set_title(f"{NICE.get(config, config)}: {scan}-only scan", loc="left", fontsize=8.5)
            if ci == 0:
                ax.set_ylabel("surviving gamma + neutron weight")
    axes[0][0].legend(loc="lower right", fontsize=7)
    fig.suptitle(f"Signal loss against background rejection as the fiducial cut tightens ({args.analysis} cut; radiological excluded, see table)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, f"signal_vs_background_{args.analysis.lower()}")

    # best S/sqrt(B) per mode
    fig, axes = plt.subplots(1, len(configs), figsize=(3.6 * len(configs), 3.4), squeeze=False, sharey=False)
    for ci, config in enumerate(configs):
        ax = axes[0][ci]
        for k, (mode, mlabel) in enumerate(MODES):
            for si, scan in enumerate(("X and Y", "X only (Y=0)", "Y only (X=0)")):
                row = next(r for r in table if r[0] == NICE.get(config, config) and r[1] == scan and r[2] == mlabel)
                v = float(row[-1]) if row[-1] != "n/a" else np.nan
                ax.bar(si + (k - (len(MODES) - 1) / 2) * 0.17, np.nan_to_num(v), 0.16, color=style[mode][0], hatch=None if mode in ("reco", "truth") else "////", edgecolor="white" if mode in ("reco", "truth") else style[mode][0],
                       label=mlabel if si == 0 else None)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["X, Y", "X only", "Y only"], fontsize=8)
        ax.set_title(NICE.get(config, config), loc="left", fontsize=9)
        ax.grid(axis="x", visible=False)
        if ci == 0:
            ax.set_ylabel("best S/√B (γ + n)")
            ax.legend(fontsize=6.5, loc="upper left")
    fig.suptitle(f"Best figure of merit by position source ({args.analysis} cut, radiological excluded)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, f"xy_scan_best_fom_{args.analysis.lower()}")
    return headers, table


def mismatch_stats(s, m):
    """Position- and purity-based mismatch rates of the events selected by m (weighted). Purity terms are NaN without the aux cache."""
    tol = truth_match_purity_config(ROOT)
    dtol, ttol = float(tol["drift_tolerance_cm"]), float(tol["transverse_tolerance_cm"])
    lo, hi = bounds(s.info)
    w = s.w[m]
    tot = w.sum()
    dx = np.abs(s.d["reco_x"] - s.d["truth_x"])[m]
    dyz = np.maximum(np.abs(s.d["reco_y"] - s.d["truth_y"]), np.abs(s.d["reco_z"] - s.d["truth_z"]))[m]
    rx = s.d["reco_x"][m]

    def f(x):
        return float(w[x].sum() / tot)

    def cond(a, b):
        return float(w[a & b].sum() / w[b].sum()) if w[b].sum() > 0 else np.nan

    bad = dx > dtol
    out = {"n_mc": int(m.sum()), "frac_recox_at_edge": f((rx <= lo[0] + 0.5) | (rx >= hi[0] - 0.5)), "frac_dx_gt_drift_tol": f(bad), "frac_dx_gt_30cm": f(dx > 30),
           "frac_dyz_gt_transverse_tol": f(dyz > ttol), "frac_any_tolerance": f(bad | (dyz > ttol)), "drift_tol_cm": dtol, "transverse_tol_cm": ttol}
    keys = ("frac_purity_zero", "frac_purity_below", "frac_purity_below_and_dx", "frac_purity_below_or_dx", "p_dx_given_purity_low", "p_purity_low_given_dx", "frac_dx_among_pure")
    out.update({k: np.nan for k in keys})
    if s.aux is not None and "MatchedOpFlashPur" in s.aux:
        pur = np.nan_to_num(s.aux["MatchedOpFlashPur"].astype(float), nan=0.0)[m]     # absent purity counts as 0
        low = pur < args.purity_threshold
        out.update(frac_purity_zero=f(pur <= 0), frac_purity_below=f(low), frac_purity_below_and_dx=f(low & bad), frac_purity_below_or_dx=f(low | bad),
                   p_dx_given_purity_low=cond(bad, low), p_purity_low_given_dx=cond(low, bad), frac_dx_among_pure=cond(bad, ~low))
    return out


CAND_KEYS = [("main", "Main"), ("mainparent", "MainParent"), ("end", "End"), ("signalparticle", "SignalParticle")]


def truth_key_records(data):
    """Which truth position key describes the reconstructed cluster? Fraction of weight with reco Y,Z (wire-based) and X within 30 cm of each candidate key."""
    recs = []
    for c in data:
        cut = best_cut(c)
        lo, hi = bounds(data[c]["info"])
        for name, s in data[c]["s"].items():
            if s.aux is None:
                continue
            configured = s.truth_keys[0][:-1]
            for label, m in (("window", s.window), ("after_cut", s.window & s.cut(cut))):
                if not m.any():
                    continue
                w = s.w[m]
                for k, nice in CAND_KEYS:
                    if f"{k}_x" not in s.aux:
                        continue
                    a = {ax: s.aux[f"{k}_{ax}"][m] for ax in "xyz"}
                    dyz = np.maximum(np.abs(s.d["reco_y"][m] - a["y"]), np.abs(s.d["reco_z"][m] - a["z"]))
                    dx = np.abs(s.d["reco_x"][m] - a["x"])
                    inbox = np.ones(len(w), bool)
                    for i, ax in enumerate("xyz"):
                        inbox &= (a[ax] >= lo[i] - _margin) & (a[ax] <= hi[i] + _margin)
                    recs.append((c, name, label, nice, nice == configured, int(m.sum()), float(w[dyz < 30].sum() / w.sum()), float(w[dx < 30].sum() / w.sum()),
                                 float(w[(dyz < 30) & (dx < 30)].sum() / w.sum()), float(w[inbox].sum() / w.sum())))
    return recs


TK_COLUMNS = ["config", "species", "selection", "candidate_key", "is_configured_key", "n_mc", "frac_yz_within_30cm", "frac_x_within_30cm", "frac_xyz_within_30cm", "frac_in_active_box"]


def table_truth_keys(data):
    recs = truth_key_records(data)
    headers = ["Config", "Component", "Sample", "N_MC", "configured key"] + [f"{n}: Y,Z [%] / X [%] / in box [%]" for _, n in CAND_KEYS]
    rows = {}
    for c, name, label, nice, conf, n, yz, x, xyz, box in recs:
        r = rows.setdefault((c, name, label), [NICE.get(c, c), name, "window" if label == "window" else "+ cut", n, "", {}])
        if conf:
            r[4] = nice
        r[5][nice] = f"{100 * yz:.0f} / {100 * x:.0f} / {100 * box:.0f}"
    out = [r[:5] + [r[5].get(n, "n/a") for _, n in CAND_KEYS] for r in rows.values()]
    write_csv(f"truth_key_check_{args.analysis.lower()}", headers, out)
    return headers, out



def table_mismatch(data):
    tol = truth_match_purity_config(ROOT)
    dtol, ttol = float(tol["drift_tolerance_cm"]), float(tol["transverse_tolerance_cm"])
    pt = args.purity_threshold
    headers = ["Config", "Component", "Sample", "N_MC", "RecoX at volume edge [%]", f"|ΔX| > {dtol:g} cm [%]", "|ΔX| > 30 cm [%]",
               f"|ΔY| or |ΔZ| > {ttol:g} cm [%]", "either position tolerance [%]",
               "Pur = 0 or absent [%]", f"Pur < {pt:g} [%]", f"Pur < {pt:g} and |ΔX| > {dtol:g} [%]", f"Pur < {pt:g} or |ΔX| > {dtol:g} [%]",
               f"P(|ΔX| > {dtol:g} | Pur < {pt:g}) [%]", f"P(Pur < {pt:g} | |ΔX| > {dtol:g}) [%]", f"|ΔX| > {dtol:g} among Pur ≥ {pt:g} [%]"]
    pc = lambda v: "n/a" if not np.isfinite(v) else f"{100 * v:.1f}"
    rows = []
    for config in [c for c in FACE_CONFIGS if c in data]:
        cut = best_cut(config)
        for name in ("marley", "gamma", "neutron", "radiological"):
            s = data[config]["s"].get(name)
            if s is None:
                continue
            for label, m in (("window", s.window), ("+ cut", s.window & s.cut(cut))):
                if not m.any():
                    continue
                st = mismatch_stats(s, m)
                rows.append([NICE.get(config, config), name, label, st["n_mc"]] + [pc(st[k]) for k in (
                    "frac_recox_at_edge", "frac_dx_gt_drift_tol", "frac_dx_gt_30cm", "frac_dyz_gt_transverse_tol", "frac_any_tolerance", "frac_purity_zero", "frac_purity_below",
                    "frac_purity_below_and_dx", "frac_purity_below_or_dx", "p_dx_given_purity_low", "p_purity_low_given_dx", "frac_dx_among_pure")])
    write_csv(f"flash_mismatch_{args.analysis.lower()}", headers, rows)
    return headers, rows


def run_faces():
    data = {c: l for c in args.config if (l := load_config(c)) is not None}
    if not data:
        sys.exit("No caches found under " + str(CACHE))
    sections = [
        ("Table F1 — Entry-face composition of the background",
         "Share of Σw by nearest wall of the truth position (walls: HD central x = ±360, HD lateral x = 0 only, VD x = ±330; Y and Z combine both faces). 'Outside' = truth position beyond the active box. "
         f"Cut = default {args.analysis} best cut; fiducial = default volume, reco position. The last column is the weight entering through the X face(s) "
         "(HD central both walls, HD lateral x = 0 only, VD the top face = X high).", fig_face_composition(data)),
        ("Table F2 — |RecoX − truth X| within 100 cm of the entry X face(s)",
         "Gamma and neutron only. 'RecoX at the volume edge' is the clipped fraction.", fig_x_entry_residuals(data)),
        ("Table F3 — Fiducial scans with truth and reco coordinates mixed",
         "Grid FiducialX/FiducialY 0–340 cm in 20 cm steps, FiducialZ = 0 (Z is not applied for Truncated). Figure of merit S/√B with B = gamma + neutron at the "
         "analysis cut, requiring ≥ 5 surviving gamma+neutron MC events; radiological (2–4 MC events) is listed but excluded. Truth modes carry the containment cut on "
         "the truth axes only, and no position-consistency cut. This is a proxy, not the significance.", fig_xy_scans(data)),
        ("Table F4 — Flash-mismatch and saturation rates",
         "Reco vs truth position disagreement per species. Position tolerances are the `truth_match_purity` values. Purity terms use the backtracked `MatchedOpFlashPur` "
         "(absent or NaN counts as 0; threshold `--purity_threshold`) and need the `extract_aux` caches.", table_mismatch(data)),
        ("Table F5 — Which truth key describes the reconstructed cluster?",
         "Weighted share of events whose reco Y,Z (wire-based cluster position) and reco X lie within 30 cm of each candidate truth key, and whose key lies inside the active box. "
         "The configured key (`BACKGROUND_SAMPLES.truth_position_keys`; marley uses SignalParticle) is in the 'configured key' column.", table_truth_keys(data)),
    ]
    out = Path(ROOT) / f"output/docs/truth_position_faces_{args.analysis.lower()}.md"
    lines = ["# Truth-position diagnostics: entry faces and X residuals", "",
             "_Generated by `src/physics/signal/truth_position_study.py --stage faces`. Weights are `SignalParticleWeight` only._", ""]
    for title, note, (h, r) in sections:
        lines += [f"## {title}", "", note, "", md_table(h, r), ""]
    out.write_text("\n".join(lines))
    print(f"[doc] {out}")


# ==========================================================================================
# export stage: tidy numeric DataFrames (pickle) for plotting outside this repository
# ==========================================================================================

EXPORT = CACHE / args.export_dir


def _write(name, rows, columns=None):
    import pandas as pd
    df = pd.DataFrame(rows, columns=columns)
    EXPORT.mkdir(parents=True, exist_ok=True)
    df.to_pickle(EXPORT / f"{name}.pkl", protocol=4)      # protocol 4: readable by any pandas >= 1.0 / Python >= 3.4
    print(f"[export] {name}.pkl  {df.shape}")
    return df


MISMATCH_COLS = ["n_mc", "frac_recox_at_edge", "frac_dx_gt_drift_tol", "frac_dx_gt_30cm", "frac_dyz_gt_transverse_tol", "frac_any_tolerance", "frac_purity_zero", "frac_purity_below",
                 "frac_purity_below_and_dx", "frac_purity_below_or_dx", "p_dx_given_purity_low", "p_purity_low_given_dx", "frac_dx_among_pure"]


def run_export():
    import subprocess
    data = {c: l for c in args.config if (l := load_config(c)) is not None}
    if not data:
        sys.exit("No caches found under " + str(CACHE))
    configs = list(data)
    args.apply_cuts_walls = False

    # ---- analysis-independent -------------------------------------------------------------
    grid = np.arange(0, 301, 1.0)
    rows = []
    for c in configs:
        for name in ("marley", "gamma", "neutron", "radiological"):
            s = data[c]["s"].get(name)
            if s is None or not s.window.any():
                continue
            for kind in ("truth", "reco"):
                dist, w = s.wall_distance(kind)[s.window], s.w[s.window]
                cdf = np.array([w[dist <= g].sum() for g in grid]) / w.sum()
                rows += [(c, name, kind, float(g), float(v), int(s.window.sum())) for g, v in zip(grid, cdf)]
    _write("wall_cdf", rows, ["config", "species", "position", "distance_cm", "cdf", "n_mc"])

    rows = []
    for c in configs:
        for bkg in ("gamma", "neutron", "radiological"):
            res = shell_ratio(data, c, bkg)
            if res:
                rows += [(c, bkg, lo, hi, float(r), float(e)) for (lo, hi), (r, e) in zip(SHELLS, res)]
    _write("shell_ratio", rows, ["config", "species", "shell_lo_cm", "shell_hi_cm", "ratio_vs_100_200cm", "ratio_err"])

    edges = np.arange(-150.0, 155.0, 5.0)
    rows, summ = [], []
    for c in configs:
        for name in ("marley", "gamma", "neutron", "radiological"):
            s = data[c]["s"].get(name)
            if s is None or not s.window.any():
                continue
            w = s.w[s.window]
            for a in "xyz":
                delta = (s.d[f"reco_{a}"] - s.d[f"truth_{a}"])[s.window]
                h, _ = np.histogram(np.clip(delta, -150 + 1e-6, 150 - 1e-6), bins=edges, weights=w)
                n, _ = np.histogram(np.clip(delta, -150 + 1e-6, 150 - 1e-6), bins=edges)
                rows += [(c, name, a, float(lo), float(hi), float(v / w.sum()), int(k)) for lo, hi, v, k in zip(edges[:-1], edges[1:], h, n)]
                summ.append((c, name, a, wmedian(delta, w), float(w[np.abs(delta) < args.agree_cm].sum() / w.sum()), float(w[np.abs(delta) >= 150].sum() / w.sum()), int(s.window.sum())))
    _write("residual_hist", rows, ["config", "species", "axis", "bin_lo_cm", "bin_hi_cm", "weight_fraction", "n_mc"])
    summ = [r + (bool(abs(r[3]) > 150.0),) for r in summ]
    _write("residual_summary", summ, ["config", "species", "axis", "median_cm", "frac_within_agree_cm", "frac_beyond_150cm", "n_mc_window", "median_off_scale_150cm"])
    wide = np.arange(-700.0, 705.0, 20.0)
    rows = []
    for c in configs:
        for name in ("marley", "gamma", "neutron", "radiological"):
            s_ = data[c]["s"].get(name)
            if s_ is None or not s_.window.any():
                continue
            w = s_.w[s_.window]
            for a in "xyz":
                delta = np.clip((s_.d[f"reco_{a}"] - s_.d[f"truth_{a}"])[s_.window], -700 + 1e-6, 700 - 1e-6)
                h, _ = np.histogram(delta, bins=wide, weights=w)
                n, _ = np.histogram(delta, bins=wide)
                rows += [(c, name, a, float(lo), float(hi), float(v / w.sum()), int(k)) for lo, hi, v, k in zip(wide[:-1], wide[1:], h, n)]
    _write("residual_hist_wide", rows, ["config", "species", "axis", "bin_lo_cm", "bin_hi_cm", "weight_fraction", "n_mc"])

    # ---- cut-dependent (one block per analysis) -------------------------------------------
    sig, vols, cuts = [], [], []
    surv, faces, entry_h, entry_s, scan, passf, neut, mism = ([] for _ in range(8))
    split, xeff, best_rows = [], [], []
    for analysis in ("DAYNIGHT", "HEP", "SENSITIVITY"):
        args.analysis = analysis
        for c in configs:
            cut = best_cut(c)
            fr, ft = volume(fiducials(), c), volume(fiducials("_fiduc_truth"), c)
            cuts.append((analysis, c, cut["NHits"], cut["OpHits"], cut["AdjCl"]))
            vols.append((analysis, c, "default", *(fr[k] if fr else np.nan for k in ("FiducialX", "FiducialY", "FiducialZ"))))
            vols.append((analysis, c, "fiduc_truth", *(ft[k] if ft else np.nan for k in ("FiducialX", "FiducialY", "FiducialZ"))))
            S = data[c]["s"]

            res, _, _ = pass_fractions(data, c)
            for group, (vals, n) in res.items():
                passf += [(analysis, c, group, k, lab, float(v), n) for k, (lab, v) in enumerate(zip(PF_LABELS, vals))]

            for name in ("marley", "gamma", "neutron", "radiological"):
                if name not in S:
                    continue
                for label, m in (("window", S[name].window), ("after_cut", S[name].window & S[name].cut(cut))):
                    if m.any():
                        st = mismatch_stats(S[name], m)
                        mism.append((analysis, c, name, label, *(st[k] for k in MISMATCH_COLS)))

            comp = {n: v for n, v in ((n, stage_masks(S[n], cut, fr)) for n in S)}
            tot_reco = sum(wsum(S[n].w, comp[n][2][1]) for n in BKG if n in S)
            tot_truth = sum(wsum(S[n].w, S[n].window & S[n].cut(cut) & S[n].fiducial(ft, "truth")) for n in BKG if n in S) if ft else np.nan
            for pipe, sel_fn in (("reco fiducial", lambda s_: s_.window & s_.cut(cut) & s_.fiducial(fr, "reco")),
                                 ("truth pipeline", lambda s_: s_.window & s_.cut(cut) & s_.fiducial(ft, "truth") if ft else s_.window & s_.cut(cut)),
                                 ("no fiducial", lambda s_: s_.window & s_.cut(cut))):
                gn = [(n, sel_fn(S[n])) for n in ("gamma", "neutron") if n in S]
                rad = sel_fn(S["radiological"]) if "radiological" in S else None
                w_gn = sum(wsum(S[n].w, m) for n, m in gn)
                w_rad = wsum(S["radiological"].w, rad) if rad is not None else 0.0
                split.append((analysis, c, pipe, float(w_gn), float(w_rad), float(w_rad / (w_gn + w_rad)) if (w_gn + w_rad) > 0 else np.nan,
                              int(sum(m.sum() for _, m in gn)), int(rad.sum()) if rad is not None else 0,
                              float(w_gn / (w_gn + w_rad)) if (w_gn + w_rad) > 0 else np.nan))
            for n in S:
                s = S[n]
                idx, dist = face_split(s)
                base = s.window & s.cut(cut)
                truth_sel = base & s.fiducial(ft, "truth") if ft else base
                w = s.w[comp[n][2][1]]
                surv.append((analysis, c, n, int(base.sum()), int(comp[n][2][1].sum()), int(truth_sel.sum()), float(w.sum()), float(w.mean()) if len(w) else np.nan,
                             neff(w), float(w.max() / w.sum()) if len(w) else np.nan,
                             float(w.sum() / tot_reco) if (n in BKG and tot_reco > 0) else np.nan,
                             float(wsum(s.w, truth_sel) / tot_truth) if (n in BKG and np.isfinite(tot_truth) and tot_truth > 0) else np.nan))
                if n in BKG:
                    for label, m in comp[n]:
                        ww = s.w[m]
                        for k, fname in enumerate(FACE_NAMES):
                            faces.append((analysis, c, n, label, fname, "face", float(ww[idx[m] == k].sum() / ww.sum()) if ww.sum() > 0 else np.nan, int(m.sum()), float(ww.sum())))
                        faces.append((analysis, c, n, label, "outside", "flag", float(ww[s.outside_box("truth")[m]].sum() / ww.sum()) if ww.sum() > 0 else np.nan, int(m.sum()), float(ww.sum())))

            lo, hi = bounds(S["marley"].info)
            for n in ("gamma", "neutron"):
                if n not in S:
                    continue
                s = S[n]
                idx, dist = face_split(s)
                for use_cut in (False, True):
                    m = s.window & (s.cut(cut) if use_cut else True) & np.isin(idx, entry_x_faces(c)) & (dist < 100)
                    if not m.any():
                        continue
                    dx = np.abs(s.d["reco_x"] - s.d["truth_x"])[m]
                    w = s.w[m]
                    rx = s.d["reco_x"][m]
                    eb = np.arange(0, 710, 10.0)
                    h, _ = np.histogram(np.clip(dx, 0, 700 - 1e-6), bins=eb, weights=w)
                    entry_h += [(analysis, c, n, use_cut, float(a), float(b), float(v / w.sum()), int(k)) for a, b, v, k in zip(eb[:-1], eb[1:], h, np.histogram(np.clip(dx, 0, 700 - 1e-6), bins=eb)[0])]
                    entry_s.append((analysis, c, n, use_cut, int(m.sum()), float(w.sum()), wmedian(dx, w), float(w[dx > 30].sum() / w.sum()), float(w[dx > 100].sum() / w.sum()),
                                    float(w[(rx <= lo[0] + 0.5) | (rx >= hi[0] - 0.5)].sum() / w.sum()), bool(m.sum() < 10)))

            for n in ("neutron", "gamma"):
                if n == "gamma" and not c.startswith("vd"):
                    continue
                s = S.get(n)
                if s is None:
                    continue
                m = s.window & s.cut(cut)
                delta = s.pos("reco") - s.pos("truth")
                neut += [(analysis, c, n, *s.pos("reco")[i], *s.pos("truth")[i], float(s.w[i]), float(np.sqrt((delta[i] ** 2).sum())), bool((np.abs(delta[i]) < args.agree_cm).all()))
                         for i in np.flatnonzero(m)]

            fxs = fys = np.arange(0, 341, 20.0)
            grids = {}
            for mode, mlabel in MODES:
                g = grids[mode] = scan_grid(data, c, mode, fxs, fys)
                for i, fx in enumerate(fxs):
                    for j, fy in enumerate(fys):
                        scan.append((analysis, c, mode, mlabel, float(fx), float(fy), float(g["eff"][i, j]), float(g["b"][i, j]), float(g["rad"][i, j]), int(g["nb"][i, j]), float(g["tot_s"])))
            for mode, mlabel in MODES:
                res_ = {}
                for scan_name, isel, jsel in (("X and Y", list(range(len(fxs))), list(range(len(fys)))), ("X only (Y=0)", list(range(len(fxs))), [0]), ("Y only (X=0)", [0], list(range(len(fys))))):
                    (bi, bj), fom = best_of(grids[mode], fxs, fys, isel, jsel)
                    res_[scan_name] = (float(fxs[bi]), float(fys[bj]), float(grids[mode]["eff"][bi, bj]), float(grids[mode]["b"][bi, bj]), float(grids[mode]["rad"][bi, bj]),
                                       int(grids[mode]["nb"][bi, bj]), float(fom) if np.isfinite(fom) else np.nan)
                for scan_name, v in res_.items():
                    same = [k for k, o in res_.items() if k != scan_name and o[:2] == v[:2]]
                    best_rows.append((analysis, c, mode, mlabel, scan_name, *v, "; ".join(same)))
            xs = {}
            if fr:
                xs["default volume X"] = fr["FiducialX"]
            if ft:
                xs["truth volume X"] = ft["FiducialX"]
            for mode, lab in (("reco", "FoM-best X, reco"), ("xT", "FoM-best X, truth X")):
                (bi, _), _fom = best_of(grids[mode], fxs, fys, list(range(len(fxs))), [0])
                xs[lab] = float(fxs[bi])
            base = {n: S[n].window & S[n].cut(cut) for n in S}
            tot_s = wsum(S["marley"].w, base["marley"])
            for lab, xv in xs.items():
                fid = {"FiducialX": xv, "FiducialY": 0, "FiducialZ": 0}
                row = [analysis, c, lab, float(xv)]
                for mode in ("reco", "xT"):
                    row.append(wsum(S["marley"].w, base["marley"] & scan_mask(S["marley"], fid, mode)) / tot_s)
                for mode in ("reco", "xT"):
                    row.append(sum(wsum(S[n].w, base[n] & scan_mask(S[n], fid, mode)) for n in ("gamma", "neutron") if n in S))
                xeff.append(tuple(row))

    nofid = {(r[0], r[1]): r[7] for r in split if r[2] == "no fiducial"}
    split = [r + (nofid[(r[0], r[1])], bool(r[7] == 0 and nofid[(r[0], r[1])] > 0)) for r in split]
    _write("background_split", split, ["analysis", "config", "selection", "sum_w_gamma_neutron", "sum_w_radiological", "radiological_fraction", "n_mc_gamma_neutron",
                                       "n_mc_radiological", "gamma_neutron_fraction", "n_mc_radiological_no_fiducial", "radiological_all_rejected"])
    _write("signal_efficiency_x", xeff, ["analysis", "config", "x_definition", "fiducial_x_cm", "signal_eff_reco_x", "signal_eff_truth_x",
                                         "sum_w_gn_reco_x", "sum_w_gn_truth_x"])
    _write("flash_mismatch", mism, ["analysis", "config", "species", "selection"] + MISMATCH_COLS)
    tk = []
    for analysis in ("DAYNIGHT", "HEP", "SENSITIVITY"):
        args.analysis = analysis
        tk += [(analysis, *r) for r in truth_key_records(data)]
    _write("truth_key_check", tk, ["analysis"] + TK_COLUMNS)
    _write("cuts", cuts, ["analysis", "config", "nhits_min", "ophits_min", "adjcl_max"])
    _write("fiducial_volumes", vols, ["analysis", "config", "variant", "fiducial_x_cm", "fiducial_y_cm", "fiducial_z_cm"])
    _write("pass_fractions", passf, ["analysis", "config", "group", "variant_index", "variant", "pass_fraction", "n_mc"])
    _write("survivor_statistics", surv, ["analysis", "config", "species", "n_mc_window_cut", "n_mc_reco_fiducial", "n_mc_truth_pipeline", "sum_w_reco_fiducial",
                                         "mean_w_per_event", "n_eff", "largest_event_share", "share_of_background_reco", "share_of_background_truth_pipeline"])
    _write("face_composition", faces, ["analysis", "config", "species", "stage", "face", "kind", "weight_fraction", "n_mc", "sum_w"])
    _write("x_entry_hist", entry_h, ["analysis", "config", "species", "after_cut", "bin_lo_cm", "bin_hi_cm", "weight_fraction", "n_mc"])
    _write("x_entry_summary", entry_s, ["analysis", "config", "species", "after_cut", "n_mc", "sum_w", "median_abs_dx_cm", "frac_dx_gt_30cm", "frac_dx_gt_100cm", "frac_recox_at_edge", "too_few_mc"])
    _write("xy_scan_best", best_rows, ["analysis", "config", "mode", "mode_label", "scan", "best_x_cm", "best_y_cm", "signal_efficiency", "sum_w_gamma_neutron",
                                       "sum_w_radiological", "n_mc_gamma_neutron", "sn_over_sqrt_b", "identical_volume_as"])
    _write("xy_scan_grid", scan, ["analysis", "config", "mode", "mode_label", "fiducial_x_cm", "fiducial_y_cm", "signal_efficiency", "sum_w_gamma_neutron", "sum_w_radiological",
                                  "n_mc_gamma_neutron", "sum_w_signal_before"])
    _write("survivors_events", neut, ["analysis", "config", "species", "reco_x", "reco_y", "reco_z", "truth_x", "truth_y", "truth_z", "weight", "dist3d_cm", "agree"])

    sigrows = []
    for c in args.config:
        for analysis, *_ in ANALYSES:
            for label in ("default", "fiduc_truth", "fiduc_truth_refvol"):
                sigrows.append((c, analysis, label, significance(c, analysis, dict(VARIANTS_BY_LABEL)[label])))
    _write("significance", sigrows, ["config", "analysis", "variant", "significance_sigma"])

    try:
        rev = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        rev = "unknown"
    import datetime
    meta = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"), "git_head": rev, "script": "src/physics/signal/truth_position_study.py --stage export",
        "folder": args.folder, "energy_variable": args.energy, "energy_window_mev": [args.emin, args.emax], "agree_cm": args.agree_cm,
        "truth_position_keys": {"marley": "SignalParticleX/Y/Z", "gamma": "EndX/Y/Z", "neutron": "MainX/Y/Z", "radiological": "MainX/Y/Z"},
        "weights": "SignalParticleWeight only (no oscillation, MC-support gate or smoothing)",
        "configs": configs,
        "notes": ["Scan figure of merit uses gamma+neutron only; radiological has 2-4 MC events with weight ~1e5.",
                  "Walls used for distances and entry faces: HD central x=-360 and +360 (x=0 is interior); HD lateral x=0 only (background piles up there; x=360 is not counted); VD x=+-330; Y and Z faces always.",
                  "VD background truth X can lie outside the active box; RecoX is confined to it."],
    }
    (EXPORT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[export] {EXPORT / 'meta.json'}")


VARIANTS_BY_LABEL = [("default", ""), ("fiduc_truth", "_fiduc_truth"), ("fiduc_truth_refvol", "_fiduc_truth_refvol")]


# ==========================================================================================
# driver
# ==========================================================================================

if args.stage == "plot":
    data = {}
    for config in args.config:
        loaded = load_config(config)
        if loaded is None:
            print(f"[warn] no cached marley sample for {config}; run the extract stage. Skipping event-level figures for it.")
            continue
        data[config] = loaded
    if not data:
        sys.exit("No caches found under " + str(CACHE))

    fig_wall_proximity(data)
    shell_rows = fig_shell_ratio(data)
    for config in data:
        fig_residuals(data, config)
    pf = fig_pass_fractions(data)

    sig = sig_table(args.config)
    fig_summary(args.config, sig)

    sections = [
        ("Table 1 — Outer-shell background-to-signal enhancement",
         "Weighted (B/S) in the shell divided by (B/S) in the 100–200 cm shell, truth position. "
         f"{'With' if args.apply_cuts_walls else 'No'} topological cut; {args.emin:g}–{args.emax:g} MeV.",
         table_shells(shell_rows)),
        ("Table 2 — Pass fractions", f"Fraction of the {args.analysis} window+cut sample kept by the fiducial cut. 'Fiducial only' evaluates the truth position with no further cut; the pipeline adds the truth containment cut and, for backgrounds, the position-consistency cut, as `--truth_fiducial` does.",
         table_passfrac(pf)),
        ("Table 3 — Default vs fiduc_truth volumes and significances",
         "`fiduc_truth_refvol` uses the truth position at the default (reco) volume, isolating what position knowledge buys at an unchanged volume. "
         "Units: DayNight and HEP in σ; Sensitivity is the cut-quality Score = ½[χ²☉(θ_react) + χ²_react(θ☉)] (a wrong-hypothesis Δχ², see 04_best_cuts.py / 06_significance.py), so its σ-equivalent is √Score.",
         table_volumes(args.config, sig)),
        ("Table 4 — Surviving MC statistics at the analysis cut",
         f"N_MC at the default {args.analysis} best cut. The weight per event and the share of the background weight show how few MC events carry the background; the last two columns give each component's share of the total background weight before and after the truth pipeline.",
         table_survivors(data)),
    ]
    lines = ["# Truth-position diagnostics (fiduc_truth)", "",
             "_Generated by `src/physics/signal/truth_position_study.py --stage plot`. Weights are `SignalParticleWeight` only; "
             "see the script docstring for the selection._", ""]
    for title, note, (h, r) in sections:
        lines += [f"## {title}", "", note, "", md_table(h, r), ""]
    DOC.write_text("\n".join(lines))
    print(f"[doc] {DOC}")


if args.stage == "faces":
    run_faces()


if args.stage == "export":
    run_export()
