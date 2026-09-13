"""
cutflow_plot.py — Successive cut effects on reconstructed energy spectra
==========================================================================
Histograms reconstructed energy at 5 successive cut stages for each named
sample (signal + background components).

All stages share the same N_total event pool from Ref pkls
(03_analysis.py --export_raw --export_fiducial), eliminating population
mismatch between fiducial and quality-cut stages.

Stages (in order):
  Raw                     : all simulated events, no cuts
  TPC-PDS Matching        : OpFlashPlane == OPFLASH_PLANE & OpFlashPE > 0
  Fiducial                : TPC-PDS Matching & FiducializationMask (spatial + surface)
  NHits                   : Fiducial + NHits >= nhits
  NHits+OpHits            : NHits + OpHits >= ophits
  NHits+OpHits+AdjCl      : NHits+OpHits + AdjCl < adjcl

Components per analysis (from 03_analysis.py --export_raw --export_fiducial):
  Sensitivity : Solar       — AnalysisWeightsSolar_{energy}_Ref
  DayNight    : Solar Day   — AnalysisWeightsSolarDay_{energy}_Ref
                Solar Night — AnalysisWeightsSolarNight_{energy}_Ref
                Solar       — AnalysisWeightsSolar_{energy}_Ref
  HEP         : 8B          — AnalysisWeights8B_{energy}_Ref
                hep         — AnalysisWeightshep_{energy}_Ref
  Background  : (name)      — AnalysisWeights_{energy}_Ref

Required Ref pkls (shared across all components):
  AnalysisData_{energy}_Ref, FiducializationMask_{energy}_{ANALYSIS},
  AnalysisNHits_{energy}_Ref, AnalysisOpHits_{energy}_Ref,
  AnalysisAdjCl_{energy}_Ref, AnalysisOpFlashPlane_{energy}_Ref,
  AnalysisOpFlashPE_{energy}_Ref

Best cuts are resolved in this order:
  1. --nhits / --ophits / --adjcls CLI flags
  2. PNFS highest-sensitivity pkl
  3. Analysis-info defaults (NHits=10, OpHits=2, AdjCl=2)

Outputs — one pkl per named sample + one combined pkl:
  output/data/solar/cutflow/{config}/{name}/{folder}/{analysis}/{config}_{name}_{energy}_{analysis}_Cutflow.pkl
  output/images/solar/cutflow/{folder}/{analysis}/{config}_{name}_{energy}_{analysis}_Cutflow_NHits{n}_...png
"""

import math
import os
import subprocess
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *

save_path  = f"{root}/output/images/solar/cutflow"
data_path  = f"{root}/output/data/solar/cutflow"
ref_path   = f"{root}/output/data/results"

analysis_info = load_analysis_info(str(root))

parser = argparse.ArgumentParser(
    description="Successive cut effects on reconstructed energy spectra (all analysis components)",
    formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=36, width=120),
)
parser.add_argument("--config", nargs="+", type=str, default=["hd_1x2x6_centralAPA"])
parser.add_argument(
    "--signal",
    nargs="+",
    type=str,
    default=["marley"],
    help="Sample names (signal + background). One pkl saved per name.",
)
parser.add_argument(
    "--folder", type=str, choices=["Reduced", "Truncated", "Nominal"], default="Truncated"
)
parser.add_argument(
    "--energy", type=str, default="SolarEnergy", help="Energy label for output naming"
)
parser.add_argument("--analysis", nargs="+", type=str, default=["Sensitivity"])
parser.add_argument("--nhits",  type=int, default=None)
parser.add_argument("--ophits", type=int, default=None)
parser.add_argument("--adjcls", type=int, default=None)
parser.add_argument("--exposure", type=float, default=None, help="Livetime in years for scaling. Defaults to EVALUATION_EXPOSURE_YEARS from params (20 if unset).")
parser.add_argument("--mc_filter_threshold", type=int, default=2, help="Min unweighted MC events per bin in the final-cut stage; bins below are zeroed (matches 03_analysis.py default).")
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug",   action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--plot",    action=argparse.BooleanOptionalAction, default=True)

parser.add_argument(
    "--membrane_veto",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "Accept only cathode/APA optical matches (QUALITY_CUTS.OPFLASH_PLANE, plane 0). "
        "This is the default. --no-membrane_veto additionally accepts membrane and endcap "
        "matches (VD planes 1-4), which HD never produces; unmatched clusters are rejected "
        "by the MatchedOpFlashPE > 0 requirement either way. Used by the membrane_veto study."
    ),
)

args = parser.parse_args()

# Self-dispatch: when multiple configs or analyses given, invoke script once per (config, analysis) pair.
if len(args.config) > 1 or len(args.analysis) > 1:
    from itertools import product as _product
    for _cfg, _ana in _product(args.config, args.analysis):
        _cmd = [
            sys.executable, __file__,
            "--config",               _cfg,
            "--analysis",             _ana,
            "--signal",               *args.signal,
            "--folder",               args.folder,
            "--energy",               args.energy,
            "--mc_filter_threshold",  str(args.mc_filter_threshold),
            "--rewrite"   if args.rewrite       else "--no-rewrite",
            "--debug"     if args.debug         else "--no-debug",
            "--plot"      if args.plot          else "--no-plot",
            "--membrane_veto" if args.membrane_veto else "--no-membrane_veto",
        ]
        if args.exposure is not None:
            _cmd += ["--exposure", str(args.exposure)]
        if args.nhits  is not None:
            _cmd += ["--nhits",  str(args.nhits)]
        if args.ophits is not None:
            _cmd += ["--ophits", str(args.ophits)]
        if args.adjcls is not None:
            _cmd += ["--adjcls", str(args.adjcls)]
        rprint(f"\n[green][CMD][/green] cutflow_plot.py --config {_cfg} --analysis {_ana}")
        subprocess.run(_cmd, check=False)
    sys.exit(0)

# Unwrap single-element lists to plain strings for the rest of the script.
args.config   = args.config[0]
args.analysis = args.analysis[0]

os.makedirs(f"{save_path}/{args.folder.lower()}/{args.analysis.lower()}", exist_ok=True)

info = json.loads(open(f"{root}/config/{args.config}/{args.config}_config.json").read())
_primary = args.signal[0]

if args.exposure is None:
    _params_path = f"{root}/config/{args.config}/{args.config}_params.json"
    _exposure = 20.0
    if os.path.exists(_params_path):
        _exposure = float(json.loads(open(_params_path).read()).get("EVALUATION_EXPOSURE_YEARS", 20.0))
else:
    _exposure = args.exposure

_detector_mass = get_full_detector_mass(args.config, info)  # kT
_scale         = _detector_mass * _exposure                 # kT·yr

# ── Resolve best cuts ─────────────────────────────────────────────────────────

def _extract_cuts(obj) -> Optional[tuple]:
    """Extract (NHits, OpHits, AdjCl) from a highest-sensitivity pkl.

    Handles two formats:
      dict  : {(config, name, energy): {"NHits":..., "OpHits":..., "AdjCl":...}}
              Written by SENSITIVITY analysis.
      DataFrame : rows indexed by metric name (NHits, OpHits, AdjCl, ...),
                  MultiIndex columns (config, name, energy).
                  Written by DayNight / HEP analyses.
    """
    if isinstance(obj, dict):
        key = (args.config, _primary, args.energy)
        row = obj.get(key)
        if row is None:
            raise KeyError(f"Key {key} not found in best-cuts pkl. Available: {list(obj.keys())[:5]}")
        return int(row["NHits"]), int(row["OpHits"]), int(row["AdjCl"])
    else:  # DataFrame
        key = (args.config, _primary, args.energy)
        if key not in obj.columns:
            raise KeyError(f"Column {key} not found in best-cuts DataFrame. Available: {list(obj.columns[:5])}")
        s = obj[key]
        return int(s["NHits"]), int(s["OpHits"]), int(s["AdjCl"])


def _load_pnfs_cuts() -> Optional[tuple]:
    _subdir = args.analysis.upper()
    _base   = f"{info['PATH']}/{_subdir}/{args.folder.lower()}/{args.config}/{_primary}"
    # 04_best_cuts.py writes "highest_SENSITIVITY.pkl"; 05_best_sigmas.py preserves
    # args.analysis casing ("highest_DayNight.pkl", "highest_HEP.pkl").
    _analysis_tag = args.analysis.upper() if args.analysis.lower() == "sensitivity" else args.analysis
    exact = f"{_base}/{args.config}_{_primary}_highest_{_analysis_tag}.pkl"
    if not os.path.exists(exact):
        raise SystemExit(
            f"[ERROR] Best-cuts pkl not found: {exact}\n"
            "Run the full pipeline (04_best_cuts.py / 05_best_sigmas.py) for this config/analysis/folder first."
        )
    return _extract_cuts(pickle.load(open(exact, "rb")))


_nhits  = args.nhits
_ophits = args.ophits
_adjcls = args.adjcls

if None in (_nhits, _ophits, _adjcls):
    _nhits, _ophits, _adjcls = _load_pnfs_cuts()
    rprint(f"[cyan][INFO][/cyan] Best cuts from PNFS: NHits={_nhits} OpHits={_ophits} AdjCl={_adjcls}")

# ── Energy histogram settings — 1 MeV bins ───────────────────────────────────

_e_range = analysis_info.get("RECO_ENERGY_RANGE", [0, 30])
_edges   = np.arange(_e_range[0], _e_range[1] + 1e-9, 1.0)  # exactly 1 MeV per bin
_centers = 0.5 * (_edges[:-1] + _edges[1:])

# ── Stage palette ─────────────────────────────────────────────────────────────

_TRUTH_STAGES: list[dict] = [
    {"label": "Truth", "color": "rgba(34,139,34,0.8)", "dash": "longdash", "width": 2},
]
_FID_STAGES: list[dict] = [
    {"label": "Raw",              "color": "rgba(50,50,50,0.9)",    "dash": "dot",    "width": 2},
    {"label": "TPC-PDS Matching", "color": "rgba(214,39,40,0.9)",   "dash": "dashdot","width": 2},
    {"label": "Fiducial",         "color": "rgba(148,103,189,0.9)", "dash": "dash",   "width": 2},
]
_CUT_STAGES: list[dict] = [
    {"label": "NHits",              "color": "rgba(31,119,180,1)",    "dash": "solid", "width": 2},
    {"label": "NHits+OpHits",       "color": "rgba(255,127,14,1)",    "dash": "solid", "width": 2},
    {"label": "NHits+OpHits+AdjCl", "color": "rgba(44,160,44,1)",    "dash": "solid", "width": 2},
]

# ── Component specs per analysis (signal/marley only) ─────────────────────────
# Each entry: (component_label, weight_pkl_prefix)
# Weight pkl: {ref_path}/{config}/{name}/{folder}/{config}_{name}_{prefix}_{energy}_Ref.pkl

_SIGNAL_COMPONENTS: dict[str, list[tuple[str, str]]] = {
    "Sensitivity": [("Solar",       "AnalysisWeightsSolar")],
    "DayNight":    [("Solar Day",   "AnalysisWeightsSolarDay"),
                    ("Solar Night", "AnalysisWeightsSolarNight"),
                    ("Solar",       "AnalysisWeightsSolar")],
    "HEP":         [("8B",          "AnalysisWeights8B"),
                    ("hep",         "AnalysisWeightshep")],
}

# AnalysisWeightsSolarDay / ...Night are *conditional* rates — each is the rate the
# detector would see if it were day (or night) for the whole exposure, because
# lib/weights.py normalises every nadir slice by its own weight sum. This script
# reports events observed in `_exposure` detector-years, so each slice is scaled by
# the fraction of the exposure it actually occupies. After scaling,
# Solar Day + Solar Night == Solar, each roughly half of it.
# Must match daynight/01_daynight.py --day_fraction (same DAY_FRACTION key).
_DAY_FRACTION = float(analysis_info.get("DAY_FRACTION", 0.493))
_COMPONENT_EXPOSURE_FRACTION: dict[str, float] = {
    "Solar Day":   _DAY_FRACTION,
    "Solar Night": 1.0 - _DAY_FRACTION,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _histogram(energy: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(energy[mask], bins=_edges, weights=weights[mask])
    return counts



def _ref_pkl(name: str, filename: str) -> str:
    return (
        f"{ref_path}/{args.config}/{name}/{args.folder.lower()}/"
        f"{args.config}_{name}_{filename}.pkl"
    )


_op_plane_cut = analysis_info.get("QUALITY_CUTS", {}).get("OPFLASH_PLANE", 0)


def _load_all_stages(name: str, weight_filename: str) -> list[np.ndarray]:
    """
    Load all 6 stage histograms from Ref pkls using a single consistent event pool.

    Stages returned (in order, matching _FID_STAGES + _CUT_STAGES):
      0  Raw                 : all events
      1  TPC-PDS Matching    : OpFlashPlane == cut & OpFlashPE > 0
      2  Fiducial            : TPC-PDS Matching & FiducializationMask
      3  NHits               : Fiducial + NHits >= nhits
      4  NHits+OpHits        : Fiducial + NHits + OpHits >= ophits
      5  NHits+OpHits+AdjCl  : Fiducial + NHits + OpHits + AdjCl < adjcl

    The optional "Interacting" stage (all truth interactions before reco efficiency)
    is handled separately by _try_load_truth_stage.

    weight_filename: pkl prefix for the component-specific weight array
      e.g. "AnalysisWeightsSolar", "AnalysisWeights8B", "AnalysisWeights".
    """
    required = [
        _ref_pkl(name, f"AnalysisData_{args.energy}_Ref"),
        _ref_pkl(name, f"{weight_filename}_{args.energy}_Ref"),
        _ref_pkl(name, f"FiducializationMask_{args.energy}_{args.analysis.upper()}"),
        _ref_pkl(name, f"AnalysisNHits_{args.energy}_Ref"),
        _ref_pkl(name, f"AnalysisOpHits_{args.energy}_Ref"),
        _ref_pkl(name, f"AnalysisAdjCl_{args.energy}_Ref"),
        _ref_pkl(name, f"AnalysisOpFlashPlane_{args.energy}_Ref"),
        _ref_pkl(name, f"AnalysisOpFlashPE_{args.energy}_Ref"),
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise SystemExit(
            f"[ERROR] Missing Ref pkls for '{name}' — run 03_analysis.py --export_raw --export_fiducial first:\n"
            + "\n".join(f"  {p}" for p in missing)
        )

    # Freshness check: Ref pkls must not predate the Rebin pkl (which feeds HEP Counts / significance_plot.py).
    # If 03_analysis.py was rerun without --export_raw, Rebin pkl is newer → data inconsistent.
    _dir_key   = "signal" if "marley" in name else "background"
    _rebin_pkl = (
        f"{info['PATH']}/{_dir_key}/{args.folder.lower()}/{args.analysis.upper()}/"
        f"{args.config}/{name}/{args.config}_{name}_{args.energy}_Rebin.pkl"
    )
    if os.path.exists(_rebin_pkl):
        _rebin_mtime = os.path.getmtime(_rebin_pkl)
        _ref_mtime   = min(os.path.getmtime(p) for p in required)
        if _ref_mtime < _rebin_mtime - 60:
            rprint(
                f"[yellow][WARNING][/yellow] Ref pkls for '{name}' are "
                f"{(_rebin_mtime - _ref_mtime) / 60:.1f} min older than the Rebin pkl — "
                f"Cutflow and HEP Counts may be from different 03_analysis.py runs.\n"
                f"  Regenerate: python3 src/physics/signal/03_analysis.py "
                f"--config {args.config} --signal {name} --folder {args.folder} "
                f"--energy {args.energy} --analysis {args.analysis} "
                f"--export_raw --export_fiducial --best_cuts_only"
            )

    reco     = np.asarray(pickle.load(open(required[0], "rb")), dtype=float)
    weights  = np.asarray(pickle.load(open(required[1], "rb")), dtype=float)
    geo_fid  = np.asarray(pickle.load(open(required[2], "rb")), dtype=bool)
    nhits    = np.asarray(pickle.load(open(required[3], "rb")), dtype=int)
    ophits   = np.asarray(pickle.load(open(required[4], "rb")), dtype=int)
    adjcl    = np.asarray(pickle.load(open(required[5], "rb")), dtype=int)
    op_plane = np.asarray(pickle.load(open(required[6], "rb")), dtype=int)
    op_pe    = np.asarray(pickle.load(open(required[7], "rb")), dtype=float)

    flash_mask = accepted_flash_planes(op_plane, str(root), args.membrane_veto) & (op_pe > 0)
    fid        = geo_fid & flash_mask
    pre_mask   = np.ones(len(reco), dtype=bool)
    nhits_mask = fid & (nhits  >= _nhits)
    op_mask    = nhits_mask & (ophits >= _ophits)
    full_mask  = op_mask    & (adjcl  <  _adjcls)

    # mc_filter on final stage only — matches 03_analysis.py Rebin pkl behaviour.
    # Bins with fewer than mc_filter_threshold unweighted MC events are zeroed.
    _mc_counts, _ = np.histogram(reco[full_mask], bins=_edges)
    _mc_filter     = (_mc_counts >= args.mc_filter_threshold).astype(float)

    h_full = _histogram(reco, weights, full_mask) * _mc_filter

    def _h_and_err(mask, mc_filter=None):
        h = _histogram(reco, weights, mask)
        w2, _ = np.histogram(reco[mask], bins=_edges, weights=weights[mask] ** 2)
        if mc_filter is not None:
            h  = h  * mc_filter
            w2 = w2 * mc_filter
        mc, _ = np.histogram(reco[mask], bins=_edges)
        err = np.sqrt(w2)
        return h, err, mc

    return [
        _h_and_err(pre_mask),
        _h_and_err(flash_mask),
        _h_and_err(fid),
        _h_and_err(nhits_mask),
        _h_and_err(op_mask),
        _h_and_err(full_mask, mc_filter=_mc_filter),
    ]

# ── Smoothing config (1-D, analysis-specific — matches significance_plot.py) ──

_smoothing_cfg = get_smoothing_config(
    str(root), analysis_name=args.analysis.upper(), dimensions="1d", stage="significance"
)


def _smooth(h: np.ndarray, component: str) -> np.ndarray:
    cfg = get_component_smoothing_config(_smoothing_cfg, component)
    return smooth_histogram_with_config(h, cfg)


# ── Truth (Interacting) stage ─────────────────────────────────────────────────
# Maps reco-weight filename prefix → truth-weight filename prefix.
# TruthEnergy_Ref and TruthWeights*_Ref are written by 03_analysis.py for
# non-radiological samples only; radiological has no single-particle-gun truth.

_TRUTH_WEIGHT_MAP: dict[str, str] = {
    "AnalysisWeightsSolar":      "TruthWeightsSolar",
    "AnalysisWeightsSolarDay":   "TruthWeightsSolarDay",
    "AnalysisWeightsSolarNight": "TruthWeightsSolarNight",
    "AnalysisWeights8B":         "TruthWeights8B",
    "AnalysisWeightshep":        "TruthWeightshep",
    "AnalysisWeights":           "TruthWeights",
}


def _try_load_truth_stage(name: str, weight_filename: str) -> Optional[tuple]:
    """Load truth-level (Interacting) histogram, or return None if unavailable.

    Uses TruthEnergy_Ref (all truth interactions) and the corresponding truth
    weight pkl.  Returns (h, err, mc) or None.
    """
    truth_weight_file = _TRUTH_WEIGHT_MAP.get(weight_filename)
    if truth_weight_file is None:
        return None

    energy_pkl = _ref_pkl(name, "TruthEnergy_Ref")
    weight_pkl = _ref_pkl(name, f"{truth_weight_file}_Ref")

    if not os.path.exists(energy_pkl) or not os.path.exists(weight_pkl):
        return None

    t_energy  = np.asarray(pickle.load(open(energy_pkl,  "rb")), dtype=float)
    t_weights = np.asarray(pickle.load(open(weight_pkl,  "rb")), dtype=float)

    h,  _ = np.histogram(t_energy, bins=_edges, weights=t_weights)
    w2, _ = np.histogram(t_energy, bins=_edges, weights=t_weights ** 2)
    mc, _ = np.histogram(t_energy, bins=_edges)
    return h, np.sqrt(w2), mc


# ── Per-name processing ───────────────────────────────────────────────────────

_cut_label    = f"NHits{_nhits}_OpHits{_ophits}_AdjCl{_adjcls}"
_all_rows: list[dict] = []  # accumulates rows from every name for the combined pkl

for _name in args.signal:

    cutflow_rows: list[dict]                         = []
    _component_order: list[tuple[str, str]]          = []
    _component_hists: dict[tuple[str, str], list]    = {}

    # ── Resolve components for this sample ────────────────────────────────────
    # Signal (marley): components defined by analysis type.
    # Background: single component resolved from Weighted_Distributions.

    _is_signal = "marley" in _name
    if _is_signal:
        _components = _SIGNAL_COMPONENTS.get(args.analysis, _SIGNAL_COMPONENTS["Sensitivity"])
    else:
        _components = [(_name, "AnalysisWeights")]

    # ── Per-component stage histograms ────────────────────────────────────────

    for _component_label, _weight_filename in _components:
        stage_hists = _load_all_stages(_name, _weight_filename)

        # Prepend "Interacting" stage if truth-level pkls are available.
        _truth_entry = _try_load_truth_stage(_name, _weight_filename)
        _active_stage_pairs = (
            [(_TRUTH_STAGES[0], _truth_entry)] if _truth_entry is not None else []
        ) + list(zip(_FID_STAGES + _CUT_STAGES, stage_hists))

        key = (_name, _component_label)
        _component_order.append(key)
        _component_hists[key] = _active_stage_pairs  # store pairs, not raw hists

        _meta = {
            "Config": args.config, "Name": _name, "Folder": args.folder,
            "Component": _component_label, "NHits": _nhits, "OpHits": _ophits, "AdjCl": _adjcls,
            "Exposure": _exposure, "ExposureUnit": "year",
            "EnergyUnit": "MeV", "CountsUnit": f"events / MeV / {_exposure:.0f} yr",
        }
        # Day/night slices occupy only part of the exposure — see _COMPONENT_EXPOSURE_FRACTION.
        _exposure_fraction = _COMPONENT_EXPOSURE_FRACTION.get(_component_label, 1.0)
        _comp_scale = _scale * _exposure_fraction

        for stage, (h, h_err, mc) in _active_stage_pairs:
            h_scaled   = h     * _comp_scale   # events/(kT·yr) × kT·yr → events/MeV (1 MeV bins)
            err_scaled = h_err * _comp_scale
            sh_scaled  = _smooth(h,     _component_label) * _comp_scale
            se_scaled  = _smooth(h_err, _component_label) * _comp_scale
            cutflow_rows.append({
                **_meta, "Stage": stage["label"],
                "Energy":               _centers.tolist(),
                "Counts":               h_scaled.tolist(),
                "CountsError":          err_scaled.tolist(),
                "SmoothedCounts":       sh_scaled.tolist(),
                "SmoothedCountsError":  se_scaled.tolist(),
                "MCCounts":             mc.tolist(),
                "MCCountsError":        np.sqrt(mc).tolist(),
            })

    if not cutflow_rows:
        rprint(f"[yellow][WARNING][/yellow] '{_name}': no data produced. Skipping.")
        continue

    _all_rows.extend(cutflow_rows)

    # ── Save pkl (one per name) ────────────────────────────────────────────────

    _pkl_filename = f"{args.energy}_{args.analysis}_Cutflow"
    save_df(
        pd.DataFrame(cutflow_rows),
        data_path,
        config=args.config,
        name=_name,
        subfolder=f"{args.folder.lower()}/{args.analysis.lower()}",
        filename=_pkl_filename,
        rm=args.rewrite,
        debug=args.plot,
    )

    # ── Plot ───────────────────────────────────────────────────────────────────

    if not (args.plot and _component_order):
        continue

    n_panels = len(_component_order)
    n_cols   = min(4, n_panels)
    n_rows   = math.ceil(n_panels / n_cols)

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=tuple(f"{comp}" for _, comp in _component_order),
        shared_yaxes=False,
    )

    _legend_added: set[str] = set()

    for panel_idx, (__, component) in enumerate(_component_order):
        p_row = panel_idx // n_cols + 1
        p_col = panel_idx %  n_cols + 1
        stage_pairs = _component_hists[(__, component)]  # list of (stage_def, (h, err, mc))

        for stage, (h, _h_err, _mc) in stage_pairs:
            show_legend = stage["label"] not in _legend_added
            if show_legend:
                _legend_added.add(stage["label"])
            total = float(np.sum(h))
            fig.add_trace(
                go.Scatter(
                    x=_centers, y=h, mode="lines",
                    name=stage["label"],
                    legendgroup=stage["label"],
                    line=dict(color=stage["color"], dash=stage["dash"], width=stage["width"]),
                    showlegend=show_legend,
                    hovertemplate=(
                        f"{stage['label']}: %{{y:.3g}} "
                        f"(total={total:.3g})<extra>{_name}/{component}</extra>"
                    ),
                ),
                row=p_row, col=p_col,
            )

        fig.update_xaxes(title_text="Reco Energy (MeV)", row=p_row, col=p_col)
        fig.update_yaxes(type="log", title_text=f"Events / MeV / {_exposure:.0f} yr", row=p_row, col=p_col)

    fig = format_coustom_plotly(
        fig,
        title=f"Cut-flow Spectra — {args.config} {_name} {args.folder} {args.analysis} {args.energy}  [{_cut_label}]",
        ranges=(None, [-3, 6]),  # matches significance_plot.py
    )

    save_figure(
        fig,
        f"{save_path}/{args.folder.lower()}/{args.analysis.lower()}",
        config=args.config, name=_name, subfolder=None,
        filename=f"{args.energy}_{args.analysis}_Cutflow_{_cut_label}",
        rm=args.rewrite, debug=args.plot,
    )

# ── Combined pkl (all names, no Name column) ──────────────────────────────────

if _all_rows:
    _combined = pd.DataFrame(_all_rows).drop(columns=["Name"], errors="ignore")
    save_df(
        _combined,
        data_path,
        config=args.config,
        name=None,
        subfolder=f"{args.folder.lower()}/{args.analysis.lower()}",
        filename=f"{args.energy}_{args.analysis}_Cutflow",
        rm=args.rewrite,
        debug=args.plot,
    )
