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
parser.add_argument("--config", type=str, default="hd_1x2x6_centralAPA")
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
parser.add_argument("--analysis", type=str, default="Sensitivity")
parser.add_argument("--nhits",  type=int, default=None)
parser.add_argument("--ophits", type=int, default=None)
parser.add_argument("--adjcls", type=int, default=None)
parser.add_argument("--exposure", type=float, default=None, help="Livetime in years for scaling. Defaults to EVALUATION_EXPOSURE_YEARS from params (20 if unset).")
parser.add_argument("--mc_filter_threshold", type=int, default=2, help="Min unweighted MC events per bin in the final-cut stage; bins below are zeroed (matches 03_analysis.py default).")
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug",   action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--plot",    action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()

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
    # Try canonical filename first (no study label suffix)
    exact = f"{_base}/{args.config}_{_primary}_highest_{args.analysis}.pkl"
    if not os.path.exists(exact):
        raise SystemExit(
            f"[ERROR] Best-cuts pkl not found: {exact}\n"
            "Run the full pipeline (05_best_sigmas.py) for this config/analysis/folder first."
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
                    ("Solar Night", "AnalysisWeightsSolarNight")],
    "HEP":         [("8B",          "AnalysisWeights8B"),
                    ("hep",         "AnalysisWeightshep")],
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

    reco     = np.asarray(pickle.load(open(required[0], "rb")), dtype=float)
    weights  = np.asarray(pickle.load(open(required[1], "rb")), dtype=float)
    geo_fid  = np.asarray(pickle.load(open(required[2], "rb")), dtype=bool)
    nhits    = np.asarray(pickle.load(open(required[3], "rb")), dtype=int)
    ophits   = np.asarray(pickle.load(open(required[4], "rb")), dtype=int)
    adjcl    = np.asarray(pickle.load(open(required[5], "rb")), dtype=int)
    op_plane = np.asarray(pickle.load(open(required[6], "rb")), dtype=int)
    op_pe    = np.asarray(pickle.load(open(required[7], "rb")), dtype=float)

    flash_mask = (op_plane == _op_plane_cut) & (op_pe > 0)
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

    return [
        _histogram(reco, weights, pre_mask),
        _histogram(reco, weights, flash_mask),
        _histogram(reco, weights, fid),
        _histogram(reco, weights, nhits_mask),
        _histogram(reco, weights, op_mask),
        h_full,
    ]

# ── Smoothing config (1-D, analysis-specific — matches significance_plot.py) ──

_smoothing_cfg = get_smoothing_config(
    str(root), analysis_name=args.analysis.upper(), dimensions="1d", stage="significance"
)


def _smooth(h: np.ndarray, component: str) -> np.ndarray:
    cfg = get_component_smoothing_config(_smoothing_cfg, component)
    return smooth_histogram_with_config(h, cfg)

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

        key = (_name, _component_label)
        _component_order.append(key)
        _component_hists[key] = stage_hists

        _meta = {
            "Config": args.config, "Name": _name, "Folder": args.folder,
            "Component": _component_label, "NHits": _nhits, "OpHits": _ophits, "AdjCl": _adjcls,
            "Exposure": _exposure, "ExposureUnit": "year",
            "EnergyUnit": "MeV", "CountsUnit": f"events / MeV / {_exposure:.0f} yr",
        }
        for stage, h in zip(_FID_STAGES + _CUT_STAGES, stage_hists):
            h_scaled  = h * _scale          # events/(kT·yr) × kT·yr = events / bin (1 MeV bin → events/MeV)
            sh_scaled = _smooth(h, _component_label) * _scale
            cutflow_rows.append({
                **_meta, "Stage": stage["label"],
                "Energy":         _centers.tolist(),
                "Counts":         h_scaled.tolist(),
                "SmoothedCounts": sh_scaled.tolist(),
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
    _all_stages = _FID_STAGES + _CUT_STAGES

    for panel_idx, (__, component) in enumerate(_component_order):
        p_row = panel_idx // n_cols + 1
        p_col = panel_idx %  n_cols + 1
        hists = _component_hists[(__, component)]

        for stage, h in zip(_all_stages[:len(hists)], hists):
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
