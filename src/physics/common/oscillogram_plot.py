"""
oscillogram_plot.py — Oscillation probability visualisation
===========================================================
Standalone plot script producing P(νe→νe) maps at the best-fit parameter
point (SOLAR_DM2, SIN13, SIN12 from physics.json). Called from
run_sensitivity.py outside the --no-computation gate, so it runs even
when analyses are skipped.

Outputs — figures
-----------------
  {images}/analysis/{analysis}/oscillogram/
    Oscillogram_{energy}.html                  — raw P(νe→νe) heatmap (nadir × energy)
    Oscillogram_NadirWeighted_{energy}.html    — P_ee × nadir PDF weight heatmap
    Oscillogram_NadirProjection_{energy}.html  — day/DUNE-weighted mean/night P_ee vs E
    Signal1D_{energy}_FidOnly.html             — 1D osc-weighted signal (--signal_1d)

Outputs — DataFrames  (compatible with LOWE_RECONSTRUCTION_PUBLICATION scripts)
-----------------
  {data}/analysis/{analysis}/
    {config}_{name}_{folder}_Oscillogram.csv / .pkl
      Long-format: one row per (NadirAngle, Energy) point.
      Columns: Config, Name, Geometry, Analysis, Folder,
               Energy (array n_energy), NadirAngle (array n_nadir),
               Pee (array n_nadir×n_energy, raw P_ee),
               PeeWeighted (array n_nadir×n_energy, P_ee × nadir_pdf),
               Dm2, Sin12, Sin13

For --signal_1d, reads pre-saved Ref pkls from output/analysis/.
Run src/physics/signal/03_analysis.py --export_fiducial first.

Run
---
  python3 src/physics/common/oscillogram_plot.py \\
      --analysis Sensitivity --config hd_1x2x6_centralAPA \\
      --folder Truncated --energy SolarEnergy \\
      --oscillation_backend file [--signal_1d]
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *
from lib.oscillation import get_oscillation_map
from lib.oscillation_backends import get_nadir_pdf_file, get_nadir_pdf_nufast

# ── CLI ─────────────────────────────────────────────────────────────────────
_analysis_info = load_analysis_info(str(root))

parser = argparse.ArgumentParser(
    description="Oscillogram and nadir-projected P_ee at best-fit oscillation parameters"
)
parser.add_argument("--analysis", type=str, choices=["DayNight", "HEP", "Sensitivity"], default="Sensitivity")
parser.add_argument("--config",   type=str, default="hd_1x2x6_centralAPA")
parser.add_argument("--name",     type=str, default="marley")
parser.add_argument("--folder",   type=str, choices=["Reduced", "Truncated", "Nominal"], default="Truncated")
parser.add_argument("--energy",   type=str, default="SolarEnergy",
    choices=["SignalParticleK", "ClusterEnergy", "TotalEnergy", "SelectedEnergy", "SolarEnergy"])
parser.add_argument("--exposure", type=float, default=30)
parser.add_argument(
    "--oscillation_backend", type=str, choices=["file", "prob3", "nufast"], default="file",
    help="'file': load pre-computed pkl; 'prob3'/'nufast': compute on-the-fly.",
)
parser.add_argument(
    "--signal_1d", action=argparse.BooleanOptionalAction, default=False,
    help=(
        "Overlay 1D oscillation-weighted signal spectrum. "
        "Reads AnalysisEnergy/Data/Weights/FiducializationMask pkls from "
        "output/analysis/. Run 03_analysis.py --export_fiducial first."
    ),
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug",   action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--plot",    action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()

config = args.config
name   = args.name

# ── Paths ─────────────────────────────────────────────────────────────────
_analysis_dir = {
    "DayNight":    "day-night",
    "HEP":         "hep",
    "Sensitivity": "sensitivity",
}.get(args.analysis, args.analysis.lower())

save_path   = f"{root}/images/analysis/{_analysis_dir}/oscillogram"
data_path   = f"{root}/data/analysis/{_analysis_dir}"
export_path = f"{root}/output/analysis"

for _p in [save_path, data_path]:
    os.makedirs(_p, exist_ok=True)

# ── Config info ────────────────────────────────────────────────────────────
info          = json.loads(open(f"{root}/config/{config}/{config}_config.json").read())
geometry      = info.get("GEOMETRY", config.split("_")[0].upper())
detector_mass = get_full_detector_mass(config, info)

# ── Validate physics.json keys ─────────────────────────────────────────────
for _key in ("SOLAR_DM2", "SIN13", "SIN12"):
    if _key not in _analysis_info:
        raise SystemExit(f"[oscillogram] '{_key}' missing from physics.json.")

dm2   = float(_analysis_info["SOLAR_DM2"])
sin13 = float(_analysis_info["SIN13"])
sin12 = float(_analysis_info["SIN12"])

rprint(
    f"[bold]oscillogram_plot[/bold] — analysis={args.analysis} "
    f"backend={args.oscillation_backend} "
    f"Δm²={dm2:.3e} sin²θ₁₂={sin12:.4f} sin²θ₁₃={sin13:.5f}"
)

# ── Load oscillation at best-fit point ────────────────────────────────────
if args.oscillation_backend == "file":
    _pkl = (
        f"{info['PATH']}/data/OSCILLATION/pkl/rebin/"
        f"osc_probability_dm2_{dm2:.3e}_sin13_{sin13:.3e}_sin12_{sin12:.3e}.pkl"
    )
    if not os.path.exists(_pkl):
        raise SystemExit(
            f"[oscillogram] Oscillation pkl not found: {_pkl}\n"
            "Run 01_process_oscillation.py first."
        )
    oscillation_df = pd.read_pickle(_pkl)
else:
    _osc_map = get_oscillation_map(
        backend=args.oscillation_backend,
        dm2=[dm2], sin13=[sin13], sin12=[sin12],
        output="df", debug=args.debug,
    )
    oscillation_df = next(iter(_osc_map.values()))

if args.debug:
    rprint(
        f"[cyan][INFO][/cyan] Oscillation df: {oscillation_df.shape} "
        f"(nadir bins × energy bins)"
    )

# ── Recover raw P_ee (undo nadir-PDF weighting) ───────────────────────────
# oscillation_df stores P_ee × nadir_PDF_weight (from combine_day_night).
# Divide by nadir_pdf to recover actual P(νe→νe) ∈ [0, 1].
_pee_energy = np.array([float(c) for c in oscillation_df.columns])
_nadir_vals = np.array([float(r) for r in oscillation_df.index])

try:
    _nadir_pdf = get_nadir_pdf_file(nadir_centers=_nadir_vals)
except Exception:
    _latitude = _analysis_info.get("DUNE_LATITUDE_DEG", 44.35)
    _nadir_pdf = get_nadir_pdf_nufast(_nadir_vals, _latitude)

_safe_pdf = np.where(_nadir_pdf > 0, _nadir_pdf, np.nan)
_raw_pee  = oscillation_df.values / _safe_pdf[:, np.newaxis]
raw_pee_df = pd.DataFrame(
    np.nan_to_num(_raw_pee, nan=0.0),
    index=oscillation_df.index,
    columns=oscillation_df.columns,
)

# ── Pre-compute nadir-projected P_ee ──────────────────────────────────────
# Mean: DUNE nadir-PDF-weighted average ⟨P_ee⟩ = sum(P_ee × nadir_pdf)
# Day/Night: arithmetic mean over day/night nadir bins from raw P_ee
_pee_mean_dune = oscillation_df.sum(axis=0).values  # = Σ_η P_ee(η,E) × pdf(η)

_day_mask   = _nadir_vals < 0
_night_mask = _nadir_vals >= 0
_pee_day    = (
    raw_pee_df.values[_day_mask].mean(axis=0)
    if np.any(_day_mask) else raw_pee_df.values.mean(axis=0)
)
_pee_night  = (
    raw_pee_df.values[_night_mask].mean(axis=0)
    if np.any(_night_mask) else raw_pee_df.values.mean(axis=0)
)

# ── Build export DataFrame ─────────────────────────────────────────────────
# One row per oscillogram: array-valued Energy, NadirAngle, Pee, PeeWeighted.
# Pee        — raw P(νe→νe), shape (n_nadir, n_energy)
# PeeWeighted — P_ee × nadir_pdf(η), shape (n_nadir, n_energy)
_osc_df = pd.DataFrame([{
    "Config":       config,
    "Name":         name,
    "Geometry":     geometry,
    "Analysis":     args.analysis,
    "Folder":       args.folder,
    "Energy":       _pee_energy,
    "NadirAngle":   _nadir_vals,
    "Pee":          raw_pee_df.values,
    "PeeWeighted":  oscillation_df.values,
    "Dm2":          dm2,
    "Sin12":        sin12,
    "Sin13":        sin13,
}])
save_df(
    _osc_df, data_path,
    config=config, name=name,
    subfolder=args.folder.lower(),
    filename=f"Oscillogram",
    rm=args.rewrite, debug=args.plot,
)
rprint(
    f"[green][OK][/green] Oscillogram DataFrame saved "
    f"({oscillation_df.shape[0]} nadir bins × {oscillation_df.shape[1]} energy bins)"
)

# ── Figures ────────────────────────────────────────────────────────────────
if not args.plot:
    rprint("[bold green]oscillogram_plot complete (no-plot mode).[/bold green]")
    raise SystemExit(0)

# Figure 1: Raw P(νe→νe) heatmap — actual survival probability before nadir weighting
_fig_osc = make_subplots(rows=1, cols=1)
_fig_osc.add_trace(go.Heatmap(
    z=raw_pee_df.values,
    x=[float(c) for c in raw_pee_df.columns],
    y=[float(r) for r in raw_pee_df.index],
    colorscale="RdYlBu_r",
    colorbar=dict(title=dict(text="P(νe→νe)", side="right")),
    zauto=True,
))
_fig_osc = format_coustom_plotly(
    _fig_osc,
    title=(
        f"P(νe→νe) — Δm²₂₁={dm2:.2e} eV², "
        f"sin²θ₁₂={sin12:.3f}, sin²θ₁₃={sin13:.4f}"
    ),
    add_watermark=True,
)
_fig_osc.update_xaxes(title_text="True Neutrino Energy (MeV)")
_fig_osc.update_yaxes(title_text="cos(η) Nadir Angle")
save_figure(
    _fig_osc, save_path, config=config, name=name,
    subfolder=args.folder.lower(),
    filename=f"Oscillogram_{args.energy}",
    rm=args.rewrite, debug=args.plot,
)

# Figure 1b: Nadir-PDF-weighted oscillogram — P_ee × nadir_pdf(η), showing DUNE exposure weight
_fig_osc_w = make_subplots(rows=1, cols=1)
_fig_osc_w.add_trace(go.Heatmap(
    z=oscillation_df.values,
    x=[float(c) for c in oscillation_df.columns],
    y=[float(r) for r in oscillation_df.index],
    colorscale="RdYlBu_r",
    colorbar=dict(title=dict(text="P(νe→νe) × w(η)", side="right")),
    zauto=True,
))
_fig_osc_w = format_coustom_plotly(
    _fig_osc_w,
    title=(
        f"P(νe→νe) × nadir weight — Δm²₂₁={dm2:.2e} eV², "
        f"sin²θ₁₂={sin12:.3f}, sin²θ₁₃={sin13:.4f}"
    ),
    add_watermark=True,
)
_fig_osc_w.update_xaxes(title_text="True Neutrino Energy (MeV)")
_fig_osc_w.update_yaxes(title_text="cos(η) Nadir Angle")
save_figure(
    _fig_osc_w, save_path, config=config, name=name,
    subfolder=args.folder.lower(),
    filename=f"Oscillogram_NadirWeighted_{args.energy}",
    rm=args.rewrite, debug=args.plot,
)

# Figure 2: Nadir-projected P_ee — day / night (raw arithmetic mean) + DUNE-weighted mean
_fig_proj = make_subplots(rows=1, cols=1)
_fig_proj.add_trace(go.Scatter(
    x=_pee_energy, y=_pee_mean_dune, mode="lines", name="Mean (DUNE weighted)",
    line=dict(color="black", width=2.5, dash="solid"),
))
_fig_proj.add_trace(go.Scatter(
    x=_pee_energy, y=_pee_day, mode="lines", name="Day (cos η < 0)",
    line=dict(color="#e8421a", width=1.5, dash="dash"),
))
_fig_proj.add_trace(go.Scatter(
    x=_pee_energy, y=_pee_night, mode="lines", name="Night (cos η ≥ 0)",
    line=dict(color="#1f6e8a", width=1.5, dash="dot"),
))
_fig_proj = format_coustom_plotly(
    _fig_proj,
    title=f"⟨P(νe→νe)⟩ Nadir Projection — {args.analysis}",
    add_watermark=True,
)
_fig_proj.update_xaxes(title_text="True Neutrino Energy (MeV)")
_fig_proj.update_yaxes(title_text="⟨P(νe→νe)⟩")
save_figure(
    _fig_proj, save_path, config=config, name=name,
    subfolder=args.folder.lower(),
    filename=f"Oscillogram_NadirProjection_{args.energy}",
    rm=args.rewrite, debug=args.plot,
)

# Figure 3 + DataFrame: 1D oscillation-weighted signal (optional)
if not args.signal_1d:
    rprint("[bold green]oscillogram_plot complete.[/bold green]")
    raise SystemExit(0)

_ref_dir = os.path.join(export_path, config, name, args.folder.lower())


def _load_ref(filename: str) -> np.ndarray:
    path = os.path.join(_ref_dir, f"{config}_{name}_{filename}.pkl")
    if not os.path.exists(path):
        raise SystemExit(
            f"[oscillogram] Required Ref pkl not found: {path}\n"
            "Run src/physics/signal/03_analysis.py --export_fiducial first."
        )
    return np.asarray(pd.read_pickle(path))


_true_energy  = _load_ref(f"AnalysisEnergy_{args.energy}_Ref").astype(float)
_reco_energy  = _load_ref(f"AnalysisData_{args.energy}_Ref").astype(float)
_base_weights = _load_ref(f"AnalysisWeights_{args.energy}_Ref").astype(float)
_fid_mask     = _load_ref(
    f"FiducializationMask_{args.energy}_{args.analysis.upper()}"
).astype(bool)

_true_sel = _true_energy[_fid_mask]
_reco_sel  = _reco_energy[_fid_mask]
_w_sel     = _base_weights[_fid_mask]

# Mean P_ee per event via interpolation from mean nadir projection
_pee_per_event = np.interp(
    _true_sel, _pee_energy, _pee_mean_dune,
    left=float(_pee_mean_dune[0]), right=float(_pee_mean_dune[-1]),
)
_final_weights = _w_sel * _pee_per_event * args.exposure * detector_mass

_sig_hist, _ = np.histogram(_reco_sel, bins=true_energy_edges, weights=_final_weights)

# Export signal 1D DataFrame
_sig_records = [
    {
        "Config":          config,
        "Name":            name,
        "Geometry":        geometry,
        "Analysis":        args.analysis,
        "Folder":          args.folder,
        "EnergyVariable":  args.energy,
        "RecoEnergy":      float(true_energy_centers[_k]),
        "Signal":          float(_sig_hist[_k]),
        "NadirSlice":      "Mean",
        "Dm2":             dm2,
        "Sin12":           sin12,
        "Sin13":           sin13,
        "FiducialCount":   int(np.sum(_fid_mask)),
    }
    for _k in range(len(true_energy_centers))
]
_sig_df = pd.DataFrame(_sig_records)
save_df(
    _sig_df, data_path,
    config=config, name=name,
    subfolder=args.folder.lower(),
    filename=f"Signal1D_{args.energy}",
    rm=args.rewrite, debug=args.debug,
)

_fig_1d = make_subplots(rows=1, cols=1)
_fig_1d.add_trace(go.Scatter(
    x=true_energy_centers,
    y=_sig_hist,
    mode="lines",
    fill="tozeroy",
    line_shape="hvh",
    name="Solar ν (mean osc.)",
    line=dict(color="black", width=2),
))
_fig_1d = format_coustom_plotly(
    _fig_1d,
    title=(
        f"1D Oscillation-Weighted Signal — {config} "
        f"({args.exposure} kt·yr, {args.analysis})"
    ),
    add_watermark=True,
)
_fig_1d.update_xaxes(title_text=f"Reconstructed Energy — {args.energy} (MeV)")
_fig_1d.update_yaxes(title_text="Events / (kt·yr · MeV)")
save_figure(
    _fig_1d, save_path, config=config, name=name,
    subfolder=args.folder.lower(),
    filename=f"Signal1D_{args.energy}_FidOnly",
    rm=args.rewrite, debug=args.plot,
)

rprint(
    f"[bold green]oscillogram_plot complete.[/bold green] "
    f"Signal 1D: {np.sum(_sig_hist):.1f} events "
    f"({np.sum(_fid_mask)} fiducialized / {len(_fid_mask)} total)"
)
