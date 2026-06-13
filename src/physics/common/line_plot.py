"""
line_plot.py — Generic line-plot export tool
=============================================
Produces publication-ready line plots and matching DataFrame pkl files for
physics quantities that don't belong to a specific analysis stage.

Each plot type is self-contained: it computes its data, saves a DataFrame,
and saves one or more HTML figures.

Supported plot types
--------------------
  nadir_pdf             DUNE SURF nadir angle weight distribution w(cos η).
                        Overlays ROOT-file interpolation and NuFast analytic.
  kinematic_threshold   CC νe + ⁴⁰Ar kinematic threshold vs neutrino energy.
                        Requires --config and --name to locate input DataFrame.

Outputs — figures
-----------------
  {images}/common/line_plots/
    NadirPDF.png               — w(cos η) from ROOT file vs NuFast analytic
  {images}/marley/stacked/
    {config}_{name}_KinematicThreshold.png

Outputs — DataFrames
--------------------
  {data}/common/
    NadirPDF.pkl
      Columns: NadirAngle (array), PDF_File (array), PDF_NuFast (array),
               LatitudeDeg (scalar), NadirBins (scalar)
  {data}/marley/stacked/{config}/{name}/
    {config}_{name}_KinematicThreshold.pkl
      Columns: Config, Name, NeutrinoEnergy (array), TotalFraction (array),
               KinematicThreshold (array)

Run
---
  # All config-independent plots:
  python3 src/physics/common/line_plot.py

  # Kinematic threshold for a specific config:
  python3 src/physics/common/line_plot.py \\
      --plot_type kinematic_threshold --config hd_1x2x6 --name marley_official

  # Skip figures, export DataFrames only:
  python3 src/physics/common/line_plot.py --no-plot
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *
from lib.oscillation_backends import get_nadir_pdf_file, get_nadir_pdf_nufast

# ── CLI ──────────────────────────────────────────────────────────────────────
_analysis_info = load_analysis_info(str(root))

_ALL_PLOT_TYPES = ["nadir_pdf", "kinematic_threshold"]

parser = argparse.ArgumentParser(
    description="Export line plots and DataFrame pkl files for common physics quantities"
)
parser.add_argument(
    "--plot_type",
    nargs="+",
    type=str,
    choices=_ALL_PLOT_TYPES,
    default=_ALL_PLOT_TYPES,
    help="Which line plots to produce. Default: all.",
)
parser.add_argument(
    "--config",
    type=str,
    default=None,
    help="Detector config short name (required for kinematic_threshold).",
)
parser.add_argument(
    "--name",
    type=str,
    default=None,
    help="Sample name (required for kinematic_threshold).",
)
parser.add_argument(
    "--rewrite",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Overwrite existing output files. Pass --no-rewrite to skip existing.",
)
parser.add_argument(
    "--plot",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Produce HTML figures. Pass --no-plot to export DataFrames only.",
)
parser.add_argument(
    "--debug",
    action=argparse.BooleanOptionalAction,
    default=False,
)

args = parser.parse_args()

# ── Paths ─────────────────────────────────────────────────────────────────────
_save_path = f"{root}/images/common/line_plots"
_data_path = f"{root}/data/common"

for _p in [_save_path, _data_path]:
    os.makedirs(_p, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _save(df, filename):
    save_df(
        df, _data_path,
        config=None, name=None, subfolder=None,
        filename=filename,
        rm=args.rewrite, debug=args.plot,
    )

def _fig(fig, filename):
    if not args.plot:
        return
    save_figure(
        fig, _save_path,
        config=None, name=None, subfolder=None,
        filename=filename,
        rm=args.rewrite, debug=args.plot,
    )


# ══════════════════════════════════════════════════════════════════════════════
# nadir_pdf — DUNE SURF nadir angle weight distribution
# ══════════════════════════════════════════════════════════════════════════════

def run_nadir_pdf():
    """
    Nadir angle PDF w(cos η) for DUNE's SURF far detector (latitude 44.35°N).

    Two curves:
      File    — interpolated from nadir.root (ROOT histogram, 2000 bins)
      NuFast  — analytic Solar_Weight from NuFast-Earth at SURF latitude

    The PDF is normalised so Σ w(cos η_i) = 1 over the analysis nadir bins.
    """
    nadir_bins   = _analysis_info.get("NADIR_BINS", 40)
    latitude_deg = _analysis_info.get("DUNE_LATITUDE_DEG", 44.35)

    nadir_edges   = np.linspace(-1.0, 1.0, nadir_bins + 1)
    nadir_centers = 0.5 * (nadir_edges[1:] + nadir_edges[:-1])

    rprint(
        f"[bold]nadir_pdf[/bold] — {nadir_bins} bins, latitude={latitude_deg}°N"
    )

    try:
        pdf_file = get_nadir_pdf_file(nadir_centers=nadir_centers)
        rprint(
            f"  [green][OK][/green] ROOT file PDF loaded "
            f"(sum={pdf_file.sum():.6f})"
        )
    except Exception as _exc:
        rprint(f"  [yellow][WARNING][/yellow] ROOT file PDF unavailable: {_exc}")
        pdf_file = np.zeros_like(nadir_centers)

    pdf_nufast = get_nadir_pdf_nufast(nadir_centers, latitude_deg)
    rprint(
        f"  [green][OK][/green] NuFast PDF computed "
        f"(sum={pdf_nufast.sum():.6f})"
    )

    # ── Export DataFrame ──────────────────────────────────────────────────────
    df = pd.DataFrame([{
        "NadirAngle":   nadir_centers,
        "PDF_File":     pdf_file,
        "PDF_NuFast":   pdf_nufast,
        "LatitudeDeg":  latitude_deg,
        "NadirBins":    nadir_bins,
    }])
    _save(df, "NadirPDF")

    # ── Figure ────────────────────────────────────────────────────────────────
    if not args.plot:
        rprint("  [cyan][SKIP][/cyan] figure (--no-plot)")
        return

    fig = make_subplots(rows=1, cols=1)

    if np.any(pdf_file > 0):
        fig.add_trace(go.Scatter(
            x=nadir_centers, y=pdf_file,
            mode="lines",
            name="ROOT file",
            line=dict(color="#1f6e8a", width=2.5, dash="solid"),
        ))

    fig.add_trace(go.Scatter(
        x=nadir_centers, y=pdf_nufast,
        mode="lines",
        name="NuFast analytic",
        line=dict(color="#e8421a", width=2.0, dash="dash"),
    ))

    fig = format_coustom_plotly(
        fig,
        title=f"DUNE SURF Nadir Angle Distribution — latitude {latitude_deg}°N",
        add_watermark=True,
    )
    fig.update_xaxes(title_text="cos(η) Nadir Angle")
    fig.update_yaxes(title_text="Normalised weight  w(cos η)")

    _fig(fig, "NadirPDF")
    rprint("  [green][OK][/green] NadirPDF figure saved")


# ══════════════════════════════════════════════════════════════════════════════
# kinematic_threshold — CC νe + 40Ar interaction threshold vs neutrino energy
# ══════════════════════════════════════════════════════════════════════════════

def run_kinematic_threshold():
    """
    Kinematic threshold for CC νe + 40Ar → e⁻ + 40K* interaction.

    Reads Neutrino_CC_Fraction DataFrame saved by TruthMarleyStacked.ipynb
    (or run_signal.py). Computes:

      T_threshold(E_nu) = (1 - Σ_i f_i(E_nu)) × E_nu

    where f_i(E_nu) is the mean fractional kinetic energy carried by daughter
    particle type i at neutrino energy E_nu. The remainder is energy deposited
    in nuclear excitation and recoil — the effective kinematic threshold.

    Requires --config and --name to locate the input DataFrame.
    """
    config = args.config
    name   = args.name
    if config is None or name is None:
        rprint(
            "[yellow][WARNING][/yellow] kinematic_threshold requires --config and --name. Skipping."
        )
        return

    _cc_data_path = f"{root}/data/marley/stacked"
    _pkl = f"{_cc_data_path}/{config}/{name}/{config}_{name}_Neutrino_CC_Fraction.pkl"

    if not os.path.exists(_pkl):
        rprint(
            f"[yellow][WARNING][/yellow] Neutrino_CC_Fraction pkl not found: {_pkl}\n"
            "  Run src/pipelines/run_signal.py or TruthMarleyStacked.ipynb first."
        )
        return

    frac_df = pd.read_pickle(_pkl)
    rprint(
        f"  [green][OK][/green] Loaded Neutrino_CC_Fraction "
        f"({len(frac_df)} rows, config={config}, name={name})"
    )

    # Sum all particle fractions at each neutrino energy bin
    total_frac = (
        frac_df.groupby("SignalParticleK")["TSignalSumK"].sum()
    )
    # Mask bins where total fraction >= 1 (unphysical/low-stats) or energy == 0
    valid = (total_frac < 1.0) & (total_frac.index > 0)
    total_frac = total_frac[valid]

    energy         = np.array(total_frac.index, dtype=float)
    threshold      = (1.0 - total_frac.values) * energy

    # ── Export DataFrame ──────────────────────────────────────────────────────
    df = pd.DataFrame([{
        "Config":          config,
        "Name":            name,
        "NeutrinoEnergy":  energy,
        "TotalFraction":   total_frac.values,
        "KinematicThreshold": threshold,
    }])
    save_df(
        df, f"{_cc_data_path}/{config}/{name}",
        config=None, name=None, subfolder=None,
        filename=f"{config}_{name}_KinematicThreshold",
        rm=args.rewrite, debug=args.plot,
    )

    # ── Figure ────────────────────────────────────────────────────────────────
    if not args.plot:
        rprint("  [cyan][SKIP][/cyan] figure (--no-plot)")
        return

    _cc_save_path = f"{root}/images/marley/stacked"
    os.makedirs(_cc_save_path, exist_ok=True)

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(
        x=energy, y=threshold,
        mode="lines",
        line_shape="hvh",
        name="Kinematic threshold",
        line=dict(color="#1f6e8a", width=2.5),
    ))
    fig = format_coustom_plotly(
        fig,
        title=f"Kinematic Threshold — CC νe + ⁴⁰Ar ({config})",
        add_watermark=True,
    )
    fig.update_xaxes(title_text="True Neutrino Energy (MeV)")
    fig.update_yaxes(title_text="Threshold energy (MeV)")
    save_figure(
        fig, _cc_save_path,
        config=None, name=None, subfolder=None,
        filename=f"{config}_{name}_KinematicThreshold",
        rm=args.rewrite, debug=args.plot,
    )
    rprint(f"  [green][OK][/green] KinematicThreshold figure saved")


# ══════════════════════════════════════════════════════════════════════════════
# Dispatch
# ══════════════════════════════════════════════════════════════════════════════

_REGISTRY = {
    "nadir_pdf":            run_nadir_pdf,
    "kinematic_threshold":  run_kinematic_threshold,
}

for _plot_type in args.plot_type:
    rprint(f"\n[bold cyan]── {_plot_type} ──[/bold cyan]")
    _REGISTRY[_plot_type]()

rprint("\n[bold green]line_plot complete.[/bold green]")
