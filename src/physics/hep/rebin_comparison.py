import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *

analysis_info = load_analysis_info(str(root))

save_path = f"{root}/output/images/analysis/hep"
data_path = f"{root}/output/data/analysis/hep"
for this_path in [save_path, data_path]:
    if not os.path.exists(this_path):
        os.makedirs(this_path)


def get_selection_cuts(config: str, name: str, energy: str, args: argparse.Namespace):
    sigma_path = (
        f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/HEP/{args.folder.lower()}/"
        f"{config}/{name}/{config}_{name}_{args.pkl_label}_HEP.pkl"
    )
    if not os.path.exists(sigma_path):
        return None

    sigma = pd.read_pickle(sigma_path)
    try:
        ref_plot = sigma[(config, name, energy)]
    except KeyError:
        return None

    nhits_value = args.nhits if args.nhits is not None else int(ref_plot["NHits"])
    ophits_value = args.ophits if args.ophits is not None else int(ref_plot["OpHits"])
    adjcl_value = args.adjcls if args.adjcls is not None else int(ref_plot["AdjCl"])
    return int(nhits_value), int(ophits_value), int(adjcl_value)


def as_curve(value):
    return np.asarray(value, dtype=float)


parser = argparse.ArgumentParser(
    description=(
        "Compare ProfileLikelihood significance curves before and after PAVA isotonic correction. "
        "Asimov/Gaussian adaptive-rebin comparisons are handled by common/exposure_plot.py --mode rebin."
    ),
    formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=32, width=120),
)
default_reference = analysis_info.get("BEST_SIGMA_SIGNIFICANCE_REFERENCE", {}).get("HEP", "Asimov")
if default_reference not in ["Gaussian", "Asimov", "ProfileLikelihood"]:
    default_reference = "Asimov"

parser.add_argument(
    "--analysis",
    type=str,
    default="HEP",
    help="Analysis type (fixed to HEP; reserved for future use).",
)
parser.add_argument(
    "--reference",
    type=str,
    choices=["Gaussian", "Asimov", "ProfileLikelihood"],
    default=default_reference,
    help="Significance reference used when selecting best cuts.",
)
parser.add_argument(
    "--config",
    nargs="+",
    type=str,
    default=["hd_1x2x6_centralAPA"],
    help="Detector configuration(s) to process.",
)
parser.add_argument(
    "--name",
    nargs="+",
    type=str,
    default=["marley"],
    help="Sample name(s) to process.",
)
parser.add_argument(
    "--folder",
    type=str,
    default="Truncated",
    choices=["Reduced", "Truncated", "Nominal"],
    help="Result folder containing the HEP pkl files.",
)
parser.add_argument(
    "--exposure",
    type=float,
    default=30,
    help="Reference exposure in years.",
)
parser.add_argument(
    "--energy",
    nargs="+",
    type=str,
    choices=["SignalParticleK", "ClusterEnergy", "TotalEnergy", "SelectedEnergy", "SolarEnergy"],
    default=["SignalParticleK", "ClusterEnergy", "TotalEnergy", "SelectedEnergy", "SolarEnergy"],
    help="Energy label(s) to process.",
)
parser.add_argument(
    "--signal_uncertainty",
    type=float,
    default=None,
    help="Fractional signal uncertainty override.",
)
parser.add_argument(
    "--background_uncertainty",
    type=float,
    default=None,
    help="Fractional background uncertainty override.",
)
parser.add_argument(
    "--nhits",
    type=int,
    default=None,
    help="NHits cut override (default: read from best-cut pkl).",
)
parser.add_argument(
    "--ophits",
    type=int,
    default=None,
    help="OpHits cut override (default: read from best-cut pkl).",
)
parser.add_argument(
    "--adjcls",
    type=int,
    default=None,
    help="AdjCl cut override (default: read from best-cut pkl).",
)
parser.add_argument(
    "--threshold",
    type=float,
    default=get_analysis_threshold(str(root), "HEP", stage="SIGNIFICANCE", fallback=0.0),
    help="Significance threshold appended to figure filenames.",
)
parser.add_argument(
    "--zoom",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Auto-scale significance y-axis to curve peak instead of fixing at [0, 6].",
)
parser.add_argument(
    "--rewrite",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Overwrite existing output files. Pass --no-rewrite to skip files that already exist.",
)
parser.add_argument(
    "--debug",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Print save paths for each output file.",
)
parser.add_argument(
    "--plot",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Save PNG figures. Pass --no-plot to suppress figure output.",
)
parser.add_argument(
    "--pkl_label",
    type=str,
    default="highest",
    help="Label of the best-cut pkl to read (e.g. 'highest', 'highest_spiked').",
)
args = parser.parse_args()

_STYLE_PRE  = dict(color=compare[1], dash="dash",  width=2)
_STYLE_POST = dict(color="black",    dash="solid", width=3)

for config, name in product(args.config, args.name):
    comparison_rows = []

    for energy in args.energy:
        selection = get_selection_cuts(config, name, energy, args)
        if selection is None:
            rprint(f"[yellow][WARNING][/yellow] Missing best-cut selection for {config} {name} {energy}.")
            continue

        nhits_value, ophits_value, adjcl_value = selection
        sigmas_path = (
            f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/HEP/{args.folder.lower()}/"
            f"{config}/{name}/{config}_{name}_{energy}_HEP_Results.pkl"
        )
        if not os.path.exists(sigmas_path):
            rprint(f"[yellow][WARNING][/yellow] Missing HEP results for {config} {name} {energy}.")
            continue

        sigmas_df = pd.read_pickle(sigmas_path)
        required_cols = ["Config", "Name", "NHits", "OpHits", "AdjCl", "Exposure"]
        missing_cols = [c for c in required_cols if c not in sigmas_df.columns]
        if sigmas_df.empty or missing_cols:
            rprint(
                f"[yellow][WARNING][/yellow] Invalid HEP results for {config} {name} {energy}: "
                f"missing columns {missing_cols}."
            )
            continue

        sigma_rows = sigmas_df.loc[
            (sigmas_df["Config"] == config)
            * (sigmas_df["Name"] == name)
            * (sigmas_df["NHits"] == int(nhits_value))
            * (sigmas_df["OpHits"] == int(ophits_value))
            * (sigmas_df["AdjCl"] == int(adjcl_value))
        ].copy()
        if sigma_rows.empty:
            rprint(
                f"[yellow][WARNING][/yellow] Missing result row for "
                f"{config} {name} {energy} NHits={nhits_value} OpHits={ophits_value} AdjCl={adjcl_value}."
            )
            continue

        sigma_row = sigma_rows.iloc[0]
        if "ProfileLikelihood" not in sigma_row.index:
            rprint(
                f"[yellow][WARNING][/yellow] No ProfileLikelihood column for {config} {name} {energy}. Skipping."
            )
            continue

        exposure_grid = as_curve(sigma_row["Exposure"])
        has_pre_pava = "PreIsotonicProfileLikelihood" in sigma_row.index

        if has_pre_pava:
            # Two-panel: top = Pre/Post-PAVA curves; bottom = isotonic correction (Post − Pre).
            profile_entries = [
                ("Pre-PAVA",  "PreIsotonicProfileLikelihood"),
                ("Post-PAVA", "ProfileLikelihood"),
            ]
            fig = make_subplots(
                rows=2, cols=1, row_heights=[0.7, 0.3], vertical_spacing=0,
                subplot_titles=(f"ProfileLikelihood significance vs exposure ({energy})", ""),
            )
            significance_peak = 0.0
            pl_curves = {}
            for label, col_key in profile_entries:
                y = np.nan_to_num(as_curve(sigma_row[col_key]), nan=0.0, posinf=0.0, neginf=0.0)
                pl_curves[label] = y
                significance_peak = max(significance_peak, float(np.max(y)))
                style = _STYLE_PRE if label == "Pre-PAVA" else _STYLE_POST
                fig.add_trace(
                    go.Scatter(
                        x=exposure_grid, y=y, mode="lines", name=label,
                        line=dict(color=style["color"], dash=style["dash"], width=style["width"]),
                        line_shape="linear",
                    ),
                    row=1, col=1,
                )
                comparison_rows.append({
                    "Config": config, "Name": name, "EnergyLabel": energy,
                    "NHits": int(nhits_value), "OpHits": int(ophits_value), "AdjCl": int(adjcl_value),
                    "Threshold": float(args.threshold),
                    "Exposure": exposure_grid.tolist(),
                    "Variable": "ProfileLikelihood",
                    "SpectrumType": label,
                    "Significance": y.tolist(),
                })

            correction = pl_curves["Post-PAVA"] - pl_curves["Pre-PAVA"]
            sym = max(0.1, 1.1 * float(np.max(np.abs(correction))))
            fig.add_trace(
                go.Scatter(
                    x=exposure_grid, y=correction, mode="lines", name="PAVA correction",
                    line=dict(color="black", dash="dot", width=1),
                    line_shape="linear", showlegend=False,
                ),
                row=2, col=1,
            )
            fig.add_hline(y=0, line=dict(color="gray", dash="dot", width=1), row=2, col=1)

            fig = format_coustom_plotly(
                fig, title=f"PL PAVA Comparison — {args.folder} — {config}",
                add_units=False, figsize=(800, 600), matches=("x", None), add_watermark=False,
            )
            fig.update_xaxes(title="", showticklabels=False, row=1, col=1)
            fig.update_xaxes(title="Exposure (kT·year)", row=2, col=1)
            fig.update_yaxes(
                title="Significance (σ)",
                range=[0, max(1.0, 1.1 * significance_peak)] if args.zoom else [0, 6],
                row=1, col=1,
            )
            fig.update_yaxes(title="PAVA correction (σ)", range=[-sym, sym], row=2, col=1)

        else:
            # Single-panel: one PL curve, no pre/post comparison available.
            y = np.nan_to_num(as_curve(sigma_row["ProfileLikelihood"]), nan=0.0, posinf=0.0, neginf=0.0)
            significance_peak = float(np.max(y))
            fig = make_subplots(rows=1, cols=1)
            fig.add_trace(
                go.Scatter(
                    x=exposure_grid, y=y, mode="lines", name="ProfileLikelihood",
                    line=dict(color="black", dash="solid", width=3),
                    line_shape="linear",
                ),
                row=1, col=1,
            )
            comparison_rows.append({
                "Config": config, "Name": name, "EnergyLabel": energy,
                "NHits": int(nhits_value), "OpHits": int(ophits_value), "AdjCl": int(adjcl_value),
                "Threshold": float(args.threshold),
                "Exposure": exposure_grid.tolist(),
                "Variable": "ProfileLikelihood",
                "SpectrumType": "ProfileLikelihood",
                "Significance": y.tolist(),
            })

            fig = format_coustom_plotly(
                fig, title=f"ProfileLikelihood — {args.folder} — {config}",
                add_units=False, figsize=(800, 400), matches=None, add_watermark=False,
            )
            fig.update_xaxes(title="Exposure (kT·year)")
            fig.update_yaxes(
                title="Significance (σ)",
                range=[0, max(1.0, 1.1 * significance_peak)] if args.zoom else [0, 6],
            )

        if args.plot:
            figure_name = f"{energy}_HEP_ProfileLikelihood_AdaptiveRebin_Comparison"
            if args.threshold is not None:
                figure_name += f"_Threshold_{args.threshold:.0f}"
            save_figure(
                fig, save_path,
                config=config, name=name, subfolder=args.folder.lower(),
                filename=figure_name, rm=args.rewrite, debug=args.debug,
            )

    if comparison_rows:
        save_df(
            pd.DataFrame(comparison_rows),
            data_path,
            config=config,
            name=name,
            subfolder=args.folder.lower(),
            filename="HEP_AdaptiveRebin_Comparison",
            rm=args.rewrite,
            debug=args.debug,
        )
