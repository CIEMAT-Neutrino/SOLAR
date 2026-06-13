import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *

analysis_info = load_analysis_info(str(root))

save_path = f"{root}/images/analysis/hep"
data_path = f"{analysis_info['PATH']}/HEP"
for this_path in [save_path]:
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
    description="Compare HEP significance curves with and without adaptive rebinning"
)
default_reference = analysis_info.get("BEST_SIGMA_SIGNIFICANCE_REFERENCE", {}).get(
    "HEP", "Asimov"
)
if default_reference not in ["Gaussian", "Asimov", "ProfileLikelihood"]:
    default_reference = "Asimov"

parser.add_argument("--analysis", type=str, default="HEP")
parser.add_argument(
    "--reference",
    type=str,
    choices=["Gaussian", "Asimov", "ProfileLikelihood"],
    default=default_reference,
)
parser.add_argument("--config", nargs="+", type=str, default=["hd_1x2x6_centralAPA"])
parser.add_argument("--name", nargs="+", type=str, default=["marley"])
parser.add_argument(
    "--folder", type=str, default="Nominal", choices=["Reduced", "Truncated", "Nominal"]
)
parser.add_argument("--exposure", type=float, default=30)
parser.add_argument(
    "--energy",
    nargs="+",
    type=str,
    choices=[
        "SignalParticleK",
        "ClusterEnergy",
        "TotalEnergy",
        "SelectedEnergy",
        "SolarEnergy",
    ],
    default=[
        "SignalParticleK",
        "ClusterEnergy",
        "TotalEnergy",
        "SelectedEnergy",
        "SolarEnergy",
    ],
)
parser.add_argument("--signal_uncertainty", type=float, default=None)
parser.add_argument("--background_uncertainty", type=float, default=None)
parser.add_argument("--nhits", type=int, default=None)
parser.add_argument("--ophits", type=int, default=None)
parser.add_argument("--adjcls", type=int, default=None)
parser.add_argument(
    "--threshold",
    type=float,
    default=get_analysis_threshold(str(root), "HEP", stage="SIGNIFICANCE", fallback=0.0),
)
parser.add_argument("--zoom", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument(
    "--pkl_label",
    type=str,
    default="highest",
    help="Label of the best-cut pkl to read (e.g. 'highest', 'highest_spiked').",
)
args = parser.parse_args()


comparison_rows = []
style_map = {
    ("Raw", "NoRebin"): dict(color=compare[1], dash="dash", width=2),
    ("Raw", "AdaptiveRebin"): dict(color=compare[1], dash="dot", width=3),
    ("Smoothed", "NoRebin"): dict(color='black', dash="dash", width=2),
    ("Smoothed", "AdaptiveRebin"): dict(color='black', dash="solid", width=3),
}


for config, name, energy in product(args.config, args.name, args.energy):
    selection = get_selection_cuts(config, name, energy, args)
    if selection is None:
        rprint(
            f"[yellow][WARNING][/yellow] Missing best-cut selection for {config} {name} {energy}."
        )
        continue

    nhits_value, ophits_value, adjcl_value = selection
    sigmas_path = (
        f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/HEP/{args.folder.lower()}/"
        f"{config}/{name}/{config}_{name}_{energy}_HEP_Results.pkl"
    )
    if not os.path.exists(sigmas_path):
        rprint(
            f"[yellow][WARNING][/yellow] Missing HEP results file for {config} {name} {energy}."
        )
        continue

    sigmas_df = pd.read_pickle(sigmas_path)
    required_sigma_columns = [
        "Config",
        "Name",
        "NHits",
        "OpHits",
        "AdjCl",
        "Exposure",
    ]
    missing_sigma_columns = [
        column for column in required_sigma_columns if column not in sigmas_df.columns
    ]
    if sigmas_df.empty or missing_sigma_columns:
        rprint(
            f"[yellow][WARNING][/yellow] Invalid HEP results payload for {config} {name} {energy}: "
            f"missing columns {missing_sigma_columns}. "
            "Skipping adaptive-rebin comparison for this selection."
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
            f"[yellow][WARNING][/yellow] Missing result row for {config} {name} {energy} NHits={nhits_value} OpHits={ophits_value} AdjCl={adjcl_value}."
        )
        continue

    sigma_row = sigma_rows.iloc[0]
    exposure_grid = as_curve(sigma_row["Exposure"])
    total_bins_default = int(len(as_curve(sigma_row.get("SignificanceEnergy", []))))
    if total_bins_default <= 0:
        total_bins_default = int(len(exposure_grid))

    for significance_type in ["Asimov", "Gaussian"]:
        curve_keys = {
            ("Raw", "NoRebin"): f"Raw{significance_type}NoRebin",
            ("Raw", "AdaptiveRebin"): f"Raw{significance_type}",
            ("Smoothed", "NoRebin"): f"{significance_type}NoRebin",
            ("Smoothed", "AdaptiveRebin"): f"{significance_type}",
        }
        bin_keys = {
            ("Raw", "NoRebin"): None,
            ("Raw", "AdaptiveRebin"): "RawAdaptiveBins",
            ("Smoothed", "NoRebin"): None,
            ("Smoothed", "AdaptiveRebin"): "AdaptiveBins",
        }

        missing = [key for key in curve_keys.values() if key not in sigma_row.index]
        if missing:
            rprint(
                f"[yellow][WARNING][/yellow] Missing columns {missing} in HEP results for {config} {name} {energy}."
            )
            continue

        fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.7, 0.3],
            vertical_spacing=0,
            subplot_titles=(
                f"{significance_type} significance vs exposure ({energy})",
                "",
            ),
        )

        total_bins = total_bins_default
        significance_peak = 0.0

        for spectrum_label, rebin_mode in [
            ("Raw", "NoRebin"),
            ("Raw", "AdaptiveRebin"),
            ("Smoothed", "NoRebin"),
            ("Smoothed", "AdaptiveRebin"),
        ]:
            curve_key = curve_keys[(spectrum_label, rebin_mode)]
            y_values = np.nan_to_num(
                as_curve(sigma_row[curve_key]), nan=0.0, posinf=0.0, neginf=0.0
            )
            significance_peak = max(significance_peak, float(np.max(y_values)))
            style = style_map[(spectrum_label, rebin_mode)]
            label = f"{spectrum_label} {rebin_mode}"

            fig.add_trace(
                go.Scatter(
                    x=exposure_grid,
                    y=y_values,
                    mode="lines",
                    name=label,
                    line=dict(
                        color=style["color"],
                        dash=style["dash"],
                        width=style["width"],
                    ),
                    line_shape="linear",
                ),
                row=1,
                col=1,
            )

            bin_key = bin_keys[(spectrum_label, rebin_mode)]
            if bin_key is None or bin_key not in sigma_row.index:
                grouped_bins = np.full_like(exposure_grid, float(total_bins), dtype=float)
            else:
                grouped_bins = np.nan_to_num(
                    as_curve(sigma_row[bin_key]),
                    nan=float(total_bins),
                    posinf=float(total_bins),
                    neginf=float(total_bins),
                )
                if len(grouped_bins) != len(exposure_grid):
                    grouped_bins = np.interp(
                        exposure_grid,
                        np.linspace(exposure_grid[0], exposure_grid[-1], len(grouped_bins)),
                        grouped_bins,
                    )

            fig.add_trace(
                go.Scatter(
                    x=exposure_grid,
                    y=grouped_bins,
                    mode="lines",
                    name=f"{label} bins",
                    line=dict(
                        color=style["color"],
                        dash=style["dash"],
                        width=max(1, style["width"] - 1),
                    ),
                    line_shape="linear",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

            comparison_rows.append(
                {
                    "Config": config,
                    "Name": name,
                    "EnergyLabel": energy,
                    "NHits": int(nhits_value),
                    "OpHits": int(ophits_value),
                    "AdjCl": int(adjcl_value),
                    "Threshold": float(args.threshold),
                    "Exposure": exposure_grid.tolist(),
                    "SignificanceType": significance_type,
                    "SpectrumType": spectrum_label,
                    "RebinMode": rebin_mode,
                    "Significance": y_values.tolist(),
                    "GroupedBins": grouped_bins.tolist(),
                }
            )

        fig = format_coustom_plotly(
            fig,
            title=(
                f"Rebin Comparison - {args.folder} - {config} "
            ),
            add_units=False,
            figsize=(800, 600),
            matches=("x", None),
            add_watermark=False,
        )
        fig.update_xaxes(title="", showticklabels=False, row=1, col=1)
        fig.update_xaxes(title="Exposure (kT·year)", row=2, col=1)
        fig.update_yaxes(
            title="Significance (σ)",
            range=[0, max(1.0, 1.1 * significance_peak)] if args.zoom else [0, 6],
            row=1,
            col=1,
        )
        fig.update_yaxes(title="Grouped bins", row=2, col=1)

        figure_name = f"{energy}_HEP_{significance_type}_AdaptiveRebin_Comparison"
        if args.threshold is not None:
            figure_name += f"_Threshold_{args.threshold:.0f}"
        save_figure(
            fig,
            save_path,
            config=config,
            name=name,
            subfolder=args.folder.lower(),
            filename=figure_name,
            rm=args.rewrite,
            debug=args.plot,
        )

    pl_required = ["ProfileLikelihood", "RawProfileLikelihood"]
    if all(c in sigma_row.index for c in pl_required):
        fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.7, 0.3],
            vertical_spacing=0,
            subplot_titles=(
                f"ProfileLikelihood significance vs exposure ({energy})",
                "",
            ),
        )
        significance_peak = 0.0
        profile_entries = [
            ("Raw", "NoRebin", "RawProfileLikelihoodNoRebin"),
            ("Raw", "AdaptiveRebin", "RawProfileLikelihood"),
            ("Smoothed", "NoRebin", "ProfileLikelihoodNoRebin"),
            ("Smoothed", "AdaptiveRebin", "ProfileLikelihood"),
        ]
        has_full_profile_shape = all(curve_key in sigma_row.index for _, _, curve_key in profile_entries)
        if not has_full_profile_shape:
            profile_entries = [
                ("Raw", "AdaptiveRebin", "RawProfileLikelihood"),
                ("Smoothed", "AdaptiveRebin", "ProfileLikelihood"),
            ]

        pl_curves = {}
        for spectrum_label, rebin_mode, curve_key in profile_entries:
            y_values = np.nan_to_num(
                as_curve(sigma_row[curve_key]), nan=0.0, posinf=0.0, neginf=0.0
            )
            pl_curves[(spectrum_label, rebin_mode)] = y_values
            significance_peak = max(significance_peak, float(np.max(y_values)))
            style = style_map[(spectrum_label, rebin_mode)]
            label = f"{spectrum_label} {rebin_mode}"
            fig.add_trace(
                go.Scatter(
                    x=exposure_grid,
                    y=y_values,
                    mode="lines",
                    name=label,
                    line=dict(color=style["color"], dash=style["dash"], width=style["width"]),
                    line_shape="linear",
                ),
                row=1,
                col=1,
            )
            comparison_rows.append(
                {
                    "Config": config,
                    "Name": name,
                    "EnergyLabel": energy,
                    "NHits": int(nhits_value),
                    "OpHits": int(ophits_value),
                    "AdjCl": int(adjcl_value),
                    "Threshold": float(args.threshold),
                    "Exposure": exposure_grid.tolist(),
                    "SignificanceType": "ProfileLikelihood",
                    "SpectrumType": spectrum_label,
                    "RebinMode": rebin_mode,
                    "Significance": y_values.tolist(),
                }
            )

        # Bottom panel: smoothing effect = Raw PL − Smoothed PL for each rebin mode.
        # Positive = smoothing reduces significance; negative = smoothing boosts it.
        smoothing_peak = 0.0
        rebin_modes_present = list(dict.fromkeys(rm for _, rm, _ in profile_entries))
        for rebin_mode in rebin_modes_present:
            raw_key = ("Raw", rebin_mode)
            smo_key = ("Smoothed", rebin_mode)
            if raw_key not in pl_curves or smo_key not in pl_curves:
                continue
            smoothing_effect = pl_curves[raw_key] - pl_curves[smo_key]
            smoothing_peak = max(smoothing_peak, float(np.max(np.abs(smoothing_effect))))
            style = style_map[smo_key]
            fig.add_trace(
                go.Scatter(
                    x=exposure_grid,
                    y=smoothing_effect,
                    mode="lines",
                    name=f"Smoothing {rebin_mode}",
                    line=dict(color=style["color"], dash=style["dash"], width=max(1, style["width"] - 1)),
                    line_shape="linear",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

        fig = format_coustom_plotly(
            fig,
            title=f"Rebin Comparison - {args.folder} - {config} ",
            add_units=False,
            figsize=(800, 600),
            matches=("x", None),
            add_watermark=False,
        )
        fig.update_xaxes(title="", showticklabels=False, row=1, col=1)
        fig.update_xaxes(title="Exposure (kT·year)", row=2, col=1)
        fig.update_yaxes(
            title="Significance (σ)",
            range=[0, max(1.0, 1.1 * significance_peak)] if args.zoom else [0, 6],
            row=1,
            col=1,
        )
        sym = max(0.1, 1.1 * smoothing_peak)
        fig.update_yaxes(title="Smoothing effect (σ)", range=[-sym, sym], row=2, col=1)
        fig.add_hline(y=0, line=dict(color="gray", dash="dot", width=1), row=2, col=1)

        figure_name = f"{energy}_HEP_ProfileLikelihood_AdaptiveRebin_Comparison"
        if args.threshold is not None:
            figure_name += f"_Threshold_{args.threshold:.0f}"
        save_figure(
            fig,
            save_path,
            config=config,
            name=name,
            subfolder=args.folder.lower(),
            filename=figure_name,
            rm=args.rewrite,
            debug=args.plot,
        )


if comparison_rows:
    comparison_df = pd.DataFrame(comparison_rows)
    save_df(
        comparison_df,
        data_path,
        config=args.config[0],
        name=name,
        subfolder=args.folder.lower(),
        filename="HEP_AdaptiveRebin_Comparison",
        rm=args.rewrite,
        debug=args.debug,
    )
