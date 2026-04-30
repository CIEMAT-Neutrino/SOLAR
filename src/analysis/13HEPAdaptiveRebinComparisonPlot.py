import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *


save_path = f"{root}/images/analysis/hep"
data_path = f"{root}/data/analysis/hep"
for this_path in [save_path, data_path]:
    if not os.path.exists(this_path):
        os.makedirs(this_path)


def get_selection_cuts(config: str, name: str, energy: str, args: argparse.Namespace):
    sigma_path = (
        f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/HEP/{args.folder.lower()}/"
        f"{config}/{name}/{config}_{name}_highest_HEP.pkl"
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
analysis_info = load_analysis_info(str(root))
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

    profile_rows = pd.DataFrame()
    profile_path = (
        f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/{args.analysis.upper()}/{args.folder.lower()}/"
        f"{config}/{name}/{config}_{name}_{energy}_{args.analysis}_ProfileLikelihood.pkl"
    )
    if os.path.exists(profile_path):
        profile_df = pd.read_pickle(profile_path)
        required_profile_columns = [
            "Config",
            "Name",
            "Energy",
            "NHits",
            "OpHits",
            "AdjCl",
            "Exposure",
            "RawProfileLikelihood",
            "ProfileLikelihood",
            "RawAdaptiveBins",
            "AdaptiveBins",
        ]
        missing_profile_columns = [
            column for column in required_profile_columns if column not in profile_df.columns
        ]
        if profile_df.empty or missing_profile_columns:
            rprint(
                f"[yellow][WARNING][/yellow] Invalid profile-likelihood payload for {config} {name} {energy}: "
                f"missing columns {missing_profile_columns}. "
                "ProfileLikelihood adaptive-rebin comparison will be skipped for this selection."
            )
            profile_rows = pd.DataFrame()
        else:
            profile_rows = profile_df.loc[
                (profile_df["Config"] == config)
                & (profile_df["Name"] == name)
                & (profile_df["Energy"] == energy)
                & (profile_df["NHits"] == int(nhits_value))
                & (profile_df["OpHits"] == int(ophits_value))
                & (profile_df["AdjCl"] == int(adjcl_value))
            ].copy()

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

    if not profile_rows.empty:
        profile_row = profile_rows.iloc[0]
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
        profile_styles = style_map
        profile_exposure = as_curve(profile_row["Exposure"])
        significance_peak = 0.0
        profile_entries = [
            ("Raw", "NoRebin", "RawProfileLikelihoodNoRebin", "RawAdaptiveBinsNoRebin"),
            ("Raw", "AdaptiveRebin", "RawProfileLikelihood", "RawAdaptiveBins"),
            ("Smoothed", "NoRebin", "ProfileLikelihoodNoRebin", "AdaptiveBinsNoRebin"),
            ("Smoothed", "AdaptiveRebin", "ProfileLikelihood", "AdaptiveBins"),
        ]
        has_full_profile_shape = all(curve_key in profile_row.index for _, _, curve_key, _ in profile_entries)
        if not has_full_profile_shape:
            profile_entries = [
                ("Raw", "AdaptiveRebin", "RawProfileLikelihood", "RawAdaptiveBins"),
                ("Smoothed", "AdaptiveRebin", "ProfileLikelihood", "AdaptiveBins"),
            ]

        for spectrum_label, rebin_mode, curve_key, bins_key in profile_entries:
            y_values = np.nan_to_num(
                as_curve(profile_row[curve_key]), nan=0.0, posinf=0.0, neginf=0.0
            )
            significance_peak = max(significance_peak, float(np.max(y_values)))
            if bins_key in profile_row.index:
                grouped_bins = np.nan_to_num(
                    as_curve(profile_row[bins_key]), nan=0.0, posinf=0.0, neginf=0.0
                )
            else:
                grouped_bins = np.full_like(profile_exposure, float(total_bins_default), dtype=float)

            if len(grouped_bins) != len(profile_exposure):
                grouped_bins = np.interp(
                    profile_exposure,
                    np.linspace(profile_exposure[0], profile_exposure[-1], len(grouped_bins)),
                    grouped_bins,
                )

            style = profile_styles[(spectrum_label, rebin_mode)]
            label = f"{spectrum_label} {rebin_mode}"
            fig.add_trace(
                go.Scatter(
                    x=profile_exposure,
                    y=y_values,
                    mode="lines",
                    name=label,
                    line=dict(color=style["color"], dash=style["dash"], width=style["width"]),
                    line_shape="linear",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=profile_exposure,
                    y=grouped_bins,
                    mode="lines",
                    name=f"{label} bins",
                    line=dict(color=style["color"], dash=style["dash"], width=max(1, style["width"] - 1)),
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
                    "Exposure": as_curve(profile_row["Exposure"]).tolist(),
                    "SignificanceType": "ProfileLikelihood",
                    "SpectrumType": spectrum_label,
                    "RebinMode": rebin_mode,
                    "Significance": y_values.tolist(),
                    "GroupedBins": grouped_bins.tolist(),
                }
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
        fig.update_yaxes(title="Grouped bins", row=2, col=1)

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
        name=None,
        subfolder=args.folder.lower(),
        filename="HEP_AdaptiveRebin_Comparison",
        rm=args.rewrite,
        debug=args.debug,
    )
