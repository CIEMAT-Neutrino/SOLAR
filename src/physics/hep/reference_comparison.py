import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *


save_path = f"{root}/images/analysis/hep"
data_path = f"{root}/data/analysis/hep"
for this_path in [save_path, data_path]:
    if not os.path.exists(this_path):
        os.makedirs(this_path)


def get_selection_cuts(config: str, name: str, energy: str, args: argparse.Namespace):
    if args.nhits is not None and args.ophits is not None and args.adjcls is not None:
        return int(args.nhits), int(args.ophits), int(args.adjcls)

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

parser = argparse.ArgumentParser(
    description="Compare Gaussian, Asimov, and ProfileLikelihood HEP significance outputs in dedicated figures"
)
parser.add_argument("--analysis", type=str, default="HEP")
parser.add_argument("--config", nargs="+", type=str, default=["hd_1x2x6_centralAPA"])
parser.add_argument("--name", nargs="+", type=str, default=["marley"])
parser.add_argument("--folder", type=str, default="Reduced")
parser.add_argument("--exposure", type=float, default=30)
parser.add_argument("--signal_uncertainty", type=float, default=0.04)
parser.add_argument("--background_uncertainty", type=float, default=0.02)
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

comparison_styles = {
    ("Asimov", "Raw"): dict(color="black", dash="dot", width=2),
    ("Asimov", "Smoothed"): dict(color="black", dash="solid", width=3),
    ("Gaussian", "Raw"): dict(color=compare[1], dash="dot", width=2),
    ("Gaussian", "Smoothed"): dict(color=compare[1], dash="solid", width=3),
    ("ProfileLikelihood", "Raw"): dict(color=compare[2], dash="dot", width=2),
    ("ProfileLikelihood", "Smoothed"): dict(color=compare[2], dash="solid", width=3),
}

reference_variables = ["Asimov", "Gaussian", "ProfileLikelihood"]

for config, name, energy in product(args.config, args.name, args.energy):
    selection = get_selection_cuts(config, name, energy, args)
    if selection is None:
        rprint(
            f"[yellow][WARNING][/yellow] Missing best-cut selection for {config} {name} {energy}."
        )
        continue
    nhits_value, ophits_value, adjcl_value = selection

    significance_file = (
        Path(data_path)
        / config
        / name
        / args.folder.lower()
        / f"{config}_{name}_HEP_Significance.pkl"
    )
    exposure_file = (
        Path(data_path)
        / config
        / name
        / args.folder.lower()
        / f"{config}_{name}_HEP_Exposure.pkl"
    )
    if not significance_file.exists() or not exposure_file.exists():
        rprint(
            f"[yellow][WARNING][/yellow] Missing saved HEP plot data for {config} {name} {energy}. Run the HEP plotting macros first."
        )
        continue

    significance_df = pd.read_pickle(significance_file)
    exposure_df = pd.read_pickle(exposure_file)
    if (
        "EnergyLabel" not in significance_df.columns
        or "EnergyLabel" not in exposure_df.columns
    ):
        rprint(
            f"[yellow][WARNING][/yellow] Missing EnergyLabel in saved HEP plot data for {config} {name}. Re-run the HEP plotting macros."
        )
        continue

    significance_rows = significance_df.loc[
        (significance_df["Config"] == config)
        & (significance_df["Name"] == name)
        & (significance_df["EnergyLabel"] == energy)
        & (significance_df["Variable"].isin(reference_variables))
    ].copy()
    exposure_rows = exposure_df.loc[
        (exposure_df["Config"] == config)
        & (exposure_df["Name"] == name)
        & (exposure_df["EnergyLabel"] == energy)
        & (exposure_df["Variable"].isin(reference_variables))
    ].copy()

    for column, value in [
        ("NHits", nhits_value),
        ("OpHits", ophits_value),
        ("AdjCl", adjcl_value),
    ]:
        if column in significance_rows.columns:
            significance_rows = significance_rows.loc[
                significance_rows[column] == int(value)
            ].copy()
        if column in exposure_rows.columns:
            exposure_rows = exposure_rows.loc[exposure_rows[column] == int(value)].copy()

    if significance_rows.empty or exposure_rows.empty:
        rprint(
            f"[yellow][WARNING][/yellow] Missing comparison inputs for {config} {name} {energy} after cut filtering NHits={nhits_value} OpHits={ophits_value} AdjCl={adjcl_value}."
        )
        continue

    plot_df = pd.read_pickle(
        f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/signal/{args.folder.lower()}/HEP/{config}/{name}/{config}_{name}_{energy}_Rebin.pkl"
    )
    for _, filepath in load_available_background_dataframes(
        str(root), "HEP", args.folder, config, energy
    ):
        bkg_df = pd.read_pickle(filepath)
        plot_df = pd.concat([plot_df, bkg_df], ignore_index=True)

    this_plot_df = plot_df.loc[
        (plot_df["NHits"] == int(nhits_value))
        & (plot_df["OpHits"] == int(ophits_value))
        & (plot_df["AdjCl"] == int(adjcl_value))
    ].copy()

    component_specs = [
        ("hep", "HEP", "Osc", "Mean", "signal", "Signal", "rgb(204,80,62)"),
        ("gamma", get_background_style(str(root), "gamma").get("label", "gamma"), "Truth", "Mean", "background", "Background", get_background_style(str(root), "gamma").get("color", "black")),
        ("neutron", get_background_style(str(root), "neutron").get("label", "neutron"), "Truth", "Mean", "background", "Background", get_background_style(str(root), "neutron").get("color", "rgb(15,133,84)")),
        ("radiological", get_background_style(str(root), "radiological").get("label", "radiological"), "Truth", "Mean", "background", "Background", get_background_style(str(root), "radiological").get("color", "rgb(120, 94, 240)")),
        ("8B", "8B", "Osc", "Mean", "background", "Background", "rgb(225,124,5)"),
    ]
    smoothing_config = get_smoothing_config(
        str(root), analysis_name="HEP", dimensions="1d", stage="significance"
    )

    fig_energy = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0,
        subplot_titles=("", ""),
    )
    positive_hist_counts = []
    for (
        component,
        component_label,
        oscillation,
        mean_label,
        legend_group,
        legend_group_title,
        color,
    ) in component_specs:
        component_df = this_plot_df.loc[
            (this_plot_df["Component"] == component)
            & (this_plot_df["Oscillation"] == oscillation)
            & (this_plot_df["Mean"] == mean_label)
        ].copy()
        if component_df.empty:
            continue

        xvals = np.asarray(component_df["Energy"].values[0], dtype=float)
        counts = np.nan_to_num(
            np.asarray(component_df["Counts"].values[0], dtype=float),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        errors = np.nan_to_num(
            np.asarray(component_df["Error"].values[0], dtype=float),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        mc_counts = (
            np.nan_to_num(np.asarray(component_df["MCCounts"].values[0], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
            if "MCCounts" in component_df.columns
            else np.zeros_like(counts)
        )
        component_smoothing_config = get_component_smoothing_config(smoothing_config, component)
        smoothed_counts = smooth_threshold_slice(counts, 0, component_smoothing_config)
        _smoothed_errors = smooth_histogram_errors(
            errors,
            component_smoothing_config,
            counts=counts,
            mc_counts=mc_counts,
        )

        for spectrum_type, yvals, width, dash, showlegend in [
            ("Raw", counts, 2, "dot", False),
            ("Smoothed", smoothed_counts, 3, "solid", True),
        ]:
            positive_hist_counts.extend(yvals[yvals > 0])
            fig_energy.add_trace(
                go.Scatter(
                    x=xvals,
                    y=yvals,
                    mode="lines",
                    name=component_label,
                    line=dict(color=color, dash=dash, width=width),
                    line_shape="hvh",
                    legend="legend",
                    legendgroup=legend_group,
                    legendgrouptitle=dict(text=legend_group_title),
                    showlegend=showlegend,
                ),
                row=1,
                col=1,
            )

    significance_max = 0.0
    for variable in reference_variables:
        for spectrum_type in ["Raw", "Smoothed"]:
            row = significance_rows.loc[
                (significance_rows["Variable"] == variable)
                & (significance_rows["SpectrumType"] == spectrum_type)
            ]
            if row.empty:
                continue
            xvals = np.asarray(row["Energy"].values[0], dtype=float)
            yvals = np.nan_to_num(
                np.asarray(row["Significance"].values[0], dtype=float),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            significance_max = max(significance_max, float(np.max(yvals)))

            style = comparison_styles[(variable, spectrum_type)]
            fig_energy.add_trace(
                go.Scatter(
                    x=xvals,
                    y=yvals,
                    mode="lines",
                    name=variable,
                    line=dict(
                        color=style["color"], dash=style["dash"], width=style["width"]
                    ),
                    line_shape="hvh",
                    legend="legend2",
                    legendgroup="reference",
                    legendgrouptitle=dict(text="Reference"),
                    showlegend=spectrum_type == "Smoothed",
                ),
                row=2,
                col=1,
            )

    add_histogram_style_legend_traces(
        fig_energy,
        row=1,
        col=1,
        legendgroup="histogram",
    )
    fig_energy = format_coustom_plotly(
        fig_energy,
        figsize=(800, 600),
        tickformat=(".1f", ".0e"),
        add_units=False,
        title=f"HEP Significance Comparison - {args.folder} - {config}",
        matches=("x", None),
    )
    fig_energy.update_layout(
        legend2=dict(
            y=0.14,
            x=0.79,
        )
    )
    if args.threshold is not None:
        fig_energy.add_vline(
            x=args.threshold,
            line_dash="dash",
            line_color="grey",
            annotation=dict(text="Threshold", showarrow=False),
            annotation_position="bottom right",
        )

    if positive_hist_counts:
        log_min = (
            np.floor(np.log10(max(min(positive_hist_counts), 1e-6)))
            if args.zoom
            else -2
        )
        log_max = np.ceil(np.log10(max(positive_hist_counts)))
        fig_energy.update_yaxes(
            type="log",
            tickformat=".0e",
            dtick=1,
            range=[log_min, log_max],
            title=f"Counts ({args.exposure:.0f} year·MeV)^-1",
            row=1,
            col=1,
        )
    else:
        fig_energy.update_yaxes(
            type="log",
            tickformat=".0e",
            dtick=1,
            title=f"Counts ({args.exposure:.0f} year·MeV)^-1",
            row=1,
            col=1,
        )
    fig_energy.update_yaxes(
        tickformat=".0f",
        range=[0, max(1.0, 1.1 * significance_max)] if args.zoom else [0, 6],
        title="Significance (σ)",
        row=2,
        col=1,
    )
    fig_energy.update_xaxes(range=[8, 26], showticklabels=False, row=1, col=1)
    fig_energy.update_xaxes(
        range=[8, 26], title="Reconstructed Neutrino Energy (MeV)", row=2, col=1
    )

    figure_name = f"{energy}_HEP_Significance_Comparison"
    if args.exposure is not None:
        figure_name += f"_Exposure_{args.exposure:.0f}"
    if args.threshold is not None:
        figure_name += f"_Threshold_{args.threshold:.0f}"

    save_figure(
        fig_energy,
        save_path,
        config=config,
        name=name,
        subfolder=args.folder.lower(),
        filename=figure_name,
        rm=args.rewrite,
        debug=args.plot,
    )

    fig_exposure = make_subplots(rows=1, cols=1)
    exposure_max = 0.0
    for variable in ["Asimov", "Gaussian"]:
        row = exposure_rows.loc[exposure_rows["Variable"] == variable]
        if row.empty:
            continue
        xvals = np.asarray(row["Exposure"].values[0], dtype=float)
        y_raw = np.nan_to_num(
            np.asarray(row["RawSignificance"].values[0], dtype=float),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        y_smooth = np.nan_to_num(
            np.asarray(row["Significance"].values[0], dtype=float),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        exposure_max = max(exposure_max, float(np.max(y_smooth)))

        raw_style = comparison_styles[(variable, "Raw")]
        smooth_style = comparison_styles[(variable, "Smoothed")]
        y_upper = None
        y_lower = None
        if (
            variable == "Asimov"
            and "SignificanceError+" in row.columns
            and "SignificanceError-" in row.columns
        ):
            y_plus = y_smooth + np.asarray(
                row["SignificanceError+"].values[0], dtype=float
            )
            y_minus = y_smooth - np.asarray(
                row["SignificanceError-"].values[0], dtype=float
            )
            y_upper = y_plus
            y_lower = y_minus

        add_reference_pair_traces(
            fig_exposure,
            x=xvals,
            y_raw=y_raw,
            y_smoothed=y_smooth,
            name=variable,
            raw_style=raw_style,
            smoothed_style=smooth_style,
            legend="legend",
            legendgroup="reference",
            legendgrouptitle="Reference",
            showlegend_raw=False,
            showlegend_smoothed=True,
            line_shape="linear",
            y_upper=y_upper,
            y_lower=y_lower,
        )

    add_histogram_style_legend_traces(
        fig_exposure,
        legendgroup="histogram",
    )
    fig_exposure = format_coustom_plotly(
        fig_exposure,
        tickformat=(".1f", ".1f"),
        add_units=False,
        title=f"Selected Sample for Solar Neutrino HEP Exposure Comparison",
        matches=(None, None),
    )
    fig_exposure.update_yaxes(
        tickformat=".1f",
        dtick=1,
        range=[0, max(1.0, 1.1 * exposure_max)] if args.zoom else [0, 6],
        title="Significance (σ)",
    )
    fig_exposure.update_xaxes(
        range=[-1, args.exposure], zeroline=False, title="Exposure (kT·year)"
    )

    figure_name = f"{energy}_HEP_Exposure_Comparison"
    if args.threshold is not None:
        figure_name += f"_Threshold_{args.threshold:.0f}"

    save_figure(
        fig_exposure,
        save_path,
        config=config,
        name=name,
        subfolder=args.folder.lower(),
        filename=figure_name,
        rm=args.rewrite,
        debug=args.plot,
    )
