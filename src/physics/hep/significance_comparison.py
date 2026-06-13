import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *

analysis_info = load_analysis_info(str(root))

save_path = f"{root}/images/analysis/hep"
data_path = f"{analysis_info['PATH']}/HEP"
for this_path in [save_path]:
    if not os.path.exists(this_path):
        os.makedirs(this_path)


def get_selection_cuts(config: str, name: str, energy: str, args: argparse.Namespace):
    if args.nhits is not None and args.ophits is not None and args.adjcls is not None:
        return int(args.nhits), int(args.ophits), int(args.adjcls)

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

parser = argparse.ArgumentParser(
    description="Compare Gaussian and Asimov HEP significance outputs in dedicated figures"
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
parser.add_argument(
    "--pkl_label",
    type=str,
    default="highest",
    help="Label of the best-cut pkl to read (e.g. 'highest', 'highest_spiked').",
)
args = parser.parse_args()

comparison_styles = {
    ("Asimov", "Raw"): dict(color="black", dash="dot", width=2),
    ("Asimov", "Smoothed"): dict(color="black", dash="solid", width=3),
    ("Gaussian", "Raw"): dict(color=compare[1], dash="dot", width=2),
    ("Gaussian", "Smoothed"): dict(color=compare[1], dash="solid", width=3),
}

reference_variables = ["Asimov", "Gaussian"]

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
    if not significance_file.exists():
        rprint(
            f"[yellow][WARNING][/yellow] Missing saved HEP plot data for {config} {name} {energy}. Run the HEP plotting macros first."
        )
        continue

    significance_df = pd.read_pickle(significance_file)
    if "EnergyLabel" not in significance_df.columns:
        rprint(
            f"[yellow][WARNING][/yellow] Missing EnergyLabel in saved HEP plot data for {config} {name}. Re-run the HEP plotting macros."
        )
        continue

    counts_file = significance_file.parent / significance_file.name.replace("Significance", "Counts")
    counts_df = pd.read_pickle(counts_file) if counts_file.exists() else pd.DataFrame()

    significance_rows = significance_df.loc[
        (significance_df["Config"] == config)
        & (significance_df["Name"] == name)
        & (significance_df["EnergyLabel"] == energy)
        & (significance_df["Variable"].isin(reference_variables))
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

    if significance_rows.empty:
        rprint(
            f"[yellow][WARNING][/yellow] Missing comparison inputs for {config} {name} {energy} after cut filtering NHits={nhits_value} OpHits={ophits_value} AdjCl={adjcl_value}."
        )
        continue

    _comp_src = counts_df if not counts_df.empty and "Component" in counts_df.columns else pd.DataFrame()
    if _comp_src.empty:
        rprint(f"[yellow][WARNING][/yellow] Missing HEP_Counts.pkl for {config} {name}. Re-run the HEP plotting macros.")
    component_rows = _comp_src.loc[
        (_comp_src["Config"] == config)
        & (_comp_src["Name"] == name)
        & (_comp_src["EnergyLabel"] == energy)
        & (_comp_src["Component"].notna())
        & (_comp_src["SpectrumType"].isin(["Raw", "Smoothed"]))
    ].copy() if not _comp_src.empty else pd.DataFrame()

    for column, value in [
        ("NHits", nhits_value),
        ("OpHits", ophits_value),
        ("AdjCl", adjcl_value),
    ]:
        if column in component_rows.columns:
            component_rows = component_rows.loc[component_rows[column] == int(value)].copy()

    fig_energy = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0,
        subplot_titles=("", ""),
    )
    component_color_overrides = {
        "gamma": "black",
        "neutron": "green",
        "neutrons": "green",
        "8b": "rgb(255,124,5)",
        "b8": "rgb(255,124,5)",
        "⁸b": "rgb(255,124,5)",
        "hep": "rgb(204,80,62)",
    }
    histogram_fallback_cycle = [
        "#1f77b4",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    component_color_map = {}
    fallback_index = 0
    positive_hist_counts = []
    radiological_present = False
    for component in component_rows["Component"].dropna().unique():
        if component not in component_color_map:
            normalized_component = (
                str(component).lower().replace(" ", "").replace("_", "")
            )
            if normalized_component in component_color_overrides:
                this_color = component_color_overrides[normalized_component]
            elif "gamma" in normalized_component:
                this_color = component_color_overrides["gamma"]
            elif "neutron" in normalized_component:
                this_color = component_color_overrides["neutron"]
            elif "8b" in normalized_component or "b8" in normalized_component or "⁸b" in normalized_component:
                this_color = component_color_overrides["8b"]
            elif "hep" in normalized_component:
                this_color = component_color_overrides["hep"]
            else:
                this_color = None
            if this_color is None:
                this_color = histogram_fallback_cycle[
                    fallback_index % len(histogram_fallback_cycle)
                ]
                fallback_index += 1
            component_color_map[component] = this_color

    for component in component_rows["Component"].dropna().unique():
        component_df = component_rows.loc[component_rows["Component"] == component]
        color = component_color_map[component]
        normalized_component = str(component).lower().replace(" ", "").replace("_", "")
        if "radiological" in normalized_component:
            radiological_present = True
        for spectrum_type, width, dash in [("Raw", 2, "dot"), ("Smoothed", 3, "solid")]:
            row = component_df.loc[component_df["SpectrumType"] == spectrum_type]
            if row.empty:
                continue
            xvals = np.asarray(row["Energy"].values[0], dtype=float)
            yvals = np.nan_to_num(
                np.asarray(row["Counts"].values[0], dtype=float),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            positive_hist_counts.extend(yvals[yvals > 0])
            fig_energy.add_trace(
                go.Scatter(
                    x=xvals,
                    y=yvals,
                    mode="lines",
                    name=component,
                    line=dict(color=color, dash=dash, width=width),
                    line_shape="hvh",
                    legend="legend",
                    legendgroup=f"component",
                    legendgrouptitle=dict(text="Components"),
                    showlegend=spectrum_type == "Smoothed",
                ),
                row=1,
                col=1,
            )

    if not radiological_present:
        radiological_style = get_background_style(str(root), "radiological")
        radiological_color = radiological_style.get("color", "rgb(120, 94, 240)")
        fig_energy.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name="radiological (not present after cuts)",
                line=dict(color=radiological_color, width=3, dash="dash"),
                line_shape="hvh",
                legend="legend",
                legendgroup="component",
                legendgrouptitle=dict(text="Components"),
                showlegend=True,
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
        add_watermark=False,
    )
    fig_energy.update_layout(
        legend2=dict(
            y=0.06,
            x=0.79,
            font=dict(size=12),
            bgcolor="rgba(255,255,255,0.5)",
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

