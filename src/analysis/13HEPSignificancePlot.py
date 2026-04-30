import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *


save_path = f"{root}/images/analysis/hep"
data_path = f"{root}/data/analysis/hep"
for this_path in [save_path, data_path]:
    if not os.path.exists(this_path):
        os.makedirs(this_path)


parser = argparse.ArgumentParser(
    description="Plot HEP significance diagnostics for a selected best-cut configuration"
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
parser.add_argument(
    "--bottom-panel-mode",
    type=str,
    choices=["rigorous", "intuitive", "both"],
    default="both",
    help="Quantity shown in the Local Proxy: rigorous grouped-bin proxy, intuitive local proxy, or both outputs",
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
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
args = parser.parse_args()


reference_for_bins = "Asimov" if args.reference == "ProfileLikelihood" else args.reference
reference_label = (
    "Asimov Proxy"
    if args.reference == "ProfileLikelihood"
    else f"{reference_for_bins} Proxy"
)
local_proxy_label = (
    "Local Discovery"
    if args.reference == "ProfileLikelihood"
    else f"Local Discovery"
)
saved_variable = "AsimovProxy" if args.reference == "ProfileLikelihood" else reference_for_bins
smoothing_config = get_smoothing_config(
    str(root), analysis_name="HEP", dimensions="1d", stage="significance"
)

hep_counts = []
hep_significance = []


def _safe_array(values):
    return np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def _group_tail_values(tail_energy, tail_values, grouped_centers, grouped_widths):
    grouped = []
    tail_energy = _safe_array(tail_energy)
    tail_values = _safe_array(tail_values)
    if tail_energy.size == 0:
        return np.zeros(len(grouped_centers), dtype=float)

    for center, width in zip(grouped_centers, grouped_widths):
        half_width = 0.5 * float(width)
        low = float(center) - half_width - 1e-9
        high = float(center) + half_width + 1e-9
        mask = (tail_energy >= low) & (tail_energy <= high)
        if not np.any(mask):
            nearest = int(np.argmin(np.abs(tail_energy - float(center))))
            grouped.append(float(tail_values[nearest]))
        else:
            grouped.append(float(np.sum(tail_values[mask])))
    return np.asarray(grouped, dtype=float)


def _density(values, widths):
    values = _safe_array(values)
    widths = _safe_array(widths)
    return np.divide(values, widths, out=np.zeros_like(values, dtype=float), where=widths > 0)


def _lower_panel_title(mode):
    if mode == "rigorous":
        return reference_label
    return local_proxy_label


def _figure_suffix(mode):
    return "BottomRigorous" if mode == "rigorous" else "BottomIntuitive"


def _subtitle():
    return (
        "Cuts selected from highest smoothed Asimov significance; "
        "profile likelihood is evaluated only in the exposure curve."
    )


def _step_fill_coordinates(centers, widths, values):
    centers = _safe_array(centers)
    widths = _safe_array(widths)
    values = _safe_array(values)
    if centers.size == 0 or widths.size == 0 or values.size == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)

    left_edges = centers - 0.5 * widths
    right_edges = centers + 0.5 * widths
    x_points = []
    y_points = []
    for left, right, value in zip(left_edges, right_edges, values):
        x_points.extend([float(left), float(right)])
        y_points.extend([float(value), float(value)])
    return np.asarray(x_points, dtype=float), np.asarray(y_points, dtype=float)


def _add_rebinned_overlay(
    fig,
    grouped_energy,
    grouped_width,
    grouped_background_counts,
    grouped_signal_counts,
):
    overlay_specs = [
        (
            "Background Rebin",
            grouped_background_counts,
            "rgba(0,0,0,0.18)",
            "black",
        ),
        (
            "HEP Rebin",
            grouped_signal_counts,
            "rgba(204,80,62,0.28)",
            "rgb(204,80,62)",
        ),
    ]
    for trace_name, trace_values, fill_color, line_color in overlay_specs:
        x_points, y_points = _step_fill_coordinates(
            grouped_energy,
            grouped_width,
            trace_values,
        )
        fig.add_trace(
            go.Scatter(
                x=x_points,
                y=y_points,
                mode="lines",
                name=trace_name,
                line=dict(color=line_color, width=1.5),
                fill="tozeroy",
                fillcolor=fill_color,
                legend="legend",
                legendgroup="adaptive_rebin",
                legendgrouptitle=dict(text="Adaptive Rebin"),
                showlegend=True,
            ),
            row=1,
            col=1,
        )


for config, name, energy in product(args.config, args.name, args.energy):
    info = json.loads(open(f"{root}/config/{config}/{config}_config.json").read())
    detector_mass = get_full_detector_mass(config, info)

    sigma_map_path = (
        f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/{args.analysis.upper()}/"
        f"{args.folder.lower()}/{config}/{name}/{config}_{name}_highest_{args.analysis}.pkl"
    )
    if not os.path.exists(sigma_map_path):
        rprint(
            f"[yellow][WARNING][/yellow] Missing best-cut map for {config} {name}: {sigma_map_path}"
        )
        continue

    sigma_map = pd.read_pickle(sigma_map_path)
    try:
        ref_plot = sigma_map[(config, name, energy)]
    except KeyError:
        rprint(
            f"[yellow][WARNING][/yellow] Missing best-cut entry for {config} {name} {energy}."
        )
        continue

    nhits_value = args.nhits if args.nhits is not None else int(ref_plot["NHits"])
    ophits_value = args.ophits if args.ophits is not None else int(ref_plot["OpHits"])
    adjcl_value = args.adjcls if args.adjcls is not None else int(ref_plot["AdjCl"])

    significance_bins_path = (
        f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/{args.analysis.upper()}/"
        f"{args.folder.lower()}/{config}/{name}/{config}_{name}_{energy}_HEP_SignificanceBins.pkl"
    )
    if not os.path.exists(significance_bins_path):
        rprint(
            f"[yellow][WARNING][/yellow] Missing per-bin significance payload for {config} {name} {energy}."
        )
        continue

    significance_bins_df = pd.read_pickle(significance_bins_path)
    required_bin_columns = [
        "Config",
        "Name",
        "EnergyLabel",
        "NHits",
        "OpHits",
        "AdjCl",
        "BinMode",
        "BinIndex",
        "RecoEnergy",
        "BinWidth",
        f"Raw{reference_for_bins}",
        f"{reference_for_bins}",
    ]
    missing_bin_columns = [
        column for column in required_bin_columns if column not in significance_bins_df.columns
    ]
    if significance_bins_df.empty or missing_bin_columns:
        rprint(
            f"[yellow][WARNING][/yellow] Invalid per-bin significance payload for {config} {name} {energy}: "
            f"missing columns {missing_bin_columns}. "
            "Skipping significance plot for this selection."
        )
        continue

    selected_no_rebin = significance_bins_df.loc[
        (significance_bins_df["Config"] == config)
        * (significance_bins_df["Name"] == name)
        * (significance_bins_df["EnergyLabel"] == energy)
        * (significance_bins_df["NHits"] == int(nhits_value))
        * (significance_bins_df["OpHits"] == int(ophits_value))
        * (significance_bins_df["AdjCl"] == int(adjcl_value))
        * (significance_bins_df["BinMode"] == "NoRebin")
    ].copy()
    selected_adaptive = significance_bins_df.loc[
        (significance_bins_df["Config"] == config)
        * (significance_bins_df["Name"] == name)
        * (significance_bins_df["EnergyLabel"] == energy)
        * (significance_bins_df["NHits"] == int(nhits_value))
        * (significance_bins_df["OpHits"] == int(ophits_value))
        * (significance_bins_df["AdjCl"] == int(adjcl_value))
        * (significance_bins_df["BinMode"] == "AdaptiveRebin")
    ].copy()
    if selected_no_rebin.empty:
        rprint(
            f"[yellow][WARNING][/yellow] Missing no-rebin significance bins for {config} {name} {energy} NHits={nhits_value} OpHits={ophits_value} AdjCl={adjcl_value}."
        )
        continue

    selected_no_rebin = selected_no_rebin.sort_values("BinIndex")
    selected_adaptive = selected_adaptive.sort_values("BinIndex")

    no_rebin_energy = _safe_array(selected_no_rebin["RecoEnergy"].values)
    no_rebin_width = _safe_array(selected_no_rebin["BinWidth"].values)
    raw_no_rebin_significance = _safe_array(selected_no_rebin[f"Raw{reference_for_bins}"].values)
    smoothed_no_rebin_significance = _safe_array(selected_no_rebin[f"{reference_for_bins}"].values)

    adaptive_energy = _safe_array(selected_adaptive["RecoEnergy"].values)
    adaptive_width = _safe_array(selected_adaptive["BinWidth"].values)
    raw_adaptive_significance = _safe_array(selected_adaptive[f"Raw{reference_for_bins}"].values)
    smoothed_adaptive_significance = _safe_array(selected_adaptive[f"{reference_for_bins}"].values)

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
        * (plot_df["OpHits"] == int(ophits_value))
        * (plot_df["AdjCl"] == int(adjcl_value))
    ].copy()
    if this_plot_df.empty:
        rprint(
            f"[yellow][WARNING][/yellow] Missing histogram payload for {config} {name} {energy}."
        )
        continue

    component_specs = [
        ("hep", "HEP", "Osc", "Mean", "signal", "Signal", "rgb(204,80,62)"),
    ]
    for bkg, oscillation, color in [
        ("gamma", "Truth", get_background_style(str(root), "gamma").get("color", "black")),
        ("neutron", "Truth", get_background_style(str(root), "neutron").get("color", "rgb(15,133,84)")),
        (
            "radiological",
            "Truth",
            get_background_style(str(root), "radiological").get("color", "rgb(120, 94, 240)"),
        ),
        ("8B", "Osc", "rgb(225,124,5)"),
    ]:
        style = get_background_style(str(root), bkg)
        component_specs.append(
            (bkg, style.get("label", bkg), oscillation, "Mean", "background", "Background", color)
        )

    scale = detector_mass * args.exposure
    total_raw_counts = np.zeros(len(hep_rebin_centers), dtype=float)
    total_smoothed_counts = np.zeros(len(hep_rebin_centers), dtype=float)
    signal_smoothed_counts = np.zeros(len(hep_rebin_centers), dtype=float)
    background_smoothed_counts = np.zeros(len(hep_rebin_centers), dtype=float)

    panel_modes = [args.bottom_panel_mode]
    if args.bottom_panel_mode == "both":
        panel_modes = ["rigorous", "intuitive"]

    base_fig = None

    for mode in panel_modes:
        fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.0,
            subplot_titles=(
                f"{energy} NHits={nhits_value} OpHits={ophits_value} AdjCl={adjcl_value}",
                "",
            ),
        )

        total_raw_counts[:] = 0.0
        total_smoothed_counts[:] = 0.0
        signal_smoothed_counts[:] = 0.0
        background_smoothed_counts[:] = 0.0
        radiological_present = False
        lower_panel_max = 0.0
        positive_count_values = []

        for (
            component,
            component_label,
            oscillation,
            mean_label,
            legend_group,
            legend_group_title,
            color,
        ) in component_specs:
            comp_df = this_plot_df.loc[
                (this_plot_df["Component"] == component)
                * (this_plot_df["Oscillation"] == oscillation)
                * (this_plot_df["Mean"] == mean_label)
            ].copy()
            if comp_df.empty:
                continue

            if component == "radiological":
                radiological_present = True

            counts = _safe_array(comp_df["Counts"].values[0])
            errors = _safe_array(comp_df["Error"].values[0])
            mc_counts = (
                _safe_array(comp_df["MCCounts"].values[0])
                if "MCCounts" in comp_df.columns
                else np.zeros_like(counts)
            )
            component_smoothing_config = get_component_smoothing_config(smoothing_config, component)
            smoothed_counts = smooth_threshold_slice(
                counts,
                0,
                component_smoothing_config,
            )
            smoothed_errors = smooth_histogram_errors(
                errors,
                component_smoothing_config,
                counts=counts,
                mc_counts=mc_counts,
            )

            total_raw_counts += counts
            total_smoothed_counts += smoothed_counts
            if component == "hep":
                signal_smoothed_counts += smoothed_counts
            else:
                background_smoothed_counts += smoothed_counts

            scaled_counts = scale * counts
            scaled_smoothed_counts = scale * smoothed_counts
            positive_count_values.extend(scaled_counts[scaled_counts > 0])
            positive_count_values.extend(scaled_smoothed_counts[scaled_smoothed_counts > 0])

            fig.add_trace(
                go.Scatter(
                    x=hep_rebin_centers,
                    y=scaled_counts,
                    mode="lines",
                    name=component_label,
                    showlegend=False,
                    line=dict(color=color, width=2, dash="dot"),
                    line_shape="hvh",
                    legend="legend",
                    legendgroup=legend_group,
                    legendgrouptitle=dict(text=legend_group_title),
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=hep_rebin_centers,
                    y=scaled_smoothed_counts,
                    mode="lines",
                    name=component_label,
                    line=dict(color=color, width=3),
                    line_shape="hvh",
                    error_y=dict(type="data", array=scale * smoothed_errors),
                    legend="legend",
                    legendgroup=legend_group,
                    legendgrouptitle=dict(text=legend_group_title),
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

        if not radiological_present:
            radiological_style = get_background_style(str(root), "radiological")
            radiological_color = radiological_style.get("color", "rgb(120, 94, 240)")
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    name="radiological (not present after cuts)",
                    line=dict(color=radiological_color, width=3, dash="dash"),
                    legend="legend",
                    legendgroup="background",
                    legendgrouptitle=dict(text="Background"),
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

        no_rebin_start = 0
        if no_rebin_energy.size > 0:
            no_rebin_start = int(np.argmin(np.abs(hep_rebin_centers - no_rebin_energy[0])))
        signal_tail_counts = signal_smoothed_counts[
            no_rebin_start:no_rebin_start + len(no_rebin_energy)
        ]
        background_tail_counts = background_smoothed_counts[
            no_rebin_start:no_rebin_start + len(no_rebin_energy)
        ]

        grouped_background_counts = _group_tail_values(
            no_rebin_energy,
            scale * background_tail_counts,
            adaptive_energy,
            adaptive_width,
        ) if adaptive_energy.size > 0 else np.zeros(0, dtype=float)
        grouped_signal_counts = _group_tail_values(
            no_rebin_energy,
            scale * signal_tail_counts,
            adaptive_energy,
            adaptive_width,
        ) if adaptive_energy.size > 0 else np.zeros(0, dtype=float)

        if adaptive_energy.size > 0:
            _add_rebinned_overlay(
                fig,
                adaptive_energy,
                adaptive_width,
                grouped_background_counts,
                grouped_signal_counts,
            )

        # add_histogram_style_legend_traces(
        #     fig,
        #     row=1,
        #     col=1,
        #     styles=[
        #         {"name": "Raw", "color": "gray", "width": 2, "dash": "dot", "showlegend": True},
        #         {"name": "Smoothed", "color": "gray", "width": 3, "dash": "solid", "showlegend": True},
        #     ],
        # )

        if mode == "rigorous":
            if adaptive_energy.size == 0:
                fig.add_annotation(
                    x=0.5,
                    y=0.15,
                    xref="paper",
                    yref="paper",
                    text="Adaptive-rebin payload not available for this selection.",
                    showarrow=False,
                )
            else:
                fig.add_trace(
                    go.Bar(
                        x=adaptive_energy,
                        y=smoothed_adaptive_significance,
                        width=adaptive_width,
                        name="Smoothed",
                        marker_color="rgba(31,119,180,0.45)",
                        marker_line=dict(color="rgb(31,119,180)", width=1.5),
                        legend="legend2",
                        legendgroup="proxy",
                        legendgrouptitle=dict(text="Local Proxy"),
                        showlegend=True,
                    ),
                    row=2,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=adaptive_energy,
                        y=raw_adaptive_significance,
                        mode="lines",
                        name="Raw",
                        line=dict(color="black", width=2, dash="dot", shape="hvh"),
                        # marker=dict(symbol="circle-open", size=7, color="black"),
                        error_x=dict(type="data", array=0.5 * adaptive_width, visible=True),
                        legend="legend2",
                        legendgroup="proxy",
                        showlegend=True,
                    ),
                    row=2,
                    col=1,
                )
                lower_panel_max = max(
                    lower_panel_max,
                    float(np.max(smoothed_adaptive_significance)),
                    float(np.max(raw_adaptive_significance)),
                )
        else:
            intuitive_density = _density(smoothed_no_rebin_significance, no_rebin_width)
            intuitive_raw_density = _density(raw_no_rebin_significance, no_rebin_width)
            lower_panel_max = max(
                lower_panel_max,
                float(np.max(intuitive_density)),
                float(np.max(intuitive_raw_density)),
            )
            fig.add_trace(
                go.Scatter(
                    x=no_rebin_energy,
                    y=intuitive_raw_density,
                    mode="lines",
                    name="Raw",
                    line=dict(color="black", width=2, dash="dot"),
                    line_shape="hvh",
                    legend="legend2",
                    legendgroup="proxy",
                    legendgrouptitle=dict(text="Local Proxy"),
                    showlegend=True,
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=no_rebin_energy,
                    y=intuitive_density,
                    mode="lines",
                    name="Smoothed",
                    line=dict(color=compare[1], width=3, dash="solid"),
                    line_shape="hvh",
                    legend="legend2",
                    legendgroup="proxy",
                    showlegend=True,
                ),
                row=2,
                col=1,
            )
            if adaptive_energy.size > 0:
                adaptive_density = _density(smoothed_adaptive_significance, adaptive_width)
                lower_panel_max = max(lower_panel_max, float(np.max(adaptive_density)))
                fig.add_trace(
                    go.Bar(
                        x=adaptive_energy,
                        y=adaptive_density,
                        width=adaptive_width,
                        name="Adaptive",
                        marker_color="rgba(31,119,180,0.22)",
                        marker_line=dict(color="rgb(31,119,180)", width=1),
                        legend="legend2",
                        legendgroup="proxy",
                        showlegend=True,
                    ),
                    row=2,
                    col=1,
                )

        fig = format_coustom_plotly(
            fig,
            tickformat=(".1f", ".0e"),
            figsize=(800, 600),
            add_units=False,
            title=(
                f"HEP Significance - {args.folder} - {config}"
            ),
            matches=("x", None),
            add_watermark=False,
        )
        fig.update_layout(
            legend=dict(font=dict(size=12), bgcolor="rgba(255,255,255,0.7)"),
            legend2=dict(font=dict(size=12), x=0.74, y=0.06, bgcolor="rgba(255,255,255,0.7)"),
        )
        fig.update_yaxes(
            type="log",
            title="Counts",
            range=[
                np.floor(np.log10(max(min(positive_count_values), 1e-6))),
                np.ceil(np.log10(max(positive_count_values))),
            ]
            if args.zoom and positive_count_values
            else [-2, 10],
            row=1,
            col=1,
        )
        if mode == "rigorous":
            fig.update_yaxes(
                title=reference_label,
                range=[0, max(1.0, 1.1 * lower_panel_max)] if args.zoom else [0, 6],
                row=2,
                col=1,
            )
        else:
            fig.update_yaxes(
                title=local_proxy_label,
                range=[0, max(1.0, 1.1 * lower_panel_max)] if args.zoom else [0, 6],
                row=2,
                col=1,
            )
        
        fig.update_xaxes(showticklabels=False, row=1, col=1, range=[8, 30])
        fig.update_xaxes(title="Reconstructed Energy (MeV)", row=2, col=1)
        
        if args.threshold is not None:
            fig.add_vline(x=float(args.threshold), line_dash="dash", line_color="grey")

        figure_name = (
            f"{energy}_HEP_Significance_{args.reference}_Exposure_{args.exposure:.0f}_{_figure_suffix(mode)}"
        )
        
        save_figure(
            fig,
            save_path,
            config=config,
            name=None,
            subfolder=args.folder.lower(),
            filename=figure_name,
            rm=args.rewrite,
            debug=args.plot,
        )

    hep_significance.append(
        {
            "Config": config,
            "Name": name,
            "EnergyLabel": energy,
            "Variable": saved_variable,
            "ProxyReference": reference_for_bins,
            "SpectrumType": "Raw",
            "Component": None,
            "BinMode": "NoRebin",
            "NHits": int(nhits_value),
            "OpHits": int(ophits_value),
            "AdjCl": int(adjcl_value),
            "Energy": no_rebin_energy.tolist(),
            "Significance": raw_no_rebin_significance.tolist(),
        }
    )
    hep_significance.append(
        {
            "Config": config,
            "Name": name,
            "EnergyLabel": energy,
            "Variable": saved_variable,
            "ProxyReference": reference_for_bins,
            "SpectrumType": "Smoothed",
            "Component": None,
            "BinMode": "NoRebin",
            "NHits": int(nhits_value),
            "OpHits": int(ophits_value),
            "AdjCl": int(adjcl_value),
            "Energy": no_rebin_energy.tolist(),
            "Significance": smoothed_no_rebin_significance.tolist(),
        }
    )
    if adaptive_energy.size > 0:
        hep_significance.append(
            {
                "Config": config,
                "Name": name,
                "EnergyLabel": energy,
                "Variable": saved_variable,
                "ProxyReference": reference_for_bins,
                "SpectrumType": "Raw",
                "Component": None,
                "BinMode": "AdaptiveRebin",
                "NHits": int(nhits_value),
                "OpHits": int(ophits_value),
                "AdjCl": int(adjcl_value),
                "Energy": adaptive_energy.tolist(),
                "Significance": raw_adaptive_significance.tolist(),
                "BinWidth": adaptive_width.tolist(),
            }
        )
        hep_significance.append(
            {
                "Config": config,
                "Name": name,
                "EnergyLabel": energy,
                "Variable": saved_variable,
                "ProxyReference": reference_for_bins,
                "SpectrumType": "Smoothed",
                "Component": None,
                "BinMode": "AdaptiveRebin",
                "NHits": int(nhits_value),
                "OpHits": int(ophits_value),
                "AdjCl": int(adjcl_value),
                "Energy": adaptive_energy.tolist(),
                "Significance": smoothed_adaptive_significance.tolist(),
                "BinWidth": adaptive_width.tolist(),
            }
        )

        raw_local_density = _density(raw_no_rebin_significance, no_rebin_width)
        smooth_local_density = _density(smoothed_no_rebin_significance, no_rebin_width)
        hep_significance.append(
            {
                "Config": config,
                "Name": name,
                "EnergyLabel": energy,
                "Variable": saved_variable,
                "ProxyReference": reference_for_bins,
                "SpectrumType": "Raw",
                "Component": None,
                "BinMode": "LocalDensity",
                "NHits": int(nhits_value),
                "OpHits": int(ophits_value),
                "AdjCl": int(adjcl_value),
                "Energy": no_rebin_energy.tolist(),
                "Significance": raw_local_density.tolist(),
                "BinWidth": no_rebin_width.tolist(),
            }
        )
        hep_significance.append(
            {
                "Config": config,
                "Name": name,
                "EnergyLabel": energy,
                "Variable": saved_variable,
                "ProxyReference": reference_for_bins,
                "SpectrumType": "Smoothed",
                "Component": None,
                "BinMode": "LocalDensity",
                "NHits": int(nhits_value),
                "OpHits": int(ophits_value),
                "AdjCl": int(adjcl_value),
                "Energy": no_rebin_energy.tolist(),
                "Significance": smooth_local_density.tolist(),
                "BinWidth": no_rebin_width.tolist(),
            }
        )

    for component, component_label, oscillation, mean_label, _, _, color in component_specs:
        comp_df = this_plot_df.loc[
            (this_plot_df["Component"] == component)
            * (this_plot_df["Oscillation"] == oscillation)
            * (this_plot_df["Mean"] == mean_label)
        ].copy()
        if comp_df.empty:
            continue
        counts = _safe_array(comp_df["Counts"].values[0])
        component_smoothing_config = get_component_smoothing_config(smoothing_config, component)
        smoothed_counts = smooth_threshold_slice(counts, 0, component_smoothing_config)
        hep_counts.append(
            {
                "Config": config,
                "Name": name,
                "EnergyLabel": energy,
                "Component": component,
                "SpectrumType": "Raw",
                "NHits": int(nhits_value),
                "OpHits": int(ophits_value),
                "AdjCl": int(adjcl_value),
                "Energy": hep_rebin_centers.tolist(),
                "Counts": (scale * counts).tolist(),
            }
        )
        hep_counts.append(
            {
                "Config": config,
                "Name": name,
                "EnergyLabel": energy,
                "Component": component,
                "SpectrumType": "Smoothed",
                "NHits": int(nhits_value),
                "OpHits": int(ophits_value),
                "AdjCl": int(adjcl_value),
                "Energy": hep_rebin_centers.tolist(),
                "Counts": (scale * smoothed_counts).tolist(),
            }
        )

if hep_counts:
    save_df(
        pd.DataFrame(hep_counts),
        data_path,
        config=args.config[0],
        name=args.name[0],
        subfolder=args.folder.lower(),
        filename=f"{args.config[0]}_{args.name[0]}_HEP_Counts",
        rm=args.rewrite,
        debug=args.debug,
    )

if hep_significance:
    save_df(
        pd.DataFrame(hep_significance),
        data_path,
        config=args.config[0],
        name=args.name[0],
        subfolder=args.folder.lower(),
        filename=f"{args.config[0]}_{args.name[0]}_HEP_Significance",
        rm=args.rewrite,
        debug=args.debug,
    )
