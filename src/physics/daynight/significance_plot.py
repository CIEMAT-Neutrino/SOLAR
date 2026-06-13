import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *

analysis_info = load_analysis_info(str(root))

save_path = f"{root}/images/analysis/day-night"
data_path = f"{analysis_info['PATH']}/DAYNIGHT"
for this_path in [save_path]:
    if not os.path.exists(this_path):
        os.makedirs(this_path)

parser = argparse.ArgumentParser(
    description="Plot the energy distribution of the particles"
)
parser.add_argument("--analysis", type=str, default="DayNight")
parser.add_argument(
    "--reference", type=str, choices=["Gaussian", "Asimov"], default="Gaussian"
)
parser.add_argument("--config", nargs="+", type=str, default=["hd_1x2x6_centralAPA"])
parser.add_argument("--name", nargs="+", type=str, default=["marley"])
parser.add_argument("--folder", type=str, default="Reduced")
parser.add_argument("--signal_uncertainty", type=float, default=0.00)
parser.add_argument("--background_uncertainty", type=float, default=0.02)
parser.add_argument("--exposure", type=float, default=30)
parser.add_argument(
    "--energy",
    nargs="+",
    type=str,
    default=["ClusterEnergy", "TotalEnergy", "SelectedEnergy", "SolarEnergy"],
)
parser.add_argument("--nhits", type=int, default=None)
parser.add_argument("--ophits", type=int, default=None)
parser.add_argument("--adjcls", type=int, default=None)
parser.add_argument("--stacked", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument(
    "--threshold",
    type=float,
    default=get_analysis_threshold(str(root), "DAYNIGHT", stage="SIGNIFICANCE", fallback=0.0),
)
parser.add_argument("--show_bin_labels", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--bin_label_digits", type=int, default=1)
parser.add_argument("--bin_label_stride", type=int, default=1)
parser.add_argument("--bin_label_y_offset_fraction", type=float, default=0.1)
parser.add_argument("--zoom", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
args = parser.parse_args()

smoothing_config = get_smoothing_config(
    str(root), analysis_name="DAYNIGHT", dimensions="1d", stage="significance"
)
day_night_counts = []
day_night_significance = []
smoothing_info = smoothing_metadata(smoothing_config)
for config, name, energy in product(args.config, args.name, args.energy):
    info = json.loads(open(f"{root}/config/{config}/{config}_config.json").read())
    sigma = pickle.load(
        open(
            f"{info['PATH']}/DAYNIGHT/{args.folder.lower()}/{config}/{args.name[0]}/{config}_{args.name[0]}_highest_DayNight.pkl",
            "rb",
        )
    )
    sigmas_df = pd.read_pickle(
        f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/{args.analysis.upper()}/{args.folder.lower()}/{config}/{name}/{config}_{name}_{energy}_{args.analysis}_Results.pkl",
    )
    significance_bins_path = (
        f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/{args.analysis.upper()}/"
        f"{args.folder.lower()}/{config}/{name}/{config}_{name}_{energy}_DayNight_SignificanceBins.pkl"
    )
    if not os.path.exists(significance_bins_path):
        rprint(
            f"[yellow][WARNING][/yellow] Missing per-bin significance payload for {config} {name} {energy}."
        )
        continue
    significance_bins_df = pd.read_pickle(significance_bins_path)
    detector_mass = get_full_detector_mass(config, info)

    plot_df = pd.read_pickle(
        f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/signal/{args.folder.lower()}/DAYNIGHT/{config}/{name}/{config}_{name}_{energy}_Rebin.pkl"
    )
    background_samples = []
    for bkg, filepath in load_available_background_dataframes(
        str(root), "DAYNIGHT", args.folder, config, energy
    ):
        bkg_df = pd.read_pickle(filepath)
        plot_df = pd.concat([plot_df, bkg_df], ignore_index=True)
        background_samples.append(bkg)

    try:
        ref_plot = sigma[(config, name, energy)]
    except KeyError:
        rprint(
            f"[yellow][WARNING] Not found highest for {config} {name} {energy}[/yellow]"
        )
        continue

    nhits_value = args.nhits if args.nhits is not None else int(ref_plot["NHits"])
    adjcl_value = args.adjcls if args.adjcls is not None else int(ref_plot["AdjCl"])
    ophits_value = args.ophits if args.ophits is not None else int(ref_plot["OpHits"])

    this_plot_df = plot_df.loc[
        (plot_df["NHits"] == nhits_value)
        * (plot_df["OpHits"] == ophits_value)
        * (plot_df["AdjCl"] == adjcl_value)
    ].copy()
    if this_plot_df.empty:
        continue

    plot_sigmas = sigmas_df.loc[
        (sigmas_df["Config"] == config)
        * (sigmas_df["Name"] == name)
        * (sigmas_df["NHits"] == nhits_value)
        * (sigmas_df["OpHits"] == ophits_value)
        * (sigmas_df["AdjCl"] == adjcl_value)
    ].copy()
    if plot_sigmas.empty:
        rprint(
            f"[yellow][WARNING][/yellow] Missing precomputed significance payload for {config} {name} {energy} NHits={nhits_value} OpHits={ophits_value} AdjCl={adjcl_value}."
        )
        continue

    energy_axis = np.asarray(this_plot_df["Energy"].values[0], dtype=float)
    threshold_idx = 0
    if args.threshold is not None:
        threshold_idx = np.where(energy_axis > args.threshold)[0][0]
    bin_width = float(np.median(np.diff(energy_axis))) if len(energy_axis) > 1 else 1.0

    selected_bins = significance_bins_df.loc[
        (significance_bins_df["Config"] == config)
        * (significance_bins_df["Name"] == name)
        * (significance_bins_df["EnergyLabel"] == energy)
        * (significance_bins_df["NHits"] == int(nhits_value))
        * (significance_bins_df["OpHits"] == int(ophits_value))
        * (significance_bins_df["AdjCl"] == int(adjcl_value))
    ].copy()
    if selected_bins.empty:
        rprint(
            f"[yellow][WARNING][/yellow] Missing per-bin significance payload for {config} {name} {energy} NHits={nhits_value} OpHits={ophits_value} AdjCl={adjcl_value}."
        )
        continue

    selected_bins = selected_bins.sort_values("BinIndex")
    stored_energy_axis = np.asarray(selected_bins["RecoEnergy"].values, dtype=float)
    raw_significance = np.nan_to_num(
        np.asarray(selected_bins["RawGaussian"].values, dtype=float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    smoothed_significance = np.nan_to_num(
        np.asarray(selected_bins["Gaussian"].values, dtype=float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0,
        subplot_titles=(
            f"{energy}, min#Hits {nhits_value:.0f}, min#OpHits {ophits_value:.0f}, max#AdjCl {adjcl_value:.0f}",
            "",
        ),
    )

    background_raw = np.zeros(len(energy_axis), dtype=float)
    background_smoothed = np.zeros(len(energy_axis), dtype=float)
    signal_day_raw = None
    signal_day_smoothed = None
    signal_night_raw = None
    signal_night_smoothed = None
    positive_count_values = []

    component_specs = [
        ("Solar", "Solar Day", "Osc", "Day", 0, "Signal", compare[1]),
        ("Solar", "Solar Night", "Osc", "Night", 0, "Signal", compare[0]),
    ]
    for bkg in background_samples:
        style = get_background_style(str(root), bkg)
        component_specs.append(
            (
                bkg,
                style.get("label", bkg).title(),
                "Truth",
                "Mean",
                1,
                "Background",
                style.get("color", "grey"),
            )
        )

    for (
        component,
        component_label,
        osc,
        mean,
        legend_group,
        legend_group_title,
        color,
    ) in component_specs:
        comp_df = this_plot_df.loc[
            (this_plot_df["Component"] == component)
            * (this_plot_df["Oscillation"] == osc)
            * (this_plot_df["Mean"] == mean)
        ].copy()
        if comp_df.empty:
            continue

        counts = np.asarray(comp_df["Counts"].values[0], dtype=float)
        errors = np.asarray(comp_df["Error"].values[0], dtype=float)
        mc_counts = (
            np.asarray(comp_df["MCCounts"].values[0], dtype=float)
            if "MCCounts" in comp_df.columns
            else None
        )
        component_smoothing_config = get_component_smoothing_config(
            smoothing_config, component
        )
        raw_counts_per_energy = np.nan_to_num(
            detector_mass
            * args.exposure
            * np.asarray(comp_df["Counts/Energy"].values[0], dtype=float),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        smoothed_counts = smooth_threshold_slice(
            counts, threshold_idx, component_smoothing_config
        )
        smoothed_errors = errors.copy()
        smoothed_errors[threshold_idx:] = smooth_histogram_errors(
            errors[threshold_idx:],
            component_smoothing_config,
            counts=counts[threshold_idx:],
            mc_counts=mc_counts[threshold_idx:] if mc_counts is not None else None,
        )
        smoothed_counts_per_energy = np.nan_to_num(
            detector_mass * args.exposure * smoothed_counts / bin_width,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        smoothed_errors_per_energy = np.nan_to_num(
            detector_mass * args.exposure * smoothed_errors / bin_width,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        positive_count_values.extend(raw_counts_per_energy[raw_counts_per_energy > 0])
        positive_count_values.extend(smoothed_counts_per_energy[smoothed_counts_per_energy > 0])

        fig.add_trace(
            go.Scatter(
                x=energy_axis,
                y=raw_counts_per_energy,
                name=component_label,
                mode="lines",
                line_shape="hvh",
                line=dict(color=color, width=2, dash="dot"),
                opacity=0.45,
                legend="legend",
                legendgroup=legend_group,
                legendgrouptitle=dict(text=f"{legend_group_title}"),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=energy_axis,
                y=smoothed_counts_per_energy,
                name=component_label,
                mode="lines",
                error_y=dict(type="data", array=smoothed_errors_per_energy),
                line_shape="hvh",
                line=dict(color=color, width=3),
                legend="legend",
                legendgroup=legend_group,
                legendgrouptitle=dict(text=f"{legend_group_title}"),
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        for counts_per_energy, errors_per_energy, spectrum_type in [
            (raw_counts_per_energy, detector_mass * args.exposure * errors, "Raw"),
            (smoothed_counts_per_energy, smoothed_errors_per_energy, "Smoothed"),
        ]:
            significance_values = None
            if component_label == "Solar Day":
                sig_source = smoothed_significance if spectrum_type == "Smoothed" else raw_significance
                energy_len = len(energy_axis)
                bin_len = len(sig_source)
                if bin_len >= energy_len:
                    significance_values = sig_source[:energy_len]
                else:
                    significance_values = np.pad(
                        sig_source,
                        (0, energy_len - bin_len),
                        mode="constant",
                        constant_values=np.nan,
                    )

            day_night_counts.append(
                {
                    "Geometry": info["GEOMETRY"],
                    "Config": config,
                    "Name": name,
                    "Exposure": args.exposure,
                    "SpectrumType": spectrum_type,
                    "Component": component_label,
                    "Energy": energy_axis,
                    "Counts": counts_per_energy,
                    "CountsError": errors_per_energy,
                    # Add the per bin significance values for this spectrum type
                    "Significance": significance_values,
                    "SignificanceLabel": f"Gaussian {spectrum_type}" if component_label == "Solar Day" else None,
                }
            )

        for spectrum_type, values in [
            ("Raw", raw_counts_per_energy),
            ("Smoothed", smoothed_counts_per_energy),
        ]:
            day_night_significance.append(
                {
                    "Geometry": info["GEOMETRY"],
                    "Config": config,
                    "Name": name,
                    "Exposure": args.exposure,
                    "Component": component_label,
                    "SpectrumType": spectrum_type,
                    "Energy": energy_axis,
                    "Counts": values,
                    "CountsError": (
                        smoothed_errors_per_energy
                        if spectrum_type == "Smoothed"
                        else detector_mass * args.exposure * errors
                    ),
                    "Significance": None,
                    **smoothing_info,
                }
            )

        if osc == "Truth":
            background_raw += counts
            background_smoothed += smoothed_counts
        else:
            if mean == "Day":
                signal_day_raw = counts
                signal_day_smoothed = smoothed_counts
            else:
                signal_night_raw = counts
                signal_night_smoothed = smoothed_counts

    if signal_day_raw is None or signal_night_raw is None:
        continue

    for label, significance, dash, showlegend in [
        (args.reference, raw_significance, "dot", False),
        (args.reference, smoothed_significance, "solid", True),
    ]:
        add_significance_series_trace(
            fig,
            x=stored_energy_axis,
            y=significance,
            name_prefix=label,
            row=2,
            col=1,
            color="black",
            width=3 if dash == "solid" else 2,
            dash=dash,
            legend="legend2",
            legendgroup="Significance",
            legendgrouptitle="Significance",
            showlegend=showlegend,
            append_total=True,
            total_digits=1,
        )
        if args.show_bin_labels and dash == "solid":
            add_significance_bin_labels(
                fig,
                x=stored_energy_axis,
                y=significance,
                row=2,
                col=1,
            text_prefix="",
                digits=args.bin_label_digits,
                label_stride=args.bin_label_stride,
                show_zero=False,
                color="black",
                font_size=10,
                textposition="top center",
                y_offset_fraction=args.bin_label_y_offset_fraction,
            )
        day_night_significance.append(
            {
                "Geometry": info["GEOMETRY"],
                "Config": config,
                "Name": name,
                "Exposure": args.exposure,
                "Component": None,
                "SpectrumType": "Raw" if dash == "dot" else "Smoothed",
                "Variable": label,
                "Energy": stored_energy_axis,
                "Counts": None,
                "CountsError": None,
                "Significance": significance,
                **smoothing_info,
            }
        )

    add_histogram_style_legend_traces(
        fig,
        row=1,
        col=1,
        legend="legend",
    )

    fig = format_coustom_plotly(
        fig,
        tickformat=(".1f", ".0e"),
        add_units=False,
        title=f"Day-Night Asymmetry - {args.folder} - {config}",
        matches=("x", None),
        figsize=(800, 600),
    )
    fig.update_layout(
        legend2=dict(
            x=0.79,
            y=0.14,
            font=dict(size=12),
        )
    )
    fig.update_yaxes(
        type="log",
        tickformat=".0e",
        dtick=1,
        range=[
            np.floor(np.log10(max(min(positive_count_values), 1e-6))),
            np.ceil(np.log10(max(positive_count_values))),
        ]
        if args.zoom and positive_count_values
        else [-2, np.log10(detector_mass * args.exposure * 1e4)],
        title=f"Counts ({args.exposure:.0f} year·MeV)⁻¹",
        row=1,
        col=1,
    )
    fig.update_yaxes(
        tickformat=".0f",
        dtick=1,
        range=[0, max(1.0, 1.1 * max(np.max(raw_significance), np.max(smoothed_significance), 1.0))]
        if args.zoom
        else [0, 6],
        title="Significance (σ)",
        row=2,
        col=1,
    )
    fig.update_xaxes(showticklabels=False, range=[6.75, 26], row=1, col=1)
    fig.update_xaxes(
        range=[6.75, 26], title="Reconstructed Neutrino Energy (MeV)", row=2, col=1
    )

    if args.threshold is not None:
        fig.add_vline(
            x=args.threshold,
            line_dash="dash",
            line_color="grey",
            annotation=dict(text="Threshold", showarrow=False),
            annotation_position="bottom right",
        )

    figure_name = f"{energy}_DayNight_Significance"
    if args.nhits is not None or args.ophits is not None or args.adjcls is not None:
        figure_name += (
            f"_NHits{nhits_value:.0f}_OpHits{ophits_value:.0f}_AdjCl{adjcl_value:.0f}"
        )
    if args.exposure is not None:
        figure_name += f"_Exposure_{args.exposure:.0f}"
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

    for df, df_name in zip(
        [pd.DataFrame(day_night_counts), pd.DataFrame(day_night_significance)],
        ["DayNight_Counts", "DayNight_Significance"],
    ):
        save_df(
            df,
            data_path,
            config,
            name,
            subfolder=args.folder.lower(),
            filename=df_name,
            rm=args.rewrite,
            debug=args.debug,
        )
