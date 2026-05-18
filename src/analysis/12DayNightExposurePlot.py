import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *

save_path = f"{root}/images/analysis/day-night"
data_path = f"{root}/data/analysis/day-night"

for this_path in [save_path, data_path]:
    if not os.path.exists(this_path):
        os.makedirs(this_path)

# Define flags for the analysis config and name with the python parser
parser = argparse.ArgumentParser(
    description="Plot the day-night analysis for a given configuration and name and plot the results as a function of exposure"
)
parser.add_argument(
    "--analysis",
    type=str,
    help="The analysis of the configuration. Supporting 'DayNight' and 'HEP'",
    default="DayNight",
)
parser.add_argument(
    "--reference",
    type=str,
    help="The name of the reference column",
    choices=["Gaussian+Error", "Gaussian", "Gaussian-Error"],
    default="Gaussian",
    required=False,
)
parser.add_argument(
    "--config",
    nargs="+",
    type=str,
    help="The configuration to load",
    default=["hd_1x2x6_centralAPA"],
)
parser.add_argument(
    "--name",
    nargs="+",
    type=str,
    help="The name of the configuration",
    default=["marley"],
)
parser.add_argument(
    "--folder",
    type=str,
    help="The name of the results folder",
    default="Nominal",
    choices=["Reduced", "Truncated", "Nominal"],
)
parser.add_argument(
    "--exposure",
    type=float,
    help="The exposure for the analysis",
    default=30,
)
parser.add_argument(
    "--energy",
    nargs="+",
    type=str,
    help="The energy for the analysis",
    default=["ClusterEnergy", "TotalEnergy", "SelectedEnergy", "SolarEnergy"],
    choices=["ClusterEnergy", "TotalEnergy", "SelectedEnergy", "SolarEnergy"],
)
parser.add_argument(
    "--nhits",
    type=int,
    help="The min niht cut for the analysis",
    default=None,
)
parser.add_argument(
    "--ophits",
    type=int,
    help="The min ophit cut for the analysis",
    default=None,
)
parser.add_argument(
    "--adjcls",
    type=int,
    help="The max adjcl cut for the analysis",
    default=None,
)
parser.add_argument(
    "--threshold",
    type=float,
    help="The threshold for the analysis",
    default=get_analysis_threshold(str(root), "DAYNIGHT", stage="SIGNIFICANCE", fallback=0.0),
)
parser.add_argument(
    "--signal_uncertainty",
    type=float,
    help="The signal uncertainty for the analysis",
    default=0.04,
)
parser.add_argument(
    "--background_uncertainty",
    type=float,
    help="The background uncertainty for the analysis",
    default=0.02,
)
parser.add_argument("--stacked", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()

smoothing_config = get_smoothing_config(
    str(root), analysis_name="DAYNIGHT", dimensions="1d", stage="significance"
)
day_night_counts = []
day_night_exposure = []

for config, name, energy in product(args.config, args.name, args.energy):
    info = json.loads(open(f"{root}/config/{config}/{config}_config.json").read())
    detector_mass = get_full_detector_mass(config, info)

    df_list = []
    signal_df = pd.read_pickle(
        f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/signal/{args.folder.lower()}/DAYNIGHT/{config}/{name}/{config}_{name}_{energy}_Rebin.pkl"
    )
    df_list.append(signal_df)
    for bkg, bkg_label in [
        ("neutron", "neutron"),
        ("gamma", "gamma"),
    ]:
        bkg_df = pd.read_pickle(
            f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/background/{args.folder.lower()}/DAYNIGHT/{config}/{bkg}/{config}_{bkg}_{energy}_Rebin.pkl"
        )
        df_list.append(bkg_df)

    plot_df = pd.concat(df_list, ignore_index=True)

    sigmas_df = pd.read_pickle(
        f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/{args.analysis.upper()}/{args.folder.lower()}/{config}/{name}/{config}_{name}_{energy}_{args.analysis}_Results.pkl",
    )

    for sigma_name, sigma_label in zip(
        ["highest"],
        ["Highest"],
    ):
        sigma_path = (
            f"{info['PATH']}/DAYNIGHT/{args.folder.lower()}/{config}/{name}/{config}_{name}_{sigma_name}_DayNight.pkl"
        )
        try:
            sigma = pickle.load(open(sigma_path, "rb"))
        except (EOFError, pickle.UnpicklingError) as exc:
            raise RuntimeError(
                f"Failed to load sigma summary from {sigma_path}. The file is likely truncated from an earlier failed write. "
                "Re-run 0ZBestSigmas.py after removing the broken pickle or with --rewrite."
            ) from exc

        try:
            ref_plot = sigma[(config, name, energy)]
        except KeyError:
            rprint(
                f"[yellow][WARNING] Not found {sigma_label} for {config} {name} {energy}[/yellow]"
            )
            continue

        if args.nhits is not None:
            ref_plot["NHits"] = args.nhits
        if args.ophits is not None:
            ref_plot["OpHits"] = args.ophits
        if args.adjcls is not None:
            ref_plot["AdjCl"] = args.adjcls

        if args.debug:
            rprint(
                f"Evaluating {sigma_label} for min#Hits {ref_plot['NHits']:.0f}, min#OpHits {ref_plot['OpHits']:.0f}, max#AdjCl {ref_plot['AdjCl']:.0f}"
            )

        this_plot_df = plot_df.loc[
            (plot_df["NHits"] == ref_plot["NHits"])
            * (plot_df["OpHits"] == ref_plot["OpHits"])
            * (plot_df["AdjCl"] == ref_plot["AdjCl"])
        ].copy()

        if args.debug:
            print(
                this_plot_df.explode("Counts")
                .groupby(["Component", "Oscillation", "Mean"])["Counts"]
                .sum()
            )

        plot_sigmas = sigmas_df.loc[
            (sigmas_df["Config"] == config) * (sigmas_df["Name"] == name)
        ].copy()

        plot_sigmas = plot_sigmas.loc[
            (plot_sigmas["NHits"] == ref_plot["NHits"])
            * (plot_sigmas["OpHits"] == ref_plot["OpHits"])
            * (plot_sigmas["AdjCl"] == ref_plot["AdjCl"])
        ].copy()

        # display(ref_plot)
        fig = make_subplots(
            rows=1,
            cols=1,
            subplot_titles=(f"{energy}, min#Hits {ref_plot['NHits']:.0f}, min#OpHits {ref_plot['OpHits']:.0f}, max#AdjCl {ref_plot['AdjCl']:.0f}",),
        )

        # for jdx, (
        #     component,
        #     component_label,
        #     osc,
        #     mean,
        #     legend_group,
        #     legend_group_title,
        #     color,
        #     dash,
        # ) in enumerate(
        #     zip(
        #         ["Solar", "Solar", "neutron", "gamma"],
        #         ["Solar Day", "Solar Night", "Neutron", "Gamma"],
        #         ["Osc", "Osc", "Truth", "Truth"],
        #         ["Day", "Night", "Mean", "Mean"],
        #         [0, 0, 1, 1],
        #         ["Signal", "Signal", "Background", "Background"],
        #         [compare[1], compare[0], "rgb(15,133,84)", "black"],
        #         ["dash", "dash", "solid", "solid"],
        #     )
        # ):

        #     comp_df = this_plot_df.loc[
        #         (this_plot_df["Component"] == component)
        #         * (this_plot_df["Oscillation"] == osc)
        #         * (this_plot_df["Mean"] == mean)
        #     ].copy()

        #     if comp_df.empty:
        #         rprint(
        #             f"[yellow][WARNING] Not found {component_label} for {config} {name} {energy}[/yellow]"
        #         )
        #         continue

        #     energy_axis = np.asarray(comp_df["Energy"].values[0], dtype=float)
        #     counts = np.asarray(comp_df["Counts"].values[0], dtype=float)
        #     errors = np.asarray(comp_df["Error"].values[0], dtype=float)
        #     mc_counts = np.asarray(comp_df["MCCounts"].values[0], dtype=float) if "MCCounts" in comp_df.columns else None
        #     component_smoothing_config = get_component_smoothing_config(smoothing_config, component)
        #     raw_counts_per_energy = (
        #         detector_mass * args.exposure * np.asarray(comp_df["Counts/Energy"].values[0], dtype=float)
        #     )
        #     threshold_idx = np.where(energy_axis > args.threshold)[0][0]
        #     smoothed_counts = smooth_threshold_slice(counts, threshold_idx, component_smoothing_config)
        #     smoothed_errors = errors.copy()
        #     smoothed_errors[threshold_idx:] = smooth_histogram_errors(
        #         errors[threshold_idx:],
        #         component_smoothing_config,
        #         counts=counts[threshold_idx:],
        #         mc_counts=mc_counts[threshold_idx:] if mc_counts is not None else None,
        #     )
        #     bin_width = float(np.median(np.diff(energy_axis))) if len(energy_axis) > 1 else 1.0
        #     smoothed_counts_per_energy = detector_mass * args.exposure * smoothed_counts / bin_width
        #     smoothed_errors_per_energy = detector_mass * args.exposure * smoothed_errors / bin_width

            # fig.add_trace(
            #     go.Scatter(
            #         x=energy_axis,
            #         y=raw_counts_per_energy,
            #         name=component_label,
            #         mode="lines",
            #         line_shape="hvh",
            #         line=dict(color=color, width=2, dash="dot"),
            #         opacity=0.45,
            #         legend="legend",
            #         legendgroup=legend_group,
            #         legendgrouptitle=dict(text=f"{legend_group_title}"),
            #         showlegend=False,
            #     ),
            #     row=1,
            #     col=1,
            # )
            # fig.add_trace(
            #     go.Scatter(
            #         x=energy_axis,
            #         y=smoothed_counts_per_energy,
            #         name=component_label,
            #         mode="lines",
            #         error_y=dict(
            #             type="data",
            #             array=smoothed_errors_per_energy,
            #         ),
            #         line_shape="hvh",
            #         line=dict(color=color),
            #         legend="legend",
            #         legendgroup=legend_group,
            #         legendgrouptitle=dict(text=f"{legend_group_title}"),
            #         showlegend=True,
            #     ),
            #     row=1,
            #     col=1,
            # )

            # day_night_counts.append(
            #     {
            #         "Geometry": info["GEOMETRY"],
            #         "Config": config,
            #         "Name": name,
            #         "Exposure": args.exposure,
            #         "Component": component_label,
            #         "Energy": energy_axis,
            #         "Counts": smoothed_counts_per_energy,
            #         "CountsError": smoothed_errors_per_energy,
            #     }
            # )

        if plot_sigmas.empty:
            rprint(
                f"[yellow][WARNING] Not found {sigma_label} for {config} {name} {energy}[/yellow]"
            )
            continue

        exposure_values = np.asarray(plot_sigmas["Exposure"].values[0], dtype=float)
        raw_significance = np.nan_to_num(
            np.asarray(plot_sigmas["Raw" + args.reference].values[0], dtype=float),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        smoothed_significance = np.nan_to_num(
            np.asarray(plot_sigmas[args.reference].values[0], dtype=float),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        significance_upper = np.nan_to_num(
            np.asarray(plot_sigmas[args.reference + "+Error"].values[0], dtype=float),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        significance_lower = np.nan_to_num(
            np.asarray(plot_sigmas[args.reference + "-Error"].values[0], dtype=float),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        day_night_exposure.append(
            {
                "Geometry": info["GEOMETRY"],
                "Config": config,
                "Name": name,
                "Exposure": exposure_values,
                "Significance": smoothed_significance,
                "RawSignificance": raw_significance,
                "SignificanceError+": np.subtract(
                    significance_upper,
                    smoothed_significance,
                ),
                "SignificanceError-": np.subtract(
                    smoothed_significance,
                    significance_lower,
                ),
            }
        )

        fig.add_trace(
            go.Scatter(
                x=exposure_values,
                y=raw_significance,
                name=args.reference,
                mode="lines",
                line=dict(color="black", width=2, dash="dot"),
                legendgroup=args.reference,
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=exposure_values,
                y=smoothed_significance,
                name=args.reference,
                mode="lines",
                line=dict(color="black"),
                legendgroup=args.reference,
                legendgrouptitle=dict(text="Significance"),
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=exposure_values,
                y=significance_upper,
                mode="lines",
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=exposure_values,
                y=significance_lower,
                marker=dict(color="#444"),
                line=dict(width=0),
                mode="lines",
                fillcolor="rgba(68, 68, 68, 0.3)",
                fill="tonexty",
                showlegend=False,
            ),
            row=1,
            col=1,
        )


        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name="Raw",
                line=dict(color="black", width=2, dash="dot"),
                legendgroup="linestyle",
                legendgrouptitle=dict(text="Data"),
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name="Smoothed",
                line=dict(color="black", width=3),
                legendgroup="linestyle",
                legendgrouptitle=dict(text="Data"),
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        fig = format_coustom_plotly(
            fig,
            tickformat=(".1f", ".0e"),
            add_units=False,
            title=f"Day-Night Asymmetry - {args.folder} - {config}",
            matches=(None, None),
            legend=dict(font=dict(size=14), bgcolor="rgba(255,255,255,0.7)"),
        )

        fig.update_yaxes(
            tickformat=".1f",
            dtick=1,
            range=[0, 4],
            title=f"Significance (σ)",
            row=1,
            col=1,
        )

        fig.update_xaxes(
            range=[-1, args.exposure],
            zeroline=False,
            title=f"Exposure (year)",
            row=1,
            col=1,
        )

        for sigma_line, cl in zip([1, 2, 3], [0.6827, 0.9545, 0.9973]):
            fig.add_hline(y=sigma_line, line_dash="dash", line_color="black")
            fig.add_annotation(
                x=2,
                y=sigma_line + 0.2,
                text=f"{100*cl:.2f}% CL",
                showarrow=False,
                row=1,
                col=1,
            )

        figure_name = f"{energy}_DayNight_Exposure"

        if (
            args.nhits is not None
            and args.ophits is not None
            and args.adjcls is not None
        ):
            figure_name += (
                f"_NHits{args.nhits:.0f}_OpHits{args.ophits:.0f}_AdjCl{args.adjcls:.0f}"
            )

        if args.threshold is not None:
            figure_name += f"_Threshold_{args.threshold:.0f}"

        if args.stacked:
            figure_name += "_Stacked"

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
            [pd.DataFrame(day_night_exposure)],
            ["DayNight_Exposure"],
        ):
            save_df(
                df,
                data_path,
                config,
                name,
                subfolder=args.folder.lower(),
                filename=df_name,
                rm=args.rewrite,
                debug=True,
            )
