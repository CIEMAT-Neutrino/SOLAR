import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *

save_path = f"{root}/images/day-night"
data_path = f"{root}/data/analysis/day-night"

for this_path in [save_path, data_path]:
    if not os.path.exists(this_path):
        os.makedirs(this_path)

# Define flags for the analysis config and name with the python parser
parser = argparse.ArgumentParser(
    description="Plot the energy distribution of the particles"
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
    type=str,
    help="The configuration to load",
    default="hd_1x2x6_centralAPA",
)
parser.add_argument(
    "--name", type=str, help="The name of the configuration", default="marley"
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
    default=100,
)
parser.add_argument(
    "--energy",
    nargs="+",
    type=str,
    help="The energy for the analysis",
    default=["ClusterEnergy", "TotalEnergy", "SelectedEnergy", "SolarEnergy"],
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
    "--threshold", type=float, help="The threshold for the analysis", default=8.0
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
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()

analysis = args.analysis
reference = args.reference
config = args.config
name = args.name
folder = args.folder

configs = {config: [name]}

user_input = {
    "threshold": args.threshold,
    "rewrite": args.rewrite,
    "debug": args.debug,
}

day_night_counts = []
day_night_significance = []

for config in configs:
    info = json.loads(open(f"{root}/config/{config}/{config}_config.json").read())
    sigma_label = "highest"
    sigma = pickle.load(
        open(
            f"{info['PATH']}/DAYNIGHT/{args.folder.lower()}/{args.config}/{args.name}/{args.config}_{args.name}_{sigma_label}_DayNight.pkl",
            "rb",
        )
    )

    for idx, (name, energy, key) in enumerate(
        product(configs[config], args.energy, sigma)
    ):
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
                f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/background/{folder.lower()}/DAYNIGHT/{config}/{bkg}/{config}_{bkg}_{energy}_Rebin.pkl"
            )
            df_list.append(bkg_df)

        plot_df = pd.concat(df_list, ignore_index=True)

        sigmas_df = pd.read_pickle(
            f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/{analysis.upper()}/{folder.lower()}/{config}/{name}/{config}_{name}_{energy}_{analysis}_Results.pkl",
        )

        try:
            ref_plot = sigma[(config, name, energy)]
        except KeyError:
            rprint(
                f"[yellow][WARNING] Not found {sigma_label} for {config} {name} {energy}[/yellow]"
            )
            continue

        if args.nhits is not None:
            nhits = args.nhits
        else:
            rprint(f"Using optimized nhits {sigma[key]['NHits']}")
            nhits = int(sigma[key]["NHits"])

        if args.adjcls is not None:
            adjcl = args.adjcls
        else:
            rprint(f"Using optimized adjcl {sigma[key]['AdjCl']}")
            adjcl = int(sigma[key]["AdjCl"])

        if args.ophits is not None:
            ophits = args.ophits
        else:
            rprint(f"Using optimized ophits {sigma[key]['OpHits']}")
            ophits = int(sigma[key]["OpHits"])

        rprint(
            f"Evaluating {sigma_label} for min#Hits {nhits:.0f}, min#OpHits {ophits:.0f}, max#AdjCl {adjcl:.0f}"
        )

        this_plot_df = plot_df.loc[
            # (plot_df["EnergyLabel"] == energy)
            (plot_df["NHits"] == nhits)
            * (plot_df["OpHits"] == ophits)
            * (plot_df["AdjCl"] == adjcl)
        ].copy()

        print(
            this_plot_df.explode("Counts")
            .groupby(["Component", "Oscillation", "Mean"])["Counts"]
            .sum()
        )

        plot_sigmas = sigmas_df.loc[
            (sigmas_df["Config"] == config) * (sigmas_df["Name"] == name)
        ].copy()

        plot_sigmas = plot_sigmas.loc[
            # (plot_sigmas["EnergyLabel"] == energy)
            (plot_sigmas["NHits"] == nhits)
            * (plot_sigmas["OpHits"] == ophits)
            * (plot_sigmas["AdjCl"] == adjcl)
        ].copy()

        # display(ref_plot)
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                f"min#Hits {nhits:.0f}, min#OpHits {ophits:.0f}, max#AdjCl {adjcl:.0f}",
                "Significance",
            ),
        )

        for jdx, (
            component,
            component_label,
            osc,
            mean,
            legend_group,
            legend_group_title,
            color,
            dash,
        ) in enumerate(
            zip(
                ["Solar", "Solar", "neutron", "gamma"],
                ["Solar Day", "Solar Night", "Neutron", "Gamma"],
                ["Osc", "Osc", "Truth", "Truth"],
                ["Day", "Night", "Mean", "Mean"],
                [0, 0, 1, 1],
                ["Signal", "Signal", "Background", "Background"],
                [compare[1], compare[0], "rgb(15,133,84)", "black"],
                ["dash", "dash", "solid", "solid"],
            )
        ):

            comp_df = this_plot_df.loc[
                (this_plot_df["Component"] == component)
                * (this_plot_df["Oscillation"] == osc)
                * (this_plot_df["Mean"] == mean)
            ].copy()

            # If the dataframe is empty, skip the iteration
            if comp_df.empty:
                rprint(
                    f"[yellow][WARNING] Not found {component_label} for {config} {name} {energy}[/yellow]"
                )
                continue

            # rprint(args.exposure * comp_df["Counts/Energy"].values[0])
            fig.add_trace(
                go.Scatter(
                    x=comp_df["Energy"].values[0],
                    y=args.exposure * np.array(comp_df["Counts/Energy"].values[0]),
                    name=f"{component_label}",
                    mode="lines",
                    error_y=dict(
                        type="data",
                        array=args.exposure * np.array(comp_df["Error"].values[0]),
                    ),
                    line_shape="hvh",
                    line=dict(
                        color=color,
                    ),
                    legend="legend",
                    legendgroup=legend_group,
                    legendgrouptitle=dict(text=f"{legend_group_title}"),
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

            day_night_counts.append(
                {
                    "Geometry": info["GEOMETRY"],
                    "Config": config,
                    "Name": name,
                    "Exposure": args.exposure,
                    "Component": component_label,
                    "Energy": comp_df["Energy"].values[0],
                    "Counts": np.array(comp_df["Counts/Energy"].values[0]),
                    "CountsError": np.array(comp_df["Error"].values[0]),
                }
            )

        if plot_sigmas.empty:
            rprint(
                f"[yellow][WARNING] Not found {sigma_label} for {config} {name} {energy}[/yellow]"
            )
            continue

        plot_sign_fid = plot_sigmas.copy()

        if len(plot_sigmas) > 1:
            print(plot_sigmas[reference])
            print(plot_sigmas[reference].max())
            plot_sign_fid = plot_sigmas.loc[
                plot_sigmas[reference] == plot_sigmas[reference].max()
            ]

        day_night_significance.append(
            {
                "Geometry": info["GEOMETRY"],
                "Config": config,
                "Name": name,
                "Exposure": plot_sign_fid["Exposure"].values[0],
                "Significance": plot_sign_fid[reference].values[0],
                "SignificanceError+": np.subtract(
                    plot_sign_fid[reference + "+Error"].values[0],
                    plot_sign_fid[reference].values[0],
                ),
                "SignificanceError-": np.subtract(
                    plot_sign_fid[reference].values[0],
                    plot_sign_fid[reference + "-Error"].values[0],
                ),
            }
        )

        fig.add_trace(
            go.Scatter(
                x=plot_sign_fid["Exposure"].values[0],
                y=plot_sign_fid[reference].values[0],
                name=f"{reference}",
                mode="lines",
                line=dict(color="black"),
                legend="legend2",
                legendgroup="Significance",
                legendgrouptitle=dict(text="Significance"),
                showlegend=True,
                # Render the line with better resolution
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=plot_sign_fid["Exposure"].values[0],
                y=plot_sign_fid[reference + "+Error"].values[0],
                mode="lines",
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=plot_sign_fid["Exposure"].values[0],
                y=plot_sign_fid[reference + "-Error"].values[0],
                marker=dict(color="#444"),
                line=dict(width=0),
                mode="lines",
                fillcolor="rgba(68, 68, 68, 0.3)",
                fill="tonexty",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        fig = format_coustom_plotly(
            fig,
            tickformat=(".1f", ".0e"),
            add_units=False,
            legend_title=f"{energy}",
            title=f"Day-Night - {folder} Sample - {config} {name}",
            matches=(None, None),
        )

        fig.update_yaxes(
            type="log",
            tickformat=".0e",
            # Reduce number of ticks
            dtick=1,
            range=[
                np.log10(args.exposure * 1e-3),
                np.log10(args.exposure * 1e4),
            ],
            title=f"Counts per Energy ({args.exposure} kT·year·MeV)⁻¹",
            row=1,
            col=1,
        )
        fig.update_xaxes(
            range=[6.75, 26],
            title=f"Reconstructed Neutrino Energy (MeV)",
            row=1,
            col=1,
        )
        fig.update_yaxes(
            tickformat=".1f",
            dtick=1,
            range=[0, 4],
            title=f"Significance (σ)",
            row=1,
            col=2,
        )
        exposure_title = f"Exposure (kT·year)"

        fig.update_xaxes(
            range=[-10, args.exposure],
            zeroline=False,
            title=exposure_title,
            row=1,
            col=2,
        )

        fig.add_vline(
            x=user_input["threshold"],
            line_dash="dash",
            line_color="grey",
            row=1,
            col=1,
            annotation=dict(text="Threshold", showarrow=False),
            annotation_position="bottom right",
        )
        for sigma_line, cl in zip([1, 2, 3], [0.6827, 0.9545, 0.9973]):
            fig.add_hline(
                y=sigma_line, line_dash="dash", line_color="black", row=1, col=2
            )
            fig.add_annotation(
                x=2,
                y=sigma_line + 0.2,
                text=f"{100*cl:.2f}% CL",
                showarrow=False,
                row=1,
                col=2,
            )

        fig.update_layout(
            legend_title_text="",
            legend=dict(y=1, x=0.30, font=dict(size=16)),
            legend2=dict(y=0.0, x=0.84, font=dict(size=16), bgcolor="rgba(0,0,0,0)"),
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
            f"{save_path}/{folder.lower()}",
            config,
            None,
            filename=figure_name,
            rm=user_input["rewrite"],
            debug=user_input["debug"],
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
                rm=user_input["rewrite"],
                debug=user_input["debug"],
            )
