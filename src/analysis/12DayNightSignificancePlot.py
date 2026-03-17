import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *

save_path = f"{root}/images/day-night"
if not os.path.exists(save_path):
    os.makedirs(save_path)

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
    choices=["Gaussian", "Asimov"],
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
    "--folder", type=str, help="The name of the results folder", default="Reduced"
)
parser.add_argument(
    "--signal_uncertainty",
    type=float,
    help="The signal uncertainty for the analysis",
    default=0.00,
)
parser.add_argument(
    "--background_uncertainty",
    type=float,
    help="The background uncertainty for the analysis",
    default=0.02,
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
parser.add_argument("--stacked", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument(
    "--threshold", type=float, help="The threshold for the analysis", default=8
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()

for config, name, energy in product(args.config, args.name, args.energy):
    info = json.loads(open(f"{root}/config/{config}/{config}_config.json").read())
    sigma_label = "highest"
    sigma = pickle.load(
        open(
            f"{info['PATH']}/DAYNIGHT/{args.folder.lower()}/{config}/{args.name[0]}/{config}_{args.name[0]}_{sigma_label}_DayNight.pkl",
            "rb",
        )
    )

    plot_df = pd.read_pickle(
        f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/signal/{args.folder.lower()}/DAYNIGHT/{config}/{name}/{config}_{name}_{energy}_Rebin.pkl"
    )

    for bkg, bkg_label, color in [
        ("neutron", "neutron", "green"),
        ("gamma", "gamma", "black"),
    ]:
        bkg_df = pd.read_pickle(
            f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/background/{args.folder.lower()}/DAYNIGHT/{config}/{bkg}/{config}_{bkg}_{energy}_Rebin.pkl"
        )
        plot_df = pd.concat([plot_df, bkg_df], ignore_index=True)

    sigmas_df = pd.read_pickle(
        f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/{args.analysis.upper()}/{args.folder.lower()}/{config}/{name}/{config}_{name}_{energy}_{args.analysis}_Results.pkl",
    )

    try:
        ref_plot = sigma[(config, name, energy)]
    except KeyError:
        rprint(
            f"[yellow][WARNING] Not found {sigma_label} for {config} {name} {energy}[/yellow]"
        )
        continue

    if args.debug:
        print(ref_plot)

    for key in sigma.keys():
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

        fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.7, 0.3],
            vertical_spacing=0,
            subplot_titles=(
                f"min#Hits {nhits:.0f}, min#OpHits {ophits:.0f}, max#AdjCl {adjcl:.0f}",
                "",
            ),
        )

        background = np.zeros(len(this_plot_df["Energy"].values[0]))

        for idx, (
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
                continue

            fig.add_trace(
                go.Scatter(
                    x=comp_df["Energy"].values[0],
                    y=args.exposure * np.asarray(comp_df["Counts/Energy"].values[0]),
                    name=f"{component_label}",
                    mode="lines",
                    error_y=dict(
                        type="data",
                        array=args.exposure * np.asarray(comp_df["Error"].values[0]),
                    ),
                    line_shape="hvh",
                    line=dict(
                        color=color,
                        width=3,
                    ),
                    legend="legend",
                    legendgroup=legend_group,
                    legendgrouptitle=dict(text=f"{legend_group_title}"),
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
            if osc == "Truth":
                background += comp_df["Counts"].values[0]
            else:
                if mean == "Day":
                    signal_day = np.asarray(comp_df["Counts"].values[0])
                if mean == "Night":
                    signal_night = np.asarray(comp_df["Counts"].values[0])

        significance = evaluate_significance(
            args.exposure * (signal_night - signal_day),
            args.exposure * (background / 2 + signal_day),
            type=args.reference.lower(),
        )
        # Substitute nan in significance with 0
        significance = np.nan_to_num(significance, nan=0.0)
        if args.threshold is not None:
            threshold_idx = np.where(
                np.asarray(comp_df["Energy"].values[0]) > args.threshold
            )[0][0]
            significance[:threshold_idx] = 0

        fig.add_trace(
            go.Scatter(
                x=comp_df["Energy"].values[0],
                y=significance,
                mode="lines",
                name=f"{args.reference}: {np.sqrt(np.sum(np.power(significance, 2))):.1f}",
                line=dict(
                    color="black",
                    # Increase line width
                    width=3,
                ),
                line_shape="hvh",
                legend="legend2",
                legendgroup="Significance",
                legendgrouptitle=dict(text="Significance"),
                showlegend=True,
            ),
            row=2,
            col=1,
        )

        fig = format_coustom_plotly(
            fig,
            tickformat=(".1f", ".0e"),
            add_units=False,
            legend_title=f"{energy}",
            title=f"{args.folder} Day-Night - {config}",
            matches=("x", None),
        )
        fig.update_yaxes(
            type="log",
            tickformat=".0e",
            # Reduce number of ticks
            dtick=1,
            range=[np.log10(args.exposure * 1e-3), np.log10(args.exposure * 1e4)],
            # range=[1, 7],
            title=f"Counts per Energy ({args.exposure:.0f} kT·year·MeV)⁻¹",
            row=1,
            col=1,
        )
        fig.update_yaxes(
            tickformat=".0f",
            dtick=1,
            range=[0, 1.5 * np.max(significance)],
            title=f"Significance (σ)",
            row=2,
            col=1,
        )
        fig.update_xaxes(showticklabels=False, range=[6.75, 26], row=1, col=1)
        fig.update_xaxes(
            range=[6.75, 26],
            title=f"Reconstructed Neutrino Energy (MeV)",
            row=2,
            col=1,
        )

        if args.threshold is not None:
            fig.add_vline(
                x=args.threshold,
                line_dash="dash",
                line_color="grey",
                annotation=dict(text="Threshold", showarrow=False),
                annotation_position="bottom right",
            )

        fig.update_layout(
            # legend_title_text="",
            legend=dict(y=0.99, x=0.76, font=dict(size=16)),
            legend2=dict(y=0.2, x=0.7, font=dict(size=16), bgcolor="rgba(0,0,0,0)"),
        )

        figure_name = f"{energy}_DayNight_Significance"

        if args.nhits is not None or args.ophits is not None or args.adjcls is not None:
            figure_name += f"_NHits{nhits:.0f}_OpHits{ophits:.0f}_AdjCl{adjcl:.0f}"

        if args.exposure is not None:
            figure_name += f"_Exposure_{args.exposure:.0f}"

        if args.threshold is not None:
            figure_name += f"_Threshold_{args.threshold:.0f}"

        if args.stacked:
            figure_name += "_Stacked"

        save_figure(
            fig,
            f"{save_path}/{args.folder.lower()}",
            config,
            None,
            filename=figure_name,
            rm=args.rewrite,
            debug=args.debug,
        )
