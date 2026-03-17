import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *

save_path = f"{root}/images/hep"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Define flags for the analysis config and name with the python parser
parser = argparse.ArgumentParser(
    description="Plot the energy distribution of the particles"
)
parser.add_argument(
    "--analysis", type=str, help="The analysis configuration", default="HEP"
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
    "--folder", type=str, help="The name of the results folder", default="Nominal"
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
    "--threshold", type=float, help="The threshold for the analysis", default=10.0
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
# Print the values of the flags
parser.add_argument(
    "--verbose",
    action="store_true",
    help="Print the values of the flags",
    default=False,
)
args = parser.parse_args()

analysis = args.analysis
config = args.config
name = args.name
folder = args.folder

for config, name, energy in product(args.config, args.name, args.energy):
    plot_df = pd.read_pickle(
        f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/signal/{args.folder.lower()}/HEP/{config}/{name}/{config}_{name}_{energy}_Rebin.pkl"
    )

    for bkg, bkg_label, color in [
        ("neutron", "neutron", "green"),
        ("gamma", "gamma", "black"),
    ]:
        bkg_df = pd.read_pickle(
            f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/background/{args.folder.lower()}/HEP/{config}/{bkg}/{config}_{bkg}_{energy}_Rebin.pkl"
        )
        plot_df = pd.concat([plot_df, bkg_df], ignore_index=True)

    sigmas_df = pd.read_pickle(
        f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/{analysis.upper()}/{args.folder.lower()}/{config}/{name}/{config}_{name}_{energy}_{analysis}_Results.pkl",
    )
    sigmas_df = sigmas_df.explode(
        [
            "Sigma2",
            "Sigma3",
            "Exposure",
            "Gaussian-Error",
            "Gaussian",
            "Gaussian+Error",
            "Asimov-Error",
            "Asimov",
            "Asimov+Error",
        ]
    )

    for sigma_name, sigma_label in zip(
        ["highest"],
        ["Highest"],
    ):

        sigma = pd.read_pickle(
            f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/{analysis.upper()}/{folder.lower()}/{config}/{name}/{config}_{name}_{sigma_name}_{analysis}.pkl",
        )

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

        rprint(
            f"Evaluating {sigma_label} for min#Hits {ref_plot['NHits']}, min#OpHits {ref_plot['OpHits']}, max#AdjCl {ref_plot['AdjCl']}"
        )

        this_plot_df = plot_df[
            (plot_df["NHits"] == ref_plot["NHits"])
            * (plot_df["OpHits"] == ref_plot["OpHits"])
            * (plot_df["AdjCl"] == ref_plot["AdjCl"])
        ].copy()

        print(
            this_plot_df.explode("Counts")
            .groupby(["Component", "Oscillation", "Mean"])["Counts"]
            .sum()
        )

        plot_sigmas = sigmas_df[
            (sigmas_df["Config"] == config) * (sigmas_df["Name"] == name)
        ].copy()

        plot_sigmas = plot_sigmas[
            (plot_sigmas["NHits"] == int(ref_plot["NHits"]))
            * (plot_sigmas["OpHits"] == int(ref_plot["OpHits"]))
            * (plot_sigmas["AdjCl"] == int(ref_plot["AdjCl"]))
        ]

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                f"min#Hits {ref_plot['NHits']:.0f}, min#OpHits {ref_plot['OpHits']:.0f}, max#AdjCl {ref_plot['AdjCl']:.0f}",
                "Significance",
            ),
        )

        for (
            component,
            osc,
            legend_group,
            legend_group_title,
            color,
        ) in zip(
            ["8B", "hep", "neutron", "gamma"],
            ["Osc", "Osc", "Truth", "Truth"],
            [1, 0, 1, 1],
            ["Background", "Signal", "Background", "Background"],
            [
                "rgb(225,124,5)",
                "rgb(204,80,62)",
                "rgb(15,133,84)",
                "black",
            ],
        ):

            this_comp_df = this_plot_df.loc[
                (this_plot_df["Component"] == component)
                * (this_plot_df["Oscillation"] == osc)
                * (this_plot_df["Mean"] == "Mean")
            ]
            if args.stacked:
                fig.add_trace(
                    go.Bar(
                        x=this_comp_df["Energy"].values[0],
                        y=np.asarray(this_comp_df["Counts/Energy"].values[0]),
                        name=component,
                        marker_color=color,
                        legend="legend",
                        legendgroup=legend_group,
                        legendgrouptitle=dict(text=f"{legend_group_title}"),
                        showlegend=True,
                    ),
                    row=1,
                    col=1,
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=this_comp_df["Energy"].values[0],
                        y=this_comp_df["Counts/Energy"].values[0],
                        name=component,
                        mode="lines",
                        error_y=dict(
                            type="data",
                            array=this_comp_df["Error"].values[0],
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

        for significance, significance_label, dash in zip(
            ["Asimov", "Gaussian"], ["asimov", "gaussian"], ["solid", "dash"]
        ):
            fig.add_trace(
                go.Scatter(
                    x=plot_sigmas["Exposure"],
                    y=plot_sigmas[significance],
                    name=f"{significance_label}",
                    mode="lines",
                    line=dict(color="black", dash=dash),
                    legend="legend2",
                    legendgroup="Significance",
                    legendgrouptitle=dict(text="Significance"),
                    showlegend=True,
                ),
                row=1,
                col=2,
            )
            if significance == "Asimov":
                fig.add_trace(
                    go.Scatter(
                        x=plot_sigmas["Exposure"],
                        y=plot_sigmas[significance + "+Error"],
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
                        x=plot_sigmas["Exposure"],
                        y=plot_sigmas[significance + "-Error"],
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

        if args.stacked:
            fig.update_layout(barmode="stack")

        fig = format_coustom_plotly(
            fig,
            tickformat=(".1f", ".0e"),
            add_units=False,
            legend_title=f"{energy}",
            title=f"Selected Sample for Solar Neutrino HEP Significance",
            matches=(None, None),
        )

        fig.update_yaxes(
            type="log",
            tickformat=".0e",
            dtick=1,
            range=[-2, 5],
            title=f"Counts per Energy (kT·year·MeV)⁻¹",
            row=1,
            col=1,
        )
        fig.update_xaxes(
            range=[10, 22] if args.stacked else [8, 22],
            title=f"Reconstructed Neutrino Energy (MeV)",
            row=1,
            col=1,
        )
        fig.update_yaxes(
            tickformat=".1f",
            dtick=1,
            range=[-0.1, 6],
            title=f"Significance (σ)",
            row=1,
            col=2,
        )
        fig.update_xaxes(
            range=[-10, args.exposure],
            zeroline=False,
            title=f"Exposure (kT·year)",
            row=1,
            col=2,
        )

        fig.add_vline(
            x=args.threshold,
            line_dash="dash",
            line_color="grey",
            row=1,
            col=1,
            annotation=dict(text="Threshold ", showarrow=False),
            annotation_position="bottom right",
        )
        for sigma, cl in zip([1, 2, 3, 4, 5], [0.6827, 0.9545, 0.9973, 0.9999, 1]):
            fig.add_hline(y=sigma, line_dash="dash", line_color="grey", row=1, col=2)
            fig.add_annotation(
                x=args.exposure * 0.1,
                y=sigma + 0.2,
                text=f"{100*cl:.2f}% CL",
                showarrow=False,
                row=1,
                col=2,
            )

        fig.update_layout(
            legend_title_text="",
            legend=dict(y=1, x=0.33, font=dict(size=14)),
            legend2=dict(y=0.01, x=0.88, font=dict(size=14), bgcolor="rgba(0,0,0,0)"),
        )

        figure_name = f"{energy}_HEP_Exposure"

        if (
            args.nhits is not None
            and args.ophits is not None
            and args.adjcls is not None
        ):
            figure_name += f"_NHits{ref_plot['NHits']:.0f}_OpHits{ref_plot['OpHits']:.0f}_AdjCl{ref_plot['AdjCl']:.0f}"

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
            rm=args.rewrite,
            debug=args.debug,
        )
