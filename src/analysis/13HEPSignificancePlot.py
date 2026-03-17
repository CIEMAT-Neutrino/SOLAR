import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *

save_path = f"{root}/images/hep"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Define flags for the args.analysis config and name with the python parser
parser = argparse.ArgumentParser(
    description="Plot the energy distribution of the particles"
)
parser.add_argument(
    "--analysis", type=str, help="The args.analysis configuration", default="HEP"
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
    default=0.3,
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
    "--threshold", type=float, help="The threshold for the analysis", default=None
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
        f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/{args.analysis.upper()}/{args.folder.lower()}/{config}/{name}/{config}_{name}_{energy}_{args.analysis}_Results.pkl",
    )

    for sigma_name, sigma_label in zip(
        ["highest"],
        ["HighestSigma"],
    ):

        sigma = pd.read_pickle(
            f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/{args.analysis.upper()}/{args.folder.lower()}/{config}/{name}/{config}_{name}_{sigma_name}_{args.analysis}.pkl",
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
            f"\nEvaluating {sigma_label} for min#Hits {ref_plot['NHits']}, min#OpHits {ref_plot['OpHits']}, max#AdjCl {ref_plot['AdjCl']}"
        )

        this_plot_df = plot_df[
            # (plot_df["EnergyLabel"] == energy)
            (plot_df["NHits"] == ref_plot["NHits"])
            * (plot_df["OpHits"] == ref_plot["OpHits"])
            * (plot_df["AdjCl"] == ref_plot["AdjCl"])
        ].copy()

        plot_sigmas = sigmas_df.loc[
            (sigmas_df["Config"] == config) * (sigmas_df["Name"] == name)
        ].copy()

        plot_sigmas = plot_sigmas.loc[
            # (plot_sigmas["EnergyLabel"] == energy)
            (plot_sigmas["NHits"] == int(ref_plot["NHits"]))
            * (plot_sigmas["OpHits"] == int(ref_plot["OpHits"]))
            * (plot_sigmas["AdjCl"] == int(ref_plot["AdjCl"]))
        ].copy()

        fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.7, 0.3],
            vertical_spacing=0,
            subplot_titles=(
                f"min#Hits {ref_plot['NHits']:.0f}, min#OpHits {ref_plot['OpHits']:.0f}, max#AdjCl {ref_plot['AdjCl']:.0f}",
                "",
            ),
        )

        # signal = np.zeros(len(this_plot_df["Energy"].values[0]))
        background = np.zeros(len(this_plot_df["Energy"].values[0]))
        background_error = np.zeros((3, len(this_plot_df["Energy"].values[0])))
        # background_uncertainty = np.zeros((3, len(this_plot_df["Energy"].values[0])))

        for idx, (
            component,
            osc,
            legend_group,
            legend_group_title,
            color,
        ) in enumerate(
            zip(
                ["neutron", "gamma", "8B", "hep"],
                ["Truth", "Truth", "Osc", "Osc"],
                [1, 1, 1, 0],
                ["Background", "Background", "Background", "Signal"],
                [
                    "rgb(15,133,84)",
                    "black",
                    "rgb(265,124,5)",
                    "rgb(204,80,62)",
                ],
            )
        ):

            comp_df = this_plot_df.loc[
                (this_plot_df["Component"] == component)
                * (this_plot_df["Oscillation"] == osc)
                * (this_plot_df["Mean"] == "Mean")
            ].copy()

            # If the dataframe is empty, skip the iteration
            if comp_df.empty:
                continue

            if args.stacked:
                fig.add_trace(
                    go.Bar(
                        x=comp_df["Energy"].values[0],
                        y=args.exposure
                        * np.asarray(comp_df["Counts/Energy"].values[0]),
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
                        x=comp_df["Energy"].values[0],
                        y=args.exposure
                        * np.asarray(comp_df["Counts/Energy"].values[0]),
                        name=component,
                        mode="lines",
                        error_y=dict(
                            type="data",
                            array=args.exposure
                            * np.asarray(comp_df["Error"].values[0]),
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
            if component == "hep":
                signal = comp_df["Counts"].values[0]
                signal_detection = (
                    args.exposure
                    * np.asarray(signal)
                    * (1 - 3 * args.signal_uncertainty)
                    > 1
                )
            else:
                background += comp_df["Counts"].values[0]
                background_statistical = np.divide(
                    comp_df["Error"].values[0],
                    comp_df["Counts"].values[0],
                    out=np.zeros_like(comp_df["Error"].values[0]),
                    where=comp_df["Counts"].values[0] != 0,
                )
                background_systematic = args.background_uncertainty * np.ones(
                    len(background_statistical)
                )
                background_error[idx] = (
                    np.sum(
                        [
                            background_statistical**2,
                            background_systematic**2,
                        ],
                        axis=0,
                    )
                    ** 0.5
                )
                # Multiply each line in background_uncertainty by the background
                background_error[idx] = np.multiply(
                    background_error[idx],
                    comp_df["Counts"].values[0],
                    out=np.zeros_like(background_error[idx]),
                    where=background_error[idx] != 0,
                )
                # Substitute nan values with 0
                background_error[idx] = np.nan_to_num(
                    background_error[idx],
                    nan=0,
                    posinf=0,
                    neginf=0,
                )
        # Sum the background error to get the total background error
        background_error = np.sum(background_error, axis=0)

        for jdx, significance_label in enumerate(["asimov", "gaussian"]):

            significance = evaluate_significance(
                signal_detection * args.exposure * signal,
                np.multiply(args.exposure, background),
                background_uncertainty=np.multiply(args.exposure, background_error),
                type=significance_label,
            )
            # Substitute nan in significance with 0
            significance = np.nan_to_num(significance, nan=0.0)
            # if args.threshold is not None:
            #     threshold_idx = np.where(
            #         np.asarray(comp_df["Energy"].values[0]) > args.threshold
            #     )[0][0]
            #     significance[:threshold_idx] = 0

            fig.add_trace(
                go.Scatter(
                    x=comp_df["Energy"].values[0],
                    y=significance,
                    mode="lines",
                    name=f"{significance_label}: {np.sqrt(np.sum(np.power(significance, 2))):.1f}",
                    line=dict(
                        color="black",
                        dash=["solid", "dash"][jdx],
                    ),
                    line_shape="hvh",
                    legend="legend2",
                    legendgroup="Significance",
                    legendgrouptitle=dict(text=""),
                    showlegend=True,
                ),
                row=2,
                col=1,
            )

        if args.stacked:
            fig.update_layout(barmode="stack")

        fig = format_coustom_plotly(
            fig,
            tickformat=(".1f", ".0e"),
            add_units=False,
            legend_title=f"{energy}",
            title=f"{args.folder} Sample for Solar Neutrino HEP Significance",
            matches=("x", None),
        )
        fig.update_yaxes(
            type="log",
            tickformat=".0e",
            dtick=1,
            range=[np.log10(args.exposure * 1e-2), np.log10(args.exposure * 1e4)],
            title=f"Counts per Energy ({args.exposure:.0f} kT·year·MeV)⁻¹",
            row=1,
            col=1,
        )
        fig.update_yaxes(
            tickformat=".0f",
            range=[0, 1.5 * np.max(significance)],
            title=f"Significance (σ)",
            row=2,
            col=1,
        )
        fig.update_xaxes(
            range=[10, 26] if args.stacked else [8, 26],
            showticklabels=False,
            row=1,
            col=1,
        )
        fig.update_xaxes(
            range=[10, 26] if args.stacked else [8, 26],
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
            legend_title_text="",
            legend=dict(y=0.99, x=0.8, font=dict(size=14)),
            legend2=dict(y=0.2, x=0.76, font=dict(size=14), bgcolor="rgba(0,0,0,0)"),
        )

        figure_name = f"{energy}_HEP_Significance"

        if args.nhits is not None or args.ophits is not None or args.adjcls is not None:
            figure_name += f"_NHits{ref_plot['NHits']:.0f}_OpHits{ref_plot['OpHits']:.0f}_AdjCl{ref_plot['AdjCl']:.0f}"

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
