import sys

sys.path.insert(0, "../../")

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
    type=str,
    help="The configuration to load",
    default="hd_1x2x6_centralAPA",
)
parser.add_argument(
    "--name", type=str, help="The name of the configuration", default="marley"
)
parser.add_argument(
    "--folder", type=str, help="The name of the results folder", default="Reduced"
)
parser.add_argument(
    "--signal_uncertanty",
    type=float,
    help="The signal uncertanty for the analysis",
    default=0.3,
)
parser.add_argument(
    "--background_uncertanty",
    type=float,
    help="The background uncertanty for the analysis",
    default=0.02,
)
parser.add_argument(
    "--exposure",
    type=float,
    help="The exposure for the analysis",
    default=100,
)
parser.add_argument(
    "--fiducial",
    nargs="+",
    type=int,
    help="The fiducial cut for the analysis",
    default=[0, 20, 40, 60, 80, 100, 120],
)
parser.add_argument(
    "--nhits",
    nargs="+",
    type=int,
    help="The min niht cut for the analysis",
    default=nhits[:10],
)
parser.add_argument(
    "--ophits",
    nargs="+",
    type=int,
    help="The min ophit cut for the analysis",
    default=nhits[3:10],
)
parser.add_argument(
    "--adjcls",
    nargs="+",
    type=int,
    help="The max adjcl cut for the analysis",
    default=nhits[::-1][10:],
)
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
if parser.parse_args().verbose:
    print(
        f"Configuration: {parser.parse_args().config}, Name: {parser.parse_args().name}, Signal Uncertainty: {parser.parse_args().signal_uncertanty}, Background Uncertainty: {parser.parse_args().background_uncertanty}, Fiducial: {parser.parse_args().fiducial}, NHits: {parser.parse_args().nhits}, OpHits: {parser.parse_args().ophits}, AdjCls: {parser.parse_args().adjcls}"
    )

analysis = parser.parse_args().analysis
config = parser.parse_args().config
name = parser.parse_args().name
folder = parser.parse_args().folder

exposure = parser.parse_args().exposure
fiducials = parser.parse_args().fiducial
hits = parser.parse_args().nhits
ophits = parser.parse_args().ophits
adjcls = parser.parse_args().adjcls
threshold = parser.parse_args().threshold

configs = {config: [name]}

user_input = {
    "signal_uncertanty": parser.parse_args().signal_uncertanty,
    "background_uncertanty": parser.parse_args().background_uncertanty,
    "threshold": parser.parse_args().threshold,
    "rewrite": parser.parse_args().rewrite,
    "debug": parser.parse_args().debug,
}

for config in configs:
    for idx, name in enumerate(configs[config]):
        plot_df = pd.read_pickle(
            f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/signal/nominal/HEP/{config}/{name}/{config}_{name}_rebin.pkl"
        )

        for bkg, bkg_label, color in [
            ("neutron", "neutron", "green"),
            ("gamma", "gamma", "black"),
        ]:
            bkg_df = pd.read_pickle(
                f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/background/{folder.lower()}/HEP/{config}/{bkg}/{config}_{bkg}_rebin.pkl"
            )
            plot_df = pd.concat([plot_df, bkg_df], ignore_index=True)

        sigmas_df = pd.read_pickle(
            f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/{analysis.upper()}/{folder.lower()}/{config}/{name}/{config}_{name}_{analysis}_Results.pkl",
        )

        for energy_label in ["Cluster", "Total", "Selected", "Solar"]:
            # for sigma_name, sigma_label in zip(
            #     ["highest", "fastest_sigma2", "fastest_sigma3"],
            #     ["Highest", "Sigma2", "Sigma3"],
            # ):
            for sigma_name, sigma_label in zip(
                ["fastest_sigma3"],
                ["Sigma3"],
            ):

                sigma = pd.read_pickle(
                    f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/{analysis.upper()}/{folder.lower()}/{config}/{name}/{config}_{name}_{sigma_name}_{analysis}.pkl",
                )

                try:
                    ref_plot = sigma[(config, name, energy_label)]
                except KeyError:
                    rprint(
                        f"[yellow][WARNING] Not found {sigma_label} for {config} {name} {energy_label}[/yellow]"
                    )
                    continue

                rprint(
                    f"Evaluating {sigma_label} for Fiducial {ref_plot['Fiducialized']} (cm), min#Hits {ref_plot['NHits']}, min#OpHits {ref_plot['OpHits']}, max#AdjCl {ref_plot['AdjCl']}"
                )

                this_plot_df = plot_df[
                    (plot_df["EnergyLabel"] == energy_label)
                    * (plot_df["Fiducialized"] == ref_plot["Fiducialized"])
                    * (plot_df["NHits"] == ref_plot["NHits"])
                    * (plot_df["OpHits"] == ref_plot["OpHits"])
                    * (plot_df["AdjCl"] == ref_plot["AdjCl"])
                ].copy()

                plot_sigmas = sigmas_df.loc[
                    (sigmas_df["Config"] == config) * (sigmas_df["Name"] == name)
                ].copy()

                plot_sigmas = plot_sigmas.loc[
                    (plot_sigmas["EnergyLabel"] == energy_label)
                    * (plot_sigmas["Fiducialized"] == int(ref_plot["Fiducialized"]))
                    * (plot_sigmas["NHits"] == int(ref_plot["NHits"]))
                    * (plot_sigmas["OpHits"] == int(ref_plot["OpHits"]))
                    * (plot_sigmas["AdjCl"] == int(ref_plot["AdjCl"]))
                ].copy()

                fig = make_subplots(
                    rows=2,
                    cols=1,
                    row_heights=[0.7, 0.3],
                    vertical_spacing=0,
                    subplot_titles=(
                        f"Fiducial {ref_plot['Fiducialized']:.0f} (cm), min#Hits {ref_plot['NHits']:.0f}, min#OpHits {ref_plot['OpHits']:.0f}, max#AdjCl {ref_plot['AdjCl']:.0f}",
                        "",
                    ),
                )

                signal = np.zeros(len(this_plot_df["Energy"].values[0]))
                background = np.zeros(len(this_plot_df["Energy"].values[0]))
                background_error = np.zeros((3, len(this_plot_df["Energy"].values[0])))
                background_uncertanty = np.zeros(
                    (3, len(this_plot_df["Energy"].values[0]))
                )

                for jdx, (
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
                            "rgb(225,124,5)",
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

                    fig.add_trace(
                        go.Scatter(
                            x=comp_df["Energy"].values[0],
                            y=exposure * np.asarray(comp_df["Counts/Energy"].values[0]),
                            name=component,
                            mode="lines",
                            error_y=dict(
                                type="data",
                                array=exposure * np.asarray(comp_df["Error"].values[0]),
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
                        signal += comp_df["Counts"].values[0]
                        signal_detection = (
                            exposure
                            * np.asarray(signal)
                            * (1 - 3 * user_input["signal_uncertanty"])
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
                        background_systematic = user_input[
                            "background_uncertanty"
                        ] * np.ones(len(background_statistical))
                        background_uncertanty[idx] = (
                            np.sum(
                                [
                                    background_statistical**2,
                                    background_systematic**2,
                                ],
                                axis=0,
                            )
                            ** 0.5
                        )
                        # Multiply each line in background_uncertanty by the background
                        background_error[idx] = np.multiply(
                            background_uncertanty[idx],
                            comp_df["Counts"].values[0],
                            out=np.zeros_like(background_uncertanty[idx]),
                            where=background_uncertanty[idx] != 0,
                        )
                # Sum the background error to get the total background error
                background_error = np.sum(background_error, axis=0)

                for jdx, significance_label in enumerate(["asimov", "gaussian"]):

                    significance = evaluate_significance(
                        signal_detection * exposure * (signal),
                        signal_detection * exposure * (background),
                        background_uncertanty=signal_detection
                        * exposure
                        * background_error,
                        type=significance_label,
                    )
                    # Substitute nan in significance with 0
                    significance = np.nan_to_num(significance, nan=0.0)
                    if threshold is not None:
                        threshold_idx = np.where(
                            np.asarray(comp_df["Energy"].values[0]) > threshold
                        )[0][0]
                        significance[:threshold_idx] = 0

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

                fig = format_coustom_plotly(
                    fig,
                    tickformat=(".1f", ".0e"),
                    add_units=False,
                    legend_title=f"{energy_label}Energy",
                    title=f"{folder} Sample for Solar Neutrino HEP Significance",
                    matches=("x", None),
                )
                fig.update_yaxes(
                    type="log",
                    tickformat=".0e",
                    # Reduce number of ticks
                    dtick=1,
                    range=[np.log10(exposure * 1e-2), np.log10(exposure * 1e5)],
                    title=f"Counts per Energy ({exposure:.0f} kT·year·MeV)⁻¹",
                    row=1,
                    col=1,
                )
                fig.update_yaxes(
                    tickformat=".0f",
                    dtick=1,
                    range=[0, 4.5],
                    title=f"Significance (σ)",
                    row=2,
                    col=1,
                )
                fig.update_xaxes(range=[4, 25], showticklabels=False, row=1, col=1)
                fig.update_xaxes(
                    range=[4, 25],
                    title=f"Reconstructed Neutrino Energy (MeV)",
                    row=2,
                    col=1,
                )

                if threshold is not None:
                    fig.add_vline(
                        x=user_input["threshold"],
                        line_dash="dash",
                        line_color="grey",
                        annotation=dict(text="Threshold", showarrow=False),
                        annotation_position="bottom right",
                    )

                fig.update_layout(
                    legend_title_text="",
                    legend=dict(y=0.99, x=0.8, font=dict(size=16)),
                    legend2=dict(
                        y=0.2, x=0.7, font=dict(size=16), bgcolor="rgba(0,0,0,0)"
                    ),
                )

                if threshold is None:
                    figure_name = f"{sigma_label}_{energy_label}_Significance_Exposure_{exposure:.0f}"

                else:
                    figure_name = f"{sigma_label}_{energy_label}_Significance_Exposure_{exposure:.0f}_Threshold_{threshold:.0f}"

                save_figure(
                    fig,
                    f"{save_path}/{folder.lower()}",
                    config,
                    None,
                    figure_name,
                    rm=user_input["rewrite"],
                    debug=user_input["debug"],
                )
