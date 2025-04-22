import sys

sys.path.insert(0, "../../")

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
    default=0.00,
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
    "--energy",
    type=str,
    help="The energy for the analysis",
    default=["Cluster", "Total", "Selected", "Solar"],
)
parser.add_argument(
    "--fiducial",
    type=int,
    help="The min niht cut for the analysis",
    default=None,
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
    "--adjcl",
    type=int,
    help="The max adjcl cut for the analysis",
    default=None,
)
parser.add_argument(
    "--threshold", type=float, help="The threshold for the analysis", default=None
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()

analysis = parser.parse_args().analysis
config = parser.parse_args().config
name = parser.parse_args().name

exposure = parser.parse_args().exposure
threshold = parser.parse_args().threshold

configs = {config: [name]}

user_input = {
    "signal_uncertanty": parser.parse_args().signal_uncertanty,
    "background_uncertanty": parser.parse_args().background_uncertanty,
    "rewrite": parser.parse_args().rewrite,
    "debug": parser.parse_args().debug,
}


for config in configs:
    info = json.loads(open(f"{root}/config/{config}/{config}_config.json").read())
    sigma_label = "fastest_sigma2"
    sigma = pickle.load(
        open(
            f"{info['PATH']}/DAYNIGHT/{args.folder.lower()}/{args.config}/{args.name}/{args.config}_{args.name}_{sigma_label}_DayNight.pkl",
            "rb",
        )
    )

    for name, key in product(configs[config], sigma):
        plot_df = pd.read_pickle(
            f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/signal/nominal/DAYNIGHT/{config}/{name}/{config}_{name}_rebin.pkl"
        )

        for bkg, bkg_label, color in [
            ("neutron", "neutron", "green"),
            ("gamma", "gamma", "black"),
        ]:
            bkg_df = pd.read_pickle(
                f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/background/{args.folder.lower()}/DAYNIGHT/{config}/{bkg}/{config}_{bkg}_rebin.pkl"
            )
            plot_df = pd.concat([plot_df, bkg_df], ignore_index=True)

        sigmas_df = pd.read_pickle(
            f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/{analysis.upper()}/{args.folder.lower()}/{config}/{name}/{config}_{name}_{analysis}_Results.pkl",
        )

        if key[2] not in args.energy:
            continue
        else:
            energy = key[2]

        try:
            ref_plot = sigma[(config, name, energy)]
        except KeyError:
            rprint(
                f"[yellow][WARNING] Not found {sigma_label} for {config} {name} {energy}[/yellow]"
            )
            continue

        if args.fiducial is not None:
            fiducial = args.fiducial
        else:
            rprint(f"Using optimized fiducial cut {sigma[key]['Fiducialized']}")
            fiducial = int(sigma[key]["Fiducialized"])

        if args.nhits is not None:
            nhits = args.nhits
        else:
            rprint(f"Using optimized nhits {sigma[key]['NHits']}")
            nhits = int(sigma[key]["NHits"])

        if args.adjcl is not None:
            adjcl = args.adjcl
        else:
            rprint(f"Using optimized adjcl {sigma[key]['AdjCl']}")
            adjcl = int(sigma[key]["AdjCl"])

        if args.ophits is not None:
            ophits = args.ophits
        else:
            rprint(f"Using optimized ophits {sigma[key]['OpHits']}")
            ophits = int(sigma[key]["OpHits"])

        rprint(
            f"Evaluating {sigma_label} for Fiducial {fiducial:.0f} (cm), min#Hits {nhits:.0f}, min#OpHits {ophits:.0f}, max#AdjCl {adjcl:.0f}"
        )

        this_plot_df = plot_df.loc[
            (plot_df["EnergyLabel"] == energy)
            * (plot_df["Fiducialized"] == fiducial)
            * (plot_df["NHits"] == nhits)
            * (plot_df["OpHits"] == ophits)
            * (plot_df["AdjCl"] == adjcl)
        ].copy()

        fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.7, 0.3],
            vertical_spacing=0,
            subplot_titles=(
                f"Fiducial {fiducial:.0f} (cm), min#Hits {nhits:.0f}, min#OpHits {ophits:.0f}, max#AdjCl {adjcl:.0f}",
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
                    y=exposure * np.asarray(comp_df["Counts/Energy"].values[0]),
                    name=f"{component_label}",
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
            if osc == "Truth":
                background += comp_df["Counts"].values[0]
            else:
                if mean == "Day":
                    signal_day = np.asarray(comp_df["Counts"].values[0])
                if mean == "Night":
                    signal_night = np.asarray(comp_df["Counts"].values[0])

        for jdx, significance_label in enumerate(["asimov", "gaussian"]):
            significance = evaluate_significance(
                exposure * (signal_night - signal_day),
                exposure * (background / 2 + signal_day),
                # background_uncertanty=(
                #     1 / ((exposure * (background / 2 + signal_day)) ** 0.5)
                # ),
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
                        dash=["dash", "solid"][jdx],
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
            legend_title=f"{energy}Energy",
            title=f"{args.folder} Sample for Solar Neutrino Day-Night Significance",
            matches=("x", None),
        )
        fig.update_yaxes(
            type="log",
            tickformat=".0e",
            # Reduce number of ticks
            dtick=1,
            range=[np.log10(exposure * 1e0), np.log10(exposure * 1e6)],
            # range=[1, 7],
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
        fig.update_xaxes(showticklabels=False, range=[4, 20], row=1, col=1)
        fig.update_xaxes(
            range=[4, 20],
            title=f"Reconstructed Neutrino Energy (MeV)",
            row=2,
            col=1,
        )

        if threshold is not None:
            fig.add_vline(
                x=threshold,
                line_dash="dash",
                line_color="grey",
                annotation=dict(text="Threshold", showarrow=False),
                annotation_position="bottom right",
            )

        fig.update_layout(
            legend_title_text="",
            legend=dict(y=0.99, x=0.76, font=dict(size=16)),
            legend2=dict(y=0.2, x=0.7, font=dict(size=16), bgcolor="rgba(0,0,0,0)"),
        )

        if threshold is None:
            figure_name = f"{sigma_label}_{energy}_Significance_Exposure_{exposure:.0f}"

        else:
            figure_name = f"{sigma_label}_{energy}_Significance_Exposure_{exposure:.0f}_Threshold_{threshold:.0f}"

        save_figure(
            fig,
            f"{save_path}/{args.folder.lower()}",
            config,
            None,
            filename=figure_name,
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )
