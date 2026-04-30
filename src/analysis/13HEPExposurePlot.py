import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *

save_path = f"{root}/images/analysis/hep"
data_path = f"{root}/data/analysis/hep"

for this_path in [save_path, data_path]:
    if not os.path.exists(this_path):
        os.makedirs(this_path)

# Define flags for the analysis config and name with the python parser
parser = argparse.ArgumentParser(
    description="Plot the HEP significance analysis for a given configuration and name and plot the results as a function of exposure"
)
analysis_info = load_analysis_info(str(root))
default_reference = analysis_info.get("BEST_SIGMA_SIGNIFICANCE_REFERENCE", {}).get("HEP", "Asimov")
parser.add_argument(
    "--analysis", type=str, help="The analysis configuration", default="HEP"
)
parser.add_argument(
    "--reference",
    type=str,
    choices=["Gaussian", "Asimov", "ProfileLikelihood"],
    default=default_reference,
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
    default=30,
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
parser.add_argument(
    "--threshold",
    type=float,
    help="The threshold for the analysis",
    default=get_analysis_threshold(str(root), "HEP", stage="SIGNIFICANCE", fallback=0.0),
)

parser.add_argument("--stacked", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--zoom", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()

hep_counts = []
hep_exposure = []
smoothing_config = get_smoothing_config(
    str(root), analysis_name="HEP", dimensions="1d", stage="significance"
)

for config, name, energy in product(args.config, args.name, args.energy):
    info = json.loads(open(f"{root}/config/{config}/{config}_config.json").read())
    detector_mass = get_full_detector_mass(config, info)

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
    profile_sigmas_df = None
    profile_path = (
        f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/{args.analysis.upper()}/{args.folder.lower()}/"
        f"{config}/{name}/{config}_{name}_{energy}_{args.analysis}_ProfileLikelihood.pkl"
    )
    if os.path.exists(profile_path):
        profile_sigmas_df = pd.read_pickle(profile_path)
    elif args.reference == "ProfileLikelihood":
        rprint(
            f"[yellow][WARNING][/yellow] Missing profile-likelihood curve payload for {config} {name} {energy}: {profile_path}"
        )

    required_base_columns = ["Config", "Name", "NHits", "OpHits", "AdjCl", "Exposure"]
    missing_base_columns = [
        column for column in required_base_columns if column not in sigmas_df.columns
    ]
    if sigmas_df.empty or missing_base_columns:
        rprint(
            f"[yellow][WARNING][/yellow] Skipping exposure plot for {config} {name} {energy}: "
            f"results dataframe is empty or missing required columns {missing_base_columns}."
        )
        continue

    for sigma_name, sigma_label in zip(
        ["highest"],
        ["Highest"],
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

        if args.debug:
            rprint(
                f"Evaluating {sigma_label} for min#Hits {ref_plot['NHits']}, min#OpHits {ref_plot['OpHits']}, max#AdjCl {ref_plot['AdjCl']}"
            )

        this_plot_df = plot_df[
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
            cols=1,
            subplot_titles=(
                f"{energy}, min#Hits {ref_plot['NHits']:.0f}, min#OpHits {ref_plot['OpHits']:.0f}, max#AdjCl {ref_plot['AdjCl']:.0f}",
            ),
        )
        significance_peak = 0.0

        if plot_sigmas.empty:
            rprint(
                f"[yellow][WARNING] Not found {sigma_label} for {config} {name} {energy}[/yellow]"
            )
            continue

        significance_plot_styles = {
            "Asimov": {"label": "asimov", "dash": "solid", "raw_dash": "dot"},
            "Gaussian": {"label": "gaussian", "dash": "dash", "raw_dash": "dashdot"},
            "ProfileLikelihood": {
                "label": "profile-likelihood",
                "dash": "solid",
                "raw_dash": "dot",
            },
        }
        for significance, style in significance_plot_styles.items():
            if significance == "ProfileLikelihood":
                if profile_sigmas_df is None or profile_sigmas_df.empty:
                    continue
                this_plot_sigmas = profile_sigmas_df[
                    (profile_sigmas_df["Config"] == config)
                    * (profile_sigmas_df["Name"] == name)
                    * (profile_sigmas_df["NHits"] == int(ref_plot["NHits"]))
                    * (profile_sigmas_df["OpHits"] == int(ref_plot["OpHits"]))
                    * (profile_sigmas_df["AdjCl"] == int(ref_plot["AdjCl"]))
                ].copy()
            else:
                this_plot_sigmas = plot_sigmas.copy()

            if this_plot_sigmas.empty:
                continue

            required_significance_columns = [
                significance,
                "Raw" + significance,
                significance + "+Error",
                significance + "-Error",
            ]
            missing_significance_columns = [
                column
                for column in required_significance_columns
                if column not in this_plot_sigmas.columns
            ]
            if missing_significance_columns:
                rprint(
                    f"[yellow][WARNING][/yellow] Missing columns {missing_significance_columns} "
                    f"for {config} {name} {energy}. Skipping {significance} exposure curve."
                )
                continue

            exposure_values = np.asarray(this_plot_sigmas["Exposure"].values[0], dtype=float)
            smoothed_significance = np.nan_to_num(
                np.asarray(this_plot_sigmas[significance].values[0], dtype=float),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            raw_significance = np.nan_to_num(
                np.asarray(this_plot_sigmas["Raw" + significance].values[0], dtype=float),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            significance_plus = np.nan_to_num(
                np.asarray(this_plot_sigmas[significance + "+Error"].values[0], dtype=float),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            significance_minus = np.nan_to_num(
                np.asarray(this_plot_sigmas[significance + "-Error"].values[0], dtype=float),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            hep_exposure.append(
                {
                    "Geometry": info["GEOMETRY"],
                    "Config": config,
                    "Name": name,
                    "NHits": int(ref_plot["NHits"]),
                    "OpHits": int(ref_plot["OpHits"]),
                    "AdjCl": int(ref_plot["AdjCl"]),
                    "Exposure": exposure_values,
                    "EnergyLabel": energy,
                    "Variable": significance,
                    "Significance": smoothed_significance,
                    "RawSignificance": raw_significance,
                    "SignificanceError+": np.subtract(
                        significance_plus,
                        smoothed_significance,
                    ),
                    "SignificanceError-": np.subtract(
                        smoothed_significance,
                        significance_minus,
                    ),
                }
            )

            if significance != args.reference:
                continue
            y_upper = None
            y_lower = None
            if significance in ["Asimov", "ProfileLikelihood"]:
                y_upper = significance_plus
                y_lower = significance_minus
            significance_peak = max(
                significance_peak,
                float(np.max(raw_significance)),
                float(np.max(smoothed_significance)),
                float(np.max(significance_plus)),
            )

            add_reference_pair_traces(
                fig,
                x=exposure_values,
                y_raw=raw_significance,
                y_smoothed=smoothed_significance,
                name=style["label"],
                raw_style={"color": "black", "dash": style["raw_dash"], "width": 2},
                smoothed_style={"color": "black", "dash": style["dash"], "width": 2},
                row=1,
                col=1,
                legend="legend",
                legendgroup=significance,
                legendgrouptitle="Significance",
                showlegend_raw=False,
                showlegend_smoothed=True,
                line_shape="linear",
                y_upper=y_upper,
                y_lower=y_lower,
            )

        fig = format_coustom_plotly(
            fig,
            tickformat=(".1f", ".0e"),
            add_units=False,
            legend_title=f"{energy}",
            title=f"HEP Discovery - {args.folder} - {config}",
            matches=(None, None),
        )

        fig.update_yaxes(
            tickformat=".1f",
            dtick=1,
            range=[0, max(1.0, 1.1 * significance_peak)] if args.zoom else [0, 6],
            title=f"Significance (σ)",
            row=1,
            col=1,
        )
        fig.update_xaxes(
            range=[-1, args.exposure],
            zeroline=False,
            title=f"Exposure (kT·year)",
            row=1,
            col=1,
        )

        for sigma, cl in zip([1, 2, 3, 4, 5], [0.6827, 0.9545, 0.9973, 0.9999, 0.999999]):
            fig.add_hline(y=sigma, line_dash="dash", line_color="grey")
            fig.add_annotation(
                x=detector_mass * args.exposure * 0.1,
                y=sigma + 0.2,
                text=f"{100*cl:.2f}% CL",
                showarrow=False,
                row=1,
                col=1,
            )

        fig.update_layout(
            legend_title_text="",
            legend=dict(font=dict(size=12), bgcolor="rgba(255,255,255,0.7)"),
        )

        figure_name = f"{energy}_HEP_Exposure_{args.reference}"

        if (
            args.nhits is not None
            or args.ophits is not None
            or args.adjcls is not None
        ):
            figure_name += f"_NHits{ref_plot['NHits']:.0f}_OpHits{ref_plot['OpHits']:.0f}_AdjCl{ref_plot['AdjCl']:.0f}"

        if args.threshold is not None:
            figure_name += f"_Threshold_{args.threshold:.0f}"

        if args.stacked:
            figure_name += "_Stacked"

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

        for df, df_name in zip(
            [pd.DataFrame(hep_counts), pd.DataFrame(hep_exposure)],
            ["HEP_Counts", "HEP_Exposure"],
        ):
            save_df(
                df,
                data_path,
                config=config,
                name=None,
                subfolder=args.folder.lower(),
                filename=df_name,
                rm=args.rewrite,
                debug=args.debug,
            )
