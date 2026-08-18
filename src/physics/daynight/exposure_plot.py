import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *

analysis_info = load_analysis_info(str(root))

save_path = f"{root}/output/images/analysis/day-night"
data_path = f"{root}/output/data/analysis/day-night"

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
    "--signal",
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
    default=0.00,
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
parser.add_argument("--study_label", type=str, default=None, help="Tag appended to image subdirectory to isolate study outputs.")

args = parser.parse_args()
_ctx = study_context(args)
_study_suffix   = _ctx.study_suffix
_save_subfolder = _ctx.save_subfolder

smoothing_config = get_smoothing_config(
    str(root), analysis_name="DAYNIGHT", dimensions="1d", stage="significance"
)
day_night_counts = []
day_night_exposure = []

for config, name, energy in product(args.config, args.signal, args.energy):
    info = json.loads(open(f"{root}/config/{config}/{config}_config.json").read())
    detector_mass = get_full_detector_mass(config, info)

    df_list = []
    signal_df = pd.read_pickle(
        f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/signal/{args.folder.lower()}/DAYNIGHT/{config}/{name}/{config}_{name}_{energy}_Rebin.pkl"
    )
    df_list.append(signal_df)
    for bkg, filepath in load_available_background_dataframes(str(root), "DAYNIGHT", args.folder, config, energy):
        df_list.append(pd.read_pickle(filepath))

    plot_df = pd.concat(df_list, ignore_index=True)

    sigmas_df = pd.read_pickle(
        f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/{args.analysis.upper()}/{args.folder.lower()}/{config}/{name}/{config}_{name}_{energy}_{args.analysis}_Results{_study_suffix}.pkl",
    )

    for sigma_name, sigma_label in zip(
        ["highest"],
        ["Highest"],
    ):
        sigma_path = (
            f"{info['PATH']}/DAYNIGHT/{args.folder.lower()}/{config}/{name}/{config}_{name}_{sigma_name}_DayNight{_study_suffix}.pkl"
        )
        try:
            sigma = pickle.load(open(sigma_path, "rb"))
        except (EOFError, pickle.UnpicklingError) as exc:
            raise RuntimeError(
                f"Failed to load sigma summary from {sigma_path}. The file is likely truncated from an earlier failed write. "
                "Re-run sensitivity/05_best_sigmas.py after removing the broken pickle or with --rewrite."
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

        fig = make_subplots(
            rows=1,
            cols=1,
            subplot_titles=(f"{energy}, min#Hits {ref_plot['NHits']:.0f}, min#OpHits {ref_plot['OpHits']:.0f}, max#AdjCl {ref_plot['AdjCl']:.0f}",),
        )

        if plot_sigmas.empty:
            rprint(
                f"[yellow][WARNING] Not found {sigma_label} for {config} {name} {energy}[/yellow]"
            )
            continue

        exposure_values = np.asarray(plot_sigmas["Exposure"].values[0], dtype=float)
        raw_significance = np.nan_to_num(
            np.asarray(plot_sigmas["RawGaussian"].values[0], dtype=float),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        smoothed_significance = np.nan_to_num(
            np.asarray(plot_sigmas["Gaussian"].values[0], dtype=float),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        significance_upper = np.nan_to_num(
            np.asarray(plot_sigmas["Gaussian+Error"].values[0], dtype=float),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        significance_lower = np.nan_to_num(
            np.asarray(plot_sigmas["Gaussian-Error"].values[0], dtype=float),
            nan=0.0, posinf=0.0, neginf=0.0,
        )

        _has_asimov = "Asimov" in plot_sigmas.columns and "RawAsimov" in plot_sigmas.columns
        if _has_asimov:
            raw_asimov = np.nan_to_num(
                np.asarray(plot_sigmas["RawAsimov"].values[0], dtype=float),
                nan=0.0, posinf=0.0, neginf=0.0,
            )
            smoothed_asimov = np.nan_to_num(
                np.asarray(plot_sigmas["Asimov"].values[0], dtype=float),
                nan=0.0, posinf=0.0, neginf=0.0,
            )
            asimov_upper = np.nan_to_num(
                np.asarray(plot_sigmas["Asimov+Error"].values[0], dtype=float),
                nan=0.0, posinf=0.0, neginf=0.0,
            )
            asimov_lower = np.nan_to_num(
                np.asarray(plot_sigmas["Asimov-Error"].values[0], dtype=float),
                nan=0.0, posinf=0.0, neginf=0.0,
            )

        # Background normalization uncertainty: ErrorGaussian/ErrorAsimov are always required.
        # Both 01_daynight.py (default background_uncertainty=0.02) and the unc_bkg* study
        # variants write these columns.  Hard-fail if absent — no silent fallback.
        error_gaussian_central = np.nan_to_num(
            np.asarray(plot_sigmas["ErrorGaussian"].values[0], dtype=float),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        error_gaussian_upper = np.nan_to_num(
            np.asarray(plot_sigmas["ErrorGaussian+Error"].values[0], dtype=float),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        error_gaussian_lower = np.nan_to_num(
            np.asarray(plot_sigmas["ErrorGaussian-Error"].values[0], dtype=float),
            nan=0.0, posinf=0.0, neginf=0.0,
        )

        # ErrorAsimov is NOT extracted: the Asimov LLR tests the day/night ratio (shape),
        # so a fully-correlated background normalization uncertainty cancels to first order.
        # Only Gaussian sensitivity is meaningfully degraded by background normalization σ_bkg.

        for spectrum_type, significance in [
            ("Raw", raw_significance),
            ("Smoothed", smoothed_significance),
        ]:
            day_night_exposure.append({
                "Geometry": info["GEOMETRY"],
                "Config": config,
                "Name": name,
                "Variable": "Gaussian",
                "Exposure": exposure_values,
                "SpectrumType": spectrum_type,
                "Significance": significance,
                "SignificanceError+": np.subtract(significance_upper, smoothed_significance) if spectrum_type == "Smoothed" else None,
                "SignificanceError-": np.subtract(smoothed_significance, significance_lower) if spectrum_type == "Smoothed" else None,
            })

        if _has_asimov:
            for spectrum_type, significance in [
                ("Raw", raw_asimov),
                ("Smoothed", smoothed_asimov),
            ]:
                day_night_exposure.append({
                    "Geometry": info["GEOMETRY"],
                    "Config": config,
                    "Name": name,
                    "Variable": "Asimov",
                    "Exposure": exposure_values,
                    "SpectrumType": spectrum_type,
                    "Significance": significance,
                    "SignificanceError+": np.subtract(asimov_upper, smoothed_asimov) if spectrum_type == "Smoothed" else None,
                    "SignificanceError-": np.subtract(smoothed_asimov, asimov_lower) if spectrum_type == "Smoothed" else None,
                })

        # Background-uncertainty scenario rows: 3 physics scenarios × 2 metrics (Gaussian + Asimov).
        # SpectrumType encodes the scenario so downstream readers can filter without a Mode column.
        # Each row has Significance = realistic (ErrorGaussian/ErrorAsimov, with bkg systematics)
        # and SignificanceError+ = ideal − realistic (band upward to no-bkg-unc estimate).
        # Background-uncertainty scenario rows.
        # Gaussian: Significance = ErrorGaussian[scenario] (realistic, with σ_bkg);
        #           SignificanceError+ = Gaussian[scenario] − ErrorGaussian[scenario] (band upward to ideal).
        # Asimov:   Significance = Asimov[scenario] (normalization-invariant, no σ_bkg degradation);
        #           SignificanceError+ = 0 (no band — bkg normalization cancels in shape test).
        _bkg_scenario_defs = [
            ("MaxScenario",     significance_upper,    error_gaussian_upper,    asimov_upper    if _has_asimov else None),
            ("CentralScenario", smoothed_significance, error_gaussian_central,  smoothed_asimov if _has_asimov else None),
            ("MinScenario",     significance_lower,    error_gaussian_lower,    asimov_lower    if _has_asimov else None),
        ]
        for scenario_key, ideal_g, real_g, asimov_scenario in _bkg_scenario_defs:
            day_night_exposure.append({
                "Geometry": info["GEOMETRY"],
                "Config": config,
                "Name": name,
                "Variable": "Gaussian",
                "Exposure": exposure_values,
                "SpectrumType": f"Smoothed/{scenario_key}",
                "Significance": real_g,
                "SignificanceError+": np.subtract(ideal_g, real_g),
                "SignificanceError-": np.zeros_like(real_g),
            })
            if _has_asimov:
                day_night_exposure.append({
                    "Geometry": info["GEOMETRY"],
                    "Config": config,
                    "Name": name,
                    "Variable": "Asimov",
                    "Exposure": exposure_values,
                    "SpectrumType": f"Smoothed/{scenario_key}",
                    "Significance": asimov_scenario,
                    "SignificanceError+": np.zeros_like(asimov_scenario),
                    "SignificanceError-": np.zeros_like(asimov_scenario),
                })

        fig.add_trace(
            go.Scatter(
                x=exposure_values, y=raw_significance, name="Gaussian",
                mode="lines", line=dict(color="black", width=2, dash="dot"),
                legendgroup="Gaussian", showlegend=False,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=exposure_values, y=smoothed_significance, name="Gaussian",
                mode="lines", line=dict(color="black"),
                legendgroup="Gaussian", legendgrouptitle=dict(text="Significance"),
                showlegend=True,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=exposure_values, y=significance_upper,
                mode="lines", marker=dict(color="#444"), line=dict(width=0),
                showlegend=False,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=exposure_values, y=significance_lower,
                marker=dict(color="#444"), line=dict(width=0), mode="lines",
                fillcolor="rgba(68, 68, 68, 0.3)", fill="tonexty", showlegend=False,
            ),
            row=1, col=1,
        )

        if _has_asimov:
            fig.add_trace(
                go.Scatter(
                    x=exposure_values, y=raw_asimov, name="Asimov",
                    mode="lines", line=dict(color="rgb(31,119,180)", width=2, dash="dot"),
                    legendgroup="Asimov", showlegend=False,
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=exposure_values, y=smoothed_asimov, name="Asimov",
                    mode="lines", line=dict(color="rgb(31,119,180)"),
                    legendgroup="Asimov", showlegend=True,
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=exposure_values, y=asimov_upper,
                    mode="lines", marker=dict(color="rgb(31,119,180)"), line=dict(width=0),
                    showlegend=False,
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=exposure_values, y=asimov_lower,
                    marker=dict(color="rgb(31,119,180)"), line=dict(width=0), mode="lines",
                    fillcolor="rgba(31,119,180,0.2)", fill="tonexty", showlegend=False,
                ),
                row=1, col=1,
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
            subfolder=_save_subfolder,
            filename=figure_name,
            rm=args.rewrite,
            debug=args.plot,
        )

        # 3-scenario plots: each asymmetry scenario (max/central/min) with background
        # normalization uncertainty bands.  Band = ideal (no bkg unc.) − realistic (with bkg unc.).
        _scenario_plot_defs = [
            ("MaxScenario",     "Maximum asymmetry (+13% PREM, +5% osc.)",
             significance_upper,    error_gaussian_upper,    asimov_upper    if _has_asimov else None),
            ("CentralScenario", "Central asymmetry estimate",
             smoothed_significance, error_gaussian_central,  smoothed_asimov if _has_asimov else None),
            ("MinScenario",     "Minimum asymmetry (-13% PREM, -5% osc.)",
             significance_lower,    error_gaussian_lower,    asimov_lower    if _has_asimov else None),
        ]
        for scenario_key, scenario_title, ideal_g, real_g, asimov_scenario in _scenario_plot_defs:
            sfig = make_subplots(rows=1, cols=1, subplot_titles=(
                f"{energy} — {scenario_title}<br>"
                f"min#Hits {ref_plot['NHits']:.0f}, min#OpHits {ref_plot['OpHits']:.0f}, max#AdjCl {ref_plot['AdjCl']:.0f}",
            ))

            # Gaussian: ideal (no bkg unc.) + shaded band down to ErrorGaussian (with σ_bkg)
            sfig.add_trace(go.Scatter(
                x=exposure_values, y=ideal_g, name="Gaussian (no bkg unc.)",
                mode="lines", line=dict(color="black", width=2),
                legendgroup="Gaussian", legendgrouptitle=dict(text="Significance"),
                showlegend=True,
            ))
            sfig.add_trace(go.Scatter(
                x=exposure_values, y=ideal_g,
                mode="lines", marker=dict(color="black"), line=dict(width=0),
                showlegend=False,
            ))
            sfig.add_trace(go.Scatter(
                x=exposure_values, y=real_g,
                mode="lines", marker=dict(color="black"), line=dict(width=0),
                fillcolor="rgba(68, 68, 68, 0.3)", fill="tonexty",
                name="Bkg. norm. unc. (Gaussian)", showlegend=True,
                legendgroup="Gaussian",
            ))

            # Asimov: plain line for this scenario — no bkg-unc band (normalization-invariant shape test)
            if _has_asimov:
                sfig.add_trace(go.Scatter(
                    x=exposure_values, y=asimov_scenario, name="Asimov",
                    mode="lines", line=dict(color="rgb(31,119,180)", width=2),
                    legendgroup="Asimov", legendgrouptitle=dict(text=""),
                    showlegend=True,
                ))

            sfig = format_coustom_plotly(
                sfig,
                tickformat=(".1f", ".0e"),
                add_units=False,
                title=f"Day-Night Asymmetry — {scenario_title}<br>{args.folder} — {config}",
                matches=(None, None),
                legend=dict(font=dict(size=14), bgcolor="rgba(255,255,255,0.7)"),
            )
            sfig.update_yaxes(
                tickformat=".1f", dtick=1, range=[0, 4],
                title="Significance (σ)",
            )
            sfig.update_xaxes(
                range=[-1, args.exposure], zeroline=False,
                title="Exposure (year)",
            )
            for sigma_line, cl in zip([1, 2, 3], [0.6827, 0.9545, 0.9973]):
                sfig.add_hline(y=sigma_line, line_dash="dash", line_color="black")
                sfig.add_annotation(
                    x=2, y=sigma_line + 0.2,
                    text=f"{100*cl:.2f}% CL", showarrow=False,
                )

            save_figure(
                sfig,
                save_path,
                config=config,
                name=name,
                subfolder=_save_subfolder,
                filename=f"{figure_name}_{scenario_key}",
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
                subfolder=_save_subfolder,
                filename=df_name,
                rm=args.rewrite,
                debug=True,
            )
