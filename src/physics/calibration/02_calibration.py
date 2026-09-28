import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *

save_path = f"{root}/output/images/workflow/calibration"
data_path = f"{root}/output/data/workflow/calibration"

if not os.path.exists(save_path):
    os.makedirs(save_path)

# Define flags for the analysis config and name with the python parser
parser = argparse.ArgumentParser(
    description="Plot the energy distribution of the particles"
)
parser.add_argument(
    "--config",
    type=str,
    help="The configuration to load",
    default="hd_1x2x6",
)
parser.add_argument(
    "--name", type=str, help="The name of the configuration", default="marley_official"
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()
config = args.config
name = args.name

analysis_info = load_analysis_info(str(root))
_calib_base = analysis_info.get("DATA_PATHS", {}).get(
    "CALIBRATION", "/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/calibration"
)

configs = {config: [name]}

user_input = {"workflow": "CALIBRATION"}

run, output = load_multi(configs, preset=user_input["workflow"], debug=args.debug)
rprint(output)
run = compute_reco_workflow(
    run,
    configs,
    params={
        "DEFAULT_ENERGY_TIME": "Time",
        "DEFAULT_ADJCL_ENERGY_TIME": "AdjClTime",
    },
    workflow=user_input["workflow"],
    debug=args.debug,
)

filtered_run, mask, output = compute_filtered_run(
    run,
    configs,
    presets=[user_input["workflow"]],
    debug=args.debug,
)
rprint(output)
data = filtered_run["Reco"]

# Plot the calibration workflow
per = (1, 99)
fit = {
    "color": "grey",
    "opacity": 1,
    "print": False,
    "func": "linear",
    "show": True,
    "spec_type": "max",
}

reco_valid = {}
reco_nhit = {}
reco_popt = {}
reco_perr = {}
data[f"CalibratedEnergy"] = np.zeros(len(data["ElectronK"]))
data[f"CalibratedElectronEnergy"] = np.zeros(len(data["ElectronK"]))
for config in configs:
    info, params, output = get_param_dict(
        f"{root}/config/{config}/{config}", {}, output, debug=args.debug
    )
    for name in configs[config]:
        for idx, (variable, variable_label) in enumerate(
            zip(["", "Electron"], ["Primary", "Cheated"])
        ):
            df_corrected = []
            corrected_list = []
            gauss_rows = []
            reco_valid[variable] = []
            reco_nhit[variable] = []
            reco_popt[variable] = {}
            reco_perr[variable] = {}
            fig = make_subplots(rows=1, cols=1)

            for nhit in nhits:
                acc = get_default_acc(len(data["ElectronK"][data["NHits"] == nhit]))
                fit["threshold"] = (
                    len(data["ElectronK"][data["NHits"] == nhit])
                    * params["CALIBRATION_THRESHOLD"]
                )

                if len(data["ElectronK"][data["NHits"] == nhit]) < 1000:
                    rprint(
                        f"Skipping NHit {nhit} due to low statistics ({len(data['ElectronK'][data['NHits'] == nhit])})"
                    )
                    continue

                this_fig = make_subplots(
                    rows=1,
                    cols=3,
                    subplot_titles=(
                        "Charge Calibration",
                        "Energy Smearing",
                        "Energy Reconstruction",
                    ),
                )

                x, y, z = get_hist2d(
                    data["ElectronK"][data["NHits"] == nhit],
                    data[f"Corrected{variable}Charge"][data["NHits"] == nhit],
                    acc=acc,
                )

                z[z == 0] = np.nan
                this_fig.add_trace(
                    go.Heatmap(
                        x=x,
                        y=y,
                        z=z.T,
                        coloraxis="coloraxis",
                    ),
                    row=1,
                    col=1,
                )

                fit["bounds"] = ([0.1, -1], [1.1, 9])
                if nhit == 1:
                    fit["trimm"] = params["FIRST_CALIBRATION_TRIM"]
                else:
                    fit["trimm"] = params["CALIBRATION_TRIM"]

                this_fig, reco_popt[variable][nhit], reco_perr[variable][nhit] = (
                    get_hist2d_fit(
                        data["ElectronK"][data["NHits"] == nhit],
                        data[f"{variable}Energy"][data["NHits"] == nhit],
                        this_fig,
                        idx=(1, 2),
                        per=per,
                        acc="y",
                        fit=fit,
                        nanz=True,
                        zoom=True,
                        debug=args.debug,
                    )
                )

                reco_nhit[variable].append(nhit)
                if reco_popt[variable][nhit] is None:
                    reco_valid[variable].append(False)
                    reco_popt[variable][nhit] = [1, 0]
                    reco_perr[variable][nhit] = [0, 0]
                else:
                    reco_valid[variable].append(True)

                data[f"Calibrated{variable}Energy"][data["NHits"] == nhit] = (
                    data[f"{variable}Energy"][data["NHits"] == nhit]
                    - reco_popt[variable][nhit][1]
                ) / reco_popt[variable][nhit][0]

                for energy, label in zip(
                    [f"{variable}Energy", f"Calibrated{variable}Energy"],
                    ["Corrected", "Calibrated"],
                ):
                    res = (
                        data["ElectronK"][data["NHits"] == nhit]
                        - data[energy][data["NHits"] == nhit]
                    )
                    # Find the resolution
                    rms = np.sqrt(np.mean(np.power(res, 2)))
                    rms_error = np.sqrt(np.mean(np.power(res, 2)) / np.sqrt(len(res)))

                    res_edges = np.linspace(-5, 10, 50)
                    res_centers = 0.5 * (res_edges[:-1] + res_edges[1:])
                    x, y, sigma, ref, output = get_hist1d(
                        x=res,
                        per=per,
                        acc=res_edges,
                        norm=False,
                        density=False,
                        debug=args.debug,
                    )

                    this_fig.add_trace(
                        go.Scatter(
                            x=res_centers,
                            y=y,
                            line=dict(shape="hvh"),
                            showlegend=True,
                            mode="lines",
                            name=f"{label} RMS: {rms:.1f}",
                        ),
                        row=1,
                        col=3,
                    )

                this_fig = format_coustom_plotly(
                    this_fig,
                    matches=(None, None),
                    title=f"Primary Cluster - NHit {nhit} - {variable_label} Charge Calibration {config}",
                )

                this_fig.update_layout(
                    coloraxis=dict(colorscale="Turbo", colorbar=dict(title="Counts")),
                    legend=dict(x=0.86, y=0.92, traceorder="normal", orientation="v"),
                    barmode="overlay",
                    xaxis1_title="True Electron Energy (MeV)",
                    xaxis2_title="True Electron Energy (MeV)",
                    yaxis1_title="Corrected Charge (ADC x tick)",
                    yaxis2_title="Reco Electron Energy (MeV)",
                    yaxis3_title="Counts",
                    xaxis3_title="Reco Energy Error (MeV)",
                )

                if nhit % 2 == 1:
                    save_figure(
                        this_fig,
                        save_path,
                        config,
                        name,
                        filename=f"{variable}Charge_Calibration_2D_NHits{nhit}",
                        rm=args.rewrite,
                        debug=args.debug,
                    )
                    if output is not None or output != "":
                        rprint(output)

            # Resolution vs NHit threshold: for each cumulative NHits >= n cut, fit a
            # single Gaussian to the fractional (True - Calibrated{variable}Energy) /
            # True residual and keep its width as the resolution at that threshold.
            # Fractional (not absolute True - Reco in MeV) is used because the true
            # energy scale sampled shifts with the NHit cut (more hits <-> longer,
            # higher-energy tracks), so only a fractional residual gives sigma values
            # that are actually comparable across thresholds; per-NHit std(residual)
            # is unreliable near threshold (occasional near-zero True energies blow up
            # the ratio), so the histogram + Gaussian fit is preferred over a raw RMS.
            def _gauss_model(x, a, mu, sigma):
                return gauss(x, [a, mu, sigma])

            truth_full = np.asarray(data["ElectronK"])
            reco_full = np.asarray(data[f"Calibrated{variable}Energy"])
            for nhit in sorted(set(nhits)):
                cut = (
                    (data["NHits"] >= nhit) & (truth_full != 0) & (reco_full != 0)
                )
                n_cut = int(np.count_nonzero(cut))
                if n_cut < 200:
                    continue

                residual = (truth_full[cut] - reco_full[cut]) / truth_full[cut]
                residual = residual[np.isfinite(residual)]
                if len(residual) < 200:
                    continue

                lo, hi = np.percentile(residual, [1, 99])
                if hi <= lo:
                    continue
                res_edges_g = np.linspace(lo, hi, 61)
                hist_counts_g, _ = np.histogram(residual, bins=res_edges_g)
                res_centers_g = 0.5 * (res_edges_g[:-1] + res_edges_g[1:])
                if np.count_nonzero(hist_counts_g) < 4:
                    continue

                # Normalize to a probability density (area = 1) so distributions from
                # very different-sized NHit-threshold cuts (n=264174 down to n=1389)
                # sit on the same y-scale and can be overlaid/compared directly.
                bin_width_g = res_edges_g[1] - res_edges_g[0]
                n_total_g = np.sum(hist_counts_g)
                hist_y_g = hist_counts_g / (n_total_g * bin_width_g)
                sigma_w = np.where(
                    hist_counts_g > 0,
                    np.sqrt(hist_counts_g) / (n_total_g * bin_width_g),
                    np.inf,
                )

                p0 = [np.max(hist_y_g), np.mean(residual), max(np.std(residual), 1e-3)]
                bounds = ([0, -np.inf, 1e-6], [np.inf, np.inf, np.inf])
                try:
                    popt, pcov = curve_fit(
                        _gauss_model,
                        res_centers_g,
                        hist_y_g,
                        p0=p0,
                        sigma=sigma_w,
                        bounds=bounds,
                        maxfev=20000,
                    )
                except (RuntimeError, ValueError):
                    popt, pcov = curve_fit(
                        _gauss_model,
                        res_centers_g,
                        hist_y_g,
                        p0=p0,
                        bounds=bounds,
                        maxfev=20000,
                    )
                perr = np.sqrt(np.diag(pcov))

                gauss_rows.append(
                    {
                        "Geometry": info["GEOMETRY"],
                        "Config": config,
                        "Name": name,
                        "Variable": variable_label,
                        "Energy": f"Calibrated{variable}Energy",
                        "Truth": "ElectronK",
                        "#HitsThreshold": int(nhit),
                        "NEvents": n_cut,
                        "Values": res_centers_g,
                        "ValuesUnit": "(True - Reco) / True",
                        "Counts": hist_y_g,
                        "CountsUnit": "",
                        "RawCounts": hist_counts_g,
                        # No live callable stored here -- a function object pickles by
                        # module reference (lib.fitting.gauss), which breaks when this
                        # pkl is opened outside this repo (see the external plotting
                        # repo's ModuleNotFoundError: No module named 'lib.fitting').
                        # FitFunctionFormula is the portable description of the fit.
                        "FitFunctionLabel": "Gaussian",
                        "FitFunctionFormula": "a * exp(-0.5 * ((x - mu) / sigma)^2)",
                        "Params": popt,
                        "ParamsLabel": ["Amplitude", "Mean", "Sigma"],
                        "ParamsFormat": [".1f", ".3f", ".3f"],
                        "ParamsError": perr,
                        "ParamsUnit": [
                            "",
                            "",
                            "",
                        ],
                        "p0": popt[0],
                        "p1": popt[1],
                        "p2": popt[2],
                        "Amplitude": popt[0],
                        "Mean": popt[1],
                        "MeanError": perr[1],
                        "Resolution": popt[2],
                        "ResolutionError": perr[2],
                    }
                )

            for nhit, (energy, label) in product(
                np.arange(1, 9),
                zip(
                    [f"Raw{variable}Energy", f"Calibrated{variable}Energy"],
                    ["Raw", "Calibrated"],
                ),
            ):
                df_corrected.append(
                    {
                        "Geometry": info["GEOMETRY"],
                        "Config": config,
                        "Name": name,
                        "#Hits": int(nhit),
                        "Threshold": None,
                        "RawEnergy": data[energy][data["NHits"] == nhit],
                        "TrueEnergy": data["ElectronK"][data["NHits"] == nhit],
                        "Calibrated": label == "Calibrated",
                    }
                )

                df_corrected.append(
                    {
                        "Geometry": info["GEOMETRY"],
                        "Config": config,
                        "Name": name,
                        "#Hits": None,
                        "Threshold": int(nhit),
                        "RawEnergy": data[energy][data["NHits"] >= nhit],
                        "TrueEnergy": data["ElectronK"][data["NHits"] >= nhit],
                        "Calibrated": label == "Calibrated",
                    }
                )

            reco_popt_nhit = []
            reco_popt_slope = []
            reco_perr_slope = []
            reco_popt_intercept = []
            reco_perr_intercept = []

            for jdx, nhit in enumerate(reco_nhit[variable]):
                if reco_valid[variable][jdx]:
                    reco_popt_nhit.append(nhit)
                    reco_popt_slope.append(reco_popt[variable][nhit][0])
                    reco_popt_intercept.append(reco_popt[variable][nhit][1])
                    reco_perr_slope.append(reco_perr[variable][nhit][0])
                    reco_perr_intercept.append(reco_perr[variable][nhit][1])

            for jdx, (
                popt_label,
                this_reco_nhit,
                this_reco_popt,
                this_reco_perr,
            ) in enumerate(
                zip(
                    ["Slope", "Intercept"],
                    [reco_popt_nhit, reco_popt_nhit],
                    [reco_popt_slope, reco_popt_intercept],
                    [reco_perr_slope, reco_perr_intercept],
                )
            ):
                corrected_list.append(
                    {
                        "Geometry": info["GEOMETRY"],
                        "Config": config,
                        "Name": name,
                        "Variable": popt_label,
                        "#Hits": this_reco_nhit,
                        "FitValue": this_reco_popt,
                        "Error": this_reco_perr,
                    }
                )

                fig.add_trace(
                    go.Scatter(
                        x=this_reco_nhit,
                        y=this_reco_popt,
                        error_y=dict(
                            type="data",
                            array=this_reco_perr,
                            visible=True,
                            color=compare[jdx],
                        ),
                        mode="markers",
                        # line_shape="hvh",
                        name=f"{variable_label} {popt_label}",
                        line=dict(
                            color=compare[jdx],
                        ),
                    ),
                    row=1,
                    col=1,
                )

            slope = interp1d(
                reco_popt_nhit,
                reco_popt_slope,
                kind="slinear",
                bounds_error=False,
                fill_value=(reco_popt_slope[0], np.mean(reco_popt_slope[-3:])),
            )
            intercept = interp1d(
                reco_popt_nhit,
                reco_popt_intercept,
                kind="slinear",
                bounds_error=False,
                fill_value=(reco_popt_intercept[0], np.mean(reco_popt_intercept[-3:])),
            )

            # Add extrapolated slope and intercept to the figure
            for jdx, (func, fit_label) in enumerate(
                zip(
                    [slope, intercept],
                    [f"{variable_label} Slope", f"{variable_label} Intercept"],
                )
            ):
                fig.add_trace(
                    go.Scatter(
                        x=nhits[:14],
                        y=func(nhits[:14]),
                        mode="lines",
                        line_shape="spline",
                        line=dict(
                            color=compare[jdx],
                            dash="dot",
                        ),
                        name=f"Extrapolated {fit_label}",
                        showlegend=False,
                    ),
                    row=1,
                    col=1,
                )

            save_pkl(
                slope,
                f"{_calib_base}/{config}/{name}/calib",
                None,
                None,
                filename=f"{config}_{variable.lower()}charge_slope_calibration",
                rm=args.rewrite,
                debug=args.debug,
            )

            save_pkl(
                intercept,
                f"{_calib_base}/{config}/{name}/calib",
                None,
                None,
                filename=f"{config}_{variable.lower()}charge_intercept_calibration",
                rm=args.rewrite,
                debug=args.debug,
            )

            fig = format_coustom_plotly(
                fig,
                title=f"Reco Electron Energy Calibration {config}",
                legend_title="Calibration Variable",
                legend=dict(x=0.72, y=0.99),
            )

            fig.update_xaxes(title_text="#Hits in Primary Cluster")
            fig.update_yaxes(title_text="Fit Value")

            save_figure(
                fig,
                save_path,
                config,
                name,
                filename=f"{variable}Charge_Calibration",
                rm=args.rewrite,
                debug=args.debug,
            )

            for df_list, df_filename in zip(
                [corrected_list, df_corrected, gauss_rows],
                [
                    f"{variable_label}_Calibration_Fit",
                    f"{variable_label}Energy_Electron_Calibration",
                    f"{variable_label}_Resolution_GaussianFit",
                ],
            ):
                df = pd.DataFrame(df_list)
                save_df(
                    df,
                    data_path,
                    config,
                    name,
                    filename=df_filename,
                    rm=args.rewrite,
                    debug=args.debug,
                )
