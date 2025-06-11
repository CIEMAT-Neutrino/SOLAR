import sys

sys.path.insert(0, "../../")

from lib import *

save_path = f"{root}/images/calibration/"

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
    default="hd_1x2x6_centralAPA",
)
parser.add_argument(
    "--name", type=str, help="The name of the configuration", default="marley_signal"
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()
config = args.config
name = args.name

configs = {config: [name]}

user_input = {"workflow": "CALIBRATION"}

run, output = load_multi(configs, preset=user_input["workflow"], debug=args.debug)
rprint(output)
run = compute_reco_workflow(
    run,
    configs,
    params={
        "DEFAULT_ENERGY_TIME": "TruthDriftTime",
        "DEFAULT_ADJCLENERGY_TIME": "TruthAdjClDriftTime",
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
acc = int(len(data["Generator"]) / 200)
if acc > 100:
    acc = 100
fit = {
    "color": "grey",
    "threshold": 0.4,
    "trimm": (int(acc/20), int(acc/3)),
    "spec_type": "max",
    "print": False,
    "show": False,
    "opacity": 1,
}


reco_popt = {}
reco_perr = {}
data[f"CalibratedEnergy"] = np.zeros(len(data["ElectronK"]))
data[f"CalibratedElectronEnergy"] = np.zeros(len(data["ElectronK"]))
for config in configs:
    for name in configs[config]:
        fit_popt = {}
        fig = make_subplots(rows=1, cols=1)

        for idx, (variable, variable_label) in enumerate(
            zip(["", "Electron"], ["Primary", "Cheated"])
        ):
            reco_popt[variable] = {}
            reco_perr[variable] = {}
            for nhit in nhits:
                if len(data["ElectronK"][data["NHits"] >= nhit]) < 100:
                    fit_f = nhit - 1
                    break
                else:
                    fit_f = nhit

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
                    data["ElectronK"][data["NHits"] >= nhit],
                    data[f"Corrected{variable}Charge"][data["NHits"] >= nhit],
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

                fit["func"] = "linear"
                this_fig, reco_popt[variable][nhit], reco_perr[variable][nhit] = (
                    get_hist2d_fit(
                        data["ElectronK"][data["NHits"] >= nhit],
                        data[f"{variable}Energy"][data["NHits"] >= nhit],
                        this_fig,
                        idx=(1, 2),
                        per=None,
                        acc=acc,
                        fit=fit,
                        nanz=True,
                        zoom=True,
                        debug=args.debug,
                    )
                )

                data[f"Calibrated{variable}Energy"][data["NHits"] >= nhit] = (
                    data[f"{variable}Energy"][data["NHits"] >= nhit]
                    - reco_popt[variable][nhit][1]
                ) / reco_popt[variable][nhit][0]

                for energy, label in zip(
                    [f"{variable}Energy", f"Calibrated{variable}Energy"],
                    ["Corrected", "Calibrated"],
                ):
                    # for idx, energy in enumerate(["Energy", "ElectronK"]):
                    res = (
                        data["ElectronK"][data["NHits"] >= nhit]
                        - data[energy][data["NHits"] >= nhit]
                    )
                    # Find the resolution
                    rms = np.sqrt(np.mean(np.power(res, 2)))
                    rms_error = np.sqrt(np.mean(np.power(res, 2)) / np.sqrt(len(res)))

                    # print(f"-> {label} Energy RMS: {rms:.1f}")
                    x, y, ref, output = get_hist1d(
                        x=res,
                        per=None,
                        acc=acc,
                        density=False,
                        debug=args.debug,
                    )

                    this_fig.add_trace(
                        go.Scatter(
                            x=x,
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
                    title=f"Reco Cluster - {variable_label} Charge Calibration {config}",
                )

                this_fig.update_layout(
                    coloraxis=dict(colorscale="Turbo", colorbar=dict(title="Density")),
                    legend=dict(x=0.86, y=0.9, traceorder="normal", orientation="v"),
                    barmode="overlay",
                    xaxis1_title="True Electron Energy (MeV)",
                    xaxis2_title="True Electron Energy (MeV)",
                    yaxis1_title="Corrected Charge (ADC x tick)",
                    yaxis2_title="Reco Electron Energy (MeV)",
                    yaxis3_title="Density",
                    xaxis3_title="Reco Energy Error (MeV)",
                )

                if nhit < 4:
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

            reco_popt_slope = []
            reco_popt_intercept = []
            reco_perr_slope = []
            reco_perr_intercept = []

            for nhit in nhits[:fit_f]:
                reco_popt_slope.append(reco_popt[variable][nhit][0])
                reco_popt_intercept.append(reco_popt[variable][nhit][1])
                reco_perr_slope.append(reco_perr[variable][nhit][0])
                reco_perr_intercept.append(reco_perr[variable][nhit][1])
            
            for popt_label, this_reco_popt, this_reco_perr, dash in zip(["Slope", "Intercept"], [reco_popt_slope, reco_popt_intercept],
                                             [reco_perr_slope, reco_perr_intercept], ["solid", "dash"]):
                fig.add_trace(
                    go.Scatter(
                        x=nhits[:fit_f],
                        y=this_reco_popt,
                        error_y=dict(
                            type="data",
                            array=this_reco_perr,
                            visible=True,
                            color=compare[idx],
                        ),
                        mode="markers+lines",
                        line_shape="hvh",
                        line_dash=dash,
                        name=f"{variable_label} {popt_label}",
                        line=dict(
                            color=compare[idx],
                        ),
                    ),
                    row=1,
                    col=1,
                )

            mean_slope = np.mean(reco_popt_slope)
            mean_intercept = np.mean(reco_popt_intercept)
            slope = interp1d(
                nhits[:fit_f],
                reco_popt_slope,
                kind="linear",
                bounds_error=False,
                fill_value=mean_slope,
            )
            intercept = interp1d(
                nhits[:fit_f],
                reco_popt_intercept,
                kind="linear",
                bounds_error=False,
                fill_value=mean_intercept,
            )

            save_pkl(
                slope,
                f"{root}/config/{config}/{name}/{config}_calib",
                None,
                None,
                filename=f"{config}_{variable.lower()}charge_slope_calibration",
                rm=args.rewrite,
                debug=args.debug,
            )

            save_pkl(
                intercept,
                f"{root}/config/{config}/{name}/{config}_calib",
                None,
                None,
                filename=f"{config}_{variable.lower()}charge_intercept_calibration",
                rm=args.rewrite,
                debug=args.debug,
            )

        fig = format_coustom_plotly(
            fig,
            title=f"Reco Electron Energy Calibration {config}",
            matches=(None, None),
            legend_title="Energy Offset (MeV)",
            legend=dict(x=0.72, y=0.99),
        )

        fig.update_xaxes(title_text="#Hits in Primary Cluster")
        fig.update_yaxes(title_text="Offset Value", row=1, col=1)

        save_figure(
            fig,
            save_path,
            config,
            name,
            filename=f"Charge_Calibration",
            rm=args.rewrite,
            debug=args.debug,
        )
