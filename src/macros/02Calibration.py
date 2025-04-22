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

config = parser.parse_args().config
name = parser.parse_args().name

configs = {config: [name]}

user_input = {
    "workflow": "CALIBRATION",
    "rewrite": True,
    "debug": False,
}

run, output = load_multi(
    configs, preset=user_input["workflow"], debug=user_input["debug"]
)
rprint(output)
run = compute_reco_workflow(
    run,
    configs,
    params={
        "DEFAULT_ENERGY_TIME": "TruthDriftTime",
        "DEFAULT_ADJCLENERGY_TIME": "TruthAdjClDriftTime",
    },
    workflow=user_input["workflow"],
    debug=user_input["debug"],
)

filtered_run, mask, output = compute_filtered_run(
    run,
    configs,
    params={("Reco", "CorrectedCharge"): ("bigger", 150)},
    presets=[user_input["workflow"]],
    debug=user_input["debug"],
)
rprint(output)
data = filtered_run["Reco"]

# Plot the calibration workflow
acc = 100
fit = {
    "color": "grey",
    "threshold": 0.4,
    "trimm": (2, 2),
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
    fit_popt = {}
    fig = make_subplots(rows=1, cols=1)
    for idx, (variable, variable_label) in enumerate(
        zip(["", "Electron"], ["Primary", "Cheated"])
    ):
        reco_popt[variable] = {}
        reco_perr[variable] = {}
        for name, nhit in product(configs[config], nhits):
            if len(data["ElectronK"][data["NHits"] >= nhit]) < 100:
                fit_f = nhit - 1
                break

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

            fit["func"] = "slope1"
            this_fig, reco_popt[variable][nhit], reco_perr[variable][nhit] = (
                get_hist2d_fit(
                    data["ElectronK"][data["NHits"] >= nhit],
                    data[f"{variable}Energy"][data["NHits"] >= nhit],
                    this_fig,
                    idx=(1, 2),
                    per=(1, 99),
                    acc=acc,
                    fit=fit,
                    nanz=True,
                    zoom=True,
                    debug=user_input["debug"],
                )
            )

            data[f"Calibrated{variable}Energy"][data["NHits"] >= nhit] = (
                data[f"{variable}Energy"][data["NHits"] >= nhit]
                - reco_popt[variable][nhit][0]
            )

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
                x, y, ref = get_hist1d(
                    x=res,
                    per=(1, 99),
                    acc=acc,
                    density=False,
                    debug=user_input["debug"],
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
                title=f"Reco Cluster - {variable_label} Charge Calibration",
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

            save_figure(
                this_fig,
                save_path,
                config,
                name,
                f"{variable}Charge_Calibration_2D_NHits{nhit}",
                rm=user_input["rewrite"],
                debug=user_input["debug"],
            )

        reco_popt_slope = []
        reco_popt_intercept = []
        reco_perr_slope = []
        reco_perr_intercept = []

        for nhit in nhits[:fit_f]:
            reco_popt_intercept.append(reco_popt[variable][nhit][0])
            reco_perr_intercept.append(reco_perr[variable][nhit][0])

        fig.add_trace(
            go.Scatter(
                x=nhits[:fit_f],
                y=reco_popt_intercept,
                error_y=dict(
                    type="data",
                    array=reco_perr_intercept,
                    visible=True,
                    color=compare[idx],
                ),
                mode="markers+lines",
                name=variable_label,
                line=dict(
                    color=compare[idx],
                ),
            ),
            row=1,
            col=1,
        )

        mean = np.mean(reco_popt_intercept)

        interp = interp1d(
            nhits[:fit_f],
            reco_popt_intercept,
            kind="cubic",
            bounds_error=False,
            fill_value=mean,
        )

        fit_popt[f"{variable}Charge"] = interp

        fig.add_trace(
            go.Scatter(
                x=nhits,
                y=interp(nhits),
                showlegend=False,
                line=dict(color=compare[idx]),
            )
        )

    fig = format_coustom_plotly(
        fig,
        ranges=(None, [-1, 0]),
        title=f"Reco Electron Charge Calibration {config}",
        matches=(None, None),
        legend_title="Energy Offset (MeV)",
        legend=dict(x=0.75, y=0.99),
    )

    fig.update_xaxes(title_text="#Hits in Primary Cluster")
    fig.update_yaxes(title_text="Offset Value", row=1, col=1)

    save_figure(
        fig,
        save_path,
        config,
        name,
        f"Energy_Correction",
        rm=user_input["rewrite"],
        debug=user_input["debug"],
    )

    save_pkl(
        fit_popt[f"{variable}Charge"],
        f"{root}/config/{config}/{name}/{config}_calib",
        None,
        None,
        filename=f"{config}_{variable.lower()}charge_calibration",
        rm=user_input["rewrite"],
        debug=user_input["debug"],
    )
