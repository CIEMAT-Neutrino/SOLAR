import sys

sys.path.insert(0, "../../")

from lib import *

save_path = f"{root}/images/correction/"

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
    "workflow": "CORRECTION",
    "rewrite": parser.parse_args().rewrite,
    "debug": parser.parse_args().debug,
}


run, output = load_multi(
    configs, preset=user_input["workflow"], debug=user_input["debug"]
)
rprint(output)
run = compute_reco_workflow(
    run, configs, workflow=user_input["workflow"], debug=user_input["debug"]
)

filtered_run, mask, output = compute_filtered_run(
    run, configs, presets=[user_input["workflow"]], debug=user_input["debug"]
)
rprint(output)
data = filtered_run["Reco"]

# Plot the calibration workflow
acc = 100
per = (1, 99)
fit = {
    "color": "grey",
    "threshold": 0.4,
    "trimm": (2, 2),
    "spec_type": "max",
    "print": True,
    "opacity": 1,
    "print": False,
    "show": False,
}

y_min, y_max, corr_popt, corr_perr = {}, {}, {}, {}
corrected_popt, corrected_perr = {}, {}
correction_factor = {}


def correction_func(x, a, b, c, d):
    return a * np.exp(-b * x) + c / (1 + np.exp(-d * x))


popt, pcov, perr = {}, {}, {}

for config in configs:
    for name, (charge, charge_label) in product(
        configs[config], zip(["", "Electron"], ["Primary", "Cheated"])
    ):

        #############################################################################
        ########################## Fit Drift Correction #############################
        #############################################################################

        fig = make_subplots(rows=1, cols=2)

        fit["func"] = "exponential"
        fig, corr_popt[f"{charge}Charge"], corr_perr[f"{charge}Charge"] = (
            get_hist2d_fit(
                np.abs(data[f"{charge}Time"]),
                data[f"{charge}Charge"] / data["ElectronK"],
                fig,
                idx=(1, 1),
                per=per,
                acc=acc,
                fit=fit,
                density=True,
                nanz=True,
                logz=False,
                zoom=True,
                debug=user_input["debug"],
            )
        )

        x, y, h = get_hist2d(
            np.abs(data[f"{charge}Time"]),
            data[f"{charge}Charge"] / data["ElectronK"],
            per=per,
            norm=False,
            acc=acc,
            density=True,
            debug=False,
        )

        z = np.mean(h, axis=0)
        z_max = np.argmax(z)
        y_min[f"{charge}Charge"] = y[np.argmin(z[:z_max])]
        y_max[f"{charge}Charge"] = y[np.argmax(z)]
        fig.add_trace(
            go.Scatter(
                x=y,
                y=z,
                line=dict(shape="hvh"),
                showlegend=True,
                mode="lines",
            ),
            row=1,
            col=2,
        )

        # Add a vertical line to the minimum
        for value, pos, shift, label in zip(
            [y_min[f"{charge}Charge"], y_max[f"{charge}Charge"]],
            ["bottom left", "top right"],
            [[-10, 0], [20, -50]],
            ["Min", "Max"],
        ):
            fig.add_vline(
                x=value,
                line_width=1,
                line_dash="dash",
                annotation_text=f"{label}: {value:.2f}",
                annotation_position=pos,
                annotation=dict(
                    yshift=shift[1],
                    xshift=shift[0],
                ),
                col=2,
                row=1,
            )
        # Energy computation
        data["Correction"] = np.exp(
            np.abs(data[f"{charge}Time"]) / corr_popt[f"{charge}Charge"][1]
        )
        data[f"Corrected{charge}Charge"] = data[f"{charge}Charge"] * data["Correction"]
        data[f"Corrected{charge}ChargePerMeV"] = (
            data[f"Corrected{charge}Charge"] / data["ElectronK"]
        )

        fig = format_coustom_plotly(
            fig,
            title=f"Drift Electron Correction {config}",
            matches=(None, None),
            tickformat=(".2f", None),
            log=(False, False),
            debug=user_input["debug"],
        )
        fig.update_layout(
            coloraxis=dict(colorscale="Turbo", colorbar=dict(title="Density")),
            showlegend=False,
            xaxis_title="Time (tick)",
            yaxis_title=f"{charge_label} Charge / Energy (ADC x tick / MeV)",
            xaxis2_title=f"{charge_label} Charge / Energy (ADC x tick / MeV)",
            yaxis2_title="Density",
        )

        save_figure(
            fig,
            save_path,
            config,
            name,
            f"{charge}Charge_Correction_2D_Scan",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )

        correction_factor[f"{charge}Charge"] = {}
        correction_factor[f"{charge}ChargeError"] = {}
        for nhit in range(1, np.max(data["NHits"]) + 1, 1):
            this_filter_idx = np.where(
                (data["NHits"] == nhit)
                & (
                    data[f"{charge}Charge"] / data["ElectronK"]
                    > y_min[f"{charge}Charge"]
                )
            )[0]

            x, y, y_error = get_variable_scan(
                data[f"Corrected{charge}Charge"][this_filter_idx],
                data[f"Corrected{charge}ChargePerMeV"][this_filter_idx],
                variable="charge",
                per=per,
                norm=False,
                acc=acc,
                debug=False,
            )

            mean_y = np.mean(y)
            std_y = np.std(y)
            if np.isnan(mean_y) == False and mean_y > 0:
                correction_factor[f"{charge}Charge"][nhit] = mean_y
                correction_factor[f"{charge}ChargeError"][nhit] = std_y

        # Plot correction factor.values() over correction factor.keys()
        x = np.asarray(list(correction_factor[f"{charge}Charge"].keys()))
        y = np.asarray(list(correction_factor[f"{charge}Charge"].values()))
        y_error = np.asarray(list(correction_factor[f"{charge}ChargeError"].values()))
        i, j = 0, len(x) - 1

        # Make a linear fit of the central values
        initial_guess = [100, 1, 1, 1]

        popt[f"{charge}Charge"], pcov[f"{charge}Charge"] = curve_fit(
            correction_func,
            x[i:j],
            y[i:j],
            p0=initial_guess,
            sigma=y_error[i:j],
            bounds=([0, 0, 0, 0], [1e3, 1, 1e3, 1]),
        )

        perr[f"{charge}Charge"] = np.sqrt(np.diag(pcov[f"{charge}Charge"]))
        print(popt[f"{charge}Charge"])

        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                error_y=dict(
                    type="data",
                    array=y_error,
                    visible=True,
                    color=colors[-2],
                ),
                mode="lines",
                line_shape="hvh",
                marker=dict(color=colors[-2]),
                name="Correction Factor",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=correction_func(x, *popt[f"{charge}Charge"]),
                mode="lines",
                line_dash="dash",
                marker=dict(color="red"),
                # name=f"Fit = {popt[f'{charge}Charge'][0]:.1f} * e^(-x/{popt[f'{charge}Charge'][1]:.1f}) + {popt[f'{charge}Charge'][2]:.1f}",
                name=f"Fit",
            )
        )

        fig.update_layout(
            xaxis_title="Number of Hits",
            yaxis_title="Correction Factor (ADC x tick / MeV)",
            title=f"{charge_label} Correction Factor vs Number of Hits",
        )
        format_coustom_plotly(fig, legend=dict(x=0.7, y=0.99))
        save_figure(
            fig,
            save_path,
            config,
            name,
            f"{charge}Charge_Correction_Factor",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )

        if not os.path.exists(f"{root}/config/{config}/{name}/{config}_calib/"):
            os.makedirs(f"{root}/config/{config}/{name}/{config}_calib/")

        if (
            not os.path.exists(
                f"{root}/config/{config}/{name}/{config}_calib/{config}_{charge.lower()}charge_correction.json"
            )
            or user_input["rewrite"]
        ):
            with open(
                f"{root}/config/{config}/{name}/{config}_calib/{config}_{charge.lower()}charge_correction.json",
                "w",
            ) as f:
                json.dump(
                    {
                        "CHARGE_AMP": corr_popt[f"{charge}Charge"][0],
                        "CHARGE_AMP_ERROR": perr[f"{charge}Charge"][0],
                        "ELECTRON_TAU": corr_popt[f"{charge}Charge"][1],
                        "ELECTRON_TAU_ERROR": perr[f"{charge}Charge"][1],
                        "CHARGE_PER_ENERGY_TRIMM": y_min[f"{charge}Charge"],
                        "CORRECTION_AMP": popt[f"{charge}Charge"][0],
                        "CORRECTION_AMP_ERROR": perr[f"{charge}Charge"][0],
                        "CORRECTION_DECAY": popt[f"{charge}Charge"][1],
                        "CORRECTION_DECAY_ERROR": perr[f"{charge}Charge"][1],
                        "CORRECTION_CONST": popt[f"{charge}Charge"][2],
                        "CORRECTION_CONST_ERROR": perr[f"{charge}Charge"][2],
                        "CORRECTION_SIGMOID": popt[f"{charge}Charge"][3],
                        "CORRECTION_SIGMOID_ERROR": perr[f"{charge}Charge"][3],
                    },
                    f,
                )
            rprint(
                f"-> Saved calibration parameters to {root}/config/{config}/{name}/{config}_calib/{config}_{charge.lower()}charge_correction.json"
            )

        else:

            rprint(
                f"-> Found {root}/config/{config}/{name}/{config}_calib/{config}_{charge.lower()}charge_correction.json"
            )
            rprint(f"-> Please set rewrite to True to overwrite the file")
