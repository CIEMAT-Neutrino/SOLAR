import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *

save_path = f"{root}/images/workflow/correction"
data_path = f"{root}/data/workflow/correction"

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

configs = {config: [name]}

user_input = {
    "workflow": "CORRECTION",
    "rewrite": args.rewrite,
    "debug": args.debug,
}


run, output = load_multi(
    configs, preset=user_input["workflow"], debug=user_input["debug"]
)
rprint(output)
run = compute_reco_workflow(
    run, configs, workflow=user_input["workflow"], debug=user_input["debug"]
)
filtered_run, mask, output = compute_filtered_run(
    run,
    configs,
    presets=[user_input["workflow"]],
    params={("Reco", "Time"): ("bigger", 500)},
    debug=user_input["debug"],
)
rprint(output)
data = filtered_run["Reco"]

# Plot the calibration workflow
per = (1, 99)
fit = {
    "color": "grey",
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
    info, params, output = get_param_dict(
        f"{root}/config/{config}/{config}", {}, output, debug=args.debug
    )
    for name, (charge, charge_label) in product(
        configs[config], zip(["", "Electron"], ["Primary", "Cheated"])
    ):

        #############################################################################
        ########################## Fit Drift Correction #############################
        #############################################################################

        fig = make_subplots(rows=1, cols=1)

        acc = get_default_acc(len(data["Generator"]))

        # Make a plot that shows the correlation between the neutrino energy and the number of hits in the primary cluster
        fig = make_subplots(rows=1, cols=1)
        max_bin = 0
        df_scan = []
        df_lifetime = []
        for nhit in nhits[:9]:
            this_filter_idx = np.where((data["NHits"] == nhit))[0]
            if len(this_filter_idx) < 1000:
                continue

            hist, bins = np.histogram(
                data[f"SignalParticleK"][this_filter_idx],
                bins=reco_energy_edges[3:],
            )
            density = np.sum(hist) * np.diff(bins)
            if np.max(hist / density) > max_bin:
                max_bin = np.max(hist / density)

            df_scan.append(
                {
                    "Config": config,
                    "Name": name,
                    "#Hits": nhit,
                    "Values": 0.5 * (bins[1:] + bins[:-1]),
                    "Density": hist / density,
                    "Counts": hist,
                    "Variable": "SignalParticleK",
                }
            )

            fig.add_trace(
                go.Scatter(
                    x=0.5 * (bins[1:] + bins[:-1]),
                    y=hist / density,
                    mode="lines",
                    line_shape="hvh",
                    name=f"{nhit}",
                    line=dict(color=colors[nhit % len(colors)]),
                ),
                row=1,
                col=1,
            )
        fig.update_layout(
            xaxis_title="True Neutrino Energy (MeV)",
            yaxis_title="Density",
        )

        format_coustom_plotly(
            fig,
            title=f"True Neutrino Energy - {config}",
            legend=dict(x=0.86, y=0.99),
            legend_title="#Hits",
            ranges=(
                [4, reco_energy_centers[-1]],
                [0, max_bin * 1.1],
            ),
            tickformat=(".0f", None),
            debug=user_input["debug"],
        )
        save_figure(
            fig,
            save_path,
            config,
            name,
            filename="SignalParticleKineticEnergy_vs_NHits",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )

        df = pd.DataFrame(df_scan)
        save_df(
            df,
            data_path,
            config,
            name,
            filename="NHit_Distributions",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )

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
                density=False,
                nanz=True,
                logz=False,
                zoom=True,
                debug=user_input["debug"],
            )
        )

        fig = format_coustom_plotly(
            fig,
            title=f"Electron-Lifetime Attenuation - {config}",
            matches=(None, None),
            tickformat=(".0f", None),
            log=(False, False),
            debug=user_input["debug"],
        )
        fig.update_layout(
            coloraxis=dict(colorbar=dict(title="Counts")),
            showlegend=False,
            xaxis_title="Time (us)",
            yaxis_title=f"Charge per Energy (ADC x tick / MeV)",
        )

        save_figure(
            fig,
            save_path,
            config,
            name,
            filename=f"{charge}Charge_Correction_Hist2D",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )
        # Energy computation
        data["Correction"] = np.exp(
            np.abs(data[f"{charge}Time"]) / corr_popt[f"{charge}Charge"][1]
        )
        data[f"Corrected{charge}Charge"] = data[f"{charge}Charge"] * data["Correction"]
        data[f"Corrected{charge}ChargePerMeV"] = (
            data[f"Corrected{charge}Charge"] / data["ElectronK"]
        )
        # Plot the corrected charge
        fig = make_subplots(rows=1, cols=2)
        for label, values in zip(
            ["Uncorrected", "Corrected"],
            [
                data[f"{charge}Charge"] / data["ElectronK"],
                data[f"Corrected{charge}ChargePerMeV"],
            ],
        ):
            x, y, h = get_hist2d(
                data[f"{charge}Time"],
                values,
                per=per,
                norm=False,
                acc=acc,
                # nanz=True,
                density=False,
                debug=False,
            )
            fig.add_trace(
                go.Scatter(
                    x=y,
                    y=np.mean(h.T, axis=1),
                    line=dict(shape="hvh"),
                    showlegend=True,
                    mode="lines",
                    name=f"{label}",
                ),
                row=1,
                col=2,
            )
        fig.add_trace(
            go.Heatmap(
                x=x,
                y=y,
                # Draw as nan if the value is less than 1
                z=np.where(h == 0, np.nan, h).T,
                colorscale="Turbo",
                colorbar=dict(title="Counts"),
            ),
            row=1,
            col=1,
        )
        fig.add_vline(
            x=y[np.argmax(np.mean(h.T, axis=1))],
            line_width=1,
            line_dash="dash",
            annotation_text=f"<br>Electron-Lifetime:<br>{1e-3*corr_popt[f'{charge}Charge'][1]:.2f} (ms)<br>Correction Factor:<br>{y[np.argmax(np.mean(h.T, axis=1))]:.2f} (ADC x tick / MeV)",
            annotation_position="top right",
            annotation=dict(
                yshift=-50,
                xshift=20,
            ),
            col=2,
            row=1,
        )
        fig = format_coustom_plotly(
            fig,
            title=f"Average Drift Electron Correction - {config}",
            legend_title="Drift Attenuation",
            legend=dict(x=0.84, y=0.1),
            matches=(None, None),
            tickformat=(".0f", None),
            log=(False, False),
            debug=user_input["debug"],
        )
        fig.update_layout(
            coloraxis=dict(colorscale="Turbo", colorbar=dict(title="Density")),
            yaxis_title=f"Charge per Energy (ADC x tick / MeV)",
            yaxis2_title="Counts",
        )
        fig.update_xaxes(title="Time (us)", col=1)
        fig.update_xaxes(title="Charge per Energy (ADC x tick / MeV)", col=2)
        save_figure(
            fig,
            save_path,
            config,
            name,
            filename=f"Corrected_{charge}Charge",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )

        correction_factor[f"{charge}Charge"] = {}
        correction_factor[f"{charge}ChargeError"] = {}
        for nhit in range(1, np.max(data["NHits"]) + 1, 1):
            this_filter_idx = np.where((data["NHits"] == nhit))[0]
            if len(this_filter_idx) < 1000:
                continue

            acc = get_default_acc(len(this_filter_idx))

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

            if nhit < 10:
                for correct in ["", "Corrected"]:
                    df_lifetime.append(
                        {
                            "Geometry": info["GEOMETRY"],
                            "Config": config,
                            "Name": name,
                            "#Hits": int(nhit),
                            "Time": data[f"{charge}Time"][this_filter_idx],
                            "ChargePerEnergy": data[f"{correct}{charge}Charge"][
                                this_filter_idx
                            ]
                            / data["ElectronK"][this_filter_idx],
                            "Corrected": True if correct == "Corrected" else False,
                        }
                    )

        # Plot correction factor.values() over correction factor.keys()
        x = np.asarray(list(correction_factor[f"{charge}Charge"].keys()))
        y = np.asarray(list(correction_factor[f"{charge}Charge"].values()))
        y_error = np.asarray(list(correction_factor[f"{charge}ChargeError"].values()))

        # Make a linear fit of the central values
        initial_guess = [100, 1, 1, 1]

        popt[f"{charge}Charge"], pcov[f"{charge}Charge"] = curve_fit(
            correction_func,
            x,
            y,
            p0=initial_guess,
            sigma=y_error,
            bounds=([0, 0, 0, 0], [1e3, 1, 1e3, 1]),
        )
        perr[f"{charge}Charge"] = np.sqrt(np.diag(pcov[f"{charge}Charge"]))

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
            title=f"{charge_label} Correction Factor - {config}",
        )
        format_coustom_plotly(fig, legend=dict(x=0.7, y=0.99))
        save_figure(
            fig,
            save_path,
            config,
            name,
            filename=f"{charge}Charge_Correction_Factor",
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
                f"Saved calibration parameters to: {root}/config/{config}/{name}/{config}_calib/{config}_{charge.lower()}charge_correction.json"
            )

        else:

            rprint(
                f"Found {root}/config/{config}/{name}/{config}_calib/{config}_{charge.lower()}charge_correction.json"
            )
            rprint(
                f"[yellow][WARNING][/yellow]: Please set rewrite to True to overwrite the file."
            )

        # Save the 2D histogram to df so that the columns are Time, rows are Charge, and values are Counts
        for correct in ["", "Corrected"]:
            df_lifetime.append(
                {
                    "Geometry": info["GEOMETRY"],
                    "Config": config,
                    "Name": name,
                    "#Hits": None,
                    "Time": data[f"{charge}Time"],
                    "ChargePerEnergy": data[f"{correct}{charge}Charge"]
                    / data["ElectronK"],
                    "Corrected": True if correct == "Corrected" else False,
                }
            )
        df = pd.DataFrame(df_lifetime)
        save_df(
            df,
            data_path,
            config,
            name,
            filename=f"{charge}Charge_Lifetime_Correction",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )
