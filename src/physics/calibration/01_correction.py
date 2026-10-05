import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

from itertools import product
from scipy.optimize import curve_fit
from plotly.subplots import make_subplots
from rich import print as rprint

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import (
    root,
    load_multi,
    compute_reco_workflow,
    compute_filtered_run,
    get_param_dict,
    get_default_acc,
    get_hist2d,
    get_hist2d_fit,
    get_variable_scan,
    format_coustom_plotly,
    save_figure,
    save_df,
    get_project_root,
    get_default_nhits,
    get_default_energies,
)

root = get_project_root()
nhits = get_default_nhits(root)
reco_energy_edges, reco_energy_centers, reco_ebin = get_default_energies(
    root, "RECO_ENERGY"
)

pio.templates.default = "none"
colors = px.colors.qualitative.Prism

save_path = f"{root}/output/images/workflow/correction"
data_path = f"{root}/output/data/workflow/correction"

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
parser.add_argument(
    "--factor_estimator",
    type=str,
    choices=["peak", "profile"],
    default="peak",
    help=(
        "How F0(NHits) is taken from the corrected charge per true MeV of each NHits value. "
        "'peak' (default since 2026-10-05): the most probable value of Q_corr / E_true, so that the core of "
        "E / E_true peaks at 1 in every NHits group. 'profile': the previous unweighted mean of the "
        "charge-binned profile, which over-weights sparse high-charge bins and biases low-NHits clusters low."
    ),
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
if args.factor_estimator == "peak":
    # The CORRECTION workflow stops before compute_cluster_time; the F0 peak needs the flash-matched
    # drift time the analysis uses (DEFAULT_ENERGY_TIME = RecoDriftTime).
    try:
        from lib.cluster import compute_cluster_time
        run, _time_output, _ = compute_cluster_time(run, configs, debug=False)
    except Exception as exc:
        rprint(f"[yellow][WARNING][/yellow] compute_cluster_time failed ({exc}); F0 peak falls back to Corrected<charge>ChargePerMeV")
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


def most_probable_ratio(values):
    """Most probable value of a charge-per-MeV distribution and its uncertainty.

    Histogram between the 1st and 99th percentiles (Freedman-Diaconis bins, 3-bin smoothing), take the
    maximum, then iterate the mean of the values within +-1 sigma_core of it (sigma_core from the MAD of
    the same window) until it moves by less than 1e-4 of itself. Returns (peak, sigma_core / sqrt(n_core)).
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    lo, hi = np.percentile(v, [1, 99])
    core = v[(v >= lo) & (v <= hi)]
    iqr = np.subtract(*np.percentile(core, [75, 25]))
    width = 2 * iqr / max(len(core), 1) ** (1 / 3) if iqr > 0 else (hi - lo) / 50
    nbins = int(np.clip((hi - lo) / width, 20, 400))
    h, edges = np.histogram(core, bins=nbins, range=(lo, hi))
    hs = np.convolve(h, np.ones(3) / 3, mode="same")
    peak = 0.5 * (edges[np.argmax(hs)] + edges[np.argmax(hs) + 1])
    sigma = 1.4826 * np.median(np.abs(core - np.median(core)))
    for _ in range(50):
        window = core[np.abs(core - peak) <= sigma]
        if len(window) < 20:
            break
        sigma = max(1.4826 * np.median(np.abs(window - np.median(window))), 1e-9)
        new_peak = float(np.mean(window))
        if abs(new_peak - peak) < 1e-4 * abs(peak):
            peak = new_peak
            break
        peak = new_peak
    window = core[np.abs(core - peak) <= sigma]
    return float(peak), float(sigma / np.sqrt(max(len(window), 1)))


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
        df_hits, df_scan = [], []
        for variable in ["SignalParticleK", "ElectronK"]:
            fig = make_subplots(rows=1, cols=1)
            max_bin = 0
            df_fit = []
            df_lifetime = []

            df_hits.append(
                {
                    "Config": config,
                    "Name": name,
                    "#Hits": data["NHits"].astype(int),
                    "Purity": data["Purity"],
                    "Energy": data[variable],
                    "Variable": variable,
                }
            )

            for nhit in nhits[:9]:
                this_filter_idx = np.where((data["NHits"] == nhit))[0]
                if len(this_filter_idx) < 100:
                    continue

                hist, bins = np.histogram(
                    data[variable][this_filter_idx],
                    bins=(
                        reco_energy_edges[3:]
                        if variable == "SignalParticleK"
                        else reco_energy_edges
                    ),
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
                        "Variable": variable,
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
                xaxis_title=(
                    "True Neutrino Energy (MeV)"
                    if variable == "SignalParticleK"
                    else "True Electron Energy (MeV)"
                ),
                yaxis_title="Density",
            )

            format_coustom_plotly(
                fig,
                title=(
                    f"True Neutrino Energy - {config}"
                    if variable == "SignalParticleK"
                    else f"True Electron Energy (MeV) - {config}"
                ),
                legend=dict(x=0.86, y=0.99),
                legend_title="#Hits",
                ranges=(
                    (
                        [4, reco_energy_centers[-1]]
                        if variable == "SignalParticleK"
                        else [0, reco_energy_centers[-1]]
                    ),
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
                filename=f"{variable}_vs_NHits",
                rm=user_input["rewrite"],
                debug=user_input["debug"],
            )

        save_df(
            pd.DataFrame(df_scan),
            data_path,
            config,
            name,
            filename="NHit_Distributions",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )

        save_df(
            pd.DataFrame(df_hits),
            data_path,
            config,
            name,
            filename="Cluster_Distributions",
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
        _time_branch = json.load(open(f"{root}/config/{config}/{config}_params.json")).get("DEFAULT_ENERGY_TIME", "RecoDriftTime")
        analysis_charge_per_mev = None
        if args.factor_estimator == "peak":
            if _time_branch in data and "Charge" in data:
                analysis_charge_per_mev = (
                    np.asarray(data["Charge"], dtype=float)
                    * np.exp(np.abs(np.asarray(data[_time_branch], dtype=float)) / corr_popt[f"{charge}Charge"][1])
                    / np.asarray(data["ElectronK"], dtype=float)
                )
            else:
                rprint(f"[yellow][WARNING][/yellow] {_time_branch} or Charge not available: F0 peak taken on Corrected{charge}ChargePerMeV")
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
            if args.factor_estimator == "peak":
                # The analysis (lib/cluster.compute_cluster_energy) divides the primary-cluster charge,
                # lifetime-corrected with the flash-matched drift time (DEFAULT_ENERGY_TIME), by F0(NHits).
                # The peak is taken on exactly that quantity, so the converted energy peaks at the true one.
                if analysis_charge_per_mev is not None:
                    mean_y, std_y = most_probable_ratio(analysis_charge_per_mev[this_filter_idx])
                else:
                    mean_y, std_y = most_probable_ratio(data[f"Corrected{charge}ChargePerMeV"][this_filter_idx])
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
            # Sigmoid rate up to 10 per hit: with the mode estimator F0 rises from ~0.83 of its plateau at
            # NHits 1 to the plateau by NHits 3, steeper than the previous bound of 1 allowed.
            bounds=([0, 0, 0, 0], [1e3, 1, 1e3, 10]),
        )
        perr[f"{charge}Charge"] = np.sqrt(np.diag(pcov[f"{charge}Charge"]))

        df_fit.append(
            {
                "Geometry": info["GEOMETRY"],
                "Config": config,
                "Name": name,
                "Clustering": charge,
                "Values": x,
                "ValuesUnit": "#Hits",
                "Factor": y,
                "FactorUnit": "ADC x tick / MeV",
                "FactorError": y_error,
                "FactorErrorUnit": "ADC x tick / MeV",
                "FitFunction": correction_func,
                "FitFunctionLabel": f"Exponential + Sigmoid",
                "FitFunctionFormula": f"a * exp(-b * x) + c / (1 + exp(-d * x))",
                "Params": popt[f"{charge}Charge"],
                "ParamsLabel": [
                    "Amplitude",
                    "Decay",
                    "Constant",
                    "Sigmoid",
                ],
                "ParamsFormat": [".0f", ".1f", ".0f", ".1f"],
                "ParamsError": perr[f"{charge}Charge"],
                "ParamsUnit": ["ADC x tick / MeV", "1 / #hits", "ADC x tick / MeV", "1 / #hits"],
                "p0": popt[f"{charge}Charge"][0],
                "p1": popt[f"{charge}Charge"][1],
                "p2": popt[f"{charge}Charge"][2],
                "p3": popt[f"{charge}Charge"][3],
            }
        )

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

        save_df(
            pd.DataFrame(df_fit),
            data_path,
            config,
            name,
            filename=f"{charge}Charge_Correction_Factor",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )
