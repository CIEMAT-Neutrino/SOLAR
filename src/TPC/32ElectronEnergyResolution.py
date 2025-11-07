import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *


def resolution(x, p0, p1, p2, b):
    """
    Resolution function.
    """
    residuals = np.sqrt(
        np.power(p2, 2) + np.power(p1 / np.sqrt(x - b), 2) + np.power(p0 / (x - b), 2)
    )
    residuals[np.isnan(residuals)] = 0
    residuals[np.isinf(residuals)] = 0
    return residuals


save_path = f"{root}/images/TPC/resolution/electron"
data_path = f"{root}/data/TPC/resolution/electron"

for path in [save_path, data_path]:
    if not os.path.exists(path):
        os.makedirs(path)

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
    "--name", type=str, help="The name of the configuration", default="marley_official"
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()
config = args.config
name = args.name

configs = {config: [name]}

user_input = {
    "workflow": "DISCRIMINATION",
    "rewrite": args.rewrite,
    "debug": args.debug,
}

run, output = load_multi(
    configs, preset=user_input["workflow"], debug=user_input["debug"]
)

filtered_run, mask, output = compute_filtered_run(
    run,
    configs,
    presets=[user_input["workflow"]],
    debug=user_input["debug"],
)
rprint(output)

RMS_data = []
for label, params in zip(
    ["True", "Reco", "None"],
    [
        {
            "DEFAULT_ENERGY_TIME": "Time",
            "DEFAULT_ADJCL_ENERGY_TIME": "AdjClTime",
        },
        {},
        {
            "DEFAULT_ENERGY_TIME": "AverageDriftTime",
            "DEFAULT_ADJCL_ENERGY_TIME": "AdjClAverageDriftTime",
        },
    ],
):
    this_run = compute_reco_workflow(
        filtered_run,
        configs,
        params=params,
        workflow=user_input["workflow"],
        debug=user_input["debug"],
    )

    this_filtered_run, mask, output = compute_filtered_run(
        this_run,
        configs,
        params={("Reco", "TrueMain"): ("equal", True)},
        debug=user_input["debug"],
    )
    rprint(output)

    data = this_filtered_run["Reco"]
    fit = {
        "color": "grey",
        "spec_type": "max",
        "print": False,
        "show": False,
        "opacity": 1,
    }
    fit_init = 2
    fit_thld = 25
    data_end = 30
    for config in configs:
        info, params, output = get_param_dict(
            f"{root}/config/{config}/{config}", {}, output, debug=args.debug
        )
        for name, (variable, variable_label), nhit in product(
            configs[config], zip(["Electron", ""], ["Ideal", "Reco"]), nhits[:3]
        ):
            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=("Energy Smearing", "Energy Resolution"),
            )

            x, y, h = get_hist2d(
                data["ElectronK"][data["NHits"] >= nhit],
                data[f"{variable}Energy"][data["NHits"] >= nhit],
                per=None,
                norm=False,
                acc=(
                    reco_energy_edges,
                    reco_energy_edges,
                ),
            )
            h = h / np.max(h)
            # Change 0 entries in h for Nan
            h = np.where(h == 0, np.nan, h)
            fig.add_trace(
                go.Heatmap(
                    x=x,
                    y=y,
                    z=h.T,
                    coloraxis="coloraxis",
                ),
                row=1,
                col=1,
            )
            # Save heatmap to pkl. Build dataframe with columns == x, rows == y, values == z
            # save_pkl(
            #     {"x": x, "y": y, "z": h},
            #     data_path,
            #     config,
            #     name,
            #     filename=f"{variable_label}_Clustering_{label}_Drift_Correction_NHits{nhit}",
            #     rm=user_input["rewrite"],
            #     debug=user_input["debug"],
            # )

            RMS = []
            RMS_error = []
            for energy_bin in reco_energy_centers:
                idx = np.where(
                    (
                        data["ElectronK"][data["NHits"] >= nhit]
                        > energy_bin - reco_ebin / 2
                    )
                    & (
                        data["ElectronK"][data["NHits"] >= nhit]
                        < energy_bin + reco_ebin / 2
                    )
                )
                rms = np.sqrt(
                    np.mean(
                        np.power(
                            (
                                data["ElectronK"][data["NHits"] >= nhit][idx]
                                - data[f"{variable}Energy"][data["NHits"] >= nhit][idx]
                            )
                            / data["ElectronK"][data["NHits"] >= nhit][idx],
                            2,
                        )
                    )
                )

                # Compute an associated error on the RMS dependent on the number of events in the bin
                RMS.append(rms)
                error = np.sqrt(
                    np.mean(
                        np.power(
                            (
                                data["ElectronK"][data["NHits"] >= nhit][idx]
                                - data[f"{variable}Energy"][data["NHits"] >= nhit][idx]
                            )
                            / data["ElectronK"][data["NHits"] >= nhit][idx],
                            2,
                        )
                    )
                    / np.sqrt(len(idx[0]))
                )

                RMS_error.append(error)

            # If RMS value is bigger than 1 substitute it with 1
            RMS = np.where(np.isnan(RMS), 0, RMS)
            RMS_error = np.where(np.isnan(RMS_error), 0, RMS_error)
            try:
                fit_params, fit_conv = curve_fit(
                    resolution,
                    reco_energy_centers[fit_init:fit_thld],
                    RMS[fit_init:fit_thld],
                    sigma=RMS_error[fit_init:fit_thld],
                    p0=[0.01, 0.01, 0.15, 1],
                    bounds=([0, 0, 0.08, 0.5], [1, 0.1, 0.2, 1]),
                )

            except ValueError:
                print("Fit did not converge with error array")
                fit_params, fit_conv = curve_fit(
                    resolution,
                    reco_energy_centers[fit_init:fit_thld],
                    RMS[fit_init:fit_thld],
                    p0=[0.01, 0.01, 0.1, 1],
                    bounds=([0, 0, 0.08, 0.5], [1, 0.1, 0.2, 1]),
                )

            fit_errors = np.sqrt(np.diag(fit_conv))
            fit_x = np.linspace(
                min(reco_energy_centers[fit_init:data_end]),
                max(reco_energy_centers[fit_init:data_end]),
                100,
            )
            fit_y = resolution(fit_x, *fit_params)

            RMS = np.where(RMS == 0, np.nan, RMS)
            RMS_error = np.where(RMS_error == 0, np.nan, RMS_error)

            # print(fit_params, fit_errors)
            fig.add_trace(
                go.Scatter(
                    x=fit_x,
                    y=fit_y,
                    line=dict(
                        color="red",
                    ),
                    name=f"Resolution Fit: ({100*fit_params[2]:.1f} ± {100*fit_errors[2]:.1f})%",
                ),
                row=1,
                col=2,
            )
            # Add error bars
            fig.add_trace(
                go.Scatter(
                    x=reco_energy_centers[fit_init:data_end],
                    y=RMS[fit_init:data_end],
                    mode="lines",
                    line_shape="hvh",
                    marker=dict(color="black", size=5),
                    name="RMS (True - Reco) / True",
                ),
                row=1,
                col=2,
            )
            # Draw grey error bands
            fig.add_trace(
                go.Scatter(
                    x=reco_energy_centers[fit_init:data_end],
                    y=RMS[fit_init:data_end] - RMS_error[fit_init:data_end],
                    mode="lines",
                    line_shape="hvh",
                    line=dict(color="grey", width=0),
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
            fig.add_trace(
                go.Scatter(
                    x=reco_energy_centers[fit_init:data_end],
                    y=RMS[fit_init:data_end] + RMS_error[fit_init:data_end],
                    mode="lines",
                    line_shape="hvh",
                    line=dict(color="grey", width=0),
                    fillcolor="rgba(128, 128, 128, 0.5)",
                    fill="tonexty",
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
            RMS_data.append(
                {
                    "Geometry": info["GEOMETRY"],
                    "Config": config,
                    "Name": name,
                    "Clustering": variable_label,
                    "Drift": label,
                    "#Hits": nhit,
                    "Values": reco_energy_centers[fit_init:data_end],
                    "RMS": RMS[fit_init:data_end],
                    "RMSError": RMS_error[fit_init:data_end],
                    "FitFunction": resolution,
                    "Params": fit_params,
                    "ParamsLabels": ["p0", "p1", "p2", "b"],
                    "ParamsFormat": [".1f", ".0e", ".2f", ".1f"],
                    "ParamsError": fit_errors,
                }
            )

            fig = format_coustom_plotly(
                fig,
                matches=("x", None),
                tickformat=(".1f", ".1f"),
                title=f"Electron Energy - Low Energy Resolution with NHit Threshold {nhit}",
                legend_title="Data",
                legend=dict(
                    y=0.01,
                    x=0.56,
                ),
            )

            fig.update_layout(
                coloraxis=dict(colorscale="turbo", colorbar=dict(title="Norm.")),
                xaxis1_title="True Electron Energy (MeV)",
                xaxis2_title="True Electron Energy (MeV)",
                yaxis1_title=f"Reconstructed Energy (MeV)",
                yaxis2_title=f"RMS (True - Reco) / True",
                # Set axis range
                yaxis2_range=[0, 0.5],
            )
            if nhit < 4:
                save_figure(
                    fig,
                    save_path,
                    config,
                    name,
                    filename=f"{variable_label}_Energy_{label}Resolution_NHits{nhit}",
                    rm=user_input["rewrite"],
                    debug=user_input["debug"],
                )

    save_pkl(
        RMS_data,
        data_path,
        config,
        name,
        filename=f"Electron_Energy_Resolution",
        rm=user_input["rewrite"],
        debug=user_input["debug"],
    )
