import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *

save_path = f"{root}/images/vertex/resolution"
data_path = f"{root}/data/vertex/resolution"


def gaussian(x, a, b, c):
    return a * np.exp(-0.5 * ((x - b) / c) ** 2)


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
    "workflow": "VERTEXING",
    "label": {
        "marley": "Neutrino",
        "neutron": "Neutron",
        "gamma": "Gamma",
        None: "Particle",
    },
    "rewrite": args.rewrite,
    "debug": args.debug,
}

run, output = load_multi(
    configs, preset=user_input["workflow"], debug=user_input["debug"]
)
run = compute_reco_workflow(
    run, configs, workflow=user_input["workflow"], debug=user_input["debug"]
)
run, mask, output = compute_filtered_run(
    run,
    configs,
    presets=["ANALYSIS"],
    debug=user_input["debug"],
)
rprint(output)

reco_df = npy2df(run, "Reco", debug=False)

sigma_list = []
for config in configs:
    info, params, output = get_param_dict(
        f"{root}/config/{config}/{config}", {}, output, debug=args.debug
    )
    for name in configs[config]:
        max_hist = 0
        hist_list, purity_list = [], []
        info = json.load(open(f"{root}/config/{config}/{config}_config.json", "r"))
        this_reco_df = reco_df[
            (reco_df["Geometry"] == info["GEOMETRY"])
            & (reco_df["Version"] == info["VERSION"])
            & (reco_df["Name"] == name)
        ]

        fig1 = make_subplots(rows=1, cols=3, subplot_titles=["RecoX", "RecoY", "RecoZ"])
        for idx, (coord, error) in enumerate(
            zip(["X", "Y", "Z"], ["ErrorX", "ErrorY", "ErrorZ"])
        ):
            #############################################################################
            ########################### Vertexing Error Plot ############################
            #############################################################################
            init = 30  # Initial array value for the gaussian fit
            fnal = 50  # Final array value for the gaussian fit

            hx, edges = np.histogram(
                this_reco_df[error][this_reco_df["MatchedOpFlashPur"] > 0],
                bins=np.arange(-20, 20.5, 0.5),
                density=True,
            )

            try:
                popt, pcov = curve_fit(
                    gaussian,
                    0.5 * (edges[1:] + edges[:-1])[init:fnal],
                    hx[init:fnal],
                    p0=[1, 0, 5],
                    # sigma=1/np.sqrt(hx_counts[init:fnal])
                )
                perr = np.sqrt(np.diag(pcov))
                fit_y = gaussian(0.5 * (edges[1:] + edges[:-1]), *popt)
            except:
                popt = [0, 0, np.nan]
                perr = [0, 0, np.nan]
                fit_y = np.zeros(len(hx))

            # Find percentage of events within 1, 2 and 3 sigma
            sigma1 = len(
                this_reco_df[error][
                    (this_reco_df[error] > (popt[1] - 1 * popt[2]))
                    & (this_reco_df[error] < (popt[1] + 1 * popt[2]))
                ]
            ) / len(this_reco_df[error])
            sigma2 = len(
                this_reco_df[error][
                    (this_reco_df[error] > (popt[1] - 3 * popt[2]))
                    & (this_reco_df[error] < (popt[1] + 3 * popt[2]))
                ]
            ) / len(this_reco_df[error])
            sigma3 = len(
                this_reco_df[error][
                    (this_reco_df[error] > (popt[1] - 5 * popt[2]))
                    & (this_reco_df[error] < (popt[1] + 5 * popt[2]))
                ]
            ) / len(this_reco_df[error])
            # rprint(
            #     f"-> {config} {name} {coord} Vertexing Resolution: {popt[2]:.2f} +/- {perr[2]:.2f} cm"
            #     + f" ({100*sigma1:.1f}%, {100*sigma2:.1f}%, {100*sigma3:.1f}%) within 1, 3 and 5 sigma"
            # )

            # Make a 2D histogram of the purity vs the error
            if coord == "X":
                drift_sample = []
                drift_sample_error = []
                for idx, (purity_label, purity_idx) in enumerate(
                    zip(
                        ["No-Match", "Low-Purity", "High-Purity"],
                        [
                            (this_reco_df["MatchedOpFlashPur"] <= 0)
                            * (this_reco_df["MatchedOpFlashPE"] <= 0),
                            (this_reco_df["MatchedOpFlashPur"] <= 0)
                            * (this_reco_df["MatchedOpFlashPE"] > 0),
                            this_reco_df["MatchedOpFlashPur"] > 0,
                        ],
                    )
                ):
                    drift_sample.append(np.sum(purity_idx) / len(this_reco_df))
                    drift_sample_error = (
                        np.sqrt(np.sum(purity_idx)) / len(this_reco_df)
                        if np.sum(purity_idx) > 0
                        else 0
                    )
                    h, this_edges = np.histogram(
                        this_reco_df[error][purity_idx],
                        density=False,
                        bins=edges,
                    )
                    if np.sum(h) > 100:
                        purity_list.append(
                            {
                                "Geometry": info["GEOMETRY"],
                                "Config": config,
                                "Name": name,
                                "Coordinate": coord,
                                "Label": purity_label,
                                "Counts": np.sum(h),
                                "Density": np.asarray(h)
                                / np.sum(h)
                                / (this_edges[1] - this_edges[0]),
                                "DensityError": np.sqrt(h)
                                / np.sum(h)
                                / (this_edges[1] - this_edges[0]),
                                "Values": 0.5 * (edges[1:] + edges[:-1]),
                                "FitFunction": gaussian,
                                "Params": popt,
                                "ParamsError": perr,
                                "ParamsLabels": ["Amp.", "Mean", "Sigma"],
                                "ParamsFormat": [".1f", ".1f", ".2f"],
                            }
                        )
                    # Print the drift sample percentages
                    rprint(
                        f"-> {config} {name} {coord} Vertexing Purity Sample: {purity_label} -> {100*drift_sample[idx]:.2f}% +/- {100*drift_sample_error:.2f}%"
                    )

            h, edges = np.histogram(
                this_reco_df[error],
                bins=edges,
                density=True,
            )
            h_total, edges_total = np.histogram(
                this_reco_df[error],
                bins=np.arange(
                    info[f"DETECTOR_MIN_{coord}"],
                    info[f"DETECTOR_MAX_{coord}"] + 20,
                    20,
                ),
                density=True,
            )

            hist_list.append(
                {
                    "Sample": "All",
                    "Energy": None,
                    "Coordinate": coord,
                    "Error": np.asarray(h),
                    "Values": 0.5 * (edges[1:] + edges[:-1]),
                    "Amplitude": popt[0],
                    "Center": popt[1],
                    "Sigma": (popt[2] if np.sum(hx) > 0 else np.nan),
                    "SigmaError": (perr[2] if np.sum(hx) > 0 else np.nan),
                    "Mean": np.mean(this_reco_df[error]),
                    "Median": np.median(this_reco_df[error]),
                    "STD": np.std(this_reco_df[error]),
                    "STDError": np.std(this_reco_df[error])
                    / np.sqrt(len(this_reco_df[error])),
                }
            )

            sigma_list.append(
                {
                    "Geometry": info["GEOMETRY"],
                    "Config": config,
                    "Name": name,
                    "Coordinate": coord,
                    "Error": np.asarray(h_total),
                    "Values": np.asarray(0.5 * (edges_total[1:] + edges_total[:-1])),
                    "Zoom": np.asarray(h),
                    "ZoomBins": np.asarray(0.5 * (edges[1:] + edges[:-1])),
                    "NoMatch": 100 * drift_sample[0],
                    "Background": 100 * drift_sample[1],
                    "Signal": 100 * drift_sample[2],
                    "Sigma": popt[2],
                    "SigmaError": perr[2],
                    "Sigma1": sigma1,
                    "Sigma3": sigma2,
                    "Sigma5": sigma3,
                    "Mean": np.mean(this_reco_df[error]),
                    "Median": np.median(this_reco_df[error]),
                    "STD": np.std(this_reco_df[error]),
                    "STDError": np.std(this_reco_df[error])
                    / np.sqrt(len(this_reco_df[error])),
                }
            )

            if np.max(h) > max_hist:
                max_hist = np.max(h)

            for energy in red_energy_centers:
                this_energy_df = this_reco_df[
                    (this_reco_df["SignalParticleK"] > (energy - 1))
                    & (this_reco_df["SignalParticleK"] < energy + 1)
                ]
                hx, edges = np.histogram(
                    this_energy_df[error][(this_energy_df["MatchedOpFlashPur"] > 0.1)],
                    bins=np.arange(-25, 25.5, 0.5),
                    density=True,
                )

                # Fit a Gaussian to the histogram
                try:
                    popt, pcov = curve_fit(
                        gaussian,
                        0.5 * (edges[1:] + edges[:-1])[init:fnal],
                        hx[init:fnal],
                        p0=[1, 0, 5],
                        # sigma=1/np.sqrt(hx_counts[init:fnal])
                    )
                    perr = np.sqrt(np.diag(pcov))
                    fit_y = gaussian(0.5 * (edges[1:] + edges[:-1]), *popt)
                except:
                    popt = [0, 0, np.nan]
                    perr = [0, 0, np.nan]
                    fit_y = np.zeros(len(hx))

                hist_list.append(
                    {
                        "Sample": f"{energy}",
                        "Energy": energy,
                        "Coordinate": coord,
                        "Error": np.asarray(hx),
                        "Values": 0.5 * (edges[1:] + edges[:-1]),
                        "Center": popt[1],
                        "Sigma": (popt[2] if np.sum(hx) > 0 else np.nan),
                        "SigmaError": (perr[2] if np.sum(hx) > 0 else np.nan),
                        "Mean": np.mean(this_energy_df[error]),
                        "Median": np.median(this_energy_df[error]),
                        "STD": np.std(this_energy_df[error]),
                        "STDError": np.std(this_energy_df[error])
                        / np.sqrt(len(this_energy_df[error])),
                    }
                )

        df = pd.DataFrame(hist_list)
        df = explode(df, ["Error", "Values"])
        df["Error"] = df["Error"].astype(float)
        df["Values"] = df["Values"].astype(float)

        fig = make_subplots(rows=1, cols=1)
        for j, coord in zip(
            range(3),
            ["X", "Y", "Z"],
        ):

            this_df = df[(df["Coordinate"] == coord) & (df["Sample"] == "All")]
            # print(this_df)
            fig.add_trace(
                go.Scatter(
                    x=this_df["Values"],
                    y=this_df["Error"],
                    mode="lines",
                    line_shape="hvh",
                    line=dict(color=default[j], width=2),
                    name=f"{coord} Sigma: {this_df['Sigma'].values[0]:.1f} cm",
                ),
                row=1,
                col=1,
            )

        fig.update_yaxes(title_text="Density", row=1, col=1)

        fig = format_coustom_plotly(
            fig,
            log=(False, True),
            tickformat=(None, ".1s"),
            ranges=([-20, 20], [-3, 0]),
            title=f"Error Distribution - {config} {name}",
            legend_title="Coordinate",
            legend=dict(
                x=0.68, y=0.99, font=dict(size=13), title=dict(font=dict(size=16))
            ),
        )
        fig.update_xaxes(title_text="Vertex True - Reco (cm)", row=1, col=1)
        save_figure(
            fig,
            save_path,
            config,
            name,
            filename=f"Vertex_Error_LogY",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )

        fig = make_subplots(rows=1, cols=1)
        for j, coord in zip(
            range(3),
            ["X", "Y", "Z"],
        ):

            this_df = df[(df["Coordinate"] == coord) & (df["Sample"] == "All")]
            # Add a double exponential decay fit to the histogram
            # try:
            #     popt, pcov = curve_fit(
            #         lambda x, a, b: a * np.exp(-np.abs(x) / b),
            #         this_df["Values"],
            #         this_df["Error"],
            #         p0=[1, 0.1],
            #     )
            #     fit_y = popt[0] * np.exp(-np.abs(this_df["Values"]) / popt[1])
            # except:
            #     rprint("Could not fit double exponential decay to histogram")
            #     fit_y = np.zeros(len(this_df["Values"]))
            #     popt = [0, 0]

            # fit_x = np.linspace(-20, 20, 1000)
            # fig.add_trace(
            #     go.Scatter(
            #         x=fit_x,
            #         y=popt[0] * np.exp(-np.abs(fit_x) / popt[1]),
            #         mode="lines",
            #         line_shape="spline",
            #         line=dict(color="red", dash="dash"),
            #         showlegend=False,
            #     ),
            #     row=1,
            #     col=1,
            # )
            # print(this_df)
            fig.add_trace(
                go.Scatter(
                    x=this_df["Values"],
                    y=this_df["Error"],
                    mode="lines",
                    line_shape="hvh",
                    line=dict(color=default[j], width=2),
                    name=f"{coord}",
                ),
                row=1,
                col=1,
            )

        fig.update_yaxes(title_text="Density", row=1, col=1)

        fig = format_coustom_plotly(
            fig,
            log=(False, False),
            ranges=([-20, 20], [0, 1.1 * max_hist]),
            title=f"Error Distribution - {config} {name}",
            legend_title="Coordinate",
            legend=dict(
                x=0.68, y=0.99, font=dict(size=13), title=dict(font=dict(size=16))
            ),
        )
        save_figure(
            fig,
            save_path,
            config,
            name,
            filename=f"Vertex_Error",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )

        # Make a plot of the error vs purity heatmaps
        purity_df = pd.DataFrame(purity_list)
        this_purity_df = explode(purity_df, ["Density", "DensityError", "Values"])
        this_purity_df["DensityError"] = this_purity_df["DensityError"].astype(float)
        this_purity_df["Density"] = this_purity_df["Density"].astype(float)
        this_purity_df["Values"] = this_purity_df["Values"].astype(float)

        fig = make_subplots(rows=1, cols=1)

        for k, purity_label in enumerate(this_purity_df["Label"].unique()[::-1]):
            this_df = this_purity_df[
                (this_purity_df["Coordinate"] == "X")
                & (this_purity_df["Label"] == purity_label)
            ]

            fig.add_trace(
                go.Scatter(
                    x=this_df["Values"],
                    y=this_df["Density"],
                    error_y=dict(type="data", array=this_df["DensityError"]),
                    mode="lines+markers",
                    line_shape="hvh",
                    line=dict(color=default[k], width=2),
                    name=f"{purity_label}",
                ),
                row=1,
                col=1,
            )
            # Add gaussian fit for the high purity sample
            if purity_label == "High-Purity":
                df_x = df[(df["Coordinate"] == "X") & (df["Sample"] == "All")]
                popt = [
                    df_x["Amplitude"].values[0],
                    df_x["Center"].values[0],
                    df_x["Sigma"].values[0],
                ]
                fig.add_trace(
                    go.Scatter(
                        x=0.5 * (edges[1:] + edges[:-1]),
                        y=gaussian(0.5 * (edges[1:] + edges[:-1]), *popt),
                        mode="lines",
                        line_shape="spline",
                        line=dict(color="red", dash="dash"),
                        name=f"Fit: Sigma {popt[2]:.1f} cm",
                    ),
                    row=1,
                    col=1,
                )

        # fig.update_layout(legend_title_text="Energy (MeV)")
        fig = format_coustom_plotly(
            fig,
            # add_watermark=False,
            title=f"X Error vs OpFlash Purity - {config}",
            legend=dict(x=0.74, y=0.99),
            legend_title="Matched OpFlash",
            log=(False, True),
            tickformat=(None, ".1s"),
            ranges=([-20, 20], [-4, 1]),
        )
        fig.update_xaxes(title_text="Vertex X True - Reco (cm)")
        fig.update_yaxes(title_text="Density")
        save_figure(
            fig,
            save_path,
            config,
            name,
            None,
            filename=f"Vertex_Error_Purity",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )

        fig = make_subplots(rows=1, cols=1)
        for j, coord in zip(
            range(3),
            ["X", "Y", "Z"],
        ):

            this_df = df[(df["Coordinate"] == coord) & (df["Sample"] != "All")]
            fig.add_trace(
                go.Scatter(
                    x=this_df["Energy"],
                    y=this_df["Sigma"],
                    error_y=dict(type="data", array=this_df["SigmaError"]),
                    mode="lines+markers",
                    line_shape="hvh",
                    line=dict(color=default[j % len(default)], width=2),
                    name=f"{coord}",
                ),
                row=1,
                col=1,
            )

        fig.update_yaxes(title_text="Resolution (cm)", row=1, col=1)

        fig = format_coustom_plotly(
            fig,
            ranges=([6, 30], [0, 4]),
            title=f"Vertex Resolution - {config} {name}",
            legend_title="Coordinate",
            legend=dict(
                x=0.82, y=0.99, font=dict(size=13), title=dict(font=dict(size=16))
            ),
        )
        fig.update_xaxes(title_text="True Neutrino Energy (MeV)", row=1, col=1)
        save_figure(
            fig,
            save_path,
            config,
            name,
            filename=f"Vertex_Resolution",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )
        # Save the sigma list to a dataframe
        sigma_df = pd.DataFrame(sigma_list)
        for filename, df in zip(
            ["Purity_Match_Resolution", "Vertex_Resolution"], [purity_df, sigma_df]
        ):
            save_df(
                df=df,
                path=data_path,
                config=config,
                name=name,
                subfolder=None,
                filename=filename,
                rm=user_input["rewrite"],
                debug=user_input["debug"],
            )
