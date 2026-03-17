import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *

save_path = f"{root}/images/vertex/resolution"
data_path = f"{root}/data/vertex/resolution"


def gaussian(x, a, b, c):
    return a * np.exp(-0.5 * ((x - b) / abs(c)) ** 2)


def exponential_decay(x, a, b, c):
    return a * np.exp(-abs(x) / abs(c)) - b


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
    signal="marley" in args.name,
    debug=user_input["debug"],
)
rprint(output)

reco_df = npy2df(run, "Reco", debug=False)

hist_list, purity_list, sigma_list = [], [], []

for config in configs:
    info, params, output = get_param_dict(
        f"{root}/config/{config}/{config}", {}, output, debug=args.debug
    )
    for name in configs[config]:
        max_hist = 0
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

            hx, edges = np.histogram(
                this_reco_df[error][this_reco_df["MatchedOpFlashPur"] > 0.5],
                bins=np.arange(-20, 20.5, 0.5),
                density=False,
            )
            if "marley" in name.lower():
                try:
                    fit_function = "Gaussian"
                    popt, pcov = curve_fit(
                        gaussian,
                        0.5 * (edges[1:] + edges[:-1]),
                        hx,
                        p0=[np.max(hx), 0, 2],
                        sigma=1 / np.sqrt(hx),
                    )
                except:
                    rprint("Gaussian fit failed, trying exponential decay fit")
                    fit_function = "Exponential"
                    popt, pcov = curve_fit(
                        exponential_decay,
                        0.5 * (edges[1:] + edges[:-1]),
                        hx,
                        p0=[np.max(hx), 0, 2],
                        sigma=1 / np.sqrt(hx),
                        bounds=([0, -5, 0.1], [np.max(hx) * 1.5, 5, 100]),
                    )
            else:
                rprint("Gaussian fit failed, trying exponential decay fit")
                fit_function = "Exponential"
                popt, pcov = curve_fit(
                    exponential_decay,
                    0.5 * (edges[1:] + edges[:-1]),
                    hx,
                    p0=[np.max(hx), 0, 2],
                    sigma=1 / np.sqrt(hx),
                    bounds=([0, -5, 0.1], [np.max(hx) * 1.5, 5, 100]),
                )

            fit_y = (
                gaussian(0.5 * (edges[1:] + edges[:-1]), *popt)
                if fit_function == "Gaussian"
                else exponential_decay(0.5 * (edges[1:] + edges[:-1]), *popt)
            )
            perr = np.sqrt(np.diag(pcov))

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

            # Make a 2D histogram of the purity vs the error
            if coord == "X":
                drift_sample = []
                drift_sample_error = []
                for idx, (purity_label, purity_idx) in enumerate(
                    zip(
                        ["No-Match", "Background", "Low-Purity", "High-Purity"],
                        [
                            (this_reco_df["MatchedOpFlashPur"] <= 0)
                            * (this_reco_df["MatchedOpFlashPE"] <= 0),
                            (this_reco_df["MatchedOpFlashPur"] == 0)
                            * (this_reco_df["MatchedOpFlashPE"] >= 0),
                            (this_reco_df["MatchedOpFlashPur"] > 0)
                            * (this_reco_df["MatchedOpFlashPur"] <= 0.5),
                            this_reco_df["MatchedOpFlashPur"] > 0.5,
                        ],
                    )
                ):
                    drift_sample.append(np.sum(purity_idx) / len(this_reco_df))
                    drift_sample_error.append(
                        np.sqrt(np.sum(purity_idx)) / len(this_reco_df)
                        if np.sum(purity_idx) > 0
                        else 0
                    )
                    h, this_edges = np.histogram(
                        this_reco_df[error][purity_idx],
                        density=False,
                        bins=np.arange(
                            -info["DETECTOR_MAX_X"], info["DETECTOR_MAX_X"], 0.5
                        ),
                    )
                    purity_list.append(
                        {
                            "Geometry": info["GEOMETRY"],
                            "Config": config,
                            "Name": name,
                            "Coordinate": coord,
                            "Label": purity_label,
                            "Percentage": 100 * drift_sample[-1],
                            "PercentageError": 100 * drift_sample_error[-1],
                            "Counts": h,
                            "CountsError": np.sqrt(h),
                            "Density": np.asarray(h)
                            / np.sum(h)
                            / (this_edges[1] - this_edges[0]),
                            "DensityError": np.sqrt(h)
                            / np.sum(h)
                            / (this_edges[1] - this_edges[0]),
                            "Values": 0.5 * (this_edges[1:] + this_edges[:-1]),
                            "FitFunction": (
                                gaussian
                                if fit_function == "Gaussian"
                                else exponential_decay
                            ),
                            "FitFunctionLabel": fit_function,
                            "FitFunctionFormula": (
                                r"A \exp\left(-\frac{(x - \mu)^2}{2 \sigma^2}\right)"
                                if fit_function == "Gaussian"
                                else r"A \exp\left(-\frac{|x|}{\tau}\right) - B"
                            ),
                            "Params": popt,
                            "ParamsError": perr,
                            "ParamsLabels": (
                                ["Amp.", "Mean", "Sigma"]
                                if fit_function == "Gaussian"
                                else ["Amp.", "Offset", "DecayConst"]
                            ),
                            "ParamsFormat": [".1f", ".1f", ".2f"],
                        }
                    )
                    # Print the drift sample percentages
                    rprint(
                        f"{purity_label}\t-> {100*drift_sample[idx]:.2f}% +/- {100*drift_sample_error[idx]:.2f}%"
                    )

            h, edges = np.histogram(
                this_reco_df[error],
                bins=edges,
                density=True,
            )
            h_total, edges_total = np.histogram(
                this_reco_df[error],
                bins=np.arange(
                    (
                        info[f"DETECTOR_MIN_{coord}"]
                        if coord != "Z"
                        else -info[f"DETECTOR_MAX_{coord}"]
                    ),
                    info[f"DETECTOR_MAX_{coord}"] + 10,
                    10,
                ),
                density=True,
            )

            hist_list.append(
                {
                    "Config": config,
                    "Name": name,
                    "Energy": None,
                    "Coordinate": coord,
                    "Error": np.asarray(h),
                    "Values": 0.5 * (edges[1:] + edges[:-1]),
                    "Sigma": popt[2],
                    "SigmaError": perr[2],
                    "FitFunctionLabel": fit_function,
                    "Params": popt,
                    "ParamsError": perr,
                    "ParamsLabels": (
                        ["Amp.", "Mean", "Sigma"]
                        if fit_function == "Gaussian"
                        else ["Amp.", "Offset", "DecayConst"]
                    ),
                    "ParamsFormat": [".1f", ".1f", ".2f"],
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
                    "LowPurity": 100 * drift_sample[2],
                    "HighPurity": 100 * drift_sample[3],
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

            for energy in lowe_energy_centers:
                this_energy_df = this_reco_df[
                    (this_reco_df["SignalParticleK"] > (energy - 1))
                    & (this_reco_df["SignalParticleK"] < energy + 1)
                ]
                hx, edges = np.histogram(
                    (
                        this_energy_df[error][
                            (this_energy_df["MatchedOpFlashPur"] > 0.5)
                        ]
                        if coord == "X"
                        else this_energy_df[error]
                    ),
                    bins=np.arange(-20, 20.5, 0.5),
                    density=True,
                )

                # Check that data exists for the fit
                if len(this_energy_df) < 50 or np.sum(hx) == 0:
                    continue

                if "marley" in name.lower():
                    # Fit a Gaussian to the histogram
                    try:
                        fit_function = "Gaussian"
                        popt, pcov = curve_fit(
                            gaussian,
                            0.5 * (edges[1:] + edges[:-1]),
                            hx,
                            p0=[np.max(hx), 0, 5],
                            # sigma=1/np.sqrt(hx_counts)
                        )
                        fit_y = gaussian(0.5 * (edges[1:] + edges[:-1]), *popt)

                    except:
                        rprint("Gaussian fit failed, trying exponential decay fit")
                        fit_function = "Exponential"
                        popt, pcov = curve_fit(
                            exponential_decay,
                            0.5 * (edges[1:] + edges[:-1]),
                            hx,
                            p0=[np.max(hx), 0, 5],
                            # sigma=1/np.sqrt(hx_counts),
                            bounds=([0, -5, 0.1], [np.max(hx) * 1.5, 5, 100]),
                        )
                        fit_y = exponential_decay(0.5 * (edges[1:] + edges[:-1]), *popt)

                else:
                    fit_function = "Exponential"
                    popt, pcov = curve_fit(
                        exponential_decay,
                        0.5 * (edges[1:] + edges[:-1]),
                        hx,
                        p0=[np.max(hx), 0, 5],
                        # sigma=1/np.sqrt(hx_counts),
                        bounds=([0, -5, 0.1], [np.max(hx) * 1.5, 5, 100]),
                    )
                    fit_y = exponential_decay(0.5 * (edges[1:] + edges[:-1]), *popt)

                perr = np.sqrt(np.diag(pcov))

                hist_list.append(
                    {
                        "Geometry": info["GEOMETRY"],
                        "Config": config,
                        "Name": name,
                        "Energy": energy,
                        "Coordinate": coord,
                        "Error": np.asarray(hx),
                        "Values": 0.5 * (edges[1:] + edges[:-1]),
                        "Sigma": popt[2],
                        "SigmaError": perr[2],
                        "FitFunctionLabel": fit_function,
                        "Params": popt,
                        "ParamsError": perr,
                        "ParamsLabels": (
                            ["Amp.", "Mean", "Sigma"]
                            if fit_function == "Gaussian"
                            else ["Amp.", "Offset", "DecayConst"]
                        ),
                        "ParamsFormat": [".1f", ".1f", ".2f"],
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

            this_df = df[(df["Coordinate"] == coord) & (df["Energy"].isna())]
            # print(this_df)
            fig.add_trace(
                go.Scatter(
                    x=this_df["Values"],
                    y=this_df["Error"],
                    mode="lines",
                    line_shape="hvh",
                    line=dict(color=default[j], width=2),
                    # Select the second entry in params which is the sigma
                    name=f"{coord} Sigma: {this_df['Params'].values[0][2]:.2f} cm",
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

            this_df = df[(df["Coordinate"] == coord) & (df["Energy"].isna())]

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

        fig = make_subplots(rows=1, cols=3)
        for j, coord in zip(
            range(3),
            ["X", "Y", "Z"],
        ):
            if coord == "X":
                this_reco_matrix = {
                    "Config": config,
                    "Name": name,
                    f"{coord}": this_reco_df[f"SignalParticle{coord}"],
                    f"Error{coord}": this_reco_df[f"Error{coord}"],
                    "Energy": this_reco_df["SignalParticleK"],
                    "Matched": this_reco_df["MatchedOpFlashPE"] > 0,
                }
                save_pkl(
                    pd.DataFrame(this_reco_matrix),
                    data_path,
                    config,
                    name,
                    None,
                    filename= f"Vertex_Matrix_{coord}",
                    rm=user_input["rewrite"],
                    debug=user_input["debug"],
                )

            h, x, y = np.histogram2d(
                this_reco_df[f"SignalParticle{coord}"],
                this_reco_df[f"Error{coord}"],
                bins=[100, 100],
                density=True,
            )

            fig.add_trace(
                go.Heatmap(
                    x=x,
                    y=y,
                    z=np.log10(h.T),
                    coloraxis="coloraxis",
                ),
                row=1,
                col=j + 1,
            )

        fig = format_coustom_plotly(
            fig,
            matches=(None, None),
            title=f"Error Heatmap - {config}",
            add_watermark=False,
        )

        # Add title to colorbar
        fig.update_coloraxes(colorbar=dict(title="log(Density)"))

        for j, coord in enumerate(["X", "Y", "Z"]):
            fig.update_yaxes(title_text="Reco - True (cm)", row=1, col=1 + j)
            fig.update_xaxes(title_text=f"True {coord} (cm)", row=1, col=1 + j)

        save_figure(
            fig,
            save_path,
            config,
            name,
            filename=f"Vertex_Matrix",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )

        # Make a plot of the error vs purity heatmaps
        purity_df = pd.DataFrame(purity_list)
        cols = ["Counts", "CountsError", "Density", "DensityError", "Values"]
        this_purity_df = explode(purity_df, cols)
        for col in cols:
            this_purity_df[col] = this_purity_df[col].astype(float)

        fig = make_subplots(rows=1, cols=1)
        maxy = 0
        for k, purity_label in enumerate(this_purity_df["Label"].unique()[::-1]):
            this_df = this_purity_df[
                (this_purity_df["Coordinate"] == "X")
                & (this_purity_df["Label"] == purity_label)
            ]

            if purity_label == "High-Purity":
                df_x = df[(df["Coordinate"] == "X") & (df["Energy"].isna())]
                popt = [
                    df_x["Params"].values[0][0],
                    df_x["Params"].values[0][1],
                    df_x["Params"].values[0][2],
                ]
                fig.add_trace(
                    go.Scatter(
                        x=0.5 * (edges[1:] + edges[:-1]),
                        y=(
                            gaussian(0.5 * (edges[1:] + edges[:-1]), *popt)
                            if df_x["FitFunctionLabel"].values[0] == "Gaussian"
                            else exponential_decay(
                                0.5 * (edges[1:] + edges[:-1]), *popt
                            )
                        ),
                        mode="lines",
                        line_shape="spline",
                        line=dict(color="red", dash="dash"),
                        name=f"Fit: Sigma {popt[2]:.1f} cm",
                    ),
                    row=1,
                    col=1,
                )
            if np.max(this_df["Counts"]) > maxy:
                maxy = np.max(this_df["Counts"])

            fig.add_trace(
                go.Scatter(
                    x=this_df["Values"],
                    y=this_df["Counts"],
                    error_y=dict(type="data", array=this_df["CountsError"]),
                    mode="lines+markers",
                    line_shape="hvh",
                    line=dict(color=default[k], width=2),
                    name=f"{purity_label}: {this_df['Percentage'].values[0]:.1f}%",
                ),
                row=1,
                col=1,
            )
            # Add gaussian fit for the high purity sample

        # fig.update_layout(legend_title_text="Energy (MeV)")
        fig.update_xaxes(title_text="Vertex X True - Reco (cm)")
        fig.update_yaxes(title_text="Counts")
        fig = format_coustom_plotly(
            fig,
            # add_watermark=False,
            title=f"X Error vs OpFlash Purity - {config}",
            legend=dict(x=0.72, y=0.99),
            legend_title="Matched OpFlash",
            log=(False, True),
            tickformat=(None, ".1s"),
            ranges=(None, [0, np.log10(1.5 * maxy)]),
        )

        for rangex in (None, [-20, 20]):
            if rangex is not None:
                fig.update_xaxes(range=rangex)

            save_figure(
                fig,
                save_path,
                config,
                name,
                None,
                filename=(
                    f"Vertex_Error_Purity"
                    if rangex is None
                    else f"Vertex_Error_Purity_Zoom"
                ),
                rm=user_input["rewrite"],
                debug=user_input["debug"],
            )

        fig = make_subplots(rows=1, cols=1)
        for j, coord in zip(
            range(3),
            ["X", "Y", "Z"],
        ):

            this_df = df[(df["Coordinate"] == coord) & (~df["Energy"].isna())]
            fig.add_trace(
                go.Scatter(
                    x=this_df["Energy"],
                    y=this_df["Params"].apply(lambda x: x[2]),
                    error_y=dict(
                        type="data", array=this_df["ParamsError"].apply(lambda x: x[2])
                    ),
                    mode="lines+markers",
                    line_shape="spline",
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
        for filename, df in zip(
            ["Resolution", "Purity_Match_Resolution", "Vertex_Resolution"],
            [
                pd.DataFrame(hist_list),
                pd.DataFrame(purity_list),
                pd.DataFrame(sigma_list),
            ],
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
