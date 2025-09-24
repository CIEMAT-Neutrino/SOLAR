import sys

sys.path.insert(0, "../../")

from lib import *

save_path = f"{root}/images/vertexing/"
data_path = f"{root}/data/vertexing/"
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
    params={
        ("Reco", "SignalParticleK"): ("smaller", 30),
        ("Reco", "TrueMain"): ("equal", True),
    },
    debug=user_input["debug"],
)
rprint(output)

reco_df = npy2df(run, "Reco", debug=False)

for config in configs:
    for name in configs[config]:
        cumsum_list, hist_list = [], []
        info = json.load(open(f"{root}/config/{config}/{config}_config.json", "r"))
        this_reco_df = reco_df[
            (reco_df["Generator"] == 1)
            & (reco_df["Geometry"] == info["GEOMETRY"])
            & (reco_df["Version"] == info["VERSION"])
            & (reco_df["Name"] == name)
        ]

        fig1 = make_subplots(rows=1, cols=3, subplot_titles=["RecoX", "RecoY", "RecoZ"])
        for idx, (coord, error) in enumerate(
            zip(["X", "Y", "Z"], ["ErrorX", "ErrorY", "ErrorZ"])
        ):

            #############################################################################
            ############################# Vertexing Smearing ############################
            #############################################################################

            min_coord = np.percentile(this_reco_df[f"SignalParticle{coord}"], 1)
            max_coord = np.percentile(this_reco_df[f"SignalParticle{coord}"], 99)
            bins = get_default_acc(len(this_reco_df[error]))
            h, x, y = np.histogram2d(
                this_reco_df[f"SignalParticle{coord}"],
                this_reco_df[f"Reco{coord}"],
                bins=(bins, np.linspace(min_coord, max_coord, bins)),
            )
            h[h == 0] = np.nan
            fig1.add_trace(
                go.Heatmap(
                    z=h.T,
                    x=x,
                    y=y,
                    colorscale="Viridis",
                    coloraxis="coloraxis",
                    colorbar=dict(title="Counts"),
                ),
                row=1,
                col=idx + 1,
            )

            #############################################################################
            ########################### Vertexing Error Plot ############################
            #############################################################################
            init = 40 # Initial array value for the gaussian fit
            fnal = 60 # Final array value for the gaussian fit

            hx, edges = np.histogram(
                this_reco_df[error],
                bins=np.arange(-25, 25.5, 0.5),
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
            sigma1 = len(this_reco_df[error][(this_reco_df[error] > (popt[1] - 1*popt[2])) & (this_reco_df[error] < (popt[1] + 1*popt[2]))])/len(this_reco_df[error])
            sigma2 = len(this_reco_df[error][(this_reco_df[error] > (popt[1] - 2*popt[2])) & (this_reco_df[error] < (popt[1] + 2*popt[2]))])/len(this_reco_df[error])
            sigma3 = len(this_reco_df[error][(this_reco_df[error] > (popt[1] - 3*popt[2])) & (this_reco_df[error] < (popt[1] + 3*popt[2]))])/len(this_reco_df[error])
            rprint(f"-> {config} {name} {coord} Vertexing Resolution: {popt[2]:.2f} +/- {perr[2]:.2f} cm" +
                f" ({100*sigma1:.1f}%, {100*sigma2:.1f}%, {100*sigma3:.1f}%) within 1, 2 and 3 sigma")

            hist_list.append(
                {
                    "Sample": "All",
                    "Energy": None,
                    "Error": np.asarray(hx),
                    "Mean": np.mean(this_reco_df[error]),
                    "Median": np.median(this_reco_df[error]),
                    "STD": np.std(this_reco_df[error]),
                    "STD_err": np.nan,
                    "Coordinate": coord,
                    "Value": 0.5 * (edges[1:] + edges[:-1]),
                }
            )

            for energy in red_energy_centers:
                hx, edges = np.histogram(
                    this_reco_df[error][(this_reco_df["SignalParticleK"] > (energy - red_ebin/2)) & (this_reco_df["SignalParticleK"] < energy + red_ebin/2)],
                    bins=np.arange(-25, 25.5, 0.5),
                    density=True,
                )
                # hx_counts, _ = np.histogram(
                #     this_reco_df[error][(this_reco_df["SignalParticleK"] > (energy - red_ebin/2)) & (this_reco_df["SignalParticleK"] < energy + red_ebin/2)],
                #     bins=np.arange(-25, 25.5, 0.5),
                #     density=False,
                # )

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
                        "Error": np.asarray(hx),
                        "Mean": popt[1] if np.sum(hx) > 0 else np.nan,
                        "Median": np.median(
                            this_reco_df[
                                (this_reco_df["SignalParticleK"] > (energy - red_ebin/2))
                                & (this_reco_df["SignalParticleK"] < energy + red_ebin/2)
                            ][error]
                        ),
                        "STD": popt[2] if np.sum(hx) > 0 else np.nan,
                        "STD_err": perr[2] if np.sum(hx) > 0 else np.nan,
                        "Coordinate": coord,
                        "Value": 0.5 * (edges[1:] + edges[:-1]),
                    }
                )

            #############################################################################
            #################### Cumulative Distribution Function #######################
            #############################################################################

            for energy in red_energy_centers:
                hx, edges = np.histogram(
                    this_reco_df[
                        (this_reco_df["SignalParticleK"] > (energy - 1))
                        & (this_reco_df["SignalParticleK"] < energy + 1)
                    ][error],
                    bins=np.linspace(0, 100, 1000),
                )
                hx = hx / np.sum(hx)
                cdfx = np.cumsum(hx)

                STD = np.std(
                    this_reco_df[
                        (this_reco_df["SignalParticleK"] > (energy - 1))
                        & (this_reco_df["SignalParticleK"] < energy + 1)
                    ][error]
                )

                cumsum_list.append(
                    {
                        "Energy": energy,
                        "Error": cdfx,
                        "STD": STD,
                        "Coordinate": coord,
                        "Value": 0.5 * (edges[1:] + edges[:-1]),
                    }
                )

        df = pd.DataFrame(hist_list)
        df = explode(df, ["Error", "Value"])
        df["Error"] = df["Error"].astype(float)
        df["Value"] = df["Value"].astype(float)

        fig = make_subplots(rows=1, cols=1)
        for j, coord in zip(
            range(3),
            ["X", "Y", "Z"],
        ):

            this_df = df[(df["Coordinate"] == coord) & (df["Sample"] == "All")]
            # print(this_df)
            fig.add_trace(
                go.Scatter(
                    x=this_df["Value"],
                    y=this_df["Error"],
                    mode="lines",
                    line_shape="hvh",
                    line=dict(color=colors[j], width=2),
                    name=f"{coord} Median: {this_df['Median'].values[0]:.1e} cm",
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
                x=0.7, y=0.99, font=dict(size=13), title=dict(font=dict(size=16))
            ),
        )
        fig.update_xaxes(title_text="Vertex True - Reco (cm)", row=1, col=1)
        save_figure(
            fig,
            save_path,
            config,
            name,
            filename=f"Vertex_Error",
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
                    y=this_df["STD"],
                    error_y=dict(type="data", array=this_df["STD_err"]),
                    mode="lines+markers",
                    line_shape="hvh",
                    line=dict(color=colors[j], width=2),
                    name=f"{coord}",
                ),
                row=1,
                col=1,
            )

        fig.update_yaxes(title_text="Resolution (cm)", row=1, col=1)

        fig = format_coustom_plotly(
            fig,
            # log=(False, True),
            # tickformat=(None, ".1s"),
            ranges=([6, 30], [0, 4]),
            title=f"Error Resolution - {config} {name}",
            legend_title="Coordinate",
            legend=dict(
                x=0.82, y=0.99, font=dict(size=13), title=dict(font=dict(size=16))
            ),
        )
        fig.update_xaxes(title_text="Energy (MeV)", row=1, col=1)
        save_figure(
            fig,
            save_path,
            config,
            name,
            filename=f"Vertex_Error_Resolution",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )

        df = pd.DataFrame(cumsum_list)
        df = explode(df, ["Error", "Value"])
        df["Error"] = df["Error"].astype(float)
        df["Value"] = df["Value"].astype(float)

        fig = make_subplots(
            rows=1, cols=3, subplot_titles=["X Error", "Y Error", "Z Error"]
        )
        for j, coord in zip(
            range(3),
            ["X", "Y", "Z"],
        ):
            for i, energy in enumerate(red_energy_centers):
                this_df = df[(df["Energy"] == energy) & (df["Coordinate"] == coord)]
                fig.add_trace(
                    go.Scatter(
                        x=this_df["Value"],
                        y=this_df["Error"],
                        mode="lines",
                        name=int(energy),
                        line_shape="hvh",
                        line=dict(color=colors[int(i) % len(colors)], width=2),
                        showlegend=(j == 0),
                    ),
                    row=1,
                    col=j + 1,
                )
                if energy == 10:
                    # Apply high-pass filter to smooth the curve
                    this_df["Error"] = savgol_filter(
                        this_df["Error"].values, window_length=20, polyorder=2
                    )

                    # Take the first derivative and find the minimum
                    first_derivative = np.gradient(this_df["Error"].values)
                    second_derivative = np.gradient(first_derivative)
                    ref_index = np.argmax(-1 * second_derivative)
                    ref_value = this_df["Value"].values[ref_index]
                    ref_error = this_df["Error"].values[ref_index]

            fig.add_vline(
                x=ref_value,
                line_dash="dash",
                annotation_text=f"{ref_value:.1f} cm",
                annotation=dict(
                    showarrow=False,
                    xanchor="left",
                    yanchor="bottom",
                    # xref="x", yref="y",
                    xshift=-130,
                    yshift=-420,
                ),
                # Annotate in log scale
                row=1,
                col=j + 1,
            )
            fig.add_hline(
                y=ref_error,
                line_dash="dash",
                annotation_text=f"{100*ref_error:.1f}%",
                annotation_position="top right",
                row=1,
                col=j + 1,
            )

        fig.update_yaxes(title_text="Cumulative Distribution Function", row=1, col=1)

        fig.update_layout(legend_title_text="Energy (MeV)")
        fig = format_coustom_plotly(
            fig,
            log=(True, False),
            tickformat=(None, ".1f"),
            ranges=([-1, 2], [0, 1.1]),
            title=f"Vertex Error Distribution - {config} {name}",
            legend_title="Energy (MeV)",
            legend=dict(x=0.92, y=0.01),
        )
        fig.update_xaxes(title_text="X Vertex Error (cm)", row=1, col=1)
        fig.update_xaxes(title_text="Y Vertex Error (cm)", row=1, col=2)
        fig.update_xaxes(title_text="Z Vertex Error (cm)", row=1, col=3)
        save_figure(
            fig,
            save_path,
            config,
            name,
            filename=f"Cumulative_Vertex_Error",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )

        fig1 = format_coustom_plotly(
            fig1,
            log=(False, False),
            tickformat=(None, None),
            matches=(None, None),
            title=f"Vertexing Smearing - {config}",
        )
        fig1.update_layout(coloraxis=dict(colorbar=dict(title="Counts")))
        fig1.update_xaxes(title_text="True Vertex (cm)")
        fig1.update_yaxes(title_text="Reco Vertex (cm)")
        save_figure(
            fig1,
            save_path,
            config,
            name,
            filename=f"Vertex_Smearing",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )
