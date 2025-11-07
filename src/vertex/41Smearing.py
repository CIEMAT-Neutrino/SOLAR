import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *

save_path = f"{root}/images/vertex/smearing"
data_path = f"{root}/data/vertex/smearing"


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
    info, params, output = get_param_dict(
        f"{root}/config/{config}/{config}", {}, output, debug=args.debug
    )
    for name in configs[config]:
        vertex_list = []
        cumsum_list = []
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

            vertex_list.append(
                {
                    "Geometry": info["GEOMETRY"],
                    "Config": config,
                    "Name": name,
                    "Coordinate": coord,
                    "RecoCoordinate": list(this_reco_df[f"Reco{coord}"].values),
                    "TrueCoordinate": list(
                        this_reco_df[f"SignalParticle{coord}"].values
                    ),
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
                        "Geometry": info["GEOMETRY"],
                        "Config": config,
                        "Name": name,
                        "Energy": energy,
                        "Sample": 100 * cdfx,
                        "STD": STD,
                        "Variable": coord,
                        "Values": 0.5 * (edges[1:] + edges[:-1]),
                    }
                )

        df = pd.DataFrame(cumsum_list)
        df = explode(df, ["Sample", "Values"])
        df["Sample"] = df["Sample"].astype(float)
        df["Values"] = df["Values"].astype(float)

        fig = make_subplots(
            rows=1, cols=3, subplot_titles=["X Error", "Y Error", "Z Error"]
        )
        for j, coord in zip(
            range(3),
            ["X", "Y", "Z"],
        ):
            ref_value, ref_error = 0, 0
            for i, energy in enumerate(red_energy_centers):
                this_df = df[(df["Energy"] == energy) & (df["Variable"] == coord)]
                fig.add_trace(
                    go.Scatter(
                        x=this_df["Values"],
                        y=this_df["Sample"],
                        mode="lines",
                        name=int(energy),
                        line_shape="hvh",
                        line=dict(color=colors[int(i) % len(colors)], width=2),
                        showlegend=(j == 0),
                    ),
                    row=1,
                    col=j + 1,
                )
                if energy == 12:
                    # Apply high-pass filter to smooth the curve
                    this_df["Sample"] = savgol_filter(
                        this_df["Sample"].values, window_length=20, polyorder=2
                    )

                    # Take the first derivative and find the minimum
                    first_derivative = np.gradient(this_df["Sample"].values)
                    second_derivative = np.gradient(first_derivative)
                    ref_index = np.argmax(-1 * second_derivative)
                    ref_value = this_df["Values"].values[ref_index]
                    ref_error = this_df["Sample"].values[ref_index]

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
                        annotation_text=f"{ref_error:.1f}%",
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
            ranges=([-1, 2], [0, 110]),
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
        for df_label, df_list in zip(
            ["Cumulative_Vertex_Error", "Vertex_Smearing"], [cumsum_list, vertex_list]
        ):
            df = pd.DataFrame(df_list)
            save_df(
                df,
                data_path,
                config,
                name,
                filename=df_label,
                rm=user_input["rewrite"],
                debug=user_input["debug"],
            )
