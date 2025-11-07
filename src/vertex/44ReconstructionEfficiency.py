import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *

save_path = f"{root}/images/vertex/reconstruction"
data_path = f"{root}/data/vertex/reconstruction"


def position_mask(
    run,
    info,
    energy: Optional[float] = 10,
    sigma: int = 1,
    position: float = 0,
    coordinate_bin: Optional[float] = 10,
    coordinate: Optional[list[str]] = ["X", "Y", "Z"],
):
    if coordinate is None:
        coordinate = ["X", "Y", "Z"]
    true_mask = np.ones(len(run["Truth"]["SignalParticleX"]), dtype=bool)
    reco_mask = np.ones(len(run["Reco"]["SignalParticleX"]), dtype=bool)
    for coord in coordinate:
        if coord == "X" and info["GEOMETRY"] == "hd":
            true_mask = true_mask * (
                (
                    np.absolute(run["Truth"][f"SignalParticle{coord}"])
                    > position - coordinate_bin / 2
                )
                * (
                    np.absolute(run["Truth"][f"SignalParticle{coord}"])
                    <= position + coordinate_bin / 2
                )
            )
            reco_mask = reco_mask * (
                (
                    np.absolute(run["Reco"][f"SignalParticle{coord}"])
                    < position + coordinate_bin / 2
                )
                * (
                    np.absolute(run["Reco"][f"SignalParticle{coord}"])
                    >= position - coordinate_bin / 2
                )
            )
        else:
            true_mask = true_mask * (
                (run["Truth"][f"SignalParticle{coord}"] > position - coordinate_bin / 2)
                * (
                    run["Truth"][f"SignalParticle{coord}"]
                    <= position + coordinate_bin / 2
                )
            )
            reco_mask = reco_mask * (
                (run["Reco"][f"SignalParticle{coord}"] < position + coordinate_bin / 2)
                * (
                    run["Reco"][f"SignalParticle{coord}"]
                    >= position - coordinate_bin / 2
                )
            )

        reco_mask = reco_mask * (
            np.absolute(run["Reco"][f"Error{coord}"])
            < df[
                (df["Coordinate"] == f"{coord}")
                * (df["Energy"] == energy if energy != None else df["Energy"].isna())
            ][f"Sigma{sigma}"].values[0]
        )

    return true_mask, reco_mask


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

# Redefine the np.ndarray lowe_energy_centers to have an extra None value at the beginning by using np.insert
lowe_energy_centers = np.append(
    np.array([None]),
    lowe_energy_centers,
)

for config in configs:
    info, params, output = get_param_dict(
        f"{root}/config/{config}/{config}", {}, output, debug=args.debug
    )
    fig = make_subplots(rows=1, cols=1)
    for name in configs[config]:
        position_list, hist_list = [], []
        info = json.load(open(f"{root}/config/{config}/{config}_config.json", "r"))
        this_filtered_run, mask, output = compute_filtered_run(
            run,
            {config: [name]},
            debug=user_input["debug"],
        )
        rprint(output)

        for (jdx, energy), (idx, (coord, error)) in product(
            enumerate(lowe_energy_centers),
            enumerate(zip(["X", "Y", "Z"], ["ErrorX", "ErrorY", "ErrorZ"])),
        ):

            #############################################################################
            ########################### Vertexing Error Plot ############################
            #############################################################################
            init = 30  # Initial array value for the gaussian fit
            fnal = 50  # Final array value for the gaussian fit
            if energy is None:
                this_energy_mask = np.ones(
                    len(this_filtered_run["Reco"]["SignalParticleK"]), dtype=bool
                )
            else:
                this_energy_mask = (
                    this_filtered_run["Reco"]["SignalParticleK"]
                    >= energy - lowe_ebin / 2
                ) * (
                    this_filtered_run["Reco"]["SignalParticleK"]
                    < energy + lowe_ebin / 2
                )

            hx, edges = np.histogram(
                this_filtered_run["Reco"][error][
                    (this_filtered_run["Reco"]["MatchedOpFlashPur"] > 0)
                    * this_energy_mask
                ],
                bins=np.arange(-20, 20.5, 0.5),
                density=True,
            )

            popt, pcov = curve_fit(
                lambda x, a, b, c: a * np.exp(-((x - b) ** 2) / (2 * c**2)),
                0.5 * (edges[1:] + edges[:-1])[init:fnal],
                hx[init:fnal],
                p0=[1, 0, 1],
            )
            perr = np.sqrt(np.diag(pcov))

            # Find percentage of events within 1, 2 and 3 standard deviations
            sigma1 = len(
                this_filtered_run["Reco"][error][
                    (np.abs(this_filtered_run["Reco"][error] - popt[1]) < (popt[2]))
                    * (this_energy_mask)
                ]
            ) / len(this_filtered_run["Reco"][error][this_energy_mask])
            sigma2 = len(
                this_filtered_run["Reco"][error][
                    (np.abs(this_filtered_run["Reco"][error] - popt[1]) < (3 * popt[2]))
                    * (this_energy_mask)
                ]
            ) / len(this_filtered_run["Reco"][error][this_energy_mask])
            sigma3 = len(
                this_filtered_run["Reco"][error][
                    (np.abs(this_filtered_run["Reco"][error] - popt[1]) < (5 * popt[2]))
                    * (this_energy_mask)
                ]
            ) / len(this_filtered_run["Reco"][error][this_energy_mask])
            if energy is None:
                rprint(
                    f"-> {config} {name} {coord} Vertexing Resolution: {popt[2]:.2f} +/- {perr[2]:.2f} cm"
                    + f" ({100*sigma1:.1f}%, {100*sigma2:.1f}%, {100*sigma3:.1f}%) within 1, 3, 5 standard deviations"
                )

            hist_list.append(
                {
                    "Sample": "All",
                    "Energy": energy,
                    "Error": np.asarray(hx),
                    "Mean": np.mean(this_filtered_run["Reco"][error]),
                    "Median": np.median(this_filtered_run["Reco"][error]),
                    "STD": np.std(this_filtered_run["Reco"][error]),
                    "STD_err": np.nan,
                    "Coordinate": coord,
                    "Value": 0.5 * (edges[1:] + edges[:-1]),
                    "Sigma1": popt[2],
                    "Sigma1Error": perr[2],
                    "Sigma3": 3 * popt[2],
                    "Sigma3Error": 3 * perr[2],
                    "Sigma5": 5 * popt[2],
                    "Sigma5Error": 5 * perr[2],
                }
            )

        df = pd.DataFrame(hist_list)
        df = df.fillna(np.nan)
        for energy, coord, sigma in product(
            lowe_energy_centers, ["X", "Y", "Z"], [3, 5, 1]
        ):
            counts, counts_error = [], []
            efficiency, efficiency_error = [], []
            if coord == "X" and info["GEOMETRY"] == "hd":
                coordinate_array = np.arange(
                    params[f"DEFAULT_{coord}_BIN"] / 2,
                    info[f"DETECTOR_MAX_{coord}"]
                    + info[f"DETECTOR_GAP_{coord}"]
                    + params[f"DEFAULT_{coord}_BIN"] / 2,
                    params[f"DEFAULT_{coord}_BIN"],
                )

            else:
                coordinate_array = np.arange(
                    info[f"DETECTOR_MIN_{coord}"]
                    - info[f"DETECTOR_GAP_{coord}"]
                    + params[f"DEFAULT_{coord}_BIN"] / 2,
                    info[f"DETECTOR_MAX_{coord}"]
                    + info[f"DETECTOR_GAP_{coord}"]
                    + params[f"DEFAULT_{coord}_BIN"] / 2,
                    params[f"DEFAULT_{coord}_BIN"],
                )

            for position in coordinate_array:
                true_mask, reco_mask = position_mask(
                    run,
                    info,
                    energy,
                    sigma,
                    position,
                    params[f"DEFAULT_{coord}_BIN"],
                    [coord],
                )
                if energy is None:
                    true_energy_mask = np.ones(
                        len(run["Truth"]["SignalParticleK"]), dtype=bool
                    )
                    reco_energy_mask = np.ones(
                        len(run["Reco"]["SignalParticleK"]), dtype=bool
                    )
                else:
                    true_energy_mask = (
                        run["Truth"]["SignalParticleK"] >= energy - lowe_ebin / 2
                    ) * (run["Truth"]["SignalParticleK"] < energy + lowe_ebin / 2)
                    reco_energy_mask = (
                        run["Reco"]["SignalParticleK"] >= energy - lowe_ebin / 2
                    ) * (run["Reco"]["SignalParticleK"] < energy + lowe_ebin / 2)

                this_true_mask = true_mask * (
                    (run["Truth"]["Geometry"] == info["GEOMETRY"])
                    * (run["Truth"]["Version"] == info["VERSION"])
                    * (run["Truth"]["Name"] == name)
                    * true_energy_mask
                )
                this_reco_mask = reco_mask * (
                    (run["Reco"]["Geometry"] == info["GEOMETRY"])
                    * (run["Reco"]["Version"] == info["VERSION"])
                    * (run["Reco"]["Name"] == name)
                    * reco_energy_mask
                )
                counts.append(sum(this_reco_mask))
                counts_error.append(np.sqrt(sum(this_reco_mask)))
                efficiency.append(
                    100
                    * (
                        sum(this_reco_mask) / sum(this_true_mask)
                        if sum(this_true_mask) > 0
                        else 0
                    )
                )
                efficiency_error.append(
                    100
                    * (
                        np.sqrt(sum(this_reco_mask)) / sum(this_true_mask)
                        if sum(this_true_mask) > 0
                        else 0
                    )
                )

            position_list.append(
                {
                    "Geometry": info["GEOMETRY"],
                    "Config": config,
                    "Name": name,
                    "Variable": coord,
                    "Energy": energy,
                    "Sigma": sigma,
                    "Values": (
                        2 * info[f"DETECTOR_MAX_{coord}"] - coordinate_array
                        if (config == "hd_1x2x6_lateralAPA" and coord == "X")
                        else (
                            coordinate_array + info[f"DETECTOR_MAX_{coord}"]
                            if (info["GEOMETRY"] == "vd" and coord == "X")
                            else coordinate_array
                        )
                    ),
                    "Counts": counts,
                    "CountsError": counts_error,
                    "Efficiency": efficiency,
                    "EfficiencyError": efficiency_error,
                }
            )

    df_position = pd.DataFrame(position_list)
    df_position = df_position.fillna(np.nan)

    fig = px.line(
        df_position[df_position["Energy"].isna()].explode(["Values", "Efficiency"]),
        x="Values",
        y="Efficiency",
        # Draw lines between points
        markers=False,
        line_shape="spline",
        # error_y="EfficiencyError",
        facet_col="Variable",
        line_dash="Sigma",
        # labels={
        #     "Values": "Coordinate (cm)",
        #     "Efficiency": "Reconstruction Efficiency (%)",
        # },
        color_discrete_sequence=default,
    )

    fig = format_coustom_plotly(
        fig,
        ranges=(None, [0, 110]),
        title=f"Vertex Reconstruction Efficiency - {config}",
        legend_title="Sigma (cm)",
        matches=(None, None),
    )
    fig.update_yaxes(title_text="")
    fig.update_yaxes(title_text="Reconstruction Efficiency (%)", row=1, col=1)

    save_figure(
        fig,
        save_path,
        config,
        name,
        filename=f"Vertex_Reconstruction_Efficiency",
        rm=user_input["rewrite"],
        debug=user_input["debug"],
    )

    for sigma in [1, 3, 5]:
        fig = px.line(
            df_position[
                (df_position["Energy"].notna()) * (df_position["Sigma"] == sigma)
            ].explode(["Values", "Efficiency"]),
            x="Values",
            y="Efficiency",
            # Draw lines between points
            markers=False,
            line_shape="spline",
            # error_y="EfficiencyError",
            color="Energy",
            facet_col="Variable",
            # line_dash="Sigma",
            labels={
                "Values": "Coordinate (cm)",
                "Efficiency": "Reconstruction Efficiency (%)",
            },
            color_discrete_sequence=colors,
        )
        fig = format_coustom_plotly(
            fig,
            ranges=(None, [0, 110]),
            title=f"Reconstruction Efficiency Sigma{sigma} - {config}",
            legend_title="Energy (MeV)",
            matches=(None, None),
        )
        fig.update_yaxes(title_text="")
        fig.update_yaxes(title_text="Vertex Reconstruction Efficiency", row=1, col=1)
        fig.add_hline(y=100, line_dash="dash", line_color="black")
        fig.update_layout(
            legend=dict(
                traceorder="normal",
                itemsizing="constant",
            )
        )
        save_figure(
            fig,
            save_path,
            config,
            name,
            filename=f"Vertex_Reconstruction_Efficiency_Energy_Sigma{sigma}",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )

    # Create a df with the values at 100 cm position cut from 8, 16, 24 MeV
    summary_list = []
    for energy in [6, 10, 14]:
        for coord in ["X", "Y", "Z"]:
            this_df = df_position[
                (df_position["Energy"] == energy) * (df_position["Variable"] == coord)
            ].explode(["Efficiency", "EfficiencyError"])
            if len(this_df) > 0:
                summary_list.append(
                    {
                        "Energy": energy,
                        "Coordinate": coord,
                        "Efficiency": np.mean(this_df["Efficiency"].values),
                        "EfficiencyError": np.std(this_df["Efficiency"].values),
                    }
                )
    summary_df = pd.DataFrame(summary_list)
    for this_filename, this_filetype, this_df in zip(
        ["Vertex", "Summary_100cm"], ["pkl", "tex"], [df_position, summary_df]
    ):
        save_df(
            this_df,
            data_path,
            config,
            name,
            filename=f"{this_filename}_Reconstruction_Efficiency",
            rm=user_input["rewrite"],
            filetype=this_filetype,
            debug=user_input["debug"],
        )
