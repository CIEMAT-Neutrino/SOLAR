import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *

save_path = f"{root}/images/vertex/fiducial"
data_path = f"{root}/data/vertex/fiducial"


def fiducial_mask(
    run,
    info,
    energy: Optional[float] = 10,
    sigma: int = 3,
    fiducial: int = 100,
    inverse: bool = False,
    coordinate: Optional[list[str]] = ["X", "Y", "Z"],
):
    if coordinate is None:
        coordinate = ["X", "Y", "Z"]
    true_mask = np.ones(len(run["Truth"]["SignalParticleX"]), dtype=bool)
    reco_mask = np.ones(len(run["Reco"]["SignalParticleX"]), dtype=bool)
    for coord in coordinate:
        if coord == "X":
            if info["GEOMETRY"] == "hd" and info["VERSION"] == "hd_1x2x6_lateralAPA":
                if inverse:
                    true_mask = true_mask * (
                        np.absolute(run["Truth"]["SignalParticleX"])
                        > fiducial - info["DETECTOR_GAP_X"]
                    )
                    reco_mask = reco_mask * (
                        np.absolute(run["Reco"]["SignalParticleX"])
                        > fiducial - info["DETECTOR_GAP_X"]
                    )
                else:
                    true_mask = true_mask * (
                        np.absolute(run["Truth"]["SignalParticleX"])
                        < fiducial - info["DETECTOR_GAP_X"]
                    )
                    reco_mask = reco_mask * (
                        np.absolute(run["Reco"]["SignalParticleX"])
                        < fiducial - info["DETECTOR_GAP_X"]
                    )
            elif info["GEOMETRY"] == "hd" and info["VERSION"] == "hd_1x2x6_centralAPA":
                if inverse:
                    true_mask = true_mask * (
                        np.absolute(run["Truth"]["SignalParticleX"])
                        < info["DETECTOR_MAX_X"] + info["DETECTOR_GAP_X"] - fiducial
                    )
                    reco_mask = reco_mask * (
                        np.absolute(run["Reco"]["SignalParticleX"])
                        < info["DETECTOR_MAX_X"] + info["DETECTOR_GAP_X"] - fiducial
                    )
                else:
                    true_mask = true_mask * (
                        np.absolute(run["Truth"]["SignalParticleX"])
                        > info["DETECTOR_MAX_X"] + info["DETECTOR_GAP_X"] - fiducial
                    )
                    reco_mask = reco_mask * (
                        np.absolute(run["Reco"]["SignalParticleX"])
                        > info["DETECTOR_MAX_X"] + info["DETECTOR_GAP_X"] - fiducial
                    )
            elif info["GEOMETRY"] == "hd" and info["VERSION"] not in [
                "hd_1x2x6_lateralAPA",
                "hd_1x2x6_centralAPA",
            ]:
                if inverse:
                    true_mask = true_mask * (
                        np.absolute(run["Truth"]["SignalParticleX"])
                        < info["DETECTOR_SIZE_X"] / 2
                        + info["DETECTOR_GAP_X"]
                        - fiducial
                    )
                    reco_mask = reco_mask * (
                        np.absolute(run["Reco"]["SignalParticleX"])
                        < info["DETECTOR_SIZE_X"] / 2
                        + info["DETECTOR_GAP_X"]
                        - fiducial
                    )
                else:
                    true_mask = true_mask * (
                        np.absolute(run["Truth"]["SignalParticleX"])
                        > info["DETECTOR_SIZE_X"] / 2
                        + info["DETECTOR_GAP_X"]
                        - fiducial
                    )
                    reco_mask = reco_mask * (
                        np.absolute(run["Reco"]["SignalParticleX"])
                        > info["DETECTOR_SIZE_X"] / 2
                        + info["DETECTOR_GAP_X"]
                        - fiducial
                    )

            elif info["GEOMETRY"] == "vd":
                if inverse:
                    true_mask = true_mask * (
                        run["Truth"]["SignalParticleX"]
                        < info["DETECTOR_MAX_X"] - fiducial
                    )
                    reco_mask = reco_mask * (
                        run["Reco"]["SignalParticleX"]
                        < info["DETECTOR_MAX_X"] - fiducial
                    )
                else:
                    true_mask = true_mask * (
                        run["Truth"]["SignalParticleX"]
                        > info["DETECTOR_MAX_X"] - fiducial
                    )
                    reco_mask = reco_mask * (
                        run["Reco"]["SignalParticleX"]
                        > info["DETECTOR_MAX_X"] - fiducial
                    )

            else:
                rprint(
                    f"[red]ERROR[/red] Unknown geometry {info['GEOMETRY']} and version {info['VERSION']} for fiducial cut in X"
                )
            reco_mask = reco_mask * (
                np.absolute(run["Reco"]["ErrorX"])
                < df[
                    (df["Coordinate"] == "X")
                    * (
                        df["Energy"] == energy
                        if energy != None
                        else df["Energy"].isna()
                    )
                ][f"Sigma{sigma}"].values[0]
            )

        if coord == "Y":
            if inverse:
                true_mask = true_mask * (
                    np.absolute(run["Truth"]["SignalParticleY"])
                    < info["DETECTOR_MAX_Y"] - fiducial
                )
                reco_mask = reco_mask * (
                    np.absolute(run["Reco"]["SignalParticleY"])
                    < info["DETECTOR_MAX_Y"] - fiducial
                )
            else:
                true_mask = true_mask * (
                    (
                        np.absolute(run["Truth"]["SignalParticleY"])
                        > info["DETECTOR_MAX_Y"] - fiducial
                    )
                )
                reco_mask = reco_mask * (
                    (
                        np.absolute(run["Reco"]["SignalParticleY"])
                        > info["DETECTOR_MAX_Y"] - fiducial
                    )
                )
            reco_mask = reco_mask * (
                np.absolute(run["Reco"]["ErrorY"])
                < df[
                    (df["Coordinate"] == "Y")
                    * (
                        df["Energy"] == energy
                        if energy != None
                        else df["Energy"].isna()
                    )
                ][f"Sigma{sigma}"].values[0]
            )

        if coord == "Z":
            if inverse:
                true_mask = true_mask * (
                    (run["Truth"]["SignalParticleZ"] > fiducial)
                    + (
                        run["Truth"]["SignalParticleZ"]
                        < info["DETECTOR_SIZE_Z"] - fiducial
                    )
                )
                reco_mask = reco_mask * (
                    (run["Reco"]["SignalParticleZ"] > fiducial)
                    + (
                        run["Reco"]["SignalParticleZ"]
                        < info["DETECTOR_SIZE_Z"] - fiducial
                    )
                )
            else:
                true_mask = true_mask * (
                    (
                        run["Truth"]["SignalParticleZ"]
                        > info["DETECTOR_SIZE_Z"] - fiducial
                    )
                    + (run["Truth"]["SignalParticleZ"] < fiducial)
                )
                reco_mask = reco_mask * (
                    (
                        (
                            run["Reco"]["SignalParticleZ"]
                            > info["DETECTOR_SIZE_Z"] - fiducial
                        )
                        + (run["Reco"]["SignalParticleZ"] < fiducial)
                    )
                )
            reco_mask = reco_mask * (
                np.absolute(run["Reco"]["ErrorZ"])
                < df[
                    (df["Coordinate"] == "Z")
                    * (
                        df["Energy"] == energy
                        if energy != None
                        else df["Energy"].isna()
                    )
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
        fiducial_list, hist_list = [], []
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

        for inverse, coord, energy, sigma in product(
            [True, False],
            ["X", "Y", "Z"],
            lowe_energy_centers,
            [5, 3, 1],
        ):
            counts = []
            counts_error = []
            efficiency = []
            efficiency_error = []
            for fiducial in np.arange(0, 120, 20):
                true_mask, reco_mask = fiducial_mask(
                    run, info, energy, sigma, fiducial, inverse, [coord]
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
                    100 * sum(this_reco_mask) / sum(this_true_mask)
                    if sum(this_true_mask) > 0
                    else 0
                )
                efficiency_error.append(
                    100 * np.sqrt(sum(this_reco_mask)) / sum(this_true_mask)
                    if sum(this_true_mask) > 0
                    else 0
                )

            fiducial_list.append(
                {
                    "Geometry": info["GEOMETRY"],
                    "Config": config,
                    "Name": name,
                    "Variable": coord if coord is not None else "XYZ",
                    "Energy": energy,
                    "Sigma": sigma,
                    "Values": np.arange(0, 120, 20),
                    "Counts": counts,
                    "CountsError": counts_error,
                    "Efficiency": efficiency,
                    "EfficiencyError": efficiency_error,
                    "Inverse": inverse,
                }
            )

    df_fiducial = pd.DataFrame(fiducial_list)
    df_fiducial = df_fiducial.fillna(np.nan)
    for inverse in [True, False]:
        this_fiducial_df = df_fiducial[df_fiducial["Inverse"] == inverse]
        fig1 = px.line(
            this_fiducial_df[(this_fiducial_df["Energy"].isna())].explode(
                ["Values", "Efficiency", "EfficiencyError"]
            ),
            x="Values",
            y="Efficiency",
            # Draw lines between points
            markers=False,
            line_shape="spline",
            # error_y="EfficiencyError",
            color="Variable",
            line_dash="Sigma",
            labels={
                "Values": "Fiducial Cut (cm)",
                "Efficiency": (
                    "Fiducialization Efficiency"
                    if not inverse
                    else "Reconstruction Efficiency"
                ),
            },
            color_discrete_sequence=default,
        )

        fig1 = format_coustom_plotly(
            fig1,
            ranges=(None, [0, 110]),
            title=(
                f"Fiducialization Efficiency - {config}"
                if not inverse
                else f"Vertex Reconstruction Efficiency vs Fiducial Cut - {config}"
            ),
            legend_title="Variable",
            legend=dict(y=0.06, x=0.82),
        )

        save_figure(
            fig1,
            save_path,
            config,
            name,
            filename=(
                f"Vertex_Fiducial_Efficiency"
                if not inverse
                else f"Vertex_Reconstruction_Efficiency_Fiducial_Scan"
            ),
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )

        fig2 = px.line(
            this_fiducial_df[(this_fiducial_df["Energy"].notna())].explode(
                ["Values", "Efficiency", "EfficiencyError"]
            ),
            x="Values",
            y="Efficiency",
            # Draw lines between points
            markers=False,
            line_shape="spline",
            # error_y="EfficiencyError",
            color="Energy",
            facet_col="Variable",
            line_dash="Sigma",
            labels={
                "Values": "Fiducial Cut (cm)",
                "Efficiency": (
                    "Fiducialization Efficiency"
                    if not inverse
                    else "Reconstruction Efficiency"
                ),
            },
            color_discrete_sequence=colors,
        )
        fig2 = format_coustom_plotly(
            fig2,
            ranges=(None, [0, 110]),
            title=(
                f"Fiducialization Efficiency vs Energy - {config}"
                if not inverse
                else f"Vertex Reconstruction Efficiency vs Fiducial Cut and Energy - {config}"
            ),
            legend_title="Energy (MeV)",
        )
        fig2.update_yaxes(title_text="")
        fig2.update_yaxes(
            title_text=(
                "Fiducialization Efficiency"
                if not inverse
                else "Reconstruction Efficiency"
            ),
            row=1,
            col=1,
        )
        fig2.add_hline(y=100, line_dash="dash", line_color="black")
        fig2.update_layout(
            legend=dict(
                traceorder="normal",
                itemsizing="constant",
            )
        )
        save_figure(
            fig2,
            save_path,
            config,
            name,
            filename=(
                f"Vertex_Fiducial_Efficiency_Energy"
                if not inverse
                else f"Vertex_Reconstruction_Efficiency_Fiducial_Energy_Scan"
            ),
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )

    # Create a df with the values at 100 cm fiducial cut from 8, 16, 24 MeV
    summary_list = []
    for energy in [6, 10, 14]:
        for coord in ["X", "Y", "Z"]:
            this_df = this_fiducial_df.explode(
                ["Values", "Efficiency", "EfficiencyError"]
            )
            this_df = this_df[
                (this_df["Energy"] == energy)
                * (this_df["Variable"] == coord)
                * (this_df["Values"] == 100)
            ]
            if len(this_df) > 0:
                summary_list.append(
                    {
                        "Energy": energy,
                        "Coordinate": coord,
                        "Efficiency": this_df["Efficiency"].values[0],
                        "EfficiencyError": this_df["EfficiencyError"].values[0],
                    }
                )

    summary_df = pd.DataFrame(summary_list)
    for this_df, df_filename, df_type in zip(
        [df_fiducial, summary_df],
        [
            "Fiducial_Efficiency",
            (
                "Fiducial_Efficiency_Summary_100cm"
                if not inverse
                else "Reconstruction_Efficiency_Summary_100cm"
            ),
        ],
        ["pkl", "tex"],
    ):
        save_df(
            this_df,
            data_path,
            config,
            name,
            filename=df_filename,
            rm=user_input["rewrite"],
            filetype=df_type,
            debug=user_input["debug"],
        )
