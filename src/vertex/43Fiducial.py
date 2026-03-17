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
    reference="Truth",
    fiducial: int = 100,
    inverse: bool = False,
    coordinate: Optional[list[str]] = ["X", "Y", "Z"],
):
    if coordinate is None:
        coordinate = ["X", "Y", "Z"]
    true_mask = np.ones(len(run[reference]["SignalParticleX"]), dtype=bool)
    reco_mask = np.ones(len(run["Reco"]["SignalParticleX"]), dtype=bool)
    for coord in coordinate:
        if coord == "X":
            if info["VERSION"] == "hd_1x2x6_lateralAPA":
                if inverse:
                    true_mask = true_mask * (
                        run[reference]["SignalParticleX"] > fiducial
                    )
                    for variable in ["RecoX", "SignalParticleX"]:
                        reco_mask = reco_mask * (run["Reco"][variable] > fiducial)
                else:
                    true_mask = true_mask * (
                        run[reference]["SignalParticleX"] < fiducial
                    )
                    for variable in ["RecoX", "SignalParticleX"]:
                        reco_mask = reco_mask * (run["Reco"][variable] < fiducial)

            elif info["VERSION"] in [
                "hd_1x2x6_centralAPA",
                "hd_1x2x6",
            ]:
                if inverse:
                    true_mask = true_mask * (
                        np.absolute(run[reference]["SignalParticleX"])
                        < info["DETECTOR_MAX_X"] - fiducial
                    )
                    for variable in ["RecoX", "SignalParticleX"]:
                        reco_mask = reco_mask * (
                            np.absolute(run["Reco"][variable])
                            < info["DETECTOR_MAX_X"] - fiducial
                        )
                else:
                    true_mask = true_mask * (
                        np.absolute(run[reference]["SignalParticleX"])
                        > info["DETECTOR_MAX_X"] - fiducial
                    )
                    for variable in ["RecoX", "SignalParticleX"]:
                        reco_mask = reco_mask * (
                            np.absolute(run["Reco"][variable])
                            > info["DETECTOR_MAX_X"] - fiducial
                        )

            elif info["GEOMETRY"] == "vd":
                if inverse:
                    true_mask = true_mask * (
                        run[reference]["SignalParticleX"]
                        < info["DETECTOR_MAX_X"] - fiducial
                    )
                    for variable in ["RecoX", "SignalParticleX"]:
                        reco_mask = reco_mask * (
                            run["Reco"][variable] < info["DETECTOR_MAX_X"] - fiducial
                        )
                else:
                    true_mask = true_mask * (
                        run[reference]["SignalParticleX"]
                        > info["DETECTOR_MAX_X"] - fiducial
                    )
                    for variable in ["RecoX", "SignalParticleX"]:
                        reco_mask = reco_mask * (
                            run["Reco"][variable] > info["DETECTOR_MAX_X"] - fiducial
                        )

            else:
                rprint(
                    f"[red]ERROR[/red] Unknown geometry {info['GEOMETRY']} and version {info['VERSION']} for fiducial cut in X"
                )

        if coord == "Y":
            if inverse:
                true_mask = true_mask * (
                    np.absolute(run[reference]["SignalParticleY"])
                    < info["DETECTOR_MAX_Y"] - fiducial
                )
                for variable in ["RecoY", "SignalParticleY"]:
                    reco_mask = reco_mask * (
                        np.absolute(run["Reco"][variable])
                        < info["DETECTOR_MAX_Y"] - fiducial
                    )
            else:
                true_mask = true_mask * (
                    (
                        np.absolute(run[reference]["SignalParticleY"])
                        > info["DETECTOR_MAX_Y"] - fiducial
                    )
                )
                for variable in ["RecoY", "SignalParticleY"]:
                    reco_mask = reco_mask * (
                        (
                            np.absolute(run["Reco"][variable])
                            > info["DETECTOR_MAX_Y"] - fiducial
                        )
                    )

        if coord == "Z":
            if inverse:
                true_mask = true_mask * (
                    (run[reference]["SignalParticleZ"] > info["DETECTOR_MIN_Z"] + fiducial)
                    + (
                        run[reference]["SignalParticleZ"]
                        < info["DETECTOR_MAX_Z"] - fiducial
                    )
                )
                for variable in ["RecoZ", "SignalParticleZ"]:
                    reco_mask = reco_mask * (
                        (run["Reco"]["SignalParticleZ"] > info["DETECTOR_MIN_Z"] + fiducial)
                        + (run["Reco"][variable] < info["DETECTOR_MAX_Z"] - fiducial)
                    )
            else:
                true_mask = true_mask * (
                    (
                        run[reference]["SignalParticleZ"]
                        > info["DETECTOR_MAX_Z"] - fiducial
                    )
                    + (run[reference]["SignalParticleZ"] < info["DETECTOR_MIN_Z"] + fiducial)
                )
                for variable in ["RecoZ", "SignalParticleZ"]:
                    reco_mask = reco_mask * (
                        (
                            (
                                run["Reco"]["SignalParticleZ"]
                                > info["DETECTOR_MAX_Z"] - fiducial
                            )
                            + (run["Reco"]["SignalParticleZ"] < info["DETECTOR_MIN_Z"] + fiducial)
                        )
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
    params={("Reco", "TrueMain"): ("equal", True)},
    presets=["ANALYSIS"],
    signal = "marley" in args.name,
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
        fiducial_list = []
        info = json.load(open(f"{root}/config/{config}/{config}_config.json", "r"))
        this_filtered_run, mask, output = compute_filtered_run(
            run,
            {config: [name]},
            debug=user_input["debug"],
        )
        rprint(output)

        for reference, inverse, coord, energy in product(
            ["Reco", "Truth"], [True, False], ["X", "Y", "Z"], lowe_energy_centers
        ):
            counts = []
            counts_error = []
            efficiency = []
            efficiency_error = []
            for fiducial in np.arange(0, 220, 20):
                true_mask, reco_mask = fiducial_mask(
                    run, info, reference, fiducial, inverse, [coord]
                )
                if energy is None:
                    true_energy_mask = np.ones(
                        len(run[reference]["SignalParticleK"]), dtype=bool
                    )
                    reco_energy_mask = np.ones(
                        len(run["Reco"]["SignalParticleK"]), dtype=bool
                    )
                else:
                    true_energy_mask = (
                        run[reference]["SignalParticleK"] >= energy - lowe_ebin / 2
                    ) * (run[reference]["SignalParticleK"] < energy + lowe_ebin / 2)
                    reco_energy_mask = (
                        run["Reco"]["SignalParticleK"] >= energy - lowe_ebin / 2
                    ) * (run["Reco"]["SignalParticleK"] < energy + lowe_ebin / 2)

                this_true_mask = true_mask * (
                    (run[reference]["Geometry"] == info["GEOMETRY"])
                    * (run[reference]["Version"] == info["VERSION"])
                    * (run[reference]["Name"] == name)
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
                    if 100 * sum(this_reco_mask) / sum(this_true_mask) < 100
                    else 100
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
                    "Values": np.arange(0, 220, 20),
                    "Counts": counts,
                    "CountsError": counts_error,
                    "Efficiency": efficiency,
                    "EfficiencyError": efficiency_error,
                    "Inverse": inverse,
                    "Reference": reference,
                }
            )

    df_fiducial = pd.DataFrame(fiducial_list)
    df_fiducial = df_fiducial.fillna(np.nan)
    for inverse, reference in product([True, False], ["Truth"]):
        this_fiducial_df = df_fiducial[
            (df_fiducial["Inverse"] == inverse)
            * (df_fiducial["Reference"] == reference)
        ]
        fig = px.line(
            this_fiducial_df[(this_fiducial_df["Energy"].isna())].explode(
                ["Values", "Efficiency", "EfficiencyError"]
            ),
            x="Values",
            y="Efficiency",
            # Draw lines between points
            markers=True,
            line_shape="hvh",
            # error_y="EfficiencyError",
            color="Variable",
            labels={
                "Values": "Fiducial Cut (cm)",
                "Efficiency": (
                    "Fiducialization Efficiency (%)"
                    if not inverse
                    else "Reconstruction Efficiency (%)"
                ),
            },
            color_discrete_sequence=default,
        )
        fig.add_hline(y=100, line_dash="dash", line_color="black")

        fig = format_coustom_plotly(
            fig,
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
            fig,
            save_path,
            config,
            name,
            filename=(
                f"Vertex_Fiducial_Efficiency_{reference}"
                if not inverse
                else f"Vertex_Reconstruction_Efficiency_{reference}"
            ),
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )

    for inverse, reference in product([True, False], ["Truth", "Reco"]):
        this_fiducial_df = df_fiducial[
            (df_fiducial["Inverse"] == inverse)
            * (df_fiducial["Reference"] == reference)
        ]
        fig = px.line(
            this_fiducial_df[(this_fiducial_df["Energy"].notna())].explode(
                ["Values", "Efficiency", "EfficiencyError"]
            ),
            x="Values",
            y="Efficiency",
            # Draw lines between points
            markers=True,
            line_shape="hvh",
            # error_y="EfficiencyError",
            color="Energy",
            facet_col="Variable",
            color_discrete_sequence=colors,
        )
        fig.add_hline(y=100, line_dash="dash", line_color="black")

        fig = format_coustom_plotly(
            fig,
            ranges=(None, [0, 110]),
            title=(
                f"Fiducialization Efficiency vs Energy - {config}"
                if not inverse
                else f"Vertex Reconstruction Efficiency vs Fiducial Cut and Energy - {config}"
            ),
            legend_title="Energy (MeV)",
        )
        fig.update_yaxes(title_text="")
        fig.update_yaxes(
            title_text=(
                "Fiducialization Efficiency (%)"
                if not inverse
                else "Reconstruction Efficiency (%)"
            ),
            row=1,
            col=1,
        )

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
            filename=(
                f"Vertex_Fiducial_Efficiency_Energy_{reference}"
                if not inverse
                else f"Vertex_Reconstruction_Efficiency_Energy_{reference}"
            ),
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )

    for this_df, df_filename, df_type in zip(
        [df_fiducial],
        [
            "Fiducial_Efficiency",
        ],
        ["pkl"],
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
