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
    reference = 'Reco',
    energy: Optional[float] = 10,
    tolerances: Optional[list[int]] = [1],
    position: float = 0,
    coordinate_bin: Optional[float] = 10,
    coordinate: Optional[str] = "X",
    sigma: bool = True,
):
    true_mask_dict = {}
    reco_mask_dict = {}
    for idx, tolerance in enumerate(tolerances):
        true_mask = np.ones(len(run[reference]["SignalParticleX"]), dtype=bool)
        reco_mask = np.ones(len(run["Reco"]["SignalParticleX"]), dtype=bool)
        if coordinate == "X" and info["GEOMETRY"] == "hd":
            true_mask = true_mask * (
                (
                    np.absolute(run[reference][f"SignalParticle{coordinate}"])
                    > position - coordinate_bin / 2
                )
                * (
                    np.absolute(run[reference][f"SignalParticle{coordinate}"])
                    <= position + coordinate_bin / 2
                )
            )
            reco_mask = reco_mask * (
                (
                    np.absolute(run["Reco"][f"SignalParticle{coordinate}"])
                    < position + coordinate_bin / 2
                )
                * (
                    np.absolute(run["Reco"][f"SignalParticle{coordinate}"])
                    >= position - coordinate_bin / 2
                )
            )
        else:
            true_mask = true_mask * (
                (
                    run[reference][f"SignalParticle{coordinate}"]
                    > position - coordinate_bin / 2
                )
                * (
                    run[reference][f"SignalParticle{coordinate}"]
                    <= position + coordinate_bin / 2
                )
            )
            reco_mask = reco_mask * (
                (
                    run["Reco"][f"SignalParticle{coordinate}"]
                    < position + coordinate_bin / 2
                )
                * (
                    run["Reco"][f"SignalParticle{coordinate}"]
                    >= position - coordinate_bin / 2
                )
            )

        if sigma:
            reco_mask = reco_mask * (
                np.absolute(run["Reco"][f"Error{coordinate}"])
                < tolerance
                * df[
                    (df["Coordinate"] == f"{coordinate}")
                    * (
                        df["Energy"] == energy
                        if energy != None
                        else df["Energy"].isna()
                    )
                ]["Sigma"].values[0]
            )
        else:
            reco_mask = reco_mask * (
                np.absolute(run["Reco"][f"Error{coordinate}"]) < tolerance
            )

        true_mask_dict[tolerance] = true_mask
        reco_mask_dict[tolerance] = reco_mask

    return true_mask_dict, reco_mask_dict


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
        position_list, hist_list = [], []
        missing_energies = []   
        info = json.load(open(f"{root}/config/{config}/{config}_config.json", "r"))
        analysis_info = json.load(open(f"{root}/import/analysis.json", "r"))

        this_filtered_run, mask, output = compute_filtered_run(
            run,
            {config: [name]},
            debug=user_input["debug"],
        )
        rprint(output)

        df = pickle.load(
        open(
            f"{root}/data/vertex/resolution/{config}/{name}/{config}_{name}_Resolution.pkl",
            "rb",
            )
        )
        print( df.head())
        df = df.fillna(np.nan)
        for energy, coord in product(lowe_energy_centers, ["X", "Y", "Z"]):
            if energy not in df["Energy"].values and energy is not None:
                missing_energies.append(energy)
                continue
            
            counts, counts_error = {True: {}, False: {}}, {True: {}, False: {}}
            efficiency, efficiency_error = {True: {}, False: {}}, {True: {}, False: {}}

            for sigma, tolerance in zip(
                (
                    [True] * len(analysis_info["VERTEX_RESOLUTION_SIGMAS"])
                    + [False] * len(analysis_info["VERTEX_RESOLUTION_TOLERANCES"])
                ),
                (
                    analysis_info["VERTEX_RESOLUTION_SIGMAS"]
                    + analysis_info["VERTEX_RESOLUTION_TOLERANCES"]
                ),
            ):
                counts[sigma][tolerance] = []
                counts_error[sigma][tolerance] = []
                efficiency[sigma][tolerance] = []
                efficiency_error[sigma][tolerance] = []

            if coord == "X" and info["GEOMETRY"] == "hd":
                coordinate_array = np.arange(
                    0,
                    info[f"DETECTOR_MAX_{coord}"]
                    + info[f"DETECTOR_GAP_{coord}"]
                    + params[f"DEFAULT_{coord}_BIN"],
                    params[f"DEFAULT_{coord}_BIN"],
                )

            else:
                coordinate_array = np.arange(
                    info[f"DETECTOR_MIN_{coord}"] - info[f"DETECTOR_GAP_{coord}"],
                    info[f"DETECTOR_MAX_{coord}"]
                    + info[f"DETECTOR_GAP_{coord}"]
                    + params[f"DEFAULT_{coord}_BIN"],
                    params[f"DEFAULT_{coord}_BIN"],
                )

            true_mask, reco_mask = {}, {}
            for position, (tolerances, sigma), reference in product(
                coordinate_array,
                zip(
                    [
                        analysis_info["VERTEX_RESOLUTION_SIGMAS"],
                        analysis_info["VERTEX_RESOLUTION_TOLERANCES"],
                    ],
                    [True, False],
                ),
                ['Reco']
            ):
                
                true_mask[sigma], reco_mask[sigma] = position_mask(
                    run=run,
                    info=info,
                    reference=reference,
                    energy=energy,
                    tolerances=tolerances,
                    position=position,
                    coordinate_bin=params[f"DEFAULT_{coord}_BIN"],
                    coordinate=coord,
                    sigma=sigma,
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

                for tolerance in tolerances:
                    this_true_mask = true_mask[sigma][tolerance] * (
                        (run[reference]["Geometry"] == info["GEOMETRY"])
                        * (run[reference]["Version"] == info["VERSION"])
                        * (run[reference]["Name"] == name)
                        * true_energy_mask
                    )
                    this_reco_mask = reco_mask[sigma][tolerance] * (
                        (run["Reco"]["Geometry"] == info["GEOMETRY"])
                        * (run["Reco"]["Version"] == info["VERSION"])
                        * (run["Reco"]["Name"] == name)
                        * reco_energy_mask
                    )
                    counts[sigma][tolerance].append(sum(this_reco_mask))
                    counts_error[sigma][tolerance].append(np.sqrt(sum(this_reco_mask)))
                    efficiency[sigma][tolerance].append(
                        100
                        * (
                            sum(this_reco_mask) / sum(this_true_mask)
                            if sum(this_true_mask) > 0
                            else 0
                        )
                    )
                    efficiency_error[sigma][tolerance].append(
                        100
                        * (
                            np.sqrt(sum(this_reco_mask)) / sum(this_true_mask)
                            if sum(this_true_mask) > 0
                            else 0
                        )
                    )

            for sigma, tolerance in zip(
                (
                    [True] * len(analysis_info["VERTEX_RESOLUTION_SIGMAS"])
                    + [False] * len(analysis_info["VERTEX_RESOLUTION_TOLERANCES"])
                ),
                (
                    analysis_info["VERTEX_RESOLUTION_SIGMAS"]
                    + analysis_info["VERTEX_RESOLUTION_TOLERANCES"]
                ),
            ):
                if energy in missing_energies:
                    continue
                position_list.append(
                    {
                        "Geometry": info["GEOMETRY"],
                        "Config": config,
                        "Name": name,
                        "Variable": coord,
                        "Energy": energy,
                        "Tolerance": tolerance,
                        "Reference": reference,
                        "Values": (
                            2 * info[f"DETECTOR_MAX_{coord}"] - coordinate_array
                            if (config == "hd_1x2x6_lateralAPA" and coord == "X")
                            else (
                                coordinate_array + info[f"DETECTOR_MAX_{coord}"]
                                if (info["GEOMETRY"] == "vd" and coord == "X")
                                else coordinate_array
                            )
                        ),
                        "Counts": counts[sigma][tolerance],
                        "CountsError": counts_error[sigma][tolerance],
                        "Efficiency": efficiency[sigma][tolerance],
                        "EfficiencyError": efficiency_error[sigma][tolerance],
                        "Sigma": sigma,
                    }
                )

    df_position = pd.DataFrame(position_list)
    df_position = df_position.fillna(np.nan)

    for sigma in [True, False]:
        fig = px.line(
            df_position[
                (df_position["Energy"].isna()) * (df_position["Sigma"] == sigma)
            ].explode(["Values", "Efficiency", "EfficiencyError"]),
            x="Values",
            y="Efficiency",
            error_y="EfficiencyError",
            markers=True,
            line_shape="spline",
            facet_col="Variable",
            color="Tolerance",
            color_discrete_sequence=default,
        )
        fig.add_hline(y=100, line_dash="dash", line_color="black")
        fig = format_coustom_plotly(
            fig,
            ranges=(None, [0, 110]),
            title=f"Vertex Reconstruction Efficiency - {config}",
            legend_title="Sigmas" if sigma else "Tolerance (cm)",
            matches=(None, None),
        )
        fig.update_yaxes(title_text="")
        fig.update_yaxes(title_text="Reconstruction Efficiency (%)", row=1, col=1)

        save_figure(
            fig,
            save_path,
            config,
            name,
            filename=f"Vertex_Reconstruction_Efficiency_{'Sigma' if sigma else 'Tolerance'}",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )

    for tolerance in analysis_info["VERTEX_RESOLUTION_SIGMAS"]:
        fig = px.line(
            df_position[
                (df_position["Energy"].notna())
                * (df_position["Sigma"] == True)
                * (df_position["Tolerance"] == tolerance)
            ].explode(["Values", "Efficiency", "EfficiencyError"]),
            x="Values",
            y="Efficiency",
            error_y="EfficiencyError",
            markers=True,
            line_shape="spline",
            facet_col="Variable",
            color="Energy",
            color_discrete_sequence=colors,
        )
        fig = format_coustom_plotly(
            fig,
            ranges=(None, [0, 110]),
            title=f"Reconstruction Efficiency Sigma{tolerance} - {config}",
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
            filename=f"Vertex_Reconstruction_Efficiency_Energy_Sigma{tolerance}",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )

    # Create a df with the values at 100 cm position cut from 8, 16, 24 MeV
    summary_list = []
    for energy in [6, 10, 14]:
        for coord in ["X", "Y", "Z"]:
            this_df = df_position[
                (df_position["Energy"] == energy)
                * (df_position["Variable"] == coord)
                * (df_position["Sigma"] == True)
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
