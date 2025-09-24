import sys

sys.path.insert(0, "../../")

from lib import *

save_path = f"{root}/images/PDS/matchedopflash/"
save_data = f"{root}/data/PDS/matchedopflash/"

for path in [save_path, save_data]:
    if not os.path.exists(path):
        os.makedirs(path)

# Define flags for the analysis config and name with the python parser
parser = argparse.ArgumentParser(
    description="Plot the MatchedOpFlash distributions of the signal"
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

user_input = {"workflow": "MATCHEDFLASH", "rewrite": args.rewrite, "debug": args.debug}

run, output = load_multi(
    configs, preset=user_input["workflow"], debug=user_input["debug"]
)
rprint(output)

run = compute_reco_workflow(
    run, configs, workflow=user_input["workflow"], debug=user_input["debug"]
)

efficiency_plot = []


def fill_plot_list(
    efficiency_plot,
    this_filtered_run,
    name,
    plot,
    energy,
    coord_x,
    coord_y,
    coord_z,
    plane,
):
    if plane is None:
        true_match = np.sum(this_filtered_run["Truth"]["PDSMatch"])
        false_match = np.sum(~this_filtered_run["Truth"]["PDSMatch"])
    else:
        true_match = np.sum(
            this_filtered_run["Truth"]["PDSMatch"],
            where=this_filtered_run["Truth"]["PDSPlane"] == plane,
        )
        false_match = np.sum(
            ~this_filtered_run["Truth"]["PDSMatch"],
            where=this_filtered_run["Truth"]["PDSPlane"] != plane,
        )

    true_PE = np.mean(
        this_filtered_run["Truth"]["PDSPE"],
        where=this_filtered_run["Truth"]["PDSMatch"],
    )
    reco_true_match = np.sum(this_filtered_run["Reco"]["MatchedOpFlashSignal"])
    reco_match = np.sum(this_filtered_run["Reco"]["MatchedOpFlashPE"] > -1e6)
    rate_PE = np.mean(
        this_filtered_run["Reco"]["MatchedOpFlashPE"],
        where=this_filtered_run["Reco"]["MatchedOpFlashSignal"],
    )
    # print(len(this_filtered_run["Reco"]["MatchedOpFlashPE"]))
    try:
        rate = 100 * reco_true_match / (reco_match)
        rate_error = 100 * np.sqrt(reco_true_match * (reco_match)) / (reco_match) ** 2
        tpc = 100 * true_match / np.sum(this_filtered_run["Truth"]["RecoMatch"])
        tpc_PE = np.mean(
            this_filtered_run["Truth"]["PDSPE"],
            where=this_filtered_run["Truth"]["RecoMatch"],
        )
        tpc_error = (
            100
            * np.sqrt(true_match * (np.sum(this_filtered_run["Truth"]["RecoMatch"])))
            / (np.sum(this_filtered_run["Truth"]["RecoMatch"]) ** 2)
        )
        true = 100 * true_match / (true_match + false_match)
        true_error = (
            100
            * np.sqrt(true_match * (true_match + false_match))
            / ((true_match + false_match) ** 2)
        )

    except ZeroDivisionError:
        rate = np.nan
        rate_error = np.nan
        tpc = np.nan
        tpc_error = np.nan
        true = np.nan
        true_error = np.nan

    efficiency_plot.append(
        {
            "Name": name,
            "Plot": plot,
            "Type": "Matched_Flash",
            "Energy": energy,
            "X": coord_x,
            "Y": coord_y,
            "Z": coord_z,
            "Plane": plane,
            "Efficiency": rate,
            "Error": rate_error,
            "PE": rate_PE,
        }
    )
    efficiency_plot.append(
        {
            "Name": name,
            "Plot": plot,
            "Type": "TPC_Cluster",
            "Energy": energy,
            "X": coord_x,
            "Y": coord_y,
            "Z": coord_z,
            "Plane": plane,
            "Efficiency": tpc,
            "Error": tpc_error,
            "PE": tpc_PE,
        }
    )
    efficiency_plot.append(
        {
            "Name": name,
            "Plot": plot,
            "Type": "True_Neutrino",
            "Energy": energy,
            "X": coord_x,
            "Y": coord_y,
            "Z": coord_z,
            "Plane": plane,
            "Efficiency": true,
            "Error": true_error,
            "PE": true_PE,
        }
    )


for config in configs:
    info, params, output = get_param_dict(
        f"{root}/config/{config}/{config}", {}, output, debug=user_input["debug"]
    )
    for idx, name in enumerate(configs[config]):
        this_run, mask, output = compute_filtered_run(
            run,
            configs,
            params={
                ("Truth", "Name"): ("equal", name),
                ("Truth", "Geometry"): ("equal", info["GEOMETRY"]),
                ("Truth", "Version"): ("equal", info["VERSION"]),
            },
            debug=user_input["debug"],
        )
        reco_energy_bin = reco_energy_centers[1] - reco_energy_centers[0]
        for energy in reco_energy_centers:
            for coord_label in ["X", "Y", "Z"]:
                for coord in np.arange(
                    info[f"DETECTOR_MIN_{coord_label}"]
                    + params[f"DEFAULT_{coord_label}_REGION"] / 2,
                    info[f"DETECTOR_MAX_{coord_label}"]
                    + params[f"DEFAULT_{coord_label}_REGION"] / 2,
                    params[f"DEFAULT_{coord_label}_REGION"],
                ):

                    this_energy_run, mask, output = compute_filtered_run(
                        this_run,
                        configs,
                        params={
                            ("Truth", "SignalParticleK"): (
                                "between",
                                [
                                    energy - reco_energy_bin / 2,
                                    energy + reco_energy_bin / 2,
                                ],
                            ),
                            ("Reco", "SignalParticleK"): (
                                "between",
                                [
                                    energy - reco_energy_bin / 2,
                                    energy + reco_energy_bin / 2,
                                ],
                            ),
                            ("Truth", f"SignalParticle{coord_label}"): (
                                "between",
                                [
                                    coord - params[f"DEFAULT_{coord_label}_REGION"] / 2,
                                    coord + params[f"DEFAULT_{coord_label}_REGION"] / 2,
                                ],
                            ),
                            ("Reco", f"SignalParticle{coord_label}"): (
                                "between",
                                [
                                    coord - params[f"DEFAULT_{coord_label}_REGION"] / 2,
                                    coord + params[f"DEFAULT_{coord_label}_REGION"] / 2,
                                ],
                            ),
                        },
                        debug=user_input["debug"],
                    )
                    fill_plot_list(
                        efficiency_plot,
                        this_energy_run,
                        name,
                        "Energy Scan",
                        energy,
                        coord_x=coord if coord_label == "X" else None,
                        coord_y=coord if coord_label == "Y" else None,
                        coord_z=coord if coord_label == "Z" else None,
                        plane=None,
                    )
        red_energy_bin = red_energy_centers[1] - red_energy_centers[0]
        for energy, coord in product(red_energy_centers, ["X", "Y", "Z"]):
            for value in np.arange(
                info[f"DETECTOR_MIN_{coord}"] + params[f"DEFAULT_{coord}_BIN"] / 2,
                info[f"DETECTOR_MAX_{coord}"] + params[f"DEFAULT_{coord}_BIN"] / 2,
                params[f"DEFAULT_{coord}_BIN"],
            ):
                this_drift_run, mask, output = compute_filtered_run(
                    this_run,
                    configs,
                    params={
                        ("Truth", "SignalParticleK"): (
                            "between",
                            [
                                energy - red_energy_bin / 2,
                                energy + red_energy_bin / 2,
                            ],
                        ),
                        ("Reco", "SignalParticleK"): (
                            "between",
                            [
                                energy - red_energy_bin / 2,
                                energy + red_energy_bin / 2,
                            ],
                        ),
                        ("Truth", f"SignalParticle{coord}"): (
                            "between",
                            [
                                value - params[f"DEFAULT_{coord}_BIN"] / 2,
                                value + params[f"DEFAULT_{coord}_BIN"] / 2,
                            ],
                        ),
                        ("Reco", f"SignalParticle{coord}"): (
                            "between",
                            [
                                value - params[f"DEFAULT_{coord}_BIN"] / 2,
                                value + params[f"DEFAULT_{coord}_BIN"] / 2,
                            ],
                        ),
                    },
                    debug=user_input["debug"],
                )
                fill_plot_list(
                    efficiency_plot,
                    this_drift_run,
                    name,
                    f"{coord} Scan",
                    energy,
                    value if coord == "X" else None,
                    value if coord == "Y" else None,
                    value if coord == "Z" else None,
                    None,
                )

        reco_energy_bin = reco_energy_centers[1] - reco_energy_centers[0]
        for energy in reco_energy_centers:
            for plane in [int(x) for x in info["OPFLASH_PLANES"].keys()]:
                this_plane_run, mask, output = compute_filtered_run(
                    this_run,
                    configs,
                    params={
                        ("Truth", "SignalParticleK"): (
                            "between",
                            [
                                energy - reco_energy_bin / 2,
                                energy + reco_energy_bin / 2,
                            ],
                        ),
                        ("Reco", "SignalParticleK"): (
                            "between",
                            [
                                energy - reco_energy_bin / 2,
                                energy + reco_energy_bin / 2,
                            ],
                        ),
                        ("Reco", "MatchedOpFlashPlane"): ("equal", plane),
                    },
                    debug=user_input["debug"],
                )
                fill_plot_list(
                    efficiency_plot,
                    this_plane_run,
                    name,
                    "Plane Scan",
                    energy,
                    None,
                    None,
                    None,
                    plane,
                )

        efficiency_df = pd.DataFrame(efficiency_plot)
        matching_type = "TPC_Cluster"
        fig = make_subplots(
            rows=1, cols=3, subplot_titles=("Energy Scan", "Drift Scan", "Plane Scan")
        )
        this_drift_df = efficiency_df[
            (efficiency_df["Name"] == name)
            * (efficiency_df["Plot"] == "X Scan")
            * (efficiency_df["Type"] == matching_type)
        ]
        save_df(
            this_drift_df,
            save_data,
            config,
            name,
            filename=f"{matching_type}_Efficiency_Energy_Scan",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )
        this_energy_df = efficiency_df[
            (efficiency_df["Name"] == name)
            * (efficiency_df["Plot"] == "Energy Scan")
            * (efficiency_df["Type"] == matching_type)
        ]
        save_df(
            this_drift_df,
            save_data,
            config,
            name,
            filename=f"{matching_type}_Efficiency_Drift_Scan",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )
        this_plane_df = efficiency_df[
            (efficiency_df["Name"] == name)
            * (efficiency_df["Plot"] == "Plane Scan")
            * (efficiency_df["Type"] == matching_type)
        ]
        save_df(
            this_plane_df,
            save_data,
            config,
            name,
            filename=f"{matching_type}_Efficiency_Plane_Scan",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )
        for jdx, energy in enumerate(np.unique(this_drift_df["Energy"])):
            this_efficiency_df = this_drift_df[this_drift_df["Energy"] == energy]
            fig.add_trace(
                go.Scatter(
                    x=this_efficiency_df["X"],
                    y=this_efficiency_df["Efficiency"],
                    legendgroup=0,
                    legendgrouptitle=dict(
                        text=f"Energy (+- {red_energy_bin/2} MeV)", font=dict(size=13)
                    ),
                    mode="lines",
                    line_shape="spline",
                    name=f"{energy}",
                    #  error_y=dict(type='data', array=this_efficiency_df["Error"]),
                    line=dict(color=colors[jdx % len(colors)]),
                ),
                row=1,
                col=1,
            )

        for kdx, drift in enumerate(np.unique(this_energy_df["X"])):
            this_efficiency_df = this_energy_df[this_energy_df["X"] == drift]
            fig.add_trace(
                go.Scatter(
                    x=this_efficiency_df["Energy"],
                    y=this_efficiency_df["Efficiency"],
                    legendgroup=1,
                    legendgrouptitle=dict(
                        text=f"Drift (+- {params['DEFAULT_X_REGION']/2} cm)",
                        font=dict(size=13),
                    ),
                    mode="lines",
                    line_shape="spline",
                    name=f"{drift}",
                    #  error_y=dict(type='data', array=this_efficiency_df["Error"]),
                    line=dict(color=colors[kdx % len(colors)]),
                ),
                row=1,
                col=2,
            )
        for pdx, plane in enumerate(np.unique(this_plane_df["Plane"])):
            this_efficiency_df = this_plane_df[this_plane_df["Plane"] == plane]
            fig.add_trace(
                go.Scatter(
                    x=this_efficiency_df["Energy"],
                    y=this_efficiency_df["Efficiency"],
                    legendgroup=2,
                    legendgrouptitle=dict(text=f"Plane", font=dict(size=13)),
                    mode="lines",
                    line_shape="spline",
                    name=f"{info['OPFLASH_PLANES'][str(int(plane))]}",
                    #  error_y=dict(type='data', array=this_efficiency_df["Error"]),
                    line=dict(color=colors[pdx % len(colors)]),
                ),
                row=1,
                col=3,
            )
        # fig = px.line(this_efficiency_df, x="Energy", y="Efficiency", title="Matching Efficiency", facet_col="Type", line_dash="Drift", line_shape="hvh", color_discrete_sequence=colors, error_y="Error")
        fig.add_hline(100, line=dict(color="grey", dash="dash"))
        fig = format_coustom_plotly(
            fig,
            log=(False, False),
            title=f"Neutrino Matching Efficiency - {config} {name}",
            tickformat=(".0f", ".1s"),
            ranges=(None, [-1, 110]),
            matches=(None, None),
        )
        fig.update_xaxes(title_text="Neutrino Energy (MeV)", row=1, col=3)
        fig.update_xaxes(title_text="Neutrino Energy (MeV)", row=1, col=2)
        fig.update_xaxes(title_text="Drift Coordinate (cm)", row=1, col=1)
        fig.update_yaxes(title_text="TPC-PDS Matching Efficiency (%)")
        # fig.update_layout(yaxis_title="TPC-PDS Matching Efficiency (%)")

        save_figure(
            fig,
            save_path,
            config,
            name,
            filename=f"{matching_type}_Efficiency_Scan",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )

        for variable in ["Efficiency", "PE"]:
            fig = make_subplots(
                rows=1,
                cols=3,
                subplot_titles=(
                    "Coordinate X Scan",
                    "Coordinate Y Scan",
                    "Coordinate Z Scan",
                ),
            )
            # this__df = efficiency_df[
            #     (efficiency_df["Name"] == name)
            #     * (efficiency_df["Plot"] == "Energy Scan")
            #     * (efficiency_df["Type"] == matching_type)
            # ]
            for jdx, coord in enumerate(["X", "Y", "Z"]):
                for kdx, value in enumerate(np.unique(this_energy_df[coord])):
                    this_efficiency_df = this_energy_df[this_energy_df[coord] == value]
                    fig.add_trace(
                        go.Scatter(
                            x=this_efficiency_df["Energy"],
                            y=this_efficiency_df[variable],
                            legendgroup=jdx,
                            legendgrouptitle=dict(
                                text=f"{coord} (+- {params[f'DEFAULT_{coord}_REGION']/2} cm)",
                                font=dict(size=13),
                            ),
                            mode="lines",
                            line_shape="spline",
                            name=f"{value}",
                            #  error_y=dict(type='data', array=this_efficiency_df["Error"]),
                            line=dict(color=colors[kdx % len(colors)]),
                        ),
                        row=1,
                        col=1 + jdx,
                    )
            if variable == "Efficiency":
                fig.add_hline(100, line=dict(color="grey", dash="dash"))
            fig = format_coustom_plotly(
                fig,
                log=(False, False),
                title=f"Neutrino Matching Efficiency - {config} {name}",
                tickformat=(".0f", ".1s"),
                ranges=(None, [-1, 110] if variable == "Efficiency" else None),
                matches=(None, None),
            )
            fig.update_xaxes(title_text="Neutrino Energy (MeV)")
            fig.update_yaxes(
                title_text=(
                    "TPC-PDS Matching Efficiency (%)"
                    if variable == "Efficiency"
                    else "<Signal MatchedOpFlashPE> (PE)"
                )
            )

            save_figure(
                fig,
                save_path,
                config,
                name,
                filename=f"{matching_type}_{variable}_Coordinate_Scan",
                rm=user_input["rewrite"],
                debug=user_input["debug"],
            )

            fig = make_subplots(
                rows=1,
                cols=3,
                subplot_titles=(
                    "Coordinate X Scan",
                    "Coordinate Y Scan",
                    "Coordinate Z Scan",
                ),
            )
            for jdx, coord in enumerate(["X", "Y", "Z"]):
                this_coord_df = efficiency_df[
                    (efficiency_df["Name"] == name)
                    * (efficiency_df["Plot"] == f"{coord} Scan")
                    * (efficiency_df["Type"] == matching_type)
                ]
                for kdx, value in enumerate(np.unique(this_coord_df["Energy"])):
                    this_efficiency_df = this_coord_df[this_coord_df["Energy"] == value]
                    fig.add_trace(
                        go.Scatter(
                            x=this_efficiency_df[coord],
                            y=this_efficiency_df[variable],
                            legendgroup=jdx,
                            legendgrouptitle=dict(
                                text=f"Energy (+- {red_energy_bin/2:.0f} MeV)",
                                font=dict(size=13),
                            ),
                            showlegend=jdx == 0,
                            mode="lines",
                            line_shape="spline",
                            name=f"{value}",
                            #  error_y=dict(type='data', array=this_efficiency_df["Error"]),
                            line=dict(color=colors[kdx % len(colors)]),
                        ),
                        row=1,
                        col=1 + jdx,
                    )
            if variable == "Efficiency":
                fig.add_hline(100, line=dict(color="grey", dash="dash"))
            fig = format_coustom_plotly(
                fig,
                log=(False, False if variable == "Efficiency" else True),
                title=f"Neutrino Matching Efficiency - {config} {name}",
                tickformat=(".0f", ".2s"),
                ranges=(None, [-1, 110] if variable == "Efficiency" else None),
                matches=(None, None),
            )
            fig.update_xaxes(title_text="Coordinate X (cm)", row=1, col=1)
            fig.update_xaxes(title_text="Coordinate Y (cm)", row=1, col=2)
            fig.update_xaxes(title_text="Coordinate Z (cm)", row=1, col=3)
            fig.update_yaxes(
                title_text=(
                    "TPC-PDS Matching Efficiency (%)"
                    if variable == "Efficiency"
                    else "<Signal MatchedOpFlashPE> (PE)"
                )
            )

            save_figure(
                fig,
                save_path,
                config,
                name,
                filename=f"{matching_type}_{variable}_Energy_Scan",
                rm=user_input["rewrite"],
                debug=user_input["debug"],
            )
