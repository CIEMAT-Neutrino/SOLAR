import sys

sys.path.insert(0, "../../")

from lib import *

save_path = f"{root}/images/PDS/adjopflash/"
data_path = f"{root}/data/PDS/adjopflash/"

for path in [save_path, data_path]:
    if not os.path.exists(path):
        os.makedirs(path)

# Define flags for the analysis config and name with the python parser
parser = argparse.ArgumentParser(
    description="Plot the AdjOpFlash distributions of the signal"
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

user_input = {"workflow": "ADJFLASH", "rewrite": args.rewrite, "debug": args.debug}

run, output = load_multi(
    configs, preset=user_input["workflow"], debug=user_input["debug"]
)

run = compute_reco_workflow(
    run, configs, workflow=user_input["workflow"], debug=user_input["debug"]
)

reco_df = npy2df(run, "Reco", debug=user_input["debug"])

for config in configs:
    info, params, output = get_param_dict(
        f"{root}/config/{config}/{config}", {}, output, debug=user_input["debug"]
    )
    for name, (y_label, y_axis_title, y_variables) in product(
        configs[config],
        zip(
            ["Num", "PE"],
            ["Number of Adj. OpFlashes", "PE of Adj. OpFlashes"],
            (
                ["TotalAdjOpFlashSameGenNum", "TotalAdjOpFlashBkgNum"],
                ["TotalAdjOpFlashSameGenPE", "TotalAdjOpFlashBkgPE"],
            ),
        ),
    ):
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Signal", "Background"),
        )

        table_list = []
        for (jdx, limit), (idx, (label, variable)) in product(
            enumerate(info["OPFLASH_RADIUS"]),
            enumerate(
                zip(
                    ["Signal", "Background"],
                    y_variables,
                ),
            ),
        ):
            per_99 = np.percentile(reco_df[f"{variable}Radius{limit}"], 99)
            hist, bins = np.histogram(
                reco_df[f"{variable}Radius{limit}"],
                bins=(
                    np.arange(0, np.max(reco_df[f"{variable}Radius{limit}"]) + 1, 1)
                    if y_label == "Num"
                    else np.arange(1.5, per_99, 100)
                ),
            )
            hist = hist / np.sum(hist)
            fig.add_trace(
                go.Scatter(
                    x=bins,
                    y=hist,
                    mode="lines",
                    line_shape="hvh",
                    showlegend=True if idx == 0 else False,
                    line=dict(color=colors[jdx], width=2),
                    name=f"{limit}",
                ),
                row=1,
                col=1 + idx,
            )
            table_list.append(
                {
                    "Type": label,
                    "Radius (cm)": f"{limit}",
                    "Mean": np.mean(reco_df[f"{variable}Radius{limit}"]),
                    "STD": np.std(reco_df[f"{variable}Radius{limit}"]),
                }
            )
        fig = format_coustom_plotly(
            fig,
            matches=(None, "y"),
            tickformat=(".0f", ".1s"),
            legend_title="Radius (cm)",
            title=f"Radial AdjOpFlashNum (Signal vs Background) - {config}",
            log=(False, True),
        )
        fig.update_xaxes(title_text=y_axis_title)
        fig.update_yaxes(title_text="Fraction of Events", row=1, col=1)
        # fig.show()
        save_figure(
            fig,
            save_path,
            config,
            name,
            filename=f"Signal_AdjOpFlash{y_label}_RadialScan",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )

        # display(pd.DataFrame(table_list).set_index("Radius (cm)").T)
        save_df(
            pd.DataFrame(table_list),
            data_path,
            config,
            name,
            filename=f"Signal_AdjOpFlash{y_label}_RadialScan",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Signal", "Background"),
        )
        table_list = []
        plane_names = info["OPFLASH_PLANES"]
        plane_names[None] = "Total"
        plane_ids = list(info["OPFLASH_PLANES"].keys())
        for (jdx, plane), (idx, (label, variable)) in product(
            enumerate(plane_ids),
            enumerate(
                zip(
                    ["Signal", "Background"],
                    y_variables,
                ),
            ),
        ):
            if np.percentile(reco_df[f"{variable}"], 99) < 1.5:
                print(f"{variable} is empty for {config} - {name}, skipping...")
                continue

            if plane is None:
                per_99 = np.percentile(reco_df[f"{variable}"], 99)
                hist, bins = np.histogram(
                    reco_df[f"{variable}"],
                    bins=(
                        np.arange(0, np.max(reco_df[f"{variable}"]) + 1, 1)
                        if y_label == "Num"
                        else np.arange(1.5, per_99 + 100, 100)
                    ),
                )
            else:
                per_99 = np.percentile(reco_df[f"{variable}Plane{plane}"], 99)
                hist, bins = np.histogram(
                    reco_df[f"{variable}Plane{plane}"],
                    bins=(
                        np.arange(0, np.max(reco_df[f"{variable}Plane{plane}"]) + 1, 1)
                        if y_label == "Num"
                        else np.arange(1.5, per_99 + 100, 100)
                    ),
                )
            hist = hist / np.sum(hist)
            fig.add_trace(
                go.Scatter(
                    x=bins,
                    y=hist,
                    mode="lines",
                    line_shape="hvh",
                    showlegend=True if idx == 0 else False,
                    line=dict(color=colors[jdx], width=2),
                    name=f"{plane_names[plane]}",
                ),
                row=1,
                col=1 + idx,
            )
            table_list.append(
                {
                    "Type": label,
                    "Plane": plane,
                    "Mean": (
                        np.mean(reco_df[f"{variable}Plane{plane}"])
                        if plane is not None
                        else np.mean(reco_df[f"{variable}"])
                    ),
                    "STD": (
                        np.std(reco_df[f"{variable}Plane{plane}"])
                        if plane is not None
                        else np.std(reco_df[f"{variable}"])
                    ),
                }
            )
        fig = format_coustom_plotly(
            fig,
            matches=(None, "y"),
            tickformat=(".0f", ".1s"),
            legend_title="Plane",
            title=f"Radial AdjOpFlashNum (Signal vs Background) - {config}",
            log=(False, True),
        )
        fig.update_xaxes(title_text="Number of Adj. OpFlashes" if y_label == "Num" else "PE of Adj. OpFlashes")
        fig.update_yaxes(title_text="Fraction of Events", row=1, col=1)
        # fig.show()
        save_figure(
            fig,
            save_path,
            config,
            name,
            filename=f"Signal_AdjOpFlash{y_label}_PlaneScan",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )

        # display(pd.DataFrame(table_list).set_index("Plane (cm)").T)
        save_df(
            pd.DataFrame(table_list),
            data_path,
            config,
            name,
            filename=f"Signal_AdjOpFlash{y_label}_PlaneScan",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )

        for x_axis_title, x_variable in zip(
            ("Drift Distance (cm)", "Coordinate Y (cm)", "Coordinate Z (cm)"),
            ("SignalParticleX", "SignalParticleY", "SignalParticleZ"),
        ):
            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=("Signal", "Background"),
            )

            for (jdx, limit), (idx, (label, variable)) in product(
                enumerate(info["OPFLASH_RADIUS"]),
                enumerate(
                    zip(
                        ["Signal", "Background"],
                        y_variables,
                    )
                ),
            ):
                values = []
                stds = []
                coord = x_variable[-1]
                x_axis = np.arange(
                    info[f"DETECTOR_MIN_{coord}"] + params[f"DEFAULT_{coord}_BIN"] / 2,
                    info[f"DETECTOR_MAX_{coord}"],
                    params[f"DEFAULT_{coord}_BIN"] / 2,
                )
                for drift in x_axis:
                    this_reco_df = reco_df[
                        (
                            reco_df[x_variable]
                            > drift - params[f"DEFAULT_{coord}_BIN"] / 2
                        )
                        & (
                            reco_df[x_variable]
                            < drift + params[f"DEFAULT_{coord}_BIN"] / 2
                        )
                    ]
                    values.append(np.mean(this_reco_df[f"{variable}Radius{limit}"]))
                    stds.append(np.std(this_reco_df[f"{variable}Radius{limit}"]))

                fig.add_trace(
                    go.Scatter(
                        x=x_axis,
                        y=values,
                        # error_y = dict(type='data', array=stds, visible=True),
                        mode="lines",
                        line_shape="hvh",
                        showlegend=True if idx == 0 else False,
                        line=dict(color=colors[jdx], width=2),
                        name=f"{limit}",
                    ),
                    row=1,
                    col=1 + idx,
                )

            fig = format_coustom_plotly(
                fig,
                log=(False, True),
                tickformat=(None, ".1s"),
                legend_title="Radius (cm)",
                title=f"AdjOpFlash{y_label} (Signal vs Background) - {config}",
            )
            fig.update_xaxes(title_text=x_axis_title)
            fig.update_yaxes(title_text=y_axis_title, row=1, col=1)
            # fig.show()
            save_figure(
                fig,
                save_path,
                config,
                name,
                filename=f"Signal_AdjOpFlash{y_label}_{coord}Scan_Radius",
                rm=user_input["rewrite"],
                debug=user_input["debug"],
            )

            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=("Signal", "Background"),
            )

            for (jdx, plane), (idx, (label, variable)) in product(
                enumerate(plane_ids),
                enumerate(
                    zip(
                        ["Signal", "Background"],
                        y_variables,
                    )
                ),
            ):
                values = []
                stds = []
                coord = x_variable[-1]
                x_axis = np.arange(
                    info[f"DETECTOR_MIN_{coord}"] + params[f"DEFAULT_{coord}_BIN"] / 2,
                    info[f"DETECTOR_MAX_{coord}"],
                    params[f"DEFAULT_{coord}_BIN"] / 2,
                )
                for drift in x_axis:
                    this_reco_df = reco_df[
                        (
                            reco_df[x_variable]
                            > drift - params[f"DEFAULT_{coord}_BIN"] / 2
                        )
                        & (
                            reco_df[x_variable]
                            < drift + params[f"DEFAULT_{coord}_BIN"] / 2
                        )
                    ]
                    if plane is None:
                        values.append(np.mean(this_reco_df[f"{variable}"]))
                        stds.append(np.std(this_reco_df[f"{variable}"]))
                    else:
                        values.append(np.mean(this_reco_df[f"{variable}Plane{plane}"]))
                        stds.append(np.std(this_reco_df[f"{variable}Plane{plane}"]))

                fig.add_trace(
                    go.Scatter(
                        x=x_axis,
                        y=values,
                        # error_y = dict(type='data', array=stds, visible=True),
                        mode="lines",
                        line_shape="hvh",
                        showlegend=True if idx == 0 else False,
                        line=dict(color=colors[jdx], width=2),
                        name=f"{plane_names[plane]}",
                    ),
                    row=1,
                    col=1 + idx,
                )

            fig = format_coustom_plotly(
                fig,
                log=(False, True),
                tickformat=(None, ".1s"),
                legend_title="Plane",
                title=f"AdjOpFlash{y_label} (Signal vs Background) - {config}",
            )
            fig.update_xaxes(title_text=x_axis_title)
            fig.update_yaxes(title_text=y_axis_title, row=1, col=1)
            # fig.show()
            save_figure(
                fig,
                save_path,
                config,
                name,
                filename=f"Signal_AdjOpFlash{y_label}_{coord}Scan_Plane",
                rm=user_input["rewrite"],
                debug=user_input["debug"],
            )
