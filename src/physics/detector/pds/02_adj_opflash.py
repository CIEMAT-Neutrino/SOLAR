import os
import sys
import warnings

warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
warnings.filterwarnings("ignore", "Degrees of freedom <= 0", RuntimeWarning)

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *

save_path = f"{root}/output/images/PDS/adjopflash/"
data_path = f"{root}/output/data/PDS/adjopflash/"

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

def _safe_mean(arr):
    return np.mean(arr) if len(arr) > 0 else np.nan


run, output = load_multi(
    configs, preset=user_input["workflow"], debug=user_input["debug"]
)

run = compute_reco_workflow(
    run, configs, workflow=user_input["workflow"], debug=user_input["debug"]
)

for config in configs:
    info, params, output = get_param_dict(
        f"{root}/config/{config}/{config}", {}, output, debug=user_input["debug"]
    )

    for name in configs[config]:
        # Avoid mutating info["OPFLASH_PLANES"] in place
        plane_names = {**info["OPFLASH_PLANES"], None: "Total"}
        plane_ids = list(plane_names.keys())

        # Pre-compute per-bin means for coordinate scans via vectorized binning.
        # Replaces N_drift × compute_filtered_run calls with one np.digitize +
        # one groupby.mean() per coordinate — critical for large samples.
        reco = run["Reco"]
        all_y_variables = [
            "TotalAdjOpFlashSameGenNum", "TotalAdjOpFlashBkgNum",
            "TotalAdjOpFlashSameGenPE", "TotalAdjOpFlashBkgPE",
        ]
        scan_cols = [
            col for col in
            [f"{v}Radius{l}" for v in all_y_variables for l in info["OPFLASH_RADIUS"]]
            + [f"{v}Plane{p}" for v in all_y_variables for p in plane_ids if p is not None]
            + all_y_variables
            if col in reco
        ]

        coord_data = {}
        for x_axis_title, x_variable in zip(
            ("Drift Distance (cm)", "Coordinate Y (cm)", "Coordinate Z (cm)"),
            ("SignalParticleX", "SignalParticleY", "SignalParticleZ"),
        ):
            coord = x_variable[-1]
            bin_half = params[f"DEFAULT_{coord}_BIN"] / 2
            x_axis = np.arange(
                info[f"DETECTOR_MIN_{coord}"] + bin_half,
                info[f"DETECTOR_MAX_{coord}"],
                bin_half,
            )
            bin_edges = np.concatenate([[x_axis[0] - bin_half], x_axis + bin_half])
            raw_idx = np.digitize(np.asarray(reco[x_variable], dtype=float), bin_edges) - 1
            valid = (raw_idx >= 0) & (raw_idx < len(x_axis))
            df_scan = pd.DataFrame({col: np.asarray(reco[col], dtype=float) for col in scan_cols})
            df_scan["_bin"] = np.where(valid, raw_idx, -1)
            mean_by_bin = (
                df_scan[df_scan["_bin"] >= 0]
                .groupby("_bin")[scan_cols]
                .mean()
                .reindex(range(len(x_axis)))
            )
            coord_data[x_variable] = (x_axis_title, x_axis, mean_by_bin)

        for y_label, y_axis_title, y_variables in zip(
            ["Num", "PE"],
            ["Number of Adj. OpFlashes", "PE of Adj. OpFlashes"],
            (
                ["TotalAdjOpFlashSameGenNum", "TotalAdjOpFlashBkgNum"],
                ["TotalAdjOpFlashSameGenPE", "TotalAdjOpFlashBkgPE"],
            ),
        ):
            # --- Radial scan ---
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
                arr_r = run["Reco"][f"{variable}Radius{limit}"]
                if len(arr_r) == 0:
                    continue
                per_99 = np.percentile(arr_r, 99)
                hist, bins = np.histogram(
                    arr_r,
                    bins=(
                        np.arange(0, np.max(arr_r) + 1, 1)
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
                        showlegend=idx == 0,
                        line=dict(color=colors[jdx], width=2),
                        name=f"{limit}",
                    ),
                    row=1,
                    col=1 + idx,
                )
                table_list.append(
                    {
                        "Geometry": info["GEOMETRY"],
                        "Config": config,
                        "Name": name,
                        "Type": label,
                        "Radius": f"{limit}",
                        "Mean": np.mean(arr_r),
                        "MeanError": np.std(arr_r),
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
            save_figure(
                fig,
                save_path,
                config,
                name,
                filename=f"Signal_AdjOpFlash{y_label}_RadialScan",
                rm=user_input["rewrite"],
                debug=user_input["debug"],
            )
            save_df(
                pd.DataFrame(table_list),
                data_path,
                config,
                name,
                filename=f"Signal_AdjOpFlash{y_label}_RadialScan",
                rm=user_input["rewrite"],
                debug=user_input["debug"],
            )

            # --- Plane scan ---
            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=("Signal", "Background"),
            )
            table_list = []
            for (jdx, plane), (idx, (label, variable)) in product(
                enumerate(plane_ids),
                enumerate(
                    zip(
                        ["Signal", "Background"],
                        y_variables,
                    ),
                ),
            ):
                full_arr = run["Reco"][f"{variable}"]
                if len(full_arr) == 0 or np.percentile(full_arr, 99) < 1.5:
                    print(f"{variable} is empty for {config} - {name}, skipping...")
                    continue
                arr = (
                    full_arr
                    if plane is None
                    else run["Reco"][f"{variable}Plane{plane}"]
                )
                if len(arr) == 0:
                    continue
                per_99 = np.percentile(arr, 99)
                hist, bins = np.histogram(
                    arr,
                    bins=(
                        np.arange(0, np.max(arr) + 1, 1)
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
                        showlegend=idx == 0,
                        line=dict(color=colors[jdx], width=2),
                        name=f"{plane_names[plane]}",
                    ),
                    row=1,
                    col=1 + idx,
                )
                table_list.append(
                    {
                        "Geometry": info["GEOMETRY"],
                        "Config": config,
                        "Name": name,
                        "Type": label,
                        "Plane": plane,
                        "Mean": np.mean(arr),
                        "MeanError": np.std(arr),
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
            fig.update_xaxes(
                title_text=(
                    "Number of Adj. OpFlashes"
                    if y_label == "Num"
                    else "PE of Adj. OpFlashes"
                )
            )
            fig.update_yaxes(title_text="Fraction of Events", row=1, col=1)
            save_figure(
                fig,
                save_path,
                config,
                name,
                filename=f"Signal_AdjOpFlash{y_label}_PlaneScan",
                rm=user_input["rewrite"],
                debug=user_input["debug"],
            )
            save_df(
                pd.DataFrame(table_list),
                data_path,
                config,
                name,
                filename=f"Signal_AdjOpFlash{y_label}_PlaneScan",
                rm=user_input["rewrite"],
                debug=user_input["debug"],
            )

            # --- Coordinate scans (vectorized: direct column lookup from mean_by_bin) ---
            for x_variable, (x_axis_title, x_axis, mean_by_bin) in coord_data.items():
                coord = x_variable[-1]
                n_bins = len(x_axis)

                # Radius variant
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
                    col = f"{variable}Radius{limit}"
                    values = (
                        mean_by_bin[col].tolist()
                        if col in mean_by_bin.columns
                        else [np.nan] * n_bins
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=x_axis,
                            y=values,
                            mode="lines",
                            line_shape="hvh",
                            showlegend=idx == 0,
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
                save_figure(
                    fig,
                    save_path,
                    config,
                    name,
                    filename=f"Signal_AdjOpFlash{y_label}_{coord}Scan_Radius",
                    rm=user_input["rewrite"],
                    debug=user_input["debug"],
                )

                # Plane variant
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
                    col = variable if plane is None else f"{variable}Plane{plane}"
                    values = (
                        mean_by_bin[col].tolist()
                        if col in mean_by_bin.columns
                        else [np.nan] * n_bins
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=x_axis,
                            y=values,
                            mode="lines",
                            line_shape="hvh",
                            showlegend=idx == 0,
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
                save_figure(
                    fig,
                    save_path,
                    config,
                    name,
                    filename=f"Signal_AdjOpFlash{y_label}_{coord}Scan_Plane",
                    rm=user_input["rewrite"],
                    debug=user_input["debug"],
                )
