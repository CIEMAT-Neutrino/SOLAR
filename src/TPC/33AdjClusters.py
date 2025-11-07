import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *

save_path = f"{root}/images/TPC/adjcluster/"
data_path = f"{root}/data/TPC/adjcluster/"

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

user_input = {"workflow": "CALIBRATION", "rewrite": args.rewrite, "debug": args.debug}

run, output = load_multi(
    configs, preset=user_input["workflow"], debug=user_input["debug"]
)

run = compute_reco_workflow(
    run, configs, workflow=user_input["workflow"], debug=user_input["debug"]
)

for config in configs:
    branches = []
    info, params, output = get_param_dict(
        f"{root}/config/{config}/{config}", {}, output, debug=user_input["debug"]
    )
    for limit in info["CLUSTER_RADIUS"]:
        branches.append(f"TotalAdjClRatio{limit}")
        branches.append(f"TotalAdjClSameGenNum{limit}")
        branches.append(f"TotalAdjClExternalBkgNum{limit}")
        branches.append(f"TotalAdjClIntrinsicBkgNum{limit}")
        branches.append(f"TotalAdjClSameGenCharge{limit}")
        branches.append(f"TotalAdjClBkgCharge{limit}")
    branches.append(f"SignalParticleX")

    reco_df = npy2df(run, "Reco", branches, debug=False)
    for name in configs[config]:
        fig = make_subplots(rows=1, cols=1)
        table_list = []
        for (jdx, limit), (idx, (label, variable)) in product(
            enumerate(info["CLUSTER_RADIUS"]),
            enumerate(
                zip(
                    ["Signal", "Intrinsic Bkg.", "External Bkg."],
                    [
                        "TotalAdjClSameGenNum",
                        "TotalAdjClIntrinsicBkgNum",
                        "TotalAdjClExternalBkgNum",
                    ],
                )
            ),
        ):
            per_99 = np.percentile(reco_df[f"{variable}{limit}"], 99)
            hist, bins = np.histogram(
                reco_df[f"{variable}{limit}"],
                bins=(
                    np.arange(0, np.max(reco_df[f"{variable}{limit}"]) + 1, 1)
                    if variable.endswith("Num")
                    else np.arange(1.5, per_99, 100)
                ),
            )
            hist = hist / np.sum(hist)
            fig.add_trace(
                go.Scatter(
                    x=bins,
                    y=hist,
                    mode="lines+markers",
                    line_shape="spline",
                    marker_symbol=["circle", "square", "diamond"][idx],
                    line_dash=["dash", "dot", "dashdot", "longdash", "solid"][jdx],
                    showlegend=True if idx == 0 else False,
                    line=dict(color=default[idx], width=2),
                    name=f"{limit} cm",
                    legendgroup="Radial Distance",
                    legendgrouptitle=dict(text="Radial Distance"),
                )
            )
            if jdx == len(info["CLUSTER_RADIUS"]) - 1:
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="markers",
                        marker=dict(
                            symbol=["circle", "square", "diamond"][idx],
                            color=default[idx],
                        ),
                        showlegend=True,
                        legendgroup="Particle Type",
                        legendgrouptitle=dict(text="Particle Type"),
                        name=label,
                    )
                )
            table_list.append(
                {
                    "Type": label,
                    "Distance": f"{limit}",
                    "Mean": np.mean(reco_df[f"{variable}{limit}"]),
                    "Error": np.std(reco_df[f"{variable}{limit}"])
                    / np.sqrt(len(reco_df[f"{variable}{limit}"])),
                    "STD": np.std(reco_df[f"{variable}{limit}"]),
                }
            )
        fig = format_coustom_plotly(
            fig,
            title=f"AdjClusters - {config}",
            log=(False, True),
            tickformat=(None, ".0e"),
            ranges=(None, (-4, 0.5)),
        )
        fig.update_xaxes(title_text="Number of Adjacent Clusters")
        fig.update_yaxes(title_text="Fraction of Events")
        save_figure(
            fig,
            save_path,
            config,
            name,
            filename="Signal_AdjClusters",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )
        # Format the df so that the column "Radius (cm)" becomes the index and the columns are multi-indexed by "Type" and the statistic
        df = pd.DataFrame(table_list)

        save_df(
            df,
            data_path,
            config,
            name,
            decimals=3,
            filename="Signal_AdjClusters",
            rm=user_input["rewrite"],
            filetype="txt",
            debug=user_input["debug"],
        )

        explode_branches = [
            "AdjClGen",
            "AdjClR",
            "AdjClCharge",
            "AdjClCorrectedCharge",
            "AdjClEnergy",
            "AdjClMainE",
            "AdjClPur",
        ]
        keep_branches = ["SignalParticleK", "TotalAdjClCharge", "NHits", "Time"]
        reco_df = npy2df(
            run, "Reco", explode_branches + keep_branches, debug=user_input["debug"]
        )
        reco_adjcl_df = explode(
            reco_df,
            explode_branches,
            keep_branches,
        )

        df_list = []
        for variable in explode_branches:
            fig = make_subplots(rows=1, cols=1)
            this_df = reco_df[reco_df["TotalAdjClCharge"] > 0]
            this_adjcl_df = reco_adjcl_df[(reco_adjcl_df["AdjClCharge"] > 0)]

            this_df_marley = this_adjcl_df[(this_adjcl_df["AdjClPur"] > 0)]
            this_df_externals = this_adjcl_df[
                (this_adjcl_df["AdjClPur"] == 0)
                * (this_adjcl_df["AdjClGen"].isin(info["EXTERNAL_BACKGROUNDS"]))
            ]
            this_df_intrinsic = this_adjcl_df[
                (this_adjcl_df["AdjClPur"] == 0)
                * (~this_adjcl_df["AdjClGen"].isin(info["EXTERNAL_BACKGROUNDS"]))
            ]

            bins = np.arange(0, np.max(this_adjcl_df[variable]), 1)
            if variable == "AdjClPur":
                bins = np.linspace(0, 1, 100)

            if variable in ["AdjClEnergy", "AdjClMainE"]:
                bins = np.arange(0.1, 10, 0.05)

            if variable == "AdjClNHits":
                bins = np.arange(0.5, np.max(this_adjcl_df[variable]), 1)

            hist_marley, edges = np.histogram(
                this_df_marley[variable], bins=bins, density=False
            )
            hist_marley = hist_marley / len(this_df)
            hist_external_bkg, edges = np.histogram(
                this_df_externals[variable], bins=bins, density=False
            )
            hist_external_bkg = hist_external_bkg / len(this_df)
            hist_intrinsic_bkg, edges = np.histogram(
                this_df_intrinsic[variable], bins=bins, density=False
            )
            hist_intrinsic_bkg = hist_intrinsic_bkg / len(this_df)

            edge_centers = (edges[1:] + edges[:-1]) / 2

            plot_list = []
            plot_list.append(
                {
                    "Geometry": info["GEOMETRY"],
                    "Config": config,
                    "Name": name,
                    "Variable": variable,
                    "Values": edge_centers,
                    "Counts": hist_marley,
                    "Density": hist_marley
                    / (np.sum(hist_marley) * (edges[1] - edges[0])),
                    "Signal": "Signal",
                }
            )
            plot_list.append(
                {
                    "Geometry": info["GEOMETRY"],
                    "Config": config,
                    "Name": name,
                    "Variable": variable,
                    "Values": edge_centers,
                    "Counts": hist_external_bkg,
                    "Density": hist_external_bkg
                    / (np.sum(hist_external_bkg) * (edges[1] - edges[0])),
                    "Signal": "External",
                }
            )
            plot_list.append(
                {
                    "Geometry": info["GEOMETRY"],
                    "Config": config,
                    "Name": name,
                    "Variable": variable,
                    "Values": edge_centers,
                    "Counts": hist_intrinsic_bkg,
                    "Density": hist_intrinsic_bkg
                    / (np.sum(hist_intrinsic_bkg) * (edges[1] - edges[0])),
                    "Signal": "Intrinsic",
                }
            )

            plot_df = pd.DataFrame(plot_list)
            this_plot_df = plot_df.explode(["Values", "Counts", "Density"])
            df_list += plot_list
            for idx, signal in enumerate(["Signal", "Intrinsic", "External"]):
                fig.add_trace(
                    go.Scatter(
                        x=this_plot_df[this_plot_df["Signal"] == signal]["Values"],
                        y=this_plot_df[this_plot_df["Signal"] == signal]["Counts"],
                        mode="lines",
                        line_shape="hvh",
                        name=signal,
                        line=dict(color=default[idx], width=2),
                    ),
                    row=1,
                    col=1,
                )

            fig = format_coustom_plotly(
                fig,
                legend_title="Particle Type",
                title=f"{variable} Distribution - {config}",
                log=(
                    (
                        True
                        if variable
                        in [
                            "AdjClCharge",
                            "AdjClCorrectedCharge",
                            "AdjClEnergy",
                            "AdjClMainE",
                        ]
                        else False
                    ),
                    True,
                ),
                ranges=(
                    (
                        (1, None)
                        if variable in ["AdjClCharge", "AdjClCorrectedCharge"]
                        else None
                    ),
                    (-5, 0),
                ),
                tickformat=(None, ".0e"),
            )
            fig.update_xaxes(title_text=variable)
            fig.update_yaxes(title_text="#Adjacent Clusters per Events", row=1, col=1)

            save_figure(
                fig,
                save_path,
                config,
                name,
                filename=f"{variable}_Distribution",
                rm=user_input["rewrite"],
                debug=user_input["debug"],
            )

        save_df(
            pd.DataFrame(df_list),
            data_path,
            config,
            name,
            filename=f"AdjCluster_Distributions",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )
