import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *

save_path = f"{root}/images/preselection/clustering"
data_path = f"{root}/data/preselection/clustering/"

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
    "workflow": "CORRECTION",
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
    presets=["PRESELECTION"],
    signal = "marley" in args.name,
    debug=user_input["debug"],
)
rprint(output)

logy = False
for config in configs:
    info, params, output = get_param_dict(
        f"{root}/config/{config}/{config}", {}, output, debug=args.debug
    )
    for name in configs[config]:
        df_list = []
        idx = (
            (run["Truth"]["Geometry"] == info["GEOMETRY"])
            & (run["Truth"]["Version"] == info["VERSION"])
            & (run["Truth"]["Name"] == name)
        )
        h_ref, edges_ref = {}, {}

        for hit in nhits[:9]:
            this_reco_data, mask, output = compute_filtered_run(
                run,
                configs,
                params={("Reco", "NHits"): ("equal", hit)},
                debug=user_input["debug"],
            )
            this_reco_data = this_reco_data["Reco"]
            if len(this_reco_data["Charge"]) == 0:
                continue
            this_reco_data["ChargeFraction"] = (
                this_reco_data["Charge"] / this_reco_data["ElectronCharge"]
            )
            bin_width = 0.05
            x_axis = np.arange(0, 1 + 2 * bin_width, bin_width)
            x_centers = x_axis[:-1] + bin_width / 2

            y, bins = np.histogram(
                this_reco_data["Purity"],
                bins=x_axis,
            )
            purity_dict = {
                "Config": config,
                "Name": name,
                "#Hits": hit,
                "Values": 100 * x_centers,
                "Counts": y,
                "Density": y / (np.sum(y) * bin_width),
                "Variable": "Purity",
            }
            df_list.append(purity_dict)

            y, bins = np.histogram(
                this_reco_data["ChargeFraction"],
                bins=x_axis,
            )
            completeness_dict = {
                "Config": config,
                "Name": name,
                "#Hits": hit,
                "Values": 100 * x_centers,
                "Counts": y,
                "Density": y / (np.sum(y) * bin_width),
                "Variable": "Completeness",
            }
            df_list.append(completeness_dict)

        plot_df = pd.DataFrame(df_list)
        for df_label in ["Purity", "Completeness"]:
            this_plot_df = plot_df[plot_df["Variable"] == df_label]
            this_plot_df = explode(this_plot_df, ["Values", "Density", "Counts"])
            fig = px.line(
                this_plot_df,
                x="Values",
                y="Density",
                color="#Hits",
                line_shape="hvh",
                color_discrete_sequence=colors,
            )

            fig = format_coustom_plotly(
                fig,
                title=f"Cluster {df_label} - {config} {name}",
                tickformat=(".1f", None),
                log=(False, True),
                debug=user_input["debug"],
                legend_title="#Hits",
            )

            fig.update_layout(yaxis_title="Density")
            fig.update_layout(xaxis_title=f"{df_label} (%)")

            save_figure(
                fig,
                save_path,
                config,
                name,
                filename=f"Cluster_{df_label}",
                rm=user_input["rewrite"],
                debug=user_input["debug"],
            )

        save_df(
            plot_df,
            data_path,
            config,
            name,
            filename=f"Clustering_Efficiency",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )
