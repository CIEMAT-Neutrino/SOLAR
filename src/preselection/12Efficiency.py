import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *

save_path = f"{root}/images/preselection/efficiency"
data_path = f"{root}/data/preselection/efficiency"

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
    signal="marley" in args.name,
    debug=user_input["debug"],
)
rprint(output)

logy = False
for config in configs:
    info, params, output = get_param_dict(
        f"{root}/config/{config}/{config}", {}, output, debug=args.debug
    )
    for name in configs[config]:
        eff_list = []
        idx = (
            (run["Truth"]["Geometry"] == info["GEOMETRY"])
            & (run["Truth"]["Version"] == info["VERSION"])
            & (run["Truth"]["Name"] == name)
        )
        h_ref, edges_ref = {}, {}
        for tree, var in product(
            ["Truth", "Reco"],
            [
                "ElectronK",
                "SignalParticleK",
                "SignalParticleX",
                "SignalParticleY",
                "SignalParticleZ",
            ],
        ):
            if var not in run[tree].keys():
                continue

            min_val = np.min(run[tree][f"{var}"])
            max_val = np.max(run[tree][f"{var}"])
            bins = (
                reco_energy_edges
                if var == "K"
                else np.linspace(min_val, max_val, 50, endpoint=True)
            )
            h, edges = np.histogram(
                run[tree][f"{var}"],
                bins=bins,
            )
            h_ref[var] = h
            edges_ref[var] = edges

            for hit in nhits[:9]:
                if tree == "Truth":
                    this_data = run[tree][f"{var}"][
                        np.where(run[tree]["HitCount"] >= hit)
                    ]
                    h, edges = np.histogram(this_data, bins=bins)

                elif tree == "Reco":
                    this_data = run[tree][f"{var}"][
                        np.where(
                            (run[tree][f"TrueMain"]) * (run[tree]["NHits"] >= hit + 1)
                        )
                    ]
                    h, edges = np.histogram(this_data, bins=bins)

                bin_centers = 0.5 * (edges[1:] + edges[:-1])
                efficiency = 100 * h / h_ref[var]
                efficiency_error = (
                    100 * np.sqrt(h * (1 - h / h_ref[var])) / h_ref[var]
                    if np.any(h_ref[var] > 0)
                    else np.zeros_like(h)
                )

                efficiency[np.isnan(efficiency)] = 0
                eff_list.append(
                    {
                        "Config": config,
                        "Name": name,
                        "Tree": tree,
                        "Variable": f"{var}",
                        "Values": bin_centers,
                        "Efficiency": efficiency,
                        "Error": efficiency_error,
                        "#Hits": hit,
                    }
                )

        eff_df = pd.DataFrame(eff_list)

        this_eff_df = explode(
            eff_df,
            ["Values", "Efficiency"],
            ["Tree", "#Hits", "Variable"],
            debug=user_input["debug"],
        )
        if this_eff_df["Efficiency"].max() < 0.1:
            logy = True

        for tree, var in product(
            ["Truth", "Reco"],
            [
                "ElectronK",
                "SignalParticleK",
                "SignalParticleX",
                "SignalParticleY",
                "SignalParticleZ",
            ],
        ):
            fig = px.line(
                this_eff_df[
                    (this_eff_df["Variable"] == f"{var}")
                    & (this_eff_df["Tree"] == tree)
                ],
                x="Values",
                y="Efficiency",
                color="#Hits",
                color_discrete_sequence=px.colors.qualitative.Prism,
                line_shape="hvh",
                render_mode="png",
            )

            if name.split("_")[0] in user_input["label"]:
                label = user_input["label"][name.split("_")[0]]
            else:
                label = user_input["label"][None]

            fig = format_coustom_plotly(
                fig,
                title=f"Detection Efficiency - {config} {name}",
                log=(False, logy),
                ranges=(None, [0, 110]),
                legend_title="min #Hits",
                debug=user_input["debug"],
            )
            x_axis_title = f"True {label}"

            if var[-1] == "K":
                x_axis_title += " Energy (MeV)"
            else:
                x_axis_title += f" {var} Coordinate (cm)"

            fig.update_xaxes(title_text=x_axis_title)
            fig.update_yaxes(title_text="Preselection Efficiency (%)")

            fig.add_hline(y=100, line_width=1, line_dash="dash", line_color="black")
            save_figure(
                fig,
                save_path,
                config,
                name,
                filename=f"{tree}_Preselection_{var}_Efficiency",
                rm=user_input["rewrite"],
                debug=user_input["debug"],
            )

        save_df(
            eff_df,
            data_path,
            config,
            name,
            filename=f"Preselection_Efficiency",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )
