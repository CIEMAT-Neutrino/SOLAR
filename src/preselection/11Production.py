import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *

save_path = f"{root}/images/preselection/production"
data_path = f"{root}/data/preselection/production"

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
    params={("Truth", "SignalParticleK"): ("bigger", 0), ("Reco", "SignalParticleK"): ("bigger", 0), ("Reco", "TrueMain"): ("equals", True)},
    signal = "marley" in args.name,
    debug=user_input["debug"],
)
rprint(output)

for config in configs:
    info, params, output = get_param_dict(
        f"{root}/config/{config}/{config}", {}, output, debug=args.debug
    )
    eff_list = []
    prod_list = []
    for name, (reference, hits) in product(configs[config], zip(["Truth", "Reco"], ["HitCount", "NHits"])):
        idx = (
            (run[reference]["Geometry"] == info["GEOMETRY"])
            & (run[reference]["Version"] == info["VERSION"])
            & (run[reference]["Name"] == name)
        )
        h_ref, edges_ref = {}, {}
        for var in ["K", "X", "Y", "Z"]:
            min_val = np.min(run[reference][f"SignalParticle{var}"])
            max_val = np.max(run[reference][f"SignalParticle{var}"])
            prod_list.append(
                {
                    "Config": config,
                    "Name": name,
                    "Variable": f"SignalParticle{var}",
                    "Limits": [min_val, max_val],
                    "Counts": len(run[reference][f"SignalParticle{var}"]),
                    "Reference": reference,
                }
            )
            bins = reco_energy_edges if var == "K" else np.linspace(
                np.min(run[reference][f"SignalParticle{var}"]),
                np.max(run[reference][f"SignalParticle{var}"]),
                50, endpoint=True
            )

            h, edges = np.histogram(
                run[reference][f"SignalParticle{var}"],
                bins=bins,
            )

            h_ref[var] = h
            edges_ref[var] = edges

            for hit in nhits[:9]:
                this_data = run[reference][f"SignalParticle{var}"][
                    np.where(run[reference][hits] >= hit-1)
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
                        "Variable": f"SignalParticle{var}",
                        "Values": bin_centers,
                        "Efficiency": efficiency,
                        "EfficiencyError": efficiency_error,
                        "#Hits": hit-1,
                        "Reference": reference,
                    }
                )

    eff_df = pd.DataFrame(eff_list)
    lim_df = pd.DataFrame(prod_list)
    this_eff_df = explode(
        eff_df,
        ["Values", "Efficiency", "EfficiencyError"],
        ["#Hits", "Variable", "Reference"],
        debug=user_input["debug"],
    )

    for var, reference in product(["K", "X", "Y", "Z"], ["Truth", "Reco"]):
        fig = px.line(
            this_eff_df[(this_eff_df["Variable"] == f"SignalParticle{var}")*(this_eff_df["Reference"] == reference)],
            x="Values",
            y="Efficiency",
            color="#Hits",
            color_discrete_sequence=px.colors.qualitative.Prism,
            line_shape="hvh",
            render_mode="png",
        )
        # Add vertical lines for the production limits
        this_limit_df = lim_df[
            (lim_df["Variable"] == f"SignalParticle{var}") * (lim_df["Reference"] == reference)
        ]

        for i in [0, 1]:
            fig.add_vline(
                x=this_limit_df["Limits"].values[0][i],
                line_width=1,
                line_dash="dash",
                line_color="red",
            )

        if name.split("_")[0] in user_input["label"]:
            label = user_input["label"][name.split("_")[0]]
        else:
            label = user_input["label"][None]

        fig = format_coustom_plotly(
            fig,
            title=f"Production - {config} {name}",
            ranges=(None, [0, 110]),
            legend_title="min #Hits",
            debug=user_input["debug"],
        )
        x_axis_title = f"True {label}"
        if var == "K":
            x_axis_title += " Energy (MeV)"
        elif var in ["X", "Y", "Z"]:
            x_axis_title += f" {var} Coordinate (cm)"
        fig.update_xaxes(title_text=x_axis_title)
        fig.update_yaxes(title_text="Production Distributions (%)")

        fig.add_hline(y=100, line_width=1, line_dash="dash", line_color="black")
        save_figure(
            fig,
            save_path,
            config,
            name,
            filename=f"Production{var}_Distribution_{reference}",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )
    
    for df, df_name in zip([lim_df, eff_df], ["Production_Limits", "Production_Distributions"]):
        save_df(
            df,
            data_path,
            config,
            name,
            filename=df_name,
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )