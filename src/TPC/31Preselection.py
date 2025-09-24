import sys

sys.path.insert(0, "../../")

from lib import *

save_path = f"{root}/images/TPC/preselction/"
data_path = f"{root}/data/TPC/preselection/"

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

logy = False
for config in configs:
    purity_list, completeness_list = [], []
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
        for var, bins in zip(
            ["K", "X", "Y", "Z"],
            [
                red_energy_edges,
                np.arange(
                    info["DETECTOR_MIN_X"],
                    info["DETECTOR_MAX_X"] + params["DEFAULT_X_BIN"],
                    params["DEFAULT_X_BIN"],
                ),
                np.arange(
                    info["DETECTOR_MIN_Y"],
                    info["DETECTOR_MAX_Y"] + params["DEFAULT_Y_BIN"],
                    params["DEFAULT_Y_BIN"],
                ),
                np.arange(
                    info["DETECTOR_MIN_Z"],
                    info["DETECTOR_MAX_Z"] + params["DEFAULT_Z_BIN"],
                    params["DEFAULT_Z_BIN"],
                ),
            ],
        ):
            h, edges = np.histogram(run["Truth"][f"SignalParticle{var}"], bins=bins)
            h_ref[var] = h
            edges_ref[var] = edges

        for hit in nhits[:9]:
            draw = True
            if np.sum(run["Truth"]["HitCount"] == hit) == 0:
                draw = False

            for var, bins in zip(
                ["K", "X", "Y", "Z"],
                [
                    red_energy_edges,
                    np.arange(
                        info["DETECTOR_MIN_X"],
                        info["DETECTOR_MAX_X"] + params["DEFAULT_X_BIN"],
                        params["DEFAULT_X_BIN"],
                    ),
                    np.arange(
                        info["DETECTOR_MIN_Y"],
                        info["DETECTOR_MAX_Y"] + params["DEFAULT_Y_BIN"],
                        params["DEFAULT_Y_BIN"],
                    ),
                    np.arange(
                        info["DETECTOR_MIN_Z"],
                        info["DETECTOR_MAX_Z"] + params["DEFAULT_Z_BIN"],
                        params["DEFAULT_Z_BIN"],
                    ),
                ],
            ):
                this_data = run["Truth"][f"SignalParticle{var}"][
                    np.where(run["Truth"]["HitCount"] >= hit)
                ]
                h, edges = np.histogram(this_data, bins=bins)
                bin_centers = 0.5 * (edges[1:] + edges[:-1])
                this_centers = bin_centers[h != 0]
                this_h_ref = h_ref[var][h != 0]
                h = h[h != 0]

                eff_list.append(
                    {
                        "Variable": var,
                        "TrueEnergy": this_centers,
                        "Efficiency": h / this_h_ref,
                        "NHits": hit,
                        "Draw": draw,
                    }
                )

            save_pkl(
                pd.DataFrame(eff_list),
                data_path,
                config,
                name,
                filename=f"Reco_NHit_Efficiency_{hit}",
                rm=user_input["rewrite"],
                debug=user_input["debug"],
            )

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
                # range=(0, 1 + 2 * bin_width),
                density=True,
            )
            purity_dict = {
                "config": config,
                "name": name,
                "#Hits": hit,
                "Purity": x_centers,
                "Counts": y,
                "label": "Purity",
            }
            purity_list.append(purity_dict)
            y, bins = np.histogram(
                this_reco_data["ChargeFraction"],
                bins=x_axis,
                # range=(0, 1 + 2 * bin_width),
                density=True,
            )
            completeness_dict = {
                "config": config,
                "name": name,
                "#Hits": hit,
                "Completeness": x_centers,
                "Counts": y,
                "label": "Completeness",
            }
            completeness_list.append(completeness_dict)

        eff_df = pd.DataFrame(eff_list)
        eff_df = explode(
            eff_df,
            ["TrueEnergy", "Efficiency"],
            ["NHits", "Draw", "Variable"],
            debug=user_input["debug"],
        )
        if eff_df["Efficiency"].max() < 0.1:
            logy = True

        for var in ["K", "X", "Y", "Z"]:
            fig = px.line(
                eff_df[eff_df["Variable"] == var],
                x="TrueEnergy",
                y="Efficiency",
                color="NHits",
                color_discrete_sequence=px.colors.qualitative.Prism,
                line_shape="spline",
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
                ranges=(None, [-0.01, 1.1]),
                legend_title="min #Hits",
                debug=user_input["debug"],
            )
            x_axis_title = f"True {label}"
            if var == "K":
                x_axis_title += " Energy (MeV)"
            elif var in ["X", "Y", "Z"]:
                x_axis_title += f" {var} Coordinate (cm)"
            fig.update_xaxes(title_text=x_axis_title)
            fig.update_yaxes(title_text="Detection Efficiency")

            fig.add_hline(y=1, line_width=1, line_dash="dash", line_color="black")
            save_figure(
                fig,
                save_path,
                config,
                name,
                filename=f"Preselection{var}_Efficiency",
                rm=user_input["rewrite"],
                debug=user_input["debug"],
            )

        for plot_list, df_label in zip(
            [purity_list, completeness_list], ["Purity", "Completeness"]
        ):
            plot_df = explode(pd.DataFrame(plot_list), [df_label, "Counts"])
            fig = px.line(
                plot_df,
                x=df_label,
                y="Counts",
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
