import sys

sys.path.insert(0, "../../")

from lib import *

save_path = f"{root}/images/solar/results/"

if not os.path.exists(save_path):
    os.makedirs(save_path)

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
    "--name", type=str, help="The name of the configuration", default="marley"
)
parser.add_argument(
    "--rewrite", action="store_true", help="Rewrite the files", default=True
)
parser.add_argument(
    "--debug", action="store_true", help="Debug the files", default=True
)

config = parser.parse_args().config
name = parser.parse_args().name

configs = {config: [name]}

user_input = {
    "yzoom": {"marley": [-2, 3], "neutron": [1, 6], "gamma": [1, 6], "alpha": [2, 7]},
    "directory": {
        "marley": "signal",
        "neutron": "background",
        "gamma": "background",
        "alpha": "background",
    },
    "weights": {
        "marley": [
            "SignalParticleWeight",
            "SignalParticleWeightb8",
            "SignalParticleWeighthep",
        ],
        "neutron": ["SignalParticleWeight"],
        "gamma": ["SignalParticleWeight"],
        "alpha": ["SignalParticleWeight"],
    },
    "weight_labels": {
        "marley": ["Solar", "8B", "hep"],
        "neutron": ["neutron"],
        "gamma": ["gamma"],
        "alpha": ["alpha"],
    },
    "colors": {
        "marley": ["grey", "rgb(225,124,5)", "rgb(204,80,62)"],
        "neutron": ["rgb(15,133,84)"],
        "gamma": ["black"],
        "alpha": ["rgb(29, 105, 150)"],
    },
    "workflow": "ANALYSIS",
    "rewrite": parser.parse_args().rewrite,
    "debug": parser.parse_args().debug,
}

run, output = load_multi(
    configs,
    preset=user_input["workflow"],
    branches={"Config": ["Geometry"]},
    debug=user_input["debug"],
)
rprint(output)
run = compute_reco_workflow(
    run, configs, workflow=user_input["workflow"], debug=user_input["debug"]
)
run, output, this_new_branches = compute_particle_weights(
    run, configs, {}, rm_branches=True, output=output, debug=user_input["debug"]
)
run, output = compute_filtered_run(
    run, configs, params={}, presets=[user_input["workflow"]], debug=user_input["debug"]
)
rprint(output)

trimmed_run = {}
for tree in ["Reco"]:
    trimmed_run[tree] = {}
    for branch in [
        "Event",
        "Geometry",
        "Version",
        "Name",
        "SignalParticleWeight",
        "SignalParticleWeightb8",
        "SignalParticleWeighthep",
        "NHits",
        "AdjClNum",
        "MatchedOpFlashNHits",
        "RecoX",
        "RecoY",
        "RecoZ",
        "ClusterEnergy",
        "TotalEnergy",
        "SelectedEnergy",
        "SolarEnergy",
    ]:
        try:
            trimmed_run[tree][branch] = run[tree][branch]
        except KeyError:
            print(f"Missing {branch} in {tree}")
            pass

plot_list = []
for config, energy in product(
    configs, ["ClusterEnergy", "TotalEnergy", "SelectedEnergy", "SolarEnergy"]
):

    fig = make_subplots(rows=1, cols=1, subplot_titles=([energy]))
    info = json.loads(open(f"{root}/config/{config}/{config}_config.json").read())
    detector_x = info["DETECTOR_SIZE_X"] + 2 * info["DETECTOR_GAP_X"]
    detector_y = info["DETECTOR_SIZE_Y"] + 2 * info["DETECTOR_GAP_Y"]
    detector_z = info["DETECTOR_SIZE_Z"] + 2 * info["DETECTOR_GAP_Z"]
    for name in configs[config]:
        for (
            (idx, fiducial),
            nhit,
            ophit,
            adjcl,
            (weight, weight_labels, color),
        ) in track(
            product(
                enumerate(np.arange(0.00, 140, 20)),
                nhits[:10],
                nhits[3:10],
                nhits[::-1][6:],
                zip(
                    user_input["weights"][name],
                    user_input["weight_labels"][name],
                    user_input["colors"][name],
                ),
            ),
            description=f"Processing {name}",
            total=len(np.arange(0.00, 140, 20))
            * len(nhits[:10])
            * len(nhits[3:10])
            * len(nhits[::-1][6:])
            * len(user_input["weights"][name]),
        ):
            if fiducial == 0:
                this_run, output = compute_filtered_run(
                    trimmed_run,
                    configs,
                    params={
                        ("Reco", "NHits"): ("bigger", (nhit - 1)),
                        ("Reco", "AdjClNum"): ("smaller", adjcl),
                    },
                    debug=user_input["debug"],
                )

            else:
                this_run, output = compute_filtered_run(
                    trimmed_run,
                    configs,
                    params={
                        ("Reco", "NHits"): ("bigger", nhit - 1),
                        ("Reco", "AdjClNum"): ("smaller", adjcl),
                        ("Reco", "MatchedOpFlashNHits"): ("bigger", (ophit - 1)),
                        ("Reco", "RecoX"): (
                            "absbetween",
                            (fiducial * 0.1, detector_x / 2),
                        ),
                        ("Reco", "RecoY"): (
                            "absbetween",
                            (0, detector_y / 2 - fiducial),
                        ),
                        ("Reco", "RecoZ"): (
                            "between",
                            (
                                fiducial - info["DETECTOR_GAP_Z"],
                                info["DETECTOR_SIZE_Z"]
                                + info["DETECTOR_GAP_Z"]
                                - fiducial,
                            ),
                        ),
                    },
                    debug=user_input["debug"],
                )

            h, bins = np.histogram(this_run["Reco"][energy], bins=energy_edges)
            h_rel_error = np.sqrt(h) / h
            h, bins = np.histogram(
                this_run["Reco"][energy],
                bins=energy_edges,
                weights=this_run["Reco"][weight],
            )
            h_error = h * h_rel_error
            h_error[np.isnan(h_error)] = 0

            counts = np.sum(h[energy_centers > 10])
            plot_list.append(
                {
                    "Idx": 0,
                    "Name": name,
                    "Component": weight_labels,
                    "Type": user_input["directory"][name],
                    "Counts": h.tolist(),
                    "Energy": energy_centers,
                    "Error": h_error,
                    "EnergyLabel": energy.split("Energy")[0],
                    "Color": color,
                    "Fiducialized": int(fiducial),
                    "NHits": nhit,
                    "OpHits": ophit,
                    "AdjCl": adjcl,
                }
            )

            if (
                nhit == 1
                and ophit == 4
                and adjcl == 10
                and weight == "SignalParticleWeight"
            ):
                fig.add_trace(
                    go.Scatter(
                        x=bins,
                        y=h,
                        mode="lines",
                        showlegend=True,
                        name=f"{fiducial/100:.1f}: {counts:.1e} counts",
                        line_shape="hvh",
                        line=dict(color=colors[1 + idx]),
                    ),
                    row=1,
                    col=1,
                )

        # Add verticlal lines
        fig.add_vline(
            10,
            line_width=1,
            line_dash="dash",
            line_color="grey",
            annotation_text=" Thld. 10 MeV",
            annotation_position="bottom right",
        )

        fig = format_coustom_plotly(
            fig,
            title=f"Weighted {energy} ({name})",
            log=(False, True),
            ranges=(None, user_input["yzoom"][name]),
            legend=dict(x=0.7, y=0.99),
            legend_title="Fiducial (m)",
            tickformat=(".0f", ".0e"),
        )

        fig.update_xaxes(title="Reconstructed K.E. (MeV)")
        fig.update_yaxes(title="Counts per (kT · year)")
        save_figure(
            fig,
            save_path,
            config,
            name,
            f"Particle_{energy}_Fiducial_Hist",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )

df = pd.DataFrame(plot_list)
save_df(
    df,
    f"{info['PATH']}/{user_input['directory'][name]}",
    None,
    None,
    f"{config}_{name}",
    rm=user_input["rewrite"],
    debug=user_input["debug"],
)
