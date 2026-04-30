import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *

save_path = f"{root}/images/solar/results"
data_path = f"{root}/data/solar/results"

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
    "--name", type=str, help="The name of the configuration", default="alpha"
)
parser.add_argument(
    "--folder", type=str, help="The name of the background folder", choices=["Reduced", "Truncated", "Nominal"], default="Nominal",
)
parser.add_argument(
    "--nhits", type=int, help="The nhit cut for the analysis", default=1
)
parser.add_argument(
    "--ophits", type=int, help="The ophit cut for the analysis", default=4
)
parser.add_argument(
    "--adjcls", type=int, help="The adjacent cluster cut for the analysis", default=10
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()
config = args.config
name = args.name

configs = {config: [name]}

for path in [save_path, data_path]:
    if not os.path.exists(f"{path}/{args.folder.lower()}"):
        os.makedirs(f"{path}/{args.folder.lower()}")

user_input = {
    "reduction": {"gamma": 3, "neutron": 2, "alpha": 1},
    "directory": {
        "neutron": f"background/{args.folder.lower()}",
        "gamma": f"background/{args.folder.lower()}",
        "alpha": f"background/{args.folder.lower()}",
    },
    "weights": {
        "neutron": ["SignalParticleWeight"],
        "gamma": ["SignalParticleWeight"],
        "alpha": ["SignalParticleWeight"],
    },
    "weight_labels": {
        "neutron": ["neutron"],
        "gamma": ["gamma"],
        "alpha": ["alpha"],
    },
    "colors": {
        "neutron": ["rgb(15,133,84)"],
        "gamma": ["black"],
        "alpha": ["rgb(29, 105, 150)"],
    },
    "yzoom": {"neutron": [0, 6], "gamma": [0, 6], "alpha": [2, 8]},
    "workflow": "ANALYSIS",
    "rewrite": True,
    "debug": True,
}

run, output = load_multi(
    configs,
    preset=user_input["workflow"],
    branches={"Config": ["Geometry"]},
    debug=user_input["debug"],
)
run = compute_reco_workflow(
    run, configs, workflow=user_input["workflow"], debug=user_input["debug"]
)
# run, output, this_new_branches = compute_particle_weights(
#     run,
#     configs,
#     rm_branches=True,
#     output=output,
#     debug=user_input["debug"],
# )
# rprint(output)

for config in configs:
    info = json.loads(open(f"{root}/config/{config}/{config}_config.json").read())
    fiducials = json.loads(open(f"{root}/data/solar/fiducial/{args.folder.lower()}/BestFiducials.json").read())
    detector_x = info["DETECTOR_SIZE_X"] + 2 * info["DETECTOR_GAP_X"]
    detector_y = info["DETECTOR_SIZE_Y"] + 2 * info["DETECTOR_GAP_Y"]
    detector_z = info["DETECTOR_SIZE_Z"] + 2 * info["DETECTOR_GAP_Z"]

    plot_list = []
    for name, energy in product(
        configs[config],
        [
            "SignalParticleK",
            "ClusterEnergy",
            "TotalEnergy",
            "SelectedEnergy",
            "SolarEnergy",
        ],
    ):
        save_pkl(
            run["Reco"]["SignalParticleK"],
            f"{data_path}/{args.folder.lower()}",
            config,
            name,
            filename=f"AnalysisEnergy_{energy}_Ref",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )
        save_pkl(
            run["Reco"][energy],
            f"{data_path}/{args.folder.lower()}",
            config,
            name,
            filename=f"AnalysisData_{energy}_Ref",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )
        save_pkl(
            run["Reco"]["SignalParticleWeight"],
            f"{data_path}/{args.folder.lower()}",
            config,
            name,
            filename=f"AnalysisWeights_{energy}_Ref",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )

        for (
            this_nhit,
            this_ophit,
            this_adjcl,
            (weight, weight_labels, color),
        ) in track(
            product(
                nhits[:10],
                nhits[3:10],
                nhits[::-1][10:],
                zip(
                    user_input["weights"][name],
                    user_input["weight_labels"][name],
                    user_input["colors"][name],
                ),
            ),
            total=10 * 7 * 10,
            description=f"Iterating over cut configurations for reco {energy}...",
        ):
            mask = (
                (run["Reco"]["SignalParticleSurface"] >= 0 if args.name.split("_")[0] in ["gamma", "neutron"] else 1)
                * ((run["Reco"]["SignalParticleSurface"] < 3) if (args.folder in ["Reduced", "Truncated"] and args.name.split("_")[0] in ["gamma", "neutron"]) else 1)
                * (run["Reco"]["NHits"] > this_nhit - 1)
                * (run["Reco"]["AdjClNum"] < this_adjcl)
                * (run["Reco"]["MatchedOpFlashPE"] > 0)
                * (run["Reco"]["MatchedOpFlashNHits"] > this_ophit - 1)
                * (
                    np.absolute(run["Reco"]["RecoX"])
                    > fiducials[config][energy]["FiducialX"]
                    if config == "hd_1x2x6_lateralAPA"
                    else np.absolute(run["Reco"]["RecoX"])
                    < detector_x / 2
                    - fiducials[config][energy]["FiducialX"] 
                    if config == "hd_1x2x6_centralAPA"
                    else run["Reco"]["RecoX"]
                    < detector_x / 2
                    - fiducials[config][energy]["FiducialX"]
                )
                * (
                    np.absolute(run["Reco"]["RecoY"])
                    < detector_y / 2 - fiducials[config][energy]["FiducialY"]
                )
                * (
                    (run["Reco"]["RecoZ"]
                    > fiducials[config][energy]["FiducialZ"] - info["DETECTOR_GAP_Z"]) if args.folder == "Nominal" else 1
                )
                * (
                    (run["Reco"]["RecoZ"]
                    < info["DETECTOR_SIZE_Z"]
                    + info["DETECTOR_GAP_Z"]
                    - fiducials[config][energy]["FiducialZ"]) if args.folder == "Nominal" else 1
                )
            )

            idx_mask = np.where(mask == True)
            h, bins = np.histogram(run["Reco"][energy][idx_mask], bins=energy_edges)
            mc_filter = h > 1
            mc_counts = h.copy()
            h_rel_error = np.sqrt(h) / h
            
            h, bins = np.histogram(
                run["Reco"][energy][idx_mask],
                bins=energy_edges,
                weights=run["Reco"][weight][idx_mask],
            )
            h *= mc_filter
            if args.folder == "Reduced":
                h = h / user_input["reduction"][name]
            h_error = h * h_rel_error
            h_error[np.isnan(h_error)] = 0

            counts = np.sum(h[energy_centers > 10])
            plot_list.append(
                {
                    "Idx": 0,
                    "Name": name,
                    "Component": weight_labels,
                    "Oscillation": "Truth",
                    "Mean": "Mean",
                    "Type": "background",
                    "MCCounts": mc_counts.tolist(),
                    "Counts": h.tolist(),
                    "Energy": energy_centers,
                    "Error": h_error,
                    "EnergyLabel": energy.split("Energy")[0],
                    "Color": color,
                    "NHits": this_nhit,
                    "OpHits": this_ophit,
                    "AdjCl": this_adjcl,
                }
            )

            if (
                this_nhit == args.nhits
                and this_ophit == args.ophits
                and this_adjcl == args.adjcls
                and weight == "SignalParticleWeight"
            ):

                    save_pkl(
                        mask,
                        f"{data_path}/{args.folder.lower()}",
                        config,
                        name,
                        filename=f"AnalysisMask_{energy}_NHits{this_nhit}_OpHits{this_ophit}_AdjCl{this_adjcl}",
                        rm=user_input["rewrite"],
                        debug=user_input["debug"],
                    )
                    save_pkl(
                        run["Reco"]["SignalParticleK"][idx_mask],
                        f"{data_path}/{args.folder.lower()}",
                        config,
                        name,
                        filename=f"AnalysisEnergy_{energy}_NHits{this_nhit}_OpHits{this_ophit}_AdjCl{this_adjcl}",
                        rm=user_input["rewrite"],
                        debug=user_input["debug"],
                    )
                    save_pkl(
                        run["Reco"][energy][idx_mask],
                        f"{data_path}/{args.folder.lower()}",
                        config,
                        name,
                        filename=f"AnalysisData_{energy}_NHits{this_nhit}_OpHits{this_ophit}_AdjCl{this_adjcl}",
                        rm=user_input["rewrite"],
                        debug=user_input["debug"],
                    )
                    save_pkl(
                        run["Reco"][weight][idx_mask],
                        f"{data_path}/{args.folder.lower()}",
                        config,
                        name,
                        filename=f"AnalysisWeights_{energy}_NHits{this_nhit}_OpHits{this_ophit}_AdjCl{this_adjcl}",
                        rm=user_input["rewrite"],
                        debug=user_input["debug"],
                    )

    df = pd.DataFrame(plot_list)
    save_df(
        df,
        f"{info['PATH']}/{user_input['directory'][name]}",
        config=config,
        name=None,
        filename=f"{name}",
        rm=user_input["rewrite"],
        debug=user_input["debug"],
    )

    for rebin, analysis in zip(
        [daynight_rebin, sensitivity_rebin, hep_rebin],
        ["DayNight", "Sensitivity", "HEP"],
    ):
        rebin_df = rebin_df_columns(
            df, rebin, "Energy", "Counts", "Counts/Energy", "Error"
        )

        save_df(
            rebin_df,
            f"{info['PATH']}/{user_input['directory'][name]}/{analysis.upper()}",
            config=config,
            name=name,
            filename=f"rebin",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )
