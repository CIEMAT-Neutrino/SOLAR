import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *

save_path = f"{root}/images/solar/results"
cuts_path = f"{root}/data/solar/cuts"
data_path = f"{root}/data/solar/weighted"

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
    "--folder",
    type=str,
    help="The name of the background folder",
    choices=["Reduced", "Truncated", "Nominal"],
    default="Nominal",
)
parser.add_argument(
    "--analysis",
    nargs="+",
    type=str,
    help="The name of the analysis",
    choices=["DayNight", "HEP", "Sensitivity"],
    default=["DayNight", "HEP", "Sensitivity"],
)
parser.add_argument(
    "--energy",
    nargs="+",
    type=str,
    help="The energy variable to plot",
    choices=[
        "SignalParticleK",
        "ClusterEnergy",
        "TotalEnergy",
        "SelectedEnergy",
        "SolarEnergy",
    ],
    default=[
        "SignalParticleK",
        "ClusterEnergy",
        "TotalEnergy",
        "SelectedEnergy",
        "SolarEnergy",
    ],
)
parser.add_argument(
    "--nhits", type=int, help="The nhits cut for the analysis", default=None
)
parser.add_argument(
    "--ophits", type=int, help="The ophit cut for the analysis", default=None
)
parser.add_argument(
    "--adjcls", type=int, help="The adjacent cluster cut for the analysis", default=None
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
    "workflow": "SIGNIFICANCE",
    "reduction": {"marley": 1, "gamma": 3, "neutron": 2, "alpha": 1},
    "directory": {
        "marley": f"signal/{args.folder.lower()}",
        "neutron": f"background/{args.folder.lower()}",
        "gamma": f"background/{args.folder.lower()}",
        "alpha": f"background/{args.folder.lower()}",
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
    "yzoom": {"marley": [0, 6], "neutron": [0, 6], "gamma": [0, 6], "alpha": [2, 8]},
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
    run,
    configs,
    params=(
        {
            "DEFAULT_SIGNAL_WEIGHT": ["truth", "osc"],
            "DEFAULT_SIGNAL_AZIMUTH": ["mean", "day", "night"],
            "PARTICLE_TYPE": "signal",
            "PARTICLE_WEIGHTING": "volume",
        }
        if "marley" in args.name
        else {"PARTICLE_TYPE": "background", "PARTICLE_WEIGHTING": "histogram"}
    ),
    workflow=user_input["workflow"],
    rm_branches=False,
    debug=user_input["debug"],
)

for config in configs:
    info = json.loads(open(f"{root}/config/{config}/{config}_config.json").read())
    fiducials = json.loads(
        open(
            f"{root}/data/solar/fiducial/{args.folder.lower()}/BestFiducials.json"
        ).read()
    )
    detector_x = info["DETECTOR_SIZE_X"] + 2 * info["DETECTOR_GAP_X"]
    detector_y = info["DETECTOR_SIZE_Y"] + 2 * info["DETECTOR_GAP_Y"]
    detector_z = info["DETECTOR_SIZE_Z"] + 2 * info["DETECTOR_GAP_Z"]

    plot_list = []
    for name, energy in product(
        configs[config],
        args.energy,
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
                nhits[:10] if args.nhits is None else [args.nhits],
                nhits[3:10] if args.ophits is None else [args.ophits],
                nhits[::-1][10:] if args.adjcls is None else [args.adjcls],
                zip(
                    user_input["weights"][name],
                    user_input["weight_labels"][name],
                    user_input["colors"][name],
                ),
            ),
            total=(
                10 * 7 * 10 * (3 if "marley" in name else 1)
                if args.nhits is None and args.ophits is None and args.adjcls is None
                else 1
            ),
            description=f"Iterating over cut configurations for reco {energy}...",
        ):

            mask = (
                (
                    (run["Reco"]["SignalParticleSurface"] >= 0)
                    * (run["Reco"]["SignalParticleSurface"] < 3)
                    if args.name.split("_")[0] in ["gamma", "neutron"]
                    else 1
                )
                * (run["Reco"]["NHits"] > this_nhit - 1)
                * (run["Reco"]["AdjClNum"] < this_adjcl)
                * (run["Reco"]["MatchedOpFlashPlane"] == 0)
                * (run["Reco"]["MatchedOpFlashPE"] > 0)
                * (run["Reco"]["MatchedOpFlashNHits"] > this_ophit - 1)
                * (
                    np.absolute(run["Reco"]["RecoX"])
                    > fiducials[config][energy]["FiducialX"]
                    if config == "hd_1x2x6_lateralAPA"
                    else (
                        np.absolute(run["Reco"]["RecoX"])
                        < detector_x / 2 - fiducials[config][energy]["FiducialX"]
                        if config == "hd_1x2x6_centralAPA"
                        else run["Reco"]["RecoX"]
                        < detector_x / 2 - fiducials[config][energy]["FiducialX"]
                    )
                )
                * (
                    np.absolute(run["Reco"]["RecoY"])
                    < detector_y / 2 - fiducials[config][energy]["FiducialY"]
                )
                * (
                    (
                        run["Reco"]["RecoZ"]
                        > fiducials[config][energy]["FiducialZ"]
                        - info["DETECTOR_GAP_Z"]
                    )
                    if args.folder == "Nominal"
                    else 1
                )
                * (
                    (
                        run["Reco"]["RecoZ"]
                        < info["DETECTOR_SIZE_Z"]
                        + info["DETECTOR_GAP_Z"]
                        - fiducials[config][energy]["FiducialZ"]
                    )
                    if args.folder == "Nominal"
                    else 1
                )
            )

            # Update the cut impact with the current cut values
            if (
                args.nhits == this_nhit
                and args.ophits == this_ophit
                and args.adjcls == this_adjcl
            ):
                # Open json file with the cut impact on MC data and weighted data
                cut_impact = {}

                events = len(run["Reco"]["Event"])
                surface = (
                    (run["Reco"]["SignalParticleSurface"] >= 0)
                    * (run["Reco"]["SignalParticleSurface"] < 3)
                    if args.name.split("_")[0] in ["gamma", "neutron"]
                    else np.ones(len(run["Reco"]["Event"]), dtype=bool)
                )
                fiducialx = (
                    np.absolute(run["Reco"]["RecoX"])
                    > fiducials[config][energy]["FiducialX"]
                    if config == "hd_1x2x6_lateralAPA"
                    else (
                        np.absolute(run["Reco"]["RecoX"])
                        < detector_x / 2 - fiducials[config][energy]["FiducialX"]
                        if config == "hd_1x2x6_centralAPA"
                        else run["Reco"]["RecoX"]
                        < detector_x / 2 - fiducials[config][energy]["FiducialX"]
                    )
                )
                fiducialy = (
                    np.absolute(run["Reco"]["RecoY"])
                    < detector_y / 2 - fiducials[config][energy]["FiducialY"]
                )
                fiducialz = (
                    run["Reco"]["RecoZ"]
                    > fiducials[config][energy]["FiducialZ"] - info["DETECTOR_GAP_Z"]
                ) & (
                    run["Reco"]["RecoZ"]
                    < info["DETECTOR_SIZE_Z"]
                    + info["DETECTOR_GAP_Z"]
                    - fiducials[config][energy]["FiducialZ"]
                )

                cut_impact = {
                    f"NHits>{this_nhit-1}_AdjClNum<{this_adjcl}_OpHits>{this_ophit-1}": {
                        "Truncate": 100 * np.sum(surface) / events,
                        "NHits": 100
                        * np.sum(run["Reco"]["NHits"] > this_nhit - 1)
                        / events,
                        "AdjClNum": 100
                        * np.sum(run["Reco"]["AdjClNum"] < this_adjcl)
                        / events,
                        "MatchedOpFlashNHits": 100
                        * np.sum(run["Reco"]["MatchedOpFlashNHits"] > this_ophit - 1)
                        / events,
                        "MatchedOpFlashPlane": 100
                        * np.sum(run["Reco"]["MatchedOpFlashPlane"] == 0)
                        / events,
                        "MatchedOpFlashPE": 100
                        * np.sum(run["Reco"]["MatchedOpFlashPE"] > 0)
                        / events,
                        "Fiducial": 100
                        * np.sum(fiducialx & fiducialy & fiducialz)
                        / events,
                        "FiducialX": 100 * np.sum(fiducialx) / events,
                        "FiducialY": 100 * np.sum(fiducialy) / events,
                        "FiducialZ": 100 * np.sum(fiducialz) / events,
                    }
                }

            idx_mask = np.where(mask == True)
            h, bins = np.histogram(
                run["Reco"][energy][idx_mask], bins=true_energy_edges
            )
            mc_filter = h > 1
            mc_counts = h.copy()
            h_rel_error = np.sqrt(h) / h

            for osc, mean, mean_label in zip(
                ["Truth", "Osc", "Osc", "Osc"] if "marley" in name else ["Truth"],
                ["Mean", "Day", "Night", "Mean"] if "marley" in name else ["Mean"],
                ["", "OscDay", "OscNight", "OscMean"] if "marley" in name else [""],
            ):
                h_true, bins = np.histogram(
                    run["Reco"]["SignalParticleK"][idx_mask],
                    bins=true_energy_edges,
                    weights=run["Reco"][f"{weight}{mean_label}"][idx_mask],
                )
                h, bins = np.histogram(
                    run["Reco"][energy][idx_mask],
                    bins=true_energy_edges,
                    weights=run["Reco"][f"{weight}{mean_label}"][idx_mask],
                )
                h *= mc_filter
                if args.folder == "Reduced":
                    h = h / user_input["reduction"][name]
                h_error = h * h_rel_error
                h_error[np.isnan(h_error)] = 0

                counts = np.sum(h[true_energy_centers > 10])
                plot_list.append(
                    {
                        "Geometry": config.split("_")[0],
                        "Config": config,
                        "Name": name,
                        "Component": weight_labels,
                        "Oscillation": osc,
                        "Mean": mean,
                        "Type": "signal" if "marley" in name else "background",
                        "MCCounts": mc_counts.tolist(),
                        "TrueCounts": h_true.tolist(),
                        "Counts": h.tolist(),
                        "Energy": true_energy_centers,
                        "Error": h_error,
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
                    np.asarray(mask),
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
                # Save json file with the cut impact on MC data. If the file already exists, update it with the new cut impact. If not, create a new file with the cut impact.
                if not os.path.exists(
                    f"{cuts_path}/{args.folder.lower()}/{config}/{args.name}"
                ):
                    os.makedirs(
                        f"{cuts_path}/{args.folder.lower()}/{config}/{args.name}"
                    )
                    existing_cut_impact = cut_impact

                else:
                    if os.path.exists(
                        f"{cuts_path}/{args.folder.lower()}/{config}/{args.name}/analysis_cuts.json"
                    ):
                        with open(
                            f"{cuts_path}/{args.folder.lower()}/{config}/{args.name}/analysis_cuts.json",
                            "r",
                        ) as f_read:
                            print(
                                f"Updating existing cut impact file for {config} {name} {args.folder}..."
                            )
                            existing_cut_impact = json.load(f_read)
                        existing_cut_impact.update(cut_impact)
                    else:
                        existing_cut_impact = cut_impact

                with open(
                    f"{cuts_path}/{args.folder.lower()}/{config}/{args.name}/analysis_cuts.json",
                    "w",
                ) as f:
                    print(
                        f"Saving cut impact file for {config} {name} {args.folder}..."
                    )
                    json.dump(existing_cut_impact, f, indent=4)

        df = pd.DataFrame(plot_list)
        save_df(
            df,
            data_path,
            config=config,
            name=name,
            subfolder=args.folder.lower(),
            filename=f"{energy}_AnalysisData",
            rm=user_input["rewrite"],
            debug=user_input["debug"],
        )
        rebin_dict = {
            "DayNight": daynight_rebin,
            "Sensitivity": sensitivity_rebin,
            "HEP": hep_rebin,
        }
        rebin_array = [rebin_dict[analysis] for analysis in args.analysis]
        for rebin, analysis in zip(
            rebin_array,
            args.analysis,
        ):
            rebin_df = rebin_df_columns(
                df, rebin, "Energy", "Counts", "Counts/Energy", "Error"
            )

            save_df(
                rebin_df,
                f"{info['PATH']}/{user_input['directory'][name]}/{analysis.upper()}",
                config=config,
                name=name,
                filename=f"{energy}_Rebin",
                rm=user_input["rewrite"],
                debug=user_input["debug"],
            )
