import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *

save_path = f"{root}/images/solar/results"
cuts_path = f"{root}/data/solar/cuts"
data_path = f"{root}/data/solar/weighted"

_ANALYSIS_LOCAL_DIR = {
    "DayNight": "daynight-json",
    "HEP": "hep-json",
    "Sensitivity": "sensitivity-json",
}


def _load_best_cuts(analysis, folder, config, name):
    dir_name = _ANALYSIS_LOCAL_DIR.get(analysis)
    if not dir_name:
        return {}
    path = f"{root}/data/analysis/{dir_name}/{folder.lower()}/{config}/{name}/{config}_{name}_highest_{analysis}.json"
    if not os.path.exists(path):
        return {}
    d = json.load(open(path))
    return d.get(config, {}).get(name, {})



def build_analysis_mask(run, args, config, info, fiducial, detector_x, detector_y, this_nhit, this_ophit, this_adjcl, sample_name, energy=None, band_fiducials=None):
    quality_mask = (
        (
            (run["Reco"]["SignalParticleSurface"] >= 0)
            * (run["Reco"]["SignalParticleSurface"] < 3)
            if is_surface_background(str(root), sample_name)
            else 1
        )
        * (run["Reco"]["NHits"] > this_nhit - 1)
        * (run["Reco"]["AdjClNum"] < this_adjcl)
        * (run["Reco"]["MatchedOpFlashPlane"] == 0)
        * (run["Reco"]["MatchedOpFlashPE"] > 0)
        * (run["Reco"]["MatchedOpFlashNHits"] > this_ophit - 1)
    )
    # Per-energy-band spatial mask: each band may have a different optimal fiducial.
    # Events outside all bands fall back to the global fiducial.
    spatial_mask = build_energy_band_spatial_mask(
        run, config, detector_x, detector_y, info, args.folder,
        fiducial, band_fiducials or [], energy or "",
    ) if (band_fiducials and energy) else build_fiducial_spatial_mask(
        run, config, detector_x, detector_y, info, args.folder, fiducial
    )
    return quality_mask * spatial_mask


def build_cut_impact(run, args, config, info, fiducial, detector_x, detector_y, this_nhit, this_ophit, this_adjcl, sample_name):
    events = len(run["Reco"]["Event"])
    surface = (
        (run["Reco"]["SignalParticleSurface"] >= 0)
        * (run["Reco"]["SignalParticleSurface"] < 3)
        if is_surface_background(str(root), sample_name)
        else np.ones(len(run["Reco"]["Event"]), dtype=bool)
    )
    fiducialx = (
        np.absolute(run["Reco"]["RecoX"]) > fiducial["FiducialX"]
        if config == "hd_1x2x6_lateralAPA"
        else (
            np.absolute(run["Reco"]["RecoX"]) < detector_x / 2 - fiducial["FiducialX"]
            if config == "hd_1x2x6_centralAPA"
            else run["Reco"]["RecoX"] < detector_x / 2 - fiducial["FiducialX"]
        )
    )
    fiducialy = np.absolute(run["Reco"]["RecoY"]) < detector_y / 2 - fiducial["FiducialY"]
    # build_cut_impact reports per-axis efficiency so keeps per-axis cuts separate
    fiducialz = (run["Reco"]["RecoZ"] > fiducial["FiducialZ"] - info["DETECTOR_GAP_Z"]) & (
        run["Reco"]["RecoZ"] < info["DETECTOR_SIZE_Z"] + info["DETECTOR_GAP_Z"] - fiducial["FiducialZ"]
    )
    return {
        f"NHits>{this_nhit-1}_AdjClNum<{this_adjcl}_OpHits>{this_ophit-1}": {
            "Truncate": 100 * np.sum(surface) / events,
            "NHits": 100 * np.sum(run["Reco"]["NHits"] > this_nhit - 1) / events,
            "AdjClNum": 100 * np.sum(run["Reco"]["AdjClNum"] < this_adjcl) / events,
            "MatchedOpFlashNHits": 100 * np.sum(run["Reco"]["MatchedOpFlashNHits"] > this_ophit - 1) / events,
            "MatchedOpFlashPlane": 100 * np.sum(run["Reco"]["MatchedOpFlashPlane"] == 0) / events,
            "MatchedOpFlashPE": 100 * np.sum(run["Reco"]["MatchedOpFlashPE"] > 0) / events,
            "Fiducial": 100 * np.sum(fiducialx & fiducialy & fiducialz) / events,
            "FiducialX": 100 * np.sum(fiducialx) / events,
            "FiducialY": 100 * np.sum(fiducialy) / events,
            "FiducialZ": 100 * np.sum(fiducialz) / events,
        }
    }


parser = argparse.ArgumentParser(description="Plot the energy distribution of the particles")
parser.add_argument("--config", type=str, help="The configuration to load", default="hd_1x2x6_centralAPA")
parser.add_argument("--name", type=str, help="The name of the configuration", default="marley")
parser.add_argument("--folder", type=str, help="The name of the background folder", choices=["Reduced", "Truncated", "Nominal"], default="Nominal")
parser.add_argument("--analysis", nargs="+", type=str, help="The name of the analysis", choices=["DayNight", "HEP", "Sensitivity"], default=["DayNight", "HEP", "Sensitivity"])
parser.add_argument("--energy", nargs="+", type=str, help="The energy variable to plot", choices=["SignalParticleK", "ClusterEnergy", "TotalEnergy", "SelectedEnergy", "SolarEnergy"], default=["SignalParticleK", "ClusterEnergy", "TotalEnergy", "SelectedEnergy", "SolarEnergy"])
parser.add_argument("--nhits", type=int, help="The nhits cut for the analysis", default=None)
parser.add_argument("--ophits", type=int, help="The ophit cut for the analysis", default=None)
parser.add_argument("--adjcls", type=int, help="The adjacent cluster cut for the analysis", default=None)
parser.add_argument(
    "--mc_filter_threshold",
    type=int,
    default=2,
    help=(
        "Minimum MC event count per energy bin. Bins below this threshold are zeroed "
        "before weighting to suppress shape artefacts from low statistics. "
        "Lower values preserve more tail bins at the cost of increased shape noise; "
        "the HEP analysis is sensitive to this choice in the high-energy tail."
    ),
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--export_raw", action=argparse.BooleanOptionalAction, default=True, help="Export raw numpy arrays (energy, mask, weights) as pkl files")
parser.add_argument(
    "--oscillation_backend",
    type=str,
    choices=["file", "prob3", "nufast"],
    default="file",
    help="Oscillation weighting backend. 'file' uses pre-computed pkl files; 'prob3'/'nufast' compute on-the-fly.",
)

args = parser.parse_args()
config = args.config
name = args.name
configs = {config: [name]}

for path in [save_path, data_path]:
    if not os.path.exists(f"{path}/{args.folder.lower()}"):
        os.makedirs(f"{path}/{args.folder.lower()}")

user_input = {
    "workflow": "SIGNIFICANCE",

    "directory": {
        "marley": f"signal/{args.folder.lower()}",
        "neutron": f"background/{args.folder.lower()}",
        "gamma": f"background/{args.folder.lower()}",
        "alpha": f"background/{args.folder.lower()}",
        "radiological": f"background/{args.folder.lower()}",
    },
    "weights": {
        "marley": ["SignalParticleWeight", "SignalParticleWeightb8", "SignalParticleWeighthep"],
        "neutron": ["SignalParticleWeight"],
        "gamma": ["SignalParticleWeight"],
        "alpha": ["SignalParticleWeight"],
        "radiological": ["SignalParticleWeight"],
    },
    "weight_labels": {
        "marley": ["Solar", "8B", "hep"],
        "neutron": ["neutron"],
        "gamma": ["gamma"],
        "alpha": ["alpha"],
        "radiological": ["radiological"],
    },
    "colors": {
        "marley": ["grey", "rgb(225,124,5)", "rgb(204,80,62)"],
        "neutron": ["rgb(15,133,84)"],
        "gamma": ["black"],
        "alpha": ["rgb(29, 105, 150)"],
        "radiological": ["rgb(120, 94, 240)"],
    },
    "yzoom": {"marley": [0, 6], "neutron": [0, 6], "gamma": [0, 6], "alpha": [2, 8], "radiological": [0, 6]},
    "rewrite": args.rewrite,
    "debug": args.debug,
}

run, output = load_multi(
    configs,
    preset=user_input["workflow"],
    branches={"Config": ["Geometry"]},
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
            "OSCILLATION_BACKEND": args.oscillation_backend,
        }
        if "marley" in args.name
        else {"PARTICLE_TYPE": "background", "PARTICLE_WEIGHTING": "histogram"}
    ),
    workflow=user_input["workflow"],
    rm_branches=False,
    debug=args.debug,
)

for config in configs:
    info = json.loads(open(f"{root}/config/{config}/{config}_config.json").read())
    fiducials = json.loads(open(f"{root}/data/solar/fiducial/{args.folder.lower()}/BestFiducials.json").read())
    detector_x = info["DETECTOR_SIZE_X"] + 2 * info["DETECTOR_GAP_X"]
    detector_y = info["DETECTOR_SIZE_Y"] + 2 * info["DETECTOR_GAP_Y"]

    for name, energy in product(configs[config], args.energy):
        if args.export_raw:
            save_pkl(run["Reco"]["SignalParticleK"], data_path, config, name, subfolder=args.folder.lower(), filename=f"AnalysisEnergy_{energy}_Ref", rm=user_input["rewrite"], debug=args.export_raw)
            save_pkl(run["Reco"][energy], data_path, config, name, subfolder=args.folder.lower(), filename=f"AnalysisData_{energy}_Ref", rm=user_input["rewrite"], debug=args.export_raw)
            save_pkl(run["Reco"]["SignalParticleWeight"], data_path, config, name, subfolder=args.folder.lower(), filename=f"AnalysisWeights_{energy}_Ref", rm=user_input["rewrite"], debug=args.export_raw)

        plot_lists = {analysis: [] for analysis in args.analysis}
        fiducials_by_analysis = {analysis: get_best_fiducial(fiducials, config, energy, analysis) for analysis in args.analysis}
        band_fiducials_by_analysis = {analysis: get_best_fiducial_bands(fiducials, config, energy, analysis) for analysis in args.analysis}
        analysis_cache = {}
        _prev_cut = None

        best_cuts_by_analysis = {}
        if args.nhits is None and args.ophits is None and args.adjcls is None and args.export_raw:
            for analysis in args.analysis:
                cuts = _load_best_cuts(analysis, args.folder, config, name)
                best_cuts_by_analysis[analysis] = cuts.get(energy, {})

        for this_nhit, this_ophit, this_adjcl, (weight, weight_labels, color) in track(
            product(
                nhits if args.nhits is None else [args.nhits],
                nhits[3:] if args.ophits is None else [args.ophits],
                nhits[::-1] if args.adjcls is None else [args.adjcls],
                zip(user_input["weights"][name], user_input["weight_labels"][name], user_input["colors"][name]),
            ),
            total=(len(nhits) * len(nhits[3:]) * len(nhits[::-1]) * (3 if "marley" in name else 1) if args.nhits is None and args.ophits is None and args.adjcls is None else 1),
            description=f"Iterating over cut configurations for reco {energy}...",
        ):
            _cur_cut = (this_nhit, this_ophit, this_adjcl)
            if _prev_cut is not None and _cur_cut != _prev_cut:
                for _a in args.analysis:
                    analysis_cache.pop((_a, *_prev_cut), None)
            _prev_cut = _cur_cut

            for analysis in args.analysis:
                fiducial = fiducials_by_analysis[analysis]
                cache_key = (analysis, this_nhit, this_ophit, this_adjcl)
                if cache_key not in analysis_cache:
                    mask = np.asarray(
                        build_analysis_mask(
                            run,
                            args,
                            config,
                            info,
                            fiducial,
                            detector_x,
                            detector_y,
                            this_nhit,
                            this_ophit,
                            this_adjcl,
                            name,
                            energy=energy,
                            band_fiducials=band_fiducials_by_analysis[analysis],
                        ),
                        dtype=bool,
                    )
                    selected_reco_energy = run["Reco"][energy][mask]
                    selected_true_energy = run["Reco"]["SignalParticleK"][mask]
                    mc_counts, _ = np.histogram(selected_reco_energy, bins=true_energy_edges)
                    mc_filter = mc_counts >= args.mc_filter_threshold
                    if args.debug and np.any(~mc_filter):
                        rprint(
                            f"[yellow][WARNING][/yellow] {np.sum(~mc_filter)} bins with MC < "
                            f"{args.mc_filter_threshold} zeroed for {energy} "
                            f"NHits={this_nhit} OpHits={this_ophit} AdjCl={this_adjcl}"
                        )
                    h_rel_error = np.zeros_like(mc_counts, dtype=float)
                    non_zero_bins = mc_counts > 0
                    h_rel_error[non_zero_bins] = (
                        np.sqrt(mc_counts[non_zero_bins]) / mc_counts[non_zero_bins]
                    )

                    analysis_cache[cache_key] = {
                        "mask": mask,
                        "selected_reco_energy": selected_reco_energy,
                        "selected_true_energy": selected_true_energy,
                        "mc_counts": mc_counts,
                        "mc_filter": mc_filter,
                        "h_rel_error": h_rel_error,
                    }

                cached = analysis_cache[cache_key]
                mask = cached["mask"]

                for osc, mean, mean_label in zip(
                    ["Truth", "Osc", "Osc", "Osc"] if "marley" in name else ["Truth"],
                    ["Mean", "Day", "Night", "Mean"] if "marley" in name else ["Mean"],
                    ["", "OscDay", "OscNight", "OscMean"] if "marley" in name else [""],
                ):
                    selected_weights = run["Reco"][f"{weight}{mean_label}"][mask]
                    h_true, bins = np.histogram(
                        cached["selected_true_energy"],
                        bins=true_energy_edges,
                        weights=selected_weights,
                    )
                    h, bins = np.histogram(
                        cached["selected_reco_energy"],
                        bins=true_energy_edges,
                        weights=selected_weights,
                    )
                    h *= cached["mc_filter"]
                    if folder_applies_reduction(str(root), args.folder):
                        h = h / get_component_reduction_factor(str(root), args.folder, name)
                    h_error = h * cached["h_rel_error"]
                    h_error[np.isnan(h_error)] = 0

                    plot_lists[analysis].append({
                        "Geometry": config.split("_")[0],
                        "Config": config,
                        "Name": name,
                        "Analysis": analysis,
                        "Component": weight_labels,
                        "Oscillation": osc,
                        "Mean": mean,
                        "Type": "signal" if "marley" in name else "background",
                        "MCCounts": cached["mc_counts"].tolist(),
                        "TrueCounts": h_true.tolist(),
                        "Counts": h.tolist(),
                        "Energy": true_energy_centers,
                        "Error": h_error,
                        "Color": color,
                        "NHits": this_nhit,
                        "OpHits": this_ophit,
                        "AdjCl": this_adjcl,
                    })

                if this_nhit == args.nhits and this_ophit == args.ophits and this_adjcl == args.adjcls and weight == "SignalParticleWeight":
                    analysis_key = analysis.upper()
                    if args.export_raw:
                        save_pkl(np.asarray(mask), cuts_path, config, name, subfolder=args.folder.lower(), filename=f"AnalysisMask_{energy}_{analysis_key}_NHits{this_nhit}_OpHits{this_ophit}_AdjCl{this_adjcl}", rm=user_input["rewrite"], debug=user_input["debug"])
                        save_pkl(run["Reco"]["SignalParticleK"][mask], cuts_path, config, name, subfolder=args.folder.lower(), filename=f"AnalysisEnergy_{energy}_{analysis_key}_NHits{this_nhit}_OpHits{this_ophit}_AdjCl{this_adjcl}", rm=user_input["rewrite"], debug=user_input["debug"])
                        save_pkl(run["Reco"][energy][mask], cuts_path, config, name, subfolder=args.folder.lower(), filename=f"AnalysisData_{energy}_{analysis_key}_NHits{this_nhit}_OpHits{this_ophit}_AdjCl{this_adjcl}", rm=user_input["rewrite"], debug=user_input["debug"])
                        save_pkl(run["Reco"][weight][mask], cuts_path, config, name, subfolder=args.folder.lower(), filename=f"AnalysisWeights_{energy}_{analysis_key}_NHits{this_nhit}_OpHits{this_ophit}_AdjCl{this_adjcl}", rm=user_input["rewrite"], debug=user_input["debug"])

                    cut_impact = build_cut_impact(run, args, config, info, fiducial, detector_x, detector_y, this_nhit, this_ophit, this_adjcl, name)
                    cut_dir = f"{cuts_path}/{config}/{args.name}/{args.folder.lower()}"
                    if not os.path.exists(cut_dir):
                        os.makedirs(cut_dir)
                    cut_path = f"{cut_dir}/analysis_cuts_{analysis_key}.json"
                    merge_and_write_json(cut_path, cut_impact)

                if args.export_raw and weight == "SignalParticleWeight" and best_cuts_by_analysis:
                    bc = best_cuts_by_analysis.get(analysis, {})
                    if bc and int(bc.get("NHits", -1)) == this_nhit and int(bc.get("OpHits", -1)) == this_ophit and int(bc.get("AdjCl", -1)) == this_adjcl:
                        analysis_key = analysis.upper()
                        save_pkl(np.asarray(mask), cuts_path, config, name, subfolder=args.folder.lower(), filename=f"AnalysisMask_{energy}_{analysis_key}_NHits{this_nhit}_OpHits{this_ophit}_AdjCl{this_adjcl}", rm=user_input["rewrite"], debug=user_input["debug"])
                        save_pkl(run["Reco"]["SignalParticleK"][mask], cuts_path, config, name, subfolder=args.folder.lower(), filename=f"AnalysisEnergy_{energy}_{analysis_key}_NHits{this_nhit}_OpHits{this_ophit}_AdjCl{this_adjcl}", rm=user_input["rewrite"], debug=user_input["debug"])
                        save_pkl(run["Reco"][energy][mask], cuts_path, config, name, subfolder=args.folder.lower(), filename=f"AnalysisData_{energy}_{analysis_key}_NHits{this_nhit}_OpHits{this_ophit}_AdjCl{this_adjcl}", rm=user_input["rewrite"], debug=user_input["debug"])
                        save_pkl(run["Reco"][weight][mask], cuts_path, config, name, subfolder=args.folder.lower(), filename=f"AnalysisWeights_{energy}_{analysis_key}_NHits{this_nhit}_OpHits{this_ophit}_AdjCl{this_adjcl}", rm=user_input["rewrite"], debug=user_input["debug"])

        if _prev_cut is not None:
            for _a in args.analysis:
                analysis_cache.pop((_a, *_prev_cut), None)

        rebin_dict = {"DayNight": daynight_rebin, "Sensitivity": sensitivity_rebin, "HEP": hep_rebin}
        for analysis in args.analysis:
            df = pd.DataFrame(plot_lists[analysis])
            if df.empty:
                continue
            save_df(df, data_path, config=config, name=name, subfolder=args.folder.lower(), filename=f"{energy}_{analysis}_AnalysisData", rm=user_input["rewrite"], debug=user_input["debug"])
            rebin_df = rebin_df_columns(df, rebin_dict[analysis], "Energy", "Counts", "Counts/Energy", "Error", "MCCounts")
            save_df(rebin_df, f"{info['PATH']}/{user_input['directory'][name]}/{analysis.upper()}", config=config, name=name, filename=f"{energy}_Rebin", rm=user_input['rewrite'], debug=user_input['debug'])
