import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *

save_path   = f"{root}/images/solar/results"
export_path = f"{root}/output/analysis"
data_path   = f"{root}/data/solar/weighted"

_ANALYSIS_LOCAL_DIR = {
    "DayNight": "daynight-json",
    "HEP": "hep-json",
    "Sensitivity": "sensitivity-json",
}


def _load_best_cuts(analysis, folder, config):
    """Load best cuts from combined (all-samples) optimization. Not sample-specific.

    Returns dict: {energy: {NHits, AdjCl, OpHits, Score, ...}}
    """
    dir_name = _ANALYSIS_LOCAL_DIR.get(analysis)
    if not dir_name:
        rprint(f"[yellow][WARNING][/yellow] _load_best_cuts: unknown analysis '{analysis}' — no best cuts loaded")
        return {}
    path = f"{root}/config/{config}/{dir_name}/{folder.lower()}/{config}_highest_{analysis}.json"
    if not os.path.exists(path):
        rprint(f"[yellow][WARNING][/yellow] _load_best_cuts: file not found — {path}")
        return {}
    d = json.load(open(path))
    # JSON structure: {config: {energy: {NHits, AdjCl, OpHits, ...}}}
    result = d.get(config, {})
    if not result:
        rprint(f"[yellow][WARNING][/yellow] _load_best_cuts: JSON exists but no entry for {config} — {path}")
    return result



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
parser.add_argument("--export_fiducial", action=argparse.BooleanOptionalAction, default=False, help="Export FiducializationMask per analysis (surface+spatial mask at best fiducial, before quality cuts)")
parser.add_argument("--skip_scan", action=argparse.BooleanOptionalAction, default=False, help="Skip cut scan after exporting raw arrays and fiducial mask (fast first-pass mode)")
parser.add_argument("--best_cuts_only", action=argparse.BooleanOptionalAction, default=False, help="Scan only the best-cut point(s) loaded from JSON; save AnalysisMask and skip DataFrame writes (Pass 3 / post-analysis export)")
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

for path in [save_path, data_path, export_path]:
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
            "DEFAULT_SIGNAL_NADIR": ["mean", "day", "night"],
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
        if args.export_raw and not args.best_cuts_only:
            save_pkl(run["Reco"]["SignalParticleK"], export_path, config, name, subfolder=args.folder.lower(), filename=f"AnalysisEnergy_{energy}_Ref", rm=user_input["rewrite"], debug=args.export_raw)
            save_pkl(run["Reco"][energy], export_path, config, name, subfolder=args.folder.lower(), filename=f"AnalysisData_{energy}_Ref", rm=user_input["rewrite"], debug=args.export_raw)
            save_pkl(run["Reco"]["SignalParticleWeight"], export_path, config, name, subfolder=args.folder.lower(), filename=f"AnalysisWeights_{energy}_Ref", rm=user_input["rewrite"], debug=args.export_raw)

        plot_lists = {analysis: [] for analysis in args.analysis}
        fiducials_by_analysis = {analysis: get_best_fiducial(fiducials, config, energy, analysis) for analysis in args.analysis}
        band_fiducials_by_analysis = {analysis: get_best_fiducial_bands(fiducials, config, energy, analysis) for analysis in args.analysis}
        analysis_cache = {}
        _prev_cut = None

        # Pre-compute spatial masks per analysis — independent of NHits/OpHits/AdjCl.
        # Avoids recomputing the expensive spatial mask N_cuts times.
        _spatial_masks = {}
        for _an in args.analysis:
            _fid = fiducials_by_analysis[_an]
            _bfid = band_fiducials_by_analysis[_an]
            if _bfid and energy:
                _spatial_masks[_an] = np.asarray(build_energy_band_spatial_mask(
                    run, config, detector_x, detector_y, info, args.folder, _fid, _bfid, energy,
                ), dtype=bool)
            else:
                _spatial_masks[_an] = np.asarray(build_fiducial_spatial_mask(
                    run, config, detector_x, detector_y, info, args.folder, _fid
                ), dtype=bool)

        # Surface cut and reco arrays extracted once to avoid per-iteration dict lookups.
        _surface_mask = np.asarray(
            (run["Reco"]["SignalParticleSurface"] >= 0) & (run["Reco"]["SignalParticleSurface"] < 3)
            if is_surface_background(str(root), name)
            else np.ones(len(run["Reco"]["Event"]), dtype=bool),
            dtype=bool,
        )

        # ── First-pass export: FiducializationMask per analysis ──────────────
        # Surface + spatial mask at best fiducial; no quality cuts applied.
        if args.export_fiducial:
            for _an in args.analysis:
                _fid_mask = _surface_mask & _spatial_masks[_an]
                save_pkl(
                    np.asarray(_fid_mask), export_path, config, name,
                    subfolder=args.folder.lower(),
                    filename=f"FiducializationMask_{energy}_{_an.upper()}",
                    rm=user_input["rewrite"], debug=args.debug,
                )
        if args.skip_scan:
            continue

        _reco_nhits    = run["Reco"]["NHits"]
        _reco_adjcl    = run["Reco"]["AdjClNum"]
        _reco_op_plane = run["Reco"]["MatchedOpFlashPlane"]
        _reco_op_pe    = run["Reco"]["MatchedOpFlashPE"]
        _reco_op_nhits = run["Reco"]["MatchedOpFlashNHits"]
        _reco_energy_arr = run["Reco"][energy]
        _true_energy_arr = run["Reco"]["SignalParticleK"]
        _n_bins = len(true_energy_edges) - 1

        # Track last quality mask to recompute only when cut values change.
        _quality_mask_cut: tuple = ()
        _quality_mask: np.ndarray = np.ones(0, dtype=bool)

        best_cuts_by_analysis = {}
        if args.nhits is None and args.ophits is None and args.adjcls is None and (args.export_raw or args.best_cuts_only):
            for analysis in args.analysis:
                # Best cuts only exist for Sensitivity (04_best_cuts.py)
                # DayNight/HEP don't have best cuts JSON files
                if analysis == "Sensitivity":
                    cuts = _load_best_cuts(analysis, args.folder, config)
                    best_cuts_by_analysis[analysis] = cuts.get(energy, {})

        # Build scan ranges: full grid by default; reduced to best-cut tuples for Pass 3.
        if args.best_cuts_only:
            _bc_nhits  = sorted({int(bc["NHits"])  for bc in best_cuts_by_analysis.values() if bc and "NHits"  in bc})
            _bc_ophits = sorted({int(bc["OpHits"]) for bc in best_cuts_by_analysis.values() if bc and "OpHits" in bc})
            _bc_adjcls = sorted({int(bc["AdjCl"])  for bc in best_cuts_by_analysis.values() if bc and "AdjCl"  in bc})
            if not _bc_nhits:
                rprint(
                    f"[yellow][WARNING][/yellow] --best_cuts_only: no best cuts JSON found for "
                    f"{config}/{name}/{energy} — skip"
                )
                continue
            _nhits_scan, _ophits_scan, _adjcls_scan = _bc_nhits, _bc_ophits, _bc_adjcls
        elif args.nhits is not None:
            _nhits_scan  = [args.nhits]
            _ophits_scan = [args.ophits]
            _adjcls_scan = [args.adjcls]
        else:
            _nhits_scan  = nhits
            _ophits_scan = nhits[3:]
            _adjcls_scan = nhits[::-1]

        _scan_total = (
            len(_nhits_scan) * len(_ophits_scan) * len(_adjcls_scan)
            * (3 if "marley" in name else 1)
        )

        for _cut_idx, (this_nhit, this_ophit, this_adjcl, (weight, weight_labels, color)) in enumerate(track(
            product(
                _nhits_scan,
                _ophits_scan,
                _adjcls_scan,
                zip(user_input["weights"][name], user_input["weight_labels"][name], user_input["colors"][name]),
            ),
            total=_scan_total,
            description=f"Cut 1/{_scan_total} for {energy}...",
        ), 1):
            _cur_cut = (this_nhit, this_ophit, this_adjcl)
            # Recompute quality mask only when cut values change (shared across analyses).
            if _cur_cut != _quality_mask_cut:
                _quality_mask = (
                    _surface_mask
                    & (_reco_nhits    > this_nhit - 1)
                    & (_reco_adjcl    < this_adjcl)
                    & (_reco_op_plane == 0)
                    & (_reco_op_pe    > 0)
                    & (_reco_op_nhits > this_ophit - 1)
                )
                _quality_mask_cut = _cur_cut
            if _prev_cut is not None and _cur_cut != _prev_cut:
                for _a in args.analysis:
                    analysis_cache.pop((_a, *_prev_cut), None)
            _prev_cut = _cur_cut

            for analysis in args.analysis:
                fiducial = fiducials_by_analysis[analysis]
                cache_key = (analysis, this_nhit, this_ophit, this_adjcl)
                if cache_key not in analysis_cache:
                    mask = _quality_mask & _spatial_masks[analysis]
                    sel_reco = _reco_energy_arr[mask]
                    sel_true = _true_energy_arr[mask]
                    # Pre-compute bin indices once; reuse with bincount for all subsequent
                    # histogram calls — avoids repeated bin-search inside the osc/weight loop.
                    reco_bin_idx = np.digitize(sel_reco, true_energy_edges) - 1
                    true_bin_idx = np.digitize(sel_true, true_energy_edges) - 1
                    reco_bin_valid = (reco_bin_idx >= 0) & (reco_bin_idx < _n_bins)
                    true_bin_valid = (true_bin_idx >= 0) & (true_bin_idx < _n_bins)
                    mc_counts = np.bincount(reco_bin_idx[reco_bin_valid], minlength=_n_bins)
                    mc_filter = mc_counts >= args.mc_filter_threshold
                    if args.debug and np.any(~mc_filter):
                        rprint(
                            f"[yellow][WARNING][/yellow] {np.sum(~mc_filter)} bins with MC < "
                            f"{args.mc_filter_threshold} zeroed for {energy} "
                            f"NHits={this_nhit} OpHits={this_ophit} AdjCl={this_adjcl}"
                        )
                    h_rel_error = np.zeros_like(mc_counts, dtype=float)
                    non_zero_bins = mc_counts > 0
                    h_rel_error[non_zero_bins] = 1.0 / np.sqrt(mc_counts[non_zero_bins])

                    analysis_cache[cache_key] = {
                        "mask": mask,
                        "reco_bin_idx": reco_bin_idx,
                        "reco_bin_valid": reco_bin_valid,
                        "true_bin_idx": true_bin_idx,
                        "true_bin_valid": true_bin_valid,
                        "mc_counts": mc_counts,
                        "mc_filter": mc_filter,
                        "h_rel_error": h_rel_error,
                    }

                cached = analysis_cache[cache_key]
                mask = cached["mask"]

                # Slice bin indices once outside the osc loop (shared across all weights/oscs).
                _reco_bidx  = cached["reco_bin_idx"][cached["reco_bin_valid"]]
                _true_bidx  = cached["true_bin_idx"][cached["true_bin_valid"]]
                _reco_bvalid = cached["reco_bin_valid"]
                _true_bvalid = cached["true_bin_valid"]
                for osc, mean, mean_label in zip(
                    ["Truth", "Osc", "Osc", "Osc"] if "marley" in name else ["Truth"],
                    ["Mean", "Day", "Night", "Mean"] if "marley" in name else ["Mean"],
                    ["", "OscDay", "OscNight", "OscMean"] if "marley" in name else [""],
                ):
                    selected_weights = run["Reco"][f"{weight}{mean_label}"][mask]
                    h = np.bincount(
                        _reco_bidx, weights=selected_weights[_reco_bvalid], minlength=_n_bins
                    ).astype(float)
                    h_true = np.bincount(
                        _true_bidx, weights=selected_weights[_true_bvalid], minlength=_n_bins
                    ).astype(float)
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
                        save_pkl(np.asarray(mask), export_path, config, name, subfolder=args.folder.lower(), filename=f"AnalysisMask_{energy}_{analysis_key}_NHits{this_nhit}_OpHits{this_ophit}_AdjCl{this_adjcl}", rm=user_input["rewrite"], debug=user_input["debug"])
                        save_pkl(run["Reco"]["SignalParticleK"][mask], export_path, config, name, subfolder=args.folder.lower(), filename=f"AnalysisEnergy_{energy}_{analysis_key}_NHits{this_nhit}_OpHits{this_ophit}_AdjCl{this_adjcl}", rm=user_input["rewrite"], debug=user_input["debug"])
                        save_pkl(run["Reco"][energy][mask], export_path, config, name, subfolder=args.folder.lower(), filename=f"AnalysisData_{energy}_{analysis_key}_NHits{this_nhit}_OpHits{this_ophit}_AdjCl{this_adjcl}", rm=user_input["rewrite"], debug=user_input["debug"])
                        save_pkl(run["Reco"][weight][mask], export_path, config, name, subfolder=args.folder.lower(), filename=f"AnalysisWeights_{energy}_{analysis_key}_NHits{this_nhit}_OpHits{this_ophit}_AdjCl{this_adjcl}", rm=user_input["rewrite"], debug=user_input["debug"])

                    cut_impact = build_cut_impact(run, args, config, info, fiducial, detector_x, detector_y, this_nhit, this_ophit, this_adjcl, name)
                    cut_dir = f"{export_path}/{config}/{args.name}/{args.folder.lower()}"
                    if not os.path.exists(cut_dir):
                        os.makedirs(cut_dir)
                    cut_path = f"{cut_dir}/analysis_cuts_{analysis_key}.json"
                    merge_and_write_json(cut_path, cut_impact)

                if args.export_raw and weight == "SignalParticleWeight" and best_cuts_by_analysis and args.best_cuts_only:
                    bc = best_cuts_by_analysis.get(analysis, {})
                    if bc and int(bc.get("NHits", -1)) == this_nhit and int(bc.get("OpHits", -1)) == this_ophit and int(bc.get("AdjCl", -1)) == this_adjcl:
                        analysis_key = analysis.upper()
                        save_pkl(np.asarray(mask), export_path, config, name, subfolder=args.folder.lower(), filename=f"AnalysisMask_{energy}_{analysis_key}_NHits{this_nhit}_OpHits{this_ophit}_AdjCl{this_adjcl}", rm=user_input["rewrite"], debug=user_input["debug"])
                        save_pkl(run["Reco"]["SignalParticleK"][mask], export_path, config, name, subfolder=args.folder.lower(), filename=f"AnalysisEnergy_{energy}_{analysis_key}_NHits{this_nhit}_OpHits{this_ophit}_AdjCl{this_adjcl}", rm=user_input["rewrite"], debug=user_input["debug"])
                        save_pkl(run["Reco"][energy][mask], export_path, config, name, subfolder=args.folder.lower(), filename=f"AnalysisData_{energy}_{analysis_key}_NHits{this_nhit}_OpHits{this_ophit}_AdjCl{this_adjcl}", rm=user_input["rewrite"], debug=user_input["debug"])
                        save_pkl(run["Reco"][weight][mask], export_path, config, name, subfolder=args.folder.lower(), filename=f"AnalysisWeights_{energy}_{analysis_key}_NHits{this_nhit}_OpHits{this_ophit}_AdjCl{this_adjcl}", rm=user_input["rewrite"], debug=user_input["debug"])

        if _prev_cut is not None:
            for _a in args.analysis:
                analysis_cache.pop((_a, *_prev_cut), None)

        if not args.best_cuts_only:
            # All analyses use identical rebinning (0-31 MeV, 1 MeV bins)
            # Cache rebin result by DataFrame id to avoid duplicate computation
            rebin_cache = {}
            rebin_dict = {"DayNight": daynight_rebin, "Sensitivity": sensitivity_rebin, "HEP": hep_rebin}
            for analysis in args.analysis:
                df = pd.DataFrame(plot_lists[analysis])
                if df.empty:
                    continue
                save_df(df, data_path, config=config, name=name, subfolder=args.folder.lower(), filename=f"{energy}_{analysis}_AnalysisData", rm=user_input["rewrite"], debug=user_input["debug"])

                # Cache check: if this exact DataFrame was already rebinned, reuse result
                df_id = id(df)
                if df_id not in rebin_cache:
                    rebin_cache[df_id] = rebin_df_columns(df, rebin_dict[analysis], "Energy", "Counts", "Counts/Energy", "Error", "MCCounts")
                rebin_df = rebin_cache[df_id]

                save_df(rebin_df, f"{info['PATH']}/{user_input['directory'][name]}/{analysis.upper()}", config=config, name=name, filename=f"{energy}_Rebin", rm=user_input['rewrite'], debug=user_input['debug'])
