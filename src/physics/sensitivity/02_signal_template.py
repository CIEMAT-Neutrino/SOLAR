import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *
from lib.fiducial import _DEFAULT_POS_KEYS, get_truth_pos_keys


TEMPLATE_NORMALIZATION_FILE = "TEMPLATE_NORMALIZATION.json"
TEMPLATE_NORMALIZATION_VERSION = 2   # v2 = per-year (mass x rate); v1 = pre-scaled by exposure


def write_template_normalization(template_dir: str, detector_mass_kT: float) -> None:
    """Stamp a template directory as per-year so consumers can refuse stale v1 templates."""
    os.makedirs(template_dir, exist_ok=True)
    _marker = os.path.join(template_dir, TEMPLATE_NORMALIZATION_FILE)
    # PNFS/dCache is write-once: opening an existing file for writing raises
    # "Operation not permitted". Remove first, exactly as prepare_file_save() does.
    if os.path.exists(_marker):
        os.remove(_marker)
    with open(_marker, "w") as fh:
        json.dump({
            "version": TEMPLATE_NORMALIZATION_VERSION,
            "per_year": True,
            "units": "detector_mass_kT * rate (counts per year)",
            "detector_mass_kT": float(detector_mass_kT),
            "note": "Multiply by exposure_yr at load time (06_significance.py).",
        }, fh, indent=2)

from lib.oscillation import get_oscillation_datafiles
from lib.template_guards import clean_and_validate_template, write_template_sampling_marker

save_path = f"{root}/output/images/analysis/sensitivity/templates"

if not os.path.exists(save_path):
    os.makedirs(save_path)

# Define flags for the analysis config and name with the python parser
parser = argparse.ArgumentParser(
    description="Plot the energy distribution of the particles"
)
parser.add_argument(
    "--reference",
    type=str,
    help="The name of the reference analysis",
    choices=["DayNight", "SENSITIVITY", "HEP"],
    default="SENSITIVITY",
)
parser.add_argument(
    "--config",
    type=str,
    help="The configuration to load",
    default="hd_1x2x6_centralAPA",
)
parser.add_argument(
    "--signal", type=str, help="The name of the configuration", default="marley"
)
parser.add_argument(
    "--folder",
    type=str,
    help="The name of the results folder",
    default="Nominal",
    choices=["Reduced", "Truncated", "Nominal"],
)
parser.add_argument(
    "--signal_uncertainty",
    type=float,
    help="The signal uncertainty for the analysis",
    default=0.04,
)
parser.add_argument(
    "--background_uncertainty",
    type=float,
    help="The background uncertainty for the analysis",
    default=0.02,
)
parser.add_argument(
    "--exposure",
    type=float,
    help="Exposure in years. Default from ANALYSIS_EXPOSURES['SENSITIVITY']['PRIMARY'] in config/analysis/config.json.",
    default=get_analysis_exposure(str(root), "Sensitivity"),
)
parser.add_argument(
    "--energy",
    type=str,
    help="The energy for the analysis",
    choices=[
        "SignalParticleK", "MainK",
        "ClusterEnergy",
        "TotalEnergy",
        "SelectedEnergy",
        "SolarEnergy",
    ],
    default="SolarEnergy",
)
parser.add_argument(
    "--nhits", type=int, help="The nhit cut for the analysis", default=None
)
parser.add_argument(
    "--ophits", type=int, help="The ophit cut for the analysis", default=None
)
parser.add_argument(
    "--adjcls", type=int, help="The adjacent cluster cut for the analysis", default=None
)
parser.add_argument("--test", action=argparse.BooleanOptionalAction)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument(
    "--oscillation_backend",
    type=str,
    choices=["file", "prob3", "nufast"],
    default="nufast",
    help="Oscillation backend for signal template convolution. 'file' uses pre-computed pkl files; 'prob3'/'nufast' compute on-the-fly.",
)
parser.add_argument(
    "--scan_mode",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Restrict oscillation grid to solar+reactor reference points only. "
         "Fast path for cut optimisation; use without --scan_mode for the full analysis grid.",
)
parser.add_argument(
    "--cuts",
    type=str,
    default=None,
    help='JSON list of cut dicts [{"NHits":N,"AdjCl":A,"OpHits":O}, ...]. '
         "When provided, processes all listed cuts in a single invocation.",
)
parser.add_argument("--charge_threshold", type=float, default=0,
    help="Charge threshold Q (ADC). When >0, applies Charge>Q cut to signal template events.")
parser.add_argument("--study_label", type=str, default=None,
    help="Tag appended to template subfolder to isolate charge study variants.")

parser.add_argument(
    "--membrane_veto",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "Accept only cathode/APA optical matches (QUALITY_CUTS.OPFLASH_PLANE, plane 0). "
        "This is the default. --no-membrane_veto additionally accepts membrane and endcap "
        "matches (VD planes 1-4), which HD never produces; unmatched clusters are rejected "
        "by the MatchedOpFlashPE > 0 requirement either way. Used by the membrane_veto study."
    ),
)
parser.add_argument(
    "--flyweight",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Enable flyweight mode: save only unoscillated base templates in coarse (30-bin) sensitivity bins "
        "instead of computing and saving ~14k oscillation templates. Skips Phase 2 (oscillation loop)."
    ),
)
parser.add_argument(
    "--truth_fiducial",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Use true MC particle coordinates (SignalParticleX/Y/Z for marley, EndX/Y/Z for gamma, "
        "MainX/Y/Z for neutron/radiological) instead of reco flash-matched coordinates (RecoX/Y/Z). "
        "Output templates are saved with a '_fiduc_truth' suffix."
    ),
)

args = parser.parse_args()
_ctx = study_context(args, analysis="Sensitivity")
_study_suffix = _ctx.study_suffix
_template_suffix = _ctx.template_suffix

smoothing_config = get_smoothing_config(
    str(root), analysis_name="SENSITIVITY", dimensions="2d", stage="significance"
)
smoothing_config = dict(smoothing_config)
smoothing_config["params"] = dict(smoothing_config.get("params", {}))
smoothing_config["params"]["sigma_y"] = 0.0
smoothing_info = smoothing_metadata(smoothing_config)


def _resolve_signal_smoothing_config(config: dict):
    """Return effective smoothing config for sensitivity signal templates."""
    signal_labels = ["signal", "solar", "8B", "hep", "marley"]
    component_configs = [get_component_smoothing_config(config, label) for label in signal_labels]
    active_configs = [item for item in component_configs if str(item.get("method", "none")).lower() != "none"]

    if active_configs:
        return active_configs[0], signal_labels

    fallback = get_component_smoothing_config(config, "signal")
    fallback["enabled"] = False
    fallback["method"] = "none"
    return fallback, signal_labels


signal_smoothing_config, signal_smoothing_labels = _resolve_signal_smoothing_config(smoothing_config)
signal_smoothing_active = str(signal_smoothing_config.get("method", "none")).lower() != "none"

if signal_smoothing_active:
    rprint(
        "[yellow][WARNING][/yellow] Signal smoothing is ACTIVE in sensitivity/02_signal_template.py. "
        "This can wash out solar-neutrino wiggles relevant for sensitivity significance. "
        f"method={signal_smoothing_config.get('method')} params={signal_smoothing_config.get('params', {})} "
        f"mode={smoothing_info.get('SmoothingComponentMode', 'all')} "
        f"components={smoothing_info.get('SmoothingComponents', [])} "
        f"checked_labels={signal_smoothing_labels}"
    )
elif args.debug:
    rprint(
        "[cyan][INFO][/cyan] Signal smoothing is disabled for sensitivity templates; "
        "using raw oscillated signal to preserve fine-structure wiggles."
    )


def _load_best_cut_map(info: dict, args):
    _suffix = f"_{args.study_label}" if getattr(args, 'study_label', None) else ""
    candidates = list(dict.fromkeys(["SENSITIVITY", args.reference.upper()]))
    tried = []
    
    # Standard location: {PATH}/SENSITIVITY/{config}/{name}/{folder}/...
    for analysis in candidates if not _suffix else ["SENSITIVITY"]:
        for suffix in ([_suffix] if _suffix else [""]):
            filepath = (
                f"{info['PATH']}/SENSITIVITY/{args.config}/{args.signal}/{args.folder.lower()}/"
                f"{args.config}_{args.signal}_highest_{analysis}{suffix}.pkl"
            )
            tried.append(filepath)
            rprint(f"[cyan][INFO][/cyan] Checking best-cut map at: {filepath}")
            if os.path.exists(filepath):
                if args.debug:
                    rprint(f"[cyan][INFO][/cyan] Using best-cut map from {analysis}{suffix}")
                rprint(f"[cyan][INFO][/cyan] Loading best-cut map from: {filepath}")
                return pickle.load(open(filepath, "rb"))

    rprint(
        "[yellow][WARNING][/yellow] Unable to load any best-cut map. Checked:\n"
        + "\n".join(tried)
    )
    return None


def _validate_signal_cut_consistency(cuts_to_process: list, info: dict, args) -> None:
    """
    Validate that signal templates will be generated for the correct cut.
    
    For SIGNAL templates, we expect to use the best cut from the best-cut file.
    This function checks that if we loaded a best-cut file, the cuts we're processing
    match what's in that file.
    """
    # Only validate if we have cuts to process
    if not cuts_to_process:
        return

    # Explicit --nhits/--adjcls/--ophits: the caller chose the cut, so no best-cut map is needed.
    # Study variants run with --skip_best_cuts never write a labeled map, and 06_significance.py
    # uses the same explicit cut, so requiring (or matching) a map here only blocks those runs.
    if args.cuts is None and None not in (args.nhits, args.adjcls, args.ophits):
        rprint(
            f"[green][CUT CONSISTENCY][/green] Signal templates use the explicit cut "
            f"NHits{args.nhits} AdjCl{args.adjcls} OpHits{args.ophits}"
        )
        return

    # If we have a study_label, we MUST have a best-cut file
    if args.study_label:
        loaded = _load_best_cut_map(info, args)
        if loaded is None:
            raise FileNotFoundError(
                f"[CRITICAL] study_label '{args.study_label}' is set but no best-cut map found. "
                f"Signal templates require a best cut to be selected first. "
                f"Run 04_best_cuts.py before generating signal templates."
            )
        
        # Check that the cuts we're processing match the best-cut file
        key = (args.config, args.signal, args.energy)
        if key in loaded:
            selected = loaded[key]
            best_nhits = int(selected["NHits"])
            best_adjcl = int(selected["AdjCl"])
            best_ophits = int(selected["OpHits"])
            
            for i, cut in enumerate(cuts_to_process):
                if (int(cut["NHits"]) != best_nhits or 
                    int(cut["AdjCl"]) != best_adjcl or 
                    int(cut["OpHits"]) != best_ophits):
                    raise ValueError(
                        f"[CRITICAL] Signal template cut mismatch! "
                        f"Best-cut file: NHits{best_nhits} AdjCl{best_adjcl} OpHits{best_ophits} | "
                        f"Processing: NHits{cut['NHits']} AdjCl{cut['AdjCl']} OpHits{cut['OpHits']} "
                        f"(index {i}). "
                        f"Signal templates must match the best-cut selection."
                    )
            
            rprint(
                f"[green][CUT CONSISTENCY][/green] Signal templates will use best cut: "
                f"NHits{best_nhits} AdjCl{best_adjcl} OpHits{best_ophits}"
            )
        else:
            raise KeyError(
                f"[CRITICAL] No entry for {key} in best-cut map. "
                f"The best-cut file may be for a different config/name/energy."
            )
    else:
        # No study_label - we may be using default cuts or best-cut file
        loaded = _load_best_cut_map(info, args)
        if loaded is not None:
            # Best-cut file exists - verify our cuts match
            key = (args.config, args.signal, args.energy)
            if key in loaded:
                selected = loaded[key]
                best_nhits = int(selected["NHits"])
                best_adjcl = int(selected["AdjCl"])
                best_ophits = int(selected["OpHits"])
                
                for i, cut in enumerate(cuts_to_process):
                    if (int(cut["NHits"]) != best_nhits or 
                        int(cut["AdjCl"]) != best_adjcl or 
                        int(cut["OpHits"]) != best_ophits):
                        rprint(
                            f"[red][CRITICAL WARNING][/red] Signal template cut mismatch! "
                            f"Best-cut file: NHits{best_nhits} AdjCl{best_adjcl} OpHits{best_ophits} | "
                            f"Processing: NHits{cut['NHits']} AdjCl{cut['AdjCl']} OpHits{cut['OpHits']} "
                            f"(index {i}). "
                            f"This may cause a mismatch with background templates. "
                            f"Use --cuts to specify the correct cut or regenerate with --rewrite."
                        )
                
                rprint(
                    f"[green][CUT CONSISTENCY][/green] Signal templates will use best cut: "
                    f"NHits{best_nhits} AdjCl{best_adjcl} OpHits{best_ophits}"
                )

folder = args.folder
configs = {args.config: [args.signal]}

for path in [save_path]:
    if not os.path.exists(f"{path}/{args.folder.lower()}"):
        os.makedirs(f"{path}/{args.folder.lower()}")

run, output = load_multi(
    configs,
    preset="SIGNIFICANCE",
    branches={"Config": ["Geometry"]},
    debug=args.debug,
)
if args.debug:
    rprint(output)
run = compute_reco_workflow(
    run,
    configs,
    params={
        "DEFAULT_SIGNAL_WEIGHT": ["truth", "osc"],
        "DEFAULT_SIGNAL_NADIR": ["mean", "day", "night"],
        "PARTICLE_TYPE": "signal",
        "PARTICLE_WEIGHTING": "volume",
        "OSCILLATION_BACKEND": args.oscillation_backend,
    } if "marley" in args.signal else {"PARTICLE_TYPE": "background", "PARTICLE_WEIGHTING": "histogram"},
    workflow="SIGNIFICANCE",
    rm_branches=False,
    debug=args.debug)

for config in configs:
    info = json.loads(
        open(f"{root}/config/{config}/{config}_config.json").read()
    )
    _fiducials_stem = "BestFiducials_fiduc_truth" if args.truth_fiducial else "BestFiducials"
    fiducials = json.loads(open(f"{root}/config/analysis/fiducial/{args.folder.lower()}/{_fiducials_stem}.json").read())
    selected_fiducial = get_best_fiducial(fiducials, config, args.energy, "SENSITIVITY")
    selected_fiducial_bands = get_best_fiducial_bands(fiducials, config, args.energy, "SENSITIVITY")
    analysis_info = load_analysis_info(str(root))
    info = json.loads(open(f"{root}/config/{config}/{config}_config.json").read())
    detector_mass = get_full_detector_mass(config, info)

    _solar_dm2 = analysis_info["SOLAR_DM2"]
    _react_dm2 = analysis_info["REACT_DM2"]
    _sin13     = analysis_info["SIN13"]
    _sin12     = analysis_info["SIN12"]

    if args.scan_mode:
        _ref_dm2s  = list(dict.fromkeys([_solar_dm2, _react_dm2]))
        dm2_list   = _ref_dm2s
        sin13_list = [_sin13] * len(_ref_dm2s)
        sin12_list = [_sin12] * len(_ref_dm2s)
    elif args.oscillation_backend == "file":
        (dm2_list, sin13_list, sin12_list) = get_oscillation_datafiles(
            dm2=None,
            sin13=None,
            sin12=None,
            path=f"{info['PATH']}/data/OSCILLATION/pkl/rebin/",
            ext="pkl",
            auto=args.test == False,
            debug=args.debug,
        )
    else:
        # nufast/prob3 full mode: build grid from OSCILLATION_GRID config (no file dependency)
        (dm2_list, sin13_list, sin12_list) = get_oscillation_datafiles(
            backend=args.oscillation_backend,
            debug=args.debug,
        )

    detector_x = info["DETECTOR_SIZE_X"] + 2 * info["DETECTOR_GAP_X"]
    detector_y = info["DETECTOR_SIZE_Y"] + 2 * info["DETECTOR_GAP_Y"]
    detector_z = info["DETECTOR_SIZE_Z"] + 2 * info["DETECTOR_GAP_Z"]

    energy = args.energy

    # Build cuts_to_process
    if args.cuts is not None:
        cuts_to_process = json.loads(args.cuts)
    elif args.nhits is not None and args.adjcls is not None and args.ophits is not None:
        cuts_to_process = [{"NHits": args.nhits, "AdjCl": args.adjcls, "OpHits": args.ophits}]
    else:
        loaded = _load_best_cut_map(info, args)
        if loaded is not None:
            cuts_to_process = [
                {"NHits": int(v["NHits"]), "AdjCl": int(v["AdjCl"]), "OpHits": int(v["OpHits"])}
                for v in loaded.values() if v is not None
            ]
        elif args.study_label:
            raise FileNotFoundError(
                f"Missing best-cut map for study '{args.study_label}'; refusing to use default cuts"
            )
        else:
            cuts_to_process = [{"NHits": 4, "AdjCl": 10, "OpHits": 4}]
            rprint("[yellow][WARNING][/yellow] Falling back to default cuts NHits4 AdjCl10 OpHits4")
    
    # =====================================================================
    # CRITICAL: Validate cut consistency before generating signal templates
    # =====================================================================
    _validate_signal_cut_consistency(cuts_to_process, info, args)

    # ── Phase 1: pre-compute per-cut histograms ──────────────────────────────────
    cut_data = []  # (nhits, adjcl, ophits, h, fig, title)
    
    # Determine which binning to use based on flyweight mode
    if args.flyweight:
        # Flyweight mode: use coarse sensitivity bins (30 bins) to avoid rebinning later
        _bin_edges = sensitivity_rebin
        _energy_centers_for_plot = sensitivity_rebin_centers
        rprint(f"[cyan][INFO][/cyan] Flyweight mode: using coarse sensitivity bins ({len(_bin_edges)-1} bins)")
    else:
        # Standard mode: use fine energy bins (120 bins)
        _bin_edges = energy_edges
        _energy_centers_for_plot = energy_centers
    
    for cut in cuts_to_process:
        nhits  = int(cut["NHits"])
        adjcl  = int(cut["AdjCl"])
        ophits = int(cut["OpHits"])

        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=(
                "Unweighted Smearing",
                "Solar Weighted Smearing",
                f"Oscillated ({'Raw' if not signal_smoothing_active else 'Smoothed'})" if not args.flyweight else "Base Template (Coarse Bins)",
            ),
            shared_xaxes=True,
            shared_yaxes=True,
        )

        # Use truth positions for fiducialization when --truth_fiducial is enabled.
        # _DEFAULT_POS_KEYS is private, so the `from lib import *` above does not bring it
        # in — import it explicitly, as 03_analysis.py does for the same pair.
        from lib.fiducial import _DEFAULT_POS_KEYS, get_truth_pos_keys
        sample_key = args.signal.split("_")[0].lower()
        _pos_keys = get_truth_pos_keys(str(root), sample_key) if args.truth_fiducial else _DEFAULT_POS_KEYS
        
        quality_mask = (
            (
                (run["Reco"]["SignalParticleSurface"] >= 0)
                & (run["Reco"]["SignalParticleSurface"] < 3)
                if args.signal.split("_")[0] in ["gamma", "neutron"]
                else np.ones(len(run["Reco"]["NHits"]), dtype=bool)
            )
            & ((run["Reco"]["SignalParticleSurface"] < 3) if (args.folder in ["Reduced", "Truncated"] and args.signal.split("_")[0] in ["gamma", "neutron"]) else np.ones(len(run["Reco"]["NHits"]), dtype=bool))
            & (run["Reco"]["NHits"] > nhits - 1)
            & (run["Reco"]["AdjClNum"] < adjcl)
            & accepted_flash_planes(
                run["Reco"]["MatchedOpFlashPlane"], str(root), args.membrane_veto
            )
            & (run["Reco"]["MatchedOpFlashPE"] > 0)
            & (run["Reco"]["MatchedOpFlashNHits"] > ophits - 1)
            & (run["Reco"]["Charge"] > args.charge_threshold if args.charge_threshold > 0 else np.ones(len(run["Reco"]["NHits"]), dtype=bool))
        )
        spatial_mask = build_energy_band_spatial_mask(
            run, config, detector_x, detector_y, info, args.folder,
            selected_fiducial, selected_fiducial_bands, energy,
            pos_keys=_pos_keys,
        )
        this_filter = np.where(quality_mask & spatial_mask)

        if args.debug:
            print(f"Selected #Events: {len(this_filter[0])} ({len(this_filter[0])/len(run['Reco']['Event'])*100:.2f}%)")

        title = f"{energy} Signal (min #NHits {nhits} / max #AdjClusters {adjcl} / min #OpHits {ophits})"
        h, xedges, yedges = np.histogram2d(
            run["Reco"][f"{energy}"][this_filter],
            run["Reco"]["SignalParticleK"][this_filter],
            bins=(_bin_edges, _bin_edges),
        )
        if args.debug:
            print(f"# of events (counts): {np.sum(h)}")
        fig.add_trace(
            go.Heatmap(z=np.log10(h), x=_energy_centers_for_plot, y=_energy_centers_for_plot, colorscale="Turbo", coloraxis="coloraxis"),
            row=1, col=1,
        )
        for weight in ["B8", "hep", ""]:
            h, xedges, yedges = np.histogram2d(
                run["Reco"][f"{energy}"][this_filter],
                run["Reco"]["SignalParticleK"][this_filter],
                bins=(_bin_edges, _bin_edges),
                weights=run["Reco"][f"SignalParticleWeight{weight.lower()}"][this_filter],
            )
            if args.debug:
                print(f"# of weighted events (counts) {(weight if weight != '' else 'solar')}: {np.sum(h):.2f}")

        # NOTE: the '<1 expected event' mask is exposure dependent, so it is NOT
        # applied here — templates are stored per-year and 06_significance.py
        # applies the mask after scaling to the requested exposure.
        h = np.where(np.isfinite(h), h, np.nan)
        fig.add_trace(
            go.Heatmap(z=np.log10(h), x=_energy_centers_for_plot, y=_energy_centers_for_plot, colorscale="Turbo", coloraxis="coloraxis"),
            row=1, col=2,
        )
        h = np.nan_to_num(h, nan=0.0)

        cut_data.append((nhits, adjcl, ophits, h, fig, title))
    
    # ── Phase 1.5: Flyweight mode - save base templates and skip oscillation loop ────
    if args.flyweight:
        rprint(f"[cyan][INFO][/cyan] Flyweight mode: saving {len(cut_data)} base template(s) in coarse bins...")
        for nhits, adjcl, ophits, h, fig, title in cut_data:
            # Apply stability guards to base histogram before scaling
            h_clean = clean_and_validate_template(
                h,
                label=f"Signal base template {args.config} {args.signal} {energy} NHits{nhits} AdjCl{adjcl} OpHits{ophits}",
                debug=args.debug,
            )
            # Save the base template (unoscillated histogram in coarse bins)
            # This is the solar-weighted histogram (weight="")
            save_pkl(
                # per-year template: detector_mass_kT x rate
                detector_mass * h_clean,
                f"{info['PATH']}/SENSITIVITY",
                config=args.config,
                name=args.signal,
                subfolder=f"{folder.lower()}/{energy}{_template_suffix}",
                filename=f"NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_BASE",
                rm=args.rewrite,
                debug=args.debug,
            )
            write_template_normalization(
                f"{info['PATH']}/SENSITIVITY/{args.config}/{args.signal}/{folder.lower()}/{energy}{_template_suffix}",
                detector_mass,
            )
            if args.debug:
                rprint(f"[cyan][INFO][/cyan] Saved flyweight base template: NHits{nhits} AdjCl{adjcl} OpHits{ophits}")
        
        # Skip Phase 2 (oscillation loop) in flyweight mode
        rprint(f"[cyan][INFO][/cyan] Flyweight mode: skipping Phase 2 (oscillation loop)")
        # Still need to save figures - jump to Phase 3
        goto_phase_3 = True
    else:
        goto_phase_3 = False

    # ── Phase 2: oscillation loop (outer) × cuts loop (inner) ───────────────────
    # P_ee is integrated over OSC_NADIR_OVERSAMPLE nadir sub-bins per row; point-sampling the
    # nadir bin centres aliases Earth regeneration at low dm2 (whole-row chi2 stripes).
    _nadir_oversample = int(analysis_info.get("OSC_NADIR_OVERSAMPLE", 1))
    if not goto_phase_3:
        for dm2, sin13, sin12 in track(
            zip(dm2_list, sin13_list, sin12_list),
            total=len(dm2_list),
            description="Convolving oscillation files...",
        ):
            if args.debug:
                rprint(f"dm2: {dm2:.3e}, sin13: {sin13:.3e}, sin12: {sin12:.3e}")

            if args.oscillation_backend == "file":
                oscillation_df = pd.read_pickle(
                    f"{info['PATH']}/data/OSCILLATION/pkl/rebin/osc_probability_dm2_{dm2:.3e}_sin13_{sin13:.3e}_sin12_{sin12:.3e}.pkl"
                )
            else:
                from lib.oscillation import get_oscillation_map
                osc_map = get_oscillation_map(
                    backend=args.oscillation_backend,
                    dm2=[float(dm2)],
                    sin13=[float(sin13)],
                    sin12=[float(sin12)],
                    output="df",
                    debug=args.debug,
                    nadir_oversample=_nadir_oversample,
                )
                oscillation_df = next(iter(osc_map.values()))

            is_solar_bf = (
                dm2 == analysis_info["SOLAR_DM2"]
                and sin13 == analysis_info["SIN13"]
                and sin12 == analysis_info["SIN12"]
            )

            for nhits, adjcl, ophits, h, fig, title in cut_data:
                convolved = np.dot(oscillation_df.values, h.T)
                rebin_x, rebin_y, rebin_z, rebin_z_per_x = rebin_hist2d(
                    energy_centers,
                    np.asarray(list(oscillation_df.index)),
                    convolved,
                    sensitivity_rebin,  # type: ignore[arg-type]
                )

                if args.debug:
                    rprint(
                        f"[cyan][INFO][/cyan] Saving signal template NHits{nhits} AdjCl{adjcl} OpHits{ophits} "
                        f"dm2={dm2:.3e} sin13={sin13:.3e} sin12={sin12:.3e}"
                    )
                # Apply stability guards before saving
                _template_to_save = clean_and_validate_template(
                    detector_mass * rebin_z,
                    label=f"Signal template {args.config} {args.signal} {energy} NHits{nhits} AdjCl{adjcl} OpHits{ophits} dm2={dm2:.3e}",
                    debug=args.debug,
                )
                save_pkl(
                    # per-year template: detector_mass_kT x rate. 06_significance.py
                    # multiplies by the requested exposure at load time.
                    _template_to_save,
                    f"{info['PATH']}/SENSITIVITY",
                    config=args.config,
                    name=args.signal,
                    subfolder=f"{folder.lower()}/{energy}{_template_suffix}",
                    filename=f"NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_dm2_{dm2:.3e}_sin13_{sin13:.3e}_sin12_{sin12:.3e}",
                    rm=args.rewrite,
                    debug=args.debug,
                )
                write_template_normalization(
                    f"{info['PATH']}/SENSITIVITY/{args.config}/{args.signal}/"
                    f"{folder.lower()}/{energy}{_template_suffix}",
                    detector_mass,
                )

                if is_solar_bf:
                    if args.debug:
                        rprint(f"# of events (absolute counts at {args.exposure} yr × {detector_mass:.2f} kT): {np.sum(convolved):.2f}")

                    _osc_fig = make_subplots(rows=1, cols=1)
                    _osc_fig.add_trace(go.Heatmap(
                        z=oscillation_df.values,
                        x=[float(c) for c in oscillation_df.columns],
                        y=list(oscillation_df.index),
                        colorscale="Viridis",
                        colorbar=dict(title="P(νe→νe)"),
                    ), row=1, col=1)
                    _osc_fig = format_coustom_plotly(
                        _osc_fig,
                        title=f"Solar Oscillogram {config} (Δm²={dm2:.2e} eV², sin²θ₁₂={sin12:.3f})",
                    )
                    _osc_fig.update_xaxes(title="True Neutrino Energy (MeV)")
                    _osc_fig.update_yaxes(title="cos(η) Zenith Angle")
                    save_figure(
                        _osc_fig, save_path, config=args.config, name=args.signal, subfolder=args.folder.lower(),
                        filename=f"Oscillogram_{energy}", rm=args.rewrite, debug=args.plot,
                    )

                    _signal_1d = args.exposure * detector_mass * np.sum(rebin_z, axis=0)
                    _sig_fig = make_subplots(rows=1, cols=1)
                    _sig_fig.add_trace(go.Scatter(
                        x=sensitivity_rebin_centers,
                        y=_signal_1d,
                        mode="lines",
                        fill="tozeroy",
                        line_shape="hvh",
                        name="Solar ν Signal",
                    ), row=1, col=1)
                    _sig_fig = format_coustom_plotly(
                        _sig_fig, title=f"1D Signal Spectrum {config} {energy} ({args.exposure} yr × {detector_mass:.2f} kT)",
                    )
                    _sig_fig.update_xaxes(title="Reconstructed Neutrino Energy (MeV)")
                    # y shows absolute counts (exposure_yr × detector_mass_kT × rate) — not a rate
                    _sig_fig.update_yaxes(title="Events / MeV")
                    save_figure(
                        _sig_fig, save_path, config=args.config, name=args.signal, subfolder=args.folder.lower(),
                        filename=f"Signal1D_{energy}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}", rm=args.rewrite, debug=args.plot,
                    )

                    fig.add_trace(
                        go.Heatmap(
                            z=np.log10(np.where(rebin_z > 0, rebin_z, np.nan)) if not signal_smoothing_active else np.log10(smooth_histogram_with_config(rebin_z, signal_smoothing_config)),
                            x=sensitivity_rebin,
                            y=rebin_y,
                            colorscale="Turbo",
                            coloraxis="coloraxis",
                        ),
                        row=1, col=3,
                    )

        # Stamp each cut only after the whole oscillation loop finished, so a partially
        # regenerated cut is never reported as current to 04_best_cuts.py / 06_significance.py.
        for nhits, adjcl, ophits, h, fig, title in cut_data:
            write_template_sampling_marker(
                f"{info['PATH']}/SENSITIVITY/{args.config}/{args.signal}/{folder.lower()}/{energy}{_template_suffix}",
                args.config, args.signal, nhits, adjcl, ophits,
                scope="scan" if args.scan_mode else "grid",
                backend=args.oscillation_backend,
                nadir_oversample=None if args.oscillation_backend == "file" else _nadir_oversample,
            )

    # ── Phase 3: format and save per-cut figures ─────────────────────────────────
    for nhits, adjcl, ophits, h, fig, title in cut_data:
        fig = format_coustom_plotly(
            fig,
            title=title,
            legend=dict(x=0.5, y=0.99),
            tickformat=(".0f", ".0f"),
            matches=("x", None),
        )
        fig.update_yaxes(title="Reconstructed Neutrino Energy (MeV)", row=1, col=1)
        fig.update_yaxes(title="", row=1, col=2)
        fig.update_xaxes(title="Reconstructed Neutrino Energy (MeV)", row=1, col=3)
        fig.update_xaxes(title="True Neutrino Energy (MeV)", row=1, col=1)
        fig.update_xaxes(title="True Neutrino Energy (MeV)", row=1, col=2)
        fig.update_yaxes(title="Nadir Angle cos(" + unicode("eta") + ")", row=1, col=3)
        fig.update_layout(coloraxis=dict(colorbar=dict(title="log(Counts)")))
        save_figure(
            fig, f"{save_path}",
            config=args.config, name=args.signal, subfolder=f"{args.folder.lower()}",
            filename=f"Selected_Signal_{energy}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}",
            rm=args.rewrite, debug=args.plot,
        )
