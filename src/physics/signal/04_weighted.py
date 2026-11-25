import os
import sys
import subprocess

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *

save_path = f"{root}/output/images/solar/nhits"
data_path = f"{root}/output/data/solar/nhits"

# Define flags for the analysis config and name with the python parser
parser = argparse.ArgumentParser(
    description="Plot the energy distribution of the particles"
)
parser.add_argument(
    "--config",
    nargs="+",
    type=str,
    help="The configuration to load",
    default=["hd_1x2x6_centralAPA"],
)
parser.add_argument(
    "--signal", nargs="+", type=str, help="The name of the configuration", default=["marley"]
)
parser.add_argument(
    "--folder",
    type=str,
    help="The name of the background folder",
    choices=["Reduced", "Truncated", "Nominal", "Pessimistic"],
    default="Nominal",
)
parser.add_argument(
    "--analysis",
    nargs="+",
    type=str,
    default=["Sensitivity"],
    help="Analysis type for PNFS best-cuts resolution (Weighted_Distributions_Fiducial only).",
)
parser.add_argument("--nhits",  type=int, default=None, help="NHits cut override (else from PNFS)")
parser.add_argument("--ophits", type=int, default=None, help="OpHits cut override (else from PNFS)")
parser.add_argument("--adjcls", type=int, default=None, help="AdjCl cut override (else from PNFS)")
parser.add_argument(
    "--cuts_signal", type=str, default=None,
    help="Signal sample whose selected cuts (PNFS best-cuts pkl) every sample of the scan is cut at. Default: the "
         "sample itself for marley, else the marley sample of the same production suffix (gamma_final -> "
         "marley_final). A missing pkl is an error, never a fallback to another sample.",
)
parser.add_argument(
    "--mc_filter_threshold", type=int, default=None,
    help="Min unweighted MC events per 1 MeV energy bin, at each scan threshold; bins below are zeroed. Default: the "
         "per-bin filter of the analysis (config.json MC_THRESHOLDS_PER_CONFIG MC_FILTER, else 2); radiological bins "
         "need at least RADIOLOGICAL_MIN_MC_PER_BIN (lib.smoothing.radiological_min_mc), as in 03_analysis.py.",
)
parser.add_argument(
    "--summary_oscillation", type=str, choices=["Mean", "Truth"], default="Mean",
    help="Signal weights of the threshold-scan summary: Mean (SignalParticleWeight*OscMean, the normalisation of the "
         "cut flows and the analysis spectra) or Truth (unoscillated).",
)
parser.add_argument(
    "--check_cutflow", action=argparse.BooleanOptionalAction, default=True,
    help="Compare the scan at the selected cuts (final stage, every panel) with the last stage of the SolarEnergy cut "
         "flow of the same config, sample, folder and analysis (cutflow_plot.py, --cutflow_label), rtol 1e-3; exit 1 "
         "on a mismatch. Skipped with a warning when the cut flow is missing or holds other cuts.",
)
parser.add_argument("--output_label", type=str, default="",
    help="Suffix of the output pkls (_LABEL), to write a re-export next to the existing outputs instead of over them.")
parser.add_argument("--output_dir", type=str, default=None,
    help="Directory of the output pkls (and figures) instead of output/data/solar/nhits, for exports that must not "
         "touch the production tree.")
parser.add_argument("--cutflow_dir", type=str, default=None,
    help="Directory of the cut flow compared by --check_cutflow (cutflow_plot.py --output_dir), instead of "
         "output/data/solar/cutflow.")
parser.add_argument("--cutflow_label", type=str, default="",
    help="Output label of the cut flow compared by --check_cutflow (cutflow_plot.py --output_label), '' for the default.")
parser.add_argument("--energy", type=str, default="SolarEnergy", help="Energy variable used to resolve FiducializationMask Ref pkl (must match 03_analysis.py --energy used during export).")

parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument(
    "--oscillation_backend",
    type=str,
    choices=["file", "prob3", "nufast"],
    default="nufast",
    help="Oscillation weighting backend. 'file' uses pre-computed pkl files; 'prob3'/'nufast' compute on-the-fly.",
)

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

args = parser.parse_args()
if args.output_dir:
    save_path = data_path = args.output_dir

# Self-dispatch: run once per (config, signal, analysis) — signal determines ROOT loading
# and compute_reco_workflow params, so it must be in the dispatch key (unlike cutflow_plot.py).
if len(args.config) > 1 or len(args.signal) > 1 or len(args.analysis) > 1:
    from itertools import product as _product
    _failed = []
    for _cfg, _sig, _ana in _product(args.config, args.signal, args.analysis):
        _cmd = [
            sys.executable, __file__,
            "--config",              _cfg,
            "--signal",              _sig,
            "--analysis",            _ana,
            "--folder",              args.folder,
            "--energy",              args.energy,
            "--oscillation_backend", args.oscillation_backend,
            "--summary_oscillation", args.summary_oscillation,
            "--cutflow_label",       args.cutflow_label,
            "--output_label",        args.output_label,
            *(["--output_dir", args.output_dir] if args.output_dir else []),
            *(["--cutflow_dir", args.cutflow_dir] if args.cutflow_dir else []),
            "--check_cutflow" if args.check_cutflow else "--no-check_cutflow",
            "--rewrite"       if args.rewrite       else "--no-rewrite",
            "--debug"         if args.debug         else "--no-debug",
            "--membrane_veto" if args.membrane_veto else "--no-membrane_veto",
        ]
        if args.nhits  is not None: _cmd += ["--nhits",  str(args.nhits)]
        if args.ophits is not None: _cmd += ["--ophits", str(args.ophits)]
        if args.adjcls is not None: _cmd += ["--adjcls", str(args.adjcls)]
        if args.mc_filter_threshold is not None: _cmd += ["--mc_filter_threshold", str(args.mc_filter_threshold)]
        if args.cuts_signal is not None: _cmd += ["--cuts_signal", args.cuts_signal]
        rprint(f"\n[green][CMD][/green] 04_weighted.py --config {_cfg} --signal {_sig} --analysis {_ana}")
        if subprocess.run(_cmd, check=False).returncode != 0:
            _failed.append(f"{_cfg} {_sig} {_ana}")
    if _failed:
        rprint(f"[red][ERROR][/red] 04_weighted.py failed for: {', '.join(_failed)}")
        sys.exit(1)
    sys.exit(0)

args.config   = args.config[0]
args.signal   = args.signal[0]
args.analysis = args.analysis[0]

config = args.config
name = args.signal
configs = {config: [name]}

_analysis_info = load_analysis_info(str(root))
# Per-bin MC filter of the analysis: run_sensitivity.py passes MC_THRESHOLDS_PER_CONFIG MC_FILTER to 03_analysis.py
# (default 2 there), which raises it to RADIOLOGICAL_MIN_MC_PER_BIN for the radiological sample.
if args.mc_filter_threshold is None:
    args.mc_filter_threshold = int(_analysis_info.get("MC_THRESHOLDS_PER_CONFIG", {}).get(config, {}).get("MC_FILTER", 2))
_mc_threshold = args.mc_filter_threshold
if name.split("_")[0].lower() == "radiological":
    _mc_threshold = max(_mc_threshold, radiological_min_mc(str(root)))
# Every sample is cut at the selected cuts of the signal (the backgrounds have no best-cuts record of their own).
if args.cuts_signal is None:
    args.cuts_signal = name if "marley" in name else "marley" + name[len(name.split("_")[0]):]
_OP_PLANE_CUT  = _analysis_info.get("QUALITY_CUTS", {}).get("OPFLASH_PLANE", 0)
_export_path   = f"{root}/output/data/results"   # mirrors 03_analysis.py export_path

if not os.path.exists(f"{save_path}/{args.folder.lower()}"):
    os.makedirs(f"{save_path}/{args.folder.lower()}")

user_input = {
    "workflow": "SIGNIFICANCE",
    "weights": {
        "marley": [
            "SignalParticleWeight",
            "SignalParticleWeightb8",
            "SignalParticleWeighthep",
        ],
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
    "rewrite": True,
    "debug": True,
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
        if "marley" in args.signal
        else {"PARTICLE_TYPE": "background", "PARTICLE_WEIGHTING": "histogram"}
    ),
    rm_branches=False,
    workflow=user_input["workflow"],
    debug=args.debug,
)


# ── Best-cuts helpers (for Weighted_Distributions_Fiducial only) ───────────────

def _extract_best_cuts(obj, cfg, sig):
    key = (cfg, sig, "SolarEnergy")
    if isinstance(obj, dict):
        row = obj.get(key)
        if row is None:
            return None
        return int(row["NHits"]), int(row["OpHits"]), int(row["AdjCl"])
    if key not in obj.columns:
        return None
    s = obj[key]
    return int(s["NHits"]), int(s["OpHits"]), int(s["AdjCl"])


def _load_best_cuts(cfg, sig, analysis, folder, info):
    if analysis.upper() == "SENSITIVITY":
        # 04_best_cuts.py / 06_significance.py: {PATH}/SENSITIVITY/{config}/{name}/{folder}/
        _base = f"{info['PATH']}/SENSITIVITY/{cfg}/{sig}/{folder.lower()}"
        _tag  = "SENSITIVITY"
    else:
        # 05_best_sigmas.py: {PATH}/{ANALYSIS}/{folder}/{config}/{name}/
        _base = f"{info['PATH']}/{analysis.upper()}/{folder.lower()}/{cfg}/{sig}"
        _tag  = analysis
    _pkl  = f"{_base}/{cfg}_{sig}_highest_{_tag}.pkl"
    if not os.path.exists(_pkl):
        raise SystemExit(f"[ERROR] Best-cuts pkl not found: {_pkl}. Run the cut optimisation for {sig} first or pass "
                         "--nhits/--ophits/--adjcls.")
    cuts = _extract_best_cuts(pickle.load(open(_pkl, "rb")), cfg, sig)
    if cuts is None:
        raise SystemExit(f"[ERROR] {_pkl} has no SolarEnergy entry for {cfg}/{sig}.")
    return cuts


# ── Threshold-scan helpers ─────────────────────────────────────────────────────

def _scan_ge(vals, wts, thresholds, w_var=None):
    """sum(wts[vals >= t]), count(vals >= t), and sum(w_var[vals >= t]) for each t."""
    if len(vals) == 0:
        z = np.zeros(len(thresholds))
        return z, z.astype(int), z
    order = np.argsort(vals)
    sv, sw = vals[order], wts[order]
    cs = np.concatenate([[0.0], np.cumsum(sw)])
    idx = np.searchsorted(sv, thresholds, side="left")
    counts   = cs[-1] - cs[idx]
    nevents  = (len(vals) - idx).astype(int)
    if w_var is not None:
        sv2 = w_var[order]
        cs2 = np.concatenate([[0.0], np.cumsum(sv2)])
        variance = cs2[-1] - cs2[idx]
    else:
        variance = np.zeros(len(thresholds))
    return counts, nevents, variance


def _scan_lt(vals, wts, thresholds, w_var=None):
    """sum(wts[vals < t]), count(vals < t), and sum(w_var[vals < t]) for each t."""
    if len(vals) == 0:
        z = np.zeros(len(thresholds))
        return z, z.astype(int), z
    order = np.argsort(vals)
    sv, sw = vals[order], wts[order]
    cs = np.concatenate([[0.0], np.cumsum(sw)])
    idx = np.searchsorted(sv, thresholds, side="left")
    counts  = cs[idx]
    nevents = idx.astype(int)
    if w_var is not None:
        sv2 = w_var[order]
        cs2 = np.concatenate([[0.0], np.cumsum(sv2)])
        variance = cs2[idx]
    else:
        variance = np.zeros(len(thresholds))
    return counts, nevents, variance


def _scan_filtered(vals, energy, wts, thresholds, rb_edges, threshold, mode):
    """Weighted counts and variance at each threshold with the per-bin MC filter of 03_analysis.py applied at that
    threshold: for each t the selected events (vals >= t for mode "ge", vals < t for "lt") are binned in energy
    (rb_edges, 1 MeV), bins with fewer than threshold unweighted events are zeroed, and the rest summed. Variance is
    sum over bins of sum(w²)/mc_count. Events outside rb_edges do not count (the analysis spectra cover rb_edges only).
    """
    nb, nt = len(rb_edges) - 1, len(thresholds)
    eb = np.digitize(energy, rb_edges) - 1
    ok = (eb >= 0) & (eb < nb)
    pos = np.searchsorted(thresholds, vals[ok], side="right")   # number of thresholds <= value, 0..nt
    key = pos * nb + eb[ok]
    w = wts[ok]
    a_n  = np.bincount(key, minlength=(nt + 1) * nb).reshape(nt + 1, nb)
    a_w  = np.bincount(key, weights=w,      minlength=(nt + 1) * nb).reshape(nt + 1, nb)
    a_w2 = np.bincount(key, weights=w ** 2, minlength=(nt + 1) * nb).reshape(nt + 1, nb)
    if mode == "ge":   # value >= thresholds[i]  <=>  i < pos
        s_n, s_w, s_w2 = (np.cumsum(a[::-1], axis=0)[::-1][1:] for a in (a_n, a_w, a_w2))
    else:              # value <  thresholds[i]  <=>  i >= pos
        s_n, s_w, s_w2 = (np.cumsum(a, axis=0)[:nt] for a in (a_n, a_w, a_w2))
    keep = s_n >= threshold
    counts = np.where(keep, s_w, 0.0).sum(axis=1)
    variance = np.where(keep, s_w2 / np.maximum(s_n, 1), 0.0).sum(axis=1)
    return counts, variance


# ── Main loop ──────────────────────────────────────────────────────────────────

nhits_list          = []   # full per-event distributions (original schema)
nhits_fiducial_list = []   # threshold-scan summary per (stage, hit variable)


# The folder's background model and statistical reduction scale the background weights, as in 03_analysis.py.
if "marley" not in args.signal:
    _divisor = background_weight_divisor(str(root), args.folder, args.signal, config, run["Reco"]["SignalParticleK"])
    run["Reco"]["SignalParticleWeight"] = np.asarray(run["Reco"]["SignalParticleWeight"], dtype=float) / _divisor

for config in configs:
    info = json.loads(open(f"{root}/config/{config}/{config}_config.json").read())
    # Counts are quoted at the evaluation livetime (EVALUATION_EXPOSURE_YEARS in
    # config/analysis/config.json): DayNight/HEP 20 yr, Sensitivity 30 yr. The self-dispatch
    # above already split a multi-analysis call into one child per analysis, so args.analysis
    # is a single analysis here and each one is scaled at its own livetime.
    _exposure    = get_evaluation_exposure(str(root), args.analysis, config=config)
    _scale       = get_full_detector_mass(config, info) * _exposure   # kT·yr → events/raw_weight

    for name in configs[config]:

        # ── Resolve best cuts (fiducial summary only) ──────────────────────────
        _nhits, _ophits, _adjcls = args.nhits, args.ophits, args.adjcls
        if None in (_nhits, _ophits, _adjcls):
            _nhits, _ophits, _adjcls = _load_best_cuts(config, args.cuts_signal, args.analysis, args.folder, info)
        rprint(f"[cyan][INFO][/cyan] {config}/{name} {args.analysis}: cuts NHits={_nhits} OpHits={_ophits} AdjCl={_adjcls} "
               f"(of {args.cuts_signal if args.nhits is None else 'the command line'}), per-bin MC filter {_mc_threshold}, "
               f"signal weights {args.summary_oscillation}")

        _has_cuts = None not in (_nhits, _ophits, _adjcls)
        _reco     = run["Reco"]
        _all_mask = np.ones(len(_reco["Event"]), dtype=bool)

        # Load FiducializationMask exported by 03_analysis.py (surface + spatial fiducial).
        # Path mirrors save_pkl(export_path, config, name, subfolder=folder, filename=...).
        _fid_pkl = (
            f"{_export_path}/{config}/{name}/{args.folder.lower()}/"
            f"{config}_{name}_FiducializationMask_{args.energy}_{args.analysis.upper()}.pkl"
        )
        if os.path.exists(_fid_pkl):
            _geo_fid = np.asarray(pickle.load(open(_fid_pkl, "rb")), dtype=bool)
        else:
            rprint(
                f"[yellow][WARNING][/yellow] FiducializationMask not found: {_fid_pkl}\n"
                f"  Run: 03_analysis.py --config {config} --signal {name} "
                f"--folder {args.folder} --energy {args.energy} --export_fiducial --skip_scan\n"
                f"  Falling back to SignalParticleSurface < 3."
            )
            _geo_fid = _reco["SignalParticleSurface"] < 3

        # Quality cuts: TPC-PDS matching — same as 03_analysis.py build_analysis_mask.
        _flash_mask = (
            accepted_flash_planes(
                _reco["MatchedOpFlashPlane"], str(root), args.membrane_veto
            )
            & (_reco["MatchedOpFlashPE"] > 0)
        )
        _fid_mask = _geo_fid & _flash_mask
        # Clusters the analysis removes (03_analysis.py AnalysisKeep export: neutron-capture veto with ACTION remove, removal of
        # the no-offset clusters) leave at the Fiducial stage, as in cutflow_plot.py; older exports have no such file.
        _keep_pkl = _fid_pkl.replace("_FiducializationMask_", "_AnalysisKeep_")
        if os.path.exists(_keep_pkl):
            _fid_mask = _fid_mask & np.asarray(pickle.load(open(_keep_pkl, "rb")), dtype=bool)

        # Global threshold ranges (full dataset so all stages share the same x-axis), always reaching the selected cut
        _t_nhits  = np.arange(0, max(int(np.max(_reco["NHits"]))               + 2, (_nhits  or 0) + 1))
        _t_ophits = np.arange(0, max(int(np.max(_reco["MatchedOpFlashNHits"])) + 2, (_ophits or 0) + 1))
        _t_adjcls = np.arange(0, max(int(np.max(_reco["AdjClNum"]))            + 2, (_adjcls or 0) + 1))

        _weights_list = list(zip(
            user_input["weights"][name.split("_")[0]],
            user_input["weight_labels"][name.split("_")[0]],
            user_input["colors"][name.split("_")[0]],
        ))
        _surfaces = [None, -1, 0, 1, 2, 3, 4] if name in ["marley", "neutron"] else [None]

        # ── Per-surface entries — original schema (unchanged) ──────────────────
        for (weight, weight_labels, color), surface in track(
            product(
                _weights_list,
                _surfaces,
            ),
            total=len(_weights_list) * len(_surfaces),
            description=f"Processing {name} - {config}",
        ):
            if surface is None:
                if folder_applies_surface_cut(str(root), args.folder):
                    mask = _reco["SignalParticleSurface"] < 3
                else:
                    mask = _all_mask
            else:
                mask = _reco["SignalParticleSurface"] == surface

            if np.sum(mask) > 0:
                nhits_list.append(
                    {
                        "Config": config,
                        "Name": name,
                        "Folder": args.folder,
                        "Component": weight_labels,
                        # These are the truth-normalised (unoscillated) weights:
                        # SignalParticleWeight*, not SignalParticleWeight*Osc{Mean,Day,Night}.
                        # "Solar" here is therefore ~1/<P_ee> (~3x) larger than the "Solar"
                        # component in the Cutflow / Counts pkls, which is OscMean. Flagged
                        # explicitly so the two are never compared as if they were the same
                        # quantity. No Day/Night split exists at this stage.
                        "Oscillation": "Truth",
                        "Weight": weight,
                        "Type": "signal" if "marley" in name else "background",
                        "Surface": surface,
                        "#Hits": _reco["NHits"][mask],
                        "#OpHits": _reco["MatchedOpFlashNHits"][mask],
                        "#AdjCls": _reco["AdjClNum"][mask],
                        "Counts": _reco[weight][mask],
                        "TrueEnergy": _reco["SignalParticleK"][mask],
                        "RecoEnergy": _reco["SolarEnergy"][mask],
                    }
                )
            else:
                if args.debug:
                    print(f"No events found for {name} - {config} - {weight} - Surface {surface}")

        # ── Fiducial threshold-scan summary ────────────────────────────────────
        # Stages: (name, nhits_base, ophits_base, adjcls_base, nc, oc, ac)
        # nhits_base / ophits_base / adjcls_base = base mask for each hit-variable scan.
        # For Raw and Fiducial all three bases are the same.
        # For NHits+OpHits+AdjCl each base fixes the other two cuts at best values.
        _scan_stages: list[tuple] = [
            ("Raw",      _all_mask, _all_mask, _all_mask, None, None, None),
            ("Fiducial", _fid_mask, _fid_mask, _fid_mask, None, None, None),
        ]
        if _has_cuts:
            _nhv = _reco["NHits"]
            _opv = _reco["MatchedOpFlashNHits"]
            _acv = _reco["AdjClNum"]
            _scan_stages.append((
                "NHits+OpHits+AdjCl",
                _fid_mask & (_opv >= _ophits) & (_acv < _adjcls),   # scan NHits
                _fid_mask & (_nhv >= _nhits)  & (_acv < _adjcls),   # scan OpHits
                _fid_mask & (_nhv >= _nhits)  & (_opv >= _ophits),  # scan AdjCls
                _nhits, _ophits, _adjcls,
            ))
        else:
            rprint(f"[yellow][WARNING][/yellow] No cut values for {config}/{name}. NHits+OpHits+AdjCl stage skipped.")

        # Per-bin MC filter at each threshold of each scan (_scan_filtered): SolarEnergy in hep_rebin (1 MeV) bins,
        # as 03_analysis.py, so the scan at the selected cuts equals the analysis selection.
        _energy_for_filter = np.asarray(_reco[args.energy], dtype=float)
        # The HEP analysis bins its spectra in SolarEnergyHEP (03_analysis.py --hep_energy, _energy_key), and so does its
        # cut flow (cutflow_plot.py): same clusters, same order, from the reco workflow or the 03_analysis.py export.
        if args.analysis == "HEP" and args.energy == "SolarEnergy":
            _hep_ref = _fid_pkl.replace(f"_FiducializationMask_{args.energy}_HEP", "_AnalysisData_SolarEnergyHEP_Ref")
            if "SolarEnergyHEP" in _reco:
                _energy_for_filter = np.asarray(_reco["SolarEnergyHEP"], dtype=float)
            elif os.path.exists(_hep_ref):
                _energy_for_filter = np.asarray(pickle.load(open(_hep_ref, "rb")), dtype=float)
                if len(_energy_for_filter) != len(_reco["Event"]):
                    raise SystemExit(f"[ERROR] {_hep_ref} has {len(_energy_for_filter)} clusters, the sample {len(_reco['Event'])}.")
            else:
                rprint(f"[yellow][WARNING][/yellow] No SolarEnergyHEP for {config}/{name}: HEP MC filter in SolarEnergy.")
        _osc_signal = "marley" in name and args.summary_oscillation == "Mean"

        for weight, weight_labels, color in _weights_list:
            # Signal: the oscillated weights of the cut flows and analysis spectra (SignalParticleWeight*OscMean) unless
            # --summary_oscillation Truth; the Weight column keeps the base name the plot lists select on.
            wts = np.asarray(_reco[f"{weight}OscMean" if _osc_signal else weight], dtype=float)
            for stage_name, m_nhits, m_ophits, m_adjcls, nc, oc, ac in _scan_stages:
                _meta = {
                    "Config":      config,
                    "Name":        name,
                    "Folder":      args.folder,
                    "Component":   weight_labels,
                    "Oscillation": args.summary_oscillation if "marley" in name else "Truth",
                    "Weight":      weight,
                    "Type":        "signal" if "marley" in name else "background",
                    "Stage":       stage_name,
                    "NHits":       nc,
                    "OpHits":      oc,
                    "AdjCl":       ac,
                    "Exposure":    _exposure,
                    "CountsUnit":  f"events / {_exposure:.0f} yr",
                }

                # NEvents: raw unweighted count (no mc_filter).
                # Counts: weighted sum with mc_filter applied (low-stat bins zeroed).
                # CountsError: scale * sqrt(sum(w²/mc_count_bin)) — same MC-stat error as 03_analysis.py.

                _nh_vals = _reco["NHits"][m_nhits]
                c, var = _scan_filtered(_nh_vals, _energy_for_filter[m_nhits], wts[m_nhits], _t_nhits, hep_rebin,
                                        _mc_threshold, "ge")
                _, n_raw, _ = _scan_ge(_nh_vals, wts[m_nhits], _t_nhits)
                nhits_fiducial_list.append({**_meta, "#Hits": _t_nhits, "#OpHits": np.nan, "#AdjCls": np.nan,
                                            "Counts": c * _scale, "CountsError": np.sqrt(var) * _scale, "NEvents": n_raw})

                _op_vals = _reco["MatchedOpFlashNHits"][m_ophits]
                c, var = _scan_filtered(_op_vals, _energy_for_filter[m_ophits], wts[m_ophits], _t_ophits, hep_rebin,
                                        _mc_threshold, "ge")
                _, n_raw, _ = _scan_ge(_op_vals, wts[m_ophits], _t_ophits)
                nhits_fiducial_list.append({**_meta, "#Hits": np.nan, "#OpHits": _t_ophits, "#AdjCls": np.nan,
                                            "Counts": c * _scale, "CountsError": np.sqrt(var) * _scale, "NEvents": n_raw})

                _ac_vals = _reco["AdjClNum"][m_adjcls]
                c, var = _scan_filtered(_ac_vals, _energy_for_filter[m_adjcls], wts[m_adjcls], _t_adjcls, hep_rebin,
                                        _mc_threshold, "lt")
                _, n_raw, _ = _scan_lt(_ac_vals, wts[m_adjcls], _t_adjcls)
                nhits_fiducial_list.append({**_meta, "#Hits": np.nan, "#OpHits": np.nan, "#AdjCls": _t_adjcls,
                                            "Counts": c * _scale, "CountsError": np.sqrt(var) * _scale, "NEvents": n_raw})

# ── Closure with the cut flow ─────────────────────────────────────────────────
# At the selected cuts every panel of the final stage is the analysis selection: it must equal the total of the last
# stage of the SolarEnergy cut flow (cutflow_plot.py, Counts summed over energy) of the same sample and analysis.
# The signal components need --summary_oscillation Mean; HEP has no "Solar" cut-flow component (8B and hep are checked).

def _check_cutflow(rows: list) -> list:
    path = (f"{args.cutflow_dir or f'{root}/output/data/solar/cutflow'}/{config}/{name}/{args.folder.lower()}/{args.analysis.lower()}/"
            f"{config}_{name}_{args.energy}_{args.analysis}_Cutflow{'_' + args.cutflow_label if args.cutflow_label else ''}.pkl")
    if not os.path.exists(path):
        rprint(f"[yellow][WARNING][/yellow] No {path}: threshold scan not checked against the cut flow.")
        return []
    cf = pd.read_pickle(path)
    held = cf[["NHits", "OpHits", "AdjCl"]].drop_duplicates().values.tolist()
    if held != [[_nhits, _ophits, _adjcls]]:
        rprint(f"[yellow][WARNING][/yellow] {path} holds cuts {held}, the scan {[_nhits, _ophits, _adjcls]}: not checked.")
        return []
    final = cf[cf["Stage"] == cf["Stage"].iloc[-1]]
    # Signal components by label; a background cut flow has one component, named after the sample (gamma_final).
    totals = {(str(c) if "marley" in name else str(c).split("_")[0]): float(np.sum(v))
              for c, v in zip(final["Component"], final["Counts"])}
    bad, checked = [], 0
    for row in rows:
        if row["Stage"] != "NHits+OpHits+AdjCl" or row["Component"] not in totals:
            continue
        if "marley" in name and args.summary_oscillation != "Mean":
            continue
        axis, cut = next((a, c) for a, c in (("#Hits", _nhits), ("#OpHits", _ophits), ("#AdjCls", _adjcls))
                         if isinstance(row[a], np.ndarray))
        ours = float(np.asarray(row["Counts"])[np.asarray(row[axis]) == cut][0])
        ref = totals[row["Component"]]
        checked += 1
        if not np.isclose(ours, ref, rtol=1e-3, atol=1e-9 * max(abs(ref), 1.0)):
            bad.append(f"{row['Component']} {axis}: scan {ours:.6g}, cut flow {ref:.6g} (ratio {ours / ref if ref else np.nan:.4f})")
    rprint(f"[cyan][INFO][/cyan] Threshold scan vs {os.path.basename(path)}: {checked} panels checked, {len(bad)} mismatched")
    return bad


if args.check_cutflow and nhits_fiducial_list and _has_cuts:
    _mismatch = _check_cutflow(nhits_fiducial_list)
    if _mismatch:
        rprint(f"[red][ERROR][/red] {config} {name} {args.folder} {args.analysis}: the threshold scan at the selected cuts "
               "differs from the cut flow:\n  " + "\n  ".join(_mismatch))
        sys.exit(1)

# ── Save ───────────────────────────────────────────────────────────────────────

_out_sfx = f"_{args.output_label}" if args.output_label else ""

save_df(
    pd.DataFrame(nhits_list),
    f"{data_path}",
    config=config,
    name=name,
    filename=f"Weighted_Distributions{_out_sfx}",
    subfolder=f"{args.folder.lower()}",
    rm=user_input["rewrite"],
    debug=user_input["debug"],
)

if nhits_fiducial_list:
    save_df(
        pd.DataFrame(nhits_fiducial_list),
        f"{data_path}",
        config=config,
        name=name,
        filename=f"Weighted_Distributions_Fiducial_{args.analysis}{_out_sfx}",
        subfolder=f"{args.folder.lower()}/{args.analysis.lower()}",
        rm=user_input["rewrite"],
        debug=user_input["debug"],
    )
