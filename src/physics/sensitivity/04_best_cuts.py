import os
import re
import sys
import subprocess
import tempfile
from glob import glob as glob_files
from shlex import quote

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *
from lib.root import Sensitivity_Fitter
from lib.fitting import SENSITIVITY_FIT_METHODS, sensitivity_pull_chi2
from lib.template_guards import check_template_sampling_marker

parser = argparse.ArgumentParser(
    description="Cut optimisation for Sensitivity analysis. "
    "Generates signal templates at solar+reactor reference points for every "
    "background-template cut candidate, scores each cut with the Sensitivity_Fitter "
    "chi2 figure-of-merit, and writes highest_SENSITIVITY.pkl."
)
parser.add_argument("--config", type=str, default="hd_1x2x6_centralAPA")
parser.add_argument("--signal",   type=str, default="marley")
parser.add_argument(
    "--folder", type=str, choices=["Reduced", "Truncated", "Nominal"], default="Nominal"
)
parser.add_argument(
    "--energy",
    type=str,
    choices=["SignalParticleK", "MainK", "ClusterEnergy", "TotalEnergy", "SelectedEnergy", "SolarEnergy"],
    default="SolarEnergy",
)
parser.add_argument(
    "--oscillation_backend",
    type=str, choices=["file", "prob3", "nufast"], default="nufast",
)
parser.add_argument("--exposure",              type=float, default=get_analysis_exposure(str(root), "Sensitivity"), help="Exposure in years. Default from ANALYSIS_EXPOSURES['SENSITIVITY']['PRIMARY'] in config/analysis/config.json.")
# Default to the configured Sensitivity uncertainties (ANALYSIS_UNCERTAINTIES.SENSITIVITY in
# config/analysis/config.json), i.e. what run_sensitivity.py forwards. With the old None -> 0
# default a manual run scored every cut with a FIXED signal normalisation (Delta chi2 20.3 for
# NHits3 AdjCl2 OpHits12 against 6.0 with the 4% prior), so the map disagreed with 06_significance.
_configured_unc = load_analysis_info(str(root)).get("ANALYSIS_UNCERTAINTIES", {}).get("SENSITIVITY", {})
parser.add_argument("--signal_uncertainty",    type=float,
                    default=float(_configured_unc.get("signal_uncertainty", load_analysis_info(str(root)).get("SIGNAL_ERROR", 0.04))),
                    help="Signal normalisation prior width. Default: ANALYSIS_UNCERTAINTIES.SENSITIVITY.signal_uncertainty.")
parser.add_argument("--background_uncertainty",type=float,
                    default=float(_configured_unc.get("background_uncertainty", load_analysis_info(str(root)).get("BACKGROUND_ERROR", 0.02))),
                    help="Background normalisation prior width. Default: ANALYSIS_UNCERTAINTIES.SENSITIVITY.background_uncertainty.")
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug",   action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--plot",    action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--background", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--study_label",     type=str,   default=None, help="Tag appended to output pkl filename to isolate study results.")
parser.add_argument("--truth_fiducial", action=argparse.BooleanOptionalAction, default=False, help="Truth-position fiducialisation variant. Must match the flag passed to 03_analysis.py so study_context selects the labeled Rebin pkl.")
parser.add_argument("--charge_threshold", type=float, default=0,   help="Charge threshold Q (ADC). When >0, reads templates from labeled subfolders.")
parser.add_argument(
    "--membrane_veto",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "Accept only cathode/APA optical matches (QUALITY_CUTS.OPFLASH_PLANE, plane 0). "
        "This is the default. --no-membrane_veto also accepts membrane and endcap matches "
        "(VD planes 1-4). Forwarded to sensitivity/02_signal_template.py. "
        "Used by the membrane_veto study."
    ),
)
parser.add_argument(
    "--min_background_mc",
    type=int,
    default=20,
    help=(
        "Reject cuts where an essential background has fewer than this many simulated events "
        "summed over the spectrum (default 10). 03_analysis.py zeroes bins with MC < "
        "--mc_filter_threshold, so a thinly simulated background reads as exactly zero and the "
        "cut optimiser is rewarded for a background it cannot see. 0 disables the guard."
    ),
)
parser.add_argument(
    "--scan_strategy",
    type=str,
    choices=["coarse", "full"],
    default="coarse",
    help="How to search the cut grid. 'coarse' (default) scores a strided seed grid, then "
         "refines around the best --scan_top cuts at halving offsets down to adjacent cuts. "
         "'full' scores every cut that has a background template (6800 cuts, ~42 h serial).",
)
parser.add_argument(
    "--scan_fraction",
    type=float,
    default=0.01,
    help="Coarse strategy: target fraction of the grid for the seed scan (default 0.01). The "
         "per-axis stride is the cube root of its inverse; axis extremes are always included.",
)
parser.add_argument(
    "--scan_top",
    type=int,
    default=3,
    help="Coarse strategy: how many leading cuts to refine around each round (default 3).",
)
parser.add_argument(
    "--scan_workers",
    type=int,
    default=8,
    help="Signal templates are built by this many concurrent 02_signal_template.py processes "
         "(default 8). Each holds its own copy of the signal dataframe, ~4 GB.",
)
parser.add_argument("--nhits",  type=int, default=None, help="Number of hits cut. If provided, adds this cut to the evaluation list.")
parser.add_argument("--adjcls", type=int, default=None, help="Adjacent clusters cut. If provided, adds this cut to the evaluation list.")
parser.add_argument("--ophits", type=int, default=None, help="Optical hits cut. If provided, adds this cut to the evaluation list.")
parser.add_argument(
    "--fit_method",
    type=str,
    choices=list(SENSITIVITY_FIT_METHODS),
    default="pull",
    help="chi2 method used to score each cut: 'pull' (closed-form pull profile, default) or 'legacy' (Sensitivity_Fitter).",
)

args = parser.parse_args()
_ctx = study_context(args, analysis="Sensitivity")
_study_suffix    = _ctx.study_suffix
_template_suffix = _ctx.template_suffix

analysis_info = load_analysis_info(str(root))
info = json.loads(open(f"{root}/config/{args.config}/{args.config}_config.json").read())

solar_dm2 = analysis_info["SOLAR_DM2"]
react_dm2 = analysis_info["REACT_DM2"]
sin13     = analysis_info["SIN13"]
sin12     = analysis_info["SIN12"]

threshold = get_analysis_threshold(str(root), "SENSITIVITY", stage="SIGNIFICANCE", fallback=0.0)
thld = int(np.where(sensitivity_rebin_centers >= threshold)[0][0]) if threshold > 0.0 else 0

expected_ecols = len(sensitivity_rebin_centers)
expected_nrows = analysis_info["NADIR_BINS"]

signal_path     = f"{info['PATH']}/SENSITIVITY/{args.config}/{args.signal}/{args.folder.lower()}/{args.energy}{_template_suffix}"
background_path = f"{info['PATH']}/SENSITIVITY/{args.config}/background/{args.folder.lower()}/{args.energy}{_template_suffix}"


# ── helpers ────────────────────────────────────────────────────────────────────

def _background_candidates():
    pattern = f"{background_path}/{args.config}_background_NHits*_AdjCl*_OpHits*.pkl"
    valid, stale = [], []
    for f in sorted(glob_files(pattern)):
        if _is_valid(f):
            valid.append(f)
        else:
            stale.append(f)
    if stale:
        rprint(
            f"[yellow][WARNING][/yellow] Ignoring {len(stale)} stale background template(s) "
            f"(wrong energy or nadir binning — orphans from old code): "
            + ", ".join(os.path.basename(f) for f in stale)
        )
    return valid


def _parse_cut(filepath: str):
    base = os.path.basename(filepath)
    m = re.search(r"NHits(?P<n>\d+)_AdjCl(?P<a>\d+)_OpHits(?P<o>\d+)", base)
    if m is None:
        return None
    return {"NHits": int(m.group("n")), "AdjCl": int(m.group("a")), "OpHits": int(m.group("o"))}


def _pkl_path(nhits, adjcl, ophits, dm2):
    return (
        f"{signal_path}/{args.config}_{args.signal}"
        f"_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}"
        f"_dm2_{dm2:.3e}_sin13_{sin13:.3e}_sin12_{sin12:.3e}.pkl"
    )


def _is_valid(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        arr = np.asarray(pd.read_pickle(path), dtype=float)
        return not (arr.ndim >= 2 and (arr.shape[1] != expected_ecols or arr.shape[0] != expected_nrows))
    except Exception:
        return False


def _sampling_current(cut) -> bool:
    """Reference templates of this cut were built with the configured P_ee sampling."""
    ok, _ = check_template_sampling_marker(
        signal_path, args.config, args.signal, cut["NHits"], cut["AdjCl"], cut["OpHits"],
        backend=args.oscillation_backend,
        nadir_oversample=int(analysis_info.get("OSC_NADIR_OVERSAMPLE", 1)),
        scopes=("scan", "grid"),
    )
    return ok


# The cut list goes to 02_signal_template.py in a file, not on the command line: a full
# pre-best-cut scan is 6800 triplets (~279 KiB of JSON), over the 128 KiB cap on a single argv
# entry, and printing it would bury every other line of the log.
def _template_command(cuts_path):
    cmd = [
        "python3", f"{root}/src/physics/sensitivity/02_signal_template.py",
        "--config",               args.config,
        "--signal",                 args.signal,
        "--folder",               args.folder,
        "--energy",               args.energy,
        "--cuts_file",            cuts_path,
        "--exposure",             str(args.exposure),
        "--oscillation_backend",  args.oscillation_backend,
        "--scan_mode",
        "--rewrite" if args.rewrite else "--no-rewrite",
        "--no-plot",
        "--no-debug",
        "--no-test",
    ]
    if args.charge_threshold > 0:
        cmd += ["--charge_threshold", str(args.charge_threshold)]
    if not args.membrane_veto:
        cmd += ["--no-membrane_veto"]
    # Without this the scan builds nominal-fiducial templates and study_context gives them an
    # empty template suffix, so they land in the unlabeled energy directory while this script
    # looks for them under {energy}_{label} — every candidate then reads as missing or stale.
    if args.truth_fiducial:
        cmd += ["--truth_fiducial"]
    if args.study_label:
        cmd += ["--study_label", args.study_label]
    return cmd


def _generate_templates(cuts):
    """Build signal templates for `cuts`, split across --scan_workers processes.

    Each cut writes its own per-cut pkls, so workers share no state; the split is round-robin
    to keep per-worker load even.
    """
    cuts = list(cuts)
    if not cuts:
        return True
    workers = max(1, min(int(args.scan_workers), len(cuts)))
    chunks = [chunk for chunk in (cuts[i::workers] for i in range(workers)) if chunk]

    _cuts_dir = f"{root}/output/tmp"
    os.makedirs(_cuts_dir, exist_ok=True)
    procs, paths, logs = [], [], []
    for chunk in chunks:
        _handle, cuts_path = tempfile.mkstemp(
            prefix=f"cuts_{args.config}_{args.signal}_{args.folder.lower()}_",
            suffix=".json",
            dir=_cuts_dir,
        )
        with os.fdopen(_handle, "w") as _cuts_file:
            json.dump(chunk, _cuts_file)
        paths.append(cuts_path)
        # Workers share the parent's terminal, so their progress bars and tracebacks would
        # interleave. Each writes to its own log; failures are reported with a tail below.
        log_path = f"{root}/output/logs/scan_worker_{len(procs)}_{args.config}_{args.folder.lower()}.log"
        log_handle = open(log_path, "w")
        logs.append((log_path, log_handle))
        procs.append(subprocess.Popen(_template_command(cuts_path), stdout=log_handle, stderr=subprocess.STDOUT))

    rprint(
        f"[cyan][INFO][/cyan] Generating signal templates for {len(cuts)} cut(s) "
        f"across {len(procs)} worker(s)."
    )
    if paths:
        rprint(f"\n[green][CMD][/green] {' '.join(quote(str(c)) for c in _template_command(paths[0]))}")

    ok = True
    try:
        for i, proc in enumerate(procs):
            if proc.wait() != 0:
                ok = False
                log_path, _ = logs[i]
                rprint(
                    f"[yellow][WARNING][/yellow] signal template worker {i} failed "
                    f"(exit {proc.returncode}); last lines of {log_path}:"
                )
                try:
                    with open(log_path) as fh:
                        for line in fh.readlines()[-15:]:
                            rprint(f"    {line.rstrip()}")
                except OSError:
                    pass
    finally:
        for _, handle in logs:
            try:
                handle.close()
            except OSError:
                pass
        for cuts_path in paths:
            try:
                os.remove(cuts_path)
            except FileNotFoundError:
                pass
    if ok:
        rprint(f"[cyan][INFO][/cyan] {len(procs)} worker(s) finished; logs in {root}/output/logs/scan_worker_*.log")
    return ok


def _load_template(path: str):
    return np.nan_to_num(np.asarray(pd.read_pickle(path), dtype=float), nan=0.0)


# ── discover cuts ───────────────────────────────────────────────────────────────

cut_candidates = [c for c in (_parse_cut(f) for f in _background_candidates()) if c is not None]

# Add manually specified cuts from flags if provided
if args.nhits is not None and args.adjcls is not None and args.ophits is not None:
    manual_cut = {"NHits": args.nhits, "AdjCl": args.adjcls, "OpHits": args.ophits}
    # Only add if not already in the list
    if manual_cut not in cut_candidates:
        cut_candidates.append(manual_cut)
        rprint(
            f"[cyan][INFO][/cyan] Added manual cut candidate: "
            f"NHits{args.nhits} AdjCl{args.adjcls} OpHits{args.ophits}"
        )

if not cut_candidates:
    rprint(
        f"[red][ERROR][/red] No background templates found in {background_path}. "
        "Run sensitivity/01_background_template.py first."
    )
    raise SystemExit(1)


def _mc_supported_cuts():
    """Cuts whose essential backgrounds carry enough simulation to be believed.

    The Rebin pkls keep the raw unweighted MC entry count per bin. Where that is thin,
    03_analysis.py has already zeroed the weighted Counts, so the background is not small
    here, it is unmeasured - and a cut scan maximises exactly that.
    """
    if args.min_background_mc <= 0:
        return None
    essential = [s for s, is_essential in get_essential_backgrounds(str(root)).items() if is_essential]
    if not essential:
        return None
    # Match the labeled Rebins 01_background_template.py reads for this variant.
    _selection_changed = (
        getattr(args, "charge_threshold", 0) > 0
        or args.truth_fiducial
        or not args.membrane_veto
    )
    _label = args.study_label if _selection_changed else None
    supported, checked = None, []
    for sample, filepath in load_available_background_dataframes(
        str(root), "SENSITIVITY", args.folder, args.config, args.energy, study_label=_label
    ):
        if sample not in essential:
            continue
        frame = pd.read_pickle(filepath)
        if "MCCounts" not in frame.columns:
            rprint(f"[yellow][WARNING][/yellow] {sample} Rebin has no MCCounts; MC support unchecked.")
            continue
        ok = set()
        for _, row in frame.iterrows():
            entries = row["MCCounts"]
            total_entries = float(np.nansum(np.asarray(list(entries), dtype=float))) if hasattr(entries, "__len__") else float(entries)
            if total_entries >= args.min_background_mc:
                ok.add((int(row["NHits"]), int(row["AdjCl"]), int(row["OpHits"])))
        checked.append(sample)
        supported = ok if supported is None else (supported & ok)
    if supported is None:
        return None
    rprint(
        f"[cyan][INFO][/cyan] MC support: {len(supported)} of {len(cut_candidates)} cuts have "
        f">= {args.min_background_mc} simulated events in every essential background "
        f"({', '.join(checked)})."
    )
    return supported


_supported = _mc_supported_cuts()
if _supported is not None:
    _rejected = [c for c in cut_candidates if (c["NHits"], c["AdjCl"], c["OpHits"]) not in _supported]
    cut_candidates = [c for c in cut_candidates if (c["NHits"], c["AdjCl"], c["OpHits"]) in _supported]
    if _rejected:
        rprint(
            f"[yellow][WARNING][/yellow] Dropping {len(_rejected)} cut(s) where an essential "
            f"background is too thinly simulated to trust (e.g. "
            + ", ".join(
                f"NHits{c['NHits']} AdjCl{c['AdjCl']} OpHits{c['OpHits']}" for c in _rejected[:3]
            )
            + "). Raise --min_background_mc to tighten, or 0 to disable."
        )
    if not cut_candidates:
        rprint(
            "[red][ERROR][/red] No cut has adequate MC support for its essential backgrounds. "
            "Lower --min_background_mc, or simulate more background statistics."
        )
        raise SystemExit(1)

rprint(
    f"[cyan][INFO][/cyan] Found {len(cut_candidates)} background-template cut candidates "
    f"for {args.config} {args.signal} {args.folder} {args.energy}."
)

# ── generate scan templates and score ──────────────────────────────────────────

cut_quality = []

def _score_cut(cut):
    """Score one cut, or return None when its templates are unusable."""
    nhits = cut["NHits"]
    adjcl = cut["AdjCl"]
    ophits = cut["OpHits"]

    solar_pkl = _pkl_path(nhits, adjcl, ophits, solar_dm2)
    react_pkl = _pkl_path(nhits, adjcl, ophits, react_dm2)

    if not _is_valid(solar_pkl) or not _is_valid(react_pkl) or not _sampling_current(cut):
        rprint(
            f"[yellow][WARNING][/yellow] Templates missing or stale after generation for "
            f"NHits{nhits} AdjCl{adjcl} OpHits{ophits}; skipping."
        )
        return None

    pred_solar = _load_template(solar_pkl)
    pred_react = _load_template(react_pkl)

    if args.background:
        bkg_pkl = f"{background_path}/{args.config}_background_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}.pkl"
        bkg = _load_template(bkg_pkl)
    else:
        bkg = np.zeros_like(pred_solar)

    # Scale per-year templates to expected counts. The '<1 expected event -> 0' floor is
    # applied for the legacy Gaussian fitter only (bb_mask=(b_t > 0): a bin with ~0 expected
    # background has an unbounded Gaussian chi2). The pull fit's Poisson deviance handles
    # small expectations, and flooring the background there throws the cleanest cells out of
    # the fit, so it would reward cuts that leave some background in every cell. Mirrors
    # 06_significance.scale_to_exposure.
    def _to_counts(arr):
        scaled = args.exposure * np.asarray(arr, dtype=float)
        if args.fit_method == "legacy":
            scaled[scaled < 1.0] = 0.0
        return scaled

    pred_solar_scaled = _to_counts(pred_solar)
    pred_react_scaled = _to_counts(pred_react)
    bkg_scaled        = _to_counts(bkg)

    obs_at_react = (pred_react_scaled + bkg_scaled)[:, thld:]
    obs_at_solar = (pred_solar_scaled + bkg_scaled)[:, thld:]
    p_s  = pred_solar_scaled[:, thld:]
    p_r  = pred_react_scaled[:, thld:]
    b_t  = bkg_scaled[:, thld:]

    if args.fit_method == "pull":
        # Same nuisances as the legacy scorer (signal + background normalisation only).
        chi2_solar_at_react = sensitivity_pull_chi2(
            obs_at_react, p_s, b_t,
            sigma_pred=args.signal_uncertainty,
            sigma_bkg=args.background_uncertainty,
        )["chi2"]
        chi2_react_at_solar = sensitivity_pull_chi2(
            obs_at_solar, p_r, b_t,
            sigma_pred=args.signal_uncertainty,
            sigma_bkg=args.background_uncertainty,
        )["chi2"]
    else:
        fitter_s = Sensitivity_Fitter(
            obs_at_react, p_s, b_t,
            SigmaPred=args.signal_uncertainty,
            SigmaBkg=args.background_uncertainty,
            bb_mask=(b_t > 0),
            fit_background=False,
            use_legacy_background_penalty=False,
        )
        chi2_solar_at_react, _, _ = fitter_s.Fit(0.0, 0.0)

        fitter_r = Sensitivity_Fitter(
            obs_at_solar, p_r, b_t,
            SigmaPred=args.signal_uncertainty,
            SigmaBkg=args.background_uncertainty,
            bb_mask=(b_t > 0),
            fit_background=False,
            use_legacy_background_penalty=False,
        )
        chi2_react_at_solar, _, _ = fitter_r.Fit(0.0, 0.0)

    if chi2_solar_at_react is None or chi2_react_at_solar is None:
        rprint(
            f"[yellow][WARNING][/yellow] Fitter returned None for "
            f"NHits{nhits} AdjCl{adjcl} OpHits{ophits}; skipping."
        )
        return None

    score = 0.5 * (float(chi2_solar_at_react) + float(chi2_react_at_solar))
    rprint(
        f"  NHits{nhits} AdjCl{adjcl} OpHits{ophits}  "
        f"solar@react={chi2_solar_at_react:.3f}  react@solar={chi2_react_at_solar:.3f}  "
        f"score={score:.3f}"
    )
    return {
        "NHits":             nhits,
        "AdjCl":             adjcl,
        "OpHits":            ophits,
        "SolarFitAtReact":   float(chi2_solar_at_react),
        "ReactorFitAtSolar": float(chi2_react_at_solar),
        "Score":             score,
    }


# ── search the cut grid ────────────────────────────────────────────────────────

_AXES = ("NHits", "AdjCl", "OpHits")


def _key(cut):
    return tuple(int(cut[k]) for k in _AXES)


def _axis_values(cuts):
    return {k: sorted({int(c[k]) for c in cuts}) for k in _AXES}


def _seed_grid(axes, stride, available):
    """Every `stride`-th value per axis, extremes always included."""
    picks = {}
    for k, values in axes.items():
        taken = list(values[::stride]) or list(values[:1])
        if values[-1] not in taken:
            taken.append(values[-1])
        picks[k] = taken
    return [
        {"NHits": n, "AdjCl": a, "OpHits": o}
        for n in picks["NHits"]
        for a in picks["AdjCl"]
        for o in picks["OpHits"]
        if (n, a, o) in available
    ]


def _neighbours(cut, axes, offset, available, scanned):
    """Unscanned grid positions `offset` lattice steps from `cut` along any axis."""
    index = {k: axes[k].index(int(cut[k])) for k in _AXES}
    out = []
    for dn in (-offset, 0, offset):
        for da in (-offset, 0, offset):
            for do in (-offset, 0, offset):
                if dn == da == do == 0:
                    continue
                position, inside = {}, True
                for k, delta in zip(_AXES, (dn, da, do)):
                    j = index[k] + delta
                    if not 0 <= j < len(axes[k]):
                        inside = False
                        break
                    position[k] = axes[k][j]
                if inside and _key(position) in available and _key(position) not in scanned:
                    out.append(position)
    return out


cut_quality = []
_scanned = set()
_available = {_key(c) for c in cut_candidates}
_axes = _axis_values(cut_candidates)


def _scan(batch, label):
    """Generate templates for the unscanned cuts in `batch`, then score them."""
    batch = [c for c in batch if _key(c) not in _scanned]
    if not batch:
        return []
    rprint(f"[cyan][INFO][/cyan] {label}: {len(batch)} cut(s)")
    stale = [
        c for c in batch
        if args.rewrite
        or not _is_valid(_pkl_path(c["NHits"], c["AdjCl"], c["OpHits"], solar_dm2))
        or not _is_valid(_pkl_path(c["NHits"], c["AdjCl"], c["OpHits"], react_dm2))
        or not _sampling_current(c)
    ]
    if stale and not _generate_templates(stale):
        rprint("[red][ERROR][/red] Template generation failed; aborting.")
        raise SystemExit(1)
    scored = []
    for cut in batch:
        _scanned.add(_key(cut))
        result = _score_cut(cut)
        if result is not None:
            scored.append(result)
            cut_quality.append(result)
    return scored


if args.scan_strategy == "full":
    _scan(cut_candidates, f"Full scan of {len(cut_candidates)} cut(s)")
else:
    _stride = max(1, int(round((1.0 / max(args.scan_fraction, 1e-9)) ** (1.0 / 3.0))))
    _scan(_seed_grid(_axes, _stride, _available), f"Coarse seed scan (stride {_stride})")
    _offset = max(1, _stride // 2)
    while cut_quality:
        _top = sorted(cut_quality, key=lambda item: item["Score"], reverse=True)[: max(1, args.scan_top)]
        rprint(
            "[cyan][INFO][/cyan] Leading cuts: "
            + ", ".join(
                f"NHits{t['NHits']} AdjCl{t['AdjCl']} OpHits{t['OpHits']} ({t['Score']:.3f})"
                for t in _top
            )
        )
        _batch, _seen = [], set()
        for _cut in _top:
            for _nb in _neighbours(_cut, _axes, _offset, _available, _scanned):
                if _key(_nb) not in _seen:
                    _seen.add(_key(_nb))
                    _batch.append(_nb)
        _scan(_batch, f"Refinement at offset {_offset} around the top {len(_top)}")
        if _offset == 1:
            break
        _offset = max(1, _offset // 2)
    rprint(
        f"[cyan][INFO][/cyan] Coarse-to-fine scan scored {len(_scanned)} of "
        f"{len(cut_candidates)} cuts ({100 * len(_scanned) / max(len(cut_candidates), 1):.1f}%)."
    )

if not cut_quality:
    rprint("[red][ERROR][/red] No valid cut candidates scored — cannot determine best cut.")
    raise SystemExit(1)

best = max(cut_quality, key=lambda x: x["Score"])
rprint(
    f"[cyan][INFO][/cyan] Best cut: NHits{best['NHits']} AdjCl{best['AdjCl']} OpHits{best['OpHits']} "
    f"(score={best['Score']:.3f})"
)

# ── save highest_SENSITIVITY.pkl and JSON ──────────────────────────────────────

best_payload = {
    (args.config, args.signal, args.energy): {
        "NHits":             int(best["NHits"]),
        "AdjCl":             int(best["AdjCl"]),
        "OpHits":            int(best["OpHits"]),
        "Score":             float(best["Score"]),
        "SolarFitAtReact":   float(best["SolarFitAtReact"]),
        "ReactorFitAtSolar": float(best["ReactorFitAtSolar"]),
    }
}

save_pkl(
    best_payload,
    f"{info['PATH']}/SENSITIVITY/",
    config=args.config,
    name=args.signal,
    subfolder=args.folder.lower(),
    filename=f"highest_SENSITIVITY{_study_suffix}",
    rm=args.rewrite,
    debug=args.debug,
)

# Print where the best-cut file was saved
best_cut_path = f"{info['PATH']}/SENSITIVITY/{args.config}/{args.signal}/{args.folder.lower()}/{args.config}_{args.signal}_highest_SENSITIVITY{_study_suffix}.pkl"
rprint(f"[cyan][INFO][/cyan] Best-cut file saved to: {best_cut_path}")

json_payload: dict = {}
for (cfg, nm, en), values in best_payload.items():
    # Flatten to config/energy level (all samples combined)
    # Best cuts are shared across all samples, not per-sample
    json_payload.setdefault(cfg, {}).setdefault(en, {}).update(values)

for local_dir in [
    f"{root}/config/{args.config}/sensitivity-json/{args.folder.lower()}",
    f"{root}/config/{args.config}/best-sigma-json/sensitivity/{args.folder.lower()}",
]:
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    merge_and_write_json(
        f"{local_dir}/{args.config}_highest_Sensitivity{_study_suffix}.json",
        json_payload,
        debug=args.debug,
    )
