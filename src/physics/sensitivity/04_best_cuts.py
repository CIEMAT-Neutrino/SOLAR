import os
import re
import sys
import subprocess
from glob import glob as glob_files
from shlex import quote

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *
from lib.root import Sensitivity_Fitter

parser = argparse.ArgumentParser(
    description="Cut optimisation for Sensitivity analysis. "
    "Generates signal templates at solar+reactor reference points for every "
    "background-template cut candidate, scores each cut with the Sensitivity_Fitter "
    "chi2 figure-of-merit, and writes highest_SENSITIVITY.pkl."
)
parser.add_argument("--config", type=str, default="hd_1x2x6_centralAPA")
parser.add_argument("--name",   type=str, default="marley")
parser.add_argument(
    "--folder", type=str, choices=["Reduced", "Truncated", "Nominal"], default="Nominal"
)
parser.add_argument(
    "--energy",
    type=str,
    choices=["SignalParticleK", "ClusterEnergy", "TotalEnergy", "SelectedEnergy", "SolarEnergy"],
    default="SolarEnergy",
)
parser.add_argument(
    "--oscillation_backend",
    type=str, choices=["file", "prob3", "nufast"], default="file",
)
parser.add_argument("--exposure",              type=float, default=30.0)
parser.add_argument("--signal_uncertainty",    type=float, default=None)
parser.add_argument("--background_uncertainty",type=float, default=None)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug",   action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--plot",    action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--background", action=argparse.BooleanOptionalAction, default=True)
args = parser.parse_args()

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

signal_path     = f"{info['PATH']}/SENSITIVITY/{args.config}/{args.name}/{args.folder.lower()}/{args.energy}"
background_path = f"{info['PATH']}/SENSITIVITY/{args.config}/background/{args.folder.lower()}/{args.energy}"


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
        f"{signal_path}/{args.config}_{args.name}"
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


def _generate_templates(cuts):
    cmd = [
        "python3", f"{root}/src/physics/sensitivity/02_signal_template.py",
        "--config",               args.config,
        "--name",                 args.name,
        "--folder",               args.folder,
        "--energy",               args.energy,
        "--cuts",                 json.dumps(cuts),
        "--exposure",             str(args.exposure),
        "--oscillation_backend",  args.oscillation_backend,
        "--scan_mode",
        "--rewrite" if args.rewrite else "--no-rewrite",
        "--no-plot",
        "--no-debug",
        "--no-test",
    ]
    cmd_str = " ".join(quote(str(c)) for c in cmd)
    rprint(f"\n[green][CMD][/green] {cmd_str}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        rprint(
            f"[yellow][WARNING][/yellow] sensitivity/02_signal_template.py failed "
            f"(exit {result.returncode})"
        )
        return False
    return True


def _load_template(path: str):
    return np.nan_to_num(np.asarray(pd.read_pickle(path), dtype=float), nan=0.0)


# ── discover cuts ───────────────────────────────────────────────────────────────

cut_candidates = [c for c in (_parse_cut(f) for f in _background_candidates()) if c is not None]
if not cut_candidates:
    rprint(
        f"[red][ERROR][/red] No background templates found in {background_path}. "
        "Run sensitivity/01_background_template.py first."
    )
    raise SystemExit(1)

rprint(
    f"[cyan][INFO][/cyan] Found {len(cut_candidates)} background-template cut candidates "
    f"for {args.config} {args.name} {args.folder} {args.energy}."
)

# ── generate scan templates and score ──────────────────────────────────────────

cut_quality = []

# Collect stale cuts and regenerate in a single subprocess call
stale_cuts = [
    c for c in cut_candidates
    if args.rewrite
    or not _is_valid(_pkl_path(c["NHits"], c["AdjCl"], c["OpHits"], solar_dm2))
    or not _is_valid(_pkl_path(c["NHits"], c["AdjCl"], c["OpHits"], react_dm2))
]
if stale_cuts:
    ok = _generate_templates(stale_cuts)
    if not ok:
        rprint("[red][ERROR][/red] Template generation failed; aborting.")
        raise SystemExit(1)

for cut in cut_candidates:
    nhits = cut["NHits"]
    adjcl = cut["AdjCl"]
    ophits = cut["OpHits"]

    solar_pkl = _pkl_path(nhits, adjcl, ophits, solar_dm2)
    react_pkl = _pkl_path(nhits, adjcl, ophits, react_dm2)

    if not _is_valid(solar_pkl) or not _is_valid(react_pkl):
        rprint(
            f"[yellow][WARNING][/yellow] Templates missing or stale after generation for "
            f"NHits{nhits} AdjCl{adjcl} OpHits{ophits}; skipping."
        )
        continue

    pred_solar = _load_template(solar_pkl)
    pred_react = _load_template(react_pkl)

    if args.background:
        bkg_pkl = f"{background_path}/{args.config}_background_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}.pkl"
        bkg = _load_template(bkg_pkl)
    else:
        bkg = np.zeros_like(pred_solar)

    # Scale to expected counts
    pred_solar_scaled = args.exposure * pred_solar
    pred_react_scaled = args.exposure * pred_react
    bkg_scaled        = args.exposure * bkg

    obs_at_react = (pred_react_scaled + bkg_scaled)[:, thld:]
    obs_at_solar = (pred_solar_scaled + bkg_scaled)[:, thld:]
    p_s  = pred_solar_scaled[:, thld:]
    p_r  = pred_react_scaled[:, thld:]
    b_t  = bkg_scaled[:, thld:]

    fitter_s = Sensitivity_Fitter(
        obs_at_react, p_s, b_t,
        SigmaPred=args.signal_uncertainty,
        SigmaBkg=args.background_uncertainty,
        bb_mask=(b_t > 0),
    )
    chi2_solar_at_react, _, _ = fitter_s.Fit(0.0, 0.0)

    fitter_r = Sensitivity_Fitter(
        obs_at_solar, p_r, b_t,
        SigmaPred=args.signal_uncertainty,
        SigmaBkg=args.background_uncertainty,
        bb_mask=(b_t > 0),
    )
    chi2_react_at_solar, _, _ = fitter_r.Fit(0.0, 0.0)

    if chi2_solar_at_react is None or chi2_react_at_solar is None:
        rprint(
            f"[yellow][WARNING][/yellow] Fitter returned None for "
            f"NHits{nhits} AdjCl{adjcl} OpHits{ophits}; skipping."
        )
        continue

    score = 0.5 * (float(chi2_solar_at_react) + float(chi2_react_at_solar))
    cut_quality.append({
        "NHits":             nhits,
        "AdjCl":             adjcl,
        "OpHits":            ophits,
        "SolarFitAtReact":   float(chi2_solar_at_react),
        "ReactorFitAtSolar": float(chi2_react_at_solar),
        "Score":             score,
    })
    rprint(
        f"  NHits{nhits} AdjCl{adjcl} OpHits{ophits}  "
        f"solar@react={chi2_solar_at_react:.3f}  react@solar={chi2_react_at_solar:.3f}  "
        f"score={score:.3f}"
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
    (args.config, args.name, args.energy): {
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
    f"{info['PATH']}/SENSITIVITY/{args.folder.lower()}",
    config=args.config,
    name=args.name,
    filename="highest_SENSITIVITY",
    rm=args.rewrite,
    debug=args.debug,
)

json_payload: dict = {}
for (cfg, nm, en), values in best_payload.items():
    # Flatten to config/energy level (all samples combined)
    # Best cuts are shared across all samples, not per-sample
    json_payload.setdefault(cfg, {}).setdefault(en, {}).update(values)

for local_dir in [
    f"{root}/data/analysis/sensitivity-json/{args.folder.lower()}/{args.config}",
    f"{root}/data/analysis/best-sigma-json/sensitivity/{args.folder.lower()}/{args.config}",
]:
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    merge_and_write_json(
        f"{local_dir}/{args.config}_highest_Sensitivity.json",
        json_payload,
        debug=args.debug,
    )
