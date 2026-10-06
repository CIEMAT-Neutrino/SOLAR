import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *
from lib.fiducial_significance import (
    FIDUCIAL_COLUMNS,
    combine_components,
    build_significance_payload,
    get_essential_background_components,
)

"""
03d_signal_background_threshold.py: signal/background classifier threshold from the analysis significance.

The threshold of 03b_signal_background.py (a signal completeness target) is replaced by the one that
maximises the significance of one analysis in its fiducial-stage region of interest: the same energy
window, signal and background components, significance type, smoothing, exposure and essential-background
MC gate as 02_best_fiducial.py, inside the fiducial volume the analysis currently uses.

- A cluster tagged background-like is not removed: its SolarEnergy loses the de-excitation offset. For
  every threshold SolarEnergy is rebuilt from the two versions kept by compute_energy_calibration
  (SolarEnergyOffset, SolarEnergyNoOffset) and the held-out classifier score (SignalBackgroundScore).
- Threshold grid: the score quantiles that keep a given fraction of the CC clusters with E_nu in [6, 20] MeV
  (the 03b completeness definition), plus threshold 0 (classifier off, every cluster keeps the offset).
- Points where an essential background has fewer than --mc_threshold MC clusters in the window are
  excluded (the classifier-off point is always kept, as the no-fiducial point of 02_best_fiducial.py).
- Tuning and check use different events: the threshold is chosen on fold 0 ((Flag + Event) % 2 == 0)
  and its significance is reported on fold 1. Each fold is scaled by 2 to the full exposure.
- Conservative choice: among the points whose fold-0 significance is within the bootstrap uncertainty of
  the fold-0 maximum, the one with the highest completeness (closest to the classifier off) is taken.
- Radiological is left out of the background by default (--exclude_radiological): without the topological
  cuts it outnumbers the signal in the window by ~1e7 and the significance stays near zero at every threshold.
- The choice is marked MCLimited when the MC gate excluded a point with a higher fold-0 significance than
  the chosen one, or when an essential background has fewer than 2 x --mc_threshold MC clusters there.
"""

parser = argparse.ArgumentParser(description="Choose the signal/background classifier threshold from the analysis significance")
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--name", type=str, default="marley_final", help="Signal sample (calibration sample)")
parser.add_argument("--backgrounds", nargs="+", default=["gamma_final", "neutron_final", "radiological_final"])
parser.add_argument("--folder", type=str, choices=["Reduced", "Truncated", "Nominal", "Test"], default="Truncated")
parser.add_argument("--fiducial_folder", type=str, choices=["Reduced", "Truncated", "Nominal", "Test"], default=None,
    help="Read BestFiducials.json of this folder instead of --folder.")
parser.add_argument("--analysis", type=str, choices=["DayNight", "HEP", "Sensitivity"], default="Sensitivity")
parser.add_argument("--energy", type=str, default="SolarEnergy")
parser.add_argument("--exposure", type=float, default=get_analysis_exposure(str(root), "Sensitivity", fallback=30.0),
    help="Exposure in years (as passed to 02_best_fiducial.py by run_sensitivity.py).")
parser.add_argument("--mc_threshold", type=float,
    default=get_analysis_threshold(str(root), "FIDUCIALIZATION", stage="MC", fallback=0.0))
parser.add_argument("--completeness", nargs="+", type=float,
    default=[0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99])
parser.add_argument("--bootstrap", type=int, default=100, help="Poisson bootstrap replicas for the significance uncertainty")
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--exclude_radiological", action=argparse.BooleanOptionalAction, default=True,
    help="Leave radiological out of the background of the significance (in the fiducial-stage selection, without the "
         "topological cuts, it outnumbers the signal in the window by ~1e7 and pins the significance near zero).")
parser.add_argument("--oscillation_backend", type=str, choices=["file", "prob3", "nufast"], default="nufast")
parser.add_argument("--write", action=argparse.BooleanOptionalAction, default=True,
    help="Write the chosen threshold to the classifier json (SB_THRESHOLD; the 03b value is kept as SB_THRESHOLD_COMPLETENESS).")
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()

config = args.config
analysis_key = args.analysis.upper()
analysis_info = load_analysis_info(str(root))
info = json.load(open(f"{root}/config/{config}/{config}_config.json"))
detector_x = info["DETECTOR_SIZE_X"] + 2 * info["DETECTOR_GAP_X"]
detector_y = info["DETECTOR_SIZE_Y"] + 2 * info["DETECTOR_GAP_Y"]
fiducials = json.load(open(f"{root}/config/analysis/fiducial/{(args.fiducial_folder or args.folder).lower()}/BestFiducials.json"))
fiducial = get_best_fiducial(fiducials, config, args.energy, analysis_key)

analysis_config = dict(get_fiducialization_config(str(root), analysis_key))
_sig_ref = str(analysis_info.get("BEST_SIGMA_SIGNIFICANCE_REFERENCE", {}).get(analysis_key, "")).lower()
if _sig_ref in {"asimov", "gaussian"}:
    analysis_config["significance_type"] = _sig_ref
smoothing_config = get_smoothing_config(str(root), analysis_name=analysis_key, dimensions="1d", stage="fiducial")
if args.exclude_radiological:
    analysis_config["background_components"] = [
        c for c in analysis_config.get("background_components", []) if not str(c).lower().startswith("radiological")
    ]
    args.backgrounds = [b for b in args.backgrounds if not b.lower().startswith("radiological")]
essential = get_essential_background_components(str(root), list(analysis_config.get("background_components", [])))
energy_min, energy_max = analysis_config.get("energy_min"), analysis_config.get("energy_max")

calib_json = f"{root}/config/{config}/{args.name}/{config}_calib/{config}_signal_background.json"
summary = json.load(open(calib_json))
n_bins = len(reco_energy_edges) - 1
weight_branches = {"8B": "SignalParticleWeightb8OscMean", "hep": "SignalParticleWeighthepOscMean"}


def load_sample(name):
    """Clusters of one sample inside the fiducial-stage selection, with both SolarEnergy versions."""
    configs = {config: [name]}
    run, output = load_multi(configs, preset="SIGNIFICANCE", branches={"Config": ["Geometry"]}, debug=args.debug)
    is_signal = name.startswith("marley")
    params = (
        {"DEFAULT_SIGNAL_WEIGHT": ["truth", "osc"], "DEFAULT_SIGNAL_NADIR": ["mean", "day", "night"],
         "PARTICLE_TYPE": "signal", "PARTICLE_WEIGHTING": "volume", "OSCILLATION_BACKEND": args.oscillation_backend}
        if is_signal
        else {"PARTICLE_TYPE": "background", "PARTICLE_WEIGHTING": "histogram"}
    )
    run = compute_reco_workflow(run, configs, params=params, workflow="SIGNIFICANCE", rm_branches=False, debug=args.debug)
    reco = run["Reco"]
    for branch in ["SignalBackgroundScore", "SolarEnergyOffset", "SolarEnergyNoOffset"]:
        if branch not in reco:
            raise KeyError(f"{config} {name}: {branch} missing (signal/background classifier off or not found)")
    # Same selection as 01_fiducialize.py: surface filter for the surface-generated backgrounds,
    # accepted optical-match plane with PE > 0; then the analysis fiducial volume.
    sample_key = name.split("_")[0].lower()
    mask = (
        surface_quality_mask(str(root), sample_key, args.folder, reco["SignalParticleSurface"])
        & accepted_flash_planes(reco["MatchedOpFlashPlane"], str(root), True)
        & (np.asarray(reco["MatchedOpFlashPE"]) > 0)
        & np.asarray(build_fiducial_spatial_mask(run, config, detector_x, detector_y, info, args.folder, fiducial), dtype=bool)
    )
    df = pd.DataFrame({
        "Score": np.asarray(reco["SignalBackgroundScore"], dtype=float)[mask],
        "BinOffset": np.digitize(np.asarray(reco["SolarEnergyOffset"], dtype=float)[mask], reco_energy_edges) - 1,
        "BinNoOffset": np.digitize(np.asarray(reco["SolarEnergyNoOffset"], dtype=float)[mask], reco_energy_edges) - 1,
        "Fold": ((np.asarray(reco["Flag"]).astype(np.int64) + np.asarray(reco["Event"]).astype(np.int64)) % 2)[mask],
    })
    if is_signal:
        for component, branch in weight_branches.items():
            df[component] = np.asarray(reco[branch], dtype=float)[mask]
        df["Reference"] = (
            (np.asarray(reco["Energy"], dtype=float) >= summary.get("MIN_ENERGY", 0.0))
            & (np.asarray(reco["SignalParticleK"], dtype=float) >= 6)
            & (np.asarray(reco["SignalParticleK"], dtype=float) <= 20)
            & ((np.asarray(reco["Generator"]) == 1) if "Generator" in reco else True)
        )[mask]
    else:
        df[sample_key] = np.asarray(reco["SignalParticleWeight"], dtype=float)[mask]
    rprint(f"{config}/{name}: {len(df)} clusters in the {analysis_key} fiducial selection")
    return df


def fill_gaps(hist):
    """As 01_fiducialize.py: half the weight of a non-zero bin moves to an empty bin on its left."""
    hist = np.asarray(hist, dtype=float)
    right, left = hist[1:], hist[:-1]
    transfer = right * 0.5 * ((right > 0) & (left == 0))
    hist[:-1] += transfer
    hist[1:] -= transfer
    return hist


samples = {name: load_sample(name) for name in [args.name] + args.backgrounds}
signal_df = samples[args.name]
reference_scores = signal_df.loc[signal_df["Reference"], "Score"].to_numpy()
grid = [(1.0, 0.0)] + [(c, float(np.quantile(reference_scores, 1 - c))) for c in sorted(args.completeness, reverse=True)]


def spectra(threshold, fold=None, poisson=None):
    """Fiducial-stage rows (Component, Energy, MCCounts, Counts, Error+/-) at one threshold."""
    rows = []
    for name, df in samples.items():
        keep = np.ones(len(df), dtype=bool) if fold is None else (df["Fold"].to_numpy() == fold)
        scale = 1.0 if fold is None else 2.0
        b = np.where(df["Score"].to_numpy() >= threshold, df["BinOffset"].to_numpy(), df["BinNoOffset"].to_numpy())
        keep &= (b >= 0) & (b < n_bins)
        resample = np.ones(len(df)) if poisson is None else poisson[name]
        h_raw = np.bincount(b[keep], weights=resample[keep], minlength=n_bins).astype(float)
        mc_counts = fill_gaps(h_raw)
        rel_error = np.zeros(n_bins)
        rel_error[h_raw > 0] = 1.0 / np.sqrt(h_raw[h_raw > 0])
        components = list(weight_branches) if name == args.name else [name.split("_")[0].lower()]
        if name != args.name:
            rel_error += 0.02
        for component in components:
            h = fill_gaps(scale * np.bincount(b[keep], weights=(df[component].to_numpy() * resample)[keep], minlength=n_bins))
            rows.append(pd.DataFrame({
                "Component": component, "Energy": reco_energy_centers, "MCCounts": mc_counts, "Counts": h,
                "Error+": h * rel_error, "Error-": h * rel_error,
                "FiducializedX": int(fiducial["FiducialX"]), "FiducializedY": int(fiducial["FiducialY"]),
                "FiducializedZ": int(fiducial["FiducialZ"]),
            }))
    return pd.concat(rows, ignore_index=True)


def significance(rows):
    """Smoothed significance of 02_best_fiducial.select_best_fiducial for one volume."""
    merged = pd.merge(
        combine_components(rows, analysis_config.get("signal_components", [])),
        combine_components(rows, analysis_config.get("background_components", [])),
        on=["Energy"] + FIDUCIAL_COLUMNS, how="outer", suffixes=("Signal", "Background"),
    ).fillna(0.0)
    payload = build_significance_payload(
        merged, analysis_config.get("significance_type", "gaussian"), analysis_config.get("combine_mode", "quadrature"),
        smoothing_config, args.exposure, energy_min, energy_max,
    )
    if payload is None:
        return 0.0, 0.0, 0.0
    window = np.ones(len(payload["Energy"]), dtype=bool)
    if energy_min is not None:
        window &= payload["Energy"] >= float(energy_min)
    if energy_max is not None:
        window &= payload["Energy"] <= float(energy_max)
    return float(payload["SmoothedTotal"]), float(np.sum(payload["RawSignal"][window])), float(np.sum(payload["RawBackground"][window]))


def essential_mc(rows):
    window = rows
    if energy_min is not None:
        window = window.loc[window["Energy"] >= float(energy_min)]
    if energy_max is not None:
        window = window.loc[window["Energy"] <= float(energy_max)]
    # MCCounts is repeated for every weight component of a sample; one component per sample here.
    return {c: float(window.loc[window["Component"] == c, "MCCounts"].sum()) for c in essential}


rng = np.random.default_rng(args.seed)
replicas = [{name: rng.poisson(1.0, len(df)).astype(float) for name, df in samples.items()} for _ in range(args.bootstrap)]
scan = []
for completeness, threshold in grid:
    full = spectra(threshold)
    mc = essential_mc(full)
    z, s, b = significance(full)
    z0 = significance(spectra(threshold, fold=0))[0]
    z0_boot = [significance(spectra(threshold, fold=0, poisson=r))[0] for r in replicas]
    scan.append({
        "Completeness": completeness, "Threshold": threshold, "Signal": s, "Background": b,
        "Significance": z, "SignificanceFold0": z0, "SignificanceFold1": significance(spectra(threshold, fold=1))[0],
        "SigmaFold0": float(np.std(z0_boot)), "EssentialMC": mc,
        "PassGate": bool(threshold == 0.0 or all(v >= args.mc_threshold for v in mc.values())),
    })
scan_df = pd.DataFrame(scan)

# Window content per component, classifier off and at the 03b threshold.
for label, threshold in [("classifier off", 0.0), ("03b threshold", summary.get("SB_THRESHOLD_COMPLETENESS", summary["SB_THRESHOLD"]))]:
    rows = spectra(threshold)
    window = rows.loc[(rows["Energy"] >= float(energy_min or -np.inf)) & (rows["Energy"] <= float(energy_max or np.inf))]
    rprint(f"  window content, {label} (threshold {threshold:.3f}, x{args.exposure:g} yr): " + ", ".join(
        f"{c} {args.exposure * window.loc[window['Component'] == c, 'Counts'].sum():.4g} ({window.loc[window['Component'] == c, 'MCCounts'].sum():.0f} MC)"
        for c in rows["Component"].unique()))

gated = scan_df.loc[scan_df["PassGate"]]
best = gated.loc[gated["SignificanceFold0"].idxmax()]
plateau = gated.loc[gated["SignificanceFold0"] >= best["SignificanceFold0"] - best["SigmaFold0"]]
chosen = plateau.loc[plateau["Completeness"].idxmax()]
ungated_best = scan_df.loc[scan_df["SignificanceFold0"].idxmax()]
reasons = []
if not bool(ungated_best["PassGate"]) and ungated_best["SignificanceFold0"] > chosen["SignificanceFold0"]:
    reasons.append(f"MC gate excluded completeness {ungated_best['Completeness']:.3f} (fold-0 significance {ungated_best['SignificanceFold0']:.3f})")
low = {c: v for c, v in chosen["EssentialMC"].items() if v < 2 * args.mc_threshold}
if low and chosen["Threshold"] > 0:
    reasons.append("essential MC below 2x threshold: " + ", ".join(f"{c} {v:.0f}" for c, v in low.items()))

rprint(f"\n{config} {analysis_key} {args.energy} ({args.folder}, volume X={fiducial['FiducialX']} Y={fiducial['FiducialY']} "
       f"Z={fiducial['FiducialZ']}, {args.exposure:g} yr, {analysis_config.get('significance_type')}, window {energy_min}-{energy_max} MeV)")
rprint("  completeness  threshold   S(window)   B(window)   Z(all)  Z(fold0) +- sigma  Z(fold1)  gate  essential MC")
for row in scan_df.itertuples():
    mark = "  <- chosen" if row.Threshold == chosen["Threshold"] else ("  <- fold-0 max" if row.Threshold == best["Threshold"] else "")
    rprint(f"  {row.Completeness:12.3f}  {row.Threshold:9.3f}  {row.Signal:10.4g}  {row.Background:10.4g}  {row.Significance:7.3f}  {row.SignificanceFold0:8.3f} +- {row.SigmaFold0:5.3f}"
           f"  {row.SignificanceFold1:8.3f}  {'ok' if row.PassGate else 'MC':>4}  "
           + " ".join(f"{c}:{v:.0f}" for c, v in row.EssentialMC.items()) + mark)
if reasons:
    rprint(f"[yellow][MC-LIMITED][/yellow] {config} {analysis_key}: " + "; ".join(reasons))

result = {
    "ANALYSIS": analysis_key, "ENERGY": args.energy, "FOLDER": args.folder,
    "FIDUCIAL": {k: fiducial[k] for k in ["FiducialX", "FiducialY", "FiducialZ"]},
    "EXPOSURE": args.exposure, "SIGNIFICANCE_TYPE": analysis_config.get("significance_type"),
    "BACKGROUND_COMPONENTS": list(analysis_config.get("background_components", [])),
    "ENERGY_WINDOW": [energy_min, energy_max], "MC_THRESHOLD": args.mc_threshold, "BOOTSTRAP": args.bootstrap,
    "RULE": "highest completeness with fold-0 significance >= fold-0 maximum - its bootstrap sigma (MC-gated); reported on fold 1",
    "CHOSEN_COMPLETENESS": float(chosen["Completeness"]), "CHOSEN_THRESHOLD": float(chosen["Threshold"]),
    "MCLimited": bool(reasons), "MCLimitedReason": "; ".join(reasons),
    "SCAN": scan_df.to_dict(orient="records"),
}
data_path = f"{root}/output/data/workflow/discrimination"
save_df(pd.DataFrame([{"Geometry": info["GEOMETRY"], "Config": config, "Name": args.name, **{k: v for k, v in result.items() if k != "SCAN"},
                       **{c: list(scan_df[c]) for c in ["Completeness", "Threshold", "Signal", "Background", "Significance", "SignificanceFold0",
                                                         "SignificanceFold1", "SigmaFold0", "PassGate"]}}]),
        data_path, config, args.name, filename="Signal_Background_Threshold", rm=args.rewrite, debug=args.debug)
if args.write:
    summary.setdefault("SB_THRESHOLD_COMPLETENESS", summary["SB_THRESHOLD"])
    summary["SB_THRESHOLD"] = float(chosen["Threshold"])
    summary["THRESHOLD_SCAN"] = result
    with open(calib_json, "w") as f:
        json.dump(summary, f, indent=2)
    rprint(f"Wrote SB_THRESHOLD {chosen['Threshold']:.3f} (completeness {chosen['Completeness']:.3f}) to {calib_json}")
