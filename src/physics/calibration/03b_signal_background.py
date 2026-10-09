import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

"""
03b_signal_background.py: signal/background cluster classifier (first SolarEnergy stage).

A random forest separates signal clusters (marley sample, Generator == 1) from the background
samples (gamma, neutron, radiological) with the reconstructed features used by the energy
discriminant. In the analysis (lib/cluster.compute_reco_energy) clusters scored below the
threshold keep their cluster energy (no de-excitation offset), the others go through the
upper/lower branch classifier as before.

- Clusters are loaded with the analysis chain (ANALYSIS workflow, no preset filter, as 03_analysis), so the
  features are exactly those seen by the analysis; only reconstructed Energy >= --min_energy enters.
- Two folds by event ((Flag + Event) % 2): the model of fold k is trained on fold k and applied to
  fold 1-k, so no background (or signal) cluster is ever scored by a model that saw it.
- Class weights: signal and background classes have equal total weight; each background sample
  has an equal share of the background class (uniform within a sample).
- Threshold: the largest score cut that keeps --completeness of the signal clusters with
  E_nu in [6, 20] MeV (completeness favoured over purity), over the whole range or, with
  --completeness_per_bin, in every E_nu bin.
- HEP working point: the HEP analysis reads SolarEnergyHEP, where clusters whose SolarEnergy (with the offset) is at
  least --hep_gate_energy use SB_THRESHOLD_HEP (--hep_completeness of the CC clusters above that gate). Above the gate
  the external gammas carry most of the HEP background; tuned on fold-split HEP significance (2026-10-08, _xchk/heptune):
  HD Lateral +14%, VD Shielded +26-30%, HD Central and VD Top unchanged within the MC noise. DayNight and Sensitivity
  keep SolarEnergy.
- Alternative HEP classifier (--target hep): a second random forest, trained only on clusters with Energy >=
  --hep_min_energy (8 MeV) against the external backgrounds (gamma, neutron; radiological left out), with the CC
  clusters weighted by their hep spectral weight (SignalParticleWeighthepOscMean, SIGNIFICANCE workflow) so that it
  learns hep-like clusters rather than the 8B mix. Same inputs as the default classifier (its EXCLUDED_FEATURES), same
  folds, same per-sample background weights. Threshold SB_HEP_MODEL_THRESHOLD keeps --hep_model_completeness of the
  CC clusters whose SolarEnergy (with the offset) is at least --hep_model_gate_energy. Writes
  <config>_signal_background_hep.json and the fold models *_signal_background_hep_fold{0,1}.pkl; the default
  classifier and its outputs are not touched. Used only by the HEP study variant hep_sbmodel (03_analysis.py
  --hep_energy SolarEnergyHEPModel). Fold-split test (2026-10-08, _xchk/heptune "hepSB gate12 C0.8"): VD Shielded
  +44%, HD Lateral +7%, HD Central +1%, VD Top -11%.
- High-energy threshold: clusters whose SolarEnergy (with the offset) is at least --gate_energy use
  SB_THRESHOLD_HIGH, the largest score cut that keeps --completeness_high of the signal clusters above the
  gate (never looser than SB_THRESHOLD). Above the gate the backgrounds left after the topological cuts are
  external gammas and neutron captures, and the CC clusters there are the most signal-like, so a stricter
  cut costs little signal (study of 2026-10-07, _xchk/hep). --gate_energy 0 switches it off.
"""

parser = argparse.ArgumentParser(description="Train the signal/background cluster classifier")
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--name", type=str, default="marley_final", help="Signal sample (calibration sample)")
parser.add_argument("--backgrounds", nargs="+", default=["gamma_final", "neutron_final", "radiological_final"])
parser.add_argument("--completeness", type=float, default=0.95)
parser.add_argument("--completeness_per_bin", action=argparse.BooleanOptionalAction, default=False,
    help="Apply the completeness target in every E_nu bin (6-8, 8-10, 10-14, 14-20 MeV) instead of 6-20 MeV overall.")
parser.add_argument("--exclude_features", nargs="*", default=[],
    help="Inputs left out of the classifier (HD Central: the matched-flash inputs, whose gamma/neutron samples lack the light overlay).")
parser.add_argument("--min_energy", type=float, default=3.0)
parser.add_argument("--gate_energy", type=float, default=12.0,
    help="SolarEnergy (with the offset, MeV) above which SB_THRESHOLD_HIGH applies; 0 switches the high-energy threshold off.")
parser.add_argument("--hep_gate_energy", type=float, default=14.0,
    help="SolarEnergy (with the offset, MeV) above which the HEP analysis uses SB_THRESHOLD_HEP; 0 switches it off.")
parser.add_argument("--hep_completeness", type=float, default=0.80,
    help="CC completeness above --hep_gate_energy of SB_THRESHOLD_HEP (HEP-only energy, SolarEnergyHEP).")
parser.add_argument("--completeness_high", type=float, default=0.95,
    help="Signal completeness of SB_THRESHOLD_HIGH among the CC clusters above --gate_energy.")
parser.add_argument("--target", choices=["default", "hep"], default="default",
    help="default: the S/B classifier of SolarEnergy; hep: the alternative HEP-trained classifier (SolarEnergyHEPModel).")
parser.add_argument("--hep_min_energy", type=float, default=8.0, help="--target hep: minimum cluster Energy of the training sample.")
parser.add_argument("--hep_model_gate_energy", type=float, default=12.0,
    help="--target hep: SolarEnergy (with the offset, MeV) above which the HEP classifier replaces the default one.")
parser.add_argument("--hep_model_completeness", type=float, default=0.80,
    help="--target hep: CC completeness above --hep_model_gate_energy of SB_HEP_MODEL_THRESHOLD.")
parser.add_argument("--max_per_sample", type=int, default=300000)
parser.add_argument("--rf_estimators", type=int, default=300)
parser.add_argument("--rf_depth", type=int, default=10)
parser.add_argument("--rf_min_leaf", type=int, default=20)
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()

config = args.config
HEP = args.target == "hep"
if HEP:
    # External backgrounds only; the inputs of the default classifier (its excluded features unless given).
    args.backgrounds = [b for b in args.backgrounds if not b.startswith("radiological")]
    _default_json = f"{root}/config/{config}/{args.name}/{config}_calib/{config}_signal_background.json"
    if not args.exclude_features and os.path.isfile(_default_json):
        args.exclude_features = list(json.load(open(_default_json)).get("EXCLUDED_FEATURES", []))
    args.min_energy = args.hep_min_energy
analysis_info = load_analysis_info(str(root))
_models_base = analysis_info.get("DATA_PATHS", {}).get(
    "CALIBRATION", "/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/calibration"
)
base_features = list(get_default_info(root, "ML_FEATURES"))
optional_features = ["MatchedOpFlashPE", "MatchedOpFlashNHits", "MatchedOpFlashMaxPE"]


def load_sample(name):
    configs = {config: [name]}
    # --target hep needs the hep spectral weight of the CC clusters: SIGNIFICANCE workflow (as 03_analysis.py).
    workflow = "SIGNIFICANCE" if HEP else "ANALYSIS"
    run, output = load_multi(configs, preset=workflow, branches={"Config": ["Geometry"]}, debug=args.debug)
    params = (
        {"PARTICLE_TYPE": "signal", "SB_CLASSIFIER": False}
        if name.startswith("marley")
        else {"PARTICLE_TYPE": "background", "SB_CLASSIFIER": False}
    )
    if HEP and name.startswith("marley"):
        params.update({"DEFAULT_SIGNAL_WEIGHT": ["truth", "osc"], "DEFAULT_SIGNAL_NADIR": ["mean", "day", "night"],
                       "PARTICLE_WEIGHTING": "volume", "OSCILLATION_BACKEND": "nufast"})
    elif HEP:
        params["PARTICLE_WEIGHTING"] = "histogram"
    run = compute_reco_workflow(run, configs, params=params, workflow=workflow, rm_branches=False, debug=args.debug)
    # No preset filter: 03_analysis.py applies none (the VD SIGNIFICANCE filter would drop every
    # surface-generated gamma/neutron); only the analysed energy range is kept below.
    reco = run["Reco"]
    features = base_features + [f for f in optional_features if f in reco and f not in base_features]
    features = [f for f in features if f not in args.exclude_features]
    df = pd.DataFrame({f: np.asarray(reco[f], dtype=float) for f in features})
    df["Flag"] = np.asarray(reco["Flag"]).astype(np.int64)
    df["Event"] = np.asarray(reco["Event"]).astype(np.int64)
    df["SignalParticleK"] = np.asarray(reco["SignalParticleK"], dtype=float)
    df["SolarEnergy"] = np.asarray(reco["SolarEnergy"], dtype=float)  # classifier off: the offset version
    df["Generator"] = np.asarray(reco["Generator"])
    df["HepWeight"] = np.asarray(reco["SignalParticleWeighthepOscMean"], dtype=float) if (HEP and name.startswith("marley")) else 0.0
    df["Sample"] = name
    df = df[(df["Energy"] >= args.min_energy) & (df["SignalParticleK"].abs() <= 30)]
    if name.startswith("marley"):
        df = df[df["Generator"] == 1]
    if len(df) > args.max_per_sample:
        df = df.sample(n=args.max_per_sample, random_state=args.seed)
    rprint(f"{config}/{name}: {len(df)} clusters with Energy >= {args.min_energy} MeV")
    return df, features


frames, features = [], None
for name in [args.name] + args.backgrounds:
    df, f = load_sample(name)
    features = f if features is None else [x for x in features if x in f]
    frames.append(df)
data = pd.concat(frames, ignore_index=True)
data[features] = data[features].replace([np.inf, -np.inf], np.nan).fillna(0)
data["Label"] = (data["Sample"] == args.name).astype(int)
data["Fold"] = ((data["Flag"] + data["Event"]) % 2).astype(int)

# Class weights: equal totals for signal and background, equal share per background sample.
weights = np.zeros(len(data))
sig = data["Label"] == 1
weights[sig.values] = 1.0 / max(sig.sum(), 1)
if HEP:  # CC clusters weighted by their hep spectral weight (same total)
    hw = np.clip(data.loc[sig, "HepWeight"].to_numpy(dtype=float), 0, None)
    weights[sig.values] = hw / hw.sum() if hw.sum() > 0 else 1.0 / max(sig.sum(), 1)
for name in args.backgrounds:
    m = (data["Sample"] == name).values
    if m.sum():
        weights[m] = 1.0 / (len(args.backgrounds) * m.sum())
data["W"] = weights / weights.mean()

scores = np.full(len(data), np.nan)
models = {}
for fold in (0, 1):
    train = data["Fold"] == fold
    model = RandomForestClassifier(
        n_estimators=args.rf_estimators,
        max_depth=args.rf_depth,
        min_samples_leaf=args.rf_min_leaf,
        random_state=args.seed + fold,
        n_jobs=4,
    )
    model.fit(data.loc[train, features], data.loc[train, "Label"], sample_weight=data.loc[train, "W"])
    apply = data["Fold"] == 1 - fold
    scores[apply.values] = model.predict_proba(data.loc[apply, features])[:, 1]
    models[fold] = model
data["Score"] = scores

if HEP:
    # Alternative HEP classifier: threshold, summary and models only (no plotting exports).
    gate = sig & (data["SolarEnergy"] >= args.hep_model_gate_energy)
    if gate.sum() < 100:
        raise SystemExit(f"[ERROR] {config}: only {gate.sum()} CC clusters above {args.hep_model_gate_energy} MeV")
    threshold = float(np.quantile(data.loc[gate, "Score"], 1 - args.hep_model_completeness))
    auc = roc_auc_score(data["Label"], data["Score"], sample_weight=data["W"])
    summary = {"SB_HEP_MODEL_THRESHOLD": threshold, "SB_HEP_MODEL_GATE_ENERGY": args.hep_model_gate_energy,
               "TARGET_COMPLETENESS": args.hep_model_completeness, "FEATURES": features,
               "EXCLUDED_FEATURES": list(args.exclude_features), "MIN_ENERGY": args.min_energy, "AUC": float(auc),
               "SIGNAL_WEIGHT": "SignalParticleWeighthepOscMean", "FOLD_RULE": "(Flag + Event) % 2; model k applied to fold 1-k",
               "SIGNAL_SAMPLE": args.name, "BACKGROUND_SAMPLES": args.backgrounds, "SEED": args.seed,
               "RF": {"n_estimators": args.rf_estimators, "max_depth": args.rf_depth, "min_samples_leaf": args.rf_min_leaf},
               "FLAGGED_SIGNAL_LIKE_ABOVE_GATE": {}}
    for name in [args.name] + args.backgrounds:
        m = (data["Sample"] == name) & (data["SolarEnergy"] >= args.hep_model_gate_energy)
        summary["FLAGGED_SIGNAL_LIKE_ABOVE_GATE"][name] = float(np.mean(data.loc[m, "Score"] >= threshold)) if m.sum() else None
    rprint(json.dumps(summary, indent=2))
    model_dir = f"{_models_base}/{config}/{args.name}/models"
    os.makedirs(model_dir, exist_ok=True)
    for fold, model in models.items():
        path = f"{model_dir}/{config}_{args.name}_signal_background_hep_fold{fold}.pkl"
        if os.path.isfile(path):  # pnfs cannot overwrite in place
            os.remove(path)
        with open(path, "wb") as f:
            pickle.dump(model, f)
    calib_dir = f"{root}/config/{config}/{args.name}/{config}_calib"
    os.makedirs(calib_dir, exist_ok=True)
    with open(f"{calib_dir}/{config}_signal_background_hep.json", "w") as f:
        json.dump(summary, f, indent=2)
    rprint(f"Saved HEP signal/background classifier ({len(features)} features, AUC {auc:.3f}) to {model_dir} and {calib_dir}")
    sys.exit(0)

# Completeness-driven threshold on held-out signal clusters with E_nu in [6, 20] MeV.
ref = sig & (data["SignalParticleK"] >= 6) & (data["SignalParticleK"] <= 20)
if args.completeness_per_bin:
    # Every E_nu bin keeps at least the target completeness (the lowest bins are the hardest).
    _thr = []
    for lo, hi in [(6, 8), (8, 10), (10, 14), (14, 20)]:
        m = sig & (data["SignalParticleK"] >= lo) & (data["SignalParticleK"] < hi)
        if m.sum() >= 100:
            _thr.append(float(np.quantile(data.loc[m, "Score"], 1 - args.completeness)))
    threshold = min(_thr)
else:
    threshold = float(np.quantile(data.loc[ref, "Score"], 1 - args.completeness))
threshold_high, high = None, sig & (data["SolarEnergy"] >= args.gate_energy)
thresholds_high = {}
if args.gate_energy > 0 and high.sum() >= 100:
    threshold_high = max(threshold, float(np.quantile(data.loc[high, "Score"], 1 - args.completeness_high)))
    # Other working points, for the studies (03_analysis.py --sb_completeness_high).
    for c in sorted({0.80, 0.85, 0.90, 0.95, args.completeness_high}):
        thresholds_high[f"{c:.2f}"] = max(threshold, float(np.quantile(data.loc[high, "Score"], 1 - c)))
threshold_hep, hep_region = None, sig & (data["SolarEnergy"] >= args.hep_gate_energy)
if args.hep_gate_energy > 0 and hep_region.sum() >= 100:
    threshold_hep = max(threshold, float(np.quantile(data.loc[hep_region, "Score"], 1 - args.hep_completeness)))
auc = roc_auc_score(data["Label"], data["Score"], sample_weight=data["W"])
summary = {"SB_THRESHOLD": threshold, "TARGET_COMPLETENESS": args.completeness, "COMPLETENESS_PER_BIN": args.completeness_per_bin, "FEATURES": features,
           "EXCLUDED_FEATURES": list(args.exclude_features),
           "MIN_ENERGY": args.min_energy, "AUC": float(auc), "FOLD_RULE": "(Flag + Event) % 2; model k applied to fold 1-k",
           "SIGNAL_SAMPLE": args.name, "BACKGROUND_SAMPLES": args.backgrounds, "SEED": args.seed,
           "RF": {"n_estimators": args.rf_estimators, "max_depth": args.rf_depth, "min_samples_leaf": args.rf_min_leaf},
           "FLAGGED_SIGNAL_LIKE": {}}
if threshold_hep is not None:
    summary.update({"SB_HEP_GATE_ENERGY": args.hep_gate_energy, "SB_THRESHOLD_HEP": threshold_hep,
                    "TARGET_COMPLETENESS_HEP": args.hep_completeness, "FLAGGED_SIGNAL_LIKE_HEP": {}})
    for name in [args.name] + args.backgrounds:
        m = (data["Sample"] == name) & (data["SolarEnergy"] >= args.hep_gate_energy)
        summary["FLAGGED_SIGNAL_LIKE_HEP"][name] = float(np.mean(data.loc[m, "Score"] >= threshold_hep)) if m.sum() else None
if threshold_high is not None:
    summary.update({"SB_GATE_ENERGY": args.gate_energy, "SB_THRESHOLD_HIGH": threshold_high,
                    "SB_THRESHOLD_HIGH_BY_COMPLETENESS": thresholds_high,
                    "TARGET_COMPLETENESS_HIGH": args.completeness_high, "FLAGGED_SIGNAL_LIKE_HIGH": {}})
    for name in [args.name] + args.backgrounds:
        m = (data["Sample"] == name) & (data["SolarEnergy"] >= args.gate_energy)
        summary["FLAGGED_SIGNAL_LIKE_HIGH"][name] = float(np.mean(data.loc[m, "Score"] >= threshold_high)) if m.sum() else None
for name in [args.name] + args.backgrounds:
    m = data["Sample"] == name
    summary["FLAGGED_SIGNAL_LIKE"][name] = float(np.mean(data.loc[m, "Score"] >= threshold))
for lo, hi in [(6, 8), (8, 10), (10, 14), (14, 20)]:
    m = sig & (data["SignalParticleK"] >= lo) & (data["SignalParticleK"] < hi)
    summary["FLAGGED_SIGNAL_LIKE"][f"{args.name}_Enu_{lo}_{hi}"] = float(np.mean(data.loc[m, "Score"] >= threshold))
rprint(json.dumps(summary, indent=2))

# Exports for the plotting repository (same layout as 03_discrimination.py: Cluster_Discriminant /
# Random_Forest_Importance), plus the signal-like fraction versus reconstructed energy.
data_path = f"{root}/output/data/workflow/discrimination"
info = json.load(open(f"{root}/config/{config}/{config}_config.json"))
label = lambda n: "CC Neutrino" if n == args.name else n.split("_")[0].capitalize()
common = {"Geometry": info["GEOMETRY"], "Config": config, "Name": args.name}
disc_rows, eff_rows, sel_rows = [], [], []
energy_edges = np.arange(3, 31, 1.0)
matched_all = (data["MatchedOpFlashPE"] > 0) if "MatchedOpFlashPE" in data else pd.Series(True, index=data.index)
for population, pmask in [("All", pd.Series(True, index=data.index)), ("Matched", matched_all)]:
    for n in [args.name] + args.backgrounds:
        m = (data["Sample"] == n) & pmask
        keep = (data.loc[m, "Score"] >= threshold).to_numpy()
        p, nn = (float(np.mean(keep)), int(m.sum())) if m.sum() else (np.nan, 0)
        disc_rows.append({**common, "Population": population, "Label": label(n),
                          "Discriminant": data.loc[m, "Score"].reset_index(drop=True), "Efficiency": 100 * p, "Threshold": threshold})
        sel_rows.append({**common, "Population": population, "Label": label(n), "Percentage": 100 * p,
                         "PercentageError": 100 * np.sqrt(p * (1 - p) / nn) if nn else np.nan, "Clusters": nn, "Threshold": threshold})
        idx = np.digitize(data.loc[m, "Energy"].to_numpy(), energy_edges) - 1
        frac = [100 * float(np.mean(keep[idx == i])) if np.sum(idx == i) >= 20 else np.nan for i in range(len(energy_edges) - 1)]
        eff_rows.append({**common, "Population": population, "Label": label(n), "Variable": "Energy",
                         "Values": 0.5 * (energy_edges[1:] + energy_edges[:-1]), "Efficiency": frac})
save_df(pd.DataFrame(sel_rows), data_path, config, args.name, filename="Signal_Background_Selection", rm=args.rewrite, debug=args.debug)
save_df(pd.DataFrame(disc_rows), data_path, config, args.name, filename="Signal_Background_Discriminant", rm=args.rewrite, debug=args.debug)
save_df(pd.DataFrame(eff_rows), data_path, config, args.name, filename="Signal_Background_Efficiency", rm=args.rewrite, debug=args.debug)
importance = np.mean([m.feature_importances_ for m in models.values()], axis=0)
save_df(pd.DataFrame([{**common, "Feature": features, "Importance": list(importance)}]), data_path, config, args.name,
        filename="Signal_Background_Importance", rm=args.rewrite, debug=args.debug)

model_dir = f"{_models_base}/{config}/{args.name}/models"
os.makedirs(model_dir, exist_ok=True)
for fold, model in models.items():
    path = f"{model_dir}/{config}_{args.name}_signal_background_fold{fold}.pkl"
    if os.path.isfile(path):  # pnfs cannot overwrite in place
        os.remove(path)
    with open(path, "wb") as f:
        pickle.dump(model, f)
calib_dir = f"{root}/config/{config}/{args.name}/{config}_calib"
os.makedirs(calib_dir, exist_ok=True)
with open(f"{calib_dir}/{config}_signal_background.json", "w") as f:
    json.dump(summary, f, indent=2)
rprint(f"Saved signal/background classifier ({len(features)} features, AUC {auc:.3f}) to {model_dir} and {calib_dir}")
