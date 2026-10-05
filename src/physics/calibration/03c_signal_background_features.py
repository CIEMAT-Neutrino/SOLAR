import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *

"""
03c_signal_background_features.py: input distributions of the signal/background classifier.

Loads the same clusters as 03b_signal_background.py (ANALYSIS workflow, no preset filter, reconstructed
Energy >= MIN_ENERGY, |SignalParticleK| <= 30, signal clusters with Generator == 1) and exports, for every
classifier feature and sample, the unit-area distribution (unweighted, as in the training) in two regions:
"All" (Energy >= MIN_ENERGY) and "Window" (6 <= Energy <= 18 MeV, the Day-Night window on the electron
scale). Rows carry the feature importance and its rank from the trained classifier summary
(config/{config}/{name}/{config}_calib/{config}_signal_background.json, Signal_Background_Importance).
Nothing is retrained and no model is written.
"""

parser = argparse.ArgumentParser(description="Export the input distributions of the signal/background classifier")
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--name", type=str, default="marley_final")
parser.add_argument("--max_per_sample", type=int, default=300000)
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()

config = args.config
summary = json.load(open(f"{root}/config/{config}/{args.name}/{config}_calib/{config}_signal_background.json"))
features = summary["FEATURES"]
backgrounds = summary["BACKGROUND_SAMPLES"]
min_energy = summary.get("MIN_ENERGY", 3.0)
data_path = f"{root}/output/data/workflow/discrimination"
imp_file = f"{data_path}/{config}/{args.name}/{config}_{args.name}_Signal_Background_Importance.pkl"
importance = {}
if os.path.exists(imp_file):
    row = pd.read_pickle(imp_file).iloc[0]
    importance = dict(zip(row["Feature"], row["Importance"]))
rank = {f: i + 1 for i, f in enumerate(sorted(importance, key=importance.get, reverse=True))}


def load_sample(name):
    configs = {config: [name]}
    run, _ = load_multi(configs, preset="ANALYSIS", branches={"Config": ["Geometry"]}, debug=args.debug)
    params = {"PARTICLE_TYPE": "signal" if name.startswith("marley") else "background", "SB_CLASSIFIER": False}
    run = compute_reco_workflow(run, configs, params=params, workflow="ANALYSIS", rm_branches=False, debug=args.debug)
    reco = run["Reco"]
    df = pd.DataFrame({f: np.asarray(reco[f], dtype=float) for f in features})
    df["SignalParticleK"] = np.asarray(reco["SignalParticleK"], dtype=float)
    df["Generator"] = np.asarray(reco["Generator"])
    df = df[(df["Energy"] >= min_energy) & (df["SignalParticleK"].abs() <= 30)]
    if name.startswith("marley"):
        df = df[df["Generator"] == 1]
    if len(df) > args.max_per_sample:
        df = df.sample(n=args.max_per_sample, random_state=args.seed)
    df[features] = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


samples = {n: load_sample(n) for n in [args.name] + backgrounds}
label = lambda n: "CC Neutrino" if n == args.name else n.split("_")[0].capitalize()
info = json.load(open(f"{root}/config/{config}/{config}_config.json"))
rows = []
for feature in features:
    pooled = np.concatenate([d[feature].to_numpy() for d in samples.values()])
    # Matched-flash inputs carry a negative default for clusters without a matched flash
    # (-1 optical hits, -1e6 PE): those go to one bin just below the matched range.
    unmatched = feature.startswith("MatchedOpFlash") and np.any(pooled < 0)
    matched = pooled[pooled >= 0] if unmatched else pooled
    integer = np.allclose(matched, np.round(matched))
    lo, hi = np.percentile(matched, [0.5, 99.5])
    if integer:
        edges = np.arange(np.floor(lo) - 0.5, np.ceil(hi) + 1.5, 1.0)
    else:
        edges = np.linspace(lo, hi if hi > lo else lo + 1, 51)
    if unmatched:
        edges = np.concatenate([[edges[0] - (edges[1] - edges[0])], edges])
    for region, sel in [("All", lambda d: np.ones(len(d), bool)), ("Window", lambda d: ((d["Energy"] >= 6) & (d["Energy"] <= 18)).to_numpy())]:
        for n, d in samples.items():
            v = d[feature].to_numpy()[sel(d)]
            if unmatched:
                v = np.where(v < 0, 0.5 * (edges[0] + edges[1]), v)
            h, _ = np.histogram(np.clip(v, edges[0], edges[-1] - 1e-9), bins=edges)
            rows.append({
                "Geometry": info["GEOMETRY"], "Config": config, "Name": args.name, "Label": label(n), "Region": region,
                "Feature": feature, "Importance": float(importance.get(feature, np.nan)), "Rank": int(rank.get(feature, 99)),
                "Bins": 0.5 * (edges[1:] + edges[:-1]), "Unmatched": bool(unmatched), "Density": h / max(h.sum(), 1) / np.diff(edges), "Clusters": int(len(v)),
            })
save_df(pd.DataFrame(rows), data_path, config, args.name, filename="Signal_Background_Features", rm=args.rewrite, debug=args.debug)
top = sorted(rank, key=rank.get)[:4]
rprint(f"{config}: exported {len(features)} feature distributions; top-4 by importance: {top}")
