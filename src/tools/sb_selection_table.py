"""sb_selection_table.py: signal-selected percentage per sample for the signal/background classifier.

Reads the held-out scores exported by src/physics/calibration/03b_signal_background.py
(Signal_Background_Discriminant) and writes Signal_Background_Selection: one row per sample with the
percentage of clusters (reconstructed Energy above the training MIN_ENERGY) classified signal-like
and its binomial uncertainty, for script_aggregate_table.py in the plotting repository.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from lib import *

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--configs", nargs="+", required=True)
parser.add_argument("--name", default="marley_final")
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()

data_path = f"{root}/output/data/workflow/discrimination"
for config in args.configs:
    path = f"{data_path}/{config}/{args.name}/{config}_{args.name}_Signal_Background_Discriminant.pkl"
    disc = pd.read_pickle(path)
    rows = []
    for _, r in disc.iterrows():
        scores = np.asarray(r["Discriminant"], dtype=float)
        n = len(scores)
        p = float(np.mean(scores >= r["Threshold"])) if n else np.nan
        rows.append({"Geometry": r["Geometry"], "Config": config, "Name": args.name, "Label": r["Label"],
                     "Percentage": 100 * p, "PercentageError": 100 * np.sqrt(p * (1 - p) / n) if n else np.nan,
                     "Clusters": n, "Threshold": float(r["Threshold"])})
    save_df(pd.DataFrame(rows), data_path, config, args.name, filename="Signal_Background_Selection", rm=True, debug=args.debug)
    rprint(pd.DataFrame(rows)[["Label", "Percentage", "PercentageError", "Clusters"]].to_string(index=False))
