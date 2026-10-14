import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *

"""
03d_signal_background_roc.py: ROC curves and threshold scans of the signal/background classifier.

Reads the held-out classifier scores written by 03b_signal_background.py (Signal_Background_Discriminant: one
score array per sample and population, every cluster scored by the fold model that did not see it) and the
base threshold of the calibration summary, and exports, for each background sample against the CC neutrino
sample and each population (All, Matched):
  Signal_Background_ROC              CC efficiency vs background signal-like fraction over the score range, the
                                     AUC (Mann-Whitney) with its Hanley-McNeil error, and the working point
  Signal_Background_Threshold_Scan   signal-like fraction of every sample vs the score threshold (0-1, step 0.005)
  Signal_Background_WorkingPoint.json signal-like fractions at the base threshold (population All)
Nothing is retrained; the exports follow the classifier of the current calibration.
"""

parser = argparse.ArgumentParser(description="ROC curves and threshold scans of the signal/background classifier")
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--name", type=str, default="marley_final")
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()

config = args.config
data_path = f"{root}/output/data/workflow/discrimination"
disc = pd.read_pickle(f"{data_path}/{config}/{args.name}/{config}_{args.name}_Signal_Background_Discriminant.pkl")
summary = json.load(open(f"{root}/config/{config}/{args.name}/{config}_calib/{config}_signal_background.json"))
threshold = float(summary["SB_THRESHOLD"])
info = json.load(open(f"{root}/config/{config}/{config}_config.json"))
common = {"Geometry": info["GEOMETRY"], "Config": config, "Name": args.name}
signal_label = "CC Neutrino"


def auc_and_error(sig: np.ndarray, bkg: np.ndarray):
    """Mann-Whitney AUC (signal scored above background) and the Hanley-McNeil standard error."""
    scores = np.concatenate([sig, bkg])
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    n1, n0 = len(sig), len(bkg)
    auc = (ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    q1, q2 = auc / (2 - auc), 2 * auc**2 / (1 + auc)
    err = np.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc**2) + (n0 - 1) * (q2 - auc**2)) / (n1 * n0))
    return float(auc), float(err)


roc_grid = np.unique(np.concatenate([np.linspace(0, 1, 1001), np.linspace(0.999, 1, 1001)]))
scan_grid = np.round(np.arange(0, 1.0001, 0.005), 6)
roc_rows, scan_rows, working_point = [], [], {}
for population, df_pop in disc.groupby("Population", sort=False):
    scores = {row.Label: np.asarray(row.Discriminant, dtype=float) for row in df_pop.itertuples()}
    if signal_label not in scores:
        continue
    sig = scores[signal_label]
    sig_wp = 100 * float(np.mean(sig >= threshold))
    for label, values in scores.items():
        scan_rows.append({**common, "Population": population, "Label": label, "Variable": "Threshold",
                          "Values": scan_grid, "Efficiency": [100 * float(np.mean(values >= t)) for t in scan_grid],
                          "Threshold": threshold})
        if population == "All":
            working_point[label] = 100 * float(np.mean(values >= threshold))
        if label == signal_label:
            continue
        auc, err = auc_and_error(sig, values)
        roc_rows.append({**common, "Population": population, "Label": label,
                         "SignalEfficiency": [100 * float(np.mean(sig >= t)) for t in roc_grid],
                         "BackgroundEfficiency": [100 * float(np.mean(values >= t)) for t in roc_grid],
                         "AUC": auc, "AUCError": err, "Clusters": int(len(values)), "Threshold": threshold,
                         "WPSignalEfficiency": sig_wp, "WPBackgroundEfficiency": 100 * float(np.mean(values >= threshold))})
        rprint(f"{config} {population:<8} {label:<12} AUC {auc:.4f} +- {err:.4f}  signal-like at {threshold:.3f}: "
               f"{roc_rows[-1]['WPBackgroundEfficiency']:.2f} % (CC {sig_wp:.2f} %)")
working_point["Threshold"] = threshold

save_df(pd.DataFrame(roc_rows), data_path, config, args.name, filename="Signal_Background_ROC", rm=args.rewrite, debug=args.debug)
save_df(pd.DataFrame(scan_rows), data_path, config, args.name, filename="Signal_Background_Threshold_Scan", rm=args.rewrite, debug=args.debug)
with open(f"{data_path}/{config}/{args.name}/{config}_{args.name}_Signal_Background_WorkingPoint.json", "w") as f:
    json.dump(working_point, f, indent=1)
rprint(f"[green]Saved ROC, threshold scan and working point for {config} {args.name}[/green]")
