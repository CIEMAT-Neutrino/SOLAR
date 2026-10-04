"""
05_flash_distributions.py — OpFlash distribution pkls
=====================================================
Flash-level counterpart of src/physics/detector/tpc/02_adj_clusters.py for the
2026-09 "flash" productions (SolarNuAna_<name>_flash.root), which store every
reconstructed OpFlash of the event in MCTruthTree (OpFlashPE, OpFlashPur, ...).

The ROOT file is read directly and in chunks: a VD event carries ~18k flashes, so the
per-event vectors are far too large for the zero-padded npy layout.

Outputs (output/data/PDS/flash/<config>/<name>/<config>_<name>_<filename>.pkl):

  OpFlash_Counts[_Slim]        per-event number of flashes, by Signal/Background/Total
                               and plane (raw values, like Adjacent_Cluster_Counts)
  OpFlash_Distributions        per-flash histograms of PE, MaxPE, NHits, Time, STD, Pur,
                               X/Y/Z and Plane, by Signal/Background and plane
                               (like Adjacent_Cluster_Distributions; Counts are per event)
  AdjOpFlash_Counts[_Slim]     per-cluster number of adjacent flashes within each
                               OPFLASH_RADIUS, by Signal/Background and cluster type
  AdjOpFlash_Distributions     per-adjacent-flash histograms (Counts are per cluster)
  MatchedOpFlash_Distributions per-cluster histograms of the matched flash variables
  MatchedOpFlash_Summary       matched fraction / signal-matched / correctly-matched
                               per cluster type

"Signal" flashes have Pur > 0 (some PE from the signal generator), "Background" have
Pur == 0. Cluster type is "Marley" for Generator == 1 and "Background" otherwise.
Histogram rows carry Underflow/Overflow counts (values outside the bins are not
folded in) plus NEvents/NClusters/NInputs so rates can be normalised downstream
(NInputs = ConfigTree entries = number of merged grid jobs).

Run
---
  python3 src/physics/detector/pds/05_flash_distributions.py \
      --config vd_1x8x14_3view_30deg_shielded --name marley_flash
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *
import awkward as ak
import uproot

save_path = f"{root}/output/images/PDS/flash"
data_path = f"{root}/output/data/PDS/flash"

for path in [save_path, data_path]:
    if not os.path.exists(path):
        os.makedirs(path)

parser = argparse.ArgumentParser(description="OpFlash distributions of the flash productions")
parser.add_argument("--config", type=str, help="The configuration to load", default="vd_1x8x14_3view_30deg_shielded")
parser.add_argument("--name", type=str, help="The name of the sample", default="marley_flash")
parser.add_argument("--step", type=int, help="Truth events per chunk", default=500)
parser.add_argument("--max_events", type=int, help="Stop after this many truth events (testing)", default=None)
parser.add_argument("--max_raw", type=int, help="Max raw entries kept in the *_Counts pkls", default=2_000_000)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
args = parser.parse_args()

config = args.config
name = args.name
user_input = {"rewrite": args.rewrite, "debug": args.debug}

info, params, output = get_param_dict(f"{root}/config/{config}/{config}", {}, "", debug=args.debug)
filename = f'{info["PATH"]}/data/{info["GEOMETRY"]}/{info["VERSION"]}/{info["NAME"]}{name}.root'
if not os.path.isfile(filename):
    sys.exit(f"ERROR: {filename} not found")

planes = {int(k): v for k, v in info["OPFLASH_PLANES"].items()}
radii = info["OPFLASH_RADIUS"]
window = info["TIMEWINDOW"] * 1e6  # us
rng = np.random.default_rng(42)

f = uproot.open(filename)
folder = [k.split(";")[0] for k, c in f.classnames().items() if c == "TDirectory"][0]
truth_tree = f[f"{folder}/MCTruthTree"]
reco_tree = f[f"{folder}/SolarNuAnaTree"]
n_inputs = f[f"{folder}/ConfigTree"].num_entries
base_row = {"Geometry": info["GEOMETRY"], "Config": config, "Name": name, "NInputs": n_inputs}


def position_bins(coord):
    return np.linspace(info[f"DETECTOR_MIN_{coord}"] - 150, info[f"DETECTOR_MAX_{coord}"] + 150, 101)


BINS = {
    "PE": np.logspace(0, 5, 101),
    "MaxPE": np.logspace(0, 5, 101),
    "NHits": np.arange(0.5, 101.5, 1),
    "Time": np.linspace(-window, window, 171),
    "STD": np.linspace(0, 1e4, 101),
    "Fast": np.linspace(0, 1.5, 76),
    "Pur": np.linspace(0, 1, 101),
    "R": np.linspace(0, 150, 76),
    "Plane": np.arange(-1.5, max(planes) + 1.5, 1),
    "X": position_bins("X"),
    "Y": position_bins("Y"),
    "Z": position_bins("Z"),
    "RecoX": position_bins("X"),
    "RecoY": position_bins("Y"),
    "RecoZ": position_bins("Z"),
}


class Hist:
    """Histogram accumulated over chunks, keyed by (variable, *labels)."""

    def __init__(self):
        self.h = {}

    def fill(self, key, variable, values):
        values = np.asarray(values, dtype=float)
        edges = BINS[variable]
        hist, under, over = self.h.get(key, (np.zeros(len(edges) - 1), 0, 0))
        hist = hist + np.histogram(values, bins=edges)[0]
        self.h[key] = (hist, under + int(np.sum(values < edges[0])), over + int(np.sum(values > edges[-1])))

    def rows(self, labels, norm, norm_label, select=lambda key: True):
        rows = []
        for key, (hist, under, over) in self.h.items():
            if not select(key):
                continue
            variable = key[0]
            edges = BINS[variable]
            widths = np.diff(edges)
            entries = np.sum(hist)
            rows.append(
                {
                    **base_row,
                    **dict(zip(labels, key)),
                    "Values": (edges[1:] + edges[:-1]) / 2,
                    "Edges": edges,
                    "Counts": hist / norm,
                    "CountsError": np.sqrt(hist) / norm,
                    "Density": hist / (entries * widths) if entries > 0 else np.zeros_like(hist),
                    "DensityError": np.sqrt(hist) / (entries * widths) if entries > 0 else np.zeros_like(hist),
                    "Entries": int(entries),
                    "Underflow": under,
                    "Overflow": over,
                    norm_label: norm,
                }
            )
        return rows


class RawSample:
    """Keeps a fixed-probability random subset of per-entry values, keyed by labels."""

    def __init__(self, n_total, max_raw):
        self.p = min(1.0, max_raw / max(n_total, 1))
        self.values = {}

    def keep_mask(self, n):
        return np.ones(n, dtype=bool) if self.p >= 1 else rng.random(n) < self.p

    def add(self, key, values):
        self.values.setdefault(key, []).append(np.asarray(values))

    def rows(self, labels, column):
        rows, slim = [], []
        for key, chunks in self.values.items():
            values = pd.Series(np.concatenate(chunks)).reset_index(drop=True)
            row = {**base_row, **dict(zip(labels, key)), "Sampled": self.p < 1, "SampleFraction": self.p}
            rows.append({**row, column: values})
            if len(values) > 10_000:
                values = values.iloc[np.sort(rng.choice(len(values), 10_000, replace=False))].reset_index(drop=True)
            slim.append({**row, column: values})
        return pd.DataFrame(rows), pd.DataFrame(slim)


def split_masks(pur, valid):
    return {"Signal": valid & (pur > 0), "Background": valid & (pur == 0), "Total": valid}


# ── Truth: every OpFlash of the event ────────────────────────────────────────
# Background productions run without the truth OpFlash branches (e.g. the FNAL VD
# nominal radiological sample); they only get the Reco-level pkls.
truth_vars = ["PE", "MaxPE", "NHits", "Time", "STD", "Pur", "X", "Y", "Z", "Plane"]
has_truth_flashes = all(f"OpFlash{v}" in truth_tree.keys() for v in truth_vars)
if not has_truth_flashes:
    rprint(f"[yellow]{config} {name}: no truth OpFlash vectors, skipping the OpFlash_* pkls[/yellow]")
else:
    n_truth = truth_tree.num_entries if args.max_events is None else min(args.max_events, truth_tree.num_entries)
    truth_hist = Hist()
    truth_raw = RawSample(n_truth, args.max_raw)
    n_events = 0
    rprint(f"[cyan]{config} {name}: {n_truth} truth events, {reco_tree.num_entries} clusters, {n_inputs} inputs[/cyan]")
    for chunk in truth_tree.iterate(
        [f"OpFlash{v}" for v in truth_vars], step_size=args.step, entry_stop=n_truth, library="ak"
    ):
        n_events += len(chunk)
        pur = chunk["OpFlashPur"]
        plane = chunk["OpFlashPlane"]
        valid = chunk["OpFlashPE"] > 0
        keep = truth_raw.keep_mask(len(chunk))
        for label, mask in split_masks(pur, valid).items():
            for plane_id, plane_name in [(None, "Total")] + list(planes.items()):
                this_mask = mask if plane_id is None else mask & (plane == plane_id)
                truth_raw.add((label, plane_name), ak.to_numpy(ak.sum(this_mask, axis=1))[keep].astype(np.int32))
                if label == "Total":
                    continue
                for variable in truth_vars:
                    if plane_id is not None and variable == "Plane":
                        continue
                    truth_hist.fill(
                        (variable, label, plane_name), variable, ak.to_numpy(ak.flatten(chunk[f"OpFlash{variable}"][this_mask]))
                    )
        rprint(f"  truth events {n_events}/{n_truth}")

    truth_dist = pd.DataFrame(truth_hist.rows(["Variable", "Signal", "Plane"], n_events, "NEvents"))
    truth_dist["Variable"] = "OpFlash" + truth_dist["Variable"]
    truth_counts, truth_counts_slim = truth_raw.rows(["Signal", "Plane"], "OpFlashNum")
    for df_, filename_ in [
        (truth_dist, "OpFlash_Distributions"),
        (truth_counts, "OpFlash_Counts"),
        (truth_counts_slim, "OpFlash_Counts_Slim"),
    ]:
        save_df(df_, data_path, config, name, filename=filename_, rm=user_input["rewrite"], debug=user_input["debug"])

# ── Reco: adjacent and matched flashes of each cluster ───────────────────────
adj_vars = ["PE", "MaxPE", "NHits", "R", "Time", "STD", "Fast", "Pur", "Plane", "RecoX", "RecoY", "RecoZ"]
matched_vars = ["PE", "MaxPE", "NHits", "R", "Time", "STD", "Fast", "Pur", "Plane", "RecoX", "RecoY", "RecoZ"]
n_reco = reco_tree.num_entries
adj_hist = Hist()
matched_hist = Hist()
adj_raw = RawSample(n_reco, args.max_raw)
n_clusters = {}
summary = {}
reco_branches = ["Generator"] + [f"AdjOpFlash{v}" for v in adj_vars] + [f"MatchedOpFlash{v}" for v in matched_vars]
has_correctly = "MatchedOpFlashCorrectly" in reco_tree.keys()  # missing in pre-2026 productions
for chunk in reco_tree.iterate(
    reco_branches + (["MatchedOpFlashCorrectly"] if has_correctly else []),
    step_size="500 MB",
    library="ak",
):
    generator = ak.to_numpy(chunk["Generator"])
    keep = adj_raw.keep_mask(len(chunk))
    for cluster, cmask in [("Marley", generator == 1), ("Background", generator != 1)]:
        if not np.any(cmask):
            continue
        sub = chunk[cmask]
        n_clusters[cluster] = n_clusters.get(cluster, 0) + len(sub)
        sub_keep = keep[cmask]

        # Adjacent flashes (vectors per cluster)
        pur = sub["AdjOpFlashPur"]
        valid = sub["AdjOpFlashPE"] > 0
        radius = abs(sub["AdjOpFlashR"])
        for label, mask in split_masks(pur, valid).items():
            for limit in radii:
                adj_raw.add(
                    (cluster, label, f"{limit}"),
                    ak.to_numpy(ak.sum(mask & (radius < limit), axis=1))[sub_keep].astype(np.int16),
                )
            if label == "Total":
                continue
            for variable in adj_vars:
                adj_hist.fill((variable, cluster, label), variable, ak.to_numpy(ak.flatten(sub[f"AdjOpFlash{variable}"][mask])))

        # Matched flash (one per cluster, -1e6 when unmatched)
        mpe = ak.to_numpy(sub["MatchedOpFlashPE"])
        mpur = ak.to_numpy(sub["MatchedOpFlashPur"])
        matched = mpe > 0
        s = summary.setdefault(cluster, {"NClusters": 0, "Matched": 0, "MatchedSignal": 0, "MatchedCorrectly": 0})
        s["NClusters"] += len(sub)
        s["Matched"] += int(np.sum(matched))
        s["MatchedSignal"] += int(np.sum(matched & (mpur > 0)))
        if has_correctly:
            s["MatchedCorrectly"] += int(np.sum(ak.to_numpy(sub["MatchedOpFlashCorrectly"])))
        for label, mask in [("Signal", matched & (mpur > 0)), ("Background", matched & (mpur == 0))]:
            for variable in matched_vars:
                matched_hist.fill((variable, cluster, label), variable, ak.to_numpy(sub[f"MatchedOpFlash{variable}"])[mask])
    rprint(f"  clusters {sum(n_clusters.values())}/{n_reco}")

adj_dist, matched_dist = [
    pd.DataFrame(
        [
            row
            for cluster in n_clusters
            for row in hist.rows(
                ["Variable", "Cluster", "Signal"], n_clusters[cluster], "NClusters",
                select=lambda key, cluster=cluster: key[1] == cluster,
            )
        ]
    )
    for hist in (adj_hist, matched_hist)
]
for df_, prefix in [(adj_dist, "AdjOpFlash"), (matched_dist, "MatchedOpFlash")]:
    if not df_.empty:
        df_["Variable"] = prefix + df_["Variable"]
adj_counts, adj_counts_slim = adj_raw.rows(["Cluster", "Signal", "Radius"], "AdjOpFlashNum")
summary_df = pd.DataFrame(
    [
        {
            **base_row,
            "Cluster": cluster,
            **s,
            "MatchedFraction": s["Matched"] / s["NClusters"],
            "MatchedSignalFraction": s["MatchedSignal"] / s["NClusters"],
            "MatchedCorrectlyFraction": s["MatchedCorrectly"] / s["NClusters"],
        }
        for cluster, s in summary.items()
    ]
)
for df_, filename_ in [
    (adj_dist, "AdjOpFlash_Distributions"),
    (adj_counts, "AdjOpFlash_Counts"),
    (adj_counts_slim, "AdjOpFlash_Counts_Slim"),
    (matched_dist, "MatchedOpFlash_Distributions"),
    (summary_df, "MatchedOpFlash_Summary"),
]:
    save_df(df_, data_path, config, name, filename=filename_, rm=user_input["rewrite"], debug=user_input["debug"])
rprint(summary_df.drop(columns=["Geometry", "Config"]))

# ── Figures: per-flash distributions, Signal vs Background ───────────────────
for variable in truth_vars if has_truth_flashes else []:
    fig = make_subplots(rows=1, cols=1)
    this = truth_dist[(truth_dist["Variable"] == f"OpFlash{variable}") & (truth_dist["Plane"] == "Total")]
    for idx, signal in enumerate(["Signal", "Background"]):
        row = this[this["Signal"] == signal]
        if row.empty:
            continue
        row = row.iloc[0]
        fig.add_trace(
            go.Scatter(
                x=row["Values"], y=row["Counts"], mode="lines", line_shape="hvh",
                name=signal, line=dict(color=default[idx], width=2),
            )
        )
    fig = format_coustom_plotly(
        fig,
        legend_title="OpFlash",
        title=f"OpFlash{variable} - {config} {name}",
        log=(variable in ["PE", "MaxPE"], True),
        tickformat=(None, ".0e"),
    )
    fig.update_xaxes(title_text=f"OpFlash{variable}")
    fig.update_yaxes(title_text="#OpFlashes per Event")
    save_figure(fig, save_path, config, name, filename=f"OpFlash{variable}_Distribution", rm=user_input["rewrite"], debug=user_input["debug"])
