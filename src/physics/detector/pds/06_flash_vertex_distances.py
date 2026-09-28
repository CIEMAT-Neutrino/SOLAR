"""
06_flash_vertex_distances.py — flash distances to the signal cluster vertex
===========================================================================
Distributions behind the adjacent-flash (matching candidate) cuts, from the 2026-09
"flash" productions, which store every reconstructed OpFlash of the event in
MCTruthTree. For every primary signal cluster (Generator == 1) all flashes of its event
that are time-compatible with it (the flash precedes the cluster by at most the
maximum drift time) are histogrammed in |dY|, |dZ| and the radial distance
R = sqrt(dY^2 + dZ^2) between the cluster's reconstructed vertex and the flash centre,
split into signal flashes (Pur > 0) and background flashes (Pur == 0). The MainOpFlash*
variables repeat this for the largest (PE) time-compatible flash of each cluster only.

Output: output/data/PDS/flash/<config>/<name>/<config>_<name>_Flash_Vertex_Distances.pkl
  columns Config, Name, Variable ([Main]OpFlashErrorY/ErrorZ/R), Signal (bool),
  X (bin centres, cm), Y (counts), NClusters.

Run
---
  python3 src/physics/detector/pds/06_flash_vertex_distances.py \\
      --config hd_1x2x6_centralAPA --name marley_flash
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *
import awkward as ak
import uproot

data_path = f"{root}/output/data/PDS/flash"

parser = argparse.ArgumentParser(description="Flash distances to the signal cluster vertex")
parser.add_argument("--config", type=str, default="hd_1x2x6_centralAPA")
parser.add_argument("--name", type=str, default="marley_flash")
parser.add_argument("--max_events", type=int, default=None, help="Stop after this many truth events (testing)")
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
args = parser.parse_args()
config, name = args.config, args.name

info, params, output = get_param_dict(f"{root}/config/{config}/{config}", {}, "", debug=args.debug)
filename = f'{info["PATH"]}/data/{info["GEOMETRY"]}/{info["VERSION"]}/{info["NAME"]}{name}.root'
if not os.path.isfile(filename):
    sys.exit(f"ERROR: {filename} not found")

f = uproot.open(filename)
folder = [k.split(";")[0] for k, c in f.classnames().items() if c == "TDirectory"][0]
truth_tree = f[f"{folder}/MCTruthTree"]
reco_tree = f[f"{folder}/SolarNuAnaTree"]

# Time frames: cluster Time is in ticks from the readout start, OpFlashTime in us
# relative to the trigger, shifted by the configured flash-time offset.
tick_us = info["TIMEWINDOW"] * 1e6 / info["EVENT_TICKS"]
config_dir = f"{info['PATH']}/data/{info['GEOMETRY']}/{info['VERSION']}/{info['NAME']}{name}/Config"
flash_offset = float(np.unique(np.load(f"{config_dir}/OpFlashTimeOffset.npy", allow_pickle=True))[0])
max_drift_us = info["TIMEWINDOW"] * 1e6

reco = reco_tree.arrays(["Event", "Flag", "Generator", "Time", "RecoY", "RecoZ"], library="np")
signal = reco["Generator"] == 1
clusters = {}
for ev, flag, t, y, z in zip(reco["Event"][signal], reco["Flag"][signal], reco["Time"][signal], reco["RecoY"][signal], reco["RecoZ"][signal]):
    clusters.setdefault((int(ev), int(flag)), []).append((t * tick_us, y, z))
rprint(f"[cyan]{len(clusters)} events with {int(signal.sum())} signal clusters; flash offset {flash_offset} us, max drift {max_drift_us:.0f} us[/cyan]")

edges = np.arange(0, 302, 2.0)
hist = {
    (p + v, s): np.zeros(len(edges) - 1)
    for p in ("", "Main")
    for v in ("OpFlashErrorY", "OpFlashErrorZ", "OpFlashR")
    for s in (True, False)
}
n_clusters, n_events = 0, 0
for chunk in truth_tree.iterate(["Event", "Flag", "OpFlashY", "OpFlashZ", "OpFlashTime", "OpFlashPur", "OpFlashPE"], step_size=500, library="ak"):
    events, flags = ak.to_numpy(chunk["Event"]), ak.to_numpy(chunk["Flag"])
    for i in range(len(chunk)):
        this = clusters.get((int(events[i]), int(flags[i])))
        if this is None:
            continue
        fy = ak.to_numpy(chunk["OpFlashY"][i])
        fz = ak.to_numpy(chunk["OpFlashZ"][i])
        ft = ak.to_numpy(chunk["OpFlashTime"][i]) + flash_offset
        fs = ak.to_numpy(chunk["OpFlashPur"][i]) > 0
        fpe = ak.to_numpy(chunk["OpFlashPE"][i])
        for t, y, z in this:
            drift = t - ft
            ok = (drift >= 0) & (drift <= max_drift_us)
            dy, dz = np.abs(fy[ok] - y), np.abs(fz[ok] - z)
            r = np.sqrt(dy**2 + dz**2)
            for s in (True, False):
                sel = fs[ok] == s
                hist[("OpFlashErrorY", s)] += np.histogram(dy[sel], bins=edges)[0]
                hist[("OpFlashErrorZ", s)] += np.histogram(dz[sel], bins=edges)[0]
                hist[("OpFlashR", s)] += np.histogram(r[sel], bins=edges)[0]
            if np.any(ok):
                k = np.argmax(fpe[ok])
                s = bool(fs[ok][k])
                hist[("MainOpFlashErrorY", s)] += np.histogram(dy[k : k + 1], bins=edges)[0]
                hist[("MainOpFlashErrorZ", s)] += np.histogram(dz[k : k + 1], bins=edges)[0]
                hist[("MainOpFlashR", s)] += np.histogram(r[k : k + 1], bins=edges)[0]
            n_clusters += 1
        n_events += 1
    rprint(f"  events {n_events}, clusters {n_clusters}")
    if args.max_events is not None and n_events >= args.max_events:
        break

centres = 0.5 * (edges[1:] + edges[:-1])
df = pd.DataFrame(
    [
        {"Config": config, "Name": name, "Variable": v, "Signal": s, "X": centres, "Y": h, "NClusters": n_clusters}
        for (v, s), h in hist.items()
    ]
)
save_df(df, data_path, config, name, filename="Flash_Vertex_Distances", rm=args.rewrite, debug=args.debug)
