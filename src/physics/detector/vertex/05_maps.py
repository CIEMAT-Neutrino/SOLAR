"""
05_maps.py — Vertex Reconstruction Efficiency Maps
==================================================
Spatial projections of the 3D vertex reconstruction efficiency of 04_reconstruction.py: an event is
reconstructed if its 3D vertex error is below r_n = n * quantile(r_hp, erf(1/sqrt 2)), the 68.27 %
radius of the high-purity flash-matched sample at the event's true energy (n = --tolerance, paper: 3).

For each projection (z-y, z-x, y-x; x is the drift coordinate, |x| for the central-APA HD workspace)
and for the full sample and the 10 MeV slice, the export holds, on a common grid:
  Events      true vertices of all primaries
  Efficiency  fraction of them reconstructed within r_n (true-vertex binning)
  Lost        reconstructed vertices of the primaries outside r_n (reco-vertex binning), i.e. where
              the reconstruction puts the events it loses

Output: output/data/vertex/maps/{config}/{name}/{config}_{name}_Vertex_Efficiency_Maps.pkl
(columns Projection, Map, Energy, XEdges, YEdges, Values[ny, nx]).
"""

import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *
from scipy.special import erf

data_path = f"{root}/output/data/vertex/maps"
os.makedirs(data_path, exist_ok=True)

parser = argparse.ArgumentParser(description="Vertex reconstruction efficiency maps")
parser.add_argument("--config", type=str, help="The configuration to load", default="hd_1x2x6_centralAPA")
parser.add_argument("--name", type=str, help="The name of the configuration", default="marley_final")
parser.add_argument("--tolerance", type=int, default=3, help="Tolerance in units of the 68.27 %% high-purity radius")
parser.add_argument("--purity_threshold", type=float, default=0.5, help="MatchedOpFlashPur of the high-purity sample")
parser.add_argument("--bin_size", type=float, default=40, help="Map bin size (cm)")
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
args = parser.parse_args()
config, name = args.config, args.name
configs = {config: [name]}
workflow = "VERTEXING"

run, output = load_multi(configs, preset=workflow, debug=args.debug)
run = compute_reco_workflow(run, configs, workflow=workflow, debug=args.debug)
run, mask, output = compute_filtered_run(
    run, configs, params={("Reco", "TrueMain"): ("equal", True)},
    presets=["ANALYSIS"], signal="marley" in name, debug=args.debug,
)
rprint(output)

info = json.load(open(f"{root}/config/{config}/{config}_config.json", "r"))
reco = run["Reco"]
energy = np.asarray(reco["SignalParticleK"], dtype=float)
error_3d = np.sqrt(np.asarray(reco["ErrorX"]) ** 2 + np.asarray(reco["ErrorY"]) ** 2 + np.asarray(reco["ErrorZ"]) ** 2)
high_purity = np.asarray(reco["MatchedOpFlashPur"]) > args.purity_threshold

# Energy-dependent tolerance, as in 04_reconstruction.py (2 MeV slices from 6 to 30 MeV).
slices = np.arange(6, 31, 2)
radius = np.full(len(slices), np.nan)
for i, centre in enumerate(slices):
    in_slice = high_purity & (energy >= centre - 1) & (energy < centre + 1)
    if np.sum(in_slice) >= 10:
        radius[i] = args.tolerance * np.quantile(error_3d[in_slice], erf(1 / np.sqrt(2)))
valid = np.isfinite(radius)
event_radius = np.interp(energy, slices[valid], radius[valid])  # clamped outside 6-30 MeV
reconstructed = error_3d < event_radius

fold = info["GEOMETRY"] == "hd" and info["VERSION"] in ["hd_1x2x6_centralAPA", "hd_1x2x6"]
true_pos = {
    "X": np.abs(np.asarray(reco["SignalParticleX"], dtype=float)) if fold else np.asarray(reco["SignalParticleX"], dtype=float),
    "Y": np.asarray(reco["SignalParticleY"], dtype=float),
    "Z": np.asarray(reco["SignalParticleZ"], dtype=float),
}
reco_pos = {
    "X": np.abs(np.asarray(reco["RecoX"], dtype=float)) if fold else np.asarray(reco["RecoX"], dtype=float),
    "Y": np.asarray(reco["RecoY"], dtype=float),
    "Z": np.asarray(reco["RecoZ"], dtype=float),
}
limits = {
    "X": (0 if fold else info["DETECTOR_MIN_X"], info["DETECTOR_MAX_X"]),
    "Y": (info["DETECTOR_MIN_Y"], info["DETECTOR_MAX_Y"]),
    "Z": (info["DETECTOR_MIN_Z"], info["DETECTOR_MAX_Z"]),
}
# Bins of about --bin_size that tile the active volume exactly.
edges = {axis: np.linspace(lo, hi, max(int(round((hi - lo) / args.bin_size)), 1) + 1) for axis, (lo, hi) in limits.items()}

rows = []
for (horizontal, vertical), energy_slice in product([("Z", "Y"), ("Z", "X"), ("Y", "X")], [None, 10]):
    sel = np.ones(len(energy), dtype=bool) if energy_slice is None else (energy >= energy_slice - 1) & (energy < energy_slice + 1)
    bins = [edges[vertical], edges[horizontal]]
    events, _, _ = np.histogram2d(true_pos[vertical][sel], true_pos[horizontal][sel], bins=bins)
    good, _, _ = np.histogram2d(true_pos[vertical][sel & reconstructed], true_pos[horizontal][sel & reconstructed], bins=bins)
    lost, _, _ = np.histogram2d(reco_pos[vertical][sel & ~reconstructed], reco_pos[horizontal][sel & ~reconstructed], bins=bins)
    maps = {
        "Events": np.where(events > 0, events, np.nan),
        "Efficiency": np.where(events >= 5, 100 * good / np.maximum(events, 1), np.nan),
        "Lost": np.where(lost > 0, lost, np.nan),
    }
    for label, values in maps.items():
        rows.append({
            "Geometry": info["GEOMETRY"], "Config": config, "Name": name, "Tolerance": args.tolerance,
            "Projection": f"{horizontal}{vertical}", "Map": label, "Energy": energy_slice,
            "Efficiency": float(np.mean(reconstructed[sel])) if np.any(sel) else np.nan,
            "XEdges": edges[horizontal], "YEdges": edges[vertical], "Values": values,
        })
save_df(pd.DataFrame(rows), data_path, config, name, filename="Vertex_Efficiency_Maps", rm=args.rewrite, debug=args.debug)
rprint(f"[green]Vertex efficiency maps saved for {config} {name}: overall efficiency {np.mean(reconstructed):.3f}[/green]")
