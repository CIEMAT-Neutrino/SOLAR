"""
pds_plane_decomposition.py — which photon-detector plane each sample's clusters are matched to (membrane-veto study)
==================================================================================================================
For every config and sample (marley, gamma, neutron, radiological) and every analysis (DayNight, HEP, Sensitivity),
follows the events down the default working point of that analysis with the membrane veto LIFTED (all planes kept):

  all                    every cached event (surface filter for surface-filtered backgrounds, as the Truncated folder)
  matched                a flash match exists (plane >= 0, MatchedOpFlashPE > 0)
  cut                    + the analysis's default best cut (NHits / AdjCl / OpHits)
  cut+window             + the analysis's energy window (config/analysis/fiducialization.json)
  cut+window+fiducial    + the default (reco) fiducial volume of the analysis

and records, per matched plane, the weighted fraction of the stage (Fraction, FractionError = binomial error from the
effective MC size), the unweighted fraction, the weight relative to plane 0 (what lifting the veto adds), and how well
the drift coordinate is reconstructed (|RecoX - truth X| < 10 cm; truth key per sample as in truth_position_study.py).

Planes (lib.fiducial.accepted_flash_planes): -1 no match; 0 cathode (VD) / APA (HD); 1, 2 membranes; 3, 4 end caps.
HD only reports -1 and 0. The default analysis keeps plane 0 only (QUALITY_CUTS.OPFLASH_PLANE, the membrane veto).

Input: the per-event caches of truth_position_study.py (output/data/solar/truth_position/{config}/). Weights are
SignalParticleWeight (no oscillation weighting for backgrounds; marley combines all fluxes).

Output (synced by the plot repo, label `default`; its script_aggregate_table.py loads {config}_{name}_{datafile}.pkl):
  output/data/analysis/{day-night|hep|sensitivity}/{config}/marley/truncated/default/{config}_{sample}_{Analysis}_PDSPlanes.pkl
  output/data/solar/truth_position/tables/pds_planes.csv   (all rows, for the documentation)

Example table in the plot repo:
  python3 scripts/script_aggregate_table.py --configs <configs> --names marley gamma neutron radiological \
      --datafile HEP_PDSPlanes --y Fraction --variables Variable --row_name Name --select Selection --save_values matched

Usage
-----
  python3 src/physics/signal/pds_plane_decomposition.py
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import numpy as np
import pandas as pd

from lib import root, get_fiducialization_config  # noqa: E402
from lib.fiducial import build_fiducial_spatial_mask, get_best_fiducial  # noqa: E402
from lib.background import is_surface_background  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
parser.add_argument("--config", nargs="+", default=[
    "hd_1x2x6_centralAPA", "hd_1x2x6_lateralAPA", "vd_1x8x14_3view_30deg_nominal", "vd_1x8x14_3view_30deg_shielded"])
parser.add_argument("--samples", nargs="+", default=["marley", "gamma", "neutron", "radiological"])
parser.add_argument("--folder", default="Truncated")
parser.add_argument("--energy", default="SolarEnergy")
parser.add_argument("--dx_cm", type=float, default=10.0, help="|RecoX - truth X| below which the drift coordinate counts as well reconstructed")
parser.add_argument("--sync", action=argparse.BooleanOptionalAction, default=True, help="Write the per-config/sample pickles into the synced analysis tree")
args = parser.parse_args()

ROOT = str(root)
CACHE = Path(ROOT) / "output/data/solar/truth_position"
SYNC_ROOT = Path(ROOT) / "output/data/analysis"
ANALYSES = {"DayNight": ("day-night", "daynight-json", "highest_DayNight"),
            "HEP": ("hep", "hep-json", "highest_HEP"),
            "Sensitivity": ("sensitivity", "sensitivity-json", "highest_Sensitivity")}
PLANES = [-1, 0, 1, 2, 3, 4]
PLANE_LABEL = {-1: "No match", 0: "P0 cathode/APA", 1: "P1 membrane", 2: "P2 membrane", 3: "P3 end cap", 4: "P4 end cap"}
PLANE_ROLE = {-1: "no flash match", 0: "cathode (VD) / APA (HD)", 1: "membrane", 2: "membrane", 3: "end cap", 4: "end cap"}
STAGES = ["all", "matched", "cut", "cut+window", "cut+window+fiducial"]


def neff(w):
    return float(w.sum() ** 2 / (w ** 2).sum()) if w.size and (w ** 2).sum() > 0 else 0.0


def stage_masks(d, config, analysis, info):
    """Boolean mask per stage for one analysis, with the membrane veto lifted."""
    _, json_dir, tag = ANALYSES[analysis]
    cut = json.load(open(f"{ROOT}/config/{config}/{json_dir}/{args.folder.lower()}/{config}_{tag}.json"))[config][args.energy]
    fcfg = get_fiducialization_config(ROOT, analysis.upper())
    lo, hi = fcfg.get("energy_min"), fcfg.get("energy_max")
    fids = json.load(open(f"{ROOT}/config/analysis/fiducial/{args.folder.lower()}/BestFiducials.json"))
    fid = get_best_fiducial(fids, config, args.energy, analysis.upper())
    dx = info["DETECTOR_SIZE_X"] + 2 * info["DETECTOR_GAP_X"]
    dy = info["DETECTOR_SIZE_Y"] + 2 * info["DETECTOR_GAP_Y"]
    matched = (d["plane"] >= 0) & (d["pe"] > 0)
    cutm = matched & (d["NHits"] >= cut["NHits"]) & (d["AdjClNum"] < cut["AdjCl"]) & (d["OpHits"] >= cut["OpHits"])
    win = cutm & (d["energy"] >= (lo if lo is not None else -np.inf)) & (d["energy"] <= (hi if hi is not None else np.inf))
    run = {"Reco": {"RecoX": d["reco_x"], "RecoY": d["reco_y"], "RecoZ": d["reco_z"]}}
    fidm = win & build_fiducial_spatial_mask(run, config, dx, dy, info, args.folder, fid)
    wp = {"NHits": int(cut["NHits"]), "OpHits": int(cut["OpHits"]), "AdjCl": int(cut["AdjCl"]),
          "EnergyMin": lo, "EnergyMax": hi, "FiducialX": fid["FiducialX"], "FiducialY": fid["FiducialY"], "FiducialZ": fid["FiducialZ"]}
    return {"all": np.ones(len(d["w"]), bool), "matched": matched, "cut": cutm, "cut+window": win, "cut+window+fiducial": fidm}, wp


def rows_for(config, sample, analysis, d, info):
    base = np.ones(len(d["w"]), bool)
    if is_surface_background(ROOT, sample):
        base &= (d["surface"] >= 0)
        if args.folder in ("Truncated", "Reduced"):
            base &= d["surface"] < 3
    masks, wp = stage_masks(d, config, analysis, info)
    good_dx = np.abs(d["reco_x"] - d["truth_x"]) < args.dx_cm
    truth_key = str(d["truth_keys"][0])[:-1]
    out = []
    for stage in STAGES:
        m = base & masks[stage]
        w_all = d["w"][m]
        n_eff = neff(w_all)
        p0 = m & (d["plane"] == 0)
        for plane in PLANES:
            mp = m & (d["plane"] == plane)
            w = d["w"][mp]
            f = float(w.sum() / w_all.sum()) if w_all.sum() > 0 else np.nan
            out.append({
                "Geometry": config.split("_")[0], "Config": config, "Name": sample, "Analysis": analysis, "Study": "default",
                "Folder": args.folder, "Selection": stage, "Variable": PLANE_LABEL[plane], "Plane": plane, "PlaneRole": PLANE_ROLE[plane],
                "Fraction": f, "FractionError": float(np.sqrt(f * (1 - f) / n_eff)) if n_eff > 0 and np.isfinite(f) else np.nan,
                "FractionMC": float(mp.sum() / m.sum()) if m.sum() else np.nan,
                "RelativeToPlane0": float(w.sum() / d["w"][p0].sum()) if plane > 0 and d["w"][p0].sum() > 0 else np.nan,
                "FractionDXWithin": float(d["w"][mp & good_dx].sum() / w.sum()) if plane >= 0 and w.sum() > 0 else np.nan,
                "DXWithinCm": args.dx_cm, "TruthKey": truth_key,
                "Weight": float(w.sum()), "WeightStage": float(w_all.sum()), "NMC": int(mp.sum()), "NMCStage": int(m.sum()), "NEffStage": n_eff,
                "WeightLabel": "SignalParticleWeight", **wp,
            })
    return out


all_rows = []
for config in args.config:
    info = json.load(open(f"{ROOT}/config/{config}/{config}_config.json"))
    for sample in args.samples:
        p = CACHE / config / f"{config}_{sample}.npz"
        if not p.exists():
            print(f"[skip] no cache {p}")
            continue
        d = dict(np.load(p, allow_pickle=True))
        for analysis, (andir, _, _) in ANALYSES.items():
            rows = rows_for(config, sample, analysis, d, info)
            all_rows += rows
            if args.sync:
                out = SYNC_ROOT / andir / config / "marley" / args.folder.lower() / "default"
                out.mkdir(parents=True, exist_ok=True)
                f = out / f"{config}_{sample}_{analysis}_PDSPlanes.pkl"
                pd.DataFrame(rows).to_pickle(f, protocol=4)
        print(f"[ok] {config} {sample}")

table = pd.DataFrame(all_rows)
csv = CACHE / "tables" / "pds_planes.csv"
csv.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(csv, index=False)
print(f"[csv] {csv}  ({len(table)} rows)")
