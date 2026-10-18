"""
export_fiducial_repo.py — bundle the fiducial-stage data for the plot/thesis repos
==================================================================================
Collects the fiducial-stage outputs of src/physics/signal/01_fiducialize.py and 02_best_fiducial.py into one
export folder that scripts/sync_solar_data.sh --fiducial (plotting repository) mirrors. Nothing is recomputed
unless --produce is given.

  output/data/solar/fiducial/export/
    fiducial_profiles{suffix}/<config>/marley_final/<config>_marley_final_Fiducial_Profiles.pkl
        margin profiles of the fiducial figure of merit (02_best_fiducial.py --export_profiles), one folder per
        profile set (PROFILE_SETS); the layout is the one the plotting commands read (--path solar/fiducial_profiles...)
    fiducial_volumes.pkl     the BestFiducials records of the default and of the labelled studies (RECORD_LABELS):
                             volume, significance, MC marks, ungated optimum, per-band volumes
    fiducial_mc_counts.pkl   simulated events per background component in the analysis windows (and at all energies)
                             at the chosen volumes and at the named volumes (MC_VOLUMES), for each scan (MC_SCANS)
    README.md, meta.json

--produce first runs 02_best_fiducial.py --export_profiles --no-write_records for every profile set (the scans of
01_fiducialize.py must exist), with the knobs of the set; the records are left untouched.

Usage:  python3 src/tools/export_fiducial_repo.py [--produce] [--configs ...]
"""

import argparse
import glob
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lib.defaults import load_analysis_info  # noqa: E402

INFO = load_analysis_info(str(ROOT))
FID = Path(INFO["PATH"]) / "FIDUCIAL" / "truncated"
RECORDS = ROOT / "config" / "analysis" / "fiducial" / "truncated"
DST = ROOT / "output" / "data" / "solar" / "fiducial" / "export"
CONFIGS = ["hd_1x2x6_centralAPA", "hd_1x2x6_lateralAPA", "vd_1x8x14_3view_30deg_nominal", "vd_1x8x14_3view_30deg_shielded"]
NAME = "marley_final"
WINDOWS = {"DAYNIGHT": (6, 18), "HEP": (14, 30), "SENSITIVITY": (10, 30)}

# Profile sets: suffix of the exported folder -> (02_best_fiducial.py environment, analyses, description).
# "" is the default rule as it stands in the repository.
PROFILE_SETS = {
    "": ({}, ["DayNight", "HEP", "Sensitivity"], "default fiducial rule"),
    "_dnone": ({"SOLAR_FIDUCIAL_SINGLE_BAND": "1", "SOLAR_FIDUCIAL_BIN_MC_FLOOR": "none"}, ["DayNight"],
               "one volume over the Day-Night window, no radiological floor (before the 2026-10-18 rule)"),
    "_dnone_radfloor": ({"SOLAR_FIDUCIAL_SINGLE_BAND": "1", "SOLAR_FIDUCIAL_BIN_MC_FLOOR": "radiological:2"}, ["DayNight"],
                        "one volume over the Day-Night window, radiological per-bin MC floor 2"),
}
# BestFiducials records exported next to the default (labelled studies of lib/study.py).
RECORD_LABELS = ["fid_noveto", "fid_radfloor_one", "fid_oldrule", "fid_bandrel5", "fid_oneband"]
# Scans whose MC content is tabulated: scan suffix -> description.
MC_SCANS = {"": "default scan", "_fid_radfloor_one": "SolarEnergy, removal veto (scan without veto shift)",
            "_fid_v100_300_60": "SolarEnergy with the no-offset veto shift"}
# Named volumes (F_x, F_y, F_z in cm) tabulated in every scan, per configuration.
MC_VOLUMES = {"hd_1x2x6_centralAPA": [(0, 200, 60), (100, 300, 60), (140, 300, 60), (0, 300, 0), (60, 200, 260), (0, 0, 0)]}


def produce(configs):
    for suffix, (env, analyses, _) in PROFILE_SETS.items():
        for cfg in configs:
            e = dict(os.environ, **env)
            if suffix:
                e["SOLAR_FIDUCIAL_PROFILE_SUFFIX"] = suffix
            cmd = [sys.executable, "src/physics/signal/02_best_fiducial.py", "--config", cfg, "--folder", "Truncated",
                   "--no-debug", "--no-plot", "--signal", NAME, "--energy", "SolarEnergy", "--analysis", *analyses,
                   "--no-write_records", "--export_profiles"]
            print(f"[produce] {cfg} {suffix or 'default'}: {' '.join(cmd[1:])}", flush=True)
            subprocess.run(cmd, cwd=ROOT, env=e, check=True)


def export_profiles(configs):
    rows = []
    for suffix in PROFILE_SETS:
        for cfg in configs:
            src = FID / cfg / NAME / f"{cfg}_{NAME}_SolarEnergy_Fiducial_Profiles{suffix}.pkl"
            if not src.exists():
                rows.append({"Set": suffix or "default", "Config": cfg, "Rows": 0})
                continue
            dst = DST / f"fiducial_profiles{suffix}" / cfg / NAME / f"{cfg}_{NAME}_Fiducial_Profiles.pkl"
            dst.parent.mkdir(parents=True, exist_ok=True)
            df = pd.read_pickle(src)
            df.to_pickle(dst)
            rows.append({"Set": suffix or "default", "Config": cfg, "Rows": len(df)})
    return pd.DataFrame(rows)


def export_volumes(configs):
    rows = []
    for label in ["default"] + RECORD_LABELS:
        path = RECORDS / ("BestFiducials.json" if label == "default" else f"BestFiducials_{label}.json")
        if not path.exists():
            continue
        data = json.load(open(path))
        for cfg in configs:
            for ana in WINDOWS:
                n = data.get(cfg, {}).get(ana, {}).get("SolarEnergy")
                if not n:
                    continue
                reason = n.get("MCLimitedReason", "") or ""
                rows.append({"Label": label, "Config": cfg, "Analysis": ana,
                             "FiducialX": n.get("FiducialX"), "FiducialY": n.get("FiducialY"), "FiducialZ": n.get("FiducialZ"),
                             "SmoothedSignificance": n.get("SmoothedSignificance"),
                             "NoFiducialSignificance": n.get("NoFiducialSignificance"),
                             "MCThreshold": n.get("MCThreshold"), "ChosenEssentialMC": n.get("ChosenEssentialMC"),
                             "Marks": ("*" if "below" in reason else "") + ("†" if "gate changed" in reason else ""),
                             "UngatedBest": n.get("UngatedBest"), "EnergyBands": n.get("EnergyBands") or [],
                             "Record": str(path.relative_to(ROOT))})
    return pd.DataFrame(rows)


def _mc_in(row, lo, hi):
    e = np.asarray(row["Energy"], float)
    mc = np.asarray(row["MCCounts"], float)
    return float(mc[(e >= lo) & (e <= hi)].sum()) if lo is not None else float(mc.sum())


def export_mc_counts(configs, volumes):
    rows = []
    for cfg in configs:
        chosen = volumes[(volumes.Label == "default") & (volumes.Config == cfg)]
        targets = {(int(r.FiducialX), int(r.FiducialY), int(r.FiducialZ)): f"default {r.Analysis}" for r in chosen.itertuples()}
        for v in MC_VOLUMES.get(cfg, []):
            targets.setdefault(tuple(v), "named")
        for scan, desc in MC_SCANS.items():
            for sample in ("gamma_final", "neutron_final", "radiological_final"):
                path = FID / cfg / sample / f"{cfg}_{sample}_SolarEnergy_Fiducial_Scan{scan}.pkl"
                if not path.exists():
                    continue
                df = pd.read_pickle(path)
                for (x, y, z), why in targets.items():
                    r = df[(df.FiducializedX == x) & (df.FiducializedY == y) & (df.FiducializedZ == z)]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    row = {"Config": cfg, "Scan": scan or "default", "ScanDescription": desc, "Sample": sample,
                           "FiducialX": x, "FiducialY": y, "FiducialZ": z, "Volume": why, "MC_all": _mc_in(r, None, None)}
                    for ana, (lo, hi) in WINDOWS.items():
                        row[f"MC_{ana}"] = _mc_in(r, lo, hi)
                    rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--produce", action="store_true", help="run 02_best_fiducial.py --export_profiles for every profile set first")
    ap.add_argument("--configs", nargs="+", default=CONFIGS)
    args = ap.parse_args()
    if args.produce:
        produce(args.configs)
    DST.mkdir(parents=True, exist_ok=True)
    prof = export_profiles(args.configs)
    vol = export_volumes(args.configs)
    vol.to_pickle(DST / "fiducial_volumes.pkl")
    mc = export_mc_counts(args.configs, vol)
    mc.to_pickle(DST / "fiducial_mc_counts.pkl")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    meta = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "commit": commit,
            "profile_sets": {k or "default": v[2] for k, v in PROFILE_SETS.items()}, "record_labels": RECORD_LABELS,
            "mc_scans": MC_SCANS, "profile_rows": prof.to_dict(orient="records")}
    json.dump(meta, open(DST / "meta.json", "w"), indent=1, ensure_ascii=False)
    (DST / "README.md").write_text(
        "# Fiducial-stage export\n\nWritten by `src/tools/export_fiducial_repo.py` (SOLAR, commit "
        f"{commit}); see its docstring for the contents. Profile sets:\n\n"
        + "".join(f"- `fiducial_profiles{k}`: {v[2]}\n" for k, v in PROFILE_SETS.items())
        + "\nPlotting repository: `scripts/sync_solar_data.sh --fiducial`, then `python run_plot_scripts.py -s fidprofile`.\n")
    print(prof.to_string(index=False))
    print(f"fiducial_volumes.pkl {len(vol)} rows, fiducial_mc_counts.pkl {len(mc)} rows -> {DST}")


if __name__ == "__main__":
    main()
