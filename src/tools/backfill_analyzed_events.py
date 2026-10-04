"""
backfill_analyzed_events.py — Write Config/AnalyzedEvents.npy for converted samples
===================================================================================
lib.weights.resolve_generated_event_count uses the SolarNuAna per-job AnalyzedEvents
count (one ConfigTree entry per job) to normalise the truth signal weights, and
cross-checks the productions.json entries against it. Samples converted before
AnalyzedEvents.npy existed only have the zero-padded SelectedEvents.npy, which has lost
the per-job lengths. This script recovers them from the sample's ROOT file: the
AnalyzedEvents branch when present, otherwise the SelectedEvents vector lengths
(SolarNuAna pushes one entry per analysed event, selected or not).

Default is a dry run that prints the per-sample totals next to productions.json;
--write creates the missing files (existing ones are only replaced with --rewrite).
It refuses to write when the ROOT ConfigTree and the npy Config folder disagree on the
number of jobs (the npy folder was then not converted from this ROOT file).

Run
---
  python3 src/tools/backfill_analyzed_events.py --config vd_1x8x14_3view_30deg_shielded hd_1x2x6_lateralAPA --name marley
  python3 src/tools/backfill_analyzed_events.py --config vd_1x8x14_3view_30deg_shielded --name marley --write
"""

import argparse
import json
import os
import sys

import awkward as ak
import numpy as np
import uproot

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.utils import get_project_root

root = get_project_root()


def analyzed_from_root(filename):
    """Per-job analysed-event counts and the branch they came from."""
    with uproot.open(filename) as f:
        tree = f["solarnuana/ConfigTree"]
        if "AnalyzedEvents" in tree.keys():
            return np.asarray(tree["AnalyzedEvents"].array(library="np"), dtype=np.int64), "AnalyzedEvents"
        if "SelectedEvents" in tree.keys():
            return ak.to_numpy(ak.num(tree["SelectedEvents"].array(), axis=1)).astype(np.int64), "SelectedEvents lengths"
    return None, None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--name", nargs="+", default=["marley"])
    parser.add_argument("--write", action="store_true", help="Create missing AnalyzedEvents.npy files")
    parser.add_argument("--rewrite", action="store_true", help="With --write, also replace existing files")
    args = parser.parse_args()

    productions = json.load(open(f"{root}/config/import/productions.json"))
    for config in args.config:
        info = json.load(open(f"{root}/config/{config}/{config}_config.json"))
        base = f"{info['PATH']}/data/{info['GEOMETRY']}/{info['VERSION']}/{info['NAME']}"
        for name in args.name:
            sample = f"{base}{name}"
            target = f"{sample}/Config/AnalyzedEvents.npy"
            label = f"{config}/{name}"
            if not os.path.isdir(f"{sample}/Config"):
                print(f"{label}: no npy Config folder, skipping")
                continue
            n_jobs = len(np.load(f"{sample}/Config/SignalLabel.npy", allow_pickle=True))
            existing = np.load(target) if os.path.isfile(target) else None

            counts, source = (None, None)
            if os.path.isfile(f"{sample}.root"):
                counts, source = analyzed_from_root(f"{sample}.root")
            if counts is None and existing is None:
                print(f"{label}: {n_jobs} jobs, no ROOT file with AnalyzedEvents/SelectedEvents and no npy; cannot backfill")
                continue
            if counts is None:
                counts, source = existing, "existing npy"

            spec = productions.get(config, {}).get(name)
            print(
                f"{label}: {n_jobs} jobs, AnalyzedEvents total {int(counts.sum())} (from {source}; "
                f"per job min {counts.min()} / max {counts.max()}, {int((counts == 0).sum())} empty), "
                f"productions.json {spec!r}"
            )
            if len(counts) != n_jobs:
                print(f"  ROOT ConfigTree has {len(counts)} entries vs {n_jobs} npy jobs: npy not converted from this ROOT file, not writing")
                continue
            if existing is not None and not np.array_equal(existing, counts):
                print(f"  existing AnalyzedEvents.npy differs (total {int(existing.sum())})")
            if args.write and (existing is None or args.rewrite) and source != "existing npy":
                np.save(target, counts)
                print(f"  wrote {target}")


if __name__ == "__main__":
    main()
