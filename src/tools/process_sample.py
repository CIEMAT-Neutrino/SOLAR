"""
process_sample.py — Non-interactive root -> npy conversion
==========================================================
Converts a SolarNuAna ROOT file into the Config/Truth/Reco npy folder layout that
`load_multi` reads, like src/tools/processing.py, but:

  - maps trees by name (ConfigTree -> Config, MCTruthTree -> Truth,
    SolarNuAnaTree -> Reco) instead of by position, so files whose TTrees come in a
    different order (e.g. the 2026-09 flash productions) land in the right folder;
  - never prompts;
  - skips the raw waveform branches (OpFlashWaveform, MatchedOpFlashWaveform) and
    anything passed with --skip;
  - skips the per-event truth OpFlash vectors (MCTruthTree OpFlashPE, ...) unless
    --truth_opflash is given: radiological events carry thousands of flashes, so zero
    padding them costs ~2 GB per branch per 25k events. Flash-level studies read them
    straight from the ROOT file (src/physics/detector/pds/05_flash_distributions.py);
  - pads jagged branches with awkward (same zero padding as lib.io.resize_subarrays)
    instead of going through python lists, which is what makes the multi-million
    entry background Reco trees tractable.

Run
---
  python3 src/tools/process_sample.py --config vd_1x8x14_3view_30deg_shielded --name marley_flash
  python3 src/tools/process_sample.py --config hd_1x2x6_centralAPA --name marley_flash --trees Config Truth
"""

import argparse
import json
import os
import sys

import awkward as ak
import numpy as np
import uproot

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from lib.io import resize_subarrays
from src.utils import get_project_root

root = get_project_root()

TREE_LABELS = {"ConfigTree": "Config", "MCTruthTree": "Truth", "SolarNuAnaTree": "Reco"}
TRUTH_OPFLASH = [
    "OpFlashPur", "OpFlashID", "OpFlashPE", "OpFlashX", "OpFlashY", "OpFlashZ",
    "OpFlashTime", "OpFlashSTD", "OpFlashNHits", "OpFlashPlane", "OpFlashMaxPE",
    "OpFlashWaveformValid",
]
DEFAULT_SKIP = [
    "OpFlashWaveform",
    "MatchedOpFlashWaveform",
    "MatchedOpFlashWaveformTime",
    "MatchedOpFlashWaveformValid",
]


def branch_to_numpy(array):
    """Flat branches go straight to numpy; jagged ones are zero-padded to the longest entry."""
    if array.ndim == 1:
        try:
            return ak.to_numpy(array)
        except Exception:
            return np.asarray(ak.to_list(array))
    try:
        counts = ak.num(array, axis=1)
        max_len = int(ak.max(counts)) if len(counts) else 0
        if array.ndim == 2 and max_len > 0 and not np.all(ak.to_numpy(counts) == max_len):
            dtype = ak.to_numpy(ak.flatten(array)).dtype  # fill_none(0) would promote to float64
            padded = ak.fill_none(ak.pad_none(array, max_len, axis=1, clip=True), 0)
            return ak.to_numpy(padded).astype(dtype)
        return ak.to_numpy(array)
    except Exception:
        # Strings, maps, nested vectors: fall back to the original converter.
        return resize_subarrays(array, 0)


def save(filename, data, rewrite):
    if os.path.isfile(filename):
        if not rewrite:
            return False
        os.remove(filename)
    np.save(filename, data)
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert a SolarNuAna ROOT file to npy folders")
    parser.add_argument("--config", required=True, help="Detector configuration (config/<config>)")
    parser.add_argument("--name", required=True, help="Sample name, e.g. marley_flash")
    parser.add_argument("--trees", nargs="+", default=["Config", "Truth", "Reco"])
    parser.add_argument("--skip", nargs="*", default=[], help="Extra branches to skip")
    parser.add_argument("--truth_opflash", action="store_true", help="Also convert the truth OpFlash vectors")
    parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    info = json.load(open(f"{root}/config/{args.config}/{args.config}_config.json"))
    path = f'{info["PATH"]}/data/{info["GEOMETRY"]}/{info["VERSION"]}/'
    name = f'{info["NAME"]}{args.name}'
    filename = f"{path}{name}.root"
    if not os.path.isfile(filename):
        sys.exit(f"ERROR: {filename} not found")

    skip = set(DEFAULT_SKIP + args.skip + ([] if args.truth_opflash else TRUTH_OPFLASH))
    tree_info = {"Path": path, "Name": name, "TreeNames": {}}
    with uproot.open(filename) as f:
        folder = [k.split(";")[0] for k, c in f.classnames().items() if c == "TDirectory"][0]
        tree_info["Folder"] = folder
        trees = {}
        for key, cls in f.classnames().items():
            tree = key.split("/")[-1].split(";")[0]
            if cls == "TTree" and TREE_LABELS.get(tree) in args.trees:
                trees[tree] = f[key]  # classnames() already keeps the highest cycle

        for tree, ttree in trees.items():
            label = TREE_LABELS[tree]
            out = f"{path}{name}/{label}"
            os.makedirs(out, exist_ok=True)
            branches = [b for b in ttree.keys() if b not in skip]
            print(f"{tree} -> {label}: {ttree.num_entries} entries, {len(branches)} branches", flush=True)
            for branch in branches:
                target = f"{out}/{branch}.npy"
                if not args.rewrite and os.path.isfile(target):
                    continue
                data = branch_to_numpy(ttree[branch].array())
                save(target, data, True)
                print(f"  {branch}: {data.dtype} {data.shape}", flush=True)
                del data
            # Older SolarNuAna builds have SelectedEvents but no AnalyzedEvents branch; the
            # padded SelectedEvents.npy loses the per-job lengths, so keep them here
            # (lib.weights.resolve_generated_event_count reads this file).
            if label == "Config" and "SelectedEvents" in branches and "AnalyzedEvents" not in branches:
                target = f"{out}/AnalyzedEvents.npy"
                if args.rewrite or not os.path.isfile(target):
                    save(target, ak.to_numpy(ak.num(ttree["SelectedEvents"].array(), axis=1)).astype(np.int64), True)
                    print(f"  AnalyzedEvents (from SelectedEvents lengths)", flush=True)
            save(f"{out}/Branches.npy", np.asarray(branches, dtype=object), True)
            tree_info[f"{tree};1"] = np.asarray(branches, dtype=object)
            tree_info["TreeNames"][f"{tree};1"] = label

    save(f"{path}{name}/TTrees.npy", np.asarray(tree_info, dtype=object), True)
    print(f"Done: {path}{name}/", flush=True)


if __name__ == "__main__":
    main()
