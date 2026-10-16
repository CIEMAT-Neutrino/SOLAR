"""Job-level subsampling of background samples (study bkgsub, run_sensitivity --bkg_subsample).

SOLAR_BKG_SUBSAMPLE="F:SEED" (or "match=<config>:SEED") keeps a random fraction F of the whole jobs (ConfigTree
entries) of the samples in SOLAR_BKG_SUBSAMPLE_SAMPLES (default gamma_final,neutron_final) and drops every Truth and
Reco entry of the other jobs. lib.io.load_multi applies the masks to every tree it loads and lib.weights.load_config_counts
to the per-job event counts, so N_gen (AnalyzedEvents / SelectedEvents / Truth entries / events_per_job x jobs) is the
count of the kept jobs and the weights rescale with it. "match=<config>" sets F per sample to the generated events
(sum of Config/AnalyzedEvents) of the same sample in <config> over those of the loaded config, capped at 1.
"""
import os
import zlib
from typing import Optional

import numpy as np

_MASK_CACHE: dict = {}


def subsample_spec() -> Optional[tuple]:
    """(fraction or "match=<config>", seed) from SOLAR_BKG_SUBSAMPLE, or None when it is not set."""
    spec = os.environ.get("SOLAR_BKG_SUBSAMPLE", "").strip()
    if not spec:
        return None
    fraction, _, seed = spec.rpartition(":")
    if not fraction:
        raise ValueError(f"SOLAR_BKG_SUBSAMPLE={spec!r}: expected F:SEED or match=<config>:SEED")
    return fraction, int(seed)


def subsampled_samples() -> list:
    return [s.strip() for s in os.environ.get("SOLAR_BKG_SUBSAMPLE_SAMPLES", "gamma_final,neutron_final").split(",") if s.strip()]


def _per_job(sample_dir: str, branch: str) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(np.load(f"{sample_dir}/Config/{branch}.npy", allow_pickle=True))
    except FileNotFoundError:
        return None
    if branch == "SelectedEvents":
        arr = arr.reshape(len(arr), -1).astype(np.int64).sum(axis=1)
    return np.asarray(arr, dtype=np.int64).ravel()


def _fraction(fraction: str, sample_dir: str, name: str, info: dict, root: str) -> float:
    if not fraction.startswith("match="):
        return float(fraction)
    import json
    other = fraction.split("=", 1)[1]
    other_info = json.load(open(f"{root}/config/{other}/{other}_config.json"))
    other_dir = f"{other_info['PATH']}/data/{other_info['GEOMETRY']}/{other_info['VERSION']}/{other_info['NAME']}{name}"
    mine, theirs = _per_job(sample_dir, "AnalyzedEvents"), _per_job(other_dir, "AnalyzedEvents")
    if mine is None or theirs is None:
        raise FileNotFoundError(f"--bkg_subsample {fraction}: Config/AnalyzedEvents.npy missing for {name} ({sample_dir} or {other_dir})")
    return min(1.0, float(theirs.sum()) / float(mine.sum()))


def subsample_masks(sample_dir: str, name: str, info: Optional[dict], root: str) -> Optional[dict]:
    """{"Config", "Truth", "Reco": boolean entry masks, "fraction", "kept", "jobs"} for a subsampled sample, else None.

    Truth entries follow the job order of the merged file, one per event the analyzer filled (per-job SelectedEvents or
    AnalyzedEvents, whichever sums to the Truth length). Reco entries are matched to their Truth event in order through
    (Flag, Event), both trees being in job order.
    """
    spec = subsample_spec()
    if spec is None or name not in subsampled_samples() or info is None:
        return None
    key = (sample_dir, spec)
    if key in _MASK_CACHE:
        return _MASK_CACHE[key]
    fraction_spec, seed = spec
    fraction = _fraction(fraction_spec, sample_dir, name, info, root)
    analyzed = _per_job(sample_dir, "AnalyzedEvents")
    if analyzed is None:
        raise FileNotFoundError(f"bkg_subsample: {sample_dir}/Config/AnalyzedEvents.npy is needed to subsample by job")
    n_jobs = len(analyzed)
    rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
    n_keep = max(1, int(round(fraction * n_jobs)))
    kept = np.zeros(n_jobs, dtype=bool)
    kept[rng.choice(n_jobs, size=n_keep, replace=False)] = True

    truth_flag = np.asarray(np.load(f"{sample_dir}/Truth/Flag.npy", allow_pickle=True))
    truth_event = np.asarray(np.load(f"{sample_dir}/Truth/Event.npy", allow_pickle=True))
    per_job = None
    for branch in ("SelectedEvents", "AnalyzedEvents"):
        counts = _per_job(sample_dir, branch)
        if counts is not None and len(counts) == n_jobs and int(counts.sum()) == len(truth_event):
            per_job = counts
            break
    if per_job is None:
        raise ValueError(f"bkg_subsample: no per-job event count of {sample_dir} sums to its {len(truth_event)} Truth entries")
    truth_job = np.repeat(np.arange(n_jobs), per_job)

    reco_flag = np.asarray(np.load(f"{sample_dir}/Reco/Flag.npy", allow_pickle=True))
    reco_event = np.asarray(np.load(f"{sample_dir}/Reco/Event.npy", allow_pickle=True))
    reco_job = np.empty(len(reco_flag), dtype=np.int64)
    if len(reco_flag):
        change = np.r_[True, (reco_flag[1:] != reco_flag[:-1]) | (reco_event[1:] != reco_event[:-1])]
        starts = np.flatnonzero(change)
        ends = np.r_[starts[1:], len(reco_flag)]
        tf, te = truth_flag.tolist(), truth_event.tolist()
        t, n_truth = 0, len(tf)
        for start, end in zip(starts.tolist(), ends.tolist()):
            flag, event = reco_flag[start], reco_event[start]
            while t < n_truth and (tf[t] != flag or te[t] != event):
                t += 1
            if t == n_truth:
                raise ValueError(f"bkg_subsample: Reco entry {start} of {sample_dir} (Flag {flag}, Event {event}) has no Truth event in order")
            reco_job[start:end] = truth_job[t]
    masks = {
        "Config": kept,
        "Truth": kept[truth_job],
        "Reco": kept[reco_job],
        "fraction": fraction,
        "kept": int(kept.sum()),
        "jobs": n_jobs,
    }
    from rich import print as rprint
    rprint(
        f"[cyan][INFO][/cyan] bkg_subsample {fraction_spec}:{seed} -> {name}: {masks['kept']} of {n_jobs} jobs "
        f"(F = {fraction:.4f}), {int(masks['Truth'].sum())} of {len(truth_event)} events, "
        f"{int(masks['Reco'].sum())} of {len(reco_flag)} clusters, generated {int(analyzed[kept].sum())} of {int(analyzed.sum())}"
    )
    _MASK_CACHE[key] = masks
    return masks
