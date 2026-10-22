"""bkgsub check (2026-10-16): weighted rate of a background sample, full vs job-subsampled (lib/subsample.py).

Loads <config> <name> with the 03_analysis preset and background weights, with SOLAR_BKG_SUBSAMPLE unset and set,
and prints the summed SignalParticleWeight (all clusters, and clusters with 6 <= SolarEnergy < 20 MeV) with its MC
error, plus the generated-event count the weights used. Usage: bkgsub_check.py CONFIG NAME SPEC
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
config, name, spec = sys.argv[1], sys.argv[2], sys.argv[3]


def rate(subsample):
    import importlib
    if subsample:
        os.environ["SOLAR_BKG_SUBSAMPLE"] = spec
    else:
        os.environ.pop("SOLAR_BKG_SUBSAMPLE", None)
    import lib.subsample, lib.weights
    importlib.reload(lib.subsample)
    lib.weights._config_npy_cache.clear()
    from lib import load_multi, compute_reco_workflow
    run, _ = load_multi({config: [name]}, preset="SIGNIFICANCE", branches={"Config": ["Geometry"]})
    run = compute_reco_workflow(run, {config: [name]}, params={"PARTICLE_TYPE": "background", "PARTICLE_WEIGHTING": "histogram"},
                                workflow="SIGNIFICANCE", rm_branches=False)
    reco = run["Reco"]
    w = np.asarray(reco["SignalParticleWeight"], dtype=float)
    e = np.asarray(reco["SolarEnergy"], dtype=float)
    win = (e >= 6) & (e < 20)
    return len(w), w.sum(), np.sqrt((w ** 2).sum()), int(win.sum()), w[win].sum(), np.sqrt((w[win] ** 2).sum())


full = rate(False)
part = rate(True)
for label, r in (("full", full), ("subsample", part)):
    print(f"{label:10s} clusters {r[0]:8d} rate {r[1]:.4e} +- {r[2]:.2e} | 6-20 MeV clusters {r[3]:7d} rate {r[4]:.4e} +- {r[5]:.2e}")
for i, lab in ((1, "all"), (4, "6-20 MeV")):
    err = np.hypot(full[i + 1], part[i + 1])
    print(f"{lab}: subsample/full = {part[i] / full[i]:.3f}, difference {(part[i] - full[i]) / err:+.2f} sigma")
