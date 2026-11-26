"""20-yr hep discovery significance of the HEP cut flows with (analysis) and without (none) the HEP working point of the
S/B classifier (tools/hep_working_point_check.sh): last stage of each cut flow, lib/sigma.py profile likelihood with a
2 % global background normalisation, raw and smoothed counts, energy windows from 0, 10 and 14 MeV, bins with background
(all) or with at least one simulated background cluster (mc1, the 01_hep.py mask). Read-only.

    python3 src/tools/hep_working_point_significance.py OUT [CONFIG ...]
"""
import os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from lib.sigma import evaluate_profile_likelihood_discovery as pl

out = sys.argv[1]
cfgs = sys.argv[2:] or ["hd_1x2x6_centralAPA", "hd_1x2x6_lateralAPA", "vd_1x8x14_3view_30deg_nominal"]


def load(variant, config):
    d = pickle.load(open(f"{out}/{variant}/{config}/truncated/hep/{config}_SolarEnergy_HEP_Cutflow.pkl", "rb"))
    d = d[d.Stage == "NHits+OpHits+AdjCl"] if "NHits+OpHits+AdjCl" in set(d.Stage) else d[d.Stage == d.Stage.iloc[-1]]
    return {r.Component: r for r in d.itertuples()}


for c in cfgs:
    print("==", c)
    for v in ("analysis", "none"):
        r = load(v, c)
        e = np.asarray(r["hep"].Energy, float)
        s = np.asarray(r["hep"].Counts, float)
        bk = [k for k in r if k != "hep"]
        for key in ("Counts", "SmoothedCounts"):
            b = sum(np.asarray(getattr(r[k], key), float) for k in bk)
            mc = sum(np.asarray(r[k].MCCounts, float) for k in bk)
            for lo in (0, 10, 14):
                for mask in ("all", "mc1"):
                    m = (e >= lo) & ((mc >= 1) if mask == "mc1" else (b > 0))
                    w = m & (e >= 14) & (e < 20)
                    print("  %-8s %-14s E>=%2d %-3s Z=%6.2f  hep14-20=%7.1f bkg14-20=%9.1f"
                          % (v, key, lo, mask, pl(s[m], b[m], 0.02), s[w].sum(), b[w].sum()))
