"""Shape check for hep_looseshape: share of the 14-30 MeV external background above E at the selected cut, at the loosest
cut and at the loosest OpHits/AdjCl cut with the same NHits (default HEP Rebin pkls, default fiducial volume).
Uncertainty: binomial on the simulated clusters of each spectrum. Usage: shape_check.py"""
import os, sys
import numpy as np, pandas as pd
PNFS = "/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/background/truncated/HEP"
CUTS = {"hd_1x2x6_centralAPA": (4, 4, 6), "hd_1x2x6_lateralAPA": (4, 10, 5), "vd_1x8x14_3view_30deg_nominal": (6, 4, 20),
        "vd_1x8x14_3view_30deg_shielded": (5, 20, 15)}  # hep_looseshape chosen cuts (NHits, OpHits, AdjCl)
E0, THR = 14.0, (18.0, 20.0, 22.0)
def share(c, mc):
    e = np.arange(len(c)) + 0.5
    w = (e >= E0) & (e < 30)
    tot, ntot = c[w].sum(), mc[w].sum()
    out = []
    for t in THR:
        f = c[w & (e >= t)].sum() / tot if tot > 0 else np.nan
        err = np.sqrt(f * (1 - f) / ntot) if ntot > 0 and np.isfinite(f) else np.nan
        out.append(f"{f:6.3f}±{err:5.3f}")
    return " ".join(out) + f"  (MC {ntot:.0f})"
for cfg, (nh, oh, ad) in CUTS.items():
    for comp in ("gamma_final", "neutron_final"):
        df = pd.read_pickle(f"{PNFS}/{cfg}/{comp}/{cfg}_{comp}_SolarEnergy_Rebin.pkl")
        df = df[(df.Oscillation == "Truth") & (df.Mean == "Mean")]
        def row(q):
            r = df[q]
            return np.asarray(r.iloc[0].Counts, float), np.asarray(r.iloc[0].MCCounts, float)
        tight = row((df.NHits == nh) & (df.OpHits == oh) & (df.AdjCl == ad))
        loose = row((df.NHits == df.NHits.min()) & (df.OpHits == df.OpHits.min()) & (df.AdjCl == df.AdjCl.max()))
        sub = df[df.NHits == nh]
        loosenh = row((df.NHits == nh) & (df.OpHits == sub.OpHits.min()) & (df.AdjCl == sub.AdjCl.max()))
        print(f"{cfg} {comp} cut {nh}/{oh}/{ad}  share above {THR} MeV of 14-30 MeV")
        print(f"   selected cut      {share(*tight)}")
        print(f"   loosest cut       {share(*loose)}")
        print(f"   loosest at NHits  {share(*loosenh)}")
print("SHAPE CHECK DONE")
