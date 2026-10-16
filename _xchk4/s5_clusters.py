"""Step 2b: background clusters in the analysis window that leave (in A, not in B) or enter between two volumes.

usage: s5_clusters.py config analysis xA yA zA xB yB zB [minfrac]
Listed: clusters carrying >= minfrac (default 0.01) of their energy bin's total background in A (all components).
Weights per year (Truncated model). Window by bin centres as in 02_best_fiducial.py. fill_gaps of 01 is not applied
(a lone cluster's weight is split half/half with an empty lower neighbour bin in the scan).
"""
import sys
sys.path.insert(0, "/pc/choozdsk01/users/manthey/SOLAR-sb/_xchk4")
from fidlib import *
config, an = sys.argv[1], sys.argv[2]
A = tuple(int(v) for v in sys.argv[3:6]); Bv = tuple(int(v) for v in sys.argv[6:9])
minfrac = float(sys.argv[9]) if len(sys.argv) > 9 else 0.01
acfg, _ = analysis_setup(an)
lo, hi = acfg["energy_min"], acfg["energy_max"]
pd.set_option("display.width", 250)
ev = {s: load_events(config, s) for s in BKG}
binA = {}
for s, d in ev.items():
    ia = in_volume(d, config, A)
    for b, w in d[ia].groupby("Bin").Weight.sum().items():
        binA[b] = binA.get(b, 0.0) + w
rows = []
for s, d in ev.items():
    win = (d.BinCenter >= lo) & (d.BinCenter <= hi)
    ia, ib = in_volume(d, config, A), in_volume(d, config, Bv)
    for lab, m in (("leaves", win & ia & ~ib), ("enters", win & ib & ~ia)):
        q = d[m].copy()
        if not len(q):
            continue
        q["Sample"] = COMP[s]; q["Move"] = lab
        q["BinBkgA"] = [binA.get(b, 0.0) for b in q.Bin]
        q["FracOfBin"] = q.Weight / q.BinBkgA
        print(f"{COMP[s]:>12s} {lab}: {len(q)} clusters, {q.Weight.sum():.4g}/yr; max fraction of a bin {q.FracOfBin.max():.3g}")
        rows.append(q)
q = pd.concat(rows)
q = q[q.FracOfBin >= minfrac].sort_values("FracOfBin", ascending=False)
cols = ["Sample", "Move", "Weight", "SolarEnergy", "BinCenter", "BinBkgA", "FracOfBin", "RecoX", "RecoY", "RecoZ",
        "SignalParticleK", "MainPDG", "MainK", "Generator", "Event"]
print(f"clusters with >= {minfrac:g} of their bin background ({CONFIGS[config]} {an} {A} -> {Bv}):")
print(q[[c for c in cols if c in q]].head(30).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
