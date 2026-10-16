"""Step 2a: decompose a profile jump between volumes A and B per component and energy bin.

usage: s3_decompose.py config analysis xA yA zA xB yB zB
Attribution: Z(B) recomputed with one component's histograms (Counts, MCCounts, Error) taken from A.
"""
import sys
sys.path.insert(0, "/pc/choozdsk01/users/manthey/SOLAR-sb/_xchk4")
from fidlib import *

config, an = sys.argv[1], sys.argv[2]
A = tuple(int(v) for v in sys.argv[3:6]); Bv = tuple(int(v) for v in sys.argv[6:9])
pdf = load_plot_df(config)
acfg, scfg = analysis_setup(an)
lo, hi = acfg["energy_min"], acfg["energy_max"]
comps = list(acfg["signal_components"]) + list(acfg["background_components"])
dA, dB = volume_df(pdf, A), volume_df(pdf, Bv)
pa, pb = payload(pdf, A, acfg, scfg), payload(pdf, Bv, acfg, scfg)
print(f"{CONFIGS[config]} {an}  A={A}: raw {pa['RawTotal']:.3f} smoothed {pa['SmoothedTotal']:.3f}   B={Bv}: raw {pb['RawTotal']:.3f} smoothed {pb['SmoothedTotal']:.3f}")
print(f"smoothing: {scfg.get('method')} params {scfg.get('params')} preserve_integral {scfg.get('preserve_integral', True)}")
E = pa["Energy"]; w = (E >= lo) & (E <= hi)
print("bin  E   | " + " | ".join(f"{c:>12s} cntA  cntB  mcA mcB" for c in comps) + " | zrawA zrawB zsmA zsmB  bkgSmA bkgSmB")
for i in np.where(w)[0]:
    e = E[i]
    cells = []
    for c in comps:
        ra = dA[(dA.Component == c) & np.isclose(dA.Energy, e)]; rb = dB[(dB.Component == c) & np.isclose(dB.Energy, e)]
        ca, cb = float(ra.Counts.sum()), float(rb.Counts.sum()); ma, mb = float(ra.MCCounts.sum()), float(rb.MCCounts.sum())
        cells.append(f"{c:>6s} {ca*100:9.3g} {cb*100:9.3g} {ma:4.0f} {mb:4.0f}")
    print(f"{i:3d} {e:5.1f} | " + " | ".join(cells) +
          f" | {pa['RawSignificance'][i]:.3f} {pb['RawSignificance'][i]:.3f} {pa['SmoothedSignificance'][i]:.3f} {pb['SmoothedSignificance'][i]:.3f}  {pa['SmoothedBackground'][i]:.3g} {pb['SmoothedBackground'][i]:.3g}")
print("(counts x100 yr exposure)")
# attribution: swap one component of B back to A
for c in comps:
    fromA = dA[dA.Component == c].copy()
    fromA[["FiducializedX", "FiducializedY", "FiducializedZ"]] = list(Bv)
    mix = pd.concat([dB[dB.Component != c], fromA])
    m = merged_for(mix, acfg)
    p = build_significance_payload(m, acfg["significance_type"], acfg.get("combine_mode", "quadrature"), scfg, 100.0, lo, hi)
    print(f"Z(B with {c:>12s} from A): raw {p['RawTotal']:.3f} smoothed {p['SmoothedTotal']:.3f}")
# window totals per component
for c in comps:
    for lab, d in (("A", dA), ("B", dB)):
        q = d[(d.Component == c) & (d.Energy >= lo) & (d.Energy <= hi)]
        print(f"  window {c:>12s} {lab}: counts {q.Counts.sum()*100:.4g} MC {q.MCCounts.sum():.0f}")
