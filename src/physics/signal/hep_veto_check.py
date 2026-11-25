"""hep_veto_check.py: what the classifier vetoes (AnalysisKeep) remove from the hep and 8B signal of the HEP analysis.

Read-only on the 03_analysis.py Ref exports (output/data/results/<config>/marley_final/<folder>), for each config:
  - checks that AnalysisKeep_SolarEnergy_HEP removes exactly the clusters whose SolarEnergyHEP carries no
    de-excitation offset (REMOVE_NO_OFFSET_CLUSTERS, lib/background.py no_offset_removal_keep);
  - the removed fraction of hep and 8B (weighted) per offset-energy range, at the Raw-stage energy of the cut flow
    (10-20 MeV of SolarEnergyHEP) and per true neutrino energy;
  - the kept fraction of the unweighted CC clusters, on which the classifier working points are set
    (03b_signal_background.py), with the HEP working point (SolarEnergyHEP) and with the default one (SolarEnergy).
Technical note of the Phase I solar paper, 2026-11-25 (first run as SOLAR-sb _xchk4/hep_veto).

    python3 src/physics/signal/hep_veto_check.py --config hd_1x2x6_centralAPA hd_1x2x6_lateralAPA vd_1x8x14_3view_30deg_nominal
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from lib import root  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--config", nargs="+", default=["hd_1x2x6_centralAPA"])
parser.add_argument("--name", default="marley_final")
parser.add_argument("--folder", default="Truncated")
args = parser.parse_args()

for cfg in args.config:
    R = f"{root}/output/data/results/{cfg}/{args.name}/{args.folder.lower()}/{cfg}_{args.name}_"
    L = lambda f, t=float: np.asarray(pickle.load(open(R + f + ".pkl", "rb")), dtype=t)
    e_sol, e_hep = L("AnalysisData_SolarEnergy_Ref"), L("AnalysisData_SolarEnergyHEP_Ref")
    e_off, e_noff = L("AnalysisData_SolarEnergyOffset_Ref"), L("AnalysisData_SolarEnergyNoOffset_Ref")
    e_true = L("AnalysisEnergy_SolarEnergy_Ref")
    keep = L("AnalysisKeep_SolarEnergy_HEP", bool)
    pe, plane = L("AnalysisOpFlashPE_SolarEnergy_Ref"), L("AnalysisOpFlashPlane_SolarEnergy_Ref", int)
    n = len(e_sol)
    assert all(len(a) == n for a in (e_hep, e_off, e_noff, e_true, keep, pe, plane)), "exports of different clusters"
    rem = ~keep
    print(f"\n#### {cfg}: {n} clusters, removed {rem.mean():.2%}")
    rule = e_hep != e_off
    print(f"mask == (SolarEnergyHEP != SolarEnergyOffset): {np.array_equal(rem, rule)} (mismatches {int((rem != rule).sum())})")
    print(f"clusters with Offset == NoOffset: {(e_off == e_noff).mean():.2%}; removed below 12 MeV offset energy: {int((rem & (e_off < 12)).sum())}")
    for comp in ("hep", "8B"):
        w = L(f"AnalysisWeights{comp}_SolarEnergy_Ref")
        W = lambda m: w[m].sum()
        print(f"== {comp} (weighted): removed {W(rem) / w.sum():.1%} of all clusters")
        for lo, hi in ((0, 12), (12, 14), (14, 99)):
            m = (e_off >= lo) & (e_off < hi)
            print(f"  E_offset {lo:2d}-{hi:2d} MeV: share {W(m) / w.sum():.1%}, removed {W(m & rem) / max(W(m), 1e-30):.1%}")
        m = (e_hep >= 10) & (e_hep < 20)
        print(f"  Raw-stage SolarEnergyHEP 10-20 MeV: removed {W(m & rem) / W(m):.1%}")
        print(f"  removed clusters: mean E_offset {np.average(e_off[rem], weights=w[rem]):.1f}, booked (no offset) "
              f"{np.average(e_noff[rem], weights=w[rem]):.1f}, true {np.average(e_true[rem], weights=w[rem]):.1f} MeV")
        for lo, hi in ((10, 14), (14, 16), (16, 20)):
            m = (e_true >= lo) & (e_true < hi)
            print(f"  true E {lo}-{hi} MeV: removed {W(m & rem) / max(W(m), 1e-30):.1%}")
        matched = (plane == 0) & (pe > 0)
        m = e_off >= 14
        print(f"  E_offset >= 14 MeV with a matched flash: removed {W(m & rem & matched) / W(m & matched):.1%}; "
              f"unmatched share {W(m & ~matched) / W(m):.1%}")
    print("== unweighted CC clusters (classifier working points)")
    print(f"  true E range {e_true.min():.1f}-{e_true.max():.1f} MeV")
    for lo, hi in ((12, 99), (12, 14), (14, 99)):
        m = (e_off >= lo) & (e_off < hi)
        print(f"  HEP working point, E_offset {lo}-{hi}: kept {1 - rem[m].mean():.1%} (median true E {np.median(e_true[m]):.1f} MeV)")
    kept_default = e_sol == e_off
    w = L("AnalysisWeightshep_SolarEnergy_Ref")
    for lo, hi in ((14, 99), (14, 16), (16, 20), (20, 99)):
        m = (e_off >= lo) & (e_off < hi)
        print(f"  default working point, E_offset {lo}-{hi}: unweighted kept {kept_default[m].mean():.1%}, "
              f"hep-weighted kept {w[m & kept_default].sum() / w[m].sum():.1%}")
