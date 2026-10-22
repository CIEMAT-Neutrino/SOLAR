"""HEP without the tail extrapolation: alternative treatments of the bins the selection leaves without external MC.

Re-scans the default HEP Rebin grid (default fiducial volume) with the 01_hep.py metric as reproduced by
02_hep_energy_limit.py (raw spectra, profile-likelihood discovery, signal x (1 - pl_conservative_sigma x sigma_s),
background normalisation nuisance, bin mask summed MC >= 1) and the 01_hep.py MC gate, and picks the best cut per
variant at EXPOSURE years:
  T    tails as quoted (validation: the default HEP result)
  N    no tails (validation: the hep_notail study)
  W25  no tails, window upper edge 25 MeV
  WE   no tails, window up to the per-config external endpoint (last bin the loosest cut populates with gamma/neutron)
  X1   no tails, a bin enters only with >= 1 simulated gamma+neutron cluster at the cut
  X3   same with >= 3
  F    no tails, every external component gets a one-event floor (1.14 x its mean event weight at the cut) in the
       bins where the cut leaves it no simulated cluster, up to its loosest-cut endpoint
  LF   no tails, one-event floor (1.14 x the loosest-cut mean event weight) only in the bins the cut empties although
       the loosest cut populates them; bins beyond the loosest-cut endpoint stay background free
  LS   no tails, each external spectrum = its loosest-cut shape scaled to the cut's total (no fit)
  LS26 / LF26  LS / LF with the window upper edge at 26 MeV
  LSX  LS, and a bin needs simulated external MC at the loosest cut
Usage: hep_alt.py [EXPOSURE]
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from lib import *  # noqa
from lib.sigma import evaluate_profile_likelihood_discovery
from lib.background import adaptive_mc_thresholds, external_tail_model, extrapolated_external_tail

EXPOSURE = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
CONFIGS = ["hd_1x2x6_centralAPA", "hd_1x2x6_lateralAPA", "vd_1x8x14_3view_30deg_nominal", "vd_1x8x14_3view_30deg_shielded"]
PNFS = "/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR"
info_all = load_analysis_info(str(root))
tail_cfg = info_all.get("HEP_TAIL_EXTRAPOLATION", {})
sig_unc = float(info_all.get("ANALYSIS_UNCERTAINTIES", {}).get("HEP", {}).get("signal_uncertainty", info_all.get("SIGNAL_ERROR", 0.04)))
scale = 1.0 - float(get_workflow_flags(str(root), "HEP").get("pl_conservative_sigma", 0)) * sig_unc
bkg_unc = float(info_all.get("BACKGROUND_ERROR", 0.02))
edges = np.asarray(hep_rebin, dtype=float)
centers = 0.5 * (edges[:-1] + edges[1:])
mc_thr_default = float(info_all["ANALYSIS_THRESHOLDS"]["HEP"]["MC"])
essential = [s for s, e in get_essential_backgrounds(str(root)).items() if e]
VARIANTS = ["T", "N", "W25", "WE", "X1", "X3", "F", "LF", "LS", "LS26", "LF26", "LSX"]


def rows(df, comp, osc):
    d = df[(df["Component"] == comp) & (df["Oscillation"] == osc) & (df["Mean"] == "Mean")]
    return {(int(r.NHits), int(r.OpHits), int(r.AdjCl)): (np.asarray(r.Counts, float), np.asarray(r.MCCounts, float)) for r in d.itertuples()}


def pl(s, b, m):
    return float(evaluate_profile_likelihood_discovery(np.where(m, s, 0.0), np.where(m, b, 0.0), background_uncertainty=bkg_unc))


for config in CONFIGS:
    info = json.load(open(f"{root}/config/{config}/{config}_config.json"))
    factor = EXPOSURE * get_full_detector_mass(config, info)
    thr = float(info_all.get("MC_THRESHOLDS_PER_CONFIG", {}).get(config, {}).get("HEP", mc_thr_default))
    sig = pd.read_pickle(f"{PNFS}/signal/truncated/HEP/{config}/marley_final/{config}_marley_final_SolarEnergy_Rebin.pkl")
    hep, b8 = rows(sig, "hep", "Osc"), rows(sig, "8B", "Osc")
    bkgs = {}
    for name in get_background_samples(str(root), "HEP"):
        p = f"{PNFS}/background/truncated/HEP/{config}/{name}/{config}_{name}_SolarEnergy_Rebin.pkl"
        if os.path.exists(p):
            bkgs[name] = rows(pd.read_pickle(p), name, "Truth")
    external = [n for n in bkgs if n.split("_")[0] in ("gamma", "neutron")]
    # loosest-cut tails and endpoints
    tails, endpoint, loose_arr = {}, {}, {}
    for name in external:
        cuts = list(bkgs[name])
        loose = (min(c[0] for c in cuts), min(c[1] for c in cuts), max(c[2] for c in cuts))
        lc, lmc = bkgs[name][loose]
        loose_arr[name] = (lc, lmc, lc.sum() / lmc.sum() if lmc.sum() > 0 else 0.0)
        pop = np.flatnonzero(lmc > 0)
        endpoint[name] = float(edges[pop.max() + 1]) if pop.size else 0.0
        if config in tail_cfg.get("ENABLED_CONFIGS", []) and name.split("_")[0] in tail_cfg.get("COMPONENTS", ["gamma", "neutron"]):
            tails[name] = external_tail_model(lc, lmc, edges, float(tail_cfg.get("FIT_THRESHOLD", 0.0)))
    e_ext = max(endpoint.values()) if endpoint else 30.0
    mc_by = {n: {c: float(v[1].sum()) for c, v in bkgs[n].items()} for n in bkgs}
    thresholds = adaptive_mc_thresholds(mc_by, essential, thr, thr)
    best = {v: (0.0, None) for v in VARIANTS}
    default_cut = {"hd_1x2x6_centralAPA": (2, 8, 4), "hd_1x2x6_lateralAPA": (3, 6, 3), "vd_1x8x14_3view_30deg_nominal": (6, 4, 20),
                   "vd_1x8x14_3view_30deg_shielded": (6, 8, 18)}[config]
    at_default = {}
    for cut in hep:
        if cut not in b8 or any(cut not in bkgs[n] for n in bkgs):
            continue
        if not all(mc_by[n].get(cut, 0.0) >= t for n, t in thresholds.items() if t > 0):
            continue
        s = factor * scale * hep[cut][0]
        b_base = factor * b8[cut][0]
        mc_tot = hep[cut][1].copy()
        ext_mc = np.zeros_like(mc_tot)
        b_nt = b_base.copy()
        b_t = b_base.copy()
        b_f = b_base.copy()
        b_lf = b_base.copy()
        b_ls = b_base.copy()
        loose_ext_mc = np.zeros_like(mc_tot)
        for n in bkgs:
            c, mc = bkgs[n][cut]
            mc_tot += mc
            b_nt += factor * c
            b_t += factor * c
            b_f += factor * c
            b_lf += factor * c
            if n not in external:
                b_ls += factor * c
            if n in external:
                lc, lmc, wl = loose_arr[n]
                loose_ext_mc += lmc
                b_lf += factor * np.where((mc == 0) & (lmc > 0), 1.14 * wl, 0.0)
                b_ls += factor * (lc * (c.sum() / lc.sum()) if lc.sum() > 0 else c)
                ext_mc += mc
                if tails.get(n) is not None:
                    b_t += factor * extrapolated_external_tail(c, mc, tails[n], edges, 0.0, tail_cfg.get("MODE", "endpoint"))
                w = c.sum() / mc.sum() if mc.sum() > 0 else 0.0
                empty = (mc == 0) & (edges[1:] <= endpoint[n])
                b_f += factor * np.where(empty, 1.14 * w, 0.0)
        base = mc_tot >= 1
        z = {
            "T": pl(s, b_t, base),
            "N": pl(s, b_nt, base),
            "W25": pl(s, b_nt, base & (edges[1:] <= 25)),
            "WE": pl(s, b_nt, base & (edges[1:] <= e_ext)),
            "X1": pl(s, b_nt, base & (ext_mc >= 1)),
            "X3": pl(s, b_nt, base & (ext_mc >= 3)),
            "F": pl(s, b_f, base),
            "LF": pl(s, b_lf, base),
            "LS": pl(s, b_ls, base),
            "LS26": pl(s, b_ls, base & (edges[1:] <= 26)),
            "LF26": pl(s, b_lf, base & (edges[1:] <= 26)),
            "LSX": pl(s, b_ls, base & (loose_ext_mc >= 1)),
        }
        if cut == default_cut:
            at_default = dict(z)
        for v, val in z.items():
            if val > best[v][0]:
                best[v] = (val, cut)
    print(f"{config}: HEP MC gate {thr:g}, external endpoints " + ", ".join(f"{n} {e:g}" for n, e in endpoint.items())
          + f" MeV, tails {sorted(k for k, v in tails.items() if v is not None) or 'none'}")
    for v in VARIANTS:
        val, cut = best[v]
        print(f"   {v:4s} Z {val:6.2f} at {cut}   (at the default cut {default_cut}: {at_default.get(v, float('nan')):.2f})")
print("HEP ALT DONE")
