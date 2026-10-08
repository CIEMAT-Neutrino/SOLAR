import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *
from lib.sigma import evaluate_profile_likelihood_discovery

"""
02_hep_energy_limit.py: HEP discovery significance versus the upper edge of the energy window, with its
MC-statistics uncertainty.

At the HEP best cut of 01_hep.py (read from the HEP_Counts pkl), the significance is recomputed for windows
14 MeV - E_max with the 01_hep.py metric: raw Rebin spectra (1 MeV hep_rebin), profile-likelihood discovery
significance with the global background-normalisation nuisance, the 01_hep.py bin mask (summed background MCCounts,
8B included, >= --min_mc_per_bin), counts = rate x exposure x full detector mass.

Uncertainty: Poisson bootstrap of the simulated events behind every bin. Each component bin is rescaled by
N'/N with N' ~ Poisson(N = MCCounts); hep and 8B are the same marley events and share the draw. The band is the
16-84% interval of the bootstrap significance.

The recommended upper edge is the largest E_max whose relative uncertainty (half the 16-84% width over the
median) stays at or below --max_relative_error: beyond it the significance is carried by a handful of simulated
events in bins without simulated background.
"""

parser = argparse.ArgumentParser(description="HEP significance versus the upper edge of the energy window")
parser.add_argument("--config", nargs="+", default=["hd_1x2x6_centralAPA", "hd_1x2x6_lateralAPA", "vd_1x8x14_3view_30deg_nominal", "vd_1x8x14_3view_30deg_shielded"])
parser.add_argument("--signal", type=str, default="marley_final")
parser.add_argument("--backgrounds", nargs="+", default=None, help="Background samples (default: backgrounds.json HEP list)")
parser.add_argument("--folder", type=str, default="Truncated")
parser.add_argument("--energy", type=str, default="SolarEnergy")
parser.add_argument("--threshold", type=float, default=14.0, help="Lower edge of the HEP window (MeV)")
parser.add_argument("--exposure", type=float, default=None, help="Years (default: the exposure of the HEP_Counts pkl)")
parser.add_argument("--background_uncertainty", type=float, default=0.02)
parser.add_argument("--min_mc_per_bin", type=float, default=1.0)
parser.add_argument("--bootstrap", type=int, default=1000)
parser.add_argument("--max_relative_error", type=float, default=0.2)
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()

analysis_info = load_analysis_info(str(root))
backgrounds = args.backgrounds or list(get_background_samples(str(root), "HEP"))
edges = np.asarray(hep_rebin, dtype=float)
lo_idx = int(np.searchsorted(edges, args.threshold))
emax_grid = edges[lo_idx + 1:]
rng = np.random.default_rng(args.seed)
pnfs = "/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR"


def row(df, component, osc, cut):
    nh, oph, adj = cut
    r = df[(df["Component"] == component) & (df["Oscillation"] == osc) & (df["Mean"] == "Mean")
           & (df["NHits"] == nh) & (df["OpHits"] == oph) & (df["AdjCl"] == adj)]
    if r.empty:
        return None, None
    return np.asarray(r.iloc[0]["Counts"], dtype=float), np.asarray(r.iloc[0]["MCCounts"], dtype=float)


def significance(s, b, mc, emax):
    m = (edges[:-1] >= args.threshold) & (edges[1:] <= emax) & (mc >= args.min_mc_per_bin)
    return float(evaluate_profile_likelihood_discovery(np.where(m, s, 0.0), np.where(m, b, 0.0),
                                                       background_uncertainty=args.background_uncertainty))


rows_out = []
for config in args.config:
    info = json.load(open(f"{root}/config/{config}/{config}_config.json"))
    counts_pkl = f"{root}/output/data/analysis/hep/{config}/{args.signal}/{args.folder.lower()}/default/{config}_{args.signal}_HEP_Counts.pkl"
    ref = pd.read_pickle(counts_pkl).iloc[0]
    cut = (int(ref["NHits"]), int(ref["OpHits"]), int(ref["AdjCl"]))
    exposure = float(args.exposure if args.exposure is not None else ref["Exposure"])
    factor = exposure * get_full_detector_mass(config, info)
    sig = pd.read_pickle(f"{pnfs}/signal/{args.folder.lower()}/HEP/{config}/{args.signal}/{config}_{args.signal}_{args.energy}_Rebin.pkl")
    hep, hep_mc = row(sig, "hep", "Osc", cut)
    b8, _ = row(sig, "8B", "Osc", cut)
    comps = [("marley", None, hep_mc)]
    bkg_counts = []
    for bkg in backgrounds:
        p = f"{pnfs}/background/{args.folder.lower()}/HEP/{config}/{bkg}/{config}_{bkg}_{args.energy}_Rebin.pkl"
        if not os.path.exists(p):
            rprint(f"[yellow][WARNING][/yellow] {config}: no {bkg} Rebin ({p})"); continue
        c, mc = row(pd.read_pickle(p), bkg, "Truth", cut)
        if c is not None:
            bkg_counts.append((bkg, c, mc))
    mc_total = hep_mc + sum(mc for _, _, mc in bkg_counts)  # 8B shares the marley MC, as in 01_hep.py

    def spectra(scale=None):
        s = factor * hep * (1 if scale is None else scale["marley"])
        b = factor * b8 * (1 if scale is None else scale["marley"])
        for name, c, _ in bkg_counts:
            b = b + factor * c * (1 if scale is None else scale[name])
        return s, b

    s0, b0 = spectra()
    nominal = np.array([significance(s0, b0, mc_total, e) for e in emax_grid])
    boot = np.zeros((args.bootstrap, len(emax_grid)))
    for k in range(args.bootstrap):
        scale = {"marley": np.divide(rng.poisson(hep_mc), hep_mc, out=np.zeros_like(hep_mc), where=hep_mc > 0)}
        for name, _, mc in bkg_counts:
            scale[name] = np.divide(rng.poisson(mc), mc, out=np.zeros_like(mc), where=mc > 0)
        s, b = spectra(scale)
        boot[k] = [significance(s, b, mc_total, e) for e in emax_grid]
    p16, p50, p84 = np.percentile(boot, [16, 50, 84], axis=0)
    rel = np.divide(0.5 * (p84 - p16), p50, out=np.full_like(p50, np.inf), where=p50 > 0)
    ok = np.where(rel <= args.max_relative_error)[0]
    chosen = float(emax_grid[ok.max()]) if ok.size else float("nan")
    hep_mc_above = np.array([float(hep_mc[(edges[:-1] >= e - 1) & (edges[1:] <= e)].sum()) for e in emax_grid])
    rprint(f"\n## {config} ({args.signal}, cut {cut}, {exposure:g} yr): recommended E_max {chosen:g} MeV (relative error <= {args.max_relative_error:g})")
    for e, z, a, b_, c, r_, n in zip(emax_grid, nominal, p16, p50, p84, rel, hep_mc_above):
        rprint(f"   E_max {e:4.0f}  Z {z:6.2f}  bootstrap 16/50/84 {a:6.2f} {b_:6.2f} {c:6.2f}  rel {r_:5.2f}  signal MC in last bin {n:5.0f}")
    rows_out.append({
        "Geometry": info["GEOMETRY"], "Config": config, "Name": args.signal, "Analysis": "HEP", "EnergyLabel": args.energy,
        "NHits": cut[0], "OpHits": cut[1], "AdjCl": cut[2], "Exposure": exposure, "ExposureUnit": "year",
        "Variable": "ProfileLikelihood", "SpectrumType": "Raw", "Study": "default",
        "EnergyMax": emax_grid, "EnergyMaxUnit": "MeV", "Significance": nominal, "SignificanceUnit": "\\sigma",
        "SignificanceP16": p16, "SignificanceP50": p50, "SignificanceP84": p84, "RelativeError": rel,
        "SignalMCLastBin": hep_mc_above, "RecommendedEnergyMax": chosen, "MaxRelativeError": args.max_relative_error,
        "Bootstrap": args.bootstrap,
    })
    data_path = f"{root}/output/data/analysis/hep"
    save_df(pd.DataFrame([rows_out[-1]]), data_path, config, args.signal, subfolder=f"{args.folder.lower()}/default",
            filename="HEP_EnergyLimit", rm=args.rewrite, debug=args.debug)
print("HEP ENERGY LIMIT DONE")
