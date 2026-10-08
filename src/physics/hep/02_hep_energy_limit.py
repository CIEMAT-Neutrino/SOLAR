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

Uncertainty: Poisson bootstrap of the simulated statistics behind every bin. Each component bin is rescaled by
N'/N with N' ~ Poisson(N), N = (Counts / Error)^2 the effective number of simulated events (Error = sqrt(sum w^2)
in the Rebin pkls): a bin of 400 marley events carries the hep weight of only one or two of them when they lie
above the hep endpoint. Components are drawn independently. The band is the 16-84% interval of the bootstrap.

Unsimulated background: a bin with hep signal but no simulated external background (gamma, neutron) has an
expectation set to zero only because the sample has no event there. Its plausible range is 0 to the 68% Poisson upper
limit for zero observed events, 1.14 times the mean weight of one simulated event of each external sample at this
cut. Z_floor is the significance with that upper limit put in every such bin of the window; the lower edge of the
band is the 16% point of the bootstrap of Z_floor, the upper edge the 84% point of the nominal bootstrap.

The recommended upper edge is the largest E_max whose relative uncertainty, (median - lower edge) / median, stays at
or below --max_relative_error: beyond it the significance rests on bins whose background the simulation cannot
estimate.
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
        return None, None, None
    c = np.asarray(r.iloc[0]["Counts"], dtype=float); e = np.asarray(r.iloc[0]["Error"], dtype=float)
    neff = np.divide(c * c, e * e, out=np.zeros_like(c), where=e > 0)
    return c, np.asarray(r.iloc[0]["MCCounts"], dtype=float), neff


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
    hep, hep_mc, hep_neff = row(sig, "hep", "Osc", cut)
    b8, _, b8_neff = row(sig, "8B", "Osc", cut)
    bkg_counts = []
    for bkg in backgrounds:
        p = f"{pnfs}/background/{args.folder.lower()}/HEP/{config}/{bkg}/{config}_{bkg}_{args.energy}_Rebin.pkl"
        if not os.path.exists(p):
            rprint(f"[yellow][WARNING][/yellow] {config}: no {bkg} Rebin ({p})"); continue
        c, mc, neff = row(pd.read_pickle(p), bkg, "Truth", cut)
        if c is not None:
            bkg_counts.append((bkg, c, mc, neff))
    mc_total = hep_mc + sum(x[2] for x in bkg_counts)  # 8B shares the marley MC, as in 01_hep.py
    # one-event background floor for bins without simulated external background
    ext = [x for x in bkg_counts if x[0].split("_")[0] in ("gamma", "neutron")]  # external samples
    ext_mc = sum(x[2] for x in ext) if ext else np.zeros_like(hep_mc)
    one_event = sum(float(np.sum(x[1][x[2] > 0]) / max(np.sum(x[2][x[2] > 0]), 1.0)) for x in ext)
    floor = np.where((ext_mc <= 0) & (edges[:-1] >= args.threshold), 1.14 * one_event, 0.0)

    def spectra(scale=None):
        s = factor * hep * (1 if scale is None else scale["hep"])
        b = factor * b8 * (1 if scale is None else scale["8B"])
        for name, c, _, _ in bkg_counts:
            b = b + factor * c * (1 if scale is None else scale[name])
        return s, b

    s0, b0 = spectra()
    nominal = np.array([significance(s0, b0, mc_total, e) for e in emax_grid])
    floored = np.array([significance(s0, b0 + factor * floor, mc_total, e) for e in emax_grid])
    boot = np.zeros((args.bootstrap, len(emax_grid))); boot_f = np.zeros_like(boot)
    for k in range(args.bootstrap):
        draw = lambda n: np.divide(rng.poisson(n), n, out=np.zeros_like(n), where=n > 0)
        scale = {"hep": draw(hep_neff), "8B": draw(b8_neff)}
        for name, _, _, neff in bkg_counts:
            scale[name] = draw(neff)
        s, b = spectra(scale)
        boot[k] = [significance(s, b, mc_total, e) for e in emax_grid]
        boot_f[k] = [significance(s, b + factor * floor, mc_total, e) for e in emax_grid]
    p16, p50, p84 = np.percentile(boot, [16, 50, 84], axis=0)
    low = np.percentile(boot_f, 16, axis=0)
    rel = np.divide(nominal - low, nominal, out=np.full_like(nominal, np.inf), where=nominal > 0)
    ok = np.where(rel <= args.max_relative_error)[0]
    chosen = float(emax_grid[ok.max()]) if ok.size else float("nan")
    hep_mc_above = np.array([float(hep_neff[(edges[:-1] >= e - 1) & (edges[1:] <= e)].sum()) for e in emax_grid])
    rprint(f"\n## {config} ({args.signal}, cut {cut}, {exposure:g} yr): recommended E_max {chosen:g} MeV (relative error <= {args.max_relative_error:g})")
    for e, z, l_, c, f_, r_, n in zip(emax_grid, nominal, low, p84, floored, rel, hep_mc_above):
        rprint(f"   E_max {e:4.0f}  Z {z:6.2f}  band {l_:6.2f} - {c:6.2f}  (Z_floor {f_:6.2f})  rel {r_:5.2f}  hep effective MC events in last bin {n:6.1f}")
    rows_out.append({
        "Geometry": info["GEOMETRY"], "Config": config, "Name": args.signal, "Analysis": "HEP", "EnergyLabel": args.energy,
        "NHits": cut[0], "OpHits": cut[1], "AdjCl": cut[2], "Exposure": exposure, "ExposureUnit": "year",
        "Variable": "ProfileLikelihood", "SpectrumType": "Raw", "Study": "default",
        "EnergyMax": emax_grid, "EnergyMaxUnit": "MeV", "Significance": nominal, "SignificanceUnit": "\\sigma",
        "SignificanceLow": low, "SignificanceHigh": p84, "SignificanceFloor": floored, "SignificanceP16": p16,
        "SignificanceP50": p50, "SignificanceP84": p84, "RelativeError": rel, "OneEventBackground": factor * one_event,
        "HepEffectiveMCLastBin": hep_mc_above, "RecommendedEnergyMax": chosen, "MaxRelativeError": args.max_relative_error,
        "Bootstrap": args.bootstrap,
    })
    data_path = f"{root}/output/data/analysis/hep"
    save_df(pd.DataFrame([rows_out[-1]]), data_path, config, args.signal, subfolder=f"{args.folder.lower()}/default",
            filename="HEP_EnergyLimit", rm=args.rewrite, debug=args.debug)
print("HEP ENERGY LIMIT DONE")
