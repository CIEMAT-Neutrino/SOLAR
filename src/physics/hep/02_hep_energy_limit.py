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
threshold - E_max (threshold: the 01_hep.py analysis threshold) with the 01_hep.py metric: raw Rebin spectra (1 MeV hep_rebin), profile-likelihood discovery
significance with the global background-normalisation nuisance, the 01_hep.py bin mask (summed background MCCounts,
8B included, >= --min_mc_per_bin), counts = rate x exposure x full detector mass.

Analysis curve: the 01_hep.py model as quoted, i.e. the signal at (1 - pl_conservative_sigma x signal uncertainty) and,
for the configs in HEP_TAIL_EXTRAPOLATION.ENABLED_CONFIGS, the external tails of HEP_TAIL_EXTRAPOLATION.COMPONENTS
(lib.background external_tail_model / extrapolated_external_tail, fit from FIT_THRESHOLD - 4 MeV): where the analysis
selection leaves a bin of an external sample empty, its expectation is the exponential tail of the same sample at the
loosest cut of the scan (same fiducial volume, far more simulated events), scaled to the analysis cut and stopped one
bin above the last bin the loosest cut populates. At the full window it reproduces the quoted HEP result.

Band (2026-10-14): 16-84% interval of a bootstrap of the analysis curve. Each replicate rescales every component bin by
N'/N with N' ~ Poisson(N), N = (Counts / Error)^2 the effective number of simulated events (a bin of 400 marley events
carries the hep weight of only one or two of them above the hep endpoint), and draws the slope of every extrapolated
tail from its fit uncertainty (pivot at the centre of the fitted range). Components are drawn independently.

SignificanceNoTail: the same metric without the tail extrapolation (bins the selection leaves empty enter as
background free), which shows how much of the reach would otherwise rest on the empty bins.

The recommended upper edge is the largest E_max whose relative uncertainty, (nominal - lower edge) / nominal, stays at
or below --max_relative_error.
"""

parser = argparse.ArgumentParser(description="HEP significance versus the upper edge of the energy window")
parser.add_argument("--config", nargs="+", default=["hd_1x2x6_centralAPA", "hd_1x2x6_lateralAPA", "vd_1x8x14_3view_30deg_nominal", "vd_1x8x14_3view_30deg_shielded"])
parser.add_argument("--signal", type=str, default="marley_final")
parser.add_argument("--backgrounds", nargs="+", default=None, help="Background samples (default: backgrounds.json HEP list)")
parser.add_argument("--folder", type=str, default="Truncated")
parser.add_argument("--energy", type=str, default="SolarEnergy")
parser.add_argument("--threshold", type=float, default=get_analysis_threshold(str(root), "HEP", stage="SIGNIFICANCE", fallback=0.0),
    help="Lower edge of the window (MeV); default: the 01_hep.py analysis threshold, so the full window reproduces the quoted result")
parser.add_argument("--exposure", type=float, default=None, help="Years (default: the exposure of the HEP_Counts pkl)")
parser.add_argument("--background_uncertainty", type=float, default=0.02)
parser.add_argument("--signal_uncertainty", type=float, default=None, help="Default: ANALYSIS_UNCERTAINTIES.HEP (as run_sensitivity.py)")
parser.add_argument("--min_mc_per_bin", type=float, default=1.0)
parser.add_argument("--bootstrap", type=int, default=1000)
parser.add_argument("--max_relative_error", type=float, default=0.2)
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--tail_mode", choices=["endpoint", "exponential"], default="endpoint",
    help="endpoint: the extrapolated external tail stops one bin above the last bin the loosest cut populates (true "
         "endpoint of the gamma/neutron response); exponential: the fitted tail continues to the end of the window.")
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()

analysis_info = load_analysis_info(str(root))
tail_cfg = analysis_info.get("HEP_TAIL_EXTRAPOLATION", {})
signal_uncertainty = float(args.signal_uncertainty if args.signal_uncertainty is not None else
    analysis_info.get("ANALYSIS_UNCERTAINTIES", {}).get("HEP", {}).get("signal_uncertainty", analysis_info.get("SIGNAL_ERROR", 0.04)))
signal_scale = 1.0 - float(get_workflow_flags(str(root), "HEP").get("pl_conservative_sigma", 0)) * signal_uncertainty
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
    # external tails extrapolated from the loosest cut of the scan, as in 01_hep.py (see the module docstring)
    from lib.background import external_tail_model, extrapolated_external_tail
    fit_thr = float(tail_cfg.get("FIT_THRESHOLD", args.threshold))
    tails = {}
    for name, c, mc, _ in bkg_counts:
        if config not in tail_cfg.get("ENABLED_CONFIGS", []) or name.split("_")[0] not in tail_cfg.get("COMPONENTS", ["gamma", "neutron"]):
            continue
        df = pd.read_pickle(f"{pnfs}/background/{args.folder.lower()}/HEP/{config}/{name}/{config}_{name}_{args.energy}_Rebin.pkl")
        df = df[(df["Component"] == name) & (df["Oscillation"] == "Truth") & (df["Mean"] == "Mean")]
        loose = df[(df["NHits"] == df["NHits"].min()) & (df["OpHits"] == df["OpHits"].min()) & (df["AdjCl"] == df["AdjCl"].max())].iloc[0]
        model = external_tail_model(loose["Counts"], loose["MCCounts"], edges, fit_thr)
        if model is None:
            continue
        slope, icpt, last, lc, lmc = model
        centers = 0.5 * (edges[:-1] + edges[1:])
        fit = (lmc >= 3) & (centers >= fit_thr - 4) & (np.arange(len(lc)) <= last) & (lc > 0)
        _, cov = np.polyfit(centers[fit], np.log(lc[fit]), 1, w=np.sqrt(lmc[fit]), cov=True)
        tails[name] = (model, float(np.sqrt(cov[0, 0])), float(centers[fit].mean()))
        rprint(f"   {config} {name}: loosest-cut tail slope {slope:.2f} +- {np.sqrt(cov[0, 0]):.2f} /MeV (fit from {fit_thr - 4:g} MeV), last populated bin {centers[last]:.1f} MeV")

    def tail_of(name, c, mc, dslope=0.0):
        model, _, x0 = tails[name]
        slope, icpt, last, lc, lmc = model
        moved = (slope + dslope, icpt - dslope * x0, last, lc, lmc)
        return extrapolated_external_tail(c, mc, moved, edges, args.threshold, tail_cfg.get("MODE", "endpoint"))

    def spectra(scale=None, dslope=None, with_tails=True):
        sc = (lambda k: 1.0) if scale is None else (lambda k: scale[k])
        s = factor * signal_scale * hep * sc("hep")
        b = factor * b8 * sc("8B")
        for name, c, mc, _ in bkg_counts:
            cc = c * sc(name)
            if with_tails and name in tails:
                # the tail fills bins without simulated events, so its spread is the fit uncertainty of the slope only
                cc = cc + tail_of(name, c, mc, 0.0 if dslope is None else dslope[name])
            b = b + factor * cc
        return s, b

    s0, b0 = spectra()
    nominal = np.array([significance(s0, b0, mc_total, e) for e in emax_grid])
    s_nt, b_nt = spectra(with_tails=False)
    notail = np.array([significance(s_nt, b_nt, mc_total, e) for e in emax_grid])
    boot = np.zeros((args.bootstrap, len(emax_grid)))
    for k in range(args.bootstrap):
        draw = lambda n: np.divide(rng.poisson(n), n, out=np.zeros_like(n), where=n > 0)
        scale = {"hep": draw(hep_neff), "8B": draw(b8_neff)}
        for name, _, _, neff in bkg_counts:
            scale[name] = draw(neff)
        dslope = {name: float(rng.normal(0.0, t[1])) for name, t in tails.items()}
        s, b = spectra(scale, dslope)
        boot[k] = [significance(s, b, mc_total, e) for e in emax_grid]
    p16, p50, p84 = np.percentile(boot, [16, 50, 84], axis=0)
    low = p16
    rel = np.divide(nominal - low, nominal, out=np.full_like(nominal, np.inf), where=nominal > 0)
    ok = np.where(rel <= args.max_relative_error)[0]
    chosen = float(emax_grid[ok.max()]) if ok.size else float("nan")
    b8_window = np.array([float(factor * b8[(edges[:-1] >= e - 1) & (edges[1:] <= e)].sum()) for e in emax_grid])
    hep_mc_above = np.array([float(hep_neff[(edges[:-1] >= e - 1) & (edges[1:] <= e)].sum()) for e in emax_grid])
    rprint(f"\n## {config} ({args.signal}, cut {cut}, {exposure:g} yr, tails {sorted(tails) or 'none'}): recommended E_max {chosen:g} MeV (relative error <= {args.max_relative_error:g})")
    for e, z, l_, c, f_, r_, n, b8w in zip(emax_grid, nominal, low, p84, notail, rel, hep_mc_above, b8_window):
        rprint(f"   E_max {e:4.0f}  Z {z:6.2f}  band {l_:6.2f} - {c:6.2f}  (no tails {f_:6.2f})  rel {r_:5.2f}  hep N_eff in last bin {n:6.1f}  8B in last bin {b8w:.2g}")
    floored = nominal  # kept for the SignificanceTail column (the analysis now includes the configured tails)
    tail_bkg = (b0 - b_nt)
    rows_out.append({
        "Geometry": info["GEOMETRY"], "Config": config, "Name": args.signal, "Analysis": "HEP", "EnergyLabel": args.energy,
        "NHits": cut[0], "OpHits": cut[1], "AdjCl": cut[2], "Exposure": exposure, "ExposureUnit": "year",
        "Variable": "ProfileLikelihood", "SpectrumType": "Raw", "Study": "default",
        "EnergyMax": emax_grid, "EnergyMaxUnit": "MeV", "Significance": nominal, "SignificanceUnit": "\\sigma",
        "SignificanceLow": low, "SignificanceHigh": p84, "SignificanceTail": floored, "SignificanceNoTail": notail, "SignificanceP16": p16,
        "SignificanceP50": p50, "SignificanceP84": p84, "RelativeError": rel, "TailBackground": tail_bkg, "TailComponents": sorted(tails), "SignalScale": signal_scale,
        "HepEffectiveMCLastBin": hep_mc_above, "B8LastBin": b8_window, "RecommendedEnergyMax": chosen, "MaxRelativeError": args.max_relative_error,
        "Bootstrap": args.bootstrap,
    })
    data_path = f"{root}/output/data/analysis/hep"
    save_df(pd.DataFrame([rows_out[-1]]), data_path, config, args.signal, subfolder=f"{args.folder.lower()}/default",
            filename="HEP_EnergyLimit", rm=args.rewrite, debug=args.debug)
print("HEP ENERGY LIMIT DONE")
