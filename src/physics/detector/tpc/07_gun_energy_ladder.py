"""
07_gun_energy_ladder.py — Where the Energy of a 10 MeV Electron or Gamma Goes
=============================================================================
Single-particle productions (FNAL justIN gun_1008: mono-energetic e- and gamma, isotropic, uniform in the
active volume, le_mc_2024a chain, SolarNuAna with SaveClusterEDep) follow each event's energy down to
the charge-response energy of its leading primary cluster:

  E_true                       kinetic energy of the gun particle
  E_dep    = SignalIDEEDep[col] energy deposited on the collection channels (SimIDEs)     -> containment
  E_hits   = SignalHitEDep[col] deposited energy that ends up inside collection hits      -> hit threshold
  E_clu    = SignalEDep         deposited energy inside the hits of the leading cluster   -> clustering
  E_charge = Q exp(|t|/tau) / G charge response of the leading cluster (as 04/06)         -> charge response

  E_charge / E_true = (E_dep/E_true) (E_hits/E_dep) (E_clu/E_hits) (E_charge/E_clu)

and describes the deposit topology from the collection-view SimIDEs (energy-weighted RMS radius, number
of separate deposit groups at a linking distance, energy outside the largest group) and, for gammas,
the first interaction (pair production when a positron is among the daughters of the gun particle).
For thermal-neutron guns (neutrongun_*) E_true is the summed energy of the capture gammas (daughters of
the neutron); the cascade multiplicity and sum are exported, and the Total step is also given against
the Ar-40(n,gamma) Q-value (--capture_q), since the default NeutronHP capture does not conserve energy
per event.

Exports to output/data/TPC/smearing/<config>/<name>/gun/:
  Gun_Ladder     ratio distributions per ladder step (Bins, Density) with peak/median metrics
  Gun_Topology   topology distributions
  Gun_Process    E_charge/E_true for gammas split by first interaction
  Gun_Events     one row per event (unbinned)
  <config>_<name>_Gun_Summary.json
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import uproot
from scipy.optimize import curve_fit
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from lib import *

data_path = f"{root}/output/data/TPC/smearing"

parser = argparse.ArgumentParser(description="Energy ladder of single 10 MeV electrons and gammas")
parser.add_argument("--config", type=str, default="hd_1x2x6_centralAPA")
parser.add_argument("--names", nargs="+", type=str, default=["electrongun_10MeV", "gammagun_10MeV"])
parser.add_argument("--input_dir", type=str, default=None,
                    help="Read SolarNuAna_<name>.root from this folder instead of the SOLAR data path (tests)")
parser.add_argument("--capture_q", type=float, default=6.099, help="Ar-40(n,gamma)Ar-41 Q-value (MeV)")
parser.add_argument("--link_distance", type=float, default=2.0, help="Deposit-group linking distance (cm)")
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()
config = args.config
info = json.load(open(f"{root}/config/{config}/{config}_config.json"))

STEPS = {
    "Containment": "Deposited / true energy",
    "HitThreshold": "In hits / deposited",
    "Clustering": "In leading cluster / in hits",
    "ChargeResponse": "Charge response / in leading cluster",
    "Total": "Charge response / true energy",
    "TotalQ": "Charge response / capture Q-value",
}
r_edges = np.arange(0.0, 1.5 + 1e-9, 0.01)
r_centers = 0.5 * (r_edges[1:] + r_edges[:-1])


def ratio_peak(ratio):
    """Iterated Gaussian fit around the mode above 0.5 (as 04/06, wider window for the ladder steps)."""
    ratio = ratio[np.isfinite(ratio) & (ratio > 0.3) & (ratio < 1.5)]
    if len(ratio) < 100:
        return np.nan, np.nan
    counts, bin_edges = np.histogram(ratio, bins=240, range=(0.3, 1.5))
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    upper = bin_centers > 0.5
    mu, sigma = bin_centers[upper][np.argmax(counts[upper])], 0.05
    for _ in range(5):
        win = (bin_centers > mu - 1.5 * sigma) & (bin_centers < mu + 1.5 * sigma) & (counts > 0)
        if win.sum() < 5:
            break
        try:
            popt, _ = curve_fit(lambda x, a, m, s: a * np.exp(-0.5 * ((x - m) / s) ** 2),
                                bin_centers[win], counts[win], p0=[counts[win].max(), mu, sigma])
        except RuntimeError:
            break
        mu, sigma = popt[1], abs(popt[2])
    return float(mu), float(sigma)


def metrics(values):
    v = values[np.isfinite(values)]
    mu, sigma = ratio_peak(v)
    if not len(v):
        return {"Events": 0, "PeakMean": np.nan, "PeakSigma": np.nan, "Mean": np.nan, "Median": np.nan,
                "P16": np.nan, "P84": np.nan, "FracBelow0p5": np.nan}
    return {"Events": int(len(v)), "PeakMean": mu, "PeakSigma": sigma, "Mean": float(np.mean(v)),
            "Median": float(np.median(v)), "P16": float(np.percentile(v, 16)), "P84": float(np.percentile(v, 84)),
            "FracBelow0p5": float(np.mean(v < 0.5))}


def topology(x, y, z, e, link):
    """Energy-weighted RMS radius, number of deposit groups and energy fraction outside the largest group."""
    if len(e) == 0 or np.sum(e) <= 0:
        return np.nan, 0, np.nan
    pos = np.column_stack([x, y, z])
    centroid = np.average(pos, axis=0, weights=e)
    rms = float(np.sqrt(np.average(np.sum((pos - centroid) ** 2, axis=1), weights=e)))
    if len(e) == 1:
        return rms, 1, 0.0
    pairs = cKDTree(pos).query_pairs(link, output_type="ndarray")
    from scipy.sparse import coo_matrix
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(e), len(e)))
    n, labels = connected_components(graph, directed=False)
    group_e = np.bincount(labels, weights=e)
    return rms, int(n), float(1 - group_e.max() / group_e.sum())


for name in args.names:
    particle = "electron" if name.startswith("electron") else "neutron" if name.startswith("neutron") else "gamma"
    path = (f"{args.input_dir}/{info['NAME']}{name}.root" if args.input_dir
            else f'{info["PATH"]}/data/{info["GEOMETRY"]}/{info["VERSION"]}/{info["NAME"]}{name}.root')
    rprint(f"\n[bold]{config} {name}[/bold]: {path}")
    f = uproot.open(path)
    truth = f["solarnuana/MCTruthTree"].arrays(
        ["Event", "Flag", "SignalParticleX", "SignalParticleY", "SignalParticleZ", "SignalParticleK", "SignalIDEEDep", "SignalIDENElectrons", "SignalHitEDep", "SignalHitNElectrons",
         "TSignalPDG", "TSignalMother", "TSignalID", "TSignalEDepList", "TSignalXDepList", "TSignalYDepList", "TSignalZDepList"],
        library="np")
    reco = f["solarnuana/SolarNuAnaTree"].arrays(
        ["Event", "Flag", "SignalParticleX", "SignalParticleY", "SignalParticleZ", "Primary", "Charge", "NHits", "Time", "MainTime", "SignalEDep", "SignalNElectrons", "Purity"], library="np")
    corr = None
    for sample in ["marley_final", "marley_official"]:
        p = f"{root}/config/{config}/{sample}/{config}_calib/{config}_electroncharge_correction.json"
        if os.path.exists(p):
            corr = json.load(open(p))
            break
    tau, gain = corr["ELECTRON_TAU"], corr["CHARGE_AMP"]

    # Events: Flag (rand() without seed) and Event repeat across the merged jobs, so the random gun vertex
    # completes the key
    def event_key(tree):
        return [f"{a}_{b}_{x:.3f}_{y:.3f}_{z:.3f}" for a, b, x, y, z in zip(
            tree["Flag"], tree["Event"], tree["SignalParticleX"], tree["SignalParticleY"], tree["SignalParticleZ"])]
    ev_key = pd.Index(event_key(truth))
    if ev_key.has_duplicates:
        rprint(f"[yellow][WARNING][/yellow] {ev_key.duplicated().sum()} duplicated event keys")
    cl = pd.DataFrame({k: reco[k] for k in ["Charge", "NHits", "Time", "MainTime", "SignalEDep", "SignalNElectrons", "Purity"]})
    cl["Key"] = event_key(reco)
    cl["Primary"] = reco["Primary"].astype(bool)
    true_time = cl["Time"] - 1e-3 * cl["MainTime"]
    cl["EChargeAll"] = cl["Charge"] * np.clip(cl["Purity"], 0, 1) * np.exp(np.abs(true_time) / tau) / gain
    cl = cl[cl["Primary"]]
    lead = cl.sort_values("EChargeAll", ascending=False).groupby("Key").head(1).set_index("Key")
    n_primary = cl.groupby("Key").size()

    col = 2
    k = truth["SignalParticleK"].astype(float)
    capture = [(np.asarray(p) == 22) & (np.asarray(m) == 1) for p, m in zip(truth["TSignalPDG"], truth["TSignalMother"])]
    cascade_sum = np.array([np.sum(np.asarray(kk, dtype=float)[c]) for kk, c in zip(f["solarnuana/MCTruthTree"]["TSignalK"].array(library="np"), capture)])
    cascade_mult = np.array([int(np.sum(c)) for c in capture])
    if particle == "neutron":
        k = np.where(cascade_sum > 0, cascade_sum, np.nan)  # no capture in the event: no truth reference
        rprint(f"  capture cascades: {100 * np.mean(cascade_mult > 0):.1f}% of events, sum median {np.nanmedian(k):.3f} MeV "
               f"(16-84%: {np.nanpercentile(k, 16):.2f}-{np.nanpercentile(k, 84):.2f}), multiplicity median {np.median(cascade_mult[cascade_mult > 0]):.0f}")
    e_dep = np.array([v[col] if len(v) > col else np.nan for v in truth["SignalIDEEDep"]], dtype=float)
    e_hits = np.array([v[col] if len(v) > col else np.nan for v in truth["SignalHitEDep"]], dtype=float)
    n_el_dep = np.array([v[col] if len(v) > col else np.nan for v in truth["SignalIDENElectrons"]], dtype=float)
    n_el_hits = np.array([v[col] if len(v) > col else np.nan for v in truth["SignalHitNElectrons"]], dtype=float)
    lead_r = lead.reindex(ev_key)
    e_clu = lead_r["SignalEDep"].to_numpy(dtype=float)
    e_charge = lead_r["EChargeAll"].to_numpy(dtype=float)
    nhits = lead_r["NHits"].to_numpy(dtype=float)
    n_prim = n_primary.reindex(ev_key).fillna(0).to_numpy()

    # First interaction of the gun particle (track ID 1): pair production if a positron is a daughter
    pair = np.array([np.any((np.asarray(p) == -11) & (np.asarray(m) == 1)) for p, m in zip(truth["TSignalPDG"], truth["TSignalMother"])])
    topo = np.array([topology(x, y, z, e, args.link_distance) for x, y, z, e in
                     zip(truth["TSignalXDepList"], truth["TSignalYDepList"], truth["TSignalZDepList"], truth["TSignalEDepList"])], dtype=object)
    rms = topo[:, 0].astype(float)
    groups = topo[:, 1].astype(float)
    outside = topo[:, 2].astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = {
            "Containment": e_dep / k,
            "HitThreshold": e_hits / e_dep,
            "Clustering": e_clu / e_hits,
            "ChargeResponse": e_charge / e_clu,
            "Total": e_charge / k,
        }
        if particle == "neutron":
            ratios["TotalQ"] = e_charge / args.capture_q
        electrons_per_mev_dep = n_el_dep / e_dep
        electrons_per_mev_hits = n_el_hits / e_hits
    has_cluster = np.isfinite(e_charge)
    rprint(f"  {len(k)} events, {100 * has_cluster.mean():.1f}% with a primary cluster, "
           f"{np.mean(n_prim[has_cluster]):.2f} primary clusters per event with one; pair production {100 * pair.mean():.1f}%")

    ladder_rows, summary_steps = [], {}
    for step, r in ratios.items():
        sel = has_cluster if step in ["Clustering", "ChargeResponse", "Total", "TotalQ"] else np.ones(len(k), dtype=bool)
        h = np.histogram(r[sel & np.isfinite(r)], bins=r_edges)[0].astype(float)
        m = metrics(r[sel])
        summary_steps[step] = m
        ladder_rows.append({"Config": config, "Name": name, "Particle": particle, "Step": step, "Label": STEPS[step],
                            "Bins": r_centers, "Counts": h, "Density": h / max(h.sum() * np.diff(r_edges)[0], 1e-300), **m})
        rprint(f"  {STEPS[step]:<40} mean {m['Mean']:.3f}  median {m['Median']:.3f}  peak {m['PeakMean']:.3f} σ {m['PeakSigma']:.3f}")
    closure = np.nanmax(np.abs(ratios["Containment"] * ratios["HitThreshold"] * ratios["Clustering"] * ratios["ChargeResponse"]
                               - ratios["Total"])[has_cluster & (e_dep > 0) & (e_hits > 0) & (e_clu > 0)])
    rprint(f"  ladder closure max |Π - total| = {closure:.2e}")

    topo_vars = {
        "RMSRadius": (rms, np.arange(0, 60.001, 1.0)),
        "DepositGroups": (groups, np.arange(0.5, 30.5, 1.0)),
        "OutsideLargestGroup": (outside, np.arange(0, 1.0001, 0.02)),
        "LeadingNHits": (nhits, np.arange(0.5, 20.5, 1.0)),
        "PrimaryClusters": (n_prim, np.arange(-0.5, 8.5, 1.0)),
        "CascadeSum": (np.where(cascade_mult > 0, cascade_sum, np.nan), np.arange(0, 20.001, 0.25)),
        "CascadeMultiplicity": (np.where(cascade_mult > 0, cascade_mult, np.nan), np.arange(0.5, 12.5, 1.0)),
        "ElectronsPerMeVDeposit": (electrons_per_mev_dep, np.linspace(0, 40000, 81)),
        "ElectronsPerMeVHits": (electrons_per_mev_hits, np.linspace(0, 40000, 81)),
    }
    topo_rows = []
    for var, (values, bins) in topo_vars.items():
        v = values[np.isfinite(values)]
        h = np.histogram(v, bins=bins)[0].astype(float)
        topo_rows.append({"Config": config, "Name": name, "Particle": particle, "Quantity": var,
                          "Bins": 0.5 * (bins[1:] + bins[:-1]), "Counts": h, "Density": h / max(h.sum(), 1e-300),
                          "Median": float(np.median(v)) if len(v) else np.nan, "Mean": float(np.mean(v)) if len(v) else np.nan})

    process_rows = []
    if particle == "gamma":
        for proc, sel in {"Pair": pair, "Compton": ~pair}.items():
            for step in ["Containment", "HitThreshold", "Clustering", "Total"]:
                r = ratios[step][sel & (has_cluster if step in ["Clustering", "Total"] else True)]
                h = np.histogram(r[np.isfinite(r)], bins=r_edges)[0].astype(float)
                process_rows.append({"Config": config, "Name": name, "Process": proc, "Step": step,
                                     "Share": float(sel.mean()), "Bins": r_centers, "Counts": h,
                                     "Density": h / max(h.sum() * np.diff(r_edges)[0], 1e-300), **metrics(r)})

    events = pd.DataFrame({"Config": config, "Name": name, "Particle": particle, "TrueEnergy": k, "EDep": e_dep,
                           "EHits": e_hits, "ECluster": e_clu, "ECharge": e_charge, "LeadingNHits": nhits,
                           "PrimaryClusters": n_prim, "Pair": pair, "RMSRadius": rms, "DepositGroups": groups,
                           "OutsideLargestGroup": outside, "CascadeSum": cascade_sum, "CascadeMultiplicity": cascade_mult,
                           **{f"Ratio{s}": r for s, r in ratios.items()}})
    subfolder = "gun"
    for rows, filename in [(ladder_rows, "Gun_Ladder"), (topo_rows, "Gun_Topology"), (process_rows, "Gun_Process")]:
        save_df(pd.DataFrame(rows), data_path, config, name, subfolder=subfolder, filename=filename, rm=args.rewrite, debug=args.debug)
    save_df(events, data_path, config, name, subfolder=subfolder, filename="Gun_Events", rm=args.rewrite, debug=args.debug)
    summary = {"config": config, "name": name, "particle": particle, "events": int(len(k)),
               "with_cluster": float(has_cluster.mean()), "primary_clusters_per_event": float(np.mean(n_prim[has_cluster])),
               "pair_fraction": float(pair.mean()),
               "capture_fraction": float(np.mean(cascade_mult > 0)),
               "cascade_sum_percentiles": np.nanpercentile(np.where(cascade_mult > 0, cascade_sum, np.nan), [1, 16, 50, 84, 99]).tolist() if np.any(cascade_mult > 0) else None,
               "cascade_sum_above_q": float(np.mean(cascade_sum[cascade_mult > 0] > args.capture_q + 0.05)) if np.any(cascade_mult > 0) else None, "electron_tau_us": tau, "flat_gain_adc_per_mev": gain,
               "ladder_closure": float(closure), "steps": summary_steps,
               "topology_medians": {r["Quantity"]: r["Median"] for r in topo_rows},
               "process": {f"{r['Process']}_{r['Step']}": {kk: r[kk] for kk in ["Share", "PeakMean", "PeakSigma", "Median"]} for r in process_rows}}
    summary_dir = f"{data_path}/{config}/{name}/{subfolder}"
    os.makedirs(summary_dir, exist_ok=True)
    with open(f"{summary_dir}/{config}_{name}_Gun_Summary.json", "w") as fjson:
        json.dump(summary, fjson, indent=1, default=float)
    rprint(f"[green]Saved gun ladder outputs to {summary_dir}[/green]")
