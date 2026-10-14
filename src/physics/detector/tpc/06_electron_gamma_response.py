"""
06_electron_gamma_response.py — Electron vs Gamma Primary-Cluster Response at a Fixed True Energy
=================================================================================================
Explains why the CC-electron slice of 04_smearing_chain.py (true ElectronK in a narrow range) is
wider than the external-gamma slice at the same true energy. Works on the charge-response step of
the chain (own-generator charge, true drift time, flat gain G; no flash, no NHits conversion):

  E_charge = Q * Purity * exp(|t_true| / tau) / G

Every cluster enters with its own-generator (generator 1, the sample's own) charge only: Q * Purity for the
primary cluster, AdjClCharge * AdjClGenPur for an adjacent cluster whose main generator is 1 (adjacent clusters
led by another generator are left out). Per event three references are built:
  Primary            the primary cluster with the highest E_charge in the event
  PrimaryAdjacent    that primary cluster plus its own-generator adjacent clusters
  Cheated            all own-generator charge of the event: every primary cluster plus its own-generator
                     adjacent clusters

and splits it by what produced the cluster and by its place in the event:

  marley  SignalFrac (charge fraction per MARLEY particle family: electron, gamma, neutron, rest)
          -> electron only / electron + merged de-excitation gamma / de-excitation gamma only / other
  gamma   leading (largest-charge) primary cluster of the event / secondary fragments

Exports (same folder as 04_smearing_chain.py, TPC/smearing/<config>/<name>/true<lo>-<hi>_unweighted):
  Response_Composition   E_charge spectrum per class, normalised to all primary clusters of the slice
  Response_Reference     E_charge / E_ref for matched references (electron charge vs ElectronK,
                         cluster vs ElectronK + prompt de-excitation gamma energy, leading cluster)
  Response_Topology      NHits (collection, induction), charge per hit, clusters per event
  Response_FlashClass    core peak/width of the lifetime and calibration steps per flash category
  Response_Depth         gamma response vs distance between the gamma start and the cluster
  Response_Clusters      one row per primary cluster (unbinned, for 2D plots)
  <config>_<name>_Response_Summary.json
No analysis selection, fiducial volume or weights: this is a property of the reconstruction alone.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from scipy.optimize import curve_fit

from lib import *

data_path = f"{root}/output/data/TPC/smearing"
os.makedirs(data_path, exist_ok=True)

parser = argparse.ArgumentParser(description="Electron vs gamma primary-cluster response at a fixed true energy")
parser.add_argument("--config", type=str, help="The configuration to load", default="hd_1x2x6_centralAPA")
parser.add_argument("--names", nargs="+", type=str, default=["marley_final", "gamma_final"],
                    help="Samples to load (marley*, gamma* and neutron* families)")
parser.add_argument("--true_range", nargs=2, type=float, default=[9.5, 10.5], metavar=("LO", "HI"),
                    help="True-energy slice (ElectronK for marley, SignalParticleK for gamma)")
parser.add_argument("--neutron_max_k", type=float, default=1.4,
                    help="Neutron samples: keep events whose neutron starts below this kinetic energy (MeV). Below the "
                         "1.46 MeV first excited state of Ar-40 there is no inelastic gamma, so every gamma cluster "
                         "belongs to a capture cascade")
parser.add_argument("--capture_q", type=float, default=6.099,
                    help="Neutron samples: true energy of the capture cascade, Ar-40(n,gamma)Ar-41 (MeV)")
parser.add_argument("--subfolder", type=str, default=None,
                    help="Output subfolder for every sample (default: true<lo>-<hi>_unweighted, or "
                         "capture_neutronK0-<max>_unweighted for neutrons). A common name lets the plots overlay "
                         "electrons, gammas and capture cascades at the cascade energy")
parser.add_argument("--bin_width", type=float, default=0.5, help="Energy histogram bin width (MeV)")
parser.add_argument("--max_energy", type=float, default=20.0, help="Energy histogram upper edge (MeV)")
parser.add_argument("--gamma_fraction_cut", type=float, default=0.05,
                    help="SignalFrac gamma share below which a marley cluster counts as electron only")
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()
config = args.config
subfolder = f"true{args.true_range[0]:g}-{args.true_range[1]:g}_unweighted"
capture_subfolder = f"capture_neutronK0-{args.neutron_max_k:g}_unweighted"

SAMPLE_TRUTH = {"marley": "ElectronK", "gamma": "SignalParticleK", "neutron": "CaptureQ"}
MARLEY_CLASSES = {
    "ElectronOnly": "Electron only",
    "ElectronGamma": "Electron + de-excitation gamma",
    "GammaOnly": "De-excitation gamma only",
    "Other": "Other",
}
GAMMA_CLASSES = {"Leading": "Leading cluster", "Secondary": "Secondary cluster"}
FLASH_LABELS = {"NoFlash": "No flash", "WrongFlash": "Wrong flash", "CorrectFlash": "Correct flash"}

info = json.load(open(f"{root}/config/{config}/{config}_config.json"))
edges = np.arange(0, args.max_energy + args.bin_width / 2, args.bin_width)
centers = 0.5 * (edges[1:] + edges[:-1])
r_edges = np.arange(0.0, 3.0 + 1e-9, 0.02)  # response histograms up to 3 (neutron-capture cheated tail, 2026-10-13)
r_centers = 0.5 * (r_edges[1:] + r_edges[:-1])


def _calib_file(name: str, pattern: str) -> dict:
    """Per-sample calibration file with the marley_official fallback of lib/cluster.py."""
    for sample in [name, "marley_official"]:
        path = f"{root}/config/{config}/{sample}/{config}_calib/{config}_{pattern.format(sample=sample)}"
        if os.path.exists(path):
            return json.load(open(path))
    raise FileNotFoundError(pattern)


def ratio_peak(ratio):
    """Full-absorption peak of E/E_ref: iterated Gaussian fit around the mode above 0.7 (as 04)."""
    ratio = ratio[np.isfinite(ratio) & (ratio > 0.5) & (ratio < 1.5)]
    if len(ratio) < 200:
        return np.nan, np.nan
    counts, bin_edges = np.histogram(ratio, bins=200, range=(0.5, 1.5))
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    upper = bin_centers > 0.7
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


def ratio_metrics(ratio):
    finite = ratio[np.isfinite(ratio)]
    mu, sigma = ratio_peak(finite)
    return {"Clusters": int(len(finite)), "PeakMean": mu, "PeakSigma": sigma,
            "Median": float(np.median(finite)) if len(finite) else np.nan,
            "P16": float(np.percentile(finite, 16)) if len(finite) else np.nan,
            "P84": float(np.percentile(finite, 84)) if len(finite) else np.nan,
            "FracBelow0p5": float(np.mean(finite < 0.5)) if len(finite) else np.nan,
            "FracAbove1p05": float(np.mean(finite > 1.05)) if len(finite) else np.nan,
            "FracAbove0p9": float(np.mean(finite > 0.9)) if len(finite) else np.nan}


def density(values, bins, norm):
    counts = np.histogram(values, bins=bins)[0].astype(float)
    return counts, counts / max(norm * np.diff(bins)[0], 1e-300)


for name in args.names:
    family = name.split("_")[0]
    if family not in SAMPLE_TRUTH:
        raise SystemExit(f"{name}: only marley*, gamma* and neutron* samples are supported")
    sample_subfolder = args.subfolder or (capture_subfolder if family == "neutron" else subfolder)
    configs = {config: [name]}
    run, output = load_multi(configs, preset="SIGNIFICANCE", branches={"Config": ["Geometry"]}, debug=args.debug)
    workflow_params = ({"PARTICLE_TYPE": "signal", "PARTICLE_WEIGHTING": "volume"} if family == "marley"
                       else {"PARTICLE_TYPE": "background", "PARTICLE_WEIGHTING": "histogram"})
    run = compute_reco_workflow(run, configs, params=workflow_params, workflow="SIGNIFICANCE", rm_branches=False, debug=args.debug)
    reco = run["Reco"]

    # Branches the SIGNIFICANCE preset does not carry, from the same npy directory (same order, see 04)
    npy_dir = f'{info["PATH"]}/data/{info["GEOMETRY"]}/{info["VERSION"]}/{info["NAME"]}{name}/Reco'
    side = ["Purity", "MainTime", "MatchedOpFlashCorrectly", "SignalFrac", "Ind0NHits", "Ind1NHits", "Event", "Flag",
            "SignalParticleX", "SignalParticleY", "SignalParticleZ", "SignalParticleK", "RecoY", "RecoZ", "TPC",
            "AdjClCharge", "AdjClGen", "AdjClGenPur", "AdjClTime"]
    if family == "marley":
        side += ["TSignalPDG", "TSignalK", "TSignalMother"]
    npy = {}
    for branch in side:
        values = np.load(f"{npy_dir}/{branch}.npy", allow_pickle=True)
        if len(values) != len(reco["Charge"]):
            raise SystemExit(f"{branch}: {len(values)} entries vs {len(reco['Charge'])} loaded clusters")
        npy[branch] = values
    if not np.allclose(np.load(f"{npy_dir}/Charge.npy"), reco["Charge"]):
        raise SystemExit("Charge read from npy does not match the loaded run: cluster order differs")

    corr_info = _calib_file(name, "electroncharge_correction.json")
    tau, gain = corr_info["ELECTRON_TAU"], corr_info["CHARGE_AMP"]
    charge = np.asarray(reco["Charge"], dtype=float)
    purity = np.clip(np.asarray(npy["Purity"], dtype=float), 0, 1)
    true_time = np.asarray(reco["Time"], dtype=float) - 1e-3 * np.asarray(npy["MainTime"], dtype=float)
    true_lifetime = np.exp(np.abs(true_time) / tau)
    e_charge = charge * purity * true_lifetime / gain
    # Own-generator adjacent clusters (generator 1), each with its own true drift time and pure charge
    adj_charge = np.asarray(npy["AdjClCharge"], dtype=float)
    adj_own = (np.asarray(npy["AdjClGen"]) == 1) & (adj_charge > 0)
    adj_true_time = np.asarray(npy["AdjClTime"], dtype=float) - 1e-3 * np.asarray(npy["MainTime"], dtype=float)[:, None]
    adj_energy = np.sum(np.where(adj_own, adj_charge * np.clip(np.asarray(npy["AdjClGenPur"], dtype=float), 0, 1)
                                 * np.exp(np.abs(adj_true_time) / tau), 0.0), axis=1) / gain
    e_primary_adj = e_charge + adj_energy
    e_lifetime = charge * np.asarray(reco["Correction"], dtype=float) / gain
    e_calibration = np.asarray(reco["Energy"], dtype=float)
    # Neutron samples: the reference is the capture-cascade energy, not the neutron kinetic energy
    true_energy = (np.full(len(reco["Charge"]), args.capture_q) if family == "neutron"
                   else np.asarray(reco[SAMPLE_TRUTH[family]], dtype=float))
    frac = np.stack([np.asarray(x, dtype=float) for x in npy["SignalFrac"]])
    nhits = np.asarray(reco["NHits"], dtype=float)

    # Prompt MARLEY products (Mother == 0): electron KE closure against ElectronK, de-excitation gamma energy
    if family == "marley":
        triples = list(zip(npy["TSignalPDG"], npy["TSignalK"], npy["TSignalMother"]))
        electron_k = np.array([max([k for p, k, m in zip(*t) if p == 11 and m == 0] or [np.nan]) for t in triples])
        gamma_k = np.array([sum(k for p, k, m in zip(*t) if p == 22 and m == 0) for t in triples])
        ok = np.isfinite(electron_k) & (true_energy > 0)
        closure = float(np.nanmax(np.abs(electron_k[ok] - true_energy[ok]))) if ok.any() else np.nan
        rprint(f"[cyan][INFO][/cyan] ElectronK closure against the prompt TSignal electron: max |Δ| = {closure:.2e} MeV")
    else:
        gamma_k = np.zeros(len(charge))
        closure = np.nan

    if family == "neutron":
        mask = np.asarray(npy["SignalParticleK"], dtype=float) < args.neutron_max_k
    else:
        mask = (true_energy >= args.true_range[0]) & (true_energy < args.true_range[1])
    if not mask.any():
        raise SystemExit(f"{name}: no clusters in the true-energy slice")

    # Event key: file flag, event number and true energy (events repeat across merged files)
    keys = pd.Series(list(zip(npy["Flag"], npy["Event"], np.round(np.asarray(npy["SignalParticleK"], dtype=float), 6),
                              np.round(np.asarray(npy["SignalParticleX"], dtype=float), 3))))
    sel = np.flatnonzero(mask)
    sel_keys = keys.iloc[sel].astype(str).to_numpy()
    df_ev = pd.DataFrame({"idx": sel, "key": sel_keys, "q": e_charge[sel], "qa": e_primary_adj[sel]})
    df_ev["rank"] = df_ev.groupby("key")["q"].rank(ascending=False, method="first")
    df_ev["n"] = df_ev.groupby("key")["q"].transform("size")
    leading = np.zeros(len(charge), dtype=bool)
    leading[df_ev.loc[df_ev["rank"] == 1, "idx"].to_numpy()] = True
    clusters_per_event = np.zeros(len(charge))
    clusters_per_event[df_ev["idx"].to_numpy()] = df_ev["n"].to_numpy()
    # Sum of the charge-response energy of all primary clusters of the event, stored on the leading cluster
    event_sum = np.full(len(charge), np.nan)
    df_ev["sum"] = df_ev.groupby("key")["q"].transform("sum")
    event_sum[df_ev.loc[df_ev["rank"] == 1, "idx"].to_numpy()] = df_ev.loc[df_ev["rank"] == 1, "sum"].to_numpy()
    # Cheated: every primary cluster of the event with its own-generator adjacent clusters, stored on the leading one
    cheated = np.full(len(charge), np.nan)
    df_ev["sum_adj"] = df_ev.groupby("key")["qa"].transform("sum")
    cheated[df_ev.loc[df_ev["rank"] == 1, "idx"].to_numpy()] = df_ev.loc[df_ev["rank"] == 1, "sum_adj"].to_numpy()

    if family == "marley":
        f_e, f_g = frac[:, 0], frac[:, 1]
        classes = {
            "ElectronOnly": (f_e >= 0.5) & (f_g < args.gamma_fraction_cut),
            "ElectronGamma": (f_e >= 0.5) & (f_g >= args.gamma_fraction_cut),
            "GammaOnly": (f_e < 0.5) & (f_g >= 0.5),
        }
        classes["Other"] = ~(classes["ElectronOnly"] | classes["ElectronGamma"] | classes["GammaOnly"])
        class_labels = MARLEY_CLASSES
    else:
        classes = {"Leading": leading.copy(), "Secondary": ~leading}
        class_labels = GAMMA_CLASSES

    n_slice = int(mask.sum())
    n_events = int(df_ev["key"].nunique())

    # Where do the separate fragments sit? SolarNuAna only calls a cluster adjacent (and so not
    # primary) when it lies within AdjClusterRad on the same drift side (TPC parity) as a larger one.
    side_of = np.asarray(npy["TPC"], dtype=int) % 2
    df_ev["side"] = side_of[sel]
    df_ev["cls"] = ""
    for cls, sel_cls in classes.items():
        df_ev.loc[sel_cls[sel], "cls"] = cls
    if family == "marley":
        df_ev["electron"] = df_ev["cls"].isin(["ElectronOnly", "ElectronGamma"])
        ref = df_ev[df_ev["electron"]].groupby("key")["side"].agg(lambda x: set(x))
    else:
        ref = df_ev[df_ev["rank"] == 1].groupby("key")["side"].agg(lambda x: set(x))
    def placement(row, own_is_ref):
        sides = ref.get(row["key"], set())
        if own_is_ref:  # the reference cluster itself: is there another reference-type primary in the event?
            return "Alone" if row["n"] == 1 else "WithOthers"
        if not sides:
            return "NoReference"
        return "SameSide" if row["side"] in sides else "OtherSide"
    if family == "marley":
        frag = df_ev[df_ev["cls"] == "GammaOnly"]
        partial = df_ev[(df_ev["cls"] == "ElectronOnly") & (df_ev["q"] < 5)]
        frag_place = frag.apply(lambda r: placement(r, False), axis=1).value_counts(normalize=True).to_dict() if len(frag) else {}
        # partial electron clusters: does the event hold another electron primary on the other side (APA/cathode split)?
        el = df_ev[df_ev["electron"]]
        n_el_sides = el.groupby("key")["side"].nunique()
        partial_split = float(np.mean([n_el_sides.get(k, 1) > 1 for k in partial["key"]])) if len(partial) else np.nan
    else:
        frag = df_ev[df_ev["rank"] > 1]
        frag_place = frag.apply(lambda r: placement(r, False), axis=1).value_counts(normalize=True).to_dict() if len(frag) else {}
        partial_split = np.nan
    rprint(f"  Fragment primaries ({'de-excitation gamma only' if family == 'marley' else 'secondary'}) relative to the "
           f"{'electron' if family == 'marley' else 'leading'} cluster: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in frag_place.items()))
    if family == "marley":
        rprint(f"  Electron-only clusters below 5 MeV with another electron primary on the other drift side: {100 * partial_split:.1f}%")
    rprint(f"\n[bold]{config} {name}[/bold] ({sample_subfolder}): {n_slice} primary clusters in {n_events} events "
           f"({n_slice / n_events:.2f} per event)")

    # ── Composition of the charge-step spectrum ──────────────────────────────────
    comp_rows = []
    for cls, sel_cls in classes.items():
        m = mask & sel_cls
        counts, dens = density(e_charge[m], edges, n_slice)
        comp_rows.append({"Config": config, "Name": name, "Class": cls, "Label": class_labels[cls],
                          "Bins": centers, "Counts": counts, "Density": dens, "Fraction": float(m.sum() / n_slice)})
        rprint(f"  {class_labels[cls]:<32} {100 * m.sum() / n_slice:5.1f}% of clusters, "
               f"{100 * np.mean(e_charge[m] < 5):5.1f}% below 5 MeV, {100 * np.mean(e_charge[m] > args.true_range[1]):5.1f}% above {args.true_range[1]:g} MeV")
    counts, dens = density(e_charge[mask], edges, n_slice)
    comp_rows.append({"Config": config, "Name": name, "Class": "All", "Label": "All primary clusters",
                      "Bins": centers, "Counts": counts, "Density": dens, "Fraction": 1.0})
    counts, dens = density(e_charge[mask & leading], edges, n_slice)
    comp_rows.append({"Config": config, "Name": name, "Class": "LeadingAll", "Label": "Leading cluster",
                      "Bins": centers, "Counts": counts, "Density": dens, "Fraction": float((mask & leading).sum() / n_slice)})

    # ── Response against matched references ───────────────────────────────────────
    references = {"Cluster": (e_charge, true_energy, mask), "Leading": (e_charge, true_energy, mask & leading),
                  "EventSum": (event_sum, true_energy, mask & leading),
                  "Primary": (e_charge, true_energy, mask & leading),
                  "PrimaryAdjacent": (e_primary_adj, true_energy, mask & leading),
                  "Cheated": (cheated, true_energy, mask & leading)}
    if family == "marley":
        references["ElectronCharge"] = (charge * np.clip(frac[:, 0], 0, 1) * true_lifetime / gain, true_energy,
                                        mask & (frac[:, 0] >= 0.5))
        references["ElectronOnly"] = (e_charge, true_energy, mask & classes["ElectronOnly"])
        references["LeadingVisible"] = (e_charge, true_energy + gamma_k, mask & leading)
    # Like-for-like comparison across samples: the clusters that carry only the particle of the slice
    # (electron-only clusters for marley, the leading cluster of each event for gamma)
    references["Matched"] = references["ElectronOnly"] if family == "marley" else references["Leading"]
    ref_labels = {"Cluster": "All primary clusters / E_true", "Leading": "Leading cluster / E_true",
                  "ElectronCharge": "Electron charge / ElectronK", "ElectronOnly": "Electron-only clusters / ElectronK",
                  "LeadingVisible": "Leading cluster / (ElectronK + de-excitation gammas)",
                  "Matched": "Single-particle clusters / E_true", "EventSum": "Sum of primary clusters / E_true",
                  "Primary": "Primary cluster / E_true",
                  "PrimaryAdjacent": "Primary + own-generator adjacent clusters / E_true",
                  "Cheated": "All own-generator charge of the event / E_true"}
    ref_rows, summary_refs = [], {}
    for ref, (num, den, m) in references.items():
        r = num[m] / den[m]
        h = np.histogram(r[np.isfinite(r)], bins=r_edges)[0].astype(float)
        metrics = ratio_metrics(r)
        summary_refs[ref] = metrics
        ref_rows.append({"Config": config, "Name": name, "Reference": ref, "Label": ref_labels[ref],
                         "Bins": r_centers, "Counts": h, "Density": h / max(h.sum() * np.diff(r_edges)[0], 1e-300), **metrics})
        rprint(f"  {ref_labels[ref]:<52} peak {metrics['PeakMean']:.3f} σ {metrics['PeakSigma']:.3f}  "
               f"median {metrics['Median']:.3f}  <0.5 {100 * metrics['FracBelow0p5']:4.1f}%  >1.05 {100 * metrics['FracAbove1p05']:4.1f}%")

    # ── Topology ───────────────────────────────────────────────────────────────────
    topo_rows = []
    topo_vars = {
        "NHits": (nhits, np.arange(0.5, 30.5, 1.0)),
        "Ind0NHits": (np.asarray(npy["Ind0NHits"], dtype=float), np.arange(-0.5, 30.5, 1.0)),
        "Ind1NHits": (np.asarray(npy["Ind1NHits"], dtype=float), np.arange(-0.5, 30.5, 1.0)),
        "ChargePerHit": (e_charge / np.maximum(nhits, 1), np.arange(0, 6.0001, 0.1)),
        "ClustersPerEvent": (clusters_per_event, np.arange(0.5, 8.5, 1.0)),
    }
    for group, m in {"All": mask, "Leading": mask & leading}.items():
        for var, (values, bins) in topo_vars.items():
            if var == "ClustersPerEvent" and group == "All":
                continue
            h = np.histogram(values[m], bins=bins)[0].astype(float)
            topo_rows.append({"Config": config, "Name": name, "Group": group, "Quantity": var,
                              "Bins": 0.5 * (bins[1:] + bins[:-1]), "Counts": h,
                              "Density": h / max(h.sum(), 1e-300), "Median": float(np.median(values[m]))})
    rprint(f"  Leading cluster: NHits median {np.median(nhits[mask & leading]):.0f}, "
           f"E_charge per hit median {np.median((e_charge / np.maximum(nhits, 1))[mask & leading]):.2f} MeV")

    # ── Flash category: core of the lifetime and calibration steps ────────────────
    no_flash = np.asarray(reco["MatchedOpFlashPE"]) <= 0
    correct_flash = np.asarray(npy["MatchedOpFlashCorrectly"]) == 1
    flash_classes = {"NoFlash": no_flash, "WrongFlash": ~no_flash & ~correct_flash, "CorrectFlash": ~no_flash & correct_flash}
    flash_rows = []
    for cls, sel_cls in flash_classes.items():
        for step, values in {"Charge": e_charge, "Lifetime": e_lifetime, "Calibration": e_calibration}.items():
            m = mask & sel_cls
            metrics = ratio_metrics(values[m] / true_energy[m])
            flash_rows.append({"Config": config, "Name": name, "Flash": cls, "Label": FLASH_LABELS[cls], "Step": step,
                               "Share": float(m.sum() / n_slice), **metrics})

    # ── Gamma: response vs distance between the gamma start and the cluster ───────
    depth_rows = []
    if family == "gamma":
        dist = np.sqrt((np.asarray(npy["RecoY"], dtype=float) - np.asarray(npy["SignalParticleY"], dtype=float)) ** 2
                       + (np.asarray(npy["RecoZ"], dtype=float) - np.asarray(npy["SignalParticleZ"], dtype=float)) ** 2)
        d_edges = np.array([0, 25, 50, 100, 200, 400, 1e4])
        for lo, hi in zip(d_edges[:-1], d_edges[1:]):
            m = mask & leading & (dist >= lo) & (dist < hi)
            if m.sum() < 50:
                continue
            metrics = ratio_metrics(e_charge[m] / true_energy[m])
            depth_rows.append({"Config": config, "Name": name, "DistanceLo": lo, "DistanceHi": hi,
                               "Distance": 0.5 * (lo + min(hi, 800)), "Share": float(m.sum() / (mask & leading).sum()), **metrics})
    else:
        dist = np.full(len(charge), np.nan)

    # ── Unbinned per-cluster table ──────────────────────────────────────────────
    cluster_class = np.full(len(charge), "", dtype=object)
    for cls, sel_cls in classes.items():
        cluster_class[sel_cls] = cls
    table = pd.DataFrame({
        "Config": config, "Name": name, "Class": cluster_class[mask], "Leading": leading[mask],
        "TrueEnergy": true_energy[mask], "GammaK": gamma_k[mask], "ChargeEnergy": e_charge[mask],
        "Response": e_charge[mask] / true_energy[mask], "LifetimeEnergy": e_lifetime[mask],
        "CalibrationEnergy": e_calibration[mask], "NHits": nhits[mask],
        "Ind0NHits": np.asarray(npy["Ind0NHits"], dtype=float)[mask], "Ind1NHits": np.asarray(npy["Ind1NHits"], dtype=float)[mask],
        "ChargePerHit": (e_charge / np.maximum(nhits, 1))[mask], "FracElectron": frac[mask, 0], "FracGamma": frac[mask, 1],
        "ClustersPerEvent": clusters_per_event[mask], "StartDistance": dist[mask],
        "Flash": np.select([no_flash[mask], correct_flash[mask]], ["NoFlash", "CorrectFlash"], "WrongFlash"),
    })

    for rows, filename in [(comp_rows, "Response_Composition"), (ref_rows, "Response_Reference"),
                           (topo_rows, "Response_Topology"), (flash_rows, "Response_FlashClass"),
                           (depth_rows, "Response_Depth")]:
        save_df(pd.DataFrame(rows), data_path, config, name, subfolder=sample_subfolder, filename=filename,
                rm=args.rewrite, debug=args.debug)
    save_df(table, data_path, config, name, subfolder=sample_subfolder, filename="Response_Clusters", rm=args.rewrite, debug=args.debug)

    summary = {
        "config": config, "name": name, "sample": sample_subfolder, "truth_variable": SAMPLE_TRUTH[family],
        "electron_k_closure_mev": closure, "clusters": n_slice, "events": n_events,
        "clusters_per_event": n_slice / n_events, "electron_tau_us": tau, "flat_gain_adc_per_mev": gain,
        "class_fractions": {cls: float((mask & m).sum() / n_slice) for cls, m in classes.items()},
        "class_below5_fractions": {cls: float(np.mean(e_charge[mask & m] < 5)) if (mask & m).any() else np.nan for cls, m in classes.items()},
        "class_above_hi_fractions": {cls: float(np.mean(e_charge[mask & m] > args.true_range[1])) if (mask & m).any() else np.nan for cls, m in classes.items()},
        "share_below5_from_class": {cls: float(np.sum(mask & m & (e_charge < 5)) / max(np.sum(mask & (e_charge < 5)), 1)) for cls, m in classes.items()},
        "share_above_hi_from_class": {cls: float(np.sum(mask & m & (e_charge > args.true_range[1])) / max(np.sum(mask & (e_charge > args.true_range[1])), 1)) for cls, m in classes.items()},
        "references": summary_refs,
        "fragment_placement": frag_place, "partial_electron_split_fraction": partial_split,
        "prompt_gamma_k_mean_mev": float(np.mean(gamma_k[mask & leading])) if family == "marley" else 0.0,
        "prompt_gamma_fraction": float(np.mean(gamma_k[mask & leading] > 0)) if family == "marley" else 0.0,
        "leading_nhits_median": float(np.median(nhits[mask & leading])),
        "adjacent_own_energy_mean_mev": float(np.mean(adj_energy[mask & leading])),
        "primary_clusters_per_event": n_slice / n_events,
        "leading_charge_per_hit_median_mev": float(np.median((e_charge / np.maximum(nhits, 1))[mask & leading])),
        "flash": {f"{r['Flash']}_{r['Step']}": {k: r[k] for k in ["Share", "PeakMean", "PeakSigma", "Median"]} for r in flash_rows},
        "depth": [{k: r[k] for k in ["DistanceLo", "DistanceHi", "Share", "PeakMean", "PeakSigma", "Median"]} for r in depth_rows],
    }
    summary_dir = f"{data_path}/{config}/{name}/{sample_subfolder}"
    os.makedirs(summary_dir, exist_ok=True)
    with open(f"{summary_dir}/{config}_{name}_Response_Summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=float)
    rprint(f"[green]Saved electron/gamma response outputs to {summary_dir}[/green]")
