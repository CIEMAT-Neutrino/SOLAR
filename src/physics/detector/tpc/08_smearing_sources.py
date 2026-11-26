"""
08_smearing_sources.py — Neutrino Energy Smearing Sources and De-excitation Gamma Tagging
=========================================================================================
Splits the near-threshold neutrino energy resolution (SelectedEnergy) into its sources,
swapping one truth ingredient at a time for its reco counterpart, all on the same
true-neutrino-energy axis (SignalParticleK) and with the same RMS (True - Reco) / True metric
as 03_energy_resolution.py (TrueMain primaries, NHits >= --nhits):

  Electron                  Energy(true t0) + (E_nu - ElectronK)     electron hit collection only
  Electron + t0             Energy(reco t0) + (E_nu - ElectronK)     + flash-matched drift time
  Electron + Gammas         SelectedEnergy(true t0)                  + adjacent-cluster (gamma) collection
  Electron + t0 + Gammas    SelectedEnergy(reco t0)                  full near-threshold reconstruction

The full reconstruction is also split by the MARLEY final state (NeutronK > 0 or not), and the
electron-only steps by the share of the electron energy left outside the primary cluster. The CC electron
steps keep the self pile-up of the interaction (de-excitation gamma charge merged into the primary cluster or
forming a primary cluster of its own); the free-electron rows (--reference, default electrongun_flat: electrons
of 2-30 MeV without nuclear de-excitation) give the electron-only resolution on the true electron energy:

  Free Electron [+ t0]               primary-cluster Energy (true [reco] t0) vs ElectronK
  Free Electron + Adjacent [+ t0]    + selected adjacent clusters (SelectedEnergy sum before the neutrino map)

Every row carries the RMS of (True - Reco) / True (the paper metric) and Sigma68, half the 16-84 % range of the
same distribution, a core width insensitive to the tails (split electrons, wrong flashes) that drive the RMS.

De-excitation gamma tagging (reco t0): adjacent clusters with signal purity > 0 are the gamma
signal, the SelectedEnergy selection (AdjClR < MIN_BKG_R or AdjClCharge > MAX_BKG_CHARGE) the tag.
Per true-neutrino-energy bin: tagging efficiency and purity (cluster counts and reco energy), and the
RMS of (GammaK - tagged energy) / GammaK for all tagged clusters and for the tagged signal only.

Outputs (output/data/TPC/resolution/sources/{config}/{name}/):
  {config}_{name}_Smearing_Sources.pkl   Source, Subset, Values, RMS, RMSError, Bias, Counts, Sigma68 and the
                                         same after a per-row response calibration (CalibratedRMS, CalibratedBias,
                                         CalibratedSigma68); ReportedRMS/ReportedSigma68 take the
                                         calibrated values for the free-electron rows and the plain ones
                                         for the neutrino rows (the convention of the paper plots)
  {config}_{name}_Gamma_Tagging.pkl      Variable, Weight, Values, Counts, CountsError
"""

import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *
from lib.cluster import _adjcl_slots, _selected_limit_arrays

data_path = f"{root}/output/data/TPC/resolution/sources"
os.makedirs(data_path, exist_ok=True)

parser = argparse.ArgumentParser(description="Neutrino energy smearing sources and gamma tagging")
parser.add_argument("--config", type=str, help="The configuration to load", default="hd_1x2x6_centralAPA")
parser.add_argument("--name", type=str, help="The name of the configuration", default="marley_final")
parser.add_argument("--nhits", type=int, default=3, help="Minimum primary-cluster NHits (paper: 3)")
parser.add_argument("--reference", type=str, default="electrongun_flat",
                    help="Free-electron sample (no nuclear de-excitation) for the electron-only rows; 'none' to skip")
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
args = parser.parse_args()
config, name = args.config, args.name
configs = {config: [name]}
workflow = "SMEARING"

# SMEARING preset plus the adjacent-cluster truth used by the electron cluster and the gamma tagging.
branches = get_workflow_branches(trees=["Config", "Truth", "Reco"], workflow=workflow)
branches["Reco"] = list(dict.fromkeys(branches["Reco"] + ["AdjClPur", "AdjClMainPDG"]))
run, output = load_multi(configs, branches=branches, debug=args.debug)

# Same drift-time options as 03_energy_resolution.py.
DRIFT = {
    "True": {"DEFAULT_ENERGY_TIME": "Time", "DEFAULT_ADJCL_ENERGY_TIME": "AdjClTime"},
    "Reco": {},
}
columns = {}
for label, params in DRIFT.items():
    this_run = compute_reco_workflow(run, configs, params=params, workflow=workflow, debug=args.debug, electron_energy=True)
    this_run, mask, output = compute_filtered_run(
        this_run, configs, params={("Reco", "TrueMain"): ("equal", True)},
        presets=[workflow], signal="marley" in name, debug=args.debug,
    )
    rprint(output)
    data = this_run["Reco"]
    keep = np.asarray(data["NHits"]) >= args.nhits
    columns[label] = {
        "Event": np.asarray(data["Event"])[keep],
        "Flag": np.asarray(data["Flag"])[keep],
        "Energy": np.asarray(data["Energy"], dtype=float)[keep],
        "SelectedEnergy": np.asarray(data["SelectedEnergy"], dtype=float)[keep],
    }
    if label == "Reco":
        info, this_params, output = get_param_dict(f"{root}/config/{config}/{config}", {}, output, debug=args.debug)
        idx = (np.arange(len(data["Event"])),)
        rlim, qlim = _selected_limit_arrays(this_run, config, info, [name], idx, this_params)
        slots = _adjcl_slots(this_run, idx)
        selected = ((np.asarray(data["AdjClR"]) < rlim[:, None]) | (np.asarray(data["AdjClCharge"]) > qlim[:, None])) & slots
        signal = slots & (np.asarray(data["AdjClPur"]) > 0)
        adj_energy = np.asarray(data["AdjClEnergy"], dtype=float)
        tag = {
            "NSignal": np.sum(signal, axis=1),
            "NSelected": np.sum(selected, axis=1),
            "NSelectedSignal": np.sum(selected & signal, axis=1),
            "ESignal": np.sum(adj_energy, where=signal, axis=1),
            "ESelected": np.sum(adj_energy, where=selected, axis=1),
            "ESelectedSignal": np.sum(adj_energy, where=selected & signal, axis=1),
        }
        columns[label].update({k: v[keep] for k, v in tag.items()})
        for branch in ["SignalParticleK", "ElectronK", "GammaK", "NeutronK", "ElectronEnergy"]:
            columns[label][branch] = np.asarray(data[branch], dtype=float)[keep]

# Both drift labels filter the same primaries; align them on (Flag, Event, Energy order) to be safe.
true_cols, reco_cols = columns["True"], columns["Reco"]
if len(true_cols["Event"]) != len(reco_cols["Event"]) or np.any(true_cols["Event"] != reco_cols["Event"]):
    raise SystemExit("True and Reco drift selections differ: cannot pair the smearing steps event by event")

E_nu = reco_cols["SignalParticleK"]
E_e = reco_cols["ElectronK"]
remainder = E_nu - E_e  # binding + de-excitation (+ neutron) energy, taken from truth
info, _, output = get_param_dict(f"{root}/config/{config}/{config}", {}, "", debug=args.debug)

estimators = {
    "Electron": true_cols["Energy"] + remainder,
    "Electron + t0": reco_cols["Energy"] + remainder,
    "Electron + Gammas": true_cols["SelectedEnergy"],
    "Electron + t0 + Gammas": reco_cols["SelectedEnergy"],
}
# Electron energy left outside the primary cluster (radiative losses, delta rays) in the ideal electron cluster.
outside = np.where(reco_cols["ElectronEnergy"] > 0, 1 - reco_cols["Energy"] / reco_cols["ElectronEnergy"], 0)
subsets = {
    "All": np.ones(len(E_nu), dtype=bool),
    "No Neutron": reco_cols["NeutronK"] <= 0,
    "Neutron": reco_cols["NeutronK"] > 0,
    "Contained Electron": outside < 0.05,
    "Split Electron": outside >= 0.05,
}


def binned_rms(truth, reco, sel, axis=None):
    """RMS of (truth - reco) / truth in bins of the true neutrino energy (or of `axis`)."""
    axis = truth if axis is None else axis
    rms, err, bias, counts, sigma68 = [], [], [], [], []
    for centre in reco_energy_centers:
        in_bin = sel & (axis > centre - reco_ebin / 2) & (axis < centre + reco_ebin / 2) & (truth > 0)
        n = int(np.sum(in_bin))
        counts.append(n)
        if n < 10:
            rms.append(np.nan), err.append(np.nan), bias.append(np.nan), sigma68.append(np.nan)
            continue
        rel = (truth[in_bin] - reco[in_bin]) / truth[in_bin]
        value = float(np.sqrt(np.mean(rel**2)))
        rms.append(value)
        err.append(value / np.sqrt(2 * n))
        bias.append(float(np.mean(rel)))
        # Core width, insensitive to the tails that dominate the RMS: half the 16-84 % range.
        q16, q84 = np.percentile(rel, [15.87, 84.13])
        sigma68.append(float((q84 - q16) / 2))
    return rms, err, bias, counts, sigma68


def calibrate(truth, reco, sel):
    """Calibrate `reco` on its own events: response R(E) = <reco>/truth from the bin means, solved per event
    as E_cal = reco / R(E_cal) (fixed point; R interpolated, constant outside the populated bins, so a flat
    response cannot diverge). Each row gets its own response, so the
    RMS measures the spread and not the mean offset left by the shared (CC) calibration.
    """
    centres, means = [], []
    for centre in reco_energy_centers:
        in_bin = sel & (truth > centre - reco_ebin / 2) & (truth < centre + reco_ebin / 2) & (truth > 0)
        if np.sum(in_bin) >= 10:
            centres.append(centre), means.append(np.mean(reco[in_bin]))
    ratio = np.array(means) / np.array(centres)
    calibrated = reco / np.interp(reco, centres, ratio)
    for _ in range(30):
        calibrated = reco / np.interp(calibrated, centres, ratio)
    return calibrated


def calibrated_columns(truth, reco, sel):
    # The response comes from all events: a subset (e.g. events with a neutron) cannot be identified in data,
    # so its own mean offset (the missing neutron energy) must stay in its resolution.
    rms, _, bias, _, sigma68 = binned_rms(truth, calibrate(truth, reco, np.ones(len(truth), dtype=bool)), sel)
    return {"CalibratedRMS": rms, "CalibratedBias": bias, "CalibratedSigma68": sigma68}


rows = []
for (source, reco), (subset, sel) in product(estimators.items(), subsets.items()):
    rms, err, bias, counts, sigma68 = binned_rms(E_nu, reco, sel)
    rows.append({
        "Geometry": info["GEOMETRY"], "Config": config, "Name": name, "#Hits": args.nhits,
        "Source": source, "Subset": subset, "Fraction": float(np.mean(sel)),
        "Values": reco_energy_centers, "RMS": rms, "RMSError": err, "Bias": bias, "Counts": counts, "Sigma68": sigma68,
        **calibrated_columns(E_nu, reco, sel),
        # Paper convention: neutrino-sample rows report the plain RMS / Sigma68.
        "ReportedRMS": rms, "ReportedSigma68": sigma68,
    })
# Free electrons: the electron-only resolution without the self pile-up of the CC interaction (de-excitation
# gamma charge in or around the primary cluster), on the true electron energy axis.
if args.reference.lower() != "none":
    ref_configs = {config: [args.reference]}
    ref_run, output = load_multi(ref_configs, preset=workflow, debug=args.debug)
    for label, params in [("", DRIFT["True"]), (" + t0", DRIFT["Reco"])]:
        this_run = compute_reco_workflow(ref_run, ref_configs, params=params, workflow=workflow, debug=args.debug)
        this_run, mask, output = compute_filtered_run(
            this_run, ref_configs, params={("Reco", "TrueMain"): ("equal", True)},
            presets=[workflow], signal=False, debug=args.debug,
        )
        data = this_run["Reco"]
        keep = np.asarray(data["NHits"]) >= args.nhits
        truth = np.asarray(data["ElectronK"], dtype=float)[keep]
        primary = np.asarray(data["Energy"], dtype=float)[keep]
        # Primary cluster alone, and with its selected adjacent clusters (the SelectedEnergy sum before the
        # neutrino-energy map), which recovers the electron energy radiated or split away from the primary cluster.
        estimators_free = {
            f"Free Electron{label}": primary,
            f"Free Electron + Adjacent{label}": primary + np.asarray(data["SelectedAdjClEnergy"], dtype=float)[keep],
        }
        for source, reco in estimators_free.items():
            rms, err, bias, counts, sigma68 = binned_rms(truth, reco, np.ones(len(truth), dtype=bool))
            rows.append({
                "Geometry": info["GEOMETRY"], "Config": config, "Name": name, "#Hits": args.nhits,
                "Source": source, "Subset": "All", "Fraction": 1.0, "Reference": args.reference,
                "Values": reco_energy_centers, "RMS": rms, "RMSError": err, "Bias": bias, "Counts": counts, "Sigma68": sigma68,
                **calibrated_columns(truth, reco, np.ones(len(truth), dtype=bool)),
            })
            # Paper convention: free-electron rows report the calibrated RMS / Sigma68.
            rows[-1]["ReportedRMS"], rows[-1]["ReportedSigma68"] = rows[-1]["CalibratedRMS"], rows[-1]["CalibratedSigma68"]
save_df(pd.DataFrame(rows), data_path, config, name, filename="Smearing_Sources", rm=args.rewrite, debug=args.debug)

# --- de-excitation gamma tagging, per true neutrino energy bin ---
tag_rows = []


def ratio(num, den):
    return (num / den if den > 0 else np.nan), (np.sqrt(num * max(den - num, 0) / den**3) if den > 0 else np.nan)


gamma_true = reco_cols["GammaK"]
for variable, weight, num_key, den_key in [
    ("Efficiency", "Clusters", "NSelectedSignal", "NSignal"),
    ("Purity", "Clusters", "NSelectedSignal", "NSelected"),
    ("Efficiency", "Energy", "ESelectedSignal", "ESignal"),
    ("Purity", "Energy", "ESelectedSignal", "ESelected"),
]:
    values, errors = [], []
    for centre in reco_energy_centers:
        in_bin = (E_nu > centre - reco_ebin / 2) & (E_nu < centre + reco_ebin / 2)
        value, error = ratio(float(np.sum(reco_cols[num_key][in_bin])), float(np.sum(reco_cols[den_key][in_bin])))
        values.append(100 * value), errors.append(100 * error)
    tag_rows.append({"Geometry": info["GEOMETRY"], "Config": config, "Name": name, "Variable": variable,
                     "Weight": weight, "Values": reco_energy_centers, "Counts": values, "CountsError": errors})

for variable, reco_gamma in [("Resolution", reco_cols["ESelected"]), ("Resolution Signal Only", reco_cols["ESelectedSignal"])]:
    rms, err, bias, counts, sigma68 = binned_rms(gamma_true, reco_gamma, gamma_true > 0, axis=E_nu)
    tag_rows.append({"Geometry": info["GEOMETRY"], "Config": config, "Name": name, "Variable": variable,
                     "Weight": "Energy", "Values": reco_energy_centers, "Counts": rms, "CountsError": err, "Bias": bias,
                     "Entries": counts, "Sigma68": sigma68})
save_df(pd.DataFrame(tag_rows), data_path, config, name, filename="Gamma_Tagging", rm=args.rewrite, debug=args.debug)
rprint(f"[green]Smearing sources and gamma tagging saved for {config} {name}[/green]")
