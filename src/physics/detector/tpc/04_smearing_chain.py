"""
04_smearing_chain.py — Primary-Cluster Energy Smearing Decomposition
====================================================================
Follows every primary cluster of one sample (default: the external gamma
background) through each step that turns the true particle energy into the
reconstructed primary-cluster energy, and measures what each step does to the
spectrum -- in particular which step pushes events above the sample's true
endpoint (14 MeV for gammas). No analysis selection, fiducial volume, MC filter
or smoothing is applied: this is a property of the reconstruction alone.

Electron-scale ladder (MeV); each step swaps one truth ingredient for its reco
counterpart, in the order the reconstruction applies them (lib/cluster.py):

  TrueEnergy   E_true                                    energy of the particle behind the cluster
                                                         (SAMPLE_TRUTH: ElectronK / SignalParticleK / MainK)
  Charge       Q * Purity * exp(|t_true| / tau) / G      own-generator charge, true drift time,
                                                         flat gain G = CHARGE_AMP (containment
                                                         + intrinsic charge response)
  PileUp       Q * exp(|t_true| / tau) / G               + charge from other generators
  Lifetime     Q * exp(|RecoDriftTime| / tau) / G        + flash-matched drift time
  Gain         Q * Correction / CF(NHits)                + NHits-dependent charge-per-MeV
  Calibration  (Gain - b(NHits)) / a(NHits) = Energy     + electron slope/intercept calibration
  Offset       Energy - OFFSET                           + discriminant-selected Q-value offset
                                                         (SolarEnergy only; ClusterEnergy carries
                                                         its offset in the Map intercept)
  Map          SolarEnergy / ClusterEnergy               + linear neutrino-energy calibration

t_true = Time - MainTime: the cluster time minus the creation time of its main particle
(the readout t0 is 0 for every sample, but radiological decays and neutron captures happen
anywhere in the window). Clusters without a matched flash are kept: their RecoX falls back to the
detector edge (lib/main.py), which fixes their lifetime correction. Every table
is split by flash category (no flash / wrong flash / correct flash).

The one-at-a-time counterfactuals (switch a single effect back to truth, keep
the rest) are order-independent, unlike the step-by-step ladder.
"""

import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from scipy.optimize import curve_fit

from lib import *

save_path = f"{root}/output/images/TPC/smearing"
data_path = f"{root}/output/data/TPC/smearing"

for path in [save_path, data_path]:
    os.makedirs(path, exist_ok=True)

parser = argparse.ArgumentParser(
    description="Decompose the primary-cluster energy smearing chain step by step (no analysis selection)"
)
parser.add_argument("--config", type=str, help="The configuration to load", default="hd_1x2x6_centralAPA")
parser.add_argument("--name", type=str, help="The sample to load", default="gamma")
parser.add_argument("--energy", nargs="+", type=str, choices=["ClusterEnergy", "SolarEnergy"],
                    default=["SolarEnergy"],
                    help="Reconstructed energy whose map closes the chain (SolarEnergy is the analysis variable)")
parser.add_argument("--true_range", nargs=2, type=float, default=None, metavar=("LO", "HI"),
                    help="Keep only events with LO <= true energy < HI (e.g. 9.5 10.5 for a 10 MeV slice)")
parser.add_argument("--true_range_variable", type=str, default=None,
                    help="Branch that --true_range selects on (default: the sample truth, SAMPLE_TRUTH). E.g. "
                         "SignalParticleK for marley selects on the neutrino energy while the TrueEnergy step "
                         "stays the electron energy; the output folder is then true<branch><LO>-<HI>")
parser.add_argument("--weight", type=str, choices=["analysis", "truth"], default="analysis",
                    help="analysis: the weights 03_analysis.py exports (marley: oscillated day+night mean, the "
                         "'Solar' component of cutflow_plot.py); truth: unoscillated truth-flux weights")
parser.add_argument("--reference_folder", type=str, choices=["Nominal", "Reduced", "Truncated"], default="Truncated",
                    help="Where to find the 03_analysis.py --export_raw Ref pkls used for the cutflow_plot.py closure "
                         "check (their content does not depend on the folder)")
parser.add_argument("--unweighted", action=argparse.BooleanOptionalAction, default=False,
                    help="Count MC events instead of truth-flux weights (natural for a narrow --true_range)")
parser.add_argument("--bin_width", type=float, default=1.0, help="Histogram bin width (MeV)")
parser.add_argument("--exposure", type=float, default=None,
                    help="Livetime (years) of the cutflow-format pkl counts; default EVALUATION_EXPOSURE_YEARS DEFAULT (20 yr)")
parser.add_argument("--max_energy", type=float, default=30.0, help="Histogram upper edge (MeV)")
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)

args = parser.parse_args()
config = args.config
name = args.name
configs = {config: [name]}
subfolder = (
    (f"true{args.true_range_variable or ''}{args.true_range[0]:g}-{args.true_range[1]:g}" if args.true_range else "full")
    + ("_unweighted" if args.unweighted else "")
)
is_signal = "marley" in name

# Truth reference and purity per sample family. The radiological clusters never carry the signal
# particle (Purity == 0, SignalParticleK is a dummy MARLEY neutrino), so they use the energy of the
# cluster's main particle and the dominant-generator charge fraction instead. Neutron clusters come
# from capture gammas / recoils, so the neutron kinetic energy is not their reference either.
SAMPLE_TRUTH = {"marley": "ElectronK", "gamma": "SignalParticleK", "neutron": "MainK", "radiological": "MainK"}
SAMPLE_PURITY = {"radiological": "GenPurity"}
family = name.split("_")[0]

STEPS = ["TrueEnergy", "Charge", "PileUp", "Lifetime", "Gain", "Calibration"]
STEP_LABELS = {
    "Charge": "+ Containment & charge response",
    "PileUp": "+ Pile-up",
    "Lifetime": "+ Flash-matched lifetime correction",
    "Gain": "+ NHits charge-per-MeV",
    "Calibration": "+ Electron calibration",
}
# One categorical hue per step (fixed order of the reference palette): the steps are ordered, but
# neighbouring steps often overlap and shades of one hue could not be told apart.
STEP_COLORS = {"TrueEnergy": "#898781", "Charge": "#2a78d6", "PileUp": "#1baf7a", "Lifetime": "#e34948",
               "Gain": "#eda100", "Calibration": "#4a3aa7", "Offset": "#008300", "Map": "#eb6834"}
# Stage names of the cutflow_plot.py-format pkl. The last stage is the reconstructed energy the
# analysis uses, i.e. the cutflow_plot.py "Raw" stage (all clusters, no cut).
CUTFLOW_STAGES = {"TrueEnergy": "True energy", "Charge": "Charge response", "PileUp": "Pile-up",
                  "Lifetime": "Lifetime correction", "Gain": "NHits gain", "Calibration": "Electron calibration",
                  "Offset": "Discriminant offset"}

# Order-independent counterfactuals: electron-scale energy with a single effect switched back to truth
COUNTERFACTUALS = {
    "NoPileUp": "Signal-only charge (Q x Purity)",
    "TrueLifetime": "True drift time in the lifetime correction",
    "FlatGain": "Flat gain G, no NHits gain / electron calibration",
}

# ── Reconstruction (the SIGNIFICANCE workflow also provides the truth-flux weights) ──
run, output = load_multi(configs, preset="SIGNIFICANCE", branches={"Config": ["Geometry"]}, debug=args.debug)
# Same weighting parameters as 03_analysis.py (its default oscillation backend), so the final step
# reproduces the exported analysis spectra one to one.
workflow_params = (
    {"DEFAULT_SIGNAL_WEIGHT": ["truth", "osc"], "DEFAULT_SIGNAL_NADIR": ["mean", "day", "night"],
     "PARTICLE_TYPE": "signal", "PARTICLE_WEIGHTING": "volume", "OSCILLATION_BACKEND": "nufast"}
    if is_signal
    else {"PARTICLE_TYPE": "background", "PARTICLE_WEIGHTING": "histogram"}
)
weight_branch = "SignalParticleWeightOscMean" if is_signal and args.weight == "analysis" else "SignalParticleWeight"
reference_weight = "AnalysisWeightsSolar" if is_signal and args.weight == "analysis" else "AnalysisWeights"
run = compute_reco_workflow(run, configs, params=workflow_params, workflow="SIGNIFICANCE", rm_branches=False, debug=args.debug)
reco = run["Reco"]

info = json.load(open(f"{root}/config/{config}/{config}_config.json"))
this_params = json.load(open(f"{root}/config/{config}/{config}_params.json"))

# The SIGNIFICANCE preset does not carry the backtracked purity; read it from the same
# npy directory load_multi used (no event filtering happens there, so the order matches).
npy_dir = f'{info["PATH"]}/data/{info["GEOMETRY"]}/{info["VERSION"]}/{info["NAME"]}{name}/Reco'
# load_multi drops the MARLEY-generated clusters (Generator == 1) of radiological samples; mirror it
npy_keep = (np.load(f"{npy_dir}/Generator.npy") != 1) if "radiological" in name.lower() else slice(None)
for branch in ["Purity", "GenPurity", "MainTime", "MainK", "MatchedOpFlashCorrectly"]:
    values = np.load(f"{npy_dir}/{branch}.npy", allow_pickle=True)[npy_keep]
    if len(values) != len(reco["Event"]):
        raise SystemExit(f"{branch}: {len(values)} entries vs {len(reco['Event'])} loaded events")
    reco[branch] = values
if not np.allclose(np.load(f"{npy_dir}/Charge.npy")[npy_keep], reco["Charge"]):
    raise SystemExit("Charge read from npy does not match the loaded run: event order differs")


def _calib_file(pattern: str) -> dict:
    """Per-sample calibration file with the marley_official fallback of lib/cluster.py."""
    for sample in [name, "marley_official"]:
        path = f"{root}/config/{config}/{sample}/{config}_calib/{config}_{pattern.format(sample=sample)}"
        if os.path.exists(path):
            return json.load(open(path))
    raise FileNotFoundError(pattern)


corr_info = _calib_file("electroncharge_correction.json")
reco_info = _calib_file("{sample}_energy_calibration.json")
discriminant_info = _calib_file("discriminant_calibration.json")
tau, gain = corr_info["ELECTRON_TAU"], corr_info["CHARGE_AMP"]

charge = np.asarray(reco["Charge"], dtype=float)
purity = np.clip(np.asarray(reco[SAMPLE_PURITY.get(family, "Purity")], dtype=float), 0, 1)
true_time = np.asarray(reco["Time"], dtype=float) - 1e-3 * np.asarray(reco["MainTime"], dtype=float)  # MainTime in ns
true_lifetime = np.exp(np.abs(true_time) / tau)
reco_lifetime = np.asarray(reco["Correction"], dtype=float)
cf = np.asarray(reco["CorrectionFactor"], dtype=float)
slope = np.asarray(reco["EnergySlope"], dtype=float)
intercept = np.asarray(reco["EnergyIntercept"], dtype=float)
energy = np.asarray(reco["Energy"], dtype=float)
true_name = SAMPLE_TRUTH.get(family, "SignalParticleK")
# Clusters whose main particle was not backtracked (MainK = -1e6) have no truth reference: they stay
# in every reco spectrum (the analysis keeps them) but drop out of the truth-based metrics.
true_energy = np.asarray(reco[true_name], dtype=float)
true_energy = np.where(true_energy > 0, true_energy, np.nan)

electron_scale = {
    "TrueEnergy": true_energy,
    "Charge": charge * purity * true_lifetime / gain,
    "PileUp": charge * true_lifetime / gain,
    "Lifetime": charge * reco_lifetime / gain,
    "Gain": charge * reco_lifetime / cf,
    "Calibration": energy,
}
electron_counterfactual = {
    "NoPileUp": (charge * purity * reco_lifetime / cf - intercept) / slope,
    "TrueLifetime": (charge * true_lifetime / cf - intercept) / slope,
    "FlatGain": charge * reco_lifetime / gain,
}
closure = np.nanmax(np.abs((electron_scale["Gain"] - intercept) / slope - energy))
rprint(f"[cyan][INFO][/cyan] Electron-scale ladder closure |(Gain-b)/a - Energy| max = {closure:.2e} MeV")

# ── Sample: every primary cluster, optionally a true-energy slice ─────────────
mask = np.ones(len(energy), dtype=bool)
if np.isnan(true_energy).any():
    rprint(f"[yellow][WARNING][/yellow] {np.isnan(true_energy).sum()} clusters ({100 * np.isnan(true_energy).mean():.2f}%) "
           f"have no {true_name}: kept in the reco spectra, left out of the truth-based metrics")
if args.true_range:
    range_energy = (np.asarray(reco[args.true_range_variable], dtype=float)
                    if args.true_range_variable else true_energy)
    mask &= (range_energy >= args.true_range[0]) & (range_energy < args.true_range[1])
if not mask.any():
    raise SystemExit("No events in the requested true-energy range")
w = np.ones(mask.sum()) if args.unweighted else np.asarray(reco[weight_branch], dtype=float)[mask]
endpoint = args.true_range[1] if args.true_range else float(np.nanmax(true_energy[mask]))

no_flash = np.asarray(reco["MatchedOpFlashPE"])[mask] <= 0
correct_flash = np.asarray(reco["MatchedOpFlashCorrectly"])[mask] == 1
FLASH_CLASSES = {"NoFlash": no_flash, "WrongFlash": ~no_flash & ~correct_flash, "CorrectFlash": ~no_flash & correct_flash}
FLASH_LABELS = {"NoFlash": "No flash (edge default)", "WrongFlash": "Wrong flash", "CorrectFlash": "Correct flash"}

edges = np.arange(0, args.max_energy + args.bin_width / 2, args.bin_width)
centers = 0.5 * (edges[1:] + edges[:-1])


def share(selection):
    return float(np.sum(w[selection]) / w.sum())


def ratio_peak(ratio):
    """Full-absorption peak of E_step/E_true: iterated Gaussian fit around the mode above 0.7."""
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


rprint(f"\n[bold]{config} {name}[/bold] ({subfolder}): {mask.sum()} primary clusters, endpoint {endpoint:g} MeV, "
       + " ".join(f"{cls} {100 * share(sel):.1f}%" for cls, sel in FLASH_CLASSES.items()))

json_summary = {
    "config": config, "name": name, "sample": subfolder, "truth_variable": true_name, "range_variable": args.true_range_variable or true_name,
    "purity_variable": SAMPLE_PURITY.get(family, "Purity"),
    "weighting": "unweighted" if args.unweighted else weight_branch,
    "mc_events": int(mask.sum()), "endpoint_mev": endpoint, "electron_tau_us": tau, "flat_gain_adc_per_mev": gain,
    "max_lifetime_correction": float(np.exp(info["TIMEWINDOW"] * 1e6 / tau)),
    "flash_categories": {cls: share(sel) for cls, sel in FLASH_CLASSES.items()},
    "impure_fraction": share(purity[mask] < 0.999),
}

# ── Energy-independent: response per step and lifetime correction by flash category ──
response_rows = []
for step in STEPS[1:]:
    r = electron_scale[step][mask] / true_energy[mask]
    finite = np.isfinite(r)
    mu, sigma = ratio_peak(r)
    response_rows.append({
        "Config": config, "Name": name, "Step": step, "Label": STEP_LABELS[step],
        "Median": float(np.median(r[finite])), "P84": float(np.percentile(r[finite], 84)),
        "P99": float(np.percentile(r[finite], 99)), "FracAbove1": share(finite & (r > 1)),
        "PeakMean": mu, "PeakSigma": sigma,
    })
json_summary["response"] = {row["Step"]: {k: row[k] for k in ["Median", "P84", "P99", "FracAbove1", "PeakMean", "PeakSigma"]}
                            for row in response_rows}
lifetime_ratio = reco_lifetime[mask] / true_lifetime[mask]
json_summary["lifetime_ratio_p01_p50_p99"] = {
    cls: np.percentile(lifetime_ratio[sel], [1, 50, 99]).tolist() for cls, sel in FLASH_CLASSES.items() if sel.any()
}

fig = make_subplots(rows=1, cols=1)
for step in STEPS[1:]:
    r = electron_scale[step][mask] / true_energy[mask]
    finite = np.isfinite(r)
    h, bin_edges = np.histogram(r[finite], bins=150, range=(0, 1.5), weights=w[finite])
    fig.add_trace(go.Scatter(x=0.5 * (bin_edges[1:] + bin_edges[:-1]), y=np.where(h > 0, h / w.sum(), np.nan),
                             mode="lines", line_shape="hvh", name=STEP_LABELS[step],
                             line=dict(color=STEP_COLORS[step], width=2)))
fig.add_vline(x=1, line_dash="dash", line_color="#898781", line_width=1)
format_coustom_plotly(fig, title=f"Primary-cluster response per step - {config} {name}", log=(False, True),
                      legend=dict(x=0.02, y=0.02), tickformat=(".1f", None), debug=args.debug)
fig.update_layout(xaxis_title=f"E<sub>step</sub> / {true_name}", yaxis_title="Fraction of clusters")
fit_y_ranges(fig, log_axes=("yaxis",), max_decades=5)
fig.update_yaxes(dtick=1, exponentformat="power")
save_figure(fig, save_path, config, name, subfolder=subfolder, filename="Chain_Response", rm=args.rewrite, debug=args.debug)

fig = make_subplots(rows=1, cols=1)
for cls, sel in FLASH_CLASSES.items():
    h, bin_edges = np.histogram(lifetime_ratio[sel], bins=110, range=(0.5, 1.6), weights=w[sel])
    fig.add_trace(go.Scatter(x=0.5 * (bin_edges[1:] + bin_edges[:-1]), y=np.where(h > 0, h / w.sum(), np.nan),
                             mode="lines", line_shape="hvh", name=f"{FLASH_LABELS[cls]} ({100 * share(sel):.0f}%)",
                             line=dict(color={"NoFlash": "#898781", "WrongFlash": "#eb6834", "CorrectFlash": "#2a78d6"}[cls], width=2)))
fig.add_vline(x=1, line_dash="dash", line_color="#898781", line_width=1)
format_coustom_plotly(fig, title=f"Lifetime correction reco / true - {config} {name}", log=(False, True),
                      legend=dict(x=0.02, y=0.02), tickformat=(".1f", None), debug=args.debug)
fig.update_layout(xaxis_title="exp(t<sub>flash</sub>/τ) / exp(t<sub>true</sub>/τ)", yaxis_title="Fraction of clusters")
fit_y_ranges(fig, log_axes=("yaxis",), max_decades=5)
fig.update_yaxes(dtick=1, exponentformat="power")
save_figure(fig, save_path, config, name, subfolder=subfolder, filename="Lifetime_Correction_Ratio",
            rm=args.rewrite, debug=args.debug)

# ── Per reconstructed energy: spectrum after each step and endpoint crossings ──
spectra_rows, endpoint_rows, cf_rows, cutflow_rows = [], [], [], {}
exposure = args.exposure if args.exposure is not None else get_evaluation_exposure(str(root), config=config)
json_summary["energies"] = {}
for energy_name in args.energy:
    if this_params["SAMPLE_FIT"][energy_name] != "linear":
        raise SystemExit(f"{energy_name} uses a {this_params['SAMPLE_FIT'][energy_name]} map; only linear is supported")
    progression = {step: electron_scale[step][mask] for step in STEPS}
    final = np.asarray(reco[energy_name], dtype=float)[mask]
    labels = {"TrueEnergy": f"True {true_name}", **STEP_LABELS}
    branches = {}
    if energy_name == "SolarEnergy":
        # SolarEnergy = (Energy - OFFSET - I) / A, with OFFSET picked per cluster by the random-forest
        # discriminant (lib/cluster.py compute_reco_energy). Undo the linear map to recover the offset
        # step, and classify each cluster by the branch whose OFFSET it received.
        solar = reco_info["SOLAR"]
        offset_step = final * solar["ENERGY_AMP"] + solar["INTERSECTION"]
        applied = energy[mask] - offset_step
        lower = np.abs(applied - discriminant_info["LOWER"]["OFFSET"]) < np.abs(applied - discriminant_info["UPPER"]["OFFSET"])
        branches = {"UpperBranch": ~lower, "LowerBranch": lower}
        progression["Offset"] = offset_step
        labels["Offset"] = (f"+ Discriminant offset (+{-discriminant_info['UPPER']['OFFSET']:.2f} / "
                            f"+{-discriminant_info['LOWER']['OFFSET']:.2f} MeV)")
        labels["Map"] = f"+ {energy_name} linear calibration"
    else:
        labels["Map"] = f"+ {energy_name} map (Q-value offset)"
    progression["Map"] = final

    for step, values in progression.items():
        counts = np.histogram(values, bins=edges, weights=w)[0]
        over = values > endpoint
        spectra_rows.append({"Config": config, "Name": name, "Energy": energy_name, "Step": step, "Label": labels[step],
                             "Bins": centers, "Counts": counts,
                             "Density": counts / (max(counts.sum(), 1e-300) * args.bin_width)})
        endpoint_rows.append({
            "Config": config, "Name": name, "Energy": energy_name, "Step": step, "Label": labels[step], "Endpoint": endpoint,
            "FracAboveEndpoint": share(over), "FracAboveTrue": share(values > true_energy[mask]),
            **{f"AboveShare{cls}": float(np.sum(w[over & sel]) / max(np.sum(w[over]), 1e-30))
               for cls, sel in {**FLASH_CLASSES, **branches}.items()},
        })
    for key, values in electron_counterfactual.items():
        over = values[mask] > endpoint
        cf_rows.append({"Config": config, "Name": name, "Energy": energy_name, "Counterfactual": key,
                        "Label": COUNTERFACTUALS[key], "FracAboveEndpoint": share(over),
                        "ReferenceFracAboveEndpoint": share(progression["Calibration"] > endpoint)})

    rows = [r for r in endpoint_rows if r["Energy"] == energy_name]
    rprint(f"  {energy_name}: fraction above the true endpoint ({endpoint:g} MeV) after each step"
           + ("".join(f", {key} {100 * share(sel):.1f}% of clusters" for key, sel in branches.items())))
    for r in rows:
        shares = " ".join(f"{cls[:-5]} {100 * r[f'AboveShare{cls}']:5.1f}%" for cls in FLASH_CLASSES)
        shares += "".join(f" {key[:5]} {100 * r[f'AboveShare{key}']:5.1f}%" for key in branches)
        rprint(f"    {r['Step']:<12} {100 * r['FracAboveEndpoint']:7.3f}%  (E>E_true {100 * r['FracAboveTrue']:7.3f}%)  [{shares}]")
    for r in [r for r in cf_rows if r["Energy"] == energy_name]:
        rprint(f"    only {r['Counterfactual']:<13} back to truth: {100 * r['FracAboveEndpoint']:7.3f}% "
               f"(vs {100 * r['ReferenceFracAboveEndpoint']:.3f}% after calibration)")
    # Same rows and units as cutflow_plot.py (1 MeV bins over RECO_ENERGY_RANGE, events / MeV at the
    # evaluation livetime over the full detector mass), one Stage per step. No smoothing: the
    # Smoothed* columns repeat the raw counts so the schema stays the same.
    e_range = load_analysis_info(str(root)).get("RECO_ENERGY_RANGE", [0, 30])
    cut_edges = np.arange(e_range[0], e_range[1] + 1e-9, 1.0)
    cut_centers = 0.5 * (cut_edges[:-1] + cut_edges[1:])
    scale = 1.0 if args.unweighted else get_full_detector_mass(config, info) * exposure
    component = "Solar" if is_signal and args.weight == "analysis" else name
    meta = {"Config": config, "Name": name, "Folder": None, "Component": component,
            "NHits": None, "OpHits": None, "AdjCl": None,
            "Exposure": None if args.unweighted else exposure, "ExposureUnit": None if args.unweighted else "year",
            "EnergyUnit": "MeV",
            "CountsUnit": "MC clusters / MeV" if args.unweighted else f"events / MeV / {exposure:.0f} yr"}
    cutflow_rows[energy_name] = []
    for step, values in progression.items():
        h = np.histogram(values, bins=cut_edges, weights=w)[0] * scale
        err = np.sqrt(np.histogram(values, bins=cut_edges, weights=w ** 2)[0]) * scale
        mc = np.histogram(values, bins=cut_edges)[0]
        cutflow_rows[energy_name].append({**meta, "Stage": CUTFLOW_STAGES.get(step, energy_name), "Step": step,
                             "Energy": cut_centers.tolist(), "Counts": h.tolist(), "CountsError": err.tolist(),
                             "SmoothedCounts": h.tolist(), "SmoothedCountsError": err.tolist(),
                             "MCCounts": mc.tolist(), "MCCountsError": np.sqrt(mc).tolist(), "Smoothing": "none"})

    # The final stage against an existing cutflow_plot.py pkl ("Raw" stage, rescaled to this livetime)
    cutflow_match = None
    final_row = cutflow_rows[energy_name][-1]
    for analysis in ["DayNight", "HEP", "Sensitivity"]:
        cut_pkl = (f"{root}/output/data/solar/cutflow/{config}/{args.reference_folder.lower()}/{analysis.lower()}/"
                   f"{config}_{energy_name}_{analysis}_Cutflow.pkl")
        if args.true_range or args.unweighted or not os.path.exists(cut_pkl):
            continue
        raw = pd.read_pickle(cut_pkl)
        raw = raw.loc[(raw["Stage"] == "Raw") & (raw["Component"] == component)]
        if raw.empty:
            continue
        theirs = np.asarray(raw["Counts"].iloc[0], dtype=float) * exposure / float(raw["Exposure"].iloc[0])
        ours = np.asarray(final_row["Counts"], dtype=float)
        cutflow_match = {"pkl": cut_pkl, "pkl_mtime": os.path.getmtime(cut_pkl),
                         "identical": bool(np.allclose(ours, theirs, rtol=1e-6, atol=0))}
        tag = "[green]identical[/green]" if cutflow_match["identical"] else "[red]DIFFERENT[/red]"
        rprint(f"  cutflow_plot.py pkl ({analysis}, Raw, {component}): {tag}")
        break

    # Closure against cutflow_plot.py: its "Raw" stage histograms the 03_analysis.py Ref exports
    # (every cluster, 1 MeV bins over RECO_ENERGY_RANGE) and only rescales by mass x exposure.
    closure_info = None
    ref_dir = f"{root}/output/data/results/{config}/{name}/{args.reference_folder.lower()}/{config}_{name}"
    ref_files = [f"{ref_dir}_AnalysisData_{energy_name}_Ref.pkl", f"{ref_dir}_{reference_weight}_{energy_name}_Ref.pkl"]
    if args.true_range or args.unweighted:
        pass
    elif not all(os.path.exists(p) for p in ref_files):
        rprint(f"[yellow][WARNING][/yellow] No 03_analysis.py Ref export for {name}/{energy_name}; cutflow closure skipped")
    else:
        ref_energy = np.asarray(pickle.load(open(ref_files[0], "rb")), dtype=float)
        ref_weights = np.asarray(pickle.load(open(ref_files[1], "rb")), dtype=float)
        e_range = load_analysis_info(str(root)).get("RECO_ENERGY_RANGE", [0, 30])
        cut_edges = np.arange(e_range[0], e_range[1] + 1e-9, 1.0)
        ours = np.histogram(progression["Map"], bins=cut_edges, weights=w)[0]
        theirs = np.histogram(ref_energy, bins=cut_edges, weights=ref_weights)[0]
        norm_diff = np.abs(ours / ours.sum() - theirs / theirs.sum()) if theirs.sum() > 0 else np.full(len(ours), np.nan)
        closure_info = {
            "reference": ref_files, "reference_mtime": os.path.getmtime(ref_files[0]),
            "clusters": [int(mask.sum()), int(len(ref_energy))],
            "weight_sums": [float(w.sum()), float(ref_weights.sum())],
            "max_rel_bin_diff": float(np.nanmax(np.abs(ours - theirs) / np.where(theirs > 0, theirs, np.nan))) if (theirs > 0).any() else np.nan,
            "max_norm_bin_diff": float(np.nanmax(norm_diff)),
            "identical": bool(len(ref_energy) == mask.sum() and np.allclose(ours, theirs, rtol=1e-6, atol=0)),
        }
        tag = "[green]identical[/green]" if closure_info["identical"] else "[red]DIFFERENT[/red]"
        rprint(f"  cutflow_plot.py closure ({os.path.basename(ref_files[1])}): {tag}; clusters {mask.sum()} vs {len(ref_energy)}, "
               f"weight sum {w.sum():.6g} vs {ref_weights.sum():.6g}, max |Δ| per bin {100 * closure_info['max_rel_bin_diff']:.3g}% "
               f"(normalised {closure_info['max_norm_bin_diff']:.2e})")
    json_summary["energies"][energy_name] = {
        "cutflow_closure": closure_info,
        "cutflow_pkl_match": cutflow_match,
        "branch_fractions": {key: share(sel) for key, sel in branches.items()},
        "steps": {r["Step"]: {k: v for k, v in r.items() if k.startswith(("Frac", "AboveShare"))} for r in rows},
        "counterfactuals": {r["Counterfactual"]: r["FracAboveEndpoint"] for r in cf_rows if r["Energy"] == energy_name},
    }

    # Unit-area overlay of the spectrum after each step
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Linear", "Log"), horizontal_spacing=0.1)
    for col in [1, 2]:
        for row in [r for r in spectra_rows if r["Energy"] == energy_name]:
            fig.add_trace(go.Scatter(x=row["Bins"], y=np.where(row["Density"] > 0, row["Density"], np.nan), mode="lines",
                                     line_shape="hvh", name=row["Label"], legendgroup=row["Step"], showlegend=col == 1,
                                     line=dict(color=STEP_COLORS[row["Step"]], width=2,
                                               dash="dot" if row["Step"] == "TrueEnergy" else "solid")),
                          row=1, col=col)
        fig.add_vline(x=endpoint, line_dash="dash", line_color="#898781", line_width=1, row=1, col=col)
    slice_tag = f", true {args.true_range[0]:g}-{args.true_range[1]:g} MeV" if args.true_range else ", full sample"
    format_coustom_plotly(fig, title=f"Normalised spectrum per step ({energy_name}{slice_tag}) - {config} {name}",
                          matches=(None, None), legend=dict(x=1.02, y=1.0), figsize=(1400, 550),
                          tickformat=(".0f", None), debug=args.debug)
    fig.update_xaxes(title="Energy after each step (MeV)", range=[0, args.max_energy])
    fig.update_yaxes(title="Fraction per MeV (unit area)", col=1)
    fig.update_yaxes(dtick=1, exponentformat="power", col=2)
    fit_y_ranges(fig, log_axes=("yaxis2",), reference_groups=("TrueEnergy",))
    save_figure(fig, save_path, config, name, subfolder=subfolder, filename=f"{energy_name}_Chain_Spectra_Normalized",
                rm=args.rewrite, debug=args.debug)

for energy_name, rows in cutflow_rows.items():
    save_df(pd.DataFrame(rows), data_path, config, name, subfolder=subfolder, filename=f"{energy_name}_Chain_Cutflow",
            rm=args.rewrite, debug=args.debug)
for rows, filename in [(spectra_rows, "Chain_Spectra"), (endpoint_rows, "Chain_Endpoint"),
                       (cf_rows, "Chain_Counterfactuals"), (response_rows, "Chain_Response")]:
    save_df(pd.DataFrame(rows), data_path, config, name, subfolder=subfolder, filename=filename,
            rm=args.rewrite, debug=args.debug)
summary_dir = f"{data_path}/{config}/{name}/{subfolder}"
os.makedirs(summary_dir, exist_ok=True)
with open(f"{summary_dir}/{config}_{name}_Chain_Summary.json", "w") as f:
    json.dump(json_summary, f, indent=1, default=float)
rprint(f"[green]Saved smearing-chain outputs to {summary_dir}[/green]")
