import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *
from lib.fiducial import get_best_fiducial, get_best_fiducial_bands

parser = argparse.ArgumentParser(
    description="Thresholds of the Sensitivity score categories (config SENSITIVITY_SCORE_CATEGORIES): quantiles of the "
    "signal/background classifier score of the CC signal in ENERGY_RANGE, after the quality cuts, the Sensitivity fiducial "
    "volume of FIDUCIAL_FOLDER (the same thresholds for every folder) and the neutron-capture veto. "
    "Writes config/<config>/<config>_score_categories.json."
)
parser.add_argument("--config", type=str, default="hd_1x2x6_centralAPA")
parser.add_argument("--signal", type=str, default="marley_final")
parser.add_argument("--folder", type=str, choices=["Reduced", "Truncated", "Nominal", "Pessimistic"], default="Truncated")
parser.add_argument("--energy", type=str, default="SolarEnergy")
args = parser.parse_args()

config = args.config
cfg = load_analysis_info(str(root)).get("SENSITIVITY_SCORE_CATEGORIES", {})
if config not in cfg.get("ENABLED_CONFIGS", []):
    rprint(f"[cyan][INFO][/cyan] Sensitivity score categories off for {config}; nothing to do.")
    raise SystemExit(0)

info = json.load(open(f"{root}/config/{config}/{config}_config.json"))
configs = {config: [args.signal]}
run, _ = load_multi(configs, preset="SIGNIFICANCE", branches={"Config": ["Geometry"]})
run = compute_reco_workflow(
    run, configs,
    params={"DEFAULT_SIGNAL_WEIGHT": ["truth", "osc"], "DEFAULT_SIGNAL_NADIR": ["mean", "day", "night"],
            "PARTICLE_TYPE": "signal", "PARTICLE_WEIGHTING": "volume", "OSCILLATION_BACKEND": "nufast"},
    workflow="SIGNIFICANCE", rm_branches=False,
)
reco = run["Reco"]
energy = np.asarray(reco[args.energy], dtype=float)
e_min, e_max = cfg.get("ENERGY_RANGE", [8, 13])
fiducial_folder = cfg.get("FIDUCIAL_FOLDER", "Truncated")
fiducials = json.load(open(f"{root}/config/analysis/fiducial/{fiducial_folder.lower()}/BestFiducials.json"))
spatial = np.asarray(build_energy_band_spatial_mask(
    run, config, info["DETECTOR_SIZE_X"] + 2 * info["DETECTOR_GAP_X"], info["DETECTOR_SIZE_Y"] + 2 * info["DETECTOR_GAP_Y"], info,
    fiducial_folder, get_best_fiducial(fiducials, config, args.energy, "SENSITIVITY"),
    get_best_fiducial_bands(fiducials, config, args.energy, "SENSITIVITY"), args.energy), dtype=bool)
selected = (
    surface_quality_mask(str(root), args.signal, args.folder, reco["SignalParticleSurface"], reco, info)
    & accepted_flash_planes(reco["MatchedOpFlashPlane"], str(root), True)
    & (np.asarray(reco["MatchedOpFlashPE"]) > 0)
    & spatial
    & neutron_capture_veto_keep(str(root), config, "Sensitivity", energy, reco["Energy"], reco=reco)
    & (energy >= e_min) & (energy < e_max)
)
score = np.asarray(reco[cfg.get("VARIABLE", "SignalBackgroundScore")], dtype=float)[selected]
quantiles = [float(q) for q in cfg.get("QUANTILES", [0.25, 0.5, 0.75])]
thresholds = [float(t) for t in np.quantile(score, quantiles)]
out = {
    "VARIABLE": cfg.get("VARIABLE", "SignalBackgroundScore"),
    "SIGNAL_SAMPLE": args.signal,
    "ENERGY": args.energy,
    "ENERGY_RANGE": [e_min, e_max],
    "FIDUCIAL_FOLDER": fiducial_folder,
    "QUANTILES": quantiles,
    "THRESHOLDS": thresholds,
    "N_CLUSTERS": int(selected.sum()),
}
path = score_categories_path(str(root), config)
# Unchanged thresholds leave the file (and its mtime, which the template staleness checks compare) untouched; the
# Truncated run and the background-model studies run this step concurrently.
if os.path.isfile(path) and json.load(open(path)) == json.loads(json.dumps(out)):
    rprint(f"[cyan][INFO][/cyan] {config}: score thresholds unchanged ({path})")
    raise SystemExit(0)
with open(f"{path}.tmp{os.getpid()}", "w") as handle:
    json.dump(out, handle, indent=2)
os.replace(f"{path}.tmp{os.getpid()}", path)
rprint(f"[cyan][INFO][/cyan] {config}: score thresholds {np.round(thresholds, 4).tolist()} from {int(selected.sum())} CC clusters -> {path}")
