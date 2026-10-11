"""
01b_lifetime_reco.py — Drift-Attenuation Correction with the Reconstructed Drift Time
=====================================================================================
Exports the primary-cluster charge per true electron energy as a function of the
cluster time for three versions of the lifetime correction:

  Raw              Q / E_e                                   no correction
  Truth Corrected  Q * exp(|t| / tau) / E_e                  cluster time (true drift, t0 = 0)
  Reco Corrected   Q * exp(|RecoDriftTime| / tau) / E_e      flash-matched drift time (Correction)

tau is the ELECTRON_TAU of the calibration (electroncharge_correction.json, with the
marley_official fallback of lib/cluster.py). Same cluster selection as
01_correction.py (CORRECTION preset, Time > 500). Only the data frame is written;
no calibration file is touched.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *

data_path = f"{root}/output/data/workflow/correction"
os.makedirs(data_path, exist_ok=True)

parser = argparse.ArgumentParser(description="Export raw, truth- and reco-corrected charge per MeV vs drift time")
parser.add_argument("--config", type=str, help="The configuration to load", default="hd_1x2x6")
parser.add_argument("--name", type=str, help="The name of the configuration", default="marley_official")
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
args = parser.parse_args()

config, name = args.config, args.name
configs = {config: [name]}

run, output = load_multi(configs, preset="CALIBRATION", debug=args.debug)
rprint(output)
run = compute_reco_workflow(run, configs, workflow="CALIBRATION", debug=args.debug)
filtered_run, mask, output = compute_filtered_run(
    run,
    configs,
    presets=["CORRECTION"],
    params={("Reco", "Time"): ("bigger", 500)},
    debug=args.debug,
)
rprint(output)
data = filtered_run["Reco"]

info = json.load(open(f"{root}/config/{config}/{config}_config.json"))
corr_file = f"{root}/config/{config}/{name}/{config}_calib/{config}_electroncharge_correction.json"
if not os.path.exists(corr_file):
    corr_file = f"{root}/config/{config}/marley_official/{config}_calib/{config}_electroncharge_correction.json"
tau = json.load(open(corr_file))["ELECTRON_TAU"]

charge = np.asarray(data["Charge"], dtype=float)
time = np.asarray(data["Time"], dtype=float)
energy = np.asarray(data["ElectronK"], dtype=float)
versions = {
    "Raw": np.ones_like(charge),
    "Truth Corrected": np.exp(np.abs(time) / tau),
    "Reco Corrected": np.asarray(data["Correction"], dtype=float),
}
rprint(f"tau = {tau:.1f} us, {len(charge)} clusters; reco/true correction median "
       f"{np.nanmedian(versions['Reco Corrected'] / versions['Truth Corrected']):.4f}")

rows = []
for label, factor in versions.items():
    rows.append(
        {
            "Geometry": info["GEOMETRY"],
            "Config": config,
            "Name": name,
            "#Hits": None,
            "Time": time,
            "ChargePerEnergy": charge * factor / energy,
            "Correction": label,
        }
    )
save_df(
    pd.DataFrame(rows),
    data_path,
    config,
    name,
    filename="Charge_Lifetime_Correction_Reco",
    rm=args.rewrite,
    debug=args.debug,
)
