import os
import sys
import subprocess
from shlex import quote
from typing import List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *


parser = argparse.ArgumentParser(
    description="Compute sensitivity templates (signal/background) without generating plots"
)
parser.add_argument("--config", type=str, default="hd_1x2x6_centralAPA")
parser.add_argument("--signal", type=str, default="marley")
parser.add_argument(
    "--reference",
    type=str,
    choices=["DayNight", "SENSITIVITY", "HEP"],
    default="SENSITIVITY",
)
parser.add_argument(
    "--folder",
    type=str,
    choices=["Reduced", "Truncated", "Nominal"],
    default="Nominal",
)
parser.add_argument("--signal_uncertainty", type=float, default=0.04)
parser.add_argument("--background_uncertainty", type=float, default=0.02)
parser.add_argument("--exposure", type=float, default=get_analysis_exposure(str(root), "Sensitivity"), help="Exposure in years. Saved templates are absolute counts scaled as exposure_yr × detector_mass_kT × rate. Default from ANALYSIS_EXPOSURES['SENSITIVITY']['PRIMARY'] in config/analysis/config.json.")
parser.add_argument(
    "--energy",
    type=str,
    choices=[
        "SignalParticleK",
        "MainK",
        "ClusterEnergy",
        "TotalEnergy",
        "SelectedEnergy",
        "SolarEnergy",
    ],
    default="SolarEnergy",
)
parser.add_argument("--nhits", type=int, default=None)
parser.add_argument("--ophits", type=int, default=None)
parser.add_argument("--adjcls", type=int, default=None)
parser.add_argument(
    "--template",
    type=str,
    choices=["signal", "background", "all"],
    default="all",
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument(
    "--oscillation_backend",
    type=str,
    choices=["file", "prob3", "nufast"],
    default="nufast",
    help="Oscillation backend forwarded to BackgroundTemplate and SignalTemplate.",
)
parser.add_argument("--charge_threshold", type=float, default=0,
    help="Charge threshold Q (ADC) forwarded to signal/background template scripts.")
parser.add_argument("--study_label", type=str, default=None,
    help="Tag forwarded to template scripts to label output subfolders for charge study variants.")
parser.add_argument(
    "--membrane_veto",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "Accept only cathode/APA optical matches (QUALITY_CUTS.OPFLASH_PLANE, plane 0). "
        "This is the default. --no-membrane_veto also accepts membrane and endcap matches "
        "(VD planes 1-4). Forwarded to sensitivity/02_signal_template.py. "
        "Used by the membrane_veto study."
    ),
)
parser.add_argument(
    "--flyweight",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Enable flyweight mode: save only unoscillated base templates in coarse bins instead of "
        "~14k oscillation templates. Forwarded to 02_signal_template.py only (signal templates)."
    ),
)
parser.add_argument(
    "--truth_fiducial",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Truth-fiducial study: forwarded to both template scripts, so signal and background "
        "templates are built from the labeled truth-fiducial Rebin pkls and best fiducials."
    ),
)
parser.add_argument(
    "--force-all-cuts",
    action="store_true",
    default=False,
    help=(
        "Forwarded to 01_background_template.py: build background templates for every cut. "
        "run_sensitivity.py sets it for studies that optimise their own cuts, which have no "
        "best-cut map yet when the background templates are produced."
    ),
)

parser.add_argument("--truth_purity", action=argparse.BooleanOptionalAction, default=True,
                    help="With --truth_fiducial: also require a pure, position-consistent flash match "
                         "(lib.fiducial.truth_match_purity_mask, BACKGROUND_SAMPLES.truth_match_purity). --no-truth_purity disables it.")
parser.add_argument("--truth_min_purity", type=float, default=None, help="Override truth_match_purity.min_purity (MatchedOpFlashPur threshold).")
parser.add_argument("--truth_drift_tol", type=float, default=None, help="Override truth_match_purity.drift_tolerance_cm (|RecoX - truthX|).")
parser.add_argument("--truth_transverse_tol", type=float, default=None, help="Override truth_match_purity.transverse_tolerance_cm (|RecoY/Z - truthY/Z|).")
parser.add_argument("--fiducial_folder", type=str, default=None, choices=["Reduced", "Truncated", "Nominal"],
                    help="Read BestFiducials*.json of this folder instead of --folder (bkgmodel studies hold the Truncated volumes).")

parser.add_argument("--fiducial_from_reco", action=argparse.BooleanOptionalAction, default=False,
                    help="With --truth_fiducial: keep the reference (reco-optimised) fiducial VOLUMES from BestFiducials.json and only swap "
                         "the position estimate to truth (fiduc_truth_refvol study). Default: read BestFiducials_fiduc_truth.json.")

args = parser.parse_args()


def build_common_args() -> List[str]:
    common = [
        "--config",
        args.config,
        "--signal",
        args.signal,
        "--reference",
        args.reference,
        "--folder",
        args.folder,
        "--energy",
        args.energy,
        "--exposure",
        str(args.exposure),
        "--signal_uncertainty",
        str(args.signal_uncertainty),
        "--background_uncertainty",
        str(args.background_uncertainty),
        "--rewrite" if args.rewrite else "--no-rewrite",
        "--debug" if args.debug else "--no-debug",
        "--plot" if args.plot else "--no-plot",
        "--oscillation_backend", args.oscillation_backend,
    ]
    if args.nhits is not None:
        common.extend(["--nhits", str(args.nhits)])
    if args.ophits is not None:
        common.extend(["--ophits", str(args.ophits)])
    if args.adjcls is not None:
        common.extend(["--adjcls", str(args.adjcls)])
    if args.charge_threshold > 0:
        common.extend(["--charge_threshold", str(args.charge_threshold)])
    if not args.membrane_veto:
        common.append("--no-membrane_veto")
    if args.study_label:
        common.extend(["--study_label", args.study_label])
    if args.fiducial_folder:
        common.extend(["--fiducial_folder", args.fiducial_folder])
    if args.fiducial_from_reco:
        common.append("--fiducial_from_reco")
    return common


def truth_purity_args() -> List[str]:
    out = ["--truth_purity" if args.truth_purity else "--no-truth_purity"]
    for flag in ("truth_min_purity", "truth_drift_tol", "truth_transverse_tol"):
        if getattr(args, flag) is not None:
            out += [f"--{flag}", str(getattr(args, flag))]
    return out


def run_macro(script_name: str, extra_args: Optional[List[str]] = None):
    command = ["python3", f"{root}/{script_name}", *build_common_args()]
    if extra_args:
        command.extend(extra_args)

    command_str = " ".join(quote(str(item)) for item in command)
    rprint(f"\n[green][CMD][/green] {command_str}")
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise SystemExit(
            f"Template computation failed in {script_name} with exit code {completed.returncode}.\n"
            f"Executed command: {command_str}"
        )


if args.template in ["background", "all"]:
    background_args = (["--truth_fiducial"] if args.truth_fiducial else []) + (["--force-all-cuts"] if args.force_all_cuts else [])
    run_macro("src/physics/sensitivity/01_background_template.py", extra_args=background_args)

if args.template in ["signal", "all"]:
    flyweight_args = ["--flyweight"] if args.flyweight else []
    truth_fiducial_args = ["--truth_fiducial"] if args.truth_fiducial else []
    run_macro("src/physics/sensitivity/02_signal_template.py", extra_args=["--no-test"] + flyweight_args + truth_fiducial_args + (truth_purity_args() if args.truth_fiducial else []))
