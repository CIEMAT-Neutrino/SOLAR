import os
import sys
import subprocess
from shlex import quote
from typing import List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *


parser = argparse.ArgumentParser(
    description="Compute sensitivity templates (signal/background) without generating plots"
)
parser.add_argument("--config", type=str, default="hd_1x2x6_centralAPA")
parser.add_argument("--name", type=str, default="marley")
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
parser.add_argument("--exposure", type=float, default=30.0)
parser.add_argument(
    "--energy",
    type=str,
    choices=[
        "SignalParticleK",
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
args = parser.parse_args()


def build_common_args() -> List[str]:
    common = [
        "--config",
        args.config,
        "--name",
        args.name,
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
        "--no-plot",
    ]
    if args.nhits is not None:
        common.extend(["--nhits", str(args.nhits)])
    if args.ophits is not None:
        common.extend(["--ophits", str(args.ophits)])
    if args.adjcls is not None:
        common.extend(["--adjcls", str(args.adjcls)])
    return common


def run_macro(script_name: str, extra_args: Optional[List[str]] = None):
    command = ["python3", f"{root}/src/analysis/{script_name}", *build_common_args()]
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
    run_macro("14SensitivityBackgroundTemplate.py")

if args.template in ["signal", "all"]:
    run_macro("14SensitivitySignalTemplate.py", extra_args=["--no-test"])
