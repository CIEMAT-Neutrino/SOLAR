# This script runs all the calibration and reconstruction steps for the workflow.

# It is designed to be run from the command line.
# It uses the argparse library to handle command line arguments.

import os
import sys
import subprocess

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *


def run_command(command: str):
    rprint(f"\n[green][CMD][/green] {command}")
    completed = subprocess.run(command, shell=True)
    if completed.returncode != 0:
        raise SystemExit(
            f"Workflow stopped with exit code {completed.returncode}.\n"
            f"Executed command: {command}"
        )

# Define flags for the analysis config and name with the python parser
parser = argparse.ArgumentParser(
    description="Plot the energy distribution of the particles"
)
parser.add_argument(
    "--config",
    type=str,
    nargs="+",
    help="The configuration to load",
    default=[
        "hd_1x2x6",
        "hd_1x2x6_centralAPA",
        "hd_1x2x6_lateralAPA",
        "vd_1x8x14_3view_30deg",
        "vd_1x8x14_3view_30deg_nominal",
    ],
)
parser.add_argument(
    "--name",
    type=str,
    nargs="+",
    help="The name of the configuration",
    default=["marley", "marley_official"],
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()

# Run the first script with the arguments
for config, name in product(args.config, args.name):
    # Your processing code here
    run_command(f"python3 {root}/src/PDS/21OpFlash.py --config {config} --name {name}")
    # os.system(f"python3 {root}/src/PDS/22AdjOpFlash.py --config {config} --name {name}")
    run_command(
        f"python3 {root}/src/PDS/23MatchedOpFlash.py --config {config} --name {name}"
    )
    run_command(
        f"python3 {root}/src/PDS/24MatchedOpFlashEfficiency.py --config {config} --name {name}"
    )
