# This script runs all the calibration and reconstruction steps for the workflow.

# It is designed to be run from the command line.
# It uses the argparse library to handle command line arguments.

import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *

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
    os.system(
        f"python3 {root}/src/TPC/32ElectronEnergyResolution.py --config {config} --name {name}"
    )
    os.system(
        f"python3 {root}/src/TPC/33AdjClusters.py --config {config} --name {name}"
    )
    os.system(
        f"python3 {root}/src/TPC/34EnergyResolution.py --config {config} --name {name}"
    )
