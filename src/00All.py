# This script runs all the calibration and reconstruction steps for the workflow.

# It is designed to be run from the command line.
# It uses the argparse library to handle command line arguments.

import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from lib import *

# Define flags for the analysis config and name with the python parser
parser = argparse.ArgumentParser(description="Run all the workflow")
parser.add_argument(
    "--config",
    type=str,
    help="The configuration to load",
    default=[
        "hd_1x2x6",
        "hd_1x2x6_centralAPA",
        "hd_1x2x6_lateralAPA",
        "vd_1x8x14_3view_30deg",
        "vd_1x8x14_3view_30deg_nominal",
        # "vd_1x8x14_3view_30deg_shielded",
    ],
)
parser.add_argument(
    "--name",
    type=str,
    help="The name of the configuration",
    default=["marley", "marley_official"],
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()
if isinstance(args.config, str):
    args.config = [args.config]
if isinstance(args.name, str):
    args.name = [args.name]

# Run the first script with the arguments
for config, name in product(args.config, args.name):
    os.system(
        f"python3 {root}/src/workflow/00AllWorkflow.py --config {config} --name {name}"
    )
    os.system(
        f"python3 {root}/src/preselection/10AllPreselection.py --config {config} --name {name}"
    )
    os.system(f"python3 {root}/src/PDS/20AllPDS.py --config {config} --name {name}")
    os.system(f"python3 {root}/src/TPC/30AllTPC.py --config {config} --name {name}")
    os.system(
        f"python3 {root}/src/vertex/40AllVertex.py --config {config} --name {name}"
    )
