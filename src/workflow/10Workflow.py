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
    help="The configuration to load",
    default="hd_1x2x6",
)
parser.add_argument(
    "--name", type=str, help="The name of the configuration", default="marley_signal"
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()

# Run the first script with the arguments
os.system(
    f"python3 {root}/src/workflow/01Correction.py --config {args.config} --name {args.name}"
)
os.system(
    f"python3 {root}/src/workflow/02Calibration.py --config {args.config} --name {args.name}"
)
os.system(
    f"python3 {root}/src/workflow/03Discrimination.py --config {args.config} --name {args.name}"
)
os.system(
    f"python3 {root}/src/workflow/04Reconstruction.py --config {args.config} --name {args.name}"
)
os.system(
    f"python3 {root}/src/workflow/05Smearing.py --config {args.config} --name {args.name}"
)
