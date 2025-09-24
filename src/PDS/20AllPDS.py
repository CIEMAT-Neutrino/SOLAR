# This script runs all the calibration and reconstruction steps for the workflow.

# It is designed to be run from the command line.
# It uses the argparse library to handle command line arguments.

import sys

sys.path.insert(0, "../../")

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
        "vd_1x8x14_3view_30deg_optimistic",
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
for config, name in track(
    product(args.config, args.name),
    description="Running all PDS macros...",
    total=len(args.config) * len(args.name),
):
    # Your processing code here
    os.system(f"python3 {root}/src/PDS/21OpFlash.py --config {config} --name {name}")
    os.system(f"python3 {root}/src/PDS/22AdjOpFlash.py --config {config} --name {name}")
    os.system(
        f"python3 {root}/src/PDS/23MatchedOpFlash.py --config {config} --name {name}"
    )
