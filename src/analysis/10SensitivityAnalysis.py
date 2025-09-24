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
    help="The configuration to load",
    default="hd_1x2x6_centralAPA",
)
parser.add_argument(
    "--name", type=str, help="The name of the configuration", default="marley"
)
parser.add_argument(
    "--reference",
    type=str,
    help="The name of the reference analysis",
    choices=["DayNight", "HEP"],
    default="HEP",
)
parser.add_argument(
    "--folder",
    type=str,
    help="The name of the results folder",
    default="Reduced",
    choices=["Reduced", "Nominal"],
)
parser.add_argument(
    "--signal_uncertanty",
    type=float,
    help="The signal uncertanty for the analysis",
    default=0.04,
)
parser.add_argument(
    "--background_uncertanty",
    type=float,
    help="The background uncertanty for the analysis",
    default=0.02,
)
parser.add_argument(
    "--energy",
    type=str,
    help="The energy for the analysis",
    choices=[
        "SignalParticleK",
        "ClusterEnergy",
        "TotalEnergy",
        "SelectedEnergy",
        "SolarEnergy",
    ],
    default="SolarEnergy",
)
parser.add_argument(
    "--fiducial", type=int, help="The fiducial cut for the analysis", default=None
)
parser.add_argument(
    "--nhits", type=int, help="The nhit cut for the analysis", default=None
)
parser.add_argument(
    "--ophits", type=int, help="The ophit cut for the analysis", default=None
)
parser.add_argument(
    "--adjcl", type=int, help="The adjacent cluster cut for the analysis", default=None
)
parser.add_argument("--background", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()


# Run the first script with the arguments
def build_command(script_name, additional_args):
    return (
        f"python3 {root}/src/analysis/{script_name} --config {args.config} --name {args.name} --reference {args.reference} "
        + " ".join(additional_args)
    )


base_args = [
    f"--energy {args.energy}",
]

if args.fiducial is not None:
    base_args.append(f"--fiducial {args.fiducial}")
if args.nhits is not None:
    base_args.append(f"--nhits {args.nhits}")
if args.ophits is not None:
    base_args.append(f"--ophits {args.ophits}")
if args.adjcl is not None:
    base_args.append(f"--adjcl {args.adjcl}")

os.system(build_command("14SensitivitySignalTemplate.py", base_args + ["--no-test"]))
os.system(build_command("14SensitivityBackgroundTemplate.py", base_args))
os.system(
    build_command(
        "14Sensitivity.py",
        base_args
        + [
            f"--signal_uncertanty {args.signal_uncertanty}",
            f"--background_uncertanty {args.background_uncertanty}",
            "--no-background",
        ],
    )
)
os.system(
    build_command(
        "14Sensitivity.py",
        base_args
        + [
            f"--signal_uncertanty {args.signal_uncertanty}",
            f"--background_uncertanty {args.background_uncertanty}",
        ],
    )
)
os.system(
    build_command(
        "14SensitivityContourPlot.py",
        base_args
        + [
            f"--signal_uncertanty {args.signal_uncertanty}",
            f"--background_uncertanty {args.background_uncertanty}",
            "--no-background",
        ],
    )
)
os.system(
    build_command(
        "14SensitivityContourPlot.py",
        base_args
        + [
            f"--signal_uncertanty {args.signal_uncertanty}",
            f"--background_uncertanty {args.background_uncertanty}",
        ],
    )
)
