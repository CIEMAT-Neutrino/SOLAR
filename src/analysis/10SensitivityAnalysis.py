import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *


# Run the first script with the arguments
def build_command(script_name, name, additional_args: Optional[list[str]] = None):
    if additional_args is None:
        additional_args = []
    command = f"python3 {root}/src/analysis/{script_name} --name {name} " + " ".join(
        additional_args
    )
    if args.debug:
        rprint(f"Running command: {command}")
    return command


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
    "--names",
    nargs="+",
    type=str,
    help="The name of the configuration",
    default=["marley", "gamma", "neutron"],
)
parser.add_argument(
    "--analysis",
    nargs="+",
    type=str,
    help="The name of the analysis",
    choices=["DayNight", "HEP", "Sensitivity"],
    default=["DayNight", "HEP", "Sensitivity"],
)
parser.add_argument(
    "--reference",
    type=str,
    help="The name of the reference analysis",
    choices=["Gaussian", "Asimov"],
    default=None,
)
parser.add_argument(
    "--folder",
    type=str,
    help="The name of the results folder",
    choices=["Reduced", "Truncated", "Nominal"],
    default="Reduced",
)
parser.add_argument(
    "--exposure",
    type=float,
    help="The exposure for the analysis in kT·y",
    default=100.0,
)
parser.add_argument(
    "--signal_uncertainty",
    type=float,
    help="The signal uncertainty for the analysis",
    default=None,
)
parser.add_argument(
    "--background_uncertainty",
    type=float,
    help="The background uncertainty for the analysis",
    default=None,
)
parser.add_argument(
    "--energy",
    nargs="+",
    type=str,
    help="The energy for the analysis",
    choices=[
        "SignalParticleK",
        "ClusterEnergy",
        "TotalEnergy",
        "SelectedEnergy",
        "SolarEnergy",
    ],
    default=["SolarEnergy"],
)
parser.add_argument(
    "--nhits", type=int, help="The nhit cut for the analysis", default=None
)
parser.add_argument(
    "--ophits", type=int, help="The ophit cut for the analysis", default=None
)
parser.add_argument(
    "--adjcls", type=int, help="The adjacent cluster cut for the analysis", default=None
)
parser.add_argument("--background", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()

# Define the arguments for the analysis scripts
base_args = [
    f"--config {args.config}",
    f"--folder {args.folder}",
    f"--energy {' '.join(args.energy)}",
    f"--debug" if args.debug else "--no-debug",
]
# Generate the analysis args for the analysis scripts
analysis_args = [
    f"--exposure {args.exposure}",
    f"--nhits {args.nhits}" if args.nhits is not None else "",
    f"--ophits {args.ophits}" if args.ophits is not None else "",
    f"--adjcls {args.adjcls}" if args.adjcls is not None else "",
    (
        f"--signal_uncertainty {args.signal_uncertainty}"
        if args.signal_uncertainty is not None
        else ""
    ),
    (
        f"--background_uncertainty {args.background_uncertainty}"
        if args.background_uncertainty is not None
        else ""
    ),
]
# Drop all empty strings from the list
analysis_args = [arg for arg in analysis_args if arg]

if args.rewrite:
    for name in args.names:
        os.system(build_command("0XFiducializeSignal.py", name, base_args))

    for name in args.names:
        if "marley" in name:
            os.system(build_command("0YBestFiducial.py", name, base_args))
        else:
            pass

    for name in args.names:
        os.system(build_command("11AnalysisSignal.py", name, base_args))

for name in args.names:
    if "marley" not in name:
        continue

    if "DayNight" in args.analysis:
        if args.reference is None:
            reference = "Gaussian"
        else:
            reference = args.reference

        os.system(build_command("12DayNight.py", name, base_args + analysis_args))
        os.system(
            build_command(
                "0ZBestSigmas.py",
                name,
                base_args + [f"--analysis DayNight", f"--reference {reference}"],
            )
        )
        os.system(
            build_command("12DayNightExposurePlot.py", name, base_args + analysis_args)
        )
        os.system(
            build_command(
                "12DayNightSignificancePlot.py", name, base_args + analysis_args
            )
        )

    elif "HEP" in args.analysis:
        if args.reference is None:
            reference = "Asimov"
        else:
            reference = args.reference
        os.system(build_command("13HEP.py", name, base_args + analysis_args))
        os.system(
            build_command(
                "0ZBestSigmas.py",
                name,
                base_args + [f"--analysis HEP", f"--reference {reference}"],
            )
        )
        os.system(
            build_command("13HEPExposurePlot.py", name, base_args + analysis_args)
        )
        os.system(
            build_command("13HEPSignificancePlot.py", name, base_args + analysis_args)
        )

    elif "Sensitivity" in args.analysis:
        os.system(
            build_command(
                "14SensitivityBackgroundTemplate.py", name, base_args + analysis_args
            )
        )
        os.system(
            build_command(
                "14SensitivitySignalTemplate.py",
                name,
                base_args + analysis_args + ["--no-test"],
            )
        )
        os.system(
            build_command(
                "14Sensitivity.py",
                name,
                base_args
                + [
                    f"--reference_analysis {args.reference_analysis}",
                    f"--signal_uncertainty {args.signal_uncertainty}",
                    f"--background_uncertainty {args.background_uncertainty}",
                ],
            )
        )
        os.system(
            build_command(
                "14SensitivityContourPlot.py",
                name,
                base_args
                + [
                    f"--signal_uncertainty {args.signal_uncertainty}",
                    f"--background_uncertainty {args.background_uncertainty}",
                ],
            )
        )
