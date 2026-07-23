"""
run_all.py — Full Chain Pipeline
=================================
Runs calibration → preselection → PDS → TPC → vertex for all configs.
For analysis (truth + sensitivity) blocks, use run_truth.py and run_sensitivity.py.
"""

import os
import sys
import subprocess
from typing import List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *


def _subprocess_env() -> dict:
    """Propagate BROWSER_PATH and SOLAR_VERBOSE to child processes."""
    env = os.environ.copy()
    chrome_exe = os.path.join(str(root), ".chrome", "chrome-linux64", "chrome")
    if os.path.isfile(chrome_exe) and "BROWSER_PATH" not in env:
        env["BROWSER_PATH"] = chrome_exe
    env["SOLAR_VERBOSE"] = str({"quiet": 0, "normal": 1, "verbose": 2}.get(args.verbose, 1))
    return env


def run_python_command(command: List[str], label: Optional[str] = None, stop_on_error: bool = True):
    rendered = " ".join(command)
    rprint(f"\n[green][CMD][/green] {rendered}")
    completed = subprocess.run(command, check=False, env=_subprocess_env())
    if completed.returncode != 0 and stop_on_error:
        script_label = label or os.path.basename(command[1])
        raise SystemExit(
            f"Pipeline stopped because {script_label} failed "
            f"(exit {completed.returncode}).\nCommand: {rendered}"
        )


# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Run full analysis chain",
    formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=32, width=120),
)
parser.add_argument(
    "--config",
    type=str,
    nargs="+",
    help="Detector configurations to process",
    default=[
        "hd_1x2x6_centralAPA",
        "hd_1x2x6_lateralAPA",
        "vd_1x8x14_3view_30deg_nominal",
    ],
)
parser.add_argument(
    "--name",
    type=str,
    nargs="+",
    help="Sample names to process",
    default=["marley"],
)
parser.add_argument(
    "--rewrite",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Overwrite existing output files. Pass --no-rewrite to skip files that already exist.",
)
parser.add_argument(
    "--verbose",
    type=str,
    choices=["quiet", "normal", "verbose"],
    default="normal",
    help=(
        "Output verbosity: 'quiet' (errors/warnings only), "
        "'normal' (progress + key results), "
        "'verbose' (all output including per-file debug messages)."
    ),
)

args = parser.parse_args()
configure_global_logging(verbose=args.verbose)

_verbose_flag = ["--verbose", args.verbose]
_rw_flag      = ["--rewrite" if args.rewrite else "--no-rewrite"]

rprint(f"\n[bold]Full chain pipeline[/bold]")
rprint(f"  Configs : {args.config}")
rprint(f"  Names   : {args.name}")
rprint(f"  Verbose : {args.verbose}")

for config, name in product(args.config, args.name):
    base = ["--config", config, "--name", name] + _rw_flag + _verbose_flag
    run_python_command(["python3", f"{root}/src/pipelines/run_calibration.py"]  + base)
    run_python_command(["python3", f"{root}/src/pipelines/run_preselection.py"] + base)
    run_python_command(["python3", f"{root}/src/pipelines/run_pds.py"]          + base)
    run_python_command(["python3", f"{root}/src/pipelines/run_tpc.py"]          + base)
    run_python_command(["python3", f"{root}/src/pipelines/run_vertex.py"]       + base)

rprint(f"\n[bold green]Full chain complete.[/bold green]")
