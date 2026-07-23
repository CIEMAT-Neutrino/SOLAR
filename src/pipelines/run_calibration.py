"""
run_calibration.py — Detector Calibration & Reconstruction Pipeline
====================================================================
Runs correction → calibration → discrimination → reconstruction → smearing
for a given config/name pair.
Steps live in src/physics/calibration/.
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


def build_command(script_name: str, additional_args: Optional[List[str]] = None) -> List[str]:
    command = ["python3", f"{root}/src/physics/calibration/{script_name}"]
    if additional_args:
        command.extend(str(a) for a in additional_args)
    return command


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


def run_calib_script(script_name: str, additional_args: Optional[List[str]] = None):
    run_python_command(build_command(script_name, additional_args), label=script_name)


# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Run the full detector calibration and reconstruction pipeline",
    formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=32, width=120),
)
parser.add_argument(
    "--config",
    type=str,
    help="Detector configuration to process",
    default="hd_1x2x6_centralAPA",
)
parser.add_argument(
    "--name",
    type=str,
    help="Sample name to process",
    default="marley",
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

_rw    = lambda: "--rewrite" if args.rewrite else "--no-rewrite"
_debug = lambda: "--debug"   if args.verbose == "verbose" else "--no-debug"

base_args = ["--config", args.config, "--name", args.name, _rw(), _debug()]

rprint(f"\n[bold]Calibration pipeline[/bold]")
rprint(f"  Config  : {args.config}")
rprint(f"  Name    : {args.name}")
rprint(f"  Rewrite : {args.rewrite}")
rprint(f"  Verbose : {args.verbose}")

run_calib_script("01_correction.py",     base_args)
run_calib_script("02_calibration.py",    base_args)
run_calib_script("03_discrimination.py", base_args)
run_calib_script("04_reconstruction.py", base_args)
run_calib_script("05_smearing.py",       base_args)

rprint(f"\n[bold green]Calibration pipeline complete: {args.config}/{args.name}[/bold green]")
