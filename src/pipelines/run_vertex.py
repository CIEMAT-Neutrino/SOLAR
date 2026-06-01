"""
run_vertex.py — Vertex Reconstruction Diagnostics Pipeline
===========================================================
Runs smearing, vertex, fiducial, and reconstruction diagnostics.
Steps live in src/physics/detector/.
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


def build_command(script_path: str, additional_args: Optional[List[str]] = None) -> List[str]:
    command = ["python3", f"{root}/{script_path}"]
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


# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Run vertex reconstruction diagnostics")
parser.add_argument(
    "--config",
    type=str,
    nargs="+",
    help="Detector configuration(s) to process",
    default=["hd_1x2x6_centralAPA"],
)
parser.add_argument(
    "--name",
    type=str,
    nargs="+",
    help="Sample name(s) to process",
    default=["marley"],
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
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

rprint(f"\n[bold]Vertex reconstruction diagnostics pipeline[/bold]")
rprint(f"  Configs : {args.config}")
rprint(f"  Names   : {args.name}")
rprint(f"  Verbose : {args.verbose}")

for config, name in product(args.config, args.name):
    base_args = ["--config", config, "--name", name, _rw(), _debug()]
    run_python_command(build_command("src/physics/detector/vertex/01_smearing.py",      base_args))
    run_python_command(build_command("src/physics/detector/vertex/02_vertex.py",       base_args))
    run_python_command(build_command("src/physics/detector/vertex/03_fiducial.py",     base_args))
    run_python_command(build_command("src/physics/detector/vertex/04_reconstruction.py", base_args))

rprint(f"\n[bold green]Vertex pipeline complete.[/bold green]")
