"""
run_presentations.py — Regenerate all SOLAR presentation decks
==============================================================
Standalone entrypoint that runs all presentation scripts without
touching any analysis computation. Safe to run at any time after
the analysis pipeline has produced its outputs.

Scripts run (in order):
  1. src/tools/presentations/reference.py     (config/folder-independent)
  2. src/tools/presentations/truth.py         (config/folder-independent)
  3. src/tools/presentations/daynight.py      (per folder × energy)
  4. src/tools/presentations/hep.py           (per folder × energy)
  5. src/tools/presentations/sensitivity.py   (per folder × energy)

Run examples
------------
  # Regenerate all decks (default: Truncated folder, SolarEnergy, PDF on):
  python3 src/tools/run_presentations.py

  # Multiple folders:
  python3 src/tools/run_presentations.py --folder Truncated Nominal

  # Skip PDF export (faster):
  python3 src/tools/run_presentations.py --no-pdf

  # Skip reference + truth decks:
  python3 src/tools/run_presentations.py --no-global_decks

  # Specific analysis only:
  python3 src/tools/run_presentations.py --analysis HEP DayNight
"""

import os
import sys
import subprocess
from typing import List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from lib import *


ENERGY_CHOICES = [
    "SignalParticleK",
    "ClusterEnergy",
    "TotalEnergy",
    "SelectedEnergy",
    "SolarEnergy",
]

ANALYSIS_PRESENTATION_SCRIPTS = {
    "DayNight":    "src/tools/presentations/daynight.py",
    "HEP":         "src/tools/presentations/hep.py",
    "Sensitivity": "src/tools/presentations/sensitivity.py",
}

GLOBAL_PRESENTATION_SCRIPTS = [
    "src/tools/presentations/reference.py",
    "src/tools/presentations/truth.py",
]


# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Regenerate all SOLAR presentation Markdown/PDF decks"
)
parser.add_argument(
    "--analysis",
    nargs="+",
    type=str,
    choices=["DayNight", "HEP", "Sensitivity"],
    default=["DayNight", "HEP", "Sensitivity"],
    help="Which analysis decks to regenerate (default: all three)",
)
parser.add_argument(
    "--folder",
    nargs="+",
    type=str,
    choices=["Truncated", "Nominal", "Reduced"],
    default=["Truncated"],
    help="Folder scope(s) for analysis decks (default: Truncated)",
)
parser.add_argument(
    "--energy",
    nargs="+",
    type=str,
    choices=ENERGY_CHOICES,
    default=["SolarEnergy"],
    help="Energy variable(s) for analysis decks (default: SolarEnergy)",
)
parser.add_argument(
    "--pdf",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Export PDF alongside Markdown for each deck (default: on)",
)
parser.add_argument(
    "--global_decks",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Run config/folder-independent decks (reference + truth). Pass --no-global-decks to skip.",
)
parser.add_argument(
    "--verbose",
    type=str,
    choices=["quiet", "normal", "verbose"],
    default="normal",
)

args = parser.parse_args()
configure_global_logging(verbose=args.verbose)

pdf_flag = "--pdf" if args.pdf else "--no-pdf"


# ── Subprocess helpers ────────────────────────────────────────────────────────

def _subprocess_env() -> dict:
    env = os.environ.copy()
    chrome_exe = os.path.join(str(root), ".chrome", "chrome-linux64", "chrome")
    if os.path.isfile(chrome_exe) and "BROWSER_PATH" not in env:
        env["BROWSER_PATH"] = chrome_exe
    env["SOLAR_VERBOSE"] = str({"quiet": 0, "normal": 1, "verbose": 2}.get(args.verbose, 1))
    return env


def run_script(command: List[str], label: Optional[str] = None):
    rendered = " ".join(command)
    rprint(f"\n[green][CMD][/green] {rendered}")
    completed = subprocess.run(command, check=False, env=_subprocess_env())
    if completed.returncode != 0:
        script_label = label or os.path.basename(command[1])
        rprint(
            f"[yellow][WARNING][/yellow] {script_label} exited with code "
            f"{completed.returncode} — continuing."
        )


def run_global(script_name: str):
    run_script(
        ["python3", f"{root}/{script_name}", pdf_flag],
        label=os.path.basename(script_name),
    )


def run_analysis(script_name: str, energy: str, folder: str):
    run_script(
        ["python3", f"{root}/{script_name}",
         "--energy", energy,
         "--folder", folder.lower(),
         pdf_flag],
        label=os.path.basename(script_name),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

rprint("\n[bold]Presentation regeneration[/bold]")
rprint(f"  Analysis  : {', '.join(args.analysis)}")
rprint(f"  Folders   : {', '.join(args.folder)}")
rprint(f"  Energy    : {', '.join(args.energy)}")
rprint(f"  PDF       : {args.pdf}")
rprint(f"  Global    : {args.global_decks}")

if args.global_decks:
    rprint("\n[bold cyan]── Global decks (reference + truth) ──[/bold cyan]")
    for script in GLOBAL_PRESENTATION_SCRIPTS:
        run_global(script)

for analysis_name in args.analysis:
    script_name = ANALYSIS_PRESENTATION_SCRIPTS.get(analysis_name)
    if script_name is None:
        continue
    rprint(f"\n[bold cyan]── {analysis_name} decks ──[/bold cyan]")
    for energy in args.energy:
        for folder in args.folder:
            run_analysis(script_name, energy, folder)

rprint("\n[bold green]Presentation regeneration complete.[/bold green]")
