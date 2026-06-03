"""
run_truth.py — Truth-Level Background & Oscillation Pipeline
============================================================
Single-entrypoint orchestrator for the src/physics/truth/ preprocessing pipeline.
Mirrors the structure and conventions of src/pipelines/run_sensitivity.py.

Pipeline stages (run in order, each skippable):

  01  oscillations     01_process_oscillation.py   --no-oscillations
  02  spectra          02_background_spectra.py     --no-spectra
  02  external PDF     02_background_spectra.py     --no-external-pdf
  03  surface PDF      03_background_pdf.py         --surface-pdf (off by default; legacy MC)
  05  signal KDE       05_signal_azimuth_kde.py     --signal-kde (off by default; superseded)
  bkg solar bkg        solar_background.py          --no-background
  bkg bkg plot         solar_background_plot.py     (controlled by --no-plot)

Run examples
------------
  # Full pipeline, NuFast oscillations, VD nominal config:
  python3 src/pipelines/run_truth.py \\
      --config vd_1x8x14_3view_30deg_nominal \\
      --oscillation_backend nufast --rewrite

  # Background PDFs only (skip oscillations and signal KDE):
  python3 src/pipelines/run_truth.py \\
      --config hd_1x2x6_centralAPA \\
      --no-oscillations --no-signal-kde

  # Regenerate signal KDE for all folders, skip everything else:
  python3 src/pipelines/run_truth.py \\
      --no-oscillations --no-spectra --no-external-pdf \\
      --no-surface-pdf --no-background

  # Refresh all oscillation pkl after updating physics.json best-fit params:
  python3 src/pipelines/run_truth.py \\
      --oscillation_backend nufast --no-spectra --no-external-pdf \\
      --no-surface-pdf --no-signal-kde --no-background --rewrite
"""

import os
import sys
import subprocess
from typing import List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *


# ── Particle classification ──────────────────────────────────────────────────
_BACKGROUND_TYPES = {"gamma", "neutron", "alpha", "radiological"}

def _is_background(name: str) -> bool:
    return name.split("_")[0].lower() in _BACKGROUND_TYPES

def _is_signal(name: str) -> bool:
    return not _is_background(name)


# ── Subprocess helpers (mirrors run_sensitivity.py) ──────────────────────────
def build_command(script_name: str, additional_args: Optional[List[str]] = None) -> List[str]:
    command = ["python3", f"{root}/src/physics/truth/{script_name}"]
    if additional_args:
        command.extend(str(arg) for arg in additional_args)
    return command


def _subprocess_env() -> dict:
    """Propagate BROWSER_PATH and SOLAR_VERBOSE to child processes."""
    env = os.environ.copy()
    chrome_exe = os.path.join(str(root), ".chrome", "chrome-linux64", "chrome")
    if os.path.isfile(chrome_exe) and "BROWSER_PATH" not in env:
        env["BROWSER_PATH"] = chrome_exe
    env["SOLAR_VERBOSE"] = str({"quiet": 0, "normal": 1, "verbose": 2}.get(args.verbose, 1))
    return env


def run_python_command(
    command: List[str],
    label: Optional[str] = None,
    stop_on_error: bool = True,
):
    rendered = " ".join(command)
    rprint(f"\n[green][CMD][/green] {rendered}")
    completed = subprocess.run(command, check=False, env=_subprocess_env())
    if completed.returncode != 0 and stop_on_error:
        script_label = label or os.path.basename(command[1])
        raise SystemExit(
            f"Pipeline stopped because {script_label} failed "
            f"(exit {completed.returncode}).\nCommand: {rendered}"
        )


def run_truth_script(script_name: str, additional_args: Optional[List[str]] = None):
    run_python_command(
        build_command(script_name, additional_args), label=script_name
    )


# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Run the full truth-level background and oscillation pipeline"
)

# ── Scope ────────────────────────────────────────────────────────────────────
# Configs and particle names are read from analysis/backgrounds.json
# (DEFAULT_CONFIGS and TRUTH_PIPELINE.BACKGROUND_NAMES / SIGNAL_NAMES).
# No CLI args for these — edit the JSON to change the defaults.
# ── Oscillation backend ──────────────────────────────────────────────────────
parser.add_argument(
    "--oscillation_backend",
    type=str,
    choices=["file", "prob3", "nufast"],
    default="nufast",
    help=(
        "'file': rebin existing ROOT oscillogram files. "
        "'prob3'/'nufast': generate templates from first-principles backends."
    ),
)
parser.add_argument(
    "--save_raw",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Save high-resolution raw pkl alongside rebinned pkl (prob3/nufast only).",
)

# ── Stage skip flags ─────────────────────────────────────────────────────────
parser.add_argument(
    "--oscillations",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Run 01_process_oscillation.py. Pass --oscillations to enable.",
)
parser.add_argument(
    "--spectra",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Run 02_background_spectra.py. Pass --no-spectra to skip.",
)
parser.add_argument(
    "--pdf",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "Run 03_background_pdf.py. Backend (truth/legacy) set by "
        "TRUTH_PIPELINE.PDF_BACKEND in backgrounds.json. Pass --no-pdf to skip."
    ),
)
parser.add_argument(
    "--signal_kde",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Run 05_signal_azimuth_kde.py (legacy pre-generation stage). "
        "Off by default — lib/weights.py now computes signal KDE on-the-fly "
        "from the oscillation pkl at the best-fit point, making pre-generation "
        "unnecessary for all three oscillation backends. "
        "Pass --signal-kde only if you need the per-nadir-bin KDE files explicitly."
    ),
)
parser.add_argument(
    "--background",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Run solar_background.py. Pass --no-background to skip.",
)
parser.add_argument(
    "--plot",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Run 0ZSolarBackgroundPlot.py. Pass --no-plot to skip.",
)

# ── Global flags ─────────────────────────────────────────────────────────────
parser.add_argument(
    "--rewrite",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "Overwrite existing output files in every stage. "
        "Off by default — stages skip files that already exist."
    ),
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

# ── Resolve config list and names from JSON ────────────────────────────────
analysis_info    = load_analysis_info(str(root))
truth_pipeline   = analysis_info.get("TRUTH_PIPELINE", {})
configs          = analysis_info.get("DEFAULT_CONFIGS", [])
background_names = truth_pipeline.get("BACKGROUND_NAMES", ["gamma", "neutron"])
signal_names     = truth_pipeline.get("SIGNAL_NAMES", ["marley"])

if not configs:
    raise SystemExit(
        "DEFAULT_CONFIGS not set in analysis/backgrounds.json."
    )

# Common flag helpers
def _rw()    -> str: return "--rewrite" if args.rewrite else "--no-rewrite"
def _debug() -> str: return "--debug"   if args.verbose == "verbose" else "--no-debug"
def _plot()  -> str: return "--plot"    if args.plot    else "--no-plot"


# ── Stage functions ───────────────────────────────────────────────────────────

def run_oscillation_stage():
    """0X — generate or rebin oscillation probability templates."""
    cmd_args = [
        "--backend", args.oscillation_backend,
        _rw(), _debug(),
    ]
    if not args.save_raw:
        cmd_args.append("--no-save-raw")
    run_truth_script("01_process_oscillation.py", cmd_args)


def run_spectra_stage():
    """0Y — load truth-level flux spectra; runs over all configs/names internally."""
    run_truth_script(
        "02_background_spectra.py",
        [_rw(), _debug(), _plot()],
    )


def run_pdf_stage():
    """0Y — build momentum PDFs; backend and names read from JSON internally."""
    run_truth_script(
        "03_background_pdf.py",
        [_rw(), _debug(), _plot()],
    )


def run_signal_kde_stage(config: str, sig_names: List[str]):
    """0Y — build per-nadir_slice oscillation-weighted signal KDEs.

    KDE files are folder-independent: the oscillation-weighted energy PDF is a
    theoretical quantity unaffected by fiducialization. The same pkl is reused
    for Nominal/Reduced/Truncated at analysis time.
    """
    if not sig_names:
        rprint(
            f"[cyan][INFO][/cyan] No signal names for KDE stage ({config}) — skipping."
        )
        return
    for name in sig_names:
        run_truth_script(
            "05_signal_azimuth_kde.py",
            ["--config", config, "--name", name, _rw(), _debug()],
        )


def run_solar_background_stage(config: str):
    """0Z — aggregate background energy distributions and save DataFrame."""
    names_arg = list(dict.fromkeys(["all"] + background_names))
    run_truth_script(
        "solar_background.py",
        ["--config", config, "--names", *names_arg, _rw(), _debug()],
    )


def run_background_plot_stage(config: str):
    """0Z — plot saved background energy distributions."""
    run_truth_script(
        "solar_background_plot.py",
        ["--config", config, _rw(), _debug()],
    )


# ── Summary ───────────────────────────────────────────────────────────────────
rprint(f"\n[bold]Truth pipeline configuration[/bold]")
rprint(f"  Configs          : {configs}")
rprint(f"  Background names : {background_names}")
rprint(f"  Signal names     : {signal_names}")
rprint(f"  Osc. backend     : {args.oscillation_backend}")
rprint(f"  Rewrite          : {args.rewrite}")
_pdf_backend = analysis_info.get("TRUTH_PIPELINE", {}).get("PDF_BACKEND", "truth")
rprint(
    f"\n  Stages enabled   : "
    + ", ".join(
        stage
        for stage, enabled in [
            ("oscillations", args.oscillations),
            ("spectra",      args.spectra),
            (f"pdf({_pdf_backend})", args.pdf),
            ("signal-kde",   args.signal_kde),
            ("background",   args.background),
            ("plot",         args.plot),
        ]
        if enabled
    )
)


# ═════════════════════════════════════════════════════════════════════════════
# Stage 0X: Oscillation templates (config-independent, run once)
# ═════════════════════════════════════════════════════════════════════════════
if args.oscillations:
    rprint("\n[bold cyan]── Stage 0X: Oscillation templates ──[/bold cyan]")
    run_oscillation_stage()
else:
    rprint("\n[cyan][SKIP][/cyan] Stage 0X: oscillations (pass --oscillations to enable)")


# ═════════════════════════════════════════════════════════════════════════════
# Stage 0Y: Spectra + PDFs (all configs iterated internally by each script)
# ═════════════════════════════════════════════════════════════════════════════
if args.spectra:
    rprint("\n[bold cyan]── Stage 0Y: Background spectra (all configs) ──[/bold cyan]")
    run_spectra_stage()
else:
    rprint("[cyan][SKIP][/cyan] Stage 0Y: spectra (--no-spectra)")

if args.pdf:
    rprint(f"\n[bold cyan]── Stage 0Y: Background PDFs ({_pdf_backend} backend, all configs) ──[/bold cyan]")
    run_pdf_stage()
else:
    rprint("[cyan][SKIP][/cyan] Stage 0Y: PDFs (--no-pdf)")


# ═════════════════════════════════════════════════════════════════════════════
# Stages 0Y signal KDE + 0Z: per-config
# ═════════════════════════════════════════════════════════════════════════════
for config in configs:
    rprint(f"\n[bold magenta]{'═'*60}[/bold magenta]")
    rprint(f"[bold magenta]  Config: {config}[/bold magenta]")
    rprint(f"[bold magenta]{'═'*60}[/bold magenta]")

    # ── 0Y: Signal nadir KDE (folder-independent) ───────────────────────
    if args.signal_kde:
        rprint("\n[bold cyan]── Stage 0Y: Signal KDE ──[/bold cyan]")
        run_signal_kde_stage(config, signal_names)
    else:
        rprint("[cyan][SKIP][/cyan] Stage 0Y: signal KDE (--no-signal-kde)")

    # ── 0Z: Background energy distribution ────────────────────────────────
    if args.background:
        rprint("\n[bold cyan]── Stage 0Z: Solar background distribution ──[/bold cyan]")
        run_solar_background_stage(config)
    else:
        rprint("[cyan][SKIP][/cyan] Stage 0Z: background (--no-background)")

    # ── 0Z: Plot ──────────────────────────────────────────────────────────
    if args.plot:
        rprint("\n[bold cyan]── Stage 0Z: Background plot ──[/bold cyan]")
        run_background_plot_stage(config)
    else:
        rprint("[cyan][SKIP][/cyan] Stage 0Z: plot (--no-plot)")


rprint(f"\n[bold green]Truth pipeline complete. Configs processed: {configs}[/bold green]")
