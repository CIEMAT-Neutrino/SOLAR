"""
run_studies.py — Chapter 9 sensitivity study orchestrator
==========================================================
Calls run_sensitivity.py for each study variant with --study_label and
--no-fiducialization / --skip_best_cuts to isolate study outputs without
touching intermediate or final files from the main analysis.

Study groups
------------
  metric    9.1.1  Raw/Smoothed histogram metric comparison
  unc       9.1.2  Signal/background uncertainty impacts
  energy    9.2.1  Energy variable (SignalParticleK, MainK)
  fiduc     9.2.2  Fiducialization (Nominal/Reduced/Truncated folders)
  charge    9.2.3  Charge threshold scan (replaces NHits/AdjCl axes)
  bkgmodel  9.2.4  Background model normalization (Nominal/Reduced folders)

Usage
-----
  python3 src/pipelines/run_studies.py --study unc
  python3 src/pipelines/run_studies.py --study energy charge --config hd_1x2x6_centralAPA
  python3 src/pipelines/run_studies.py --all
  python3 src/pipelines/run_studies.py --all --dry_run
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional
from typing_extensions import TypedDict, NotRequired

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from lib import root, load_analysis_info
from rich import print as rprint

_analysis_info = load_analysis_info(str(root))
_data_root     = Path(_analysis_info["PATH"])
_background_components: List[str] = list(
    _analysis_info.get("BACKGROUND_SAMPLES", {}).get("default", [])
)

PIPELINE_SCRIPT = "src/pipelines/run_sensitivity.py"


# ---------------------------------------------------------------------------
# Variant schema
# ---------------------------------------------------------------------------

class StudyVariant(TypedDict):
    skip_rebin:        bool                         # reuse existing Rebin DataFrames
    skip_best_cuts:    bool                         # skip 04_best_cuts.py phase
    label:             NotRequired[Optional[str]]   # None → folder label used instead
    folder:            NotRequired[Optional[str]]   # None → use --folder from CLI
    fiducialization:   NotRequired[bool]            # default False (skip fiducialization)
    energy_override:   NotRequired[Optional[str]]   # single energy replacing CLI --energy
    analysis_override: NotRequired[Optional[List[str]]]  # override --analysis for this variant only
    extra:             NotRequired[List[str]]       # verbatim flags appended last


# ---------------------------------------------------------------------------
# Study variant definitions
# ---------------------------------------------------------------------------

STUDY_VARIANTS: dict[str, list[StudyVariant]] = {
    # 9.1.1 — histogram metric / smoothing comparison
    # Raw vs Smoothed results are part of the default pipeline (--all_metrics).
    # No separate study runs needed — extract directly from default output pkls.
    # 9.1.2 — uncertainty impacts
    "unc": [
        # Signal uncertainty — HEP only (DayNight uses σ_sig=0 by statistical design)
        # Scan bracketing the default 30%: tighter (20%) and looser (40%)
        {"label": "unc_sig20", "skip_rebin": True, "skip_best_cuts": True, "analysis_override": ["HEP"], "extra": ["--signal_uncertainty", "0.20"]},
        {"label": "unc_sig40", "skip_rebin": True, "skip_best_cuts": True, "analysis_override": ["HEP"], "extra": ["--signal_uncertainty", "0.40"]},
        # Background uncertainty — both analyses; effect enters significance when σ_bkg²·N_bkg > 1.
        # Extended scan: systematic dominates only for large backgrounds or high σ_bkg.
        # σ_bkg² · N_bkg > 1  →  N_bkg > 1/σ_bkg²  (6%→278, 10%→100, 20%→25 events)
        {"label": "unc_bkg0",  "skip_rebin": True, "skip_best_cuts": True, "extra": ["--background_uncertainty", "0.00"]},
        {"label": "unc_bkg4",  "skip_rebin": True, "skip_best_cuts": True, "extra": ["--background_uncertainty", "0.04"]},
        {"label": "unc_bkg6",  "skip_rebin": True, "skip_best_cuts": True, "extra": ["--background_uncertainty", "0.06"]},
        {"label": "unc_bkg10", "skip_rebin": True, "skip_best_cuts": True, "extra": ["--background_uncertainty", "0.10"]},
        {"label": "unc_bkg20", "skip_rebin": True, "skip_best_cuts": True, "extra": ["--background_uncertainty", "0.20"]},
    ],
    # 9.2.1 — energy variable: energy_override replaces CLI --energy for this variant
    # fiducialization=True required — Fiducial_Scan.pkl for these energies may not exist
    "energy": [
        {"label": "energy_spk",   "skip_rebin": False, "skip_best_cuts": False, "fiducialization": True, "energy_override": "SignalParticleK"},
        {"label": "energy_maink", "skip_rebin": False, "skip_best_cuts": False, "fiducialization": True, "energy_override": "MainK"},
    ],
    # 9.2.2 — fiducialization (folder provides isolation; no study_label needed)
    "fiduc": [
        {"folder": "Nominal",   "skip_rebin": True, "skip_best_cuts": True},
        {"folder": "Reduced",   "skip_rebin": True, "skip_best_cuts": True},
        {"folder": "Truncated", "skip_rebin": True, "skip_best_cuts": True},
    ],
    # 9.2.3 — charge threshold scan
    # AdjCl energy features are recomputed with AdjClCharge > Q before the Rebin pkl is
    # written, so the energy axis itself reflects the charge cut — not just event selection.
    # SelectedEnergy (= Energy + SelectedAdjClEnergy) is used as the analysis metric:
    # it is a direct calorimetric sum that needs no BDT retraining.
    "charge": [
        {"label": "charge_Q50",  "skip_rebin": False, "skip_best_cuts": False, "energy_override": "SelectedEnergy", "extra": ["--charge_threshold",  "50"]},
        {"label": "charge_Q100", "skip_rebin": False, "skip_best_cuts": False, "energy_override": "SelectedEnergy", "extra": ["--charge_threshold", "100"]},
        {"label": "charge_Q200", "skip_rebin": False, "skip_best_cuts": False, "energy_override": "SelectedEnergy", "extra": ["--charge_threshold", "200"]},
    ],
    # 9.2.4 — background model normalization (folder provides isolation)
    "bkgmodel": [
        {"folder": "Nominal", "skip_rebin": True, "skip_best_cuts": True},
        {"folder": "Reduced", "skip_rebin": True, "skip_best_cuts": True},
    ],
    # 9.2.5 — oscillation best-fit point: solar (Δm²₂₁=6e-5) vs reactor (Δm²₂₁=7.54e-5)
    # Solar variant reuses nominal Rebin pkls (skip_rebin=True); reactor variant regenerates
    # Rebin pkls with the reactor dm2 point (skip_rebin=False) using a labeled filename to
    # avoid overwriting the nominal solar-dm2 Rebin.  Background Rebin pkls are dm2-independent
    # (backgrounds use Truth weights, not oscillation weights) and are reused unchanged.
    "oscpoint": [
        {"label": "oscpoint_solar",   "skip_rebin": True,  "skip_best_cuts": True},
        {"label": "oscpoint_reactor", "skip_rebin": False, "skip_best_cuts": False, "extra": ["--dm2", "7.54e-5"]},
    ],
}

ALL_GROUPS: list[str] = list(STUDY_VARIANTS.keys())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Run Chapter 9 sensitivity study variants without overwriting main analysis files.",
    formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=36, width=120),
)
parser.add_argument("--study",    nargs="+", choices=ALL_GROUPS, default=None, help="Study group(s) to run")
parser.add_argument("--all",      action="store_true",                          help="Run all study groups")
parser.add_argument("--dry_run",  action="store_true",                          help="Print commands without executing")
# Pipeline passthrough args — mirror run_sensitivity.py defaults
parser.add_argument("--config",              nargs="+", default=["hd_1x2x6_centralAPA"])
parser.add_argument("--signals",               nargs="+", default=["marley"],
                    help="Signal source name(s). Background components auto-discovered from analysis config.")
parser.add_argument("--analysis",            nargs="+", choices=["DayNight", "HEP", "Sensitivity"], default=["DayNight", "HEP", "Sensitivity"])
parser.add_argument("--folder",              nargs="+", default=["Truncated"],  help="Default folder(s) for groups that don't fix their own folder")
parser.add_argument("--energy",              nargs="+", default=["SolarEnergy"], help="Default energy variable(s); energy group overrides per variant")
parser.add_argument("--oscillation_backend", choices=["file", "prob3", "nufast"], default="nufast")
parser.add_argument("--verbose",             choices=["quiet", "normal", "verbose"], default="normal")
parser.add_argument("--rewrite",        dest="rewrite",        action=argparse.BooleanOptionalAction, default=True, help="Overwrite existing pkl outputs (default: True)")
parser.add_argument("--no-computation", dest="no_computation", action="store_true", help="Pass --no-computation to run_sensitivity.py (plots only, skip all computation)")
parser.add_argument("--no-plot",        dest="no_plot",        action="store_true", help="Pass --no-plot to run_sensitivity.py (skip figure output)")

args = parser.parse_args()

if not args.study and not args.all:
    parser.error("Provide --study <group> [<group> ...] or --all")

selected_groups: list[str] = ALL_GROUPS if args.all else args.study


# ---------------------------------------------------------------------------
# Prerequisite checks
# Path roots come from load_analysis_info — no PNFS strings hardcoded here.
# Naming conventions (FIDUCIAL/, signal/, background/, *_Fiducial_Scan.pkl,
# *_Rebin.pkl) mirror what 01_fiducialize.py and 03_analysis.py produce.
# ---------------------------------------------------------------------------

def _all_fiducial_exist(configs: List[str], folders: List[str],
                        names: List[str], energies: List[str]) -> bool:
    """True only if every (config, folder, name, energy) has a Fiducial_Scan pkl."""
    return all(
        (_data_root / "FIDUCIAL" / folder.lower() / config / name
         / f"{energy}_Fiducial_Scan.pkl").exists()
        for config   in configs
        for folder   in folders
        for name     in names
        for energy   in energies
    )


def _all_rebin_exist(configs: List[str], folders: List[str], names: List[str],
                     energies: List[str], analyses: List[str]) -> bool:
    """True only if every (config, folder, name, energy, analysis) has a Rebin pkl.

    Checks both signal/ and background/ subtrees so the orchestrator does not
    need to know which names map to which directory kind.
    """
    return all(
        any(
            (_data_root / kind / folder.lower() / analysis.upper()
             / config / name / f"{energy}_Rebin.pkl").exists()
            for kind in ("signal", "background")
        )
        for config   in configs
        for folder   in folders
        for name     in names
        for energy   in energies
        for analysis in analyses
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_command(cmd: List[str]) -> None:
    rendered = " ".join(str(a) for a in cmd)
    rprint(f"\n[green][STUDY-CMD][/green] {rendered}")
    if not args.dry_run:
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise SystemExit(
                f"Command failed (exit {result.returncode}):\n{rendered}"
            )


def _base_pipeline_args(
    folder: Optional[str],
    energy: List[str],
    analysis: Optional[List[str]] = None,
) -> List[str]:
    """Config/name/analysis/folder/energy/backend args shared across all variants."""
    effective_folder   = [folder] if folder is not None else args.folder
    effective_analysis = analysis if analysis is not None else list(args.analysis)
    base = [
        "--config",              *args.config,
        "--signals",             *args.signals,
        "--analysis",            *effective_analysis,
        "--folder",              *effective_folder,
        "--energy",              *energy,
        "--oscillation_backend", args.oscillation_backend,
        "--verbose",             args.verbose,
    ]
    if args.no_computation:
        base.append("--no-computation")
    if args.no_plot:
        base.append("--no-plot")
    if args.rewrite:
        base.append("--rewrite")
    else:
        base.append("--no-rewrite")
    return base


def _run_variant(group: str, variant: StudyVariant) -> None:
    label             = variant.get("label")
    folder            = variant.get("folder")
    energy_override   = variant.get("energy_override")
    analysis_override = variant.get("analysis_override")
    fiducialization   = variant.get("fiducialization", False)
    skip_rebin        = variant["skip_rebin"]
    skip_best_cuts    = variant["skip_best_cuts"]

    energy            = [energy_override] if energy_override else args.energy
    folders           = [folder] if folder is not None else args.folder
    effective_analysis = analysis_override if analysis_override is not None else list(args.analysis)

    # Safety layer: override skip flags when the expected prerequisite files are
    # absent (e.g. first run of a new config).  Path root from load_analysis_info
    # so these checks survive PNFS location changes without modification here.
    if not fiducialization and not _all_fiducial_exist(args.config, folders, args.signals, energy):
        rprint(f"  [yellow][STUDY-WARN][/yellow] Fiducial_Scan pkls missing — enabling fiducialization stage")
        fiducialization = True

    if skip_rebin and not _all_rebin_exist(args.config, folders, list(args.signals) + _background_components, energy, effective_analysis):
        rprint(f"  [yellow][STUDY-WARN][/yellow] Rebin pkls missing — enabling rebin stage")
        skip_rebin = False

    cmd: List[str] = [
        "python3", f"{root}/{PIPELINE_SCRIPT}",
        *_base_pipeline_args(folder, energy, analysis=effective_analysis),
    ]

    if not fiducialization:
        cmd.append("--no-fiducialization")
    if skip_best_cuts:
        cmd.append("--skip_best_cuts")
    if skip_rebin:
        cmd.append("--no-rebin")
    if label:
        cmd += ["--study_label", label]

    cmd += variant.get("extra", [])

    variant_id = label or f"folder={folder}"
    rprint(f"  [bold]→[/bold] [{group}] {variant_id}")
    _run_command(cmd)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

for group in selected_groups:
    variants = STUDY_VARIANTS[group]
    rprint(
        f"\n[bold cyan]══ Study group: {group}  ({len(variants)} variant{'s' if len(variants) != 1 else ''}) ══[/bold cyan]"
    )
    for variant in variants:
        _run_variant(group, variant)

rprint("\n[bold green]All selected study groups complete.[/bold green]")
