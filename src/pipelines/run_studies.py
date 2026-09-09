"""
run_studies.py — Chapter 9 sensitivity study orchestrator
==========================================================
Calls run_sensitivity.py for each study variant with --study_label and
--no-fiducialization / --skip_best_cuts to isolate study outputs without
touching intermediate or final files from the main analysis.

Study groups
------------
  metric      9.1.1  Raw/Smoothed histogram metric comparison
  unc         9.1.2  Signal/background uncertainty impacts
  oscpoint    9.1.3  Oscillation parameter choice (solar vs reactor Δm²₂₁)
  energy      9.2.1  Energy variable (SignalParticleK, MainK)
  fiduc_truth 9.2.2  Truth x-fiducialisation (SignalParticleX/Y/Z vs RecoX/Y/Z)
  fiduc       9.2.3  Fiducialization folder comparison (Nominal/Reduced/Truncated)
  charge      9.2.4  Charge threshold scan (replaces NHits/AdjCl axes)
  bkg_gamma   9.2.5  Background gamma model (ClusterEnergy as calorimetric proxy)
  bkgmodel    9.2.6  Background model normalization (Nominal/Reduced folders)
  membrane_veto      Membrane/endcap optical matches (VD planes 1-4) on vs off

Usage
-----
  python3 src/pipelines/run_studies.py --study unc
  python3 src/pipelines/run_studies.py --study energy charge --config hd_1x2x6_centralAPA
  python3 src/pipelines/run_studies.py --study charge --variant charge_Q500
  python3 src/pipelines/run_studies.py --study unc --variant unc_bkg0 unc_bkg4
  python3 src/pipelines/run_studies.py --all
  python3 src/pipelines/run_studies.py --all --dry_run
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import IO, List, Optional
from typing_extensions import TypedDict, NotRequired

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from lib import root, load_analysis_info
from lib.study import ALL_GROUPS, STUDY_VARIANTS, StudyVariant
from rich import print as rprint

_analysis_info = load_analysis_info(str(root))
_data_root     = Path(_analysis_info["PATH"])
_background_components: List[str] = list(
    _analysis_info.get("BACKGROUND_SAMPLES", {}).get("default", [])
)

PIPELINE_SCRIPT = "src/pipelines/run_sensitivity.py"
# ---------------------------------------------------------------------------
# Variant schema and registry
# ---------------------------------------------------------------------------
# StudyVariant / STUDY_VARIANTS / ALL_GROUPS live in lib/study.py so the plot
# scripts can resolve `--study all` against the same table this orchestrator runs.



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Run Chapter 9 sensitivity study variants without overwriting main analysis files.",
    formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=36, width=120),
)
parser.add_argument("--study",    nargs="+", choices=ALL_GROUPS, default=None, help="Study group(s) to run")
parser.add_argument("--all",      action="store_true",                          help="Run all study groups")
parser.add_argument("--variant",  nargs="+", default=None,                      help="Restrict to specific variant labels/folders within the selected groups (e.g. charge_Q500, unc_bkg0). Omit to run all variants.")
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
parser.add_argument("--no-fit_background", dest="no_fit_background", action="store_true",
                   help="Pass --no-fit_background to run_sensitivity.py (fix background normalization, only fit signal amplitude)")
parser.add_argument("--log_file",       dest="log_file",       default=None,        help="Tee all subprocess output to this file (appended); useful for post-run review")

args = parser.parse_args()

if not args.study and not args.all:
    parser.error("Provide --study <group> [<group> ...] or --all")

selected_groups: list[str] = ALL_GROUPS if args.all else args.study

_variant_filter: Optional[set] = set(args.variant) if args.variant else None


def _matches_variant_filter(variant: StudyVariant) -> bool:
    if _variant_filter is None:
        return True
    label  = variant.get("label")
    folder = variant.get("folder")
    return (label is not None and label in _variant_filter) or \
           (folder is not None and folder in _variant_filter)


# ---------------------------------------------------------------------------
# Prerequisite checks
# Path roots come from load_analysis_info — no PNFS strings hardcoded here.
# Naming conventions (FIDUCIAL/, signal/, background/, *_Fiducial_Scan.pkl,
# *_Rebin.pkl) mirror what 01_fiducialize.py and 03_analysis.py produce.
# ---------------------------------------------------------------------------

def _all_fiducial_exist(configs: List[str], folders: List[str],
                        names: List[str], energies: List[str]) -> bool:
    """True only if every (config, folder, name, energy) has a Fiducial_Scan pkl.

    save_df writes these as {config}_{name}_{energy}_Fiducial_Scan.pkl; omitting that
    prefix here made the check never match, so the safety layer below silently
    re-enabled fiducialization on every study run.
    """
    return all(
        (_data_root / "FIDUCIAL" / folder.lower() / config / name
         / f"{config}_{name}_{energy}_Fiducial_Scan.pkl").exists()
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

    As with the fiducial check, the files carry a {config}_{name}_ prefix; leaving it
    out made this always report missing and forced the rebin stage back on.
    """
    return all(
        any(
            (_data_root / kind / folder.lower() / analysis.upper()
             / config / name / f"{config}_{name}_{energy}_Rebin.pkl").exists()
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

def _run_command(cmd: List[str], log: Optional[IO[str]] = None) -> None:
    rendered = " ".join(str(a) for a in cmd)
    rprint(f"\n[green][STUDY-CMD][/green] {rendered}")
    if log:
        log.write(f"\n### CMD: {rendered}\n")
        log.flush()
    if not args.dry_run:
        if log is None:
            result = subprocess.run(cmd, check=False)
            returncode = result.returncode
        else:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
            log.flush()
            proc.wait()
            returncode = proc.returncode
        if returncode != 0:
            raise SystemExit(
                f"Command failed (exit {returncode}):\n{rendered}"
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
    label                = variant.get("label")
    folder               = variant.get("folder")
    energy_override      = variant.get("energy_override")
    analysis_override    = variant.get("analysis_override")
    fiducialization      = variant.get("fiducialization", False)
    ignore_energy_window = variant.get("ignore_energy_window", False)
    skip_best_sigmas     = variant.get("skip_best_sigmas", False)
    skip_rebin           = variant["skip_rebin"]
    skip_best_cuts       = variant["skip_best_cuts"]

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
    if ignore_energy_window:
        cmd.append("--ignore_energy_window")
    if skip_best_sigmas:
        cmd.append("--skip_best_sigmas")
    if args.no_fit_background:
        cmd.append("--no-fit_background")

    cmd += variant.get("extra", [])

    variant_id = label or f"folder={folder}"
    rprint(f"  [bold]→[/bold] [{group}] {variant_id}")
    _run_command(cmd, log=_log)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

_log: Optional[IO[str]] = open(args.log_file, "a") if args.log_file else None
if _log:
    import datetime
    _log.write(f"\n\n{'='*72}\n")
    _log.write(f"# run_studies.py  {datetime.datetime.now().isoformat(timespec='seconds')}\n")
    _log.write(f"# argv: {' '.join(sys.argv)}\n")
    _log.write(f"{'='*72}\n")
    _log.flush()

try:
    for group in selected_groups:
        variants = STUDY_VARIANTS[group]
        active   = [v for v in variants if _matches_variant_filter(v)]
        if not active:
            rprint(f"\n[yellow]══ Study group: {group} — no variants match --variant filter, skipping ══[/yellow]")
            continue
        skipped  = len(variants) - len(active)
        suffix   = f"  ({skipped} filtered out)" if skipped else ""
        rprint(
            f"\n[bold cyan]══ Study group: {group}  ({len(active)} variant{'s' if len(active) != 1 else ''}{suffix}) ══[/bold cyan]"
        )
        for variant in active:
            _run_variant(group, variant)
finally:
    if _log:
        _log.close()

rprint("\n[bold green]All selected study groups complete.[/bold green]")
