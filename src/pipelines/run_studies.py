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

  # Run studies in draft mode (coarse grid for fast validation):
  python3 src/pipelines/run_studies.py --study unc --draft
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import IO, List, Optional
from typing_extensions import TypedDict, NotRequired
from concurrent.futures import ThreadPoolExecutor, as_completed

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
# NOTE: --exposure and --secondary-exposure are intentionally NOT included here.
# Secondary exposure runs are only for the default sensitivity pipeline, not studies.
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
parser.add_argument("--fit_background", dest="fit_background", action="store_true",
                   help="Pass --fit_background to run_sensitivity.py (LEGACY: fit background normalization as free parameter). Default is --no-fit_background.")
parser.add_argument("--legacy_background_penalty", dest="legacy_background_penalty", action="store_true",
                   help="Pass --legacy_background_penalty to run_sensitivity.py to use the old constant penalty for background uncertainty. Default uses profiling.")
parser.add_argument("--skip-templates", action="store_true", default=False,
                   help="Force --skip-templates for ALL variants (overrides per-variant skip_templates setting). Use when templates are already computed and you want to skip regeneration for all studies.")
parser.add_argument("--draft", action=argparse.BooleanOptionalAction, default=False,
                   help="Pass --draft to run_sensitivity.py to enable draft mode (coarse grid) for Sensitivity analysis.")
parser.add_argument("--parallel", type=int, default=1, metavar="N",
                   help="Run N variants in parallel using subprocess. Use --parallel 2-4 for testing. Default is 1 (sequential).")
parser.add_argument("--quiet", action="store_true", default=False,
                   help="Suppress all output except errors and warnings. Useful for parallel runs to reduce log clutter.")
parser.add_argument("--log_file",       dest="log_file",       default=None,        help="Tee all subprocess output to this file (appended); useful for post-run review")
parser.add_argument("--nhits", type=int, default=None,
                   help="NHits cut value (passed to run_sensitivity.py)")
parser.add_argument("--ophits", type=int, default=None,
                   help="OpHits cut value (passed to run_sensitivity.py)")
parser.add_argument("--adjcls", type=int, default=None,
                   help="AdjCl cut value (passed to run_sensitivity.py)")
parser.add_argument("--flyweight", action=argparse.BooleanOptionalAction, default=False,
                   help="Pass --flyweight to run_sensitivity.py for flyweight mode in Sensitivity analysis")
parser.add_argument("--no-templates", dest="skip_templates", action="store_true",
                   help="Alias for --skip-templates: skip template computation")

args = parser.parse_args()

if not args.study and not args.all:
    parser.error("Provide --study <group> [<group> ...] or --all")

# Quiet mode: suppress rich output
if args.quiet:
    import logging
    logging.basicConfig(level=logging.WARNING)
    # Redirect rich print to devnull
    import io
    from rich.console import Console
    rprint = Console(file=io.StringIO(), force_terminal=False, width=120).print

selected_groups: list[str] = ALL_GROUPS if args.all else args.study

_variant_filter: Optional[set] = set(args.variant) if args.variant else None


def _matches_variant_filter(variant: StudyVariant) -> bool:
    if _variant_filter is not None:
        label  = variant.get("label")
        folder = variant.get("folder")
        if not ((label is not None and label in _variant_filter) or \
                (folder is not None and folder in _variant_filter)):
            return False
    
    # Filter by analysis: skip variants with analysis_override that doesn't match user's --analysis
    analysis_override = variant.get("analysis_override")
    if analysis_override is not None:
        # Check if any of the user's requested analyses are in the variant's override
        user_analyses = set(a.upper() for a in args.analysis)
        variant_analyses = set(a.upper() for a in analysis_override)
        if not user_analyses.intersection(variant_analyses):
            # Variant requires specific analyses that user didn't request
            return False
    
    return True


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


def _all_templates_exist(configs: List[str], folders: List[str], names: List[str],
                         energies: List[str], analyses: List[str], study_label: Optional[str] = None) -> bool:
    """True only if every (config, folder, name, energy, analysis) has both background and signal template pkls.

    Template files are written as {config}_{name}_{energy}_Template_{background|signal}{_study_label}.pkl
    in the SENSITIVITY/ folder tree. This check verifies that both template types exist
    for all combinations, optionally with a study label suffix.
    """
    suffix = f"_{study_label}" if study_label else ""
    return all(
        all(
            (_data_root / "SENSITIVITY" / folder.lower() / analysis.upper()
             / config / name / f"{config}_{name}_{energy}_Template_{template_type}{suffix}.pkl").exists()
            for template_type in ("background", "signal")
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
    if args.draft:
        base.append("--draft")
    if args.nhits is not None:
        base.extend(["--nhits", str(args.nhits)])
    if args.ophits is not None:
        base.extend(["--ophits", str(args.ophits)])
    if args.adjcls is not None:
        base.extend(["--adjcls", str(args.adjcls)])
    return base


def _run_variant(group: str, variant: StudyVariant) -> None:
    """Run a single variant (sequential execution)."""
    variant_id = variant.get("label") or f"folder={variant.get('folder')}"
    rprint(f"  [bold]→[/bold] [{group}] {variant_id}")
    cmd = _build_variant_command(group, variant)
    _run_command(cmd, log=_log)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _variant_priority(variant: StudyVariant) -> int:
    """Priority for variant ordering: background uncertainty variants run first."""
    label = variant.get("label", "")
    # Background uncertainty variants (unc_bkg*, fit_background_unc_bkg*) get highest priority (0)
    if "unc_bkg" in label or "fit_background_unc_bkg" in label:
        return 0
    # Other uncertainty variants get next priority (1)
    if "unc_" in label or "fit_background" in label:
        return 1
    # All other variants get lowest priority (2)
    return 2


_log: Optional[IO[str]] = open(args.log_file, "a") if args.log_file else None
if _log:
    import datetime
    _log.write(f"\n\n{'='*72}\n")
    _log.write(f"# run_studies.py  {datetime.datetime.now().isoformat(timespec='seconds')}\n")
    _log.write(f"# argv: {' '.join(sys.argv)}\n")
    _log.write(f"{'='*72}\n")
    _log.flush()

def _build_variant_command(group: str, variant: StudyVariant) -> List[str]:
    """Build the command list for a variant without executing it. Used for parallel execution."""
    label                = variant.get("label")
    folder               = variant.get("folder")
    energy_override      = variant.get("energy_override")
    analysis_override    = variant.get("analysis_override")
    fiducialization      = variant.get("fiducialization", False)
    ignore_energy_window = variant.get("ignore_energy_window", False)
    skip_best_sigmas     = variant.get("skip_best_sigmas", False)
    skip_rebin           = variant["skip_rebin"]
    skip_best_cuts       = variant["skip_best_cuts"]
    variant_skip_templates = variant.get("skip_templates", False)

    # Use global flag OR per-variant flag
    skip_templates = args.skip_templates or variant_skip_templates

    energy            = [energy_override] if energy_override else args.energy
    folders           = [folder] if folder is not None else args.folder
    effective_analysis = analysis_override if analysis_override is not None else list(args.analysis)

    # Check prerequisites - return None if we need to enable stages
    if not fiducialization and not _all_fiducial_exist(args.config, folders, args.signals, energy):
        fiducialization = True

    if skip_rebin and not _all_rebin_exist(args.config, folders, list(args.signals) + _background_components, energy, effective_analysis):
        skip_rebin = False

    # Check templates
    sensitivity_analyses = [a for a in effective_analysis if a.upper() == "SENSITIVITY"]
    if skip_templates and sensitivity_analyses:
        if not _all_templates_exist(args.config, folders, list(args.signals) + _background_components, energy, sensitivity_analyses, label):
            skip_templates = False

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
    if skip_templates:
        cmd.append("--skip-templates")
    if label:
        cmd += ["--study_label", label]
    if ignore_energy_window:
        cmd.append("--ignore_energy_window")
    if skip_best_sigmas:
        cmd.append("--skip_best_sigmas")
    if args.fit_background:
        cmd.append("--fit_background")
    else:
        cmd.append("--no-fit_background")
    if args.legacy_background_penalty:
        cmd.append("--legacy_background_penalty")
    if args.flyweight:
        cmd.append("--flyweight")

    cmd += variant.get("extra", [])
    return cmd


def _run_single_variant(group: str, variant: StudyVariant) -> None:
    """Run a single variant with logging."""
    variant_id = variant.get("label") or f"folder={variant.get('folder')}"
    rprint(f"  [bold]→[/bold] [{group}] {variant_id}")
    cmd = _build_variant_command(group, variant)
    _run_command(cmd, log=_log)


try:
    if args.parallel > 1:
        rprint(f"\n[bold yellow]Running with parallelism={args.parallel}[/bold yellow]")
        # Collect all variants across all groups
        all_variants = []
        for group in selected_groups:
            variants = STUDY_VARIANTS[group]
            active   = [v for v in variants if _matches_variant_filter(v)]
            if not active:
                rprint(f"\n[yellow]══ Study group: {group} — no variants match --variant filter, skipping ══[/yellow]")
                continue
            active.sort(key=_variant_priority)
            skipped  = len(variants) - len(active)
            suffix   = f"  ({skipped} filtered out)" if skipped else ""
            rprint(
                f"\n[bold cyan]══ Study group: {group}  ({len(active)} variant{'s' if len(active) != 1 else ''}{suffix}) ══[/bold cyan]"
            )
            all_variants.extend([(group, v) for v in active])

        # Run variants in parallel batches
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(_run_single_variant, group, variant): (group, variant)
                for group, variant in all_variants
            }
            # Wait for all to complete, raise on first failure
            for future in as_completed(futures):
                future.result()  # This will raise if the task raised
    else:
        # Sequential execution (original behavior)
        for group in selected_groups:
            variants = STUDY_VARIANTS[group]
            active   = [v for v in variants if _matches_variant_filter(v)]
            if not active:
                rprint(f"\n[yellow]══ Study group: {group} — no variants match --variant filter, skipping ══[/yellow]")
                continue
            # Sort variants: background uncertainty variants first, then other uncertainty, then rest
            active.sort(key=_variant_priority)
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
