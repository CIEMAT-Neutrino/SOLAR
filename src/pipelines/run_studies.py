"""
run_studies.py — Chapter 9 sensitivity study orchestrator
==========================================================
Calls run_sensitivity.py for each study variant with --study_label and
--no-fiducialization / --skip_best_cuts to isolate study outputs without
touching intermediate or final files from the main analysis.

Study groups
------------
  metric        9.1.1  Raw/Smoothed histogram metric comparison
  unc           9.1.2  Signal/background uncertainty impacts
  oscpoint      9.1.3  Oscillation parameter choice (solar vs reactor Δm²₂₁)
  fiduc_truth   9.2.1  Truth x-fiducialisation (SignalParticleX/Y/Z and MainX/Y/Z and EndX/Y/Z vs RecoX/Y/Z)
  energy        9.2.2  Energy variable (SignalParticleK, MainK)
  bkg_gamma     9.2.3  Background gamma model (ClusterEnergy and TotalEnergy as calorimetric proxy)
  charge        9.2.4  Charge threshold scan (uses SelectedEnergy for different charge thresholds)
  bkgmodel      9.2.5  Background model normalization (Nominal/Reduced folders)
  membrane_veto 9.2.6  Membrane/endcap optical matches (VD planes 1-4) on vs off
  nuisance             Nuisance-profile decomposition (Sensitivity)
  legacy_fit           Legacy nested-minimiser fit on the default templates (Sensitivity)

Defaults: Sensitivity runs use --flyweight templates and --fit_method pull; the legacy_fit
study is the only legacy-fit run.

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
from lib.study import ALL_GROUPS, STUDY_VARIANTS, StudyVariant, study_context
from lib.template_guards import check_template_sampling_marker
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
parser.add_argument("--skip-templates", action="store_true", default=False,
                   help="Force --skip-templates for ALL variants (overrides per-variant skip_templates setting). Use when templates are already computed and you want to skip regeneration for all studies.")
parser.add_argument("--draft", action=argparse.BooleanOptionalAction, default=False,
                   help="Pass --draft to run_sensitivity.py to enable draft mode (coarse grid) for Sensitivity analysis.")
parser.add_argument("--fit_method", choices=["legacy", "pull"], default="pull",
                   help="Sensitivity chi2 method passed to run_sensitivity.py (default 'pull'). The legacy_fit study overrides it per variant.")
parser.add_argument("--strict_validation", action=argparse.BooleanOptionalAction, default=None,
                   help="Pass --strict_validation / --no-strict_validation to run_sensitivity.py. Unset = 06_significance.py default (strict for pull, warn-only for legacy). A strict gate failure stops the whole study run.")
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
parser.add_argument("--flyweight", action=argparse.BooleanOptionalAction, default=True,
                   help="Flyweight Sensitivity templates (default). --no-flyweight uses the per-point template grid.")
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


def _variant_flag(extra: List[str], flag: str, default: bool) -> bool:
    """Value of a --flag / --no-flag pair in a variant's extra args (last one wins)."""
    value = default
    for token in extra:
        if token == f"--{flag}":
            value = True
        elif token == f"--no-{flag}":
            value = False
    return value


def _variant_option(extra: List[str], option: str, default=None):
    """Value following --option in a variant's extra args (last one wins)."""
    value = default
    for idx, token in enumerate(extra[:-1]):
        if token == f"--{option}":
            value = extra[idx + 1]
    return value


def _templates_ready(variant: StudyVariant, folders: List[str], energies: List[str]) -> bool:
    """True only if every Sensitivity template the variant reads under --skip-templates exists.

    Layout written by 01_background_template.py / 02_signal_template.py:
      {PATH}/SENSITIVITY/{config}/background/{folder}/{energy}{template_suffix}/
          TEMPLATE_NORMALIZATION.json, {config}_background_NHits{n}_AdjCl{a}_OpHits{o}.pkl
      {PATH}/SENSITIVITY/{config}/{signal}/{folder}/{energy}{template_suffix}/
          TEMPLATE_NORMALIZATION.json and, per cut,
          {config}_{signal}_NHits{n}_AdjCl{a}_OpHits{o}_BASE.pkl       (--flyweight)
          {config}_{signal}_NHits{n}_AdjCl{a}_OpHits{o}_SAMPLING.json  (per-point grid)
    template_suffix follows lib/study.py study_context, so only selection-changing variants
    (charge, dm2, truth fiducial, membrane veto) get their own directories. Without
    --nhits/--adjcls/--ophits the variant's cut comes from a best-cut map this orchestrator does
    not read, so any complete cut counts.
    """
    extra = list(variant.get("extra", []))
    flyweight = _variant_flag(extra, "flyweight", args.flyweight)
    backend = _variant_option(extra, "oscillation_backend", args.oscillation_backend)
    manual = None not in (args.nhits, args.adjcls, args.ophits)
    cut = f"NHits{args.nhits}_AdjCl{args.adjcls}_OpHits{args.ophits}" if manual else "NHits*_AdjCl*_OpHits*"
    oversample = int(_analysis_info.get("OSC_NADIR_OVERSAMPLE", 1))

    def _has(directory: Path, pattern: str) -> bool:
        return (directory / pattern).exists() if manual else any(directory.glob(pattern))

    for folder in folders:
        ctx = study_context(argparse.Namespace(
            study_label=variant.get("label"),
            charge_threshold=float(_variant_option(extra, "charge_threshold", 0) or 0),
            dm2=_variant_option(extra, "dm2"),
            exposure=None,
            truth_fiducial=_variant_flag(extra, "truth_fiducial", False),
            membrane_veto=_variant_flag(extra, "membrane_veto", True),
            folder=folder,
        ), analysis="Sensitivity")
        for config in args.config:
            for energy in energies:
                subdir = f"{energy}{ctx.template_suffix}"
                bkg_dir = _data_root / "SENSITIVITY" / config / "background" / folder.lower() / subdir
                if not (bkg_dir / "TEMPLATE_NORMALIZATION.json").exists():
                    return False
                if not _has(bkg_dir, f"{config}_background_{cut}.pkl"):
                    return False
                for signal in args.signals:
                    sig_dir = _data_root / "SENSITIVITY" / config / signal / folder.lower() / subdir
                    if not (sig_dir / "TEMPLATE_NORMALIZATION.json").exists():
                        return False
                    if flyweight:
                        if not _has(sig_dir, f"{config}_{signal}_{cut}_BASE.pkl"):
                            return False
                    elif manual:
                        current, _ = check_template_sampling_marker(
                            str(sig_dir), config, signal, args.nhits, args.adjcls, args.ophits,
                            backend=backend, nadir_oversample=oversample, scopes=("grid",),
                        )
                        if not current:
                            return False
                    elif not _has(sig_dir, f"{config}_{signal}_{cut}_SAMPLING.json"):
                        return False
    return True


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
    label = variant.get("label") or ""
    if "unc_bkg" in label:
        return 0
    if "unc_" in label:
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

    # Templates: --skip-templates is only honoured when this variant's templates really exist
    sensitivity_analyses = [a for a in effective_analysis if a.upper() == "SENSITIVITY"]
    if skip_templates and sensitivity_analyses and not _templates_ready(variant, folders, energy):
        rprint(
            f"[yellow][WARNING][/yellow] Templates for {label or folders} are missing or stale; "
            "regenerating them instead of honouring --skip-templates."
        )
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
    cmd.append("--flyweight" if args.flyweight else "--no-flyweight")
    cmd += ["--fit_method", args.fit_method]        # variant "extra" below may override it
    if args.strict_validation is not None:
        cmd.append("--strict_validation" if args.strict_validation else "--no-strict_validation")

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
