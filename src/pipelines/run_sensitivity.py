import os
import sys
import subprocess
from typing import List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *


ENERGY_CHOICES = [
    "SignalParticleK",
    "ClusterEnergy",
    "TotalEnergy",
    "SelectedEnergy",
    "SolarEnergy",
]
PRESENTATION_SCRIPTS = {
    "DayNight": "src/tools/presentations/daynight.py",
    "HEP": "src/tools/presentations/hep.py",
    "Sensitivity": "src/tools/presentations/sensitivity.py",
}


def build_command(script_name: str, additional_args: Optional[List[str]] = None) -> List[str]:
    command = ["python3", f"{root}/{script_name}"]
    if additional_args:
        command.extend(str(arg) for arg in additional_args)
    return command



def _subprocess_env() -> dict:
    """Return an environment dict with Apptainer-specific workarounds applied.

    BROWSER_PATH: points kaleido at the project-local Chrome installation
    (.chrome/chrome-linux64/chrome) so static image export works without
    needing a system-wide Chrome install.

    Note: do NOT set LD_PRELOAD here. Chrome is spawned as a child of the
    analysis subprocesses, and it would inherit LD_PRELOAD, causing it to
    crash immediately. The pyarrow/libarrow exit-segfault is handled instead
    by the os._exit(0) atexit handler in lib/__init__.py.
    """
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
            f"Workflow stopped because {script_label} failed with exit code {completed.returncode}.\n"
            f"Executed command: {rendered}"
        )



def run_analysis_script(script_name: str, additional_args: Optional[List[str]] = None):
    run_python_command(build_command(script_name, additional_args), label=script_name)



def run_presentation_script(script_name: str, energy: str, stop_on_error: bool = True):
    script_path = f"{root}/{script_name}"
    command = ["python3", script_path, "--energy", energy]
    run_python_command(command, label=script_name, stop_on_error=stop_on_error)



def sample_input_exists(config: str, name: str) -> bool:
    info = json.load(open(f"{root}/config/{config}/{config}_config.json", "r"))
    required_file = (
        f"{info['PATH']}/data/{info['GEOMETRY']}/{info['VERSION']}/"
        f"{info['NAME']}{name}/Config/GEANT4Label.npy"
    )
    return os.path.exists(required_file)



def get_selected_background_components(analysis_names: List[str], analysis_info: dict) -> set[str]:
    background_config = analysis_info.get("BACKGROUND_SAMPLES", {})
    per_analysis = background_config.get("ANALYSES", {})
    selected = set()
    for analysis_name in analysis_names:
        selected.update(
            [
                str(component).lower()
                for component in per_analysis.get(
                    analysis_name.upper(),
                    background_config.get("default", []),
                )
            ]
        )
    return selected



def is_essential_component(sample_key: str, analysis_info: dict) -> bool:
    if sample_key == "marley":
        return True
    essential_map = analysis_info.get("BACKGROUND_SAMPLES", {}).get("ESSENTIAL", {})
    return bool(essential_map.get(sample_key, False))


parser = argparse.ArgumentParser(
    description="Run DayNight, HEP, and Sensitivity workflows from a single entrypoint"
)
parser.add_argument(
    "--config",
    nargs="+",
    type=str,
    help="The configurations to load",
    default=["hd_1x2x6_centralAPA"],
)
parser.add_argument(
    "--names",
    nargs="+",
    type=str,
    help="The sample names to process",
    default=["marley", "gamma", "neutron", "radiological"],
)
parser.add_argument(
    "--analysis",
    nargs="+",
    type=str,
    help="The analyses to run",
    choices=["DayNight", "HEP", "Sensitivity"],
    default=["DayNight", "HEP", "Sensitivity"],
)
parser.add_argument(
    "--reference",
    type=str,
    help="The histogram reference used by sensitivity/05_best_sigmas.py when selecting best curves",
    choices=["Smoothed", "Raw"],
    default=None,
)
parser.add_argument(
    "--folder",
    nargs="+",
    type=str,
    help="The result folders to process",
    choices=["Reduced", "Truncated", "Nominal"],
    default=["Truncated"],
)
parser.add_argument(
    "--exposure",
    type=float,
    help="The exposure for the analysis in years",
    default=30,
)
parser.add_argument(
    "--signal_uncertainty",
    type=float,
    help="Global signal uncertainty override for all selected analyses",
    default=None,
)
parser.add_argument(
    "--background_uncertainty",
    type=float,
    help="Global background uncertainty override for all selected analyses",
    default=None,
)
parser.add_argument(
    "--energy",
    nargs="+",
    type=str,
    help="The energy for the analysis",
    choices=ENERGY_CHOICES,
    default=["SolarEnergy"],
)
parser.add_argument("--nhits", type=int, help="The nhit cut for the analysis", default=None)
parser.add_argument("--ophits", type=int, help="The ophit cut for the analysis", default=None)
parser.add_argument("--adjcls", type=int, help="The adjacent cluster cut for the analysis", default=None)
parser.add_argument(
    "--fiducial_mc_threshold",
    type=float,
    help="Minimum summed MCCounts per essential background component required in fiducialization",
    default=get_analysis_threshold(str(root), "FIDUCIALIZATION", stage="MC", fallback=0.0),
)
parser.add_argument(
    "--daynight_mc_threshold",
    type=float,
    help=(
        "Minimum summed MCCounts per essential background component required in DayNight cut scan. "
        "This is configured in analysis/config.json and can be overridden on the CLI."
    ),
    default=get_analysis_threshold(str(root), "DAYNIGHT", stage="MC", fallback=0.0),
)
parser.add_argument(
    "--earth_density_band",
    type=float,
    default=0.13,
    help="Fractional spread on predicted day-night asymmetry from Earth density profile variations (MSW/PREM). Default 13%%.",
)
parser.add_argument(
    "--oscillation_band",
    type=float,
    default=0.05,
    help="Fractional uncertainty on predicted asymmetry from oscillation parameter uncertainties (theta12, dm221). Default 5%%.",
)
parser.add_argument(
    "--day_fraction",
    type=float,
    default=0.493,
    help="Fraction of total exposure attributed to daytime. Default 0.493 (SURF latitude ~44.3°N, averaged over full year).",
)
parser.add_argument(
    "--day_fraction_band",
    type=float,
    default=0.01,
    help="Absolute uncertainty on --day_fraction from imperfect solar zenith angle cut knowledge. Default 1%%.",
)
parser.add_argument(
    "--hep_mc_threshold",
    type=float,
    help=(
        "Minimum summed MCCounts per essential background component required in HEP cut scan. "
        "This is configured in analysis/config.json and can be overridden on the CLI."
    ),
    default=None,
)
parser.add_argument("--background", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument(
    "--significance",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Run main significance computing macros (01_daynight.py, 01_hep.py, 06_significance.py). Pass --no-significance to skip.",
)
parser.add_argument(
    "--computation",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Run all analysis macros (fiducialization, best-sigma selection, significance). Pass --no-computation to run only plot-producing macros.",
)
parser.add_argument(
    "--fiducialization",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Run signal/01_fiducialize.py steps. Pass --no-fiducialization to skip when fiducial outputs are already up to date.",
)
parser.add_argument(
    "--rebin",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Run full cut scan in signal/03_analysis.py (rebinned DataFrames + AnalysisMask). Pass --no-rebin to skip; raw array and FiducializationMask export always runs.",
)
parser.add_argument(
    "--all_metrics",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Override config defaults and run all test statistic metrics in 01_daynight.py and 01_hep.py (Gaussian, Asimov, etc). Pass --all_metrics to enable.",
)
parser.add_argument(
    "--test_statistic",
    type=str,
    choices=["asimov", "gaussian", "all"],
    default="asimov",
    help=(
        "DayNight test statistic: 'asimov' (default), 'gaussian', or 'all' for both. "
        "Controls which significance metrics are computed in 01_daynight.py."
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
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument(
    "--nuisance_profiles",
    nargs="+",
    type=str,
    default=None,
    help=(
        "Nuisance profile names to run (keys in NUISANCE_PROFILES in analysis/config.json). "
        "Defaults to all profiles defined in analysis/config.json. "
        "Example: --nuisance_profiles full nominal"
    ),
)
parser.add_argument(
    "--oscillation_backend",
    type=str,
    choices=["file", "prob3", "nufast"],
    default="nufast",
    help=(
        "Oscillation backend propagated to all workflow stages. "
        "'file' uses pre-computed pkl oscillograms; "
        "'prob3'/'nufast' compute oscillation on-the-fly (no pkl files required)."
    ),
)
parser.add_argument(
    "--optimization",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Run Silverman/Scott bandwidth optimization before each analysis stage and apply "
        "the per-config/name recommended sigma at runtime. Off by default. "
        "Pass --optimization to enable."
    ),
)
parser.add_argument(
    "--smoothing_strategy",
    type=str,
    default="silverman",
    choices=["silverman", "scott", "none"],
    help=(
        "Bandwidth rule used when --optimization is enabled. "
        "'silverman' (default) and 'scott' apply analytic rule-of-thumb estimation."
    ),
)
parser.add_argument(
    "--apply_smoothing",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "Write the optimized sigma into analysis/smoothing.json (CONFIG_OVERRIDES + global fallback) "
        "when --optimization is enabled. Pass --no-apply-smoothing to compute and report only."
    ),
)
parser.add_argument(
    "--stacked",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Pass --stacked to all 0ZSignificancePlot.py calls to generate stacked-area "
        "significance plots alongside the standard line plots."
    ),
)

args = parser.parse_args()
configure_global_logging(verbose=args.verbose)
analysis_info = load_analysis_info(str(root))
selected_background_components = get_selected_background_components(args.analysis, analysis_info)

if not args.computation:
    rprint(
        "[cyan][INFO][/cyan] Computation disabled (--no-computation): running plot-producing macros only."
    )
elif not args.significance:
    rprint(
        "[cyan][INFO][/cyan] Significance disabled (--no-significance): skipping DayNight/HEP/Sensitivity computation."
    )



def get_analysis_uncertainties(analysis_name: str) -> tuple[float, float]:
    analysis_key = str(analysis_name).upper()
    configured = analysis_info.get("ANALYSIS_UNCERTAINTIES", {}).get(analysis_key, {})
    signal_uncertainty = float(
        args.signal_uncertainty
        if args.signal_uncertainty is not None
        else configured.get("signal_uncertainty", analysis_info.get("SIGNAL_ERROR", 0.04))
    )
    background_uncertainty = float(
        args.background_uncertainty
        if args.background_uncertainty is not None
        else configured.get(
            "background_uncertainty", analysis_info.get("BACKGROUND_ERROR", 0.02)
        )
    )
    return signal_uncertainty, background_uncertainty



def uncertainty_args_for(analysis_name: str) -> List[str]:
    signal_uncertainty, background_uncertainty = get_analysis_uncertainties(analysis_name)
    return [
        "--signal_uncertainty",
        str(signal_uncertainty),
        "--background_uncertainty",
        str(background_uncertainty),
    ]



def energy_args_for(energies: List[str]) -> List[str]:
    return ["--energy", *energies]



def exposure_arg_for() -> List[str]:
    if args.exposure is None:
        return []
    return ["--exposure", str(args.exposure)]



def cut_args_for() -> List[str]:
    cut_args: List[str] = []
    if args.nhits is not None:
        cut_args.extend(["--nhits", str(args.nhits)])
    if args.ophits is not None:
        cut_args.extend(["--ophits", str(args.ophits)])
    if args.adjcls is not None:
        cut_args.extend(["--adjcls", str(args.adjcls)])
    return cut_args



def common_analysis_args_for(energies: List[str]) -> List[str]:
    return energy_args_for(energies) + exposure_arg_for() + cut_args_for()



def base_args_for(config: str, folder: str, include_background: bool = False) -> List[str]:
    base_args = [
        "--config",
        config,
        "--folder",
        folder,
        "--rewrite" if args.rewrite else "--no-rewrite",
        "--debug" if args.verbose == "verbose" else "--no-debug",
        "--plot" if args.plot else "--no-plot",
    ]
    if include_background:
        base_args.append("--background" if args.background else "--no-background")
    return base_args


def oscillation_args_for() -> List[str]:
    return ["--oscillation_backend", args.oscillation_backend]


def stacked_args_for() -> List[str]:
    return ["--stacked"] if args.stacked else ["--no-stacked"]


def all_metrics_args_for() -> List[str]:
    return ["--all_metrics"] if args.all_metrics else []


def test_statistic_args_for() -> List[str]:
    return ["--test_statistic", args.test_statistic]


def reference_args_for(reference: str) -> List[str]:
    # Omit --reference when --all_metrics enabled to show all available metrics
    return [] if args.all_metrics else ["--reference", reference]


def nuisance_profile_args_for(profile_name: str) -> List[str]:
    return ["--nuisance_profile", profile_name]



def run_shared_prerequisites(config: str, folder: str, available_names: List[str]):
    if not args.computation:
        return

    base_args = base_args_for(config, folder, include_background=False)
    analysis_selection_args = ["--analysis", *args.analysis]
    energy_args = energy_args_for(args.energy)
    exposure_args = exposure_arg_for()
    cut_args = cut_args_for()

    if args.fiducialization:
        for name in available_names:
            sample_args = base_args + ["--name", name]
            run_analysis_script("src/physics/signal/01_fiducialize.py", sample_args + energy_args + oscillation_args_for())
    else:
        rprint("[cyan][INFO][/cyan] Skipping signal/01_fiducialize.py (--no-fiducialization).")

    for name in available_names:
        if "marley" not in name:
            continue
        sample_args = base_args + ["--name", name]
        run_analysis_script(
            "src/physics/signal/02_best_fiducial.py",
            sample_args
            + energy_args
            + exposure_args
            + analysis_selection_args
            + ["--mc_threshold", str(args.fiducial_mc_threshold)],
        )

    # Visualize fiducial results before lengthy analysis
    plot_base_args = base_args_for(config, folder, include_background=False) + ["--name", available_names[0]]
    run_analysis_script(
        "src/physics/common/significance_plot.py",
        plot_base_args + energy_args_for(args.energy) + exposure_arg_for()
        + ["--analysis", "Fiducial", "--fiducial-analyses", *args.analysis],
    )

    # Pass 1+2 (merged): Ref arrays + FiducializationMask + full cut scan in one data load
    if args.rebin:
        for name in available_names:
            sample_args = base_args + ["--name", name]
            run_analysis_script(
                "src/physics/signal/03_analysis.py",
                sample_args
                + analysis_selection_args
                + energy_args
                + cut_args
                + oscillation_args_for()
                + ["--export_fiducial"],
            )
    else:
        rprint("[cyan][INFO][/cyan] Skipping signal/03_analysis.py cut scan (--no-rebin).")
        # Still export raw arrays + FiducializationMask even when skipping cut scan
        for name in available_names:
            sample_args = base_args + ["--name", name]
            run_analysis_script(
                "src/physics/signal/03_analysis.py",
                sample_args
                + analysis_selection_args
                + energy_args
                + cut_args
                + oscillation_args_for()
                + ["--export_fiducial", "--skip_scan", "--no-plot"],
            )



def _set_smoothing_env(analysis_name: str, config: str, name: str, folder: str) -> None:
    """Read recommended sigma from optimizer output and export as env var for child processes."""
    from pathlib import Path
    import json as _json
    sigmas = []
    for energy in args.energy:
        sigma_path = (
            Path(str(root)) / "data" / "smoothing" / config / name
            / f"{folder.lower()}_{energy}_{analysis_name}_sigma.json"
        )
        if sigma_path.exists():
            try:
                data = _json.loads(sigma_path.read_text())
                s = float(data.get("recommended_sigma", 0.0))
                if s > 0:
                    sigmas.append(s)
                elif s == 0.0:
                    rprint(
                        f"[yellow][WARNING][/yellow] Smoothing sigma=0.0 in {sigma_path} "
                        "(key 'recommended_sigma' missing or zero) — not applied."
                    )
            except Exception as _exc:
                rprint(
                    f"[yellow][WARNING][/yellow] Could not read smoothing sigma from "
                    f"{sigma_path}: {_exc} — skipping."
                )
    if sigmas:
        sigma = max(sigmas)
        os.environ[f"SOLAR_SMOOTHING_SIGMA_{analysis_name.upper()}"] = str(sigma)
        rprint(
            f"[green][SMOOTHING][/green] {config}/{name} {analysis_name}: "
            f"env sigma={sigma:.4f} bins"
        )


def run_smoothing_optimization(config: str, folder: str, name: str, analysis_name: str):
    if not args.optimization or args.smoothing_strategy == "none":
        return
    script_path = f"{root}/src/tools/optimize_smoothing.py"
    if not os.path.exists(script_path):
        rprint(f"[yellow][WARNING][/yellow] Smoothing optimizer not found: {script_path}. Skipping.")
        return
    command = [
        "python3", script_path,
        "--config", config,
        "--name", name,
        "--folder", folder,
        "--analysis", analysis_name,
        "--strategy", args.smoothing_strategy,
        "--energy", *args.energy,
        "--rewrite" if args.rewrite else "--no-rewrite",
        "--patch" if args.apply_smoothing else "--no-patch",
    ]
    run_python_command(command, label="optimize_smoothing.py", stop_on_error=False)
    _set_smoothing_env(analysis_name, config, name, folder)


def run_daynight_stage(config: str, folder: str, name: str):
    run_smoothing_optimization(config, folder, name, "DayNight")
    plot_base_args = base_args_for(config, folder, include_background=False) + ["--name", name]
    analysis_base_args = base_args_for(config, folder, include_background=False) + ["--name", name]
    common_args = common_analysis_args_for(args.energy)
    selector_args = energy_args_for(args.energy) + cut_args_for()
    reference = args.reference or "Smoothed"
    uncertainty_args = uncertainty_args_for("DAYNIGHT")

    if args.computation:
        if args.significance:
            daynight_args = [
                "--mc_threshold", str(args.daynight_mc_threshold),
                "--earth_density_band", str(args.earth_density_band),
                "--oscillation_band", str(args.oscillation_band),
                "--day_fraction", str(args.day_fraction),
                "--day_fraction_band", str(args.day_fraction_band),
            ]
            run_analysis_script("src/physics/daynight/01_daynight.py", analysis_base_args + common_args + uncertainty_args + daynight_args + test_statistic_args_for() + all_metrics_args_for())
        run_analysis_script(
            "src/physics/sensitivity/05_best_sigmas.py",
            plot_base_args + selector_args + ["--analysis", "DayNight", "--reference", reference],
        )
    run_analysis_script("src/physics/common/exposure_plot.py", analysis_base_args + common_args + uncertainty_args + ["--analysis", "DayNight"])
    run_analysis_script(
        "src/physics/common/significance_plot.py",
        analysis_base_args + common_args + uncertainty_args + stacked_args_for() + ["--analysis", "DayNight"],
    )



def run_hep_stage(config: str, folder: str, name: str):
    run_smoothing_optimization(config, folder, name, "HEP")
    plot_base_args = base_args_for(config, folder, include_background=False) + ["--name", name]
    analysis_base_args = base_args_for(config, folder, include_background=False) + ["--name", name]
    common_args = common_analysis_args_for(args.energy)
    selector_args = energy_args_for(args.energy) + cut_args_for()
    reference = args.reference or "Smoothed"
    uncertainty_args = uncertainty_args_for("HEP")
    hep_significance_reference = str(
        analysis_info.get("BEST_SIGMA_SIGNIFICANCE_REFERENCE", {}).get("HEP", "Asimov")
    )
    if hep_significance_reference not in ["Asimov", "Gaussian", "ProfileLikelihood"]:
        hep_significance_reference = "Asimov"

    def _component_oscillation(component: str) -> str:
        return "Osc" if str(component).lower() in {"8b", "hep"} else "Truth"

    def _auto_select_hep_mc_threshold(energies: List[str]) -> int:
        background_config = analysis_info.get("BACKGROUND_SAMPLES", {})
        selected_components = [
            str(component).lower()
            for component in background_config.get("ANALYSES", {}).get(
                "HEP", background_config.get("default", [])
            )
        ]
        essential_map = {
            str(component).lower(): bool(is_essential)
            for component, is_essential in background_config.get("ESSENTIAL", {}).items()
        }
        essential_components = [
            component for component in selected_components if essential_map.get(component, False)
        ]

        if not essential_components:
            rprint(
                "[cyan][INFO][/cyan] No essential HEP background components configured; using mc_threshold=0."
            )
            return 0

        candidate_thresholds = [10, 5, 2, 1, 0]
        selected_by_energy = []

        for energy in energies:
            signal_path = (
                f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/signal/{folder.lower()}/HEP/"
                f"{config}/{name}/{config}_{name}_{energy}_Rebin.pkl"
            )
            if not os.path.exists(signal_path):
                rprint(
                    f"[yellow][WARNING][/yellow] Missing HEP signal file for auto mc-threshold selection: {signal_path}. "
                    "Using mc_threshold=0 for this energy."
                )
                selected_by_energy.append(0)
                continue

            plot_df = pd.read_pickle(signal_path)
            for _, filepath in load_available_background_dataframes(
                str(root), "HEP", folder, config, energy
            ):
                bkg_df = pd.read_pickle(filepath)
                plot_df = pd.concat([plot_df, bkg_df], ignore_index=True)

            if plot_df.empty or "Energy" not in plot_df.columns:
                rprint(
                    f"[yellow][WARNING][/yellow] Invalid HEP input dataframe while auto-selecting mc-threshold "
                    f"for {config} {name} {energy}. Using mc_threshold=0."
                )
                selected_by_energy.append(0)
                continue

            energy_axis = np.asarray(plot_df.iloc[0]["Energy"], dtype=float)
            hep_threshold = get_analysis_threshold(
                str(root), "HEP", stage="SIGNIFICANCE", fallback=0.0
            )
            threshold_idx = int(np.searchsorted(energy_axis, hep_threshold, side="left"))
            cut_rows = plot_df[["NHits", "OpHits", "AdjCl"]].drop_duplicates()
            if cut_rows.empty:
                selected_by_energy.append(0)
                continue

            mc_by_component = {component: {} for component in essential_components}
            for component in essential_components:
                component_df = plot_df.loc[
                    (plot_df["Component"].astype(str).str.lower() == component)
                    & (plot_df["Oscillation"] == _component_oscillation(component))
                    & (plot_df["Mean"] == "Mean")
                ].copy()

                if component_df.empty:
                    continue

                for _, row in component_df.iterrows():
                    mc_counts = np.asarray(row["MCCounts"], dtype=float)
                    mc_sum = float(np.nansum(mc_counts[threshold_idx:]))
                    cut_key = (int(row["NHits"]), int(row["OpHits"]), int(row["AdjCl"]))
                    mc_by_component[component][cut_key] = mc_sum

            best_for_energy = 0
            for threshold in candidate_thresholds:
                pass_count = 0
                for _, cut in cut_rows.iterrows():
                    cut_key = (int(cut["NHits"]), int(cut["OpHits"]), int(cut["AdjCl"]))
                    if all(
                        mc_by_component.get(component, {}).get(cut_key, 0.0) >= threshold
                        for component in essential_components
                    ):
                        pass_count += 1
                if pass_count > 0:
                    best_for_energy = threshold
                    break

            selected_by_energy.append(int(best_for_energy))
            rprint(
                f"[cyan][INFO][/cyan] Auto-selected HEP mc-threshold={best_for_energy} for "
                f"{config} {folder} {energy} (essential: {', '.join(essential_components)})."
            )

        return int(min(selected_by_energy)) if selected_by_energy else 0

    resolved_hep_mc_threshold = (
        int(args.hep_mc_threshold)
        if args.hep_mc_threshold is not None
        else _auto_select_hep_mc_threshold(args.energy)
    )
    if args.hep_mc_threshold is None:
        rprint(
            f"[cyan][INFO][/cyan] Using auto-selected HEP mc-threshold={resolved_hep_mc_threshold} "
            f"for config={config} folder={folder}."
        )

    if args.computation:
        if args.significance:
            run_analysis_script(
                "src/physics/hep/01_hep.py",
                analysis_base_args + common_args + uncertainty_args + ["--mc_threshold", str(resolved_hep_mc_threshold)] + all_metrics_args_for(),
            )
        run_analysis_script(
            "src/physics/sensitivity/05_best_sigmas.py",
            plot_base_args + selector_args + ["--analysis", "HEP", "--reference", reference],
        )
    run_analysis_script(
        "src/physics/common/exposure_plot.py",
        analysis_base_args + common_args + uncertainty_args + ["--analysis", "HEP", "--mode", "exposure"] + reference_args_for(hep_significance_reference),
    )
    run_analysis_script(
        "src/physics/common/exposure_plot.py",
        analysis_base_args + common_args + uncertainty_args + ["--analysis", "HEP", "--mode", "exposure", "--pkl_label", "highest_spiked"] + reference_args_for(hep_significance_reference),
    )
    run_analysis_script(
        "src/physics/common/significance_plot.py",
        analysis_base_args
        + common_args
        + uncertainty_args
        + stacked_args_for()
        + ["--analysis", "HEP", "--reference", hep_significance_reference, "--bottom-panel-mode", "both"],
    )
    run_analysis_script(
        "src/physics/common/significance_plot.py",
        analysis_base_args
        + common_args
        + uncertainty_args
        + stacked_args_for()
        + ["--analysis", "HEP", "--reference", hep_significance_reference, "--bottom-panel-mode", "both", "--pkl-label", "highest_spiked"],
    )
    run_analysis_script("src/physics/hep/significance_comparison.py", analysis_base_args + common_args + uncertainty_args)
    run_analysis_script("src/physics/common/exposure_plot.py", analysis_base_args + common_args + uncertainty_args + ["--analysis", "HEP", "--mode", "comparison"])
    run_analysis_script(
        "src/physics/common/exposure_plot.py",
        analysis_base_args + common_args + uncertainty_args + ["--analysis", "HEP", "--mode", "rebin"] + reference_args_for(hep_significance_reference),
    )

    # HEP stage summary
    rprint(
        "\n[cyan][INFO][/cyan] HEP stage outputs (with --all_metrics, additional metric variants will appear):"
        "\n  • exposure_plot (main) — exposure vs significance curve"
        "\n  • exposure_plot (highest_spiked) — same with spiked cuts only"
        "\n  • significance_plot (main) — 2D contours [reference={hep_significance_reference}]"
        "\n  • significance_plot (highest_spiked) — 2D contours for spiked cuts only"
        "\n  • significance_comparison — Gaussian vs Asimov overlay"
        "\n  • exposure_plot (comparison mode) — exposure comparison across metrics"
        "\n  • exposure_plot (rebin mode) — rebinned energy intervals"
    )



def run_sensitivity_stage(config: str, folder: str, name: str):
    run_smoothing_optimization(config, folder, name, "Sensitivity")
    plot_base_args = base_args_for(config, folder, include_background=False) + ["--name", name]
    background_base_args = base_args_for(config, folder, include_background=True) + ["--name", name]
    uncertainty_args = uncertainty_args_for("SENSITIVITY")
    reference_args = ["--reference", "SENSITIVITY"]

    _nuisance_profiles = analysis_info.get("NUISANCE_PROFILES", {})
    _default_profile = analysis_info.get("DEFAULT_NUISANCE_PROFILE", "full")
    _all_profiles = list(_nuisance_profiles.keys()) if _nuisance_profiles else [_default_profile or "full"]
    if args.nuisance_profiles:
        profile_names = [p for p in args.nuisance_profiles if p in _nuisance_profiles]
        unknown = [p for p in args.nuisance_profiles if p not in _nuisance_profiles]
        if unknown:
            rprint(f"[yellow][WARNING][/yellow] Unknown nuisance profiles (ignored): {unknown}. Available: {_all_profiles}")
        if not profile_names:
            raise SystemExit(f"No valid nuisance profiles selected. Available: {_all_profiles}")
    else:
        profile_names = [_default_profile] if _default_profile in _nuisance_profiles else _all_profiles[:1]

    for energy in args.energy:
        energy_args = common_analysis_args_for([energy])
        if args.computation and args.significance:
            # Phase 1 — background templates (all cuts)
            run_analysis_script(
                "src/physics/sensitivity/03_template_compute.py",
                plot_base_args + reference_args + energy_args + uncertainty_args + ["--template", "background"] + oscillation_args_for(),
            )
            # Phase 2 — cut optimisation: score all background-candidate cuts, write highest_SENSITIVITY.pkl
            run_analysis_script(
                "src/physics/sensitivity/04_best_cuts.py",
                background_base_args + energy_args + uncertainty_args + oscillation_args_for(),
            )
            # Phase 3 — full signal template grid for the selected best cut
            run_analysis_script(
                "src/physics/sensitivity/03_template_compute.py",
                plot_base_args + reference_args + energy_args + uncertainty_args + ["--template", "signal"] + oscillation_args_for(),
            )
            # Phase 4 — sensitivity analysis with best cut (no re-optimisation)
            for profile_name in profile_names:
                run_analysis_script(
                    "src/physics/sensitivity/06_significance.py",
                    background_base_args + energy_args + nuisance_profile_args_for(profile_name) + oscillation_args_for(),
                )
        run_analysis_script(
            "src/physics/sensitivity/template_plot.py",
            plot_base_args + reference_args + energy_args + ["--template", "all"] + oscillation_args_for(),
        )
        for profile_name in profile_names:
            run_analysis_script(
                "src/physics/sensitivity/contour_plot.py",
                background_base_args + reference_args + energy_args + nuisance_profile_args_for(profile_name),
            )
            run_analysis_script(
                "src/physics/common/exposure_plot.py",
                background_base_args + energy_args + ["--analysis", "Sensitivity", "--compare"] + nuisance_profile_args_for(profile_name),
            )
        run_analysis_script(
            "src/physics/common/significance_plot.py",
            plot_base_args + energy_args + uncertainty_args + stacked_args_for() + ["--analysis", "Sensitivity"],
        )



def run_presentations():
    for analysis_name in args.analysis:
        script_name = PRESENTATION_SCRIPTS.get(analysis_name)
        if script_name is None:
            continue
        for energy in args.energy:
            run_presentation_script(script_name, energy)


for config, folder in product(args.config, args.folder):
    available_names = []
    skipped_by_policy = []
    missing_essential = []
    missing_optional = []

    for name in args.names:
        sample_key = name.split("_")[0].lower()
        component_is_essential = is_essential_component(sample_key, analysis_info)
        is_explicitly_in_analysis = sample_key in selected_background_components

        if sample_key != "marley" and not component_is_essential and not is_explicitly_in_analysis:
            skipped_by_policy.append(name)
            continue

        if sample_input_exists(config, name):
            available_names.append(name)
        elif component_is_essential:
            missing_essential.append(name)
        else:
            missing_optional.append(name)

    if skipped_by_policy:
        rprint(
            "[cyan][INFO][/cyan] Skipping non-essential samples not included in analysis component list for "
            f"{config}: {', '.join(skipped_by_policy)}"
        )

    if missing_optional:
        rprint(
            f"[yellow][WARNING][/yellow] Skipping unavailable optional samples for {config}: {', '.join(missing_optional)}"
        )

    if missing_essential:
        raise SystemExit(
            "Workflow stopped because essential samples are missing before computation.\n"
            f"Config={config} Folder={folder}\n"
            f"Missing essential samples: {', '.join(missing_essential)}"
        )

    if not available_names:
        rprint(
            f"[yellow][WARNING][/yellow] No available samples found for {config} in folder {folder}. Skipping config."
        )
        continue

    rprint(f"Processing config={config} folder={folder} analyses={','.join(args.analysis)}")
    run_shared_prerequisites(config, folder, available_names)

    # Show fiducial selections early, right after best_fiducial optimization completes

    for name in available_names:
        if "marley" not in name:
            continue
        if "DayNight" in args.analysis:
            run_daynight_stage(config, folder, name)
        if "HEP" in args.analysis:
            run_hep_stage(config, folder, name)
        if "Sensitivity" in args.analysis:
            run_sensitivity_stage(config, folder, name)

    # Pass 3: post-analysis — export AnalysisMask at best cuts for all productions
    # Best-cuts JSONs exist for marley; apply same cuts to all samples
    if args.computation:
        base_args = base_args_for(config, folder, include_background=False)
        analysis_selection_args = ["--analysis", *args.analysis]
        energy_args = energy_args_for(args.energy)
        for name in available_names:
            sample_args = base_args + ["--name", name]
            run_analysis_script(
                "src/physics/signal/03_analysis.py",
                sample_args
                + analysis_selection_args
                + energy_args
                + oscillation_args_for()
                + ["--best_cuts_only", "--no-plot"],
            )

run_presentations()
