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
    "DayNight": "generate_daynight_presentation.py",
    "HEP": "generate_hep_presentation.py",
    "Sensitivity": "generate_sensitivity_presentation.py",
}


def build_command(script_name: str, additional_args: Optional[List[str]] = None) -> List[str]:
    command = ["python3", f"{root}/src/analysis/{script_name}"]
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
    script_path = f"{root}/scripts/{script_name}"
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
    help="The histogram reference used by 0ZBestSigmas when selecting best curves",
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
    "--hep_mc_threshold",
    type=float,
    help=(
        "Minimum summed MCCounts per essential background component required in HEP cut scan. "
        "This is configured in import/analysis.json and can be overridden on the CLI."
    ),
    default=get_analysis_threshold(str(root), "HEP", stage="MC", fallback=0.0),
)
parser.add_argument("--background", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument(
    "--significance",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Run main significance computing macros (12DayNight.py, 13HEP.py, 14Sensitivity.py). Pass --no-significance to skip.",
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
    help="Run 0XFiducializeSignal.py steps. Pass --no-fiducialization to skip when fiducial outputs are already up to date.",
)
parser.add_argument(
    "--rebin",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Run 11AnalysisSignal.py rebinning step. Pass --no-rebin to skip when rebinned data is already up to date.",
)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument(
    "--smoothing_strategy",
    type=str,
    default="silverman",
    choices=["silverman", "scott", "none"],
    help=(
        "Bandwidth rule for smoothing optimization run before each analysis stage. "
        "'silverman' (default) and 'scott' apply analytic rule-of-thumb estimation; "
        "'none' skips the optimization step entirely."
    ),
)
parser.add_argument(
    "--apply_smoothing",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "Write the optimized sigma back into import/analysis.json before running each "
        "analysis stage. Pass --no-apply-smoothing to compute and report only."
    ),
)

args = parser.parse_args()
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
        "--debug" if args.debug else "--no-debug",
        "--plot" if args.plot else "--no-plot",
    ]
    if include_background:
        base_args.append("--background" if args.background else "--no-background")
    return base_args



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
            run_analysis_script("0XFiducializeSignal.py", sample_args + energy_args)
    else:
        rprint("[cyan][INFO][/cyan] Skipping 0XFiducializeSignal.py (--no-fiducialization).")

    for name in available_names:
        if "marley" not in name:
            continue
        sample_args = base_args + ["--name", name]
        run_analysis_script(
            "0YBestFiducial.py",
            sample_args
            + energy_args
            + exposure_args
            + analysis_selection_args
            + ["--mc_threshold", str(args.fiducial_mc_threshold)],
        )

    if args.rebin:
        for name in available_names:
            sample_args = base_args + ["--name", name]
            run_analysis_script(
                "11AnalysisSignal.py",
                sample_args + analysis_selection_args + energy_args + cut_args,
            )
    else:
        rprint("[cyan][INFO][/cyan] Skipping 11AnalysisSignal.py (--no-rebin).")



def run_smoothing_optimization(config: str, folder: str, name: str, analysis_name: str):
    if args.smoothing_strategy == "none":
        return
    script_path = f"{root}/scripts/optimize_smoothing.py"
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


def run_daynight_stage(config: str, folder: str, name: str):
    run_smoothing_optimization(config, folder, name, "DayNight")
    plot_base_args = base_args_for(config, folder, include_background=False) + ["--name", name]
    analysis_base_args = base_args_for(config, folder, include_background=False) + ["--name", name]
    common_args = common_analysis_args_for(args.energy)
    selector_args = energy_args_for(args.energy) + cut_args_for()
    reference = args.reference or "Smoothed"
    uncertainty_args = uncertainty_args_for("DAYNIGHT")

    run_analysis_script("10FiducializationPlot.py", plot_base_args + ["--analysis", "DayNight"])
    if args.computation:
        if args.significance:
            run_analysis_script("12DayNight.py", analysis_base_args + common_args + uncertainty_args)
        run_analysis_script(
            "0ZBestSigmas.py",
            plot_base_args + selector_args + ["--analysis", "DayNight", "--reference", reference],
        )
    run_analysis_script("12DayNightExposurePlot.py", analysis_base_args + common_args + uncertainty_args)
    run_analysis_script("12DayNightSignificancePlot.py", analysis_base_args + common_args + uncertainty_args)



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

    run_analysis_script("10FiducializationPlot.py", plot_base_args + ["--analysis", "HEP"])
    if args.computation:
        if args.significance:
            run_analysis_script(
                "13HEP.py",
                analysis_base_args + common_args + uncertainty_args + ["--mc_threshold", str(resolved_hep_mc_threshold)],
            )
        run_analysis_script(
            "0ZBestSigmas.py",
            plot_base_args + selector_args + ["--analysis", "HEP", "--reference", reference],
        )
    run_analysis_script(
        "13HEPExposurePlot.py",
        analysis_base_args + common_args + uncertainty_args + ["--reference", hep_significance_reference],
    )
    run_analysis_script(
        "13HEPExposurePlot.py",
        analysis_base_args + common_args + uncertainty_args + ["--reference", hep_significance_reference, "--pkl_label", "highest_spiked"],
    )
    run_analysis_script(
        "13HEPSignificancePlot.py",
        analysis_base_args
        + common_args
        + uncertainty_args
        + ["--reference", hep_significance_reference, "--bottom-panel-mode", "both"],
    )
    run_analysis_script(
        "13HEPSignificancePlot.py",
        analysis_base_args
        + common_args
        + uncertainty_args
        + ["--reference", hep_significance_reference, "--bottom-panel-mode", "both", "--pkl_label", "highest_spiked"],
    )
    run_analysis_script("13HEPSignificanceComparisonPlot.py", analysis_base_args + common_args + uncertainty_args)
    run_analysis_script("13HEPExposureComparisonPlot.py", analysis_base_args + common_args + uncertainty_args)
    run_analysis_script(
        "13HEPAdaptiveRebinComparisonPlot.py",
        analysis_base_args + common_args + uncertainty_args + ["--reference", hep_significance_reference],
    )



def run_sensitivity_stage(config: str, folder: str, name: str):
    run_smoothing_optimization(config, folder, name, "Sensitivity")
    plot_base_args = base_args_for(config, folder, include_background=False) + ["--name", name]
    background_base_args = base_args_for(config, folder, include_background=True) + ["--name", name]
    uncertainty_args = uncertainty_args_for("SENSITIVITY")
    reference_args = ["--reference", "SENSITIVITY"]

    run_analysis_script("10FiducializationPlot.py", plot_base_args + ["--analysis", "Sensitivity"])
    for energy in args.energy:
        energy_args = common_analysis_args_for([energy])
        if args.computation and args.significance:
            run_analysis_script(
                "14SensitivityTemplateCompute.py",
                plot_base_args + reference_args + energy_args + uncertainty_args + ["--template", "all"],
            )
            run_analysis_script(
                "14Sensitivity.py",
                background_base_args + energy_args,
            )
        run_analysis_script(
            "14SensitivityTemplatePlot.py",
            plot_base_args + reference_args + energy_args + ["--template", "all"],
        )
        run_analysis_script(
            "14SensitivityContourPlot.py",
            background_base_args + reference_args + energy_args,
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

    for name in available_names:
        if "marley" not in name:
            continue
        if "DayNight" in args.analysis:
            run_daynight_stage(config, folder, name)
        if "HEP" in args.analysis:
            run_hep_stage(config, folder, name)
        if "Sensitivity" in args.analysis:
            run_sensitivity_stage(config, folder, name)

run_presentations()
