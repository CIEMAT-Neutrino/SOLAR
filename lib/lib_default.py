import json
import os
import numpy as np
from typing import Any, Dict, List, Optional


DEFAULT_ANALYSIS_THRESHOLDS = {
    "DAYNIGHT": {"SIGNIFICANCE": 0.0, "MC": 0.0},
    "HEP": {"SIGNIFICANCE": 0.0, "MC": 0.0},
    "SENSITIVITY": {"SIGNIFICANCE": 0.0, "MC": 0.0},
    "FIDUCIALIZATION": {"MC": 0.0},
}


def load_analysis_info(root: str):
    """
    Load and merge analysis configuration from split files.
    """
    analysis_info = json.load(open(f"{root}/import/analysis.json", "r"))

    for extra_file in ["calibration.json", "workflow.json"]:
        extra_path = f"{root}/import/{extra_file}"
        if os.path.exists(extra_path):
            analysis_info.update(json.load(open(extra_path, "r")))

    return analysis_info


def get_default_info(root: str, variable: str):
    """
    This function returns the default analysis information.
    """
    analysis_info = load_analysis_info(root)
    return analysis_info[variable]


def get_analysis_threshold(
    root: str,
    analysis_name: str,
    stage: str = "SIGNIFICANCE",
    fallback: Optional[float] = None,
) -> float:
    """Resolve analysis threshold from centralized JSON config with sane fallbacks."""
    analysis_key = str(analysis_name).upper()
    stage_key = str(stage).upper()

    analysis_info = load_analysis_info(root)
    configured = analysis_info.get("ANALYSIS_THRESHOLDS", {})
    analysis_thresholds = configured.get(analysis_key, {})

    value = None
    if isinstance(analysis_thresholds, dict):
        value = analysis_thresholds.get(stage_key, analysis_thresholds.get("DEFAULT"))
    elif analysis_thresholds is not None:
        value = analysis_thresholds

    if value is None:
        default_thresholds = DEFAULT_ANALYSIS_THRESHOLDS.get(analysis_key, {})
        value = default_thresholds.get(stage_key, fallback)

    if value is None:
        raise KeyError(
            f"Missing threshold for analysis={analysis_key} stage={stage_key} and no fallback provided"
        )

    return float(value)


def load_folder_config(root: str) -> Dict[str, Any]:
    """Load folder mode configuration from analysis/folder_configs.json."""
    with open(f"{root}/analysis/folder_configs.json") as f:
        return json.load(f)


def get_folder_choices(root: str) -> List[str]:
    """Return the list of valid folder names."""
    return load_folder_config(root)["FOLDER_CHOICES"]


def get_default_folder(root: str) -> str:
    """Return the default folder name."""
    return load_folder_config(root)["DEFAULT_FOLDER"]


def get_folder_flags(root: str, folder: str) -> Dict[str, Any]:
    """Return the flag dict for a given folder (apply_z_fiducial, apply_surface_cut, apply_reduction, path_key)."""
    config = load_folder_config(root)
    folders = config.get("FOLDERS", {})
    if folder not in folders:
        raise KeyError(f"Unknown folder '{folder}'. Valid: {list(folders)}")
    return folders[folder]


def folder_path_key(folder: str) -> str:
    """Return the lowercase path component for a folder (e.g. 'Nominal' → 'nominal')."""
    return folder.lower()


def get_default_nhits(root: str, variable: str = "NHITS"):
    """
    This function returns the default energy binning used in the analysis.
    """
    return get_default_info(root, variable)


def get_default_energies(root: str, variable: str = "ENERGY"):
    """
    This function returns the default energy binning used in the analysis.
    """
    analysis_info = load_analysis_info(root)
    e_range = analysis_info[f"{variable}_RANGE"]
    e_bins = analysis_info[f"{variable}_BINS"]
    energy_edges = np.linspace(e_range[0], e_range[-1], e_bins + 1)
    energy_centers = (energy_edges[1:] + energy_edges[:-1]) / 2
    return energy_edges, energy_centers, energy_edges[1] - energy_edges[0]


DEFAULT_WORKFLOW_FLAGS: Dict[str, Dict[str, bool]] = {
    "HEP": {
        "pl_isotonic": False,
        "pl_signal_bands": False,
        "significance_bins": False,
    },
    "DAYNIGHT": {
        "background_error": False,
        "significance_bins": False,
    },
    "SENSITIVITY": {},
}


def get_workflow_flags(root: str, analysis_name: str) -> Dict[str, bool]:
    """Return workflow feature flags for the given analysis, with defaults applied.

    Flags default to False (features off). Override via WORKFLOW section in analysis.json.
    """
    analysis_key = str(analysis_name).upper()
    analysis_info = load_analysis_info(root)
    configured = analysis_info.get("WORKFLOW", {}).get(analysis_key, {})
    defaults = DEFAULT_WORKFLOW_FLAGS.get(analysis_key, {})
    return {**defaults, **configured}
