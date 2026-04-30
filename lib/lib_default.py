import json
import os
import numpy as np
from typing import Optional


DEFAULT_ANALYSIS_THRESHOLDS = {
    "DAYNIGHT": {"SIGNIFICANCE": 0.0},
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
