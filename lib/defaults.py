import json
import os
from pathlib import Path
import numpy as np
from typing import Any, Dict, List, Optional


DEFAULT_ANALYSIS_THRESHOLDS = {
    "DAYNIGHT": {"SIGNIFICANCE": 0.0, "MC": 0.0},
    "HEP": {"SIGNIFICANCE": 0.0, "MC": 0.0},
    "SENSITIVITY": {"SIGNIFICANCE": 0.0, "MC": 0.0},
    "FIDUCIALIZATION": {"MC": 0.0},
}


def load_analysis_info(root: str, physics_override: str = "physics.json"):
    """Load and merge analysis configuration from split files in config/analysis/.

    Args:
        root: Project root path
        physics_override: Override which physics config to load (e.g., "physics_file_backend_legacy.json")
    """
    analysis_dir = Path(root) / "config" / "analysis"
    merged: Dict[str, Any] = {}
    for fname in (physics_override, "config.json", "smoothing.json", "fiducialization.json", "backgrounds.json"):
        path = analysis_dir / fname
        if path.exists():
            merged.update(json.loads(path.read_text()))

    for fname in ("calibration.json", "workflow.json"):
        path = Path(root) / "config" / "import" / fname
        if path.exists():
            merged.update(json.loads(path.read_text()))

    return merged


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


# ── Exposure ──────────────────────────────────────────────────────────────────
# Two different, both analysis-dependent, quantities live in
# config/analysis/config.json. Keep them apart:
#
#   ANALYSIS_EXPOSURES         the exposure the analysis is RUN to -- top of the
#                              significance exposure grid (01_daynight.py, 01_hep.py),
#                              the scale factor baked into the sensitivity templates,
#                              and the x-axis range of the exposure plots. 30 years for
#                              every analysis, plus a 10 year SECONDARY pass for
#                              Sensitivity (run_sensitivity.py --secondary_exposure).
#
#   EVALUATION_EXPOSURE_YEARS  the reference livetime the numbers are QUOTED at --
#                              spectra scaled to counts in common/significance_plot.py,
#                              and the expected-count columns of signal/cutflow_plot.py
#                              and signal/04_weighted.py. 20 years for DayNight and HEP,
#                              30 years for Sensitivity.
#
# A single geometry can override either one in its config/{cfg}/{cfg}_params.json,
# as a plain number or as an analysis-keyed dict mirroring the schema here.

DEFAULT_ANALYSIS_EXPOSURES: Dict[str, Dict[str, Optional[float]]] = {
    "DEFAULT":     {"PRIMARY": 30.0, "SECONDARY": None},
    "DAYNIGHT":    {"PRIMARY": 30.0, "SECONDARY": None},
    "HEP":         {"PRIMARY": 30.0, "SECONDARY": None},
    "SENSITIVITY": {"PRIMARY": 30.0, "SECONDARY": 10.0},
}

DEFAULT_EVALUATION_EXPOSURES: Dict[str, float] = {
    "DEFAULT":     20.0,
    "DAYNIGHT":    20.0,
    "HEP":         20.0,
    "SENSITIVITY": 30.0,
}


def _normalise_analysis_key(analysis_name: Optional[str]) -> str:
    """'DayNight' → 'DAYNIGHT', 'Sensitivity' → 'SENSITIVITY', None/'' → 'DEFAULT'."""
    if analysis_name is None:
        return "DEFAULT"
    if isinstance(analysis_name, (list, tuple)):
        # A pipeline may carry --analysis as a list; a single entry is unambiguous,
        # several analyses share no common exposure so fall back to DEFAULT.
        if len(analysis_name) != 1:
            return "DEFAULT"
        analysis_name = analysis_name[0]
    key = str(analysis_name).strip().upper()
    return key or "DEFAULT"


def _exposure_stage_value(entry: Any, stage_key: str) -> Optional[float]:
    """Pull one stage out of an ANALYSIS_EXPOSURES entry (dict) or a bare number."""
    if entry is None:
        return None
    if isinstance(entry, dict):
        value = entry.get(stage_key)
        if value is None and stage_key == "PRIMARY":
            value = entry.get("DEFAULT")
        return None if value is None else float(value)
    # A bare number only defines the primary exposure.
    return float(entry) if stage_key == "PRIMARY" else None


def get_analysis_exposure(
    root: str,
    analysis_name: Optional[str] = None,
    stage: str = "PRIMARY",
    config: Optional[str] = None,
    fallback: Optional[float] = None,
) -> Optional[float]:
    """Resolve the exposure (years) an analysis is run to.

    This is the exposure that sets the significance grid, the template scaling and the
    exposure-plot range -- not the livetime results are quoted at (get_evaluation_exposure).

    Resolution order, first hit wins:
      1. ANALYSIS_EXPOSURES in config/{config}/{config}_params.json
         (per-geometry override; number or analysis-keyed dict)
      2. ANALYSIS_EXPOSURES[<analysis>][<stage>] in config/analysis/config.json
      3. ANALYSIS_EXPOSURES["DEFAULT"][<stage>]
      4. DEFAULT_ANALYSIS_EXPOSURES (hard-coded mirror of the shipped config)
      5. `fallback`

    Returns None when the stage is not defined anywhere and no fallback is given
    (the normal outcome for stage="SECONDARY" outside Sensitivity).
    """
    analysis_key = _normalise_analysis_key(analysis_name)
    stage_key = str(stage).upper()

    if config:
        params_path = Path(root) / "config" / str(config) / f"{config}_params.json"
        if params_path.exists():
            params = json.loads(params_path.read_text())
            override = params.get("ANALYSIS_EXPOSURES")
            if isinstance(override, dict):
                value = _exposure_stage_value(override.get(analysis_key), stage_key)
                if value is None:
                    value = _exposure_stage_value(override.get("DEFAULT"), stage_key)
            else:
                value = _exposure_stage_value(override, stage_key)
            if value is not None:
                return value

    configured = load_analysis_info(root).get("ANALYSIS_EXPOSURES", {})
    for key in (analysis_key, "DEFAULT"):
        value = _exposure_stage_value(configured.get(key), stage_key)
        if value is not None:
            return value

    for key in (analysis_key, "DEFAULT"):
        value = _exposure_stage_value(DEFAULT_ANALYSIS_EXPOSURES.get(key), stage_key)
        if value is not None:
            return value

    return fallback


def get_evaluation_exposure(
    root: str,
    analysis_name: Optional[str] = None,
    config: Optional[str] = None,
    fallback: Optional[float] = None,
) -> float:
    """Resolve the reference livetime (years) an analysis quotes its numbers at.

    DayNight and HEP quote 20 years, Sensitivity 30. This is the scale applied to
    spectra and expected-count tables; the exposure the analysis is run to is
    get_analysis_exposure (30 years everywhere).

    Resolution order, first hit wins:
      1. EVALUATION_EXPOSURE_YEARS in config/{config}/{config}_params.json
         (per-geometry override; number or analysis-keyed dict)
      2. EVALUATION_EXPOSURE_YEARS[<analysis>] in config/analysis/config.json
      3. EVALUATION_EXPOSURE_YEARS["DEFAULT"]
      4. DEFAULT_EVALUATION_EXPOSURES (hard-coded mirror of the shipped config)
      5. `fallback`
    """
    analysis_key = _normalise_analysis_key(analysis_name)

    def _value(entry: Any) -> Optional[float]:
        if entry is None:
            return None
        if isinstance(entry, dict):
            picked = entry.get(analysis_key, entry.get("DEFAULT"))
            # An analysis-keyed dict may nest the ANALYSIS_EXPOSURES stage schema.
            if isinstance(picked, dict):
                picked = picked.get("PRIMARY", picked.get("DEFAULT"))
            return None if picked is None else float(picked)
        return float(entry)

    if config:
        params_path = Path(root) / "config" / str(config) / f"{config}_params.json"
        if params_path.exists():
            value = _value(json.loads(params_path.read_text()).get("EVALUATION_EXPOSURE_YEARS"))
            if value is not None:
                return value

    value = _value(load_analysis_info(root).get("EVALUATION_EXPOSURE_YEARS"))
    if value is not None:
        return value

    value = DEFAULT_EVALUATION_EXPOSURES.get(analysis_key, DEFAULT_EVALUATION_EXPOSURES["DEFAULT"])
    return float(value) if value is not None else float(fallback if fallback is not None else 20.0)


def load_folder_config(root: str) -> Dict[str, Any]:
    with open(f"{root}/config/analysis/folder_configs.json") as f:
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


DEFAULT_WORKFLOW_FLAGS: Dict[str, Dict] = {
    "HEP": {
        "pl_isotonic": False,
        "pl_signal_bands": False,
        "pl_conservative_sigma": 0,
        "significance_bins": False,
        "asimov_significance_bins": True,
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


# ── Metrics defaults ─────────────────────────────────────────────────────────
# Controls which significance metrics each macro computes by default.
# More expensive or less-used metrics default to False and are only activated
# with --all_metrics. The --all_metrics flag overrides all METRICS entries to True.

DEFAULT_WORKFLOW_METRICS: Dict[str, Any] = {
    "HEP": {
        "DEFAULT_METRIC": "profile_likelihood",
        "asimov": False,
        "gaussian": False,
        "profile_likelihood": True,
        "significance_bins": False,
        "asimov_significance_bins": True,
    },
    "DAYNIGHT": {
        "DEFAULT_METRIC": "asimov",
        "gaussian": False,
        "asimov": True,
        "raw_variants": False,
        "error_bands": False,          # requires background_error workflow flag
        "significance_bins": False,    # requires significance_bins workflow flag
    },
    "SENSITIVITY": {
        "DEFAULT_METRIC": "sin12",
        "sin12": True,
        "sin13": False,
        "templates": False,
        "nuisance_comparison": False,
        "significance_spectra": False,
    },
}


def get_metrics_config(root: str, analysis_name: str, all_metrics: bool = False) -> Dict[str, Any]:
    """Return which significance metrics to compute for the given analysis.

    Merges DEFAULT_WORKFLOW_METRICS with METRICS overrides from analysis/config.json.
    When all_metrics=True every key is forced to True (honours --all_metrics CLI flag).
    """
    analysis_key = str(analysis_name).upper()
    analysis_info = load_analysis_info(root)
    workflow_section = analysis_info.get("WORKFLOW", {}).get(analysis_key, {})
    configured_metrics = workflow_section.get("METRICS", {})
    defaults = DEFAULT_WORKFLOW_METRICS.get(analysis_key, {})
    merged = {**defaults, **configured_metrics}
    if all_metrics:
        return {k: True for k in merged}
    return merged
