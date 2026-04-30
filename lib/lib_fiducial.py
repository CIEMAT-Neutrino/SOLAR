import json
from copy import deepcopy
from typing import Any, Dict, Literal, Optional

from .lib_default import load_analysis_info


DEFAULT_FIDUCIALIZATION_CONFIG: Dict[str, Any] = {
    "combine_mode": "quadrature",
    "significance_type": "gaussian",
    "energy_min": None,
    "energy_max": None,
    "signal_components": ["8B", "hep"],
    "background_components": ["gamma", "neutron"],
}


def get_detector_mass(
    config: str,
    info: Dict[str, Any],
    lar_density: float = 1.396,
    z_size_key: Literal["FD_SIZE_Z", "DETECTOR_SIZE_Z"] = "FD_SIZE_Z",
) -> float:
    detector_x = info["DETECTOR_SIZE_X"] + 2 * info.get("DETECTOR_GAP_X", 0)
    detector_y = info["DETECTOR_SIZE_Y"] + 2 * info.get("DETECTOR_GAP_Y", 0)
    if z_size_key not in info:
        raise KeyError(f"Missing {z_size_key} in config info")
    detector_size_z = info[z_size_key]

    detector_z = detector_size_z + 2 * info.get("DETECTOR_GAP_Z", 0)
    detector_mass = detector_x * detector_y * detector_z * lar_density / 1e9
    if str(config).lower() == "hd_1x2x6_lateralapa":
        detector_mass *= 2
    return float(detector_mass)


def get_full_detector_mass(config: str, info: Dict[str, Any], lar_density: float = 1.396) -> float:
    return get_detector_mass(
        config,
        info,
        lar_density=lar_density,
        z_size_key="FD_SIZE_Z",
    )


def get_workspace_detector_mass(
    config: str,
    info: Dict[str, Any],
    lar_density: float = 1.396,
) -> float:
    return get_detector_mass(
        config,
        info,
        lar_density=lar_density,
        z_size_key="DETECTOR_SIZE_Z",
    )


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def get_fiducialization_config(root: str, analysis_name: Optional[str] = None) -> Dict[str, Any]:
    analysis_info = load_analysis_info(root)
    fiducial_info = analysis_info.get("FIDUCIALIZATION", {})
    config = _deep_update(DEFAULT_FIDUCIALIZATION_CONFIG, fiducial_info)
    if analysis_name is not None:
        overrides = fiducial_info.get("ANALYSES", {}).get(str(analysis_name).upper(), {})
        config = _deep_update(config, overrides)
    return config


def get_best_fiducial(
    fiducials: Dict[str, Any],
    config: str,
    energy: str,
    analysis_name: Optional[str] = None,
) -> Dict[str, Any]:
    config_entries = fiducials.get(config, {})
    if analysis_name is not None:
        analysis_entries = config_entries.get(str(analysis_name).upper())
        if isinstance(analysis_entries, dict) and energy in analysis_entries:
            return analysis_entries[energy]
    if energy in config_entries:
        return config_entries[energy]
    raise KeyError(
        f"Best fiducial not found for config={config}, energy={energy}, analysis={analysis_name}"
    )
