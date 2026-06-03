import json
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional

import numpy as np

from .defaults import load_analysis_info, get_folder_flags


def folder_applies_z_fiducial(root: str, folder: str) -> bool:
    """Return True if Z endcap rejection uses a geometric fiducial cut (z_endcap_rejection == 'fiducial')."""
    return get_folder_flags(root, folder)["z_endcap_rejection"] == "fiducial"


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


def build_fiducial_spatial_mask(run: dict, config: str, detector_x: float, detector_y: float, info: dict, folder: str, fiducial: Dict[str, Any]) -> np.ndarray:
    """Spatial-only fiducial mask (X/Y/Z cuts). Shared by analysis scripts."""
    return np.asarray(
        (
            (
                np.absolute(run["Reco"]["RecoX"]) > fiducial["FiducialX"]
                if config == "hd_1x2x6_lateralAPA"
                else (
                    np.absolute(run["Reco"]["RecoX"]) < detector_x / 2 - fiducial["FiducialX"]
                    if config == "hd_1x2x6_centralAPA"
                    else run["Reco"]["RecoX"] < detector_x / 2 - fiducial["FiducialX"]
                )
            )
            * (np.absolute(run["Reco"]["RecoY"]) < detector_y / 2 - fiducial["FiducialY"])
            * (((run["Reco"]["RecoZ"] > fiducial["FiducialZ"] - info["DETECTOR_GAP_Z"])) if folder == "Nominal" else 1)
            * (((run["Reco"]["RecoZ"] < info["DETECTOR_SIZE_Z"] + info["DETECTOR_GAP_Z"] - fiducial["FiducialZ"])) if folder == "Nominal" else 1)
        ),
        dtype=bool,
    )


def build_energy_band_spatial_mask(run: dict, config: str, detector_x: float, detector_y: float, info: dict, folder: str, fiducial: Dict[str, Any], band_fiducials: List[Dict[str, Any]], energy: str) -> np.ndarray:
    """Spatial mask with per-energy-band overrides. Events outside all bands use the global fiducial."""
    spatial_mask = build_fiducial_spatial_mask(run, config, detector_x, detector_y, info, folder, fiducial)
    if not band_fiducials:
        return spatial_mask
    event_energies = run["Reco"][energy]
    for band in band_fiducials:
        e_mask = (event_energies >= band["energy_min"]) & (event_energies < band["energy_max"])
        if not np.any(e_mask):
            continue
        band_fid = {"FiducialX": band["FiducialX"], "FiducialY": band["FiducialY"], "FiducialZ": band["FiducialZ"]}
        spatial_mask[e_mask] = build_fiducial_spatial_mask(run, config, detector_x, detector_y, info, folder, band_fid)[e_mask]
    return spatial_mask


def get_best_fiducial_bands(
    fiducials: Dict[str, Any],
    config: str,
    energy: str,
    analysis_name: Optional[str] = None,
) -> list:
    """Return per-energy-band fiducials stored under 'EnergyBands' key, or empty list."""
    try:
        best = get_best_fiducial(fiducials, config, energy, analysis_name)
    except KeyError:
        return []
    return list(best.get("EnergyBands", []))


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
