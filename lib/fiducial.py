import json
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

_DEFAULT_POS_KEYS: Tuple[str, str, str] = ("RecoX", "RecoY", "RecoZ")
_TRUTH_POS_KEYS:   Tuple[str, str, str] = ("SignalParticleX", "SignalParticleY", "SignalParticleZ")


def get_truth_pos_keys(root: str, sample_name: str) -> Tuple[str, str, str]:
    """Truth-position branches to fiducialise `sample_name` on.

    Defaults to SignalParticle*, but samples without a valid signal particle
    (radiological carries SignalParticleSurface == -1 for every event) are
    mapped to their Main* energy-deposit position via
    BACKGROUND_SAMPLES.truth_position_keys in config/analysis/backgrounds.json.
    """
    samples_config = load_analysis_info(root).get("BACKGROUND_SAMPLES", {})
    overrides = samples_config.get("truth_position_keys", {})
    keys = overrides.get(str(sample_name).split("_")[0].lower())
    if keys is None:
        return _TRUTH_POS_KEYS
    if len(keys) != 3:
        raise ValueError(
            f"truth_position_keys for '{sample_name}' must list exactly 3 branches, got {keys}"
        )
    return (keys[0], keys[1], keys[2])

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


def build_fiducial_spatial_mask(
    run: dict,
    config: str,
    detector_x: float,
    detector_y: float,
    info: dict,
    folder: str,
    fiducial: Dict[str, Any],
    pos_keys: Tuple[str, str, str] = _DEFAULT_POS_KEYS,
) -> np.ndarray:
    """Spatial-only fiducial mask (X/Y/Z cuts). Shared by analysis scripts.

    pos_keys selects which coordinate arrays to read from run["Reco"]:
      default  = ("RecoX", "RecoY", "RecoZ")   — reco flash-matched position
      truth    = ("SignalParticleX", "SignalParticleY", "SignalParticleZ")
    """
    xk, yk, zk = pos_keys
    return np.asarray(
        (
            (
                np.absolute(run["Reco"][xk]) > fiducial["FiducialX"]
                if config == "hd_1x2x6_lateralAPA"
                else (
                    np.absolute(run["Reco"][xk]) < detector_x / 2 - fiducial["FiducialX"]
                    if config == "hd_1x2x6_centralAPA"
                    else run["Reco"][xk] < detector_x / 2 - fiducial["FiducialX"]
                )
            )
            * (np.absolute(run["Reco"][yk]) < detector_y / 2 - fiducial["FiducialY"])
            * (((run["Reco"][zk] > fiducial["FiducialZ"] - info["DETECTOR_GAP_Z"])) if folder == "Nominal" else 1)
            * (((run["Reco"][zk] < info["DETECTOR_SIZE_Z"] + info["DETECTOR_GAP_Z"] - fiducial["FiducialZ"])) if folder == "Nominal" else 1)
        ),
        dtype=bool,
    )


def build_energy_band_spatial_mask(
    run: dict,
    config: str,
    detector_x: float,
    detector_y: float,
    info: dict,
    folder: str,
    fiducial: Dict[str, Any],
    band_fiducials: List[Dict[str, Any]],
    energy: str,
    pos_keys: Tuple[str, str, str] = _DEFAULT_POS_KEYS,
) -> np.ndarray:
    """Spatial mask with per-energy-band overrides. Events outside all bands use the global fiducial."""
    spatial_mask = build_fiducial_spatial_mask(run, config, detector_x, detector_y, info, folder, fiducial, pos_keys)
    if not band_fiducials:
        return spatial_mask
    event_energies = run["Reco"][energy]
    for band in band_fiducials:
        e_mask = (event_energies >= band["energy_min"]) & (event_energies < band["energy_max"])
        if not np.any(e_mask):
            continue
        band_fid = {"FiducialX": band["FiducialX"], "FiducialY": band["FiducialY"], "FiducialZ": band["FiducialZ"]}
        spatial_mask[e_mask] = build_fiducial_spatial_mask(run, config, detector_x, detector_y, info, folder, band_fid, pos_keys)[e_mask]
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
