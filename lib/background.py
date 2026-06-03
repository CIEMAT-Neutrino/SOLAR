import json
import os
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional

from .defaults import load_analysis_info, get_folder_flags


def folder_applies_surface_cut(root: str, folder: str) -> bool:
    """Return True if Z endcap rejection uses SignalParticleSurface < 3 filter (z_endcap_rejection == 'surface')."""
    return get_folder_flags(root, folder)["z_endcap_rejection"] == "surface"


def folder_applies_reduction(root: str, folder: str) -> bool:
    """Return True if the per-component statistical reduction factor should be applied for this folder."""
    return bool(get_folder_flags(root, folder)["apply_reduction"])


def get_folder_reduction_factors(root: str, folder: str) -> Dict[str, float]:
    """Return per-component reduction factors for the given folder (e.g. gamma→3 for Reduced)."""
    return dict(get_folder_flags(root, folder).get("reduction_factors", {}))


def get_component_reduction_factor(root: str, folder: str, component: str) -> float:
    """Return the reduction factor for a single component (by base name). Defaults to 1.0 if not listed."""
    factors = get_folder_reduction_factors(root, folder)
    key = component.split("_")[0].lower()
    return float(factors.get(key, 1.0))


def get_folder_reduction_threshold(root: str, folder: str) -> float:
    """Return the energy threshold (MeV) below which reduction is not applied. Default 0.0."""
    return float(get_folder_flags(root, folder).get("reduction_threshold_mev", 0.0))

DEFAULT_BACKGROUND_CONFIG: Dict[str, Any] = {
    "default": ["gamma", "neutron"],
    "surface_filtered": ["gamma", "neutron"],
    "ANALYSES": {},
    "STYLE": {},
}


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def get_background_config(root: str) -> Dict[str, Any]:
    analysis_info = load_analysis_info(root)
    return _deep_update(DEFAULT_BACKGROUND_CONFIG, analysis_info.get("BACKGROUND_SAMPLES", {}))


def get_background_samples(root: str, analysis_name: Optional[str] = None) -> List[str]:
    config = get_background_config(root)
    if analysis_name is None:
        return list(config.get("default", []))
    return list(config.get("ANALYSES", {}).get(str(analysis_name).upper(), config.get("default", [])))


def get_background_style(root: str, sample_name: str) -> Dict[str, Any]:
    config = get_background_config(root)
    return deepcopy(config.get("STYLE", {}).get(sample_name, {"label": sample_name, "color": "grey", "reduction": 1}))


def is_surface_background(root: str, sample_name: str) -> bool:
    config = get_background_config(root)
    return sample_name.split("_")[0].lower() in set(config.get("surface_filtered", []))


def get_essential_backgrounds(root: str) -> Dict[str, bool]:
    config = get_background_config(root)
    return deepcopy(config.get("ESSENTIAL", {}))


def load_available_background_dataframes(
    root: str,
    analysis_name: str,
    folder: str,
    config: str,
    energy: str,
) -> List[Any]:
    essential = get_essential_backgrounds(root)
    frames = []
    for sample in get_background_samples(root, analysis_name):
        filepath = f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/background/{folder.lower()}/{analysis_name.upper()}/{config}/{sample}/{config}_{sample}_{energy}_Rebin.pkl"
        if os.path.exists(filepath):
            frames.append((sample, filepath))
        elif essential.get(sample, False):
            warnings.warn(
                f"Essential background '{sample}' missing for {analysis_name} {config} {energy}: {filepath}",
                RuntimeWarning,
                stacklevel=2,
            )
    return frames
