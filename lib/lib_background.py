import json
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional

from .lib_default import load_analysis_info

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


def load_available_background_dataframes(
    root: str,
    analysis_name: str,
    folder: str,
    config: str,
    energy: str,
) -> List[Any]:
    frames = []
    for sample in get_background_samples(root, analysis_name):
        filepath = f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/background/{folder.lower()}/{analysis_name.upper()}/{config}/{sample}/{config}_{sample}_{energy}_Rebin.pkl"
        if os.path.exists(filepath):
            frames.append((sample, filepath))
    return frames
