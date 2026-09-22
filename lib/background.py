import os
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .defaults import load_analysis_info, get_folder_flags


def adaptive_mc_thresholds(
    mc_by_component: Mapping[str, Mapping[Any, float]],
    essential: Iterable[str],
    essential_threshold: float,
    nonessential_threshold: float,
) -> Dict[str, float]:
    """Per-component MC-support threshold for a cut/fiducial-volume scan, essential or not.

    ``mc_by_component[component]`` maps each candidate (cut or fiducial point) to that
    component's summed MCCounts there. Essential components always gate at
    ``essential_threshold`` (the existing behaviour, e.g. 04_best_cuts.py --min_background_mc).

    Non-essential components (radiological, for every analysis currently configured) were
    previously exempt from any MC-support gate, so a candidate could be rewarded for a
    background that has near-zero simulated events there rather than a background that is
    genuinely small (see [[sensitivity-one-event-floor]] and the 2026-09-18 energy-study
    finding). Gating them at the SAME threshold as essential components is usually too strict:
    radiological's window support varies by three orders of magnitude across analysis windows
    (e.g. HEP: literally absent for some configs, thin for others; DayNight: normally
    well supported). So each non-essential component gets an adaptive threshold from its own
    grid-wide maximum support:
      - max == 0   -> 0 (no-op: this component is structurally absent from this window for
                     every candidate, not selectively thin at some; gating it would reject
                     every candidate for a physical zero, not a statistics problem)
      - 0 < max < nonessential_threshold -> 1 (require at least one simulated event; the full
                     threshold is unreachable everywhere, but "some support" still beats "none")
      - max >= nonessential_threshold -> nonessential_threshold
    """
    essential_lower = {str(s).lower() for s in essential}
    out: Dict[str, float] = {}
    for component, per_candidate in mc_by_component.items():
        if str(component).lower() in essential_lower:
            out[component] = float(essential_threshold)
            continue
        grid_max = max(per_candidate.values()) if per_candidate else 0.0
        if grid_max <= 0:
            out[component] = 0.0
        elif grid_max < nonessential_threshold:
            out[component] = 1.0
        else:
            out[component] = float(nonessential_threshold)
    return out


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
    study_label: str = None,
) -> List[Any]:
    essential = get_essential_backgrounds(root)
    frames = []
    for sample in get_background_samples(root, analysis_name):
        base = f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/background/{folder.lower()}/{analysis_name.upper()}/{config}/{sample}/{config}_{sample}_{energy}"
        labeled = f"{base}_Rebin_{study_label}.pkl" if study_label else None
        standard = f"{base}_Rebin.pkl"
        if labeled:
            if os.path.exists(labeled):
                filepath = labeled
            else:
                raise RuntimeError(
                    f"Missing labeled background Rebin for study '{study_label}': {labeled}\n"
                    "Run 03_analysis.py for this config/folder/study (including backgrounds) first."
                )
        else:
            filepath = standard
        if os.path.exists(filepath):
            frames.append((sample, filepath))
        elif essential.get(sample, False):
            raise RuntimeError(
                f"Essential background '{sample}' missing for {analysis_name} {config} {energy}: {filepath}"
            )
    return frames
