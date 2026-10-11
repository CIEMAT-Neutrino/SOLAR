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


def _component_factor(factors: Mapping[str, Any], folder: str, component: str, config: Optional[str], what: str) -> float:
    """Factor of one component (by base name) from a {component: factor} dict; 1.0 if not listed.

    A factor can also be a dict keyed by config name or geometry prefix (e.g. {"hd": 10, "vd": 10});
    the config name takes precedence, then its geometry prefix (text before the first underscore).
    """
    key = component.split("_")[0].lower()
    value = factors.get(key, 1.0)
    if isinstance(value, dict):
        if config is None:
            raise ValueError(f"Folder {folder}: the {key} {what} depends on the configuration; pass config")
        value = value.get(config, value.get(config.split("_")[0], 1.0))
    return float(value)


def get_component_reduction_factor(root: str, folder: str, component: str, config: Optional[str] = None) -> float:
    """Return the statistical reduction factor of a component (reduction_factors). Defaults to 1.0 if not listed."""
    return _component_factor(get_folder_reduction_factors(root, folder), folder, component, config, "reduction factor")


def get_component_model_factor(root: str, folder: str, component: str, config: Optional[str] = None) -> float:
    """Return the background-model factor of a component (background_model_factors), applied at all energies
    in every folder that lists it, independently of apply_reduction. Defaults to 1.0 if not listed."""
    factors = dict(get_folder_flags(root, folder).get("background_model_factors", {}))
    return _component_factor(factors, folder, component, config, "background-model factor")


def neutron_capture_veto_keep(root: str, config: str, analysis: str, energy, main_energy, override=None, reco=None, rows=None):
    """Clusters kept by the neutron-capture veto (config NEUTRON_CAPTURE_VETO) in one analysis.

    Removed: ENERGY_MIN <= energy < ENERGY_MAX with main-cluster Energy < MAIN_CLUSTER_ENERGY_MIN. On for the configs in
    ENABLED_CONFIGS; override (True/False) or the environment variable SOLAR_NCVETO (1/0, set by run_sensitivity for
    the study variants) takes precedence. Returns an all-True mask when off or for analyses not listed.
    """
    import os
    import numpy as np

    energy = np.asarray(energy, dtype=float)
    keep = np.ones(len(energy), dtype=bool)
    veto = load_analysis_info(root).get("NEUTRON_CAPTURE_VETO", {})
    if not veto or analysis not in veto.get("ANALYSES", []):
        return keep
    if override is None and os.environ.get("SOLAR_NCVETO") in ("0", "1"):
        override = os.environ["SOLAR_NCVETO"] == "1"
    on = (config in veto.get("ENABLED_CONFIGS", [])) if override is None else bool(override)
    if not on:
        return keep
    par = {**{k: veto[k] for k in ("ENERGY_MIN", "ENERGY_MAX", "MAIN_CLUSTER_ENERGY_MIN")}, **veto.get("PER_CONFIG", {}).get(config, {})}
    if par.get("MODE") == "classifier":
        # Neutron-capture classifier (03b --target ncap): drop its region clusters scored below the threshold of COMPLETENESS.
        import json as _json
        jpath = f"{root}/config/{config}/marley_final/{config}_calib/{config}_neutron_capture.json"
        if reco is None or "NeutronCaptureScore" not in reco or not os.path.isfile(jpath):
            raise FileNotFoundError(f"Neutron-capture veto MODE classifier for {config}: NeutronCaptureScore or {jpath} missing")
        info = _json.load(open(jpath))
        t = float(info["NCAP_THRESHOLDS"][f"{float(par['COMPLETENESS']):.2f}"])
        score = np.asarray(reco["NeutronCaptureScore"], dtype=float)
        if rows is not None:
            score = score[rows]
        return ~(score < t)
    return ~((energy >= float(par["ENERGY_MIN"])) & (energy < float(par["ENERGY_MAX"]))
             & (np.asarray(main_energy, dtype=float) < float(par["MAIN_CLUSTER_ENERGY_MIN"])))


def score_categories_path(root: str, config: str) -> str:
    """Thresholds of the Sensitivity score categories of one config (written by sensitivity/00_score_categories.py)."""
    return os.path.join(root, "config", config, f"{config}_score_categories.json")


def score_categories(root: str, config: str) -> list:
    """Sensitivity score categories (config SENSITIVITY_SCORE_CATEGORIES): the signal/background classifier score
    thresholds of this config, or [] when the categories are off for it. With K thresholds the Sensitivity templates
    hold K + 1 categories stacked along the nadir axis, the most signal-like first."""
    import json as _json

    cfg = load_analysis_info(root).get("SENSITIVITY_SCORE_CATEGORIES", {})
    if not cfg or config not in cfg.get("ENABLED_CONFIGS", []):
        return []
    path = score_categories_path(root, config)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Sensitivity score categories on for {config} but {path} is missing: run src/physics/sensitivity/00_score_categories.py")
    return [float(t) for t in _json.load(open(path))["THRESHOLDS"]]


def score_category_index(score, thresholds):
    """Category of each cluster: 0 for score >= the highest threshold, ..., len(thresholds) below the lowest."""
    import numpy as np

    t = np.sort(np.asarray(thresholds, dtype=float))
    return len(t) - np.searchsorted(t, np.asarray(score, dtype=float), side="right")


def score_category_of(root: str, config: str, reco) -> "Optional[Any]":
    """Score category of every cluster in reco (score_category_index), or None when the categories are off for config."""
    thresholds = score_categories(root, config)
    if not thresholds:
        return None
    variable = load_analysis_info(root)["SENSITIVITY_SCORE_CATEGORIES"].get("VARIABLE", "SignalBackgroundScore")
    return score_category_index(reco[variable], thresholds)


def n_score_categories(root: str, config: str) -> int:
    """Number of category blocks in the Sensitivity templates of config (1 when the categories are off)."""
    return len(score_categories(root, config)) + 1


def collapse_score_categories(template, rows: int):
    """Sum a category-stacked template (K x rows, N) over its categories; other arrays are returned unchanged."""
    import numpy as np

    a = np.asarray(template, dtype=float)
    if a.ndim != 2 or a.shape[0] == rows or a.shape[0] % rows:
        return a
    return a.reshape(-1, rows, a.shape[1]).sum(axis=0)


def background_weight_divisor(root: str, folder: str, sample_name: str, config: str, true_energy=None):
    """Divisor of the weights of one background sample in a folder.

    The background-model factor (background_model_factors) applies at all energies. In folders with
    apply_reduction the statistical reduction (reduction_factors) multiplies it: for gammas only above
    reduction_threshold_mev of true energy (true_energy, per cluster), for the other components at all energies.
    Returns a scalar, or an array matching true_energy when the gamma threshold applies.
    """
    import numpy as np

    divisor = get_component_model_factor(root, folder, sample_name, config=config)
    if not folder_applies_reduction(root, folder):
        return divisor
    factor = get_component_reduction_factor(root, folder, sample_name, config=config)
    threshold = get_folder_reduction_threshold(root, folder)
    if "gamma" in sample_name.lower() and threshold > 0:
        if true_energy is None:
            raise ValueError(f"Folder {folder}: the {sample_name} reduction applies above {threshold} MeV; pass true_energy")
        return divisor * np.where(np.asarray(true_energy, dtype=float) >= threshold, factor, 1.0)
    return divisor * factor


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


def surface_quality_mask(root: str, sample_name: str, folder: str, surface, reco=None, info=None) -> "np.ndarray":
    """Quality cut of the background samples.

    Surface-generated backgrounds (gamma, neutron): an assigned generation surface (SignalParticleSurface >= 0)
    and, in folders with z_endcap_rejection == "surface" (Reduced, Truncated), no end-cap surface
    (SignalParticleSurface < 3).
    Radiological, with reco and info (the config json): the clusters of the external generators that the
    gamma/neutron samples already model are removed (radiological_external_overlap).
    Other samples pass.
    """
    import numpy as np

    surface = np.asarray(surface)
    mask = np.ones(len(surface), dtype=bool)
    if is_surface_background(root, sample_name):
        mask &= surface >= 0
        if folder_applies_surface_cut(root, folder):
            mask &= surface < 3
    elif sample_name.lower().startswith("radiological") and reco is not None and info is not None:
        mask &= ~radiological_external_overlap(root, reco, info)
    return mask


def radiological_external_overlap(root: str, reco, info) -> "np.ndarray":
    """Clusters of the radiological sample that the dedicated gamma/neutron samples also model.

    The external generators (config EXTERNAL_BACKGROUNDS) follow the SolarNuAna label order of
    generator_labels.fcl: cavern-wall gammas, foam gammas, cavern-wall neutrons, cryostat n-gammas, cavern n-gammas.
    All neutron-generator clusters overlap neutron_final; gamma-generator clusters overlap gamma_final when the
    primary gamma (the parent of the main particle, or the main particle itself) had at least the lower edge of
    the gamma sample's generated range.
    Clusters whose generator SolarNuAna could not match (Generator 0) are treated the same way when their main
    particle is a gamma above that edge: no internal radiological decay emits a gamma of 4 MeV or more, so these
    are neutron-capture gammas of the external generators (e.g. 10.83 MeV, 14N(n,g), in HD Central; 8.65 MeV in
    VD Shielded, where a single such cluster carries a weight of 1.8e5 and sits in the HEP window).
    """
    import numpy as np

    external = list(info.get("EXTERNAL_BACKGROUNDS", []))
    if len(external) != 5:
        return np.zeros(len(np.asarray(reco["Generator"])), dtype=bool)
    generator = np.asarray(reco["Generator"])
    neutron_generator = external[2]
    gamma_generators = [g for g in external if g != neutron_generator]
    gamma_min = float(get_background_config(root).get("GENERATED_MOMENTUM_MEV", {}).get("gamma", [4.0, 14.0])[0])
    main_k = np.asarray(reco["MainK"], dtype=float)
    parent_k = np.asarray(reco["MainParentK"], dtype=float) if "MainParentK" in reco else main_k
    primary_k = np.maximum(main_k, parent_k)
    unmatched_gamma = (generator == 0) & (np.asarray(reco["MainPDG"]) == 22) if "MainPDG" in reco else np.zeros(len(generator), dtype=bool)
    return (generator == neutron_generator) | ((np.isin(generator, gamma_generators) | unmatched_gamma) & (primary_k >= gamma_min))


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
    category: Optional[int] = None,
) -> List[Any]:
    """(sample, Rebin pkl) of the configured backgrounds; category k reads the Sensitivity score-category Rebins."""
    essential = get_essential_backgrounds(root)
    frames = []
    cat = f"_ScoreCat{category}" if category is not None else ""
    for sample in get_background_samples(root, analysis_name):
        base = f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/background/{folder.lower()}/{analysis_name.upper()}/{config}/{sample}/{config}_{sample}_{energy}"
        labeled = f"{base}_Rebin_{study_label}{cat}.pkl" if study_label else None
        standard = f"{base}_Rebin{cat}.pkl"
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
