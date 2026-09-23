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

_TRUTH_PURITY_DEFAULTS = {
    "min_purity": 0.0,                  # MatchedOpFlashPur threshold; <= 0 disables the purity term
    "drift_tolerance_cm": 100.0,         # |RecoX - truthX|; <= 0 disables (gross mismatch = wrong flash)
    "transverse_tolerance_cm": 50.0,     # |RecoY/Z - truthY/Z|; <= 0 disables (cluster is not the truth deposit)
    "apply_to": ["marley", "gamma", "neutron", "radiological"],   # samples the cut applies to
    "per_sample": {},                    # {sample: {min_purity/drift_tolerance_cm/transverse_tolerance_cm}}
}


def truth_match_purity_config(root: str) -> Dict[str, float]:
    """BACKGROUND_SAMPLES.truth_match_purity from config/analysis/backgrounds.json, with defaults."""
    cfg = dict(_TRUTH_PURITY_DEFAULTS)
    cfg.update({k: v for k, v in load_analysis_info(root).get("BACKGROUND_SAMPLES", {}).get("truth_match_purity", {}).items()
                if not str(k).startswith("_")})
    return cfg


def truth_match_purity_mask(
    run: dict,
    root: str,
    sample_name: str,
    min_purity: Optional[float] = None,
    drift_tol: Optional[float] = None,
    transverse_tol: Optional[float] = None,
) -> np.ndarray:
    """Truth-fiducial purity cut: keep events whose matched flash is genuinely theirs.

    The reco fiducial cut silently rejects backgrounds whose flash match is wrong (their
    reconstructed drift coordinate lands outside the volume); a truth-position fiducial cut
    loses that rejection, so the fiduc_truth study makes it explicit here, using truth in
    every position-related quantity:
      * MatchedOpFlashPur >= min_purity   (backtracked purity of the matched flash), and
      * |RecoX - truthX| <= drift_tolerance_cm, |RecoY/Z - truthY/Z| <= transverse_tolerance_cm
        (the reconstructed position agrees with the true one, truth keys per sample as in
        get_truth_pos_keys).
    Either term is skipped when its branches are absent or its threshold is <= 0. A sample whose
    backtracking is unreliable (radiological) fails the purity term wholesale, which is the
    conservative direction for a study that only wants to keep well-matched events.
    """
    cfg = truth_match_purity_config(root)
    key = str(sample_name).split("_")[0].lower()
    reco = run["Reco"]
    ok = np.ones(len(reco["NHits"]), dtype=bool)
    if key not in [str(s).lower() for s in cfg.get("apply_to", [])]:
        return ok
    cfg.update(cfg.get("per_sample", {}).get(key, {}))
    min_purity = cfg["min_purity"] if min_purity is None else float(min_purity)
    drift_tol = cfg["drift_tolerance_cm"] if drift_tol is None else float(drift_tol)
    transverse_tol = cfg["transverse_tolerance_cm"] if transverse_tol is None else float(transverse_tol)
    if min_purity > 0 and "MatchedOpFlashPur" in reco:
        pur = np.asarray(reco["MatchedOpFlashPur"], dtype=float)
        ok &= np.isfinite(pur) & (pur >= min_purity)
    tx, ty, tz = get_truth_pos_keys(root, str(sample_name).split("_")[0].lower())
    for reco_key, truth_key, tol in (("RecoX", tx, drift_tol), ("RecoY", ty, transverse_tol), ("RecoZ", tz, transverse_tol)):
        if tol > 0 and reco_key in reco and truth_key in reco:
            delta = np.abs(np.asarray(reco[reco_key], dtype=float) - np.asarray(reco[truth_key], dtype=float))
            ok &= np.isfinite(delta) & (delta <= tol)
    return ok



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


def accepted_flash_planes(
    plane: np.ndarray,
    root: str,
    membrane_veto: bool = True,
) -> np.ndarray:
    """Optical-match planes accepted by the analysis.

    Plane 0 is the cathode (VD) / APA (HD) photon-detector plane. HD only ever
    reports planes -1 and 0, but VD additionally reports membrane (1, 2) and
    endcap (3, 4) matches. The default QUALITY_CUTS.OPFLASH_PLANE == 0 therefore
    acts as a membrane veto: free for HD, but it discards ~22% of VD clusters
    whose drift coordinate is reconstructed as well as plane 0's (>94% within
    10 cm, against 96.8% for the cathode).

    membrane_veto=False keeps every real plane instead. A failed match always
    carries plane == -1 together with PE == 0, so callers pairing this with
    `MatchedOpFlashPE > 0` still reject unmatched clusters either way.
    """
    if membrane_veto:
        return np.asarray(
            plane == load_analysis_info(root)["QUALITY_CUTS"]["OPFLASH_PLANE"],
            dtype=bool,
        )
    return np.asarray(plane >= 0, dtype=bool)


def truth_containment_mask(run: dict, info: dict, pos_keys: Tuple[str, str, str]) -> np.ndarray:
    """True where the (truth) position lies inside the active volume plus BACKGROUND_SAMPLES.truth_containment_margin_cm."""
    from . import root

    n = len(run["Reco"][pos_keys[0]])
    margin = float(load_analysis_info(str(root)).get("BACKGROUND_SAMPLES", {}).get("truth_containment_margin_cm", 10.0))
    inside = np.ones(n, dtype=bool)
    if margin < 0:
        return inside
    for axis, key in zip("XYZ", pos_keys):
        pos = np.asarray(run["Reco"][key], dtype=float)
        inside &= (pos >= info[f"DETECTOR_MIN_{axis}"] - margin) & (pos <= info[f"DETECTOR_MAX_{axis}"] + margin)
    return inside


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
      truth    = ("SignalParticleX", "SignalParticleY", "SignalParticleZ") for marley signal
      truth    = ("MainX", "MainY", "MainZ") for neutron backgrounds
      truth    = ("EndX", "EndY", "EndZ") for gamma/radiological backgrounds
    """
    from . import root

    # Z is only geometrically fiducialised for folders whose endcap rejection is "fiducial" (Nominal); Truncated/Reduced
    # reject the Z endcaps via the SignalParticleSurface filter instead (config/analysis/folder_configs.json). This used
    # to hard-code `folder == "Nominal"`, which happens to agree with the flag for every folder configured so far -- reading
    # the flag instead removes that coincidence and keeps the two in sync if a folder's endcap-rejection mode ever changes.
    _apply_z = folder_applies_z_fiducial(str(root), folder)
    xk, yk, zk = pos_keys
    mask = np.asarray(
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
            * (((run["Reco"][zk] > fiducial["FiducialZ"] - info["DETECTOR_GAP_Z"])) if _apply_z else 1)
            * (((run["Reco"][zk] < info["DETECTOR_SIZE_Z"] + info["DETECTOR_GAP_Z"] - fiducial["FiducialZ"])) if _apply_z else 1)
        ),
        dtype=bool,
    )
    if tuple(pos_keys) != _DEFAULT_POS_KEYS:
        mask &= truth_containment_mask(run, info, pos_keys)
    return mask


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
