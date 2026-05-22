import json
import os
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from .lib_default import load_analysis_info


DEFAULT_SMOOTHING_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "method": "none",
    "preserve_integral": True,
    "qa": False,
    "report": False,
    "component_mode": "all",
    "components": [],
    "dimensions": {
        "1d": {"sigma": 0.0},
        "2d": {"sigma_x": 0.0, "sigma_y": 0.0},
    },
}


DEFAULT_ADAPTIVE_REBIN_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "min_expected_events": 1.0,
    "min_count_probability": 0.6321205588,
    "min_group_bins": 1,
    "max_group_bins": 0,
    "objective": "max_sigma_proxy",
    "isolate_detectable_bins": False,
    "report": False,
}


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _normalize_dimension_key(dimensions: Any) -> str:
    key = str(dimensions).lower()
    return "1d" if key in {"1", "1d"} else "2d"


def _normalize_method(method: Optional[str], enabled: bool) -> str:
    normalized = "none" if method is None else str(method).lower()
    if not enabled:
        return "none"
    return normalized


def _normalize_stage_key(stage: Optional[str]) -> str:
    if stage is None:
        return ""
    return str(stage).strip().upper()


def _strip_meta_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy without nested override blocks used for lookup only."""
    cleaned = deepcopy(config)
    cleaned.pop("ANALYSES", None)
    cleaned.pop("ENERGIES", None)
    return cleaned


def get_smoothing_config(
    root: str,
    analysis_name: Optional[str] = None,
    energy: Optional[str] = None,
    dimensions: Any = "1d",
    stage: Optional[str] = None,
    config_name: Optional[str] = None,
    sample_name: Optional[str] = None,
) -> Dict[str, Any]:
    analysis_info = load_analysis_info(root)
    config = deepcopy(DEFAULT_SMOOTHING_CONFIG)
    smoothing_info = analysis_info.get("SMOOTHING", {})
    config = _deep_update(config, _strip_meta_overrides(smoothing_info))

    analysis_overrides: Dict[str, Any] = {}

    if analysis_name is not None:
        analysis_overrides = smoothing_info.get("ANALYSES", {}).get(
            str(analysis_name).upper(), {}
        )
        config = _deep_update(config, _strip_meta_overrides(analysis_overrides))

    if energy is not None:
        global_energy_overrides = smoothing_info.get("ENERGIES", {}).get(
            str(energy), {}
        )
        config = _deep_update(config, _strip_meta_overrides(global_energy_overrides))

        if analysis_overrides:
            analysis_energy_overrides = analysis_overrides.get("ENERGIES", {}).get(
                str(energy), {}
            )
            config = _deep_update(
                config, _strip_meta_overrides(analysis_energy_overrides)
            )

    stage_key = _normalize_stage_key(stage)
    if stage_key:
        global_stage_overrides = smoothing_info.get("STAGES", {}).get(stage_key, {})
        config = _deep_update(config, _strip_meta_overrides(global_stage_overrides))

        if analysis_overrides:
            analysis_stage_overrides = analysis_overrides.get("STAGES", {}).get(
                stage_key, {}
            )
            config = _deep_update(
                config, _strip_meta_overrides(analysis_stage_overrides)
            )

    dimension_key = _normalize_dimension_key(dimensions)
    config["dimension"] = dimension_key
    config["params"] = deepcopy(config.get("dimensions", {}).get(dimension_key, {}))
    config["method"] = _normalize_method(
        config.get("method"), config.get("enabled", False)
    )

    # Per-config/name sigma from CONFIG_OVERRIDES (written by optimize_smoothing.py --patch).
    # Active only when config_name and sample_name are provided; lower priority than env var.
    if analysis_name is not None and config_name is not None and sample_name is not None:
        _env_key = f"SOLAR_SMOOTHING_SIGMA_{str(analysis_name).upper()}"
        if _env_key not in os.environ:
            _co = smoothing_info.get("CONFIG_OVERRIDES", {})
            _recommended = (
                _co.get(str(config_name), {})
                .get(str(sample_name), {})
                .get(str(analysis_name).upper(), {})
                .get("recommended_sigma")
            )
            if _recommended is not None:
                try:
                    _sigma = float(_recommended)
                    if _sigma > 0:
                        if dimension_key == "1d":
                            config["params"]["sigma"] = _sigma
                        else:
                            config["params"]["sigma_x"] = _sigma
                except (ValueError, TypeError):
                    pass

    # Per-config/name sigma override: set by orchestrator via env var after optimization.
    # Takes priority over analysis.json values so each production uses its own bandwidth.
    if analysis_name is not None:
        env_sigma_str = os.environ.get(f"SOLAR_SMOOTHING_SIGMA_{str(analysis_name).upper()}")
        if env_sigma_str is not None:
            try:
                env_sigma = float(env_sigma_str)
                if env_sigma > 0:
                    if dimension_key == "1d":
                        config["params"]["sigma"] = env_sigma
                    else:
                        config["params"]["sigma_x"] = env_sigma
                        config["params"]["sigma_y"] = env_sigma
            except (ValueError, TypeError):
                pass

    return config


def get_adaptive_rebin_config(
    root: str,
    analysis_name: Optional[str] = None,
) -> Dict[str, Any]:
    analysis_info = load_analysis_info(root)
    config = deepcopy(DEFAULT_ADAPTIVE_REBIN_CONFIG)
    adaptive_info = analysis_info.get("ADAPTIVE_REBIN", {})
    config = _deep_update(config, _strip_meta_overrides(adaptive_info))

    if analysis_name is not None:
        analysis_overrides = adaptive_info.get("ANALYSES", {}).get(
            str(analysis_name).upper(), {}
        )
        config = _deep_update(config, _strip_meta_overrides(analysis_overrides))

    config["enabled"] = bool(config.get("enabled", False))
    config["min_expected_events"] = max(
        0.0, float(config.get("min_expected_events", 1.0))
    )
    min_probability = float(config.get("min_count_probability", 0.6321205588))
    config["min_count_probability"] = float(np.clip(min_probability, 0.0, 1.0 - 1e-12))
    config["min_group_bins"] = max(1, int(config.get("min_group_bins", 1)))
    config["max_group_bins"] = max(0, int(config.get("max_group_bins", 0)))
    objective = str(config.get("objective", "max_sigma_proxy")).strip().lower()
    if objective not in {"max_sigma_proxy", "max_detectable_bins", "greedy_tail"}:
        objective = "max_sigma_proxy"
    config["objective"] = objective
    config["isolate_detectable_bins"] = bool(config.get("isolate_detectable_bins", False))
    config["report"] = bool(config.get("report", False))
    return config


def adaptive_rebin_metadata(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "AdaptiveRebinEnabled": bool(config.get("enabled", False)),
        "AdaptiveRebinMinExpectedEvents": float(config.get("min_expected_events", 1.0)),
        "AdaptiveRebinMinCountProbability": float(
            config.get("min_count_probability", 0.6321205588)
        ),
        "AdaptiveRebinMinGroupBins": int(config.get("min_group_bins", 1)),
        "AdaptiveRebinMaxGroupBins": int(config.get("max_group_bins", 0)),
        "AdaptiveRebinObjective": str(config.get("objective", "max_sigma_proxy")),
        "AdaptiveRebinIsolateDetectableBins": bool(config.get("isolate_detectable_bins", False)),
        "AdaptiveRebinReport": bool(config.get("report", False)),
    }


def _group_sigma_proxy(
    signal_sum: float,
    background_sum: float,
    uncertainty_sum: float,
) -> float:
    signal_value = max(0.0, float(signal_sum))
    background_value = max(0.0, float(background_sum))
    uncertainty_value = max(0.0, float(uncertainty_sum))
    denominator = float(np.sqrt(background_value + uncertainty_value * uncertainty_value))
    if denominator <= 0.0:
        return signal_value
    return signal_value / denominator


def _build_adaptive_tail_starts_max_sigma_proxy(
    signal: np.ndarray,
    background: np.ndarray,
    uncertainty: np.ndarray,
    detection: np.ndarray,
    threshold: float,
    min_group_bins: int,
    max_group_bins: int,
) -> np.ndarray:
    size = signal.size
    if size == 0:
        return np.zeros(0, dtype=int)

    if threshold <= 0 and min_group_bins <= 1 and max_group_bins == 0:
        return np.arange(size, dtype=int)

    signal_prefix = np.zeros(size + 1, dtype=float)
    signal_prefix[1:] = np.cumsum(signal, dtype=float)
    background_prefix = np.zeros(size + 1, dtype=float)
    background_prefix[1:] = np.cumsum(background, dtype=float)
    uncertainty_prefix = np.zeros(size + 1, dtype=float)
    uncertainty_prefix[1:] = np.cumsum(uncertainty, dtype=float)
    detection_prefix = np.zeros(size + 1, dtype=float)
    detection_prefix[1:] = np.cumsum(detection, dtype=float)

    dp_score = np.full(size + 1, -np.inf, dtype=float)
    dp_detectable = np.full(size + 1, -10**9, dtype=int)
    dp_groups = np.full(size + 1, -10**9, dtype=int)
    choice = np.full(size + 1, -1, dtype=int)
    dp_score[size] = 0.0
    dp_detectable[size] = 0
    dp_groups[size] = 0

    for start in range(size - 1, -1, -1):
        max_end = size if max_group_bins <= 0 else min(size, start + max_group_bins)
        min_end = min(size, start + min_group_bins)
        if min_end > max_end:
            min_end = size
            max_end = size

        for end in range(min_end, max_end + 1):
            if not np.isfinite(dp_score[end]):
                continue

            det_sum = float(detection_prefix[end] - detection_prefix[start])
            detectable = 1 if det_sum >= threshold else 0
            if detectable:
                s_sum = float(signal_prefix[end] - signal_prefix[start])
                b_sum = float(background_prefix[end] - background_prefix[start])
                u_sum = float(uncertainty_prefix[end] - uncertainty_prefix[start])
                group_score = _group_sigma_proxy(s_sum, b_sum, u_sum)
            else:
                group_score = 0.0

            candidate_score = float(group_score + dp_score[end])
            candidate_detectable = int(detectable + dp_detectable[end])
            candidate_groups = int(1 + dp_groups[end])

            better_score = candidate_score > dp_score[start] + 1e-12
            score_tie_more_detectable = (
                abs(candidate_score - dp_score[start]) <= 1e-12
                and candidate_detectable > dp_detectable[start]
            )
            score_tie_more_groups = (
                abs(candidate_score - dp_score[start]) <= 1e-12
                and candidate_detectable == dp_detectable[start]
                and candidate_groups > dp_groups[start]
            )

            if better_score or score_tie_more_detectable or score_tie_more_groups:
                dp_score[start] = candidate_score
                dp_detectable[start] = candidate_detectable
                dp_groups[start] = candidate_groups
                choice[start] = end

        if choice[start] < 0:
            choice[start] = size
            det_sum = float(detection_prefix[size] - detection_prefix[start])
            detectable = 1 if det_sum >= threshold else 0
            if detectable:
                s_sum = float(signal_prefix[size] - signal_prefix[start])
                b_sum = float(background_prefix[size] - background_prefix[start])
                u_sum = float(uncertainty_prefix[size] - uncertainty_prefix[start])
                dp_score[start] = _group_sigma_proxy(s_sum, b_sum, u_sum)
            else:
                dp_score[start] = 0.0
            dp_detectable[start] = detectable
            dp_groups[start] = 1

    starts_list = []
    idx = 0
    while idx < size:
        starts_list.append(idx)
        next_idx = int(choice[idx])
        if next_idx <= idx or next_idx > size:
            next_idx = idx + 1
        idx = next_idx
    return np.asarray(starts_list, dtype=int)


def _effective_detection_threshold(config: Dict[str, Any]) -> float:
    min_expected_events = float(config.get("min_expected_events", 1.0))
    min_probability = float(config.get("min_count_probability", 0.6321205588))
    if min_probability <= 0:
        probability_events = 0.0
    else:
        probability_events = -np.log(1.0 - min(min_probability, 1.0 - 1e-12))
    return max(min_expected_events, probability_events)


def _build_adaptive_tail_starts_greedy(
    signal: np.ndarray,
    threshold: float,
    min_group_bins: int,
    max_group_bins: int,
) -> np.ndarray:
    size = signal.size
    if size == 0:
        return np.zeros(0, dtype=int)

    if threshold <= 0 and min_group_bins <= 1 and max_group_bins == 0:
        return np.arange(size, dtype=int)

    groups_reversed = []
    end = size
    while end > 0:
        start = end - 1
        cumulative = signal[start]
        width = 1

        single_bin_allowed = cumulative >= threshold and min_group_bins <= 1
        if not single_bin_allowed:
            while start > 0:
                needs_more = cumulative < threshold or width < min_group_bins
                if not needs_more:
                    break
                if max_group_bins > 0 and width >= max_group_bins:
                    break
                start -= 1
                cumulative += signal[start]
                width += 1

        groups_reversed.append((start, end))
        end = start

    groups = list(reversed(groups_reversed))
    return np.asarray([group_start for group_start, _ in groups], dtype=int)


def _build_adaptive_tail_starts_max_detectable(
    signal: np.ndarray,
    threshold: float,
    min_group_bins: int,
    max_group_bins: int,
) -> np.ndarray:
    size = signal.size
    if size == 0:
        return np.zeros(0, dtype=int)

    if threshold <= 0 and min_group_bins <= 1 and max_group_bins == 0:
        return np.arange(size, dtype=int)

    prefix = np.zeros(size + 1, dtype=float)
    prefix[1:] = np.cumsum(signal, dtype=float)

    # Dynamic programming over contiguous partitions:
    # maximize number of groups above threshold; tie-break on finer partitions.
    dp_detectable = np.full(size + 1, -10**9, dtype=int)
    dp_groups = np.full(size + 1, -10**9, dtype=int)
    choice = np.full(size + 1, -1, dtype=int)
    dp_detectable[size] = 0
    dp_groups[size] = 0

    for start in range(size - 1, -1, -1):
        max_end = size if max_group_bins <= 0 else min(size, start + max_group_bins)
        min_end = min(size, start + min_group_bins)

        if min_end > max_end:
            # If constraints are impossible at the boundary, force one final group.
            min_end = size
            max_end = size

        for end in range(min_end, max_end + 1):
            if dp_detectable[end] < 0:
                continue
            group_sum = float(prefix[end] - prefix[start])
            detectable = 1 if group_sum >= threshold else 0
            candidate_detectable = detectable + int(dp_detectable[end])
            candidate_groups = 1 + int(dp_groups[end])

            better_detectable = candidate_detectable > dp_detectable[start]
            same_detectable_more_groups = (
                candidate_detectable == dp_detectable[start]
                and candidate_groups > dp_groups[start]
            )

            if better_detectable or same_detectable_more_groups:
                dp_detectable[start] = candidate_detectable
                dp_groups[start] = candidate_groups
                choice[start] = end

        if choice[start] < 0:
            # Conservative fallback: make the remaining suffix one group.
            choice[start] = size
            group_sum = float(prefix[size] - prefix[start])
            dp_detectable[start] = 1 if group_sum >= threshold else 0
            dp_groups[start] = 1

    starts = []
    idx = 0
    while idx < size:
        starts.append(idx)
        next_idx = int(choice[idx])
        if next_idx <= idx or next_idx > size:
            next_idx = idx + 1
        idx = next_idx

    return np.asarray(starts, dtype=int)


def build_adaptive_tail_starts(
    expected_signal: np.ndarray, config: Dict[str, Any]
) -> np.ndarray:
    signal = np.clip(
        np.nan_to_num(
            np.asarray(expected_signal, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
        ),
        0.0,
        None,
    )
    threshold = _effective_detection_threshold(config)
    min_group_bins = max(1, int(config.get("min_group_bins", 1)))
    max_group_bins = max(0, int(config.get("max_group_bins", 0)))
    objective = str(config.get("objective", "max_sigma_proxy")).strip().lower()

    if objective == "max_sigma_proxy":
        zeros = np.zeros_like(signal)
        return _build_adaptive_tail_starts_max_sigma_proxy(
            signal, zeros, zeros, signal, threshold, min_group_bins, max_group_bins,
        )

    if objective == "greedy_tail":
        return _build_adaptive_tail_starts_greedy(
            signal,
            threshold,
            min_group_bins,
            max_group_bins,
        )

    return _build_adaptive_tail_starts_max_detectable(
        signal,
        threshold,
        min_group_bins,
        max_group_bins,
    )


def rebin_with_starts(values: np.ndarray, starts: np.ndarray) -> np.ndarray:
    array = np.nan_to_num(
        np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
    )
    starts_array = np.asarray(starts, dtype=int)
    if array.size == 0:
        return array.copy()
    if starts_array.size == 0:
        return np.zeros(0, dtype=float)
    return np.add.reduceat(array, starts_array)


def apply_adaptive_tail_rebin(
    signal: np.ndarray,
    background: np.ndarray,
    background_uncertainty: np.ndarray,
    detection_signal: np.ndarray,
    config: Dict[str, Any],
    apply_detection_mask: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    signal_array = np.nan_to_num(
        np.asarray(signal, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
    )
    background_array = np.nan_to_num(
        np.asarray(background, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
    )
    uncertainty_array = np.nan_to_num(
        np.asarray(background_uncertainty, dtype=float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    detection_array = np.nan_to_num(
        np.asarray(detection_signal, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
    )

    if not bool(config.get("enabled", False)):
        if apply_detection_mask:
            detection_mask = detection_array >= _effective_detection_threshold(config)
            masked_signal = np.where(detection_mask, signal_array, 0.0)
            masked_background = np.where(detection_mask, background_array, 0.0)
            masked_uncertainty = np.where(detection_mask, uncertainty_array, 0.0)
        else:
            masked_signal = signal_array
            masked_background = background_array
            masked_uncertainty = uncertainty_array
        starts = np.arange(signal_array.size, dtype=int)
        return masked_signal, masked_background, masked_uncertainty, starts

    objective = str(config.get("objective", "max_sigma_proxy")).strip().lower()
    threshold = _effective_detection_threshold(config)
    min_group_bins = max(1, int(config.get("min_group_bins", 1)))
    max_group_bins = max(0, int(config.get("max_group_bins", 0)))
    isolate_detectable_bins = bool(config.get("isolate_detectable_bins", True))

    def _starts_for_segment(
        signal_segment: np.ndarray,
        background_segment: np.ndarray,
        uncertainty_segment: np.ndarray,
        detection_segment: np.ndarray,
    ) -> np.ndarray:
        if objective == "max_sigma_proxy":
            return _build_adaptive_tail_starts_max_sigma_proxy(
                signal_segment,
                background_segment,
                uncertainty_segment,
                detection_segment,
                threshold,
                min_group_bins,
                max_group_bins,
            )
        if objective == "greedy_tail":
            return _build_adaptive_tail_starts_greedy(
                detection_segment,
                threshold,
                min_group_bins,
                max_group_bins,
            )
        return _build_adaptive_tail_starts_max_detectable(
            detection_segment,
            threshold,
            min_group_bins,
            max_group_bins,
        )

    size = signal_array.size
    if size == 0:
        starts = np.zeros(0, dtype=int)
    elif not isolate_detectable_bins:
        starts = _starts_for_segment(
            signal_array,
            background_array,
            uncertainty_array,
            detection_array,
        )
    else:
        detectable_mask = detection_array >= threshold
        starts_list = []
        idx = 0
        while idx < size:
            if detectable_mask[idx]:
                # Keep already-detectable bins intact; merge only sub-threshold bins.
                starts_list.append(idx)
                idx += 1
                continue

            end = idx
            while end < size and not detectable_mask[end]:
                end += 1

            local_starts = _starts_for_segment(
                signal_array[idx:end],
                background_array[idx:end],
                uncertainty_array[idx:end],
                detection_array[idx:end],
            )
            if local_starts.size == 0:
                starts_list.append(idx)
            else:
                starts_list.extend((idx + local_starts).tolist())
            idx = end

        starts = np.asarray(sorted(set(starts_list)), dtype=int)
    rebinned_signal = rebin_with_starts(signal_array, starts)
    rebinned_background = rebin_with_starts(background_array, starts)
    rebinned_uncertainty = rebin_with_starts(uncertainty_array, starts)

    if apply_detection_mask:
        rebinned_detection = rebin_with_starts(detection_array, starts)
        detection_mask = rebinned_detection >= _effective_detection_threshold(config)
        rebinned_signal = np.where(detection_mask, rebinned_signal, 0.0)
        rebinned_background = np.where(detection_mask, rebinned_background, 0.0)
        rebinned_uncertainty = np.where(detection_mask, rebinned_uncertainty, 0.0)

    return rebinned_signal, rebinned_background, rebinned_uncertainty, starts


def _normalize_component_name(component: Optional[str]) -> str:
    if component is None:
        return ""
    return str(component).strip().lower()


def should_smooth_component(
    config: Dict[str, Any], component: Optional[str] = None
) -> bool:
    if _normalize_method(config.get("method"), config.get("enabled", False)) == "none":
        return False
    mode = str(config.get("component_mode", "all")).strip().lower()
    selected = [
        _normalize_component_name(item)
        for item in config.get("components", [])
        if str(item).strip()
    ]
    if component is None or mode in {"all", "any", "*"} or not selected:
        return True
    component_name = _normalize_component_name(component)
    if mode in {"only", "include", "includes", "selected", "whitelist"}:
        return component_name in selected
    if mode in {"exclude", "except", "blacklist"}:
        return component_name not in selected
    return True


def get_component_smoothing_config(
    config: Dict[str, Any], component: Optional[str] = None
) -> Dict[str, Any]:
    component_config = deepcopy(config)
    enabled = should_smooth_component(component_config, component)
    component_config["enabled"] = enabled
    component_config["method"] = _normalize_method(
        component_config.get("method"), enabled
    )
    component_config["component"] = component
    return component_config


def smooth_histogram(
    data: np.ndarray,
    method: str = "none",
    preserve_integral: bool = True,
    **params: Any,
) -> np.ndarray:
    array = np.nan_to_num(
        np.asarray(data, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
    )
    if array.size == 0:
        return array.copy()

    original_integral = float(np.sum(array))
    if original_integral <= 0:
        return np.clip(array, 0.0, None)

    normalized_method = str(method).lower()
    if normalized_method == "none":
        smoothed = array.copy()
    elif normalized_method == "gaussian":
        if array.ndim == 1:
            sigma = float(params.get("sigma", 0.0))
            if sigma <= 0.0:
                smoothed = array.copy()
            else:
                smoothed = gaussian_filter1d(array, sigma=sigma, mode="nearest")
        elif array.ndim == 2:
            sigma_x = float(params.get("sigma_x", 0.0))
            sigma_y = float(params.get("sigma_y", 0.0))
            if sigma_x <= 0.0 and sigma_y <= 0.0:
                smoothed = array.copy()
            elif sigma_x <= 0.0:
                smoothed = gaussian_filter1d(
                    array, sigma=sigma_y, axis=0, mode="nearest"
                )
            elif sigma_y <= 0.0:
                smoothed = gaussian_filter1d(
                    array, sigma=sigma_x, axis=1, mode="nearest"
                )
            else:
                smoothed = gaussian_filter(
                    array, sigma=(sigma_y, sigma_x), mode="nearest"
                )
        else:
            raise ValueError(f"Unsupported histogram dimension: {array.ndim}")
    else:
        raise ValueError(f"Unknown smoothing method: {method}")

    smoothed = np.clip(
        np.nan_to_num(smoothed, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None
    )
    if preserve_integral:
        smoothed_integral = float(np.sum(smoothed))
        if smoothed_integral > 0:
            smoothed *= original_integral / smoothed_integral

    return smoothed


def smooth_histogram_with_config(
    data: np.ndarray, config: Dict[str, Any]
) -> np.ndarray:
    return smooth_histogram(
        data,
        method=config.get("method", "none"),
        preserve_integral=config.get("preserve_integral", True),
        **config.get("params", {}),
    )


def smooth_threshold_slice(
    data: np.ndarray,
    threshold_idx: int,
    config: Dict[str, Any],
) -> np.ndarray:
    array = np.asarray(data, dtype=float).copy()
    if threshold_idx >= len(array):
        return array
    array[threshold_idx:] = smooth_histogram_with_config(array[threshold_idx:], config)
    return array


def smoothing_metadata(config: Dict[str, Any]) -> Dict[str, Any]:
    metadata = {
        "SmoothingEnabled": bool(config.get("enabled", False)),
        "SmoothingMethod": config.get("method", "none"),
        "SmoothingPreserveIntegral": bool(config.get("preserve_integral", True)),
        "SmoothingQA": bool(config.get("qa", False)),
        "SmoothingReport": bool(config.get("report", False)),
        "SmoothingComponentMode": str(config.get("component_mode", "all")),
        "SmoothingComponents": list(config.get("components", [])),
    }

    params = config.get("params", {})
    if config.get("dimension") == "1d":
        metadata["SmoothingSigma"] = float(params.get("sigma", 0.0))
    else:
        metadata["SmoothingSigmaX"] = float(params.get("sigma_x", 0.0))
        metadata["SmoothingSigmaY"] = float(params.get("sigma_y", 0.0))

    return metadata


def compute_crossing_exposure(
    exposures: np.ndarray,
    significance_values: np.ndarray,
    threshold: float,
) -> float:
    values = np.asarray(significance_values, dtype=float)
    mask = values > threshold
    if not np.any(mask):
        return 0.0
    return float(np.asarray(exposures, dtype=float)[np.argmax(mask)])


def compute_crossing_summary(
    exposures: np.ndarray,
    raw_significance: np.ndarray,
    smoothed_significance: np.ndarray,
) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for sigma in (2.0, 3.0):
        raw_crossing = compute_crossing_exposure(exposures, raw_significance, sigma)
        smoothed_crossing = compute_crossing_exposure(
            exposures, smoothed_significance, sigma
        )
        delta = smoothed_crossing - raw_crossing
        fraction = delta / raw_crossing if raw_crossing > 0 else 0.0
        label = f"Sigma{int(sigma)}"
        summary[f"Raw{label}Crossing"] = raw_crossing
        summary[f"Smoothed{label}Crossing"] = smoothed_crossing
        summary[f"{label}CrossingDelta"] = delta
        summary[f"{label}CrossingDeltaFraction"] = fraction
    return summary


def smoothing_matrix_1d(size: int, method: str = "none", **params: Any) -> np.ndarray:
    if size <= 0:
        return np.zeros((0, 0), dtype=float)
    if str(method).lower() == "none":
        return np.eye(size, dtype=float)
    basis = np.eye(size, dtype=float)
    operator = np.zeros((size, size), dtype=float)
    for idx in range(size):
        operator[:, idx] = smooth_histogram(
            basis[idx],
            method=method,
            preserve_integral=False,
            **params,
        )
    return operator


def smooth_histogram_errors(
    errors: np.ndarray,
    config: Dict[str, Any],
    counts: Optional[np.ndarray] = None,
    mc_counts: Optional[np.ndarray] = None,
) -> np.ndarray:
    error_array = np.nan_to_num(
        np.asarray(errors, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
    )
    if error_array.ndim != 1:
        raise ValueError(
            "Smoothed histogram error propagation currently supports 1D arrays only"
        )

    variance = error_array**2
    if counts is not None and mc_counts is not None:
        count_array = np.nan_to_num(
            np.asarray(counts, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
        )
        mc_array = np.nan_to_num(
            np.asarray(mc_counts, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
        )
        if count_array.shape == mc_array.shape:
            mc_variance = np.divide(
                count_array**2,
                mc_array,
                out=np.zeros_like(count_array, dtype=float),
                where=mc_array > 0,
            )
            variance = np.maximum(variance, mc_variance)

    operator = smoothing_matrix_1d(
        len(error_array),
        method=config.get("method", "none"),
        **config.get("params", {}),
    )
    propagated_variance = (operator**2) @ variance

    if config.get("preserve_integral", True) and counts is not None:
        count_array = np.nan_to_num(
            np.asarray(counts, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
        )
        unsmoothed = operator @ count_array
        original_integral = float(np.sum(count_array))
        smoothed_integral = float(np.sum(unsmoothed))
        if original_integral > 0 and smoothed_integral > 0:
            propagated_variance *= (original_integral / smoothed_integral) ** 2

    return np.sqrt(np.clip(propagated_variance, 0.0, None))
