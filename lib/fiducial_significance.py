"""Fiducial-stage significance of 02_best_fiducial.py.

The per-bin counts of the signal and background components inside an analysis energy window are
combined, smoothed and turned into the significance used to choose the fiducial volume.
"""
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from .background import get_background_config
from .sigma import evaluate_significance
from .smoothing import smooth_histogram_with_config, smooth_histogram_errors

GROUP_COLUMNS = ["Energy", "FiducializedX", "FiducializedY", "FiducializedZ"]
FIDUCIAL_COLUMNS = ["FiducializedX", "FiducializedY", "FiducializedZ"]


def combine_components(df: pd.DataFrame, components: List[str]) -> pd.DataFrame:
    component_df = df[df["Component"].isin(components)].copy()
    if component_df.empty:
        return pd.DataFrame(columns=GROUP_COLUMNS + ["MCCounts", "Counts", "Error+", "Error-"])
    return (
        component_df.groupby(GROUP_COLUMNS)
        .agg({
            "MCCounts": "sum",
            "Counts": "sum",
            "Error+": lambda x: float(np.sqrt(np.sum(np.square(np.asarray(x, dtype=float))))),
            "Error-": lambda x: float(np.sqrt(np.sum(np.square(np.asarray(x, dtype=float))))),
        })
        .reset_index()
    )


def aggregate_significance(significance: np.ndarray, mode: str) -> float:
    values = np.nan_to_num(np.asarray(significance, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if str(mode).lower() == "sum":
        return float(np.sum(values))
    return float(np.sqrt(np.sum(np.square(values))))


def build_significance_payload(
    merged_df: pd.DataFrame,
    significance_type: str,
    combine_mode: str,
    smoothing_config: dict,
    exposure: float,
    energy_min=None,
    energy_max=None,
) -> Optional[Dict]:
    if merged_df.empty:
        return None
    ordered = merged_df.sort_values("Energy").copy()
    energy = ordered["Energy"].to_numpy(dtype=float)
    signal_counts = np.nan_to_num(ordered["CountsSignal"].to_numpy(dtype=float), nan=0.0)
    background_counts = np.nan_to_num(ordered["CountsBackground"].to_numpy(dtype=float), nan=0.0)
    signal_errors = np.nan_to_num(ordered["Error+Signal"].to_numpy(dtype=float), nan=0.0)
    background_errors = np.nan_to_num(ordered["Error+Background"].to_numpy(dtype=float), nan=0.0)
    signal_mc = np.nan_to_num(ordered["MCCountsSignal"].to_numpy(dtype=float), nan=0.0)
    background_mc = np.nan_to_num(ordered["MCCountsBackground"].to_numpy(dtype=float), nan=0.0)

    smoothed_signal_counts = smooth_histogram_with_config(signal_counts, smoothing_config)
    smoothed_background_counts = smooth_histogram_with_config(background_counts, smoothing_config)
    smoothed_signal_errors = smooth_histogram_errors(
        signal_errors, smoothing_config, counts=signal_counts, mc_counts=signal_mc
    )
    smoothed_background_errors = smooth_histogram_errors(
        background_errors, smoothing_config, counts=background_counts, mc_counts=background_mc
    )

    window_mask = np.ones_like(energy, dtype=bool)
    if energy_min is not None:
        window_mask = window_mask & (energy >= float(energy_min))
    if energy_max is not None:
        window_mask = window_mask & (energy <= float(energy_max))

    signal_counts_eval = signal_counts[window_mask]
    background_counts_eval = background_counts[window_mask]
    signal_errors_eval = signal_errors[window_mask]
    background_errors_eval = background_errors[window_mask]
    smoothed_signal_counts_eval = smoothed_signal_counts[window_mask]
    smoothed_background_counts_eval = smoothed_background_counts[window_mask]
    smoothed_signal_errors_eval = smoothed_signal_errors[window_mask]
    smoothed_background_errors_eval = smoothed_background_errors[window_mask]

    significance_kind = str(significance_type).lower()
    if significance_kind == "asimov":
        raw_significance_eval = evaluate_significance(
            exposure * signal_counts_eval,
            exposure * background_counts_eval,
            background_uncertainty=exposure * background_errors_eval,
            type=significance_kind,
        )
        smoothed_significance_eval = evaluate_significance(
            exposure * smoothed_signal_counts_eval,
            exposure * smoothed_background_counts_eval,
            background_uncertainty=exposure * smoothed_background_errors_eval,
            type=significance_kind,
        )
    else:
        raw_significance_eval = evaluate_significance(
            exposure * signal_counts_eval,
            exposure * background_counts_eval,
            exposure * signal_errors_eval,
            exposure * background_errors_eval,
            type=significance_kind,
        )
        smoothed_significance_eval = evaluate_significance(
            exposure * smoothed_signal_counts_eval,
            exposure * smoothed_background_counts_eval,
            exposure * smoothed_signal_errors_eval,
            exposure * smoothed_background_errors_eval,
            type=significance_kind,
        )
    raw_significance_eval = np.nan_to_num(raw_significance_eval, nan=0.0, posinf=0.0, neginf=0.0)
    smoothed_significance_eval = np.nan_to_num(smoothed_significance_eval, nan=0.0, posinf=0.0, neginf=0.0)

    raw_significance = np.zeros_like(energy, dtype=float)
    smoothed_significance = np.zeros_like(energy, dtype=float)
    raw_significance[window_mask] = raw_significance_eval
    smoothed_significance[window_mask] = smoothed_significance_eval

    if significance_kind == "gaussian":
        # Correct global formula: S_total / sqrt(B_total + Σσ_s² + Σσ_b²)
        # Per-bin quadrature overestimates significance when bins differ in S/B ratio.
        def _gaussian_global(s, b, se, be):
            s_sum = float(np.sum(s))
            denom_sq = float(np.sum(b) + np.sum(np.square(se)) + np.sum(np.square(be)))
            return float(s_sum / np.sqrt(denom_sq)) if denom_sq > 0 else 0.0

        raw_total = _gaussian_global(
            exposure * signal_counts_eval,
            exposure * background_counts_eval,
            exposure * signal_errors_eval,
            exposure * background_errors_eval,
        )
        smoothed_total = _gaussian_global(
            exposure * smoothed_signal_counts_eval,
            exposure * smoothed_background_counts_eval,
            exposure * smoothed_signal_errors_eval,
            exposure * smoothed_background_errors_eval,
        )
    else:
        # Asimov: q0 is additive over independent bins → sqrt(Σz_i²) is exact.
        raw_total = aggregate_significance(raw_significance, combine_mode)
        smoothed_total = aggregate_significance(smoothed_significance, combine_mode)

    return {
        "Energy": energy,
        "RawSignal": exposure * signal_counts,
        "RawBackground": exposure * background_counts,
        "SmoothedSignal": exposure * smoothed_signal_counts,
        "SmoothedBackground": exposure * smoothed_background_counts,
        "RawSignificance": raw_significance,
        "SmoothedSignificance": smoothed_significance,
        "RawTotal": raw_total,
        "SmoothedTotal": smoothed_total,
    }


def get_essential_background_components(root_path: str, components: List[str]) -> Set[str]:
    background_cfg = get_background_config(root_path)
    essential_map = {
        str(component).lower(): bool(is_essential)
        for component, is_essential in background_cfg.get("ESSENTIAL", {}).items()
    }
    return {
        str(component)
        for component in components
        if essential_map.get(str(component).lower(), False)
    }
