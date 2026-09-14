"""
Template Guards for Likelihood Stability
=======================================

Guards to ensure templates used in likelihood computations are numerically stable.
These guards prevent:
- Negative values that cause PL LLR to blow up at high exposures
- Non-finite values (NaN, Inf) that corrupt calculations

The same guards should be applied both at template creation time (proactive) and
optionally at load time (reactive/defensive).
"""

import json
import os

import numpy as np


def clean_and_validate_template(
    arr,
    label: str = "template",
    expected_shape=None,
    debug: bool = False,
    clip_negative: bool = True,
) -> np.ndarray:
    """
    Apply stability guards to a template array.

    Transforms the input array to ensure it is suitable for likelihood computations:
    - Converts to float64
    - Replaces NaN/Inf with 0
    - Optionally clips negative values to 0
    - Optionally validates shape
    - Optionally warns if clipping occurred

    Parameters
    ----------
    arr : array-like
        Input template (any shape, any numeric dtype)
    label : str, default="template"
        Label for warning/error messages
    expected_shape : tuple of int, optional
        If provided, raises ValueError if arr.shape doesn't match
    debug : bool, default=False
        If True, prints warnings when clipping occurs
    clip_negative : bool, default=True
        If True, clips negative values to 0

    Returns
    -------
    np.ndarray
        Cleaned and validated template with dtype=float64

    Raises
    ------
    ValueError
        If expected_shape is provided and doesn't match arr.shape

    Examples
    --------
    >>> import numpy as np
    >>> from lib.template_guards import clean_and_validate_template
    >>> arr = np.array([1.0, -2.0, np.nan, np.inf])
    >>> clean_and_validate_template(arr, "test")
    array([1., 0., 0., 0.])
    """
    # Convert to float64
    arr = np.asarray(arr, dtype=np.float64)

    # Track original extremes for warning
    original_min = float(np.min(arr)) if arr.size > 0 else 0.0

    # Clean non-finite values: replace NaN, +Inf, -Inf with 0
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Clip negative values if requested
    if clip_negative:
        arr = np.clip(arr, 0.0, None)

    # Validate shape if provided
    if expected_shape is not None and arr.shape != expected_shape:
        raise ValueError(
            f"{label} shape mismatch: got {arr.shape}, expected {expected_shape}"
        )

    # Warn if clipping occurred (only when debug=True)
    if debug and clip_negative and original_min < 0:
        try:
            from rich.print import print as rprint
            rprint(
                f"[yellow][WARNING][/yellow] {label} had negative values "
                f"(min={original_min:.2e}, clipped to 0)"
            )
        except ImportError:
            print(
                f"[WARNING] {label} had negative values "
                f"(min={original_min:.2e}, clipped to 0)"
            )

    return arr


# ─── Oscillation-sampling markers for signal templates ──────────────────────────
# Per-point signal templates carry no record of how P_ee was sampled, and the old centre
# sampling of nadir bins aliases Earth regeneration at low dm2. 02_signal_template.py stamps
# every cut it finishes with a marker; 04_best_cuts.py and 06_significance.py refuse templates
# whose marker is missing or does not match the configured sampling, so old and new templates
# can never be mixed in one scan.

TEMPLATE_SAMPLING_VERSION = 1


def template_sampling_marker_path(template_dir, config, name, nhits, adjcl, ophits) -> str:
    return os.path.join(
        template_dir, f"{config}_{name}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}_SAMPLING.json"
    )


def write_template_sampling_marker(
    template_dir, config, name, nhits, adjcl, ophits, scope, backend, nadir_oversample
) -> None:
    """Record how the templates of one cut were built.

    scope: "grid" (full OSCILLATION_GRID) or "scan" (solar/reactor reference points only).
    A scan never downgrades an existing grid marker with the same sampling, because the
    reference points are part of that grid and were regenerated identically.
    """
    path = template_sampling_marker_path(template_dir, config, name, nhits, adjcl, ophits)
    payload = {
        "version": TEMPLATE_SAMPLING_VERSION,
        "scope": scope,
        "backend": backend,
        "nadir_oversample": None if nadir_oversample is None else int(nadir_oversample),
    }
    if scope == "scan" and os.path.exists(path):
        try:
            existing = json.load(open(path))
        except (OSError, ValueError):
            existing = {}
        if (
            existing.get("scope") == "grid"
            and existing.get("backend") == backend
            and existing.get("nadir_oversample") == payload["nadir_oversample"]
            and existing.get("version") == TEMPLATE_SAMPLING_VERSION
        ):
            return
    os.makedirs(template_dir, exist_ok=True)
    if os.path.exists(path):   # PNFS/dCache is write-once: remove before rewriting
        os.remove(path)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def check_template_sampling_marker(
    template_dir, config, name, nhits, adjcl, ophits, backend, nadir_oversample, scopes=("grid",)
) -> tuple:
    """Return (is_current, reason) for the templates of one cut."""
    path = template_sampling_marker_path(template_dir, config, name, nhits, adjcl, ophits)
    if not os.path.exists(path):
        return False, f"no sampling marker {os.path.basename(path)} (templates predate integrated nadir sampling)"
    try:
        marker = json.load(open(path))
    except (OSError, ValueError) as exc:
        return False, f"unreadable sampling marker {path}: {exc}"
    if marker.get("version") != TEMPLATE_SAMPLING_VERSION:
        return False, f"sampling marker version {marker.get('version')} != {TEMPLATE_SAMPLING_VERSION}"
    if marker.get("scope") not in scopes:
        return False, f"templates cover scope '{marker.get('scope')}', need one of {list(scopes)}"
    if marker.get("backend") != backend:
        return False, f"templates built with backend '{marker.get('backend')}', analysis uses '{backend}'"
    if backend != "file" and marker.get("nadir_oversample") != int(nadir_oversample):
        return False, (
            f"templates built with nadir_oversample={marker.get('nadir_oversample')}, "
            f"config OSC_NADIR_OVERSAMPLE={int(nadir_oversample)}"
        )
    return True, "current"
