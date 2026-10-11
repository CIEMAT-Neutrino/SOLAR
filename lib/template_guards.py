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


def check_signal_template_inputs(
    template_path, root, config, name, folder, *, truth_fiducial=False,
    fiducial_folder=None,
) -> tuple:
    """Reject a signal template older than inputs that change its selected events.

    Sampling markers describe the oscillation calculation, but do not cover the
    fiducial volume or signal/background classifier. Those can change while a
    numerically valid template remains on disk. Compare the actual pkl timestamp
    (not its marker) so a marker rewrite cannot make old counts appear current.
    """
    if not os.path.isfile(template_path):
        return False, f"missing signal template {template_path}"
    root = os.fspath(root)
    analysis_config = os.path.join(root, "config", "analysis", "config.json")
    fiducial_stem = "BestFiducials_fiduc_truth" if truth_fiducial else "BestFiducials"
    fiducial_path = os.path.join(
        root, "config", "analysis", "fiducial", (fiducial_folder or folder).lower(),
        f"{fiducial_stem}.json",
    )
    dependencies = [analysis_config, fiducial_path]
    # A branch switch or reconstruction/weighting fix changes the response even
    # if the JSON inputs have not moved. Only existing files are included so the
    # helper also works in small standalone validation environments.
    source_inputs = (
        "lib/cluster.py", "lib/weights.py", "lib/fiducial.py",
        "src/physics/sensitivity/02_signal_template.py",
    )
    dependencies.extend(
        path for relative in source_inputs
        if os.path.isfile(path := os.path.join(root, relative))
    )
    score_categories_json = os.path.join(root, "config", config, f"{config}_score_categories.json")
    if os.path.isfile(score_categories_json):
        dependencies.append(score_categories_json)
    if not os.path.isfile(analysis_config):
        return False, f"missing analysis configuration {analysis_config}"
    if not os.path.isfile(fiducial_path):
        return False, f"missing fiducial definition {fiducial_path}"
    with open(analysis_config) as handle:
        classifier_enabled = json.load(handle).get("SIGNAL_BACKGROUND_CLASSIFIER", False)
    if classifier_enabled:
        for sample in dict.fromkeys((name, "marley_final" if name.endswith(("_final", "_test")) else "")):
            if not sample:
                continue
            classifier_path = os.path.join(
                root, "config", config, sample, f"{config}_calib",
                f"{config}_signal_background.json",
            )
            if os.path.isfile(classifier_path):
                dependencies.append(classifier_path)
                break
    template_mtime = os.path.getmtime(template_path)
    for dependency in dependencies:
        if os.path.getmtime(dependency) > template_mtime:
            return False, f"{os.path.basename(dependency)} is newer than {os.path.basename(template_path)}"
    return True, "current"


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
