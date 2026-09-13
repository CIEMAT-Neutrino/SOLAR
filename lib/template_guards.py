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
