import numpy as np

from typing import Optional


def _solve_beta_hat_background_only(
    observed: np.ndarray,
    background: np.ndarray,
    rel_background_uncertainty: np.ndarray,
) -> np.ndarray:
    """Solve profiled background scale factors for mu=0 with Gaussian priors."""
    obs = np.nan_to_num(np.asarray(observed, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    bkg = np.nan_to_num(np.asarray(background, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    rel_unc = np.nan_to_num(
        np.asarray(rel_background_uncertainty, dtype=float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    beta_hat = np.ones_like(bkg, dtype=float)
    mask = (bkg > 0.0) & (rel_unc > 0.0)
    if not np.any(mask):
        return beta_hat

    n = obs[mask]
    b = bkg[mask]
    sigma2 = np.power(rel_unc[mask], 2)

    # Closed-form root of derivative equation for each bin under mu=0.
    # beta^2 + (b*sigma^2 - 1) * beta - n*sigma^2 = 0
    linear = b * sigma2 - 1.0
    discriminant = np.maximum(np.power(linear, 2) + 4.0 * n * sigma2, 0.0)
    positive_root = (-linear + np.sqrt(discriminant)) / 2.0
    beta_hat[mask] = np.maximum(positive_root, 1e-12)
    return beta_hat


def evaluate_profile_likelihood_discovery(
    signal: np.ndarray,
    background: np.ndarray,
    background_uncertainty: Optional[np.ndarray] = None,
    min_expected: float = 1e-12,
) -> float:
    """
    Evaluate median discovery significance from a profile-likelihood ratio test.

    This computes q0 for Asimov data generated at s+b with mu=1 and tests mu=0,
    profiling per-bin background normalization nuisance parameters constrained by
    Gaussian priors derived from `background_uncertainty / background`.
    """
    signal_arr = np.nan_to_num(np.asarray(signal, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    background_arr = np.nan_to_num(
        np.asarray(background, dtype=float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if signal_arr.size != background_arr.size:
        raise ValueError("signal and background must have the same length")

    if background_uncertainty is None:
        rel_unc = np.zeros_like(background_arr, dtype=float)
    else:
        bkg_unc = np.nan_to_num(
            np.asarray(background_uncertainty, dtype=float),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if bkg_unc.size != background_arr.size:
            raise ValueError("background_uncertainty and background must have the same length")
        rel_unc = np.divide(
            bkg_unc,
            background_arr,
            out=np.zeros_like(background_arr, dtype=float),
            where=background_arr > 0.0,
        )

    observed = signal_arr + background_arr
    expected_sb = np.maximum(observed, min_expected)

    # Asimov s+b point is the unconditional MLE: mu=1 and beta=1.
    ll_sb = np.sum(observed * np.log(expected_sb) - expected_sb)

    beta_hat_null = _solve_beta_hat_background_only(observed, background_arr, rel_unc)
    expected_null = np.maximum(beta_hat_null * background_arr, min_expected)
    ll_null = np.sum(observed * np.log(expected_null) - expected_null)

    pull_mask = rel_unc > 0.0
    if np.any(pull_mask):
        ll_null -= 0.5 * np.sum(
            np.power((beta_hat_null[pull_mask] - 1.0) / rel_unc[pull_mask], 2)
        )

    q0 = max(0.0, 2.0 * (ll_sb - ll_null))
    return float(np.sqrt(q0))


def evaluate_significance(
    signal: np.ndarray,
    background: np.ndarray,
    signal_uncertainty: Optional[np.ndarray] = None,
    background_uncertainty: Optional[np.ndarray] = None,
    type="gaussian",
) -> float:
    """
    Calculate the significance of a signal and background.

    Parameters
    ----------
    signal : float
        The signal value.
    background : float
        The background value.
    signal_uncertainty : float
        The signal uncertainty.
    background_uncertainty : float
        The background uncertainty.
    type : str
        The type of significance to calculate. Currently only 'gaussian' is supported.

    Returns
    -------
    float
        The significance of the signal over background.
    """
    significance = np.zeros_like(signal)
    if type == "gaussian":
        mask = (signal > 0) * (background > 0)
        if signal_uncertainty is None and background_uncertainty is None:
            signal = signal[mask]
            background = background[mask]
            significance[mask] = signal / (background**0.5)

        elif background_uncertainty is None and signal_uncertainty is not None:
            signal = signal[mask]
            background = background[mask]
            signal_uncertainty = signal_uncertainty[mask]
            significance[mask] = signal / ((background + signal_uncertainty**2) ** 0.5)

        elif signal_uncertainty is None and background_uncertainty is not None:
            signal = signal[mask]
            background = background[mask]
            background_uncertainty = background_uncertainty[mask]
            significance[mask] = signal / (
                (background + background_uncertainty**2) ** 0.5
            )

        else:
            signal = signal[mask]
            background = background[mask]
            signal_uncertainty = signal_uncertainty[mask]
            background_uncertainty = background_uncertainty[mask]

            significance[mask] = signal / (
                (+background + signal_uncertainty**2 + background_uncertainty**2) ** 0.5
            )

    elif type == "asimov":
        if signal_uncertainty is None and background_uncertainty is None:
            # Z_A = \sqrt{2 \left[ (s + b) \ln\left(1 + \frac{s}{b} \right) - s \right]}
            mask = (signal > 0) * (background > 0)
            signal = signal[mask]
            background = background[mask]
            significance[mask] = np.sqrt(
                2 * ((signal + background) * np.log(1 + signal / background) - signal)
            )
        elif signal_uncertainty is None:
            # Z = \sqrt{2 \left[ (s + b) \ln \left( \frac{(s + b)(b + \sigma_b^2)}{b^2 + (s + b)\sigma_b^2} \right) - \frac{b^2}{\sigma_b^2} \ln \left(1 + \frac{\sigma_b^2 s}{b(b + \sigma_b^2)} \right) \right]}
            mask = (signal > 0) & (background > 0) & (background_uncertainty >= 0)
            signal = signal[mask]
            background = background[mask]
            background_uncertainty = background_uncertainty[mask]
            significance[mask] = np.sqrt(
                2
                * (
                    (signal + background)
                    * np.log(
                        (signal + background)
                        * (background + background_uncertainty**2)
                        / (
                            background**2
                            + (signal + background) * background_uncertainty**2
                        )
                    )
                    - (background**2 / background_uncertainty**2)
                    * np.log(
                        1
                        + (background_uncertainty**2 * signal)
                        / (background * (background + background_uncertainty**2))
                    )
                )
            )
    elif type in ["profile", "profile_likelihood", "profile-likelihood"]:
        mask = (signal > 0) & (background >= 0)
        if not np.any(mask):
            return significance

        this_signal = signal[mask]
        this_background = background[mask]
        this_bkg_unc = (
            np.asarray(background_uncertainty[mask], dtype=float)
            if background_uncertainty is not None
            else None
        )
        significance[mask] = evaluate_profile_likelihood_discovery(
            this_signal,
            this_background,
            background_uncertainty=this_bkg_unc,
        )

    else:
        raise ValueError(f"Unknown significance type: {type}")

    return significance
