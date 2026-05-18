import numpy as np

from typing import Optional, Union


def _solve_global_beta_hat(
    N_total: float,
    B_total: float,
    sigma_rel: float,
) -> float:
    """Solve global background scale factor for mu=0 with a Gaussian prior.

    Closed-form positive root of:  β² + (B·σ² – 1)·β – N·σ² = 0
    """
    if B_total <= 0.0 or sigma_rel <= 0.0:
        return 1.0
    sigma2 = sigma_rel ** 2
    linear = B_total * sigma2 - 1.0
    discriminant = max(float(linear ** 2) + 4.0 * N_total * sigma2, 0.0)
    return max((-linear + discriminant ** 0.5) / 2.0, 1e-12)


def evaluate_profile_likelihood_discovery(
    signal: np.ndarray,
    background: np.ndarray,
    background_uncertainty: Optional[Union[float, np.ndarray]] = None,
    min_expected: float = 1e-12,
) -> float:
    """Evaluate median discovery significance from a profile-likelihood ratio test.

    Computes q0 for Asimov data at s+b testing mu=0 with a single global
    background normalization nuisance β ~ Gaussian(1, σ_rel) profiled jointly
    across all bins.  One global β per analysis window correctly represents a
    rate systematic that is fully correlated across bins; independent per-bin
    nuisances are over-parameterized and allow the null hypothesis to absorb
    signal bin-by-bin, causing an artificial plateau in significance vs exposure.

    background_uncertainty:
        float  → global fractional σ_rel applied to the summed background.
        array  → per-bin absolute errors σ_i = σ_rel · b_i; effective σ_rel
                 recovered as Σσ_i / Σb_i (exact when σ_rel is uniform, which
                 is the normal case).  Provided for backward compatibility with
                 single-bin callers in evaluate_significance.
        None   → no background uncertainty; β is fixed at 1.
    """
    signal_arr = np.nan_to_num(np.asarray(signal, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    background_arr = np.nan_to_num(
        np.asarray(background, dtype=float), nan=0.0, posinf=0.0, neginf=0.0,
    )
    if signal_arr.size != background_arr.size:
        raise ValueError("signal and background must have the same length")

    if background_uncertainty is None:
        sigma_rel = 0.0
    elif np.ndim(background_uncertainty) == 0:
        sigma_rel = float(background_uncertainty)
    else:
        bkg_unc = np.nan_to_num(
            np.asarray(background_uncertainty, dtype=float), nan=0.0, posinf=0.0, neginf=0.0,
        )
        B_sum = float(np.sum(background_arr))
        sigma_rel = float(np.sum(bkg_unc)) / B_sum if B_sum > 0.0 else 0.0

    observed = signal_arr + background_arr

    N_total = float(np.sum(observed))
    B_total = float(np.sum(background_arr))
    beta_hat = _solve_global_beta_hat(N_total, B_total, sigma_rel)

    # Compute the LLR directly as the Poisson deviance between Asimov data
    # and the null-hypothesis expectation, avoiding catastrophic cancellation.
    # The naive form  2*(ll_sb - ll_null) subtracts two O(N·logN) quantities
    # whose difference is O(S²/B) at low exposure — losing log10(N·logN·B/S²)
    # significant digits and producing large oscillations in sqrt(q0).
    #
    # The stable form:  q0 = 2·Σ [n_i·log(n_i/μ_null_i) − (n_i − μ_null_i)]
    #                           + ((β̂−1)/σ_rel)²
    # Each per-bin term is O((n−μ)²/μ) when n≈μ — numerically tiny directly.
    expected_null = np.maximum(beta_hat * background_arr, min_expected)
    n_over_mu = np.where(
        (observed > 0) & (expected_null > 0),
        np.maximum(observed, min_expected) / expected_null,
        1.0,
    )
    per_bin_llr = observed * np.log(n_over_mu) - (np.maximum(observed, min_expected) - expected_null)
    q0 = 2.0 * float(np.sum(per_bin_llr))
    if sigma_rel > 0.0:
        q0 += ((beta_hat - 1.0) / sigma_rel) ** 2

    q0 = max(0.0, q0)
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
