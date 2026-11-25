import numpy as np

from typing import Optional, Union


def _solve_global_beta_shift(
    S_total: float,
    B_total: float,
    sigma_rel: float,
) -> float:
    """δ = β̂ − 1 of the global background scale for Asimov data N = S + B (mu=0, Gaussian prior).

    With β = 1 + δ the profile equation  β² + (B·σ² − 1)·β − N·σ² = 0  becomes
        δ² + (1 + B·σ²)·δ − S·σ² = 0,
    whose positive root is taken in the cancellation-free form
        δ = 2·S·σ² / ((1 + B·σ²) + sqrt((1 + B·σ²)² + 4·S·σ²)).
    Solving for β directly loses about log10(B·σ²) digits: with B ~ 1e12 and σ = 0.02
    (the HEP spectrum from 0 MeV, radiological bins included) β̂ was off by ~1e-7, i.e.
    β̂·b off by ~1e5 events per low-energy bin, which turned q0 into rounding noise
    (2026-11-20: isolated 0.67 sigma spike at E_max = 24 MeV in the VD Top hep energy-limit
    cross-check, true value 0.14).
    """
    if B_total <= 0.0 or sigma_rel <= 0.0:
        return 0.0
    sigma2 = sigma_rel ** 2
    a = 1.0 + B_total * sigma2
    c = S_total * sigma2
    disc = a * a + 4.0 * c
    if c >= 0.0:
        return 2.0 * c / (a + disc ** 0.5)
    # negative total signal (not used by the analyses): plain root, beta clipped at 1e-12 below
    return (-a + max(disc, 0.0) ** 0.5) / 2.0


def _solve_global_beta_hat(
    N_total: float,
    B_total: float,
    sigma_rel: float,
) -> float:
    """Solve global background scale factor for mu=0 with a Gaussian prior.

    Positive root of  β² + (B·σ² – 1)·β – N·σ² = 0, computed as 1 + δ
    (_solve_global_beta_shift) to avoid cancellation when B·σ² >> 1.
    """
    if B_total <= 0.0 or sigma_rel <= 0.0:
        return 1.0
    return max(1.0 + _solve_global_beta_shift(N_total - B_total, B_total, sigma_rel), 1e-12)


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

    S_total = float(np.sum(signal_arr))
    B_total = float(np.sum(background_arr))
    # δ = β̂ − 1 from the signal sum: N − B is never formed from two large totals
    delta = _solve_global_beta_shift(S_total, B_total, sigma_rel)
    delta = max(delta, 1e-12 - 1.0)

    # q0 = 2·Σ [n_i·log(n_i/μ_i) − (n_i − μ_i)] + (δ/σ_rel)²,  μ_i = (1 + δ)·b_i  (Asimov n_i = s_i + b_i).
    # Written as n·log1p(d/μ) − d with d = n − μ = s − δ·b, so no per-bin term subtracts two
    # O(b) numbers: each term stays accurate to ~eps·|d| even for b ~ 1e12 (the naive
    # n·log(n/μ) − (n − μ) loses log10(b) digits per bin and its sum can exceed the O(S²/B) q0).
    # Bins with n or μ below min_expected keep the previous flooring:
    #   n_i·log(max(n_i, m)/max(μ_i, m)) − (max(n_i, m) − max(μ_i, m)),  and max(μ_i, m) − m for n_i = 0.
    mu_raw = (1.0 + delta) * background_arr
    expected_null = np.maximum(mu_raw, min_expected)
    observed_floor = np.maximum(observed, min_expected)
    direct = (observed >= min_expected) & (mu_raw >= min_expected)
    diff = np.where(direct, signal_arr - delta * background_arr, observed_floor - expected_null)
    ratio_m1 = np.maximum(diff / expected_null, -1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_bin_llr = np.where(
            observed > 0,
            observed * np.log1p(ratio_m1) - diff,
            expected_null - min_expected,
        )
    q0 = 2.0 * float(np.sum(per_bin_llr))
    if sigma_rel > 0.0:
        q0 += (delta / sigma_rel) ** 2

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
            positive_uncertainty = (
                (signal > 0) & (background > 0) & (background_uncertainty > 0)
            )
            zero_uncertainty = (
                (signal > 0) & (background > 0) & (background_uncertainty == 0)
            )
            if np.any(zero_uncertainty):
                s0 = signal[zero_uncertainty]
                b0 = background[zero_uncertainty]
                significance[zero_uncertainty] = np.sqrt(
                    2 * ((s0 + b0) * np.log1p(s0 / b0) - s0)
                )
            signal = signal[positive_uncertainty]
            background = background[positive_uncertainty]
            background_uncertainty = background_uncertainty[positive_uncertainty]
            significance[positive_uncertainty] = np.sqrt(
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
