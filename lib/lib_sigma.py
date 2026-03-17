import numpy as np

from typing import Optional


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
    else:
        raise ValueError(f"Unknown significance type: {type}")

    return significance
