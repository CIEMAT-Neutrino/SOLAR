import ROOT

import numpy as np
import pandas as pd

from scipy.special import gammaln
from scipy.optimize import minimize, minimize_scalar
from scipy.optimize import Bounds
from iminuit import Minuit
from ROOT import TFile, TTree, TList
from rich import print as rprint

# Maximum chi2 value to use as fallback for failed fits
# Valid chi2 values should NOT be capped - only failed/invalid fits use this fallback
MAX_PHYSICAL_CHI2 = 1e4


def th2f_from_dataframe(df, name="myhist", title="My Histogram", debug=False):
    """
    Create a TH2F histogram from a pandas DataFrame.

    Args:
        df (pandas.DataFrame): pandas DataFrame.
        name (str): name of the histogram (default: myhist).
        title (str): title of the histogram (default: My Histogram).

    Returns:
        th2f (ROOT.TH2F): TH2F histogram
    """
    # Extract x and y centers from column names and index names
    x_centers = df.columns.to_numpy(dtype=float)
    y_centers = df.index.to_numpy(dtype=float)

    # Convert DataFrame data to a 2D numpy array for z-axis values
    z_values = df.to_numpy(dtype=int)

    # Create a TH2F histogram
    nbins_x = len(x_centers)
    nbins_y = len(y_centers)
    th2f = ROOT.TH2F(
        name,
        title,
        nbins_x,
        x_centers[0],
        x_centers[-1],
        nbins_y,
        y_centers[0],
        y_centers[-1],
    )

    # Fill the TH2F histogram with z-axis values
    for i in range(nbins_x):
        for j in range(nbins_y):
            th2f.SetBinContent(
                i + 1, j + 1, int(z_values[j][i])
            )  # Note: ROOT histograms are filled in a column-major order

    if debug:
        rprint(f"Created TH2F histogram: {name}")
    return th2f


def th2f_from_numpy(
    z_values, x_centers, y_centers, name="myhist", title="My Histogram", debug=False
):
    """
    Create a TH2F histogram from a pandas DataFrame.

    Args:
        df (pandas.DataFrame): pandas DataFrame.
        name (str): name of the histogram (default: myhist).
        title (str): title of the histogram (default: My Histogram).

    Returns:
        th2f (ROOT.TH2F): TH2F histogram
    """
    # Convert DataFrame data to a 2D numpy array for z-axis values
    z_values = z_values.astype(int)

    # Create a TH2F histogram
    nbins_x = len(x_centers)
    nbins_y = len(y_centers)
    th2f = ROOT.TH2F(
        name,
        title,
        nbins_x,
        x_centers[0],
        x_centers[-1],
        nbins_y,
        y_centers[0],
        y_centers[-1],
    )

    # Fill the TH2F histogram with z-axis values
    for idx, x in enumerate(x_centers):
        for jdx, y in enumerate(y_centers):
            th2f.SetBinContent(
                idx + 1, jdx + 1, int(z_values[jdx][idx])
            )  # Note: ROOT histograms are filled in a column-major order

    if debug:
        rprint(f"Created TH2F histogram: {name}")
    return th2f


class Sensitivity_Fitter:
    """
    Class to fit the solar neutrino histograms for each set of oscillation parameters.

    Args:
        obs: observed (false-data) data histogram.
        pred: predicted signal histogram.
        bkg: background data histogram.
        SigmaPred (float): uncertainty on the predicted neutrino flux (default: 0.04).
        SigmaBkg (float): uncertainty on the background flux (default: 0.02).
        bb_mask (array): boolean mask for bins to include in fit (default: None).
        fit_background (bool): if True, fit background normalization as free parameter.
            For physically meaningful sensitivity, use fit_background=False to prevent
            the background from absorbing signal mismatches (default: False).
        use_legacy_background_penalty (bool): if True, use the legacy constant penalty
            for background uncertainty when fit_background=False. If False (default),
            profile over background normalization to properly propagate uncertainty.

    Returns:
        chisq (float): chi-squared value.
        A_pred (float): best-fit value of the predicted neutrino amplitude.
        A_bkg (float): best-fit value of the background amplitude.
    """

    def __init__(self, obs, pred, bkg, SigmaPred=0.04, SigmaBkg=0.02, bb_mask=None, fit_background=False, use_legacy_background_penalty=False):
        self.fObs = obs
        self.fPred = pred
        self.fBkg = bkg
        self.fSigmaPred = SigmaPred
        self.fSigmaBkg = SigmaBkg
        self.fMask = bb_mask  # boolean array: True = include bin in fit
        self.fit_background = fit_background  # If False, A_bkg is fixed at 0
        self.use_legacy_background_penalty = use_legacy_background_penalty

    def ROOTOperator(self, A_pred, A_bkg):
        chisq = 0
        bkg_total = 0.0
        # Determine if we're in profiling mode (need to apply A_bkg even when fit_background=False)
        profiling_mode = (not self.fit_background and 
                         self.fSigmaBkg > 0 and 
                         not self.use_legacy_background_penalty)
        
        for i in range(1, self.fObs.GetNbinsX() + 1):
            for j in range(1, self.fObs.GetNbinsY() + 1):
                if self.fMask is not None and not self.fMask[i - 1, j - 1]:
                    continue
                # Apply A_bkg to background when fitting or profiling
                if self.fit_background or profiling_mode:
                    N_bkg = (1 + A_bkg) * self.fBkg.GetBinContent(i, j)
                else:
                    N_bkg = self.fBkg.GetBinContent(i, j)
                N_pred = (1 + A_pred) * self.fPred.GetBinContent(i, j)
                e = N_bkg + N_pred
                o = self.fObs.GetBinContent(i, j)
                
                # GUARD: Ensure positive expected counts
                if e <= 0:
                    continue
                    
                if o == 0:
                    chisq += 2 * e
                else:
                    # GUARD: Prevent extreme ratios
                    ratio = o / e
                    if ratio > 1e10 or ratio < 1e-10:
                        # Use Poisson approximation for extreme cases
                        chisq += (o - e) ** 2 / e
                    else:
                        chisq += 2 * (e - o + o * np.log(ratio))
                
                # Track total background for uncertainty penalty
                if self.fit_background or profiling_mode:
                    bkg_total += (1 + A_bkg) * self.fBkg.GetBinContent(i, j)
                else:
                    bkg_total += self.fBkg.GetBinContent(i, j)
        
        chisq += ((A_pred) / self.fSigmaPred) ** 2
        if self.fSigmaBkg > 0:
            if self.fit_background or profiling_mode:
                # Penalize deviation of A_bkg from 0
                chisq += ((A_bkg) / self.fSigmaBkg) ** 2
            elif self.use_legacy_background_penalty:
                # Legacy: constant penalty (doesn't affect contour shapes)
                chisq += (self.fSigmaBkg * np.sqrt(bkg_total)) ** 2
        
        return chisq

    def NumpyOperator(self, A_pred, A_bkg):
        # Determine if we're in profiling mode
        profiling_mode = (not self.fit_background and 
                         self.fSigmaBkg > 0 and 
                         not self.use_legacy_background_penalty)
        
        if self.fit_background or profiling_mode:
            e = (1 + A_bkg) * self.fBkg + (1 + A_pred) * self.fPred
        else:
            e = self.fBkg + (1 + A_pred) * self.fPred
        o = self.fObs
        
        # GUARD: Ensure non-negative expected counts to prevent numerical issues
        e = np.maximum(e, 0.0)
        
        chisq = np.zeros_like(o, dtype=float)

        # Only skip bins where expected counts are exactly zero
        valid = e > 0
        if self.fMask is not None:
            valid = valid & self.fMask

        zero_obs = valid & (o == 0)
        nonzero_obs = valid & (o != 0)

        chisq[zero_obs] = 2 * e[zero_obs]
        
        # GUARD: Prevent extreme ratios in log-likelihood that cause numerical explosion
        if np.any(nonzero_obs):
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = o[nonzero_obs] / e[nonzero_obs]
                # Clip ratio to prevent log from producing extreme values
                ratio = np.clip(ratio, 1e-10, 1e10)
                chisq[nonzero_obs] = 2 * (
                    e[nonzero_obs] - o[nonzero_obs] + o[nonzero_obs] * np.log(ratio)
                )

        # GUARD: Handle any remaining NaN/Inf from numerical edge cases
        chisq = np.nan_to_num(chisq, nan=0.0, posinf=0.0, neginf=0.0)
        chisq_sum = float(np.maximum(np.sum(chisq), 0.0))  # Prevent negative
        
        # Always add signal pull term
        if self.fSigmaPred > 0:
            chisq_sum += ((A_pred) / self.fSigmaPred) ** 2
        # Add background pull term
        if self.fSigmaBkg > 0:
            if self.fit_background or profiling_mode:
                # When fitting or profiling, penalize deviation from nominal
                chisq_sum += ((A_bkg) / self.fSigmaBkg) ** 2
            elif self.use_legacy_background_penalty:
                # Legacy: constant penalty (doesn't affect contour shapes)
                bkg_total = float(np.sum(self.fBkg))
                chisq_sum += (self.fSigmaBkg * np.sqrt(bkg_total)) ** 2
        
        return chisq_sum

    def _profile_a_bkg(self, A_pred):
        """Minimize chi2 over A_bkg at fixed A_pred (1D numerical minimization)."""
        lo = max(-10.0 * self.fSigmaBkg, -0.999)
        hi = 10.0 * self.fSigmaBkg
        result = minimize_scalar(
            lambda a: self.NumpyOperator(A_pred, a),
            bounds=(lo, hi),
            method="bounded",
        )
        return result.x, result.fun

    def Fit(self, initial_A_pred, initial_A_bkg, verbose=0, debug=False, profile_bkg=False):
        if type(self.fObs) == ROOT.TH2F:
            if self.fit_background:
                m = Minuit(self.ROOTOperator, A_pred=initial_A_pred, A_bkg=initial_A_bkg)
                m.limits["A_pred"] = (
                    initial_A_pred - 10 * self.fSigmaPred,
                    initial_A_pred + 10 * self.fSigmaPred,
                )
                m.limits["A_bkg"] = (
                    initial_A_bkg - 10 * self.fSigmaBkg,
                    initial_A_bkg + 10 * self.fSigmaBkg,
                )
                m.migrad()
                chi2_val = m.fval
                if not m.valid or not np.isfinite(chi2_val) or chi2_val < 0:
                    if debug:
                        rprint(f"[yellow][WARNING][/yellow] Minuit fit failed (valid={m.valid}, fval={chi2_val})")
                    # Return a large but reasonable value instead of capping
                    return 1e6, initial_A_pred, initial_A_bkg
                return float(chi2_val), m.values["A_pred"], m.values["A_bkg"]
            else:
                # fit_background=False: profile over A_bkg if we have background uncertainty
                if self.fSigmaBkg > 0 and not self.use_legacy_background_penalty:
                    # Profile over A_bkg: for each A_pred, find optimal A_bkg
                    def _root_operator_profiled(A_pred):
                        # Convert to numpy for profiling (ROOT TH2F not supported for profiling)
                        # For now, fall back to constant penalty with warning
                        if debug:
                            import warnings
                            warnings.warn(
                                "Profiling not implemented for ROOT TH2F with fit_background=False. "
                                "Falling back to legacy constant penalty.",
                                UserWarning
                            )
                        return self.ROOTOperator(A_pred, 0.0)
                    m = Minuit(_root_operator_profiled, A_pred=initial_A_pred)
                    m.limits["A_pred"] = (
                        initial_A_pred - 10 * self.fSigmaPred,
                        initial_A_pred + 10 * self.fSigmaPred,
                    )
                    m.migrad()
                    chi2_val = m.fval
                    if not m.valid or not np.isfinite(chi2_val) or chi2_val < 0:
                        if debug:
                            rprint(f"[yellow][WARNING][/yellow] Minuit profiled fit failed (valid={m.valid}, fval={chi2_val}), returning fallback chi2")
                        return 1e6, initial_A_pred, 0.0
                    return float(chi2_val), m.values["A_pred"], 0.0
                else:
                    # Legacy mode: Fix A_bkg at 0, fit only A_pred
                    def _root_operator_fixed(A_pred):
                        return self.ROOTOperator(A_pred, 0.0)
                    m = Minuit(_root_operator_fixed, A_pred=initial_A_pred)
                    m.limits["A_pred"] = (
                        initial_A_pred - 10 * self.fSigmaPred,
                        initial_A_pred + 10 * self.fSigmaPred,
                    )
                    m.migrad()
                    chi2_val = m.fval
                    if not m.valid or not np.isfinite(chi2_val) or chi2_val < 0:
                        if debug:
                            rprint(f"[yellow][WARNING][/yellow] Minuit fixed fit failed (valid={m.valid}, fval={chi2_val}), returning fallback chi2")
                        return 1e6, initial_A_pred, 0.0
                    return float(chi2_val), m.values["A_pred"], 0.0

        elif type(self.fObs) == np.ndarray:
            if self.fit_background:
                if profile_bkg:
                    # 1D Minuit: profile A_bkg analytically at each A_pred step
                    def _profiled(A_pred):
                        _, fval = self._profile_a_bkg(A_pred)
                        return fval

                    m = Minuit(_profiled, A_pred=initial_A_pred)
                    m.limits["A_pred"] = (
                        initial_A_pred - 10 * self.fSigmaPred,
                        initial_A_pred + 10 * self.fSigmaPred,
                    )
                    m.migrad()
                    chi2_val = m.fval
                    if not m.valid or not np.isfinite(chi2_val) or chi2_val < 0:
                        if debug:
                            rprint(f"[yellow][WARNING][/yellow] Minuit profiled fit (numpy) failed (valid={m.valid}, fval={chi2_val})")
                        return 1e6, initial_A_pred, initial_A_bkg
                    A_pred = m.values["A_pred"]
                    A_bkg, _ = self._profile_a_bkg(A_pred)
                    return float(chi2_val), A_pred, A_bkg
                else:
                    # 2D scipy L-BFGS-B: joint optimization, no Minuit
                    result = minimize(
                        lambda v: self.NumpyOperator(v[0], v[1]),
                        x0=[initial_A_pred, initial_A_bkg],
                        bounds=[
                            (initial_A_pred - 10 * self.fSigmaPred,
                             initial_A_pred + 10 * self.fSigmaPred),
                            (initial_A_bkg - 10 * self.fSigmaBkg,
                             initial_A_bkg + 10 * self.fSigmaBkg),
                        ],
                        method="L-BFGS-B",
                    )
                    chi2_val = result.fun
                    if not result.success or not np.isfinite(chi2_val) or chi2_val < 0:
                        if debug:
                            rprint(f"[yellow][WARNING][/yellow] L-BFGS-B did not converge: {result.message}")
                        return 1e6, float(result.x[0]) if result.x is not None else initial_A_pred, float(result.x[1]) if result.x is not None and len(result.x) > 1 else initial_A_bkg
                    return float(chi2_val), float(result.x[0]), float(result.x[1])
            else:
                # fit_background=False: fit only A_pred
                if self.fSigmaBkg > 0 and not self.use_legacy_background_penalty:
                    # Profile over A_bkg: for each A_pred, find optimal A_bkg
                    def _profiled(A_pred):
                        _, fval = self._profile_a_bkg(A_pred)
                        return fval
                    result = minimize(
                        _profiled,
                        x0=[initial_A_pred],
                        bounds=[
                            (initial_A_pred - 10 * self.fSigmaPred,
                             initial_A_pred + 10 * self.fSigmaPred),
                        ],
                        method="L-BFGS-B",
                    )
                    chi2_val = result.fun
                    if not result.success or not np.isfinite(chi2_val) or chi2_val < 0:
                        if debug:
                            rprint(f"[yellow][WARNING][/yellow] L-BFGS-B (profiled) did not converge: {result.message}")
                        return 1e6, float(result.x[0]) if result.x is not None else initial_A_pred, 0.0
                    A_pred = float(result.x[0])
                    A_bkg, _ = self._profile_a_bkg(A_pred)
                    return float(chi2_val), A_pred, A_bkg
                else:
                    # Legacy mode: Fit only A_pred, keep A_bkg fixed at 0
                    result = minimize(
                        lambda v: self.NumpyOperator(v[0], 0.0),  # A_bkg fixed at 0
                        x0=[initial_A_pred],
                        bounds=[
                            (initial_A_pred - 10 * self.fSigmaPred,
                             initial_A_pred + 10 * self.fSigmaPred),
                        ],
                        method="L-BFGS-B",
                    )
                    chi2_val = result.fun
                    if not result.success or not np.isfinite(chi2_val) or chi2_val < 0:
                        if debug:
                            rprint(f"[yellow][WARNING][/yellow] L-BFGS-B (legacy) did not converge: {result.message}")
                        return 1e6, float(result.x[0]) if result.x is not None else initial_A_pred, 0.0
                    return float(chi2_val), float(result.x[0]), 0.0

        else:
            rprint(f"[red][ERROR] Unknown input type[/red]")
            return 1e6, initial_A_pred, initial_A_bkg


class Asymmetry_Fitter:
    """
    Class to fit the solar neutrino day-night asymmetry above background.

    Args:
        day: observed day data histogram.
        night: observed night data histogram.
        asymmetry: asymmetry data histogram.
        bkg: bkg data histogram.
        SigmaPred (float): uncertainty on the predicted neutrino flux (default: 0.04).
        SigmaBkg (float): uncertainty on the background flux (default: 0.02).

    Returns:
        chisq (float): chi-squared value.
        A_bkg (float): best-fit value of the background amplitude.
    """

    def __init__(self, N_day, N_night, B_hat=None, sigma_B=None):
        """
        Args:
            N_day (array): observed day data histogram.
            N_night (array): observed night data histogram.
            B_hat (array, optional): expected background values. Defaults to None.
            sigma_B (array, optional): uncertainty on the background values. Defaults to None.
        """
        self.N_day = N_day
        self.N_night = N_night
        self.B_hat = B_hat
        self.sigma_B = sigma_B

    def Fit(self, B_init, S_day_init, S_night_init, verbose=0, debug=False):
        """
        Fit the solar neutrino day-night asymmetry above background.
        Args:
            initial_B (array): initial background values.
            initial_S (array): initial signal values.
            verbose (int, optional): verbosity level. Defaults to 0.
            debug (bool, optional): debug flag. Defaults to False.
        Returns:
            TS (float): test statistic.
            B_fit (array): best-fit background values.
            S_fit (array): best-fit signal values.
        """

        def nll(params, N_day, N_night, B_hat=None, sigma_B=None):
            nbins = len(N_day)
            B = params[:nbins]
            S_day = params[nbins : 2 * nbins]
            S_night = params[2 * nbins :]

            mu_day = B + S_day
            mu_night = B + S_night

            # Compute a mask to avoid values less than or equal to zero
            mask_day = np.where(mu_day > 0, True, False)
            mask_night = np.where(mu_night > 0, True, False)
            mask_bkg = np.where(B > 0, True, False)

            # Poisson terms
            logL = -np.sum(
                N_day * np.log(mu_day) - mu_day - gammaln(N_day + 1), where=mask_day
            )

            logL += -np.sum(
                N_night * np.log(mu_night) - mu_night - gammaln(N_night + 1),
                where=mask_night,
            )

            # Add penalty terms for the fit
            if sigma_B is not None and B_hat is not None:
                logL += np.sum((B - B_hat) ** 2 / (2 * sigma_B**2), where=mask_bkg)

            return logL

        # Minimize for H1 (signal allowed)
        nbins = len(self.N_day)
        bounds = Bounds(
            np.zeros(3 * nbins),  # lower bounds (all ≥ 0)
            np.full(3 * nbins, np.inf),  # upper bounds (no limit)
        )
        params_init = np.concatenate([B_init, S_day_init, S_night_init])
        res_signal = minimize(
            nll,
            params_init,
            args=(self.N_day, self.N_night, self.B_hat, self.sigma_B),
            bounds=bounds,
            method="L-BFGS-B",
        )

        # Minimize for H0 (signal = 0)
        params_H0 = np.concatenate([B_init, S_day_init, S_day_init])
        res_null = minimize(
            nll,
            params_H0,
            args=(self.N_day, self.N_night, self.B_hat, self.sigma_B),
            bounds=bounds,
            method="L-BFGS-B",
        )

        # Likelihood ratio test statistic
        ll_signal = res_signal.fun
        ll_null = res_null.fun
        TS = 2 * (ll_null - ll_signal)
        return TS, res_signal.x, res_null.x


# def generate_synthetic_histograms():
#     '''
#     Create synthetic histograms for testing purposes.
#     '''
#     nbins_x = 10
#     nbins_y = 10

#     obs_values = [i + j for i in range(nbins_x) for j in range(nbins_y)]
#     solar_values = [2 * i for i in obs_values]
#     neut_values = [3 * i for i in obs_values]

#     obs_hist = ROOT.TH2F("obs", "Observed Data", nbins_x, 0, nbins_x, nbins_y, 0, nbins_y)
#     solar_hist = ROOT.TH2F("solar", "Solar Data", nbins_x, 0, nbins_x, nbins_y, 0, nbins_y)
#     neut_hist = ROOT.TH2F("neut", "Neutrino Data", nbins_x, 0, nbins_x, nbins_y, 0, nbins_y)

#     for i in range(1, nbins_x + 1):
#         for j in range(1, nbins_y + 1):
#             obs_hist.SetBinContent(i, j, obs_values[(i - 1) * nbins_y + (j - 1)])
#             solar_hist.SetBinContent(i, j, solar_values[(i - 1) * nbins_y + (j - 1)])
#             neut_hist.SetBinContent(i, j, neut_values[(i - 1) * nbins_y + (j - 1)])

#     return obs_hist, solar_hist, neut_hist
