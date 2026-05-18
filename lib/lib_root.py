import ROOT

import numpy as np
import pandas as pd

from scipy.special import gammaln
from scipy.optimize import minimize, minimize_scalar
from scipy.optimize import Bounds
from iminuit import Minuit
from ROOT import TFile, TTree, TList
from rich import print as rprint


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
        solar: solar data histogram.
        bkg: bkg data histogram.
        SigmaPred (float): uncertainty on the predicted neutrino flux (default: 0.04).
        SigmaBkg (float): uncertainty on the background flux (default: 0.02).

    Returns:
        chisq (float): chi-squared value.
        A_pred (float): best-fit value of the predicted neutrino amplitude.
        A_bkg (float): best-fit value of the background amplitude.
    """

    def __init__(self, obs, pred, bkg, SigmaPred=0.04, SigmaBkg=0.02, bb_mask=None):
        self.fObs = obs
        self.fPred = pred
        self.fBkg = bkg
        self.fSigmaPred = SigmaPred
        self.fSigmaBkg = SigmaBkg
        self.fMask = bb_mask  # boolean array: True = include bin in fit

    def ROOTOperator(self, A_pred, A_bkg):
        chisq = 0
        for i in range(1, self.fObs.GetNbinsX() + 1):
            for j in range(1, self.fObs.GetNbinsY() + 1):
                if self.fMask is not None and not self.fMask[i - 1, j - 1]:
                    continue
                N_bkg = (1 + A_bkg) * self.fBkg.GetBinContent(i, j)
                N_pred = (1 + A_pred) * self.fPred.GetBinContent(i, j)
                e = N_bkg + N_pred
                o = self.fObs.GetBinContent(i, j)
                if e == 0:
                    continue
                if o == 0:
                    chisq += 2 * e
                else:
                    chisq += 2 * (e - o + o * np.log(o / e))
        chisq += ((A_pred) / self.fSigmaPred) ** 2 + ((A_bkg) / self.fSigmaBkg) ** 2
        return chisq

    def NumpyOperator(self, A_pred, A_bkg):
        e = (1 + A_bkg) * self.fBkg + (1 + A_pred) * self.fPred
        o = self.fObs
        chisq = np.zeros_like(o, dtype=float)

        valid = e != 0
        if self.fMask is not None:
            valid = valid & self.fMask

        zero_obs = valid & (o == 0)
        nonzero_obs = valid & (o != 0)

        chisq[zero_obs] = 2 * e[zero_obs]
        chisq[nonzero_obs] = 2 * (
            e[nonzero_obs] - o[nonzero_obs] + o[nonzero_obs] * np.log(o[nonzero_obs] / e[nonzero_obs])
        )

        chisq_sum = float(chisq.sum())
        if self.fSigmaBkg > 0 and self.fSigmaPred > 0:
            chisq_sum += ((A_pred) / self.fSigmaPred) ** 2 + ((A_bkg) / self.fSigmaBkg) ** 2
        elif self.fSigmaBkg > 0 and self.fSigmaPred <= 0:
            chisq_sum += ((A_bkg) / self.fSigmaBkg) ** 2
        elif self.fSigmaPred > 0 and self.fSigmaBkg <= 0:
            chisq_sum += ((A_pred) / self.fSigmaPred) ** 2
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
            return m.fval, m.values["A_pred"], m.values["A_bkg"]

        elif type(self.fObs) == np.ndarray:
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
                A_pred = m.values["A_pred"]
                A_bkg, _ = self._profile_a_bkg(A_pred)
                return m.fval, A_pred, A_bkg
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
                if not result.success and debug:
                    rprint(f"[yellow][WARNING][/yellow] L-BFGS-B did not converge: {result.message}")
                return result.fun, float(result.x[0]), float(result.x[1])

        else:
            rprint(f"[red][ERROR] Unknown input type[/red]")
            return -1, -1, -1


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
