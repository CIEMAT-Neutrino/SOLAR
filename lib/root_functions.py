import ROOT

import numpy as np
import pandas as pd

from iminuit import Minuit
from ROOT    import TFile, TTree, TList

from .io_functions import print_colored

def th2f_from_dataframe(df, name="myhist", title="My Histogram", debug=False):
    '''
    Create a TH2F histogram from a pandas DataFrame.
    
    Args:
        df (pandas.DataFrame): pandas DataFrame.
        name (str): name of the histogram (default: myhist).
        title (str): title of the histogram (default: My Histogram).

    Returns:
        th2f (ROOT.TH2F): TH2F histogram
    '''
    # Extract x and y centers from column names and index names
    x_centers = df.columns.to_numpy(dtype=float)
    y_centers = df.index.to_numpy(dtype=float)

    # Convert DataFrame data to a 2D numpy array for z-axis values
    z_values = df.to_numpy(dtype=int)

    # Create a TH2F histogram
    nbins_x = len(x_centers)
    nbins_y = len(y_centers)
    th2f = ROOT.TH2F(name, title, nbins_x, x_centers[0], x_centers[-1], nbins_y, y_centers[0], y_centers[-1])

    # Fill the TH2F histogram with z-axis values
    for i in range(nbins_x):
        for j in range(nbins_y):
            th2f.SetBinContent(i + 1, j + 1, int(z_values[j][i]))  # Note: ROOT histograms are filled in a column-major order

    if debug: print_colored("Created TH2F histogram: %s"%name,"INFO")
    return th2f


class Fitter:
    '''
    Class to fit the solar neutrino histograms for each set of oscillation parameters.

    Args:
        obs (ROOT.TH2F): observed (false-data) data histogram.
        solar (ROOT.TH2F): solar data histogram.
        bkg (ROOT.TH2F): bkg data histogram.
        DayNight (bool): True if Night asymmetry is included, False only include Day data (upturn) (default: True).
        SigmaPred (float): uncertainty on the predicted neutrino flux (default: 0.04).
        SigmaBkg (float): uncertainty on the background flux (default: 0.02).

    Returns:
        chisq (float): chi-squared value.
        A_pred (float): best-fit value of the predicted neutrino amplitude.
        A_bkg (float): best-fit value of the background amplitude.
    '''
    def __init__(self, obs, pred, bkg, DayNight=True, SigmaPred=0.04, SigmaBkg=0.02):
        self.fObs = obs
        self.fPred = pred
        self.fBkg = bkg
        self.fDayNight = DayNight
        self.fSigmaPred = SigmaPred
        self.fSigmaBkg = SigmaBkg

    def ROOTOperator(self, A_pred, A_bkg):
        chisq = 0
        for i in range(1, self.fObs.GetNbinsX() + 1):
            for j in range(1, self.fObs.GetNbinsY() + 1):
                if self.fDayNight == False and j < (self.fObs.GetNbinsY() + 1)/2:
                    e = 0
                    o = 0
                else:
                    N_bkg = (1 + A_bkg) * self.fBkg.GetBinContent(i, j)
                    N_pred = (1 + A_pred) * self.fPred.GetBinContent(i, j)
                
                    e = N_bkg + N_pred
                    o = self.fObs.GetBinContent(i, j)
                
                if e == 0:
                    continue
                
                if o == 0:
                    chisq += abs(2 * (e - o))
                
                else:
                    chisq += abs(2 * (e - o + o * np.log(o / e)))

        # Add penalty terms for the fit
        chisq += ((A_pred) / self.fSigmaPred)**2 + ((A_bkg) / self.fSigmaBkg)**2

        return chisq

    def Fit(self, initial_A_pred, initial_A_bkg, verbose=0, debug=False):
        
        if type(self.fObs) == ROOT.TH2F:
            m = Minuit(self.ROOTOperator, A_pred=initial_A_pred, A_bkg=initial_A_bkg)

        else:
            print_colored("ERROR: Unknown input type","ERROR")
            return -1, -1, -1
        
        m.limits['A_pred'] = (initial_A_pred - 10, initial_A_pred + 10)
        m.limits['A_bkg'] = (initial_A_bkg - 10, initial_A_bkg + 10)

        m.migrad()

        A_pred = m.values['A_pred']
        A_bkg = m.values['A_bkg']

        return m.fval, A_pred, A_bkg
    

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