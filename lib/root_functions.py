import ROOT
import numpy as np
import pandas as pd
from iminuit import Minuit

from .io_functions import print_colored

def create_th2f_from_dataframe(df, name="myhist", title="My Histogram", debug=False):
    '''
    Create a TH2F histogram from a pandas DataFrame
    VARIABLES:
        \n - df: pandas DataFrame
        \n - name: name of the histogram (default: myhist)
        \n - title: title of the histogram (default: My Histogram)
    '''
    # Extract x and y centers from column names and index names
    x_centers = df.columns.to_numpy(dtype=float)
    y_centers = df.index.to_numpy(dtype=float)

    # Convert DataFrame data to a 2D numpy array for z-axis values
    z_values = df.to_numpy(dtype=float)

    # Create a TH2F histogram
    nbins_x = len(x_centers)
    nbins_y = len(y_centers)
    th2f = ROOT.TH2F(name, title, nbins_x, x_centers[0], x_centers[-1], nbins_y, y_centers[0], y_centers[-1])

    # Fill the TH2F histogram with z-axis values
    for i in range(nbins_x):
        for j in range(nbins_y):
            th2f.SetBinContent(i + 1, j + 1, z_values[j][i])  # Note: ROOT histograms are filled in a column-major order

    if debug: print_colored("Created TH2F histogram: %s"%name,"INFO")
    return th2f

def create_synthetic_histograms():
    nbins_x = 10
    nbins_y = 10

    obs_values = [i + j for i in range(nbins_x) for j in range(nbins_y)]
    solar_values = [2 * i for i in obs_values]
    neut_values = [3 * i for i in obs_values]

    obs_hist = ROOT.TH2F("obs", "Observed Data", nbins_x, 0, nbins_x, nbins_y, 0, nbins_y)
    solar_hist = ROOT.TH2F("solar", "Solar Data", nbins_x, 0, nbins_x, nbins_y, 0, nbins_y)
    neut_hist = ROOT.TH2F("neut", "Neutrino Data", nbins_x, 0, nbins_x, nbins_y, 0, nbins_y)

    for i in range(1, nbins_x + 1):
        for j in range(1, nbins_y + 1):
            obs_hist.SetBinContent(i, j, obs_values[(i - 1) * nbins_y + (j - 1)])
            solar_hist.SetBinContent(i, j, solar_values[(i - 1) * nbins_y + (j - 1)])
            neut_hist.SetBinContent(i, j, neut_values[(i - 1) * nbins_y + (j - 1)])

    return obs_hist, solar_hist, neut_hist

class Fitter:
    def __init__(self, obs, solar, neut, DayNight=True, SigmaSolar=0.04, SigmaNeut=0.02):
        self.fObs = obs
        self.fSolar = solar
        self.fNeut = neut
        self.fDayNight = DayNight
        self.fSigmaSolar = SigmaSolar
        self.fSigmaNeut = SigmaNeut

    def ROOTOperator(self, A_solar, A_neut):
        chisq = 0
        for i in range(1, self.fObs.GetNbinsX() + 1):
            for j in range(1, self.fObs.GetNbinsY() + 1):
                if self.fDayNight == False and j < (self.fObs.GetNbinsY() + 1)/2:
                    e = 0
                    o = 0
                else:
                    N_neut = (1 + A_neut) * self.fNeut.GetBinContent(i, j)
                    N_solar = (1 + A_solar) * self.fSolar.GetBinContent(i, j)
                
                    e = N_neut + N_solar
                    o = self.fObs.GetBinContent(i, j)
                
                if o == 0:
                    continue
                
                if o == 0:
                    chisq += 2 * (e - o)
                else:
                    chisq += 2 * (e - o + o * np.log(o / e))

        chisq += ((A_solar) / self.fSigmaSolar)**2 + ((A_neut) / self.fSigmaNeut)**2
        return chisq
    
    def PyOperator(self, A_solar, A_neut):
        # Repeat the same operation as ROOTOperator but using pandas DataFrames as input
        chisq = 0
        for i in self.fObs.columns:
            for j in self.fObs.index:
                if self.fDayNight == False and j < (self.fObs.shape[0] + 1)/2:
                    e = 0
                    o = 0
                else:
                    N_neut = (1 + A_neut) * self.fNeut.loc[j,i]
                    N_solar = (1 + A_solar) * self.fSolar.loc[j,i]
                
                    e = N_neut + N_solar
                    o = self.fObs.loc[j,i]
                
                if o == 0:
                    continue
                
                if o == 0:
                    chisq += 2 * (e - o)
                else:
                    chisq += 2 * (e - o + o * np.log(o / e))

    def Fit(self, initial_A_solar, initial_A_neut, verbose=0, debug=False):
        if type(self.fObs) == ROOT.TH2F:
            m = Minuit(self.ROOTOperator, A_solar=initial_A_solar, A_neut=initial_A_neut)
        elif type(self.fObs) == pd.DataFrame:
            m = Minuit(self.PyOperator, A_solar=initial_A_solar, A_neut=initial_A_neut)
        else:
            print_colored("ERROR: Unknown input type","ERROR")
            return
        
        m.limits['A_solar'] = (initial_A_solar - 10, initial_A_solar + 10)
        m.limits['A_neut'] = (initial_A_neut - 10, initial_A_neut + 10)

        m.migrad()

        A_solar = m.values['A_solar']
        A_neut = m.values['A_neut']

        return m.fval, A_solar, A_neut