import sys; sys.path.insert(0, '../');
import os, ROOT
import pandas as pd
import numpy as np

from rich               import print as rprint
from rich.progress      import track
from ROOT               import TFile, TTree, TList

from lib.root_functions import th2f_from_dataframe, Fitter
from lib.io_functions   import read_input_file, print_colored
from lib.osc_functions  import get_oscillation_datafiles

path = "../data/OSCILLATION/pkl/eval/"
analysis_info = json.load(open('../import/analysis.json', 'r'))

# (dm2_list, sin13_list, sin12_list) = get_oscillation_datafiles(dm2="DEFAULT",sin13="DEFAULT",sin12="DEFAULT",path=path+"Marley/",ext='pkl',auto=True,debug=True)
(dm2_list, sin13_list, sin12_list) = get_oscillation_datafiles(dm2="DEFAULT",sin13="DEFAULT",sin12="DEFAULT",path=path,ext='pkl',auto=True,debug=True)
react_sin13_df = pd.DataFrame(columns=np.unique(sin13_list),index=np.unique(dm2_list))
react_sin12_df = pd.DataFrame(columns=np.unique(sin12_list),index=np.unique(dm2_list))
solar_sin13_df = pd.DataFrame(columns=np.unique(sin13_list),index=np.unique(dm2_list))
solar_sin12_df = pd.DataFrame(columns=np.unique(sin12_list),index=np.unique(dm2_list))

react_df = pd.DataFrame(columns=["dm2","sin13","sin12","chi2"])
solar_df = pd.DataFrame(columns=["dm2","sin13","sin12","chi2"])

solar_tuple = (analysis_info["SOLAR_DM2"],analysis_info["SIN13"],analysis_info["SIN12"])
react_tuple = (analysis_info["REACT_DM2"],analysis_info["SIN13"],analysis_info["SIN12"])
pred1_df = pd.read_pickle(path+"osc_probability_dm2_%.3e_sin13_%.3e_sin12_%.3e.pkl"%solar_tuple)
pred2_df = pd.read_pickle(path+"osc_probability_dm2_%.3e_sin13_%.3e_sin12_%.3e.pkl"%react_tuple)

# bkg_df = pd.read_pickle(path+"APA/APA_events.pkl")
# Make a copy of a dataframe but fill it with zeros
bkg_df = pred1_df.copy(deep=True)
for col in bkg_df.columns:
    bkg_df[col].values[:] = 0

fake_df_dict = {}
for dm2,sin13,sin12 in zip(dm2_list,sin13_list,sin12_list):
    fake_df_dict[(dm2,sin13,sin12)] = pd.read_pickle(path+"osc_probability_dm2_{:.3e}_sin13_{:.3e}_sin12_{:.3e}.pkl".format(dm2,sin13,sin12))+bkg_df

# # Set initial parameter values for the fit
initial_A_solar = 0.0
initial_A_bkg   = 0.0

# Create the Fitter object
solar_hist = th2f_from_dataframe(pred1_df)
react_hist = th2f_from_dataframe(pred2_df)
bkg_hist   = th2f_from_dataframe(bkg_df)

for i in track(range(len(fake_df_dict.keys())), description="Computing data..."):
    params = list(fake_df_dict.keys())[i]
    obs_df = list(fake_df_dict.values())[i]

    rprint("\n--------------------------------",'\n# Parameter Combination '+str(i)+': ',params)
    obs_hist = th2f_from_dataframe(obs_df)

    # Perform the fit
    fitter = Fitter(obs_hist, solar_hist, bkg_hist, DayNight=True)
    # fitter = Fitter(obs_df, pred1_df, bkg_df, DayNight=True)
    chi2, best_A_solar, best_A_bkg = fitter.Fit(initial_A_solar, initial_A_bkg)
    rprint("Solar Chi2:", chi2)
    if params[2] == analysis_info["SIN12"]: 
        rprint("Saving data to sin13")
        solar_sin13_df.loc[params[0],params[1]] = chi2
    if params[1] == analysis_info["SIN13"]: 
        rprint("Saving data to sin12")
        solar_sin12_df.loc[params[0],params[2]] = chi2
    if params[2] != analysis_info['SIN12'] and params[1] != analysis_info['SIN13']:
        print_colored("ERROR: sin12 and sin13 do not match","ERROR")

    solar_df.loc[i] = [params[0],params[1],params[2],chi2]

    fitter = Fitter(obs_hist, react_hist, bkg_hist, DayNight=True)
    # fitter = Fitter(obs_df, pred2_df, bkg_df, DayNight=True)
    chi2, best_A_solar, best_A_bkg = fitter.Fit(initial_A_solar, initial_A_bkg)
    rprint("Reactor Chi2:", chi2)
    if params[2] == analysis_info["SIN12"]: 
        rprint("Saving data to sin13")
        react_sin13_df.loc[params[0],params[1]] = chi2
    if params[1] == analysis_info["SIN13"]: 
        rprint("Saving data to sin12")
        react_sin12_df.loc[params[0],params[2]] = chi2
    if params[2] != analysis_info['SIN12'] and params[1] != analysis_info['SIN13']:
        print_colored("ERROR: sin12 and sin13 do not match","ERROR")

    react_df.loc[i] = [params[0],params[1],params[2],chi2]

path = "../sensitivity"
if not os.path.exists(path+"/results/"): os.makedirs(path+"/results/")
solar_sin12_df.to_pickle(path+"/results/ideal_solar_sin12_df.pkl")
solar_sin13_df.to_pickle(path+"/results/ideal_solar_sin13_df.pkl")
react_sin12_df.to_pickle(path+"/results/ideal_react_sin12_df.pkl")
react_sin13_df.to_pickle(path+"/results/ideal_react_sin13_df.pkl")
solar_df.to_pickle(path+"/results/ideal_solar_df.pkl")
react_df.to_pickle(path+"/results/ideal_react_df.pkl")