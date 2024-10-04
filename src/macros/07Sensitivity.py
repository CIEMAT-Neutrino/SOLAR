import sys, json
sys.path.insert(0, "../../")
from lib import *

from rich               import print as rprint
from rich.progress      import track

from lib.root_functions import th2f_from_dataframe, Fitter
from lib.osc_functions  import get_oscillation_datafiles

path = f"{root}/data/SENSITIVITY/hd_1x2x6/"
analysis_info = json.load(open(f'{root}/lib/import/analysis.json', 'r'))

(dm2_list, sin13_list, sin12_list) = get_oscillation_datafiles(dm2=None,sin13=[0.021],sin12=None,path=path+"wbkg/",ext='pkl',auto=False,debug=True)
react_sin13_df = pd.DataFrame(columns=np.unique(sin13_list),index=np.unique(dm2_list))
react_sin12_df = pd.DataFrame(columns=np.unique(sin12_list),index=np.unique(dm2_list))
solar_sin13_df = pd.DataFrame(columns=np.unique(sin13_list),index=np.unique(dm2_list))
solar_sin12_df = pd.DataFrame(columns=np.unique(sin12_list),index=np.unique(dm2_list))

react_df = pd.DataFrame(columns=["dm2","sin13","sin12","chi2"])
solar_df = pd.DataFrame(columns=["dm2","sin13","sin12","chi2"])

solar_tuple = (analysis_info["SOLAR_DM2"],analysis_info["SIN13"],analysis_info["SIN12"])
react_tuple = (analysis_info["REACT_DM2"],analysis_info["SIN13"],analysis_info["SIN12"])
pred1_df = pd.read_pickle(path+"wbkg/solar_events_dm2_%.3e_sin13_%.3e_sin12_%.3e.pkl"%solar_tuple)
pred2_df = pd.read_pickle(path+"wbkg/solar_events_dm2_%.3e_sin13_%.3e_sin12_%.3e.pkl"%react_tuple)

name = "extrabkg"
bkg_df = pd.read_pickle(path+"NeutronsInCavernwall/NeutronsInCavernwall_events.pkl")
# Make a copy of a dataframe but fill it with zeros
extra_bkg_df = bkg_df.copy()
extra_bkg_df = extra_bkg_df/extra_bkg_df.max()
extra_bkg_df = extra_bkg_df.mul(1e9)
extra_bkg_df.loc[:,10:] = 0
# Make all nan values zero
extra_bkg_df = extra_bkg_df.fillna(1e9)
bkg_df = bkg_df + extra_bkg_df
print(bkg_df)
# Load the fake dataframes and add the background estimate
fake_df_dict = {}
for dm2,sin13,sin12 in zip(dm2_list,sin13_list,sin12_list):
    fake_df_dict[(dm2,sin13,sin12)] = pd.read_pickle(path+"wbkg/solar_events_dm2_{:.3e}_sin13_{:.3e}_sin12_{:.3e}.pkl".format(dm2,sin13,sin12))+bkg_df

# Set initial parameter values for the fit
initial_A_pred = 0.0
initial_A_bkg  = 0.0

# Create the Fitter object
pred1_hist = th2f_from_dataframe(pred1_df)
pred2_hist = th2f_from_dataframe(pred2_df)
bkg_hist   = th2f_from_dataframe(bkg_df)

for i in track(range(len(fake_df_dict.keys())), description="Computing data..."):
    params = list(fake_df_dict.keys())[i]
    obs_df = list(fake_df_dict.values())[i]

    rprint("\n--------------------------------",'\n# Parameter Combination '+str(i)+': ',params)
    obs_hist = th2f_from_dataframe(obs_df)

    # Perform the fit
    fitter = Fitter(obs_hist, pred1_hist, bkg_hist, DayNight=True, SigmaPred=0.04, SigmaBkg=0.1)
    chi2, best_A_pred, best_A_bkg = fitter.Fit(initial_A_pred, initial_A_bkg)
    rprint("Solar Chi2:", chi2)
    if params[2] == analysis_info["SIN12"]: 
        rprint("Saving data to sin13")
        solar_sin13_df.loc[params[0],params[1]] = chi2
    if params[1] == analysis_info["SIN13"]: 
        rprint("Saving data to sin12")
        solar_sin12_df.loc[params[0],params[2]] = chi2
    if params[2] != analysis_info['SIN12'] and params[1] != analysis_info['SIN13']:
        rprint(f"[red]ERROR: sin12 and sin13 do not match[/red]")

    solar_df.loc[i] = [params[0],params[1],params[2],chi2]

    fitter = Fitter(obs_hist, pred2_hist, bkg_hist, DayNight=True, SigmaPred=0.04, SigmaBkg=0.1)
    chi2, best_A_pred, best_A_bkg = fitter.Fit(initial_A_pred, initial_A_bkg)
    rprint("Reactor Chi2:", chi2)
    if params[2] == analysis_info["SIN12"]: 
        rprint("Saving data to sin13")
        react_sin13_df.loc[params[0],params[1]] = chi2
    if params[1] == analysis_info["SIN13"]: 
        rprint("Saving data to sin12")
        react_sin12_df.loc[params[0],params[2]] = chi2
    if params[2] != analysis_info['SIN12'] and params[1] != analysis_info['SIN13']:
        rprint(f"[red]ERROR: sin12 and sin13 do not match[/red]")

    react_df.loc[i] = [params[0],params[1],params[2],chi2]

# Delete files if they already exist
if os.path.exists(f"{path}{name}_solar_sin12_df.pkl"): os.remove(f"{path}{name}_solar_sin12_df.pkl")
if os.path.exists(f"{path}{name}_solar_sin13_df.pkl"): os.remove(f"{path}{name}_solar_sin13_df.pkl")
if os.path.exists(f"{path}{name}_react_sin12_df.pkl"): os.remove(f"{path}{name}_react_sin12_df.pkl")
if os.path.exists(f"{path}{name}_react_sin13_df.pkl"): os.remove(f"{path}{name}_react_sin13_df.pkl")
if os.path.exists(f"{path}{name}_solar_df.pkl"):       os.remove(f"{path}{name}_solar_df.pkl")
if os.path.exists(f"{path}{name}_react_df.pkl"):       os.remove(f"{path}{name}_react_df.pkl")

# Save the dataframes to pickle files
solar_sin12_df.to_pickle(f"{path}{name}_solar_sin12_df.pkl")
solar_sin13_df.to_pickle(f"{path}{name}_solar_sin13_df.pkl")
react_sin12_df.to_pickle(f"{path}{name}_react_sin12_df.pkl")
react_sin13_df.to_pickle(f"{path}{name}_react_sin13_df.pkl")
solar_df.to_pickle(      f"{path}{name}_solar_df.pkl")
react_df.to_pickle(      f"{path}{name}_react_df.pkl")