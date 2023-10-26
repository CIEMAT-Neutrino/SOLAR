import sys; sys.path.insert(0, '../'); from lib.__init__ import *
np.seterr(divide='ignore', invalid='ignore')
plt.rcParams.update({'font.size': 15})

# Load macro configuration
user_input = initialize_macro("02Reconstruction",["config_file","root_file","rewrite","debug"],default_dict={}, debug=True)
user_input = check_macro_config(user_input,debug=user_input["debug"])

# Load analysis data and configuration
config = user_input["config_file"].split("/")[-1].split("_config")[0]      
info = read_input_file(user_input["config_file"],path="../config/",debug=user_input["debug"])
params = get_param_dict(info,in_params={},debug=user_input["debug"])
calibration_info = read_input_file(config+"_corr_config",path="../config/"+config+"/"+config+"_calib/",DOUBLES=["CHARGE_AMP","ELECTRON_TAU"],debug=False)
corr_popt = [calibration_info["CHARGE_AMP"][0],calibration_info["ELECTRON_TAU"][0]]

# Load root file
if user_input["debug"]: print_colored("\nLoading data...","DEBUG")
input_file = TFile(info["PATH"][0]+info["NAME"][0]+user_input["root_file"][0]+".root")
folder_name = input_file.GetListOfKeys()[0].GetName()
tree = input_file.Get(folder_name + "/" + "SolarNuAnaTree")
print_colored("-> Found tree: %s"%tree.GetName(), "SUCCESS")

# Start the analysis
max_energy = 30; acc = 50
filter_idx = np.array([],dtype=int)
data = {"Primary":          np.zeros(tree.GetEntries(), dtype=bool),
        "Correction":       np.ones(tree.GetEntries(),  dtype=float),
        "Energy":           np.zeros(tree.GetEntries(), dtype=float),
        "TNuE":             np.zeros(tree.GetEntries(), dtype=float),
        "ElectronE":        np.zeros(tree.GetEntries(), dtype=float),
        "GammaE":           np.zeros(tree.GetEntries(), dtype=float),
        "NeutronP":         np.zeros(tree.GetEntries(), dtype=float),
        "VisEnergy":        np.zeros(tree.GetEntries(), dtype=float),
        "TotalEnergy":      np.zeros(tree.GetEntries(), dtype=float),
        "RecoEnergy":       np.zeros(tree.GetEntries(), dtype=float),
        "MaxAdjClEnergy":   np.zeros(tree.GetEntries(), dtype=float),
        "TotalAdjClEnergy": np.zeros(tree.GetEntries(), dtype=float),}

for i in track(range(tree.GetEntries()),description="Computing reco energy..."):
    tree.GetEntry(i)
    # Primary computation
    try: data["Primary"][i] = tree.Charge > max(tree.AdjClCharge)
    except ValueError: data["Primary"][i] = False
    
    # Energy computation
    data["Correction"][i]  = np.exp(np.abs(tree.Time)/corr_popt[1])
    data["Energy"][i] = tree.Charge*data["Correction"][i]/corr_popt[0]
    data["TNuE"][i] = 1e3*tree.TNuE
    data["ElectronE"][i] = 1e3*tree.TMarleyE[2]
    for j in range(len(tree.TMarleyPDG)):
        if tree.TMarleyPDG[j] == 22: data["GammaE"][i]+=1e3*tree.TMarleyE[j]
        if tree.TMarleyPDG[j] == 2112: data["NeutronP"][i]+=1e3*tree.TMarleyP[j] 
    data["VisEnergy"][i] = data["ElectronE"][i] + data["GammaE"][i]

    # Compute event energy
    for z in range(len(tree.AdjClR)):
        try: adj_cl_correction = np.exp(np.abs(data[params["DEFAULT_ADJCL_ENERGY_TIME"]][i])/corr_popt[1])
        except KeyError: adj_cl_correction = np.exp(np.abs(tree.AdjClTime.at(z))/corr_popt[1])
        adj_cl_energy = tree.AdjClCharge.at(z)*adj_cl_correction/corr_popt[0]
        if adj_cl_energy > data["MaxAdjClEnergy"][i]: data["MaxAdjClEnergy"][i] = adj_cl_energy
        data["TotalAdjClEnergy"][i] += adj_cl_energy
    data["TotalEnergy"][i] = data["TotalAdjClEnergy"][i] + data["Energy"][i] + 1.9

    # Filter events
    if 1e3*tree.TNuE > max_energy: continue
    # if tree.Generator != 1: continue
    if data["Primary"][i] == False: continue
    if abs(tree.Time) > info["EVENT_TICKS"][0]: continue
    for j in range(len(tree.TMarleyPDG)):
        if tree.TMarleyPDG[j] == 2112: continue

    filter_idx = np.append(filter_idx,i)

fig = make_subplots(rows=2, cols=3,subplot_titles=("Electron","Gamma","Electron+Gamma"))
fig, true_popt, true_perr = get_hist2d_fit(data["ElectronE"][filter_idx],data["TNuE"][filter_idx],acc,fig,1,1,trimm=15,spec_type="bottom",func_type="linear",debug=user_input["debug"])

# Compute reco energy
idx = data["TotalAdjClEnergy"] > 1.5
data["RecoEnergy"][idx] = data["Energy"][idx]*true_popt[0] + true_popt[1]
data["RecoEnergy"][~idx] = data["Energy"][~idx]*true_popt[0] + true_popt[1] + 2.5

reco_df  = npy2df(data, "", debug=user_input["debug"])
this_df = reco_df[(reco_df["NeutronP"] == 0) & (reco_df["Primary"] == True) & (reco_df["TNuE"] < max_energy)]
fig.add_trace(go.Histogram2d(nbinsx=acc,nbinsy=acc,y=this_df["TNuE"],x=this_df["Energy"],coloraxis="coloraxis"),row=2,col=1)
fig.add_trace(go.Histogram2d(nbinsx=acc,nbinsy=acc,y=this_df["TNuE"],x=this_df["GammaE"],coloraxis="coloraxis"),row=1,col=2)
fig.add_trace(go.Histogram2d(nbinsx=acc,nbinsy=acc,y=this_df["TNuE"],x=this_df["TotalAdjClEnergy"],coloraxis="coloraxis"),row=2,col=2)
fig.add_trace(go.Histogram2d(nbinsx=acc,nbinsy=acc,y=this_df["TNuE"],x=this_df["VisEnergy"],coloraxis="coloraxis"),row=1,col=3)
fig.add_trace(go.Histogram2d(nbinsx=acc,nbinsy=acc,y=this_df["TNuE"],x=this_df["TotalEnergy"],coloraxis="coloraxis"),row=2,col=3)
fig.update_layout(title="True Neutrino Energy vs. Reco Visible Energy",height=800,width=1600,template="presentation",coloraxis=dict(colorscale="Turbo",colorbar=dict(title="Counts")),
    xaxis1_title="True Electron Energy [MeV]",
    xaxis2_title="True Gamma Energy [MeV]",
    xaxis3_title="True Electron+Gamma Energy [MeV]",
    xaxis4_title="Reco Electron Energy [MeV]",
    xaxis5_title="Reco Gamma Energy [MeV]",
    xaxis6_title="Reco Electron+Gamma Energy [MeV]",
    )

fig.update_yaxes(title_text="True Neutrino Energy [MeV]")
fig.update_layout(showlegend=False)

if not os.path.exists("../plots/calibration/%s_calibration/"%(config)): os.makedirs("../plots/calibration/%s_calibration/"%(config))
fig.write_image("../plots/calibration/%s_calibration/%s_reco_energy.png"%(config,config), width=2400, height=1080)
print_colored("-> Saved reco energy plot to ../plots/calibration/%s_calibration/%s_reco_energy.png"%(config,config),"SUCCESS")
fig.show()

# Save the true energy fit parameters to a txt file
if not os.path.exists("../config/"+config+"/"+config+"_calib/"): os.makedirs("../config/"+config+"/"+config+"_calib/")
with open("../config/"+config+"/"+config+"_calib/"+config+"_calib_config.txt",'w') as f:
    f.write("ENERGY_AMP: %f\n"%true_popt[0])
    f.write("INTERSECTION: %f\n"%true_popt[1])
print_colored("-> Saved reco energy fit parameters to ../config/"+config+"/"+config+"_calib/"+config+"_calib_config.txt","SUCCESS")