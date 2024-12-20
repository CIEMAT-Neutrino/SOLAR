import sys, json 
sys.path.insert(0, '../'); from lib.__init__ import *
np.seterr(divide='ignore', invalid='ignore')
plt.rcParams.update({'font.size': 15})

# Load macro configuration
user_input = initialize_macro("02Reconstruction",["config_file","root_file","rewrite","show","debug"],default_dict={}, debug=True)
user_input = check_macro_config(user_input,debug=user_input["debug"])

# Load analysis data and configuration
config = user_input["config_file"].split("/")[-1].split("_config")[0]      
info = json.load(open('../config/'+user_input['config_file']+'.json', 'r'))

acc = 50
data_filter = {"max_energy": 20, "min_energy": 0, "pre_nhits": 3, "primary": True, "neutron": True}
true, data, filter_idx = compute_root_workflow(user_input, info, data_filter, workflow="RECONSTRUCTION", debug=user_input["debug"])
print_colored("-> Found %i electron candidates out of %i events!"%(len(filter_idx),data["Event"].size),"SUCCESS")

density = ""
fig = make_subplots(rows=2, cols=3,subplot_titles=("Electron","Gamma","Electron+Gamma"))
fig, top_bottom_popt, top_bottom_perr = get_hist2d_fit(data["ElectronE"][filter_idx],data["SignalParticleE"][filter_idx],acc,fig,1,1,trimm=(15,15),threshold=0.3,spec_type="top+bottom",func_type="linear",density=density,debug=user_input["debug"])
top_popt = top_bottom_popt[:2]; bottom_popt = top_bottom_popt[2:]
top_perr = top_bottom_perr[:2]; bottom_perr = top_bottom_perr[2:]
top_func = lambda x: top_popt[0]*x + top_popt[1]
bottom_func = lambda x: bottom_popt[0]*x + bottom_popt[1]

# Compute reco energy
idx = data["TotalAdjClEnergy"] > 1.5
data["SolarEnergy"][idx] = top_func(data["Energy"][idx])
data["SolarEnergy"][~idx] = bottom_func(data["Energy"][~idx])

reco_df  = npy2df(data, "", debug=user_input["debug"])
this_df = reco_df[(reco_df["NeutronP"] == 0) & (reco_df["Primary"] == data_filter["primary"]) & (reco_df["SignalParticleE"] < data_filter["max_energy"])]
fig.add_trace(go.Histogram2d(histnorm=density,nbinsx=acc,nbinsy=acc,y=this_df["SignalParticleE"],x=this_df["GammaE"],coloraxis="coloraxis"),row=1,col=2)
fig.add_trace(go.Histogram2d(histnorm=density,nbinsx=acc,nbinsy=acc,y=this_df["SignalParticleE"],x=this_df["VisEnergy"],coloraxis="coloraxis"),row=1,col=3)
fig.add_trace(go.Histogram2d(histnorm=density,nbinsx=acc,nbinsy=acc,y=this_df["SignalParticleE"],x=this_df["Energy"],coloraxis="coloraxis"),row=2,col=1)
fig.add_trace(go.Histogram2d(histnorm=density,nbinsx=acc,nbinsy=acc,y=this_df["SignalParticleE"],x=this_df["TotalAdjClEnergy"],coloraxis="coloraxis"),row=2,col=2)
fig.add_trace(go.Histogram2d(histnorm=density,nbinsx=acc,nbinsy=acc,y=this_df["SignalParticleE"],x=this_df["TotalEnergy"],coloraxis="coloraxis"),row=2,col=3)
fig.update_layout(title="True Neutrino Energy vs. Reco Visible Energy",width=2400, height=1080,template="presentation",coloraxis=dict(colorscale="Turbo"),
    xaxis1_title="True Electron Energy [MeV]",
    xaxis2_title="True Gamma Energy [MeV]",
    xaxis3_title="True Electron+Gamma Energy [MeV]",
    xaxis4_title="Reco Electron Energy [MeV]",
    xaxis5_title="Reco Gamma Energy [MeV]",
    xaxis6_title="Reco Electron+Gamma Energy [MeV]",
    )

fig.update_yaxes(title_text="True Neutrino Energy [MeV]")
fig.update_layout(showlegend=False)

if not os.path.exists("../images/calibration/%s_calibration/"%(config)): os.makedirs("../images/calibration/%s_calibration/"%(config))
fig.write_image("../images/calibration/%s_calibration/%s_reco_energy.png"%(config,config), width=2400, height=1080)
print_colored("-> Saved reco energy plot to ../images/calibration/%s_calibration/%s_reco_energy.png"%(config,config),"SUCCESS")

# Save as json file
if not os.path.exists("../config/"+config+"/"+config+"_calib/"): os.makedirs("../config/"+config+"/"+config+"_calib/")
with open("../config/"+config+"/"+config+"_calib/"+config+"_energy_calibration.json",'w') as f:
    json.dump({"ENERGY_AMP": bottom_popt[0], "INTERSECTION": bottom_popt[1]}, f)

print_colored("-> Saved reco energy fit parameters to ../config/"+config+"/"+config+"_calib/"+config+"_energy_calibration.txt","SUCCESS")
if user_input["show"]: fig.show()