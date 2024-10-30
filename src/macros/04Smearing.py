import sys; sys.path.insert(0, '../'); from lib.__init__ import *
import ROOT, root_numpy
from ROOT import TFile, TTree
np.seterr(divide='ignore', invalid='ignore')
plt.rcParams.update({'font.size': 15})

# Load macro configuration
user_input = initialize_macro("03Smearing",["config_file","root_file","rewrite","show","debug"],default_dict={}, debug=True)
user_input = check_macro_config(user_input,debug=user_input["debug"])

# Format input file names and load analysis data
config = user_input["config_file"].split("/")[-1].split("_config")[0]    
info = json.load(open('../config/'+user_input['config_file']+'.json', 'r'))

# analysis_info = read_input_file("analysis",INTEGERS=["RECO_ENERGY_RANGE","RECO_ENERGY_BINS","NADIR_RANGE","NADIR_BINS"],debug=False)
analysis_info = json.load(open('../import/analysis.json', 'r'))
energy_edges = np.linspace(analysis_info["RECO_ENERGY_RANGE"][0],analysis_info["RECO_ENERGY_RANGE"][1],analysis_info["RECO_ENERGY_BINS"]+1)
energy_centers = (energy_edges[1:]+energy_edges[:-1])/2
bin_width = energy_edges[1]-energy_edges[0]

# Load solar spectrum
eff_flux = get_detected_solar_spectrum(bins=energy_centers,components=["b8","hep"])
eff_flux_b8 = get_detected_solar_spectrum(bins=energy_centers,components=["b8"])
eff_flux_hep = get_detected_solar_spectrum(bins=energy_centers,components=["hep"])

list_hist = []
output_dict = {}
data_filter = {"max_energy": 30, "min_energy": 0, "pre_nhits": 3, "primary": True, "neutron": True}
true, data, filter_idx = compute_root_workflow(user_input, info, data_filter, workflow="SMEARING", debug=user_input["debug"])
print_colored("-> Found %i electron candidates out of %i events!"%(len(filter_idx),data["Event"].size),"SUCCESS")

for idx,energy_bin in enumerate(energy_centers):
    neutrino_energy_filter = (data["TNuE"] > (energy_bin - bin_width/2))*(data["TNuE"] < (energy_bin + bin_width/2))                         # Filtering genereted neutrinos in 1GeV energy bin
    filter = neutrino_energy_filter
    hist, bin_edges = np.histogram(data[info["DEFAULT_ANALYSIS_ENERGY"]][filter],bins=energy_edges)
    hist = hist/np.sum(hist)
    hist = np.nan_to_num(hist,0)
    flux = hist*eff_flux[idx]
    fluxb8 = hist*eff_flux_b8[idx]
    fluxhep = hist*eff_flux_hep[idx]
    
    list_hist.append(
        {"Geometry":info["GEOMETRY"],
            "Version": info["VERSION"],
            "Name": user_input["root_file"][0],
            "Generator": 1,
            "TrueEnergy":energy_bin,
            "Hist":hist,
            "Flux":flux,
            "FluxB8":fluxb8,
            "FluxHEP":fluxhep,
            "SolarEnergy":energy_centers,
        }
    )

    fig = make_subplots(rows=1, cols=1,subplot_titles=("Charge Calibration"))
    fig, popt, perr = get_hist1d_fit(data[info["DEFAULT_ANALYSIS_ENERGY"]][filter],energy_edges,fig,1,1,func_type="gauss",debug=user_input["debug"])
    if len(popt) == 0: popt = [0,0,0]; perr = [0,0,0]
    fig.update_layout(coloraxis=dict(colorscale="Turbo",colorbar=dict(title="Counts")),showlegend=False,
        title="Calibration",
        xaxis1_title="Reco Electron Energy [MeV]",
        yaxis1_title="Norm.",
        )
    
    fig = format_coustom_plotly(fig,figsize=(800,600))
    if not os.path.exists("../images/calibration/%s_calibration/%s_calibration_mono/"%(config,config)): os.makedirs("../images/calibration/%s_calibration/%s_calibration_mono/"%(config,config))
    fig.write_image("../images/calibration/%s_calibration/%s_calibration_mono/%s_calibration_%.2f.png"%(config,config,config,energy_bin), width=2400, height=1080)
    print_colored("-> Saved images to ../images/calibration/%s_calibration/%s_calibration_mono/%s_calibration_%.2f.png"%(config,config,config,energy_bin),"SUCCESS")
    
    # with open("../config/"+config+"/"+config+"_calib/"+config+"_{:.2f}".format(energy_bin)+"MeV_energy_calibration.txt",'w') as f:
    #     f.write("ENERGY: %f\n"%(energy_bin))
    #     f.write("SIGMA: %f\n"%popt[2])
    # plt.close()
    output_dict["{:.2f}".format(energy_bin)] = {"ENERGY":energy_bin,"SIGMA":popt[2],"SIGMA_ERR":perr[2]}

with open("../config/"+config+"/"+config+"_calib/"+config+"_energy_calibration.json",'w') as f: json.dump(output_dict, f)
print_colored("-> Saved reco energy fit parameters to ../config/"+config+"/"+config+"_calib/"+config+"_energy_calibration.json","SUCCESS")
    
df = pd.DataFrame(list_hist)
smearing_df = df.drop(columns=["Generator","Name"])
norm_smearing_df = smearing_df.copy()
data = list(smearing_df["Flux"])
norm_data = list(norm_smearing_df["Hist"])
y_array = smearing_df["TrueEnergy"].to_numpy()
x_array = energy_centers

smearing_df = pd.DataFrame(data,columns=y_array,index=x_array).T
norm_smearing_df = pd.DataFrame(norm_data,columns=y_array,index=x_array).T
smearing_df.to_pickle("../config/"+config+"/"+config+"_calib/"+config+"_smearing.pkl")
norm_smearing_df.to_pickle("../config/"+config+"/"+config+"_calib/"+config+"_norm_smearing.pkl")

fig = px.imshow(smearing_df,
    title="Smearing Matrix %s"%(config),
    origin="lower",
    aspect=None,
    color_continuous_scale="turbo",
    labels=dict(y="Recon Energy [MeV]",x="True Energy [MeV]"))

fig = format_coustom_plotly(fig,figsize=(2400,1080),fontsize=18)
fig.write_image("../images/calibration/%s_calibration/%s_smearing.png"%(config,config), width=2400, height=1080)
print_colored("-> Saved images to ../images/calibration/%s_calibration/%s_smearing.png"%(config,config),"SUCCESS")
if user_input["show"]: fig.show()

df = pd.DataFrame(list_hist)
this_df = explode(df,["Flux","Hist","SolarEnergy"])
this_df["TrueEnergy"] = this_df["TrueEnergy"].astype(float)
print(this_df.groupby("Version")["Flux"].sum())
fig=px.bar(this_df,
    x="SolarEnergy",
    y="Flux",
    log_y=False,
    color="TrueEnergy",
    barmode="stack",
    template="presentation",
    facet_col="Version",
    color_continuous_scale="turbo"
    )

# 'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid', 'turbo', 'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr','ylorrd'
fig = format_coustom_plotly(fig,log=(False,False),tickformat=(".1s",".1s"),matches=(None,None))
fig.update_layout(bargap=0)
fig.update_layout(coloraxis=dict(colorscale="Turbo",colorbar=dict(title="Counts")),showlegend=False,
    title="Calibration",
    xaxis1_title="Reco Electron Energy [MeV]",
    yaxis1_title="Norm.",
    )
if user_input["rewrite"]:
    fig.write_image("../images/calibration/%s_calibration/%s_smearing_hist.png"%(config,config), width=2400, height=1080)
    print_colored("-> Saved images to ../images/calibration/%s_calibration/%s_smearing_hist.png"%(config,config),"SUCCESS")
if user_input["show"]: fig.show()