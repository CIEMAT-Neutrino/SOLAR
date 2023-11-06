# ln -s /pc/choozdsk01/palomare/DUNE/SOLAR/data/ .

# Load function libraries and set up the environment
import sys; sys.path.insert(0, '../'); from lib.__init__ import *
np.seterr(divide='ignore', invalid='ignore')
plt.rcParams.update({'font.size': 15})

# Load macro configuration
default_dict = {}
user_input = initialize_macro("03Smearing",["config_file","root_file","rewrite","debug"],default_dict=default_dict, debug=True)
user_input = check_macro_config(user_input,debug=user_input["debug"])

# Format input file names and load analysis data
config = user_input["config_file"].split("/")[-1].split("_config")[0]    
config_files = {config:config+"_config"}
names = {config:user_input["root_file"]}

truth_labels, reco_labels = get_workflow_branches(workflow="ANALYSIS",debug=user_input["debug"])
run = load_multi(names,config_files,load_all=False,preset="",branches={"Truth":truth_labels,"Reco":reco_labels},debug=user_input["debug"])

# Load analysis configuration
analysis_info = read_input_file("analysis",INTEGERS=["RECO_ENERGY_RANGE","RECO_ENERGY_BINS","NADIR_RANGE","NADIR_BINS"],debug=False)
energy_edges = np.linspace(analysis_info["RECO_ENERGY_RANGE"][0],analysis_info["RECO_ENERGY_RANGE"][1],analysis_info["RECO_ENERGY_BINS"][0]+1)
energy_centers = (energy_edges[1:]+energy_edges[:-1])/2
bin_width = energy_edges[1]-energy_edges[0]

eff_flux = get_detected_solar_spectrum(bins=energy_centers,components=["b8","hep"])
eff_flux_b8 = get_detected_solar_spectrum(bins=energy_centers,components=["b8"])
eff_flux_hep = get_detected_solar_spectrum(bins=energy_centers,components=["hep"])

# Compute the calibration workflow
run = compute_reco_workflow(run,config_files,workflow="ANALYSIS",debug=user_input["debug"])
run = compute_reco_energy(run,config_files,debug=user_input["debug"])

# Filter the data for calibration
max_energy = 30; acc = 50
info = read_input_file(config+'/'+config_files[config],debug=False)
total_energy_filter = run["Reco"]["TNuE"] < max_energy*1e-3
# electron_filter     = run["Reco"]["MarleyFrac"][:,0] == 1
geo_filter          = np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"][0]
version_filter      = np.asarray(run["Reco"]["Version"]) == info["VERSION"][0]
time_filter         = abs(run["Reco"]["Time"]) < info["EVENT_TICKS"][0]
neutron_filter      = (run["Reco"]["TMarleyPDG"][:,:] != 2112).all(axis=1)

filter1 = (total_energy_filter)*(geo_filter)*(version_filter)*(neutron_filter)*(time_filter)*(run["Reco"]["Primary"])
list_hist = []

for idx,energy_bin in enumerate(energy_centers):
    neutrino_energy_filter = (run["Reco"]["TNuE"] > 1e-3*(energy_bin - bin_width/2))*(run["Reco"]["TNuE"] < 1e-3*(energy_bin + bin_width/2))                         # Filtering genereted neutrinos in 1GeV energy bin
    filter2 = (filter1)*(neutrino_energy_filter)
    hist, bin_edges = np.histogram(run["Reco"][info["DEFAULT_ANALYSIS_ENERGY"][0]][filter2],bins=energy_edges)
    hist = hist/np.sum(hist)
    hist = np.nan_to_num(hist,0)
    flux = hist*eff_flux[idx]
    fluxb8 = hist*eff_flux_b8[idx]
    fluxhep = hist*eff_flux_hep[idx]
    
    list_hist.append(
        {"Geometry":info["GEOMETRY"][0],
            "Version": info["VERSION"][0],
            "Name": run["Reco"]["Name"][0],
            "Generator": 1,
            "TrueEnergy":energy_bin,
            "Hist":hist,
            "Flux":flux,
            "FluxB8":fluxb8,
            "FluxHEP":fluxhep,
            "RecoEnergy":energy_centers,
        }
    )

    fig = make_subplots(rows=1, cols=1,subplot_titles=("Charge Calibration"))
    fig, popt, perr = get_hist1d_fit(run["Reco"][info["DEFAULT_ANALYSIS_ENERGY"][0]][filter2],energy_edges,fig,1,1,func_type="gauss",debug=user_input["debug"])
    if len(popt) == 0: popt = [0,0,0]; perr = [0,0,0]
    # Save figures to "../config/"+config+"/"+config+"_calib/" and delete the figures
    fig.update_layout(coloraxis=dict(colorscale="Turbo",colorbar=dict(title="Counts")),showlegend=False,
        title="Calibration",
        xaxis1_title="Reco Electron Energy [MeV]",
        yaxis1_title="Norm.",
        )
    fig = format_coustom_plotly(fig,figsize=(800,600))
    # Check if the directory exists
    if not os.path.exists("../images/calibration/%s_calibration/%s_calibration_mono/"%(config,config)): os.makedirs("../images/calibration/%s_calibration/%s_calibration_mono/"%(config,config))
    fig.write_image("../images/calibration/%s_calibration/%s_calibration_mono/%s_calibration_%.2f.png"%(config,config,config,energy_bin), width=2400, height=1080)
    print_colored("-> Saved images to ../images/calibration/%s_calibration/%s_calibration_mono/%s_calibration_%.2f.png"%(config,config,config,energy_bin),"SUCCESS")
    
    with open("../config/"+config+"/"+config+"_calib/"+config+"_{:.2f}".format(energy_bin)+"MeV_energy_calibration.txt",'w') as f:
        # f.write("CHARGE_AMP: %f\n"%popt1[0])
        # f.write("TAU: %f\n"%popt1[1])
        f.write("ENERGY: %f\n"%(energy_bin))
        f.write("SIGMA: %f\n"%popt[2])
        # f.write("CORRECTED_CHARGE: %f\n"%np.mean(run["Reco"]["TotalCharge"][filter2]))
    plt.close()
    
df = pd.DataFrame(list_hist)
df = df[(df["Geometry"] == info["GEOMETRY"][0])&(df["Version"] == info["VERSION"][0])]

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

fig = format_coustom_plotly(fig,figsize=(800,600),fontsize=18)
fig.write_image("../images/calibration/%s_calibration/%s_smearing.png"%(config,config), width=2400, height=1080)
print_colored("-> Saved images to ../images/calibration/%s_calibration/%s_smearing.png"%(config,config),"SUCCESS")
# fig.show()

df = pd.DataFrame(list_hist)
this_df = explode(df,["Flux","Hist","RecoEnergy"])
this_df["TrueEnergy"] = this_df["TrueEnergy"].astype(float)
print(this_df.groupby("Version")["Flux"].sum())
fig=px.bar(this_df,
    x="RecoEnergy",
    y="Flux",
    log_y=False,
    color="TrueEnergy",
    barmode="stack",
    template="presentation",
    facet_col="Version",
    color_continuous_scale="turbo"
    )

# 'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid', 'turbo', 'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr','ylorrd'
fig = format_coustom_plotly(fig,log=(False,False),tickformat=(".1s",".1s"))
fig.update_layout(bargap=0)
fig.update_layout(coloraxis=dict(colorscale="Turbo",colorbar=dict(title="Counts")),showlegend=False,
    title="Calibration",
    xaxis1_title="Reco Electron Energy [MeV]",
    yaxis1_title="Norm.",
    )
fig.write_image("../images/calibration/%s_calibration/%s_smearing_hist.png"%(config,config), width=2400, height=1080)
print_colored("-> Saved images to ../images/calibration/%s_calibration/%s_smearing_hist.png"%(config,config),"SUCCESS")
# fig.show()