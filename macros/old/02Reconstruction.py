import sys; sys.path.insert(0, '../'); from lib.__init__ import *
np.seterr(divide='ignore', invalid='ignore')
plt.rcParams.update({'font.size': 15})

# Load macro configuration
default_dict = {}
user_input = initialize_macro("02Reconstruction",["config_file","root_file","rewrite","debug"],default_dict=default_dict, debug=True)
user_input = check_macro_config(user_input,debug=user_input["debug"])

# Format input file names and load analysis data
config = user_input["config_file"].split("/")[-1].split("_config")[0]      
configs = {config:config+"_config"}
names = {config:user_input["root_file"]}

truth_labels, reco_labels = get_workflow_branches(workflow="ANALYSIS",debug=user_input["debug"])
run = load_multi(names,configs,load_all=False,preset="",branches={"Truth":truth_labels,"Reco":reco_labels},debug=user_input["debug"])

### DATA SELECTION ###
analysis_info = read_input_file("analysis",INTEGERS=["RECO_ENERGY_RANGE","RECO_ENERGY_BINS","NADIR_RANGE","NADIR_BINS"],debug=False)
energy_edges = np.linspace(analysis_info["RECO_ENERGY_RANGE"][0],analysis_info["RECO_ENERGY_RANGE"][1],analysis_info["RECO_ENERGY_BINS"]+1)
energy_centers = (energy_edges[1:]+energy_edges[:-1])/2
bin_width = energy_edges[1]-energy_edges[0]

# eff_flux = get_detected_solar_spectrum(bins=energy_centers,components=["b8","hep"])
# eff_flux_b8 = get_detected_solar_spectrum(bins=energy_centers,components=["b8"])
# eff_flux_hep = get_detected_solar_spectrum(bins=energy_centers,components=["hep"])

run = compute_reco_workflow(run,configs,workflow="ANALYSIS",rm_branches=False,debug=user_input["debug"])

for jdx,config in enumerate(configs):
    print("Processing %s"%config)
    info = json.load(open('../config/'+config+'/'+configs[config]+'.json', 'r'))

    max_energy = 30; acc = 50
    total_energy_filter = run["Reco"]["TNuE"] < max_energy*1e-3
    # electron_filter     = run["Reco"]["MarleyFrac"][:,0] > 0.9
    geo_filter          = np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"][0]
    version_filter      = np.asarray(run["Reco"]["Version"]) == info["VERSION"][0]
    time_filter         = abs(run["Reco"]["Time"]) < info["EVENT_TICKS"][0]
    neutron_filter      = (run["Reco"]["TMarleyPDG"][:,:] != 2112).all(axis=1)
    
    filter1 = (total_energy_filter)*(geo_filter)*(version_filter)*(neutron_filter)*(time_filter)*(run["Reco"]["Primary"])

    calibration_info = read_input_file(config+"_charge_correction",path="../config/"+config+"/"+config+"_calib/",DOUBLES=["CHARGE_AMP","ELECTRON_TAU"],debug=False)
    corr_popt = [calibration_info["CHARGE_AMP"][0],calibration_info["ELECTRON_TAU"][0]]

    run["Reco"]["ElectronE"] = 1e3*run["Reco"]["TMarleyE"][:,2]
    run["Reco"]["GammaE"] = 1e3*np.sum(np.sum(run["Reco"]["TMarleyP"]*[run["Reco"]["TMarleyPDG"][:,:] == 22],axis=0),axis=1)
    run["Reco"]["NeutronP"] = 1e3*np.sum(np.sum(run["Reco"]["TMarleyP"]*[run["Reco"]["TMarleyPDG"][:,:] == 2112],axis=0),axis=1)
    run["Reco"]["VisEnergy"] = run["Reco"]["ElectronE"] + run["Reco"]["GammaE"]

    fig = make_subplots(rows=2, cols=3,subplot_titles=("Electron","Gamma","Electron+Gamma"))
    fig, true_popt, true_perr = get_hist2d_fit(run["Reco"]["ElectronE"][filter1],1e3*run["Reco"]["TNuE"][filter1],acc,fig,1,1,trimm=15,spec_type="bottom",func_type="linear",debug=user_input["debug"])
    
    # Save the true energy fit parameters to a txt file
    if not os.path.exists("../config/"+config+"/"+config+"_calib/"): os.makedirs("../config/"+config+"/"+config+"_calib/")
    with open("../config/"+config+"/"+config+"_calib/"+config+"_energy_calibration.txt",'w') as f:
        f.write("ENERGY_AMP: %f\n"%true_popt[0])
        f.write("INTERSECTION: %f\n"%true_popt[1])
    plt.close()

    # run = compute_cluster_energy(run,configs,params={"DEFAULT_ENERGY_TIME":"DriftTime","DEFAULT_ADJCL_ENERGY_TIME":"AdjClDriftTime"},rm_branches=False,debug=False)
    run = compute_reco_energy(run,configs,params={},rm_branches=False,debug=user_input["debug"])

    reco_df  = npy2df(run,"Reco",debug=False)
    this_df = reco_df[(reco_df["NeutronP"] == 0) & (reco_df["Primary"] == True) & (reco_df["TNuE"] < max_energy*1e-3)]
    fig.add_trace(go.Histogram2d(nbinsx=acc,nbinsy=acc,y=1e3*this_df["TNuE"],x=this_df["Energy"],coloraxis="coloraxis"),row=2,col=1)
    fig.add_trace(go.Histogram2d(nbinsx=acc,nbinsy=acc,y=1e3*this_df["TNuE"],x=this_df["GammaE"],coloraxis="coloraxis"),row=1,col=2)
    fig.add_trace(go.Histogram2d(nbinsx=acc,nbinsy=acc,y=1e3*this_df["TNuE"],x=this_df["TotalAdjClEnergy"],coloraxis="coloraxis"),row=2,col=2)
    fig.add_trace(go.Histogram2d(nbinsx=acc,nbinsy=acc,y=1e3*this_df["TNuE"],x=this_df["VisEnergy"],coloraxis="coloraxis"),row=1,col=3)
    fig.add_trace(go.Histogram2d(nbinsx=acc,nbinsy=acc,y=1e3*this_df["TNuE"],x=this_df["TotalEnergy"],coloraxis="coloraxis"),row=2,col=3)
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
    if not os.path.exists("../images/calibration/%s_calibration/"%(config)): os.makedirs("../images/calibration/%s_calibration/"%(config))
    fig.write_image("../images/calibration/%s_calibration/%s_reco_energy.png"%(config,config), width=2400, height=1080)
    print_colored("-> Saved reco energy plot to ../images/calibration/%s_calibration/%s_reco_energy.png"%(config,config),"SUCCESS")