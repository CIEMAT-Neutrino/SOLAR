# Load function libraries and set up the environment
import sys; sys.path.insert(0, '../'); from lib.__init__ import *
np.seterr(divide='ignore', invalid='ignore')
plt.rcParams.update({'font.size': 15})

# Load macro configuration
user_input = initialize_macro("01Calibration",["config_file","root_file","rewrite","debug"],default_dict={}, debug=True)
user_input = check_macro_config(user_input,debug=user_input["debug"])

# Format input file names and load analysis data
config = user_input["config_file"].split("/")[-1].split("_config")[0]
info = read_input_file(user_input["config_file"],path="../config/",debug=user_input["debug"])

if user_input["debug"]: print_colored("\nLoading data...","DEBUG")
input_file = TFile(info["PATH"][0]+info["NAME"][0]+user_input["root_file"][0]+".root")
folder_name = input_file.GetListOfKeys()[0].GetName()
tree = input_file.Get(folder_name + "/" + "SolarNuAnaTree")
print_colored("-> Found tree: %s"%tree.GetName(), "SUCCESS")

max_energy = 20; acc = 50
calib_time = np.array([],dtype=float)
calib_charge = np.array([],dtype=float)
calib_energy = np.array([],dtype=float)
calib_idx = np.array([],dtype=int)
data = {"Primary":    np.zeros(tree.GetEntries(), dtype=bool),
        "Correction": np.ones(tree.GetEntries(), dtype=float),
        "Energy":     np.zeros(tree.GetEntries(), dtype=float),}

for i in track(range(tree.GetEntries()),description="Filtering electron data..."):
    tree.GetEntry(i)

    # Primary computation
    try: data["Primary"][i] = tree.Charge > max(tree.AdjClCharge)
    except ValueError: data["Primary"][i] = False

    if 1e3*tree.TNuE > max_energy: continue
    if abs(tree.Time) > info["EVENT_TICKS"][0]: continue
    if data["Primary"][i] == False: continue
    if (np.asarray(tree.TMarleyPDG) == 2112).any(): continue # Very time consuming!

    calib_time = np.append(calib_time,abs(tree.Time))
    calib_charge = np.append(calib_charge,tree.Charge)
    calib_energy = np.append(calib_energy,1e3*tree.TMarleyE[2])
    calib_idx = np.append(calib_idx,i)
print_colored("-> Found %i electron candidates out of %i events!"%(calib_idx.size,i),"SUCCESS")

# Plot the calibration workflow
fig = make_subplots(rows=2, cols=2,subplot_titles=("Time Profile","Correction","Charge Calibration"))
fig, corr_popt, perr = get_hist2d_fit(calib_time,calib_charge/calib_energy,acc,fig,1,1,func_type="exponential",debug=user_input["debug"])

# Compute cluster energy
for i in track(range(tree.GetEntries()),description="Computing cluster energy..."):
    tree.GetEntry(i)

    # Energy computation
    data["Correction"][i]  = np.exp(np.abs(tree.Time)/corr_popt[1])
    data["Energy"][i] = tree.Charge*data["Correction"][i]/corr_popt[0]

fig.add_trace(go.Histogram2d(x=calib_time,
    y=(calib_charge*(data["Correction"][calib_idx])/(corr_popt[0]*calib_energy)),
    coloraxis="coloraxis",nbinsx=acc,nbinsy=acc),row=1,col=2)

fig, reco_popt, perr = get_hist2d_fit(calib_energy,data["Energy"][calib_idx],acc,fig,2,1,func_type="linear",debug=user_input["debug"])
fig, res_popt, perr = get_hist1d_fit(data["Energy"][calib_idx]/calib_energy,2*acc,fig,2,2,func_type="gauss",debug=user_input["debug"])

fig.update_layout(coloraxis=dict(colorscale="Turbo",colorbar=dict(title="Counts")),showlegend=False,
    title="Calibration",
    xaxis1_title="Time [ticks]",
    xaxis2_title="Time [ticks]",
    xaxis3_title="True Electron Energy [MeV]",
    xaxis4_title="Corr. Charge/Energy [Norm]",
    yaxis1_title="Charge/Energy [ADC x ticks/MeV]",
    yaxis2_title="Corr. Charge/Energy [Norm]",
    yaxis3_title="Reco Electron Energy [MeV]",
    yaxis4_title="Norm.",
    )

fig = format_coustom_plotly(fig,fontsize=18)
if not os.path.exists("../plots/calibration/%s_calibration/"%config): os.makedirs("../plots/calibration/%s_calibration/"%config)
fig.write_image("../plots/calibration/%s_calibration/%s_calibration.png"%(config,config), width=2400, height=1080)
print_colored("-> Saved plots to ../plots/calibration/%s_calibration/%s_calibration.png"%(config,config),"SUCCESS")
fig.show()

# Save the true energy fit parameters to a txt file
if not os.path.exists("../config/"+config+"/"+config+"_calib/"): os.makedirs("../config/"+config+"/"+config+"_calib/")
with open("../config/"+config+"/"+config+"_calib/"+config+"_corr_config.txt",'w') as f:
    f.write("CHARGE_AMP: %f\n"%corr_popt[0])
    f.write("ELECTRON_TAU: %f\n"%corr_popt[1])
print_colored("-> Saved calibration parameters to ../config/"+config+"/"+config+"_calib/"+config+"_corr_config.txt","SUCCESS")