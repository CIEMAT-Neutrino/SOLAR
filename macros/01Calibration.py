
import sys
sys.path.insert(0, '../')

import os,ROOT,root_numpy
import numpy                 as np
import plotly.graph_objects  as go
 
from ROOT          import TFile
from rich.progress import track
from lib import initialize_macro, check_macro_config, read_input_file, print_colored, make_subplots, format_coustom_plotly, get_hist2d_fit, get_hist1d_fit, compute_root_workflow
np.seterr(divide='ignore', invalid='ignore')

# Load macro configuration
user_input = initialize_macro("01Calibration",["config_file","root_file","rewrite","show","debug"],default_dict={}, debug=True)
if user_input["debug"]: user_input = check_macro_config(user_input,debug=user_input["debug"])
# Load analysis data and configuration
config = user_input["config_file"].split("/")[-1].split("_config")[0]
info = read_input_file(user_input["config_file"],path="../config/",debug=user_input["debug"])

acc = 50
data_filter = {"max_energy": 20, "min_energy": 0, "pre_nhits": 4, "primary": True, "neutron": True}
data, filter_idx = compute_root_workflow(user_input, info, data_filter, workflow="BASIC", debug=user_input["debug"])
print_colored("-> Found %i electron candidates out of %i events!"%(len(filter_idx),data["Event"].size),"SUCCESS")

# Plot the calibration workflow
fig = make_subplots(rows=2, cols=2,subplot_titles=("Time Profile","Correction","Charge Calibration"))
fig, corr_popt, perr = get_hist2d_fit(np.abs(data["Time"][filter_idx]),(data["Charge"]/data["ElectronE"])[filter_idx],acc,fig,1,1,func_type="exponential",debug=user_input["debug"])

# Energy computation
data["Correction"] = np.exp(np.abs(data["Time"])/corr_popt[1])
data["Energy"]     = data["Charge"]*data["Correction"]/corr_popt[0]

fig.add_trace(go.Histogram2d(x=np.abs(data["Time"][filter_idx]),
    y=(data["Charge"]*data["Correction"]/(corr_popt[0]*data["ElectronE"]))[filter_idx],
    coloraxis="coloraxis",nbinsx=acc,nbinsy=acc),row=1,col=2)

fig, reco_popt, perr = get_hist2d_fit(data["ElectronE"][filter_idx],data["Energy"][filter_idx],acc,fig,2,1,func_type="linear",debug=user_input["debug"])
fig, res_popt, perr = get_hist1d_fit((data["Energy"]/data["ElectronE"])[filter_idx],2*acc,fig,2,2,func_type="gauss",debug=user_input["debug"])

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

if not os.path.exists("../config/"+config+"/"+config+"_calib/"): 
    os.makedirs("../config/"+config+"/"+config+"_calib/")

fig = format_coustom_plotly(fig,fontsize=18,matches=(None,None))
if not os.path.exists("../images/calibration/%s_calibration/"%config): os.makedirs("../images/calibration/%s_calibration/"%config)
if user_input["rewrite"]:
    fig.write_image("../images/calibration/%s_calibration/%s_calibration.png"%(config,config), width=2400, height=1080)
    print_colored("-> Saved images to ../images/calibration/%s_calibration/%s_calibration.png"%(config,config),"SUCCESS")

# Save the true energy fit parameters to a txt file
if not os.path.exists("../config/"+config+"/"+config+"_calib/"+config+"_charge_correction.txt") or user_input["rewrite"]:
    with open("../config/"+config+"/"+config+"_calib/"+config+"_charge_correction.txt",'w') as f:
        f.write("CHARGE_AMP: %f\n"%corr_popt[0])
        f.write("ELECTRON_TAU: %f\n"%corr_popt[1])
    print_colored("-> Saved calibration parameters to ../config/"+config+"/"+config+"_calib/"+config+"_charge_correction.txt","SUCCESS")
else:
    print_colored("-> Found ../config/"+config+"/"+config+"_calib/"+config+"_charge_correction.txt","WARNING")
    print_colored("-> Please set rewrite to True to overwrite the file","WARNING")

if user_input["show"]: fig.show()