import sys
sys.path.insert(0, '../')
import ROOT, root_numpy

import numpy          as np
import pandas         as pd
import plotly.express as px

from ROOT import TFile, TTree

from lib import compute_solarnuana_filters, compute_solar_spectrum, get_truth_count, get_gen_label, initialize_macro, check_macro_config, read_input_file, print_colored, make_subplots, format_coustom_plotly, compute_root_workflow, explode, get_simple_name, get_bkg_color
np.seterr(divide='ignore', invalid='ignore')

# Load macro configuration
user_input = initialize_macro("04Analysis",["config_file","root_file","rewrite","show","debug"],default_dict={}, debug=True)
user_input = check_macro_config(user_input,debug=user_input["debug"])

# Format input file names and load analysis data
config = user_input["config_file"].split("/")[-1].split("_config")[0]    
info = read_input_file(user_input["config_file"],path="../config/",debug=user_input["debug"])

analysis_info = read_input_file("analysis",INTEGERS=["RECO_ENERGY_RANGE","RECO_ENERGY_BINS","NADIR_RANGE","NADIR_BINS"],debug=False)
energy_edges = np.linspace(analysis_info["RECO_ENERGY_RANGE"][0],analysis_info["RECO_ENERGY_RANGE"][1],analysis_info["RECO_ENERGY_BINS"][0]+1)
energy_centers = (energy_edges[1:]+energy_edges[:-1])/2
bin_width = energy_edges[1]-energy_edges[0]

data_filter = {"max_energy": 30, "min_energy": 0, "pre_nhits": 3, "primary": True, "neutron": True}
true, reco, filter_idx = compute_root_workflow(user_input, info, data_filter, workflow="ANALYSIS", debug=user_input["debug"])
print_colored("-> Found %i electron candidates out of %i events!"%(len(filter_idx),reco["Event"].size),"SUCCESS")
run = {"Truth":true,"Reco":reco}

dict_array = []
df_dict, df_dict[config] = {}, {}
count_truth_df = get_truth_count(run,info,config,{config:user_input["root_file"][0]},debug=False)

# Get the list of generators for this file
for jdx,name in enumerate(user_input["root_file"]):
    truth_filter = (np.asarray(run["Truth"]["Name"]) == name)
    reco_filter = (np.asarray(run["Reco"]["Name"]) == name)
    gen_list = np.unique(run["Reco"]["Generator"][reco_filter])
    print("\n- Generators found for",name,":",gen_list,"\n")
    # Get the total number of events for this file
    if "wbkg" not in user_input["root_file"]:
        int_time = count_truth_df[name]*info["TIMEWINDOW"][0]
        print("Total number of events for",name,"is",str(count_truth_df[name]))
    else: 
        int_time = count_truth_df["Marley"].values[0]*info["TIMEWINDOW"][0]
        print("Total number of events for",name,"is",str(count_truth_df["Marley"]))
    # Start generator loop
    for kdx,gen in enumerate(gen_list):
        filters = compute_solarnuana_filters(run,{config:config+'_config'},config,name,gen,filter_list=["Primary"],
            params={"FIDUTIAL_FACTOR":0.075,"MIN_CL_E":5,"MAX_CL_E":20},cummulative=True,debug=False)

        gen_label = get_gen_label({config:config+'_config'})[(info["GEOMETRY"][0],info["VERSION"][0],gen)]
        this_dict_array,df_dict[config][gen_label] = compute_solar_spectrum(run,info,{config:config+'_config'},config,{config:user_input["root_file"]},name,
            gen,energy_edges,int_time,filters,truth_filter,reco_filter,input_dm2="DEFAULT",input_sin13="DEFAULT",input_sin12="DEFAULT",auto=False,save=True,debug=True)
                
        dict_array = dict_array + this_dict_array

solar_df = pd.DataFrame(dict_array)
plot_df = explode(solar_df,['TotalCounts','Energy','Efficiency'])

plot_df["SimpleName"] = plot_df["GenLabel"].map(get_simple_name(plot_df["GenLabel"].unique()))
this_plot_df = plot_df[(plot_df["TotalCounts"] > 0)]
# display(this_plot_df[this_plot_df["Energy"] > 0].groupby(["Name","Version","Filter","Generator"])["TotalCounts"].sum())
fig = px.bar(
    data_frame=this_plot_df,
    x="Energy",
    y="TotalCounts",
    facet_col="Version",
    facet_row="Filter",
    facet_row_spacing=0.1,
    color="GenLabel",
    color_discrete_sequence=pd.Series(this_plot_df["GenLabel"].unique()).map(get_bkg_color(this_plot_df["GenLabel"].unique())),
    barmode="overlay",
    )
        
# fig.update_layout(xaxis1_title="Reco Energy [MeV]",xaxis_title="Reco Energy [MeV]",yaxis_title="Counts [Hz]",yaxis1_title="Counts [Hz]")
fig = format_coustom_plotly(fig,log=(False,True),figsize=(1200,800),fontsize=18,ranges=([0,30],[0,9]),tickformat=(",.0f",".0s"),add_units=True)
fig.update_yaxes(title="Events/400kt-yr")
fig.update_layout(bargap=0)
fig.show()

solar_df.groupby("GenLabel")["TotalCounts"].sum()