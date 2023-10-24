import numpy as np
from lib.io_functions import print_colored,read_input_file

def get_bkg_config(info,debug=False):
    '''
    This function returns a dictionary of background names according to the input file.
    Each key of the dictionary should be a tuple of the form (geometry,version) and each value should be a list of background names.
    **VARIABLES:**
    ** - info:**  dictionary containing the input file information
    ** - debug:** boolean to print debug messages
    '''
    bkg_dict = {}; color_dict = {}
    if (info["GEOMETRY"][0] == "hd" and info["VERSION"][0] == "hd_1x2x6"):
        bkg_list = ["Unknown",
            "Marley",
            "Ar39GenInLAr",
            "Kr85GenInLAr",
            "Ar42GenInLAr",
            "K42From42ArGenInLAr",
            "Rn222ChainRn222GenInLAr",
            "Rn222ChainPo218GenInLAr",
            "Rn222ChainPb214GenInLAr",
            "Rn222ChainBi214GenInLAr",
            "Rn222ChainPb210GenInLAr",
            "Rn220ChainPb212GenInLAr",
            "K40GenInCPA",
            "U238ChainGenInCPA",
            "K42From42ArGenInCPA",
            "Rn222ChainPo218GenInCPA",
            "Rn222ChainPb214GenInCPA",
            "Rn222ChainBi214GenInCPA",
            "Rn222ChainPb210GenInCPA",
            "Rn222ChainFromBi210GenInCPA",
            "Rn220ChainFromPb212GenInCPA",
            "Co60GenInAPA",
            "U238ChainGenInAPA",
            "Th232ChainGenInAPA",
            "Rn222ChainGenInPDS",
            "GammasInCavernwall",
            "GammasInFoam",
            "NeutronsInCavernwall",
            "GammasInCryostat",
            "GammasInCavern"]
            
    elif info["GEOMETRY"][0] == "hd" and info["VERSION"][0] == "hd_1x2x6_legacy":
        bkg_list = ["Unknown",
            "Marley",
            "APA",
            "Neutron",
            "Po210InLAr",
            "CPA",
            "Ar42InLAr",
            "Kr85InLAr",
            "Ar39InLAr",
            "Rn222InLAr",
            ]

    elif info["GEOMETRY"][0] == "vd" and info["VERSION"][0] == "vd_1x8x14_3view_30deg":
        bkg_list = ["Unknown",
            "Marley",
            "Ar39InLAr",
            "Kr85InLAr",
            "Ar42InLAr",
            "K42-Ar42InLAr",
            "Rn222InLAr",
            "CPAK42-Ar42",
            "CPAK40",
            "CPAU238",
            "PDSRn222",
            "Neutron",
            "Gamma"]
        
    elif info["GEOMETRY"][0] == "vd" and info["VERSION"][0] == "vd_1x8x6_3view_30deg":
        bkg_list = ["Unknown",
            "marley",
            "Ar39GenInLAr",
            "Kr85GenInLAr",
            "Ar42GenInLAr",
            "K42From42ArGenInLAr",
            "Rn222ChainRn222GenInLAr",
            "Rn220ChainPb212GenInLAr",
            "K40GenInCathode",
            "U238ChainGenInCathode",
            "K42From42ArGenInCathode",
            "Rn220ChainFromPb212GenInCathode",
            "Rn222ChainGenInPDS",
            "GammasInCavernwall",
            "GammasInFoam",
            "NeutronsInCavernwall",
            "GammasInCryostat",
            "GammasInCavern"]
    
    color_ass = get_bkg_color(bkg_list)
    for idx,bkg in enumerate(bkg_list):
        bkg_dict[idx] = bkg
        color_dict[idx] = color_ass[bkg]
        
    if debug: print_colored("Loaded background dictionary: %s"%str(bkg_dict),"INFO")
    return bkg_dict,color_dict

def get_gen_label(config_files, debug=False):
    gen_dict = dict()
    for idx,config in enumerate(config_files):
        info = read_input_file(config_files[config],path="../config/"+config+"/",debug=debug)
        geo = info["GEOMETRY"][0]
        version = info["VERSION"][0]
        for idx,gen in enumerate(get_bkg_config(info,debug)[0].values()):
            gen_dict[(geo,version,idx)] = gen
    return gen_dict

def weight_lists(mean_truth_df, count_truth_df, count_reco_df, config, debug=False):
    info = read_input_file(config,path="../config/"+config+"/",debug=debug)
    weight_list = get_bkg_weights(info)
    truth_values = []; reco_values = []
    for bkg_idx,bkg in enumerate(mean_truth_df.index):
        truth_values.append(mean_truth_df.values[bkg_idx]*np.power(info["TIMEWINDOW"][0]*weight_list[bkg],-1))
        reco_values.append(count_reco_df.values[0][bkg_idx]*np.power(count_truth_df.values[bkg_idx]*info["TIMEWINDOW"][0],-1))
        if debug: print(count_truth_df.values[bkg_idx])
        if debug: print(reco_values)
    return truth_values,reco_values

def get_bkg_color(name_list, debug=False):
    '''
    Get the color for each background according to its "simple" name.
    **VARIABLES:**
    ** - name_list:** list of background names.
    ** - debug:** boolean to print debug messages.
    '''
    color_dict = dict()
    simple_name_list = get_simple_name(name_list,debug=debug)

    for name in name_list:
        if simple_name_list[name] == "Unknown":
            color_dict[name] = "black"
        elif simple_name_list[name] == "Marley":
            color_dict[name] = "orange"
        elif simple_name_list[name] == "APA":
            color_dict[name] = "violet"
        elif simple_name_list[name] == "Neutron":
            color_dict[name] = "green"
        elif simple_name_list[name] == "CPA":
            color_dict[name] = "purple"
        elif simple_name_list[name] == "Ar42":
            color_dict[name] = "blue"
        elif simple_name_list[name] == "K42":
            color_dict[name] = "blue"
        elif simple_name_list[name] == "Kr85":
            color_dict[name] = "pink"
        elif simple_name_list[name] == "Ar39":
            color_dict[name] = "grey"
        elif simple_name_list[name] == "Rn22":
            color_dict[name] = "yellow"
        elif simple_name_list[name] == "Po210":
            color_dict[name] = "brown"
        elif simple_name_list[name] == "PDS":
            color_dict[name] = "red"
        else:
            color_dict[name] = "black"

    return color_dict

def reorder_df(df,info, bkg_dict, color_dict, debug=False):
    if info["GEOMETRY"][0] == "hd" and info["VERSION"][0] == "hd_1x2x6_legacy":
        order = ["Po210InLAr",
            "CPA",
            "APA",
            "Ar42InLAr",
            "Neutron",
            "Rn222InLAr",
            "Kr85InLAr",
            "Ar39InLAr"]
    
    elif info["GEOMETRY"][0] == "hd" and info["VERSION"][0] == "hd_1x2x6":
        order = ["Rn222ChainGenInPDS",
            "K40GenInCPA",
            "U238ChainGenInCPA",
            "K42From42ArGenInCPA",
            "Rn222ChainPo218GenInCPA",
            "Rn222ChainPb214GenInCPA",
            "Rn222ChainBi214GenInCPA",
            "Rn222ChainPb210GenInCPA",
            "Rn222ChainFromBi210GenInCPA",
            "Rn220ChainFromPb212GenInCPA",
            "Co60GenInAPA",
            "U238ChainGenInAPA",
            "Th232ChainGenInAPA",
            "Ar42GenInLAr",
            "K42From42ArGenInLAr",
            "NeutronsInCavernwall",
            "Rn222ChainRn222GenInLAr",
            "Rn222ChainPo218GenInLAr",
            "Rn222ChainPb214GenInLAr",
            "Rn222ChainBi214GenInLAr",
            "Rn222ChainPb210GenInLAr",
            "Rn220ChainPb212GenInLAr",
            "Kr85GenInLAr",
            "Ar39GenInLAr",
            "GammasInCryostat",
            "GammasInCavern",
            "GammasInFoam",
            "GammasInCavernwall",
            ]
            
    elif info["GEOMETRY"][0] == "vd" and info["VERSION"][0] == "vd_1x8x14_3view_30deg":
        order = ["CPAU238",
            "CPAK42-Ar42",
            "CPAK40",
            "Ar42",
            "K42-Ar42",
            "Neutron",
            "PDSRn222",
            "Rn222",
            "Kr85",
            "Ar39",
            "Gamma"]

    else: order = list(bkg_dict.values())[2:]

    df = df[order]
    color_list = []
    for bkg in order:
        color_list.append(color_dict[list(bkg_dict.values()).index(bkg)])
    
    if debug: print_colored("Reordered dataframe with columns: %s"%order,"INFO")
    return df,color_list

def get_simple_name(name_list, debug=False):
    simple_name = dict()
    basic_names = ["Ar42","Ar39","Kr85","Po210","Rn22"]
    for name in name_list:
        if "LAr" in name:
            for basic_name in basic_names:
                if basic_name in name:
                    simple_name[name] = basic_name
            if "K42" in name:
                simple_name[name] = "Ar42"
        
        elif "Gamma" in name:
            simple_name[name] = "Gamma"
        elif "Neutron" in name:
            simple_name[name] = "Neutron"
        elif "CPA" in name:
            simple_name[name] = "CPA"
        elif "APA" in name:
            simple_name[name] = "APA"
        elif "PDS" in name:
            simple_name[name] = "PDS"
        else:
            simple_name[name] = name

    if debug: print_colored("Loaded simple name dictionary: %s"%str(simple_name),"INFO")
    return simple_name

def get_gen_weights(config_files, names, debug=False):
    weights_dict = dict()
    for idx,config in enumerate(config_files):
        info = read_input_file(config_files[config],path="../config/"+config+"/",debug=debug)
        # Write a function that returns a dictionary of background names according to the input file. Each key of the dictionary should be a tuple of the form (geometry,version) and each value should be a list of background names.
        geo = info["GEOMETRY"][0]
        name_list = names[config]
        geo_weights_dict = get_bkg_weights(info,name_list)
        for idx,name in enumerate(name_list):
            weights_dict[(geo,name)] = geo_weights_dict[name]
    return weights_dict

def get_bkg_weights(info, names, debug = False):
    bkg_dict,color_dict = get_bkg_config(info,debug=False)
    weights_dict = dict()
    for bkg in bkg_dict.values():
        weights_dict[bkg] = 1
    if "wbkg" in names:
        weights_dict["wbkg"] = 1
        return weights_dict
    else:
        if info["GEOMETRY"][0] == "hd" and info["VERSION"][0] == "hd_1x2x6":
            custom_weights = {"NeutronsInCavernwall":1e3}
        if info["GEOMETRY"][0] == "hd" and info["VERSION"][0] == "hd_1x2x6_legacy":
            custom_weights = {"Po210":1e4,"APA":1e4,"CPA":1e2,"Ar42":1e4,"Neutron":1e2,"Rn222":1e4}
        if info["GEOMETRY"][0] == "vd" and info["VERSION"][0] == "vd_1x8x14_3view_30deg":
            custom_weights = {"Neutron":1e2}
        
        for bkg in custom_weights:
            weights_dict[bkg] = custom_weights[bkg]
        return weights_dict