import ROOT, root_numpy, json

import numpy as np

from ROOT          import TFile
from rich.progress import track

from lib.io_functions import print_colored, read_input_file
from lib.reco_functions import get_param_dict

def compute_root_workflow(user_input, info, data_filter, workflow="BASICS", debug=False):
    config = user_input["config_file"].split("/")[0]
    if user_input["debug"]: print_colored("\nLoading data...","DEBUG")
    input_file = TFile(info["PATH"][0]+info["NAME"][0]+user_input["root_file"][0]+".root")
    folder_name = input_file.GetListOfKeys()[0].GetName()
    reco = input_file.Get(folder_name + "/" + "SolarNuAnaTree")
    
    params = get_param_dict(user_input["config_file"],in_params={},debug=user_input["debug"])

    data = {}
    root_branches, new_branches, filter_idx = [], [], np.array([],dtype=int)
    object_types = {"INTEGERS":int,"DOUBLES":float,"STRINGS":str,"BOOLS":bool}

    data_config = json.load(open("../lib/workflow/"+workflow+".json","r"))
    for object_type in object_types.keys():
        new_branches = new_branches+list(data_config[object_type].keys())
    
    for branch in new_branches:
        if branch in reco.GetListOfBranches():
            root_branches.append(branch) 
        
    root = root_numpy.tree2array(reco, branches=root_branches)
    for object_type in object_types.keys():
        for key in data_config[object_type]: 
            try:
                data[key] = data_config[object_type][key]*np.asarray(root[key],dtype=object_types[object_type])
                if debug: print_colored("\t-> Found branch %s with type %s"%(key,object_types[object_type]),"DEBUG")
            except ValueError:
                data[key] = data_config[object_type][key]*np.ones(reco.GetEntries(),dtype=object_types[object_type])
                branch_info = (key,object_types[object_type],data_config[object_type][key])
                if debug: print_colored("\t-> Created branch %s as type %s with factor %.2e"%branch_info,"DEBUG")

    # analysis_info = read_input_file("analysis",INTEGERS=["RECO_ENERGY_RANGE","RECO_ENERGY_BINS","NADIR_RANGE","NADIR_BINS"],debug=False)
    # energy_edges = np.linspace(analysis_info["RECO_ENERGY_RANGE"][0],analysis_info["RECO_ENERGY_RANGE"][1],analysis_info["RECO_ENERGY_BINS"][0]+1)
    # energy_centers = (energy_edges[1:]+energy_edges[:-1])/2
    # bin_width = energy_edges[1]-energy_edges[0]
    
    if workflow in ["CALIBRATION","SMEARING","VERTEXING","ANALYSIS","FULL"]:
        calibration_info = read_input_file(config+"_charge_correction",path="../config/"+config+"/"+config+"_calib/",DOUBLES=["CHARGE_AMP","ELECTRON_TAU"],debug=user_input["debug"])
        corr_popt = [calibration_info["CHARGE_AMP"][0],calibration_info["ELECTRON_TAU"][0]]
    
    if workflow in ["SMEARING","VERTEXING","ANALYSIS","FULL"]:
        calib_info = read_input_file(config+"_energy_calibration",path="../config/"+config+"/"+config+"_calib/",DOUBLES=["ENERGY_AMP","INTERSECTION"],debug=False)

    for i in track(range(reco.GetEntries()),description="Computing data..."):
        reco.GetEntry(i)
        try: data["Primary"][i] = reco.Charge > max(reco.AdjClCharge)
        except ValueError: data["Primary"][i] = False
        if workflow in ["BASIC","CALIBRATION","SMEARING","VERTEXING","ANALYSIS","FULL"]:
            ############################
            # Primary Computation
            ############################
            data["AdjClNum"][i] = len(reco.AdjClR)
            data["AdjOpFlashNum"][i] = len(reco.AdjOpFlashR)
            # print("Primary: ",data["Primary"][i])

            ############################
            # True Computation
            ############################
            data["ElectronE"][i] = 1e3*reco.TMarleyE[2]
            for j in range(len(reco.TMarleyPDG)):
                if reco.TMarleyPDG[j] == 22: data["GammaE"][i]+=1e3*reco.TMarleyE[j]
                if reco.TMarleyPDG[j] == 2112: data["NeutronP"][i]+=1e3*reco.TMarleyP[j] 
            data["VisEnergy"][i] = data["ElectronE"][i] + data["GammaE"][i]

        if workflow in ["VERTEXING","ANALYSIS","FULL"]:
            ############################
            # RecoY Computation
            ############################
            if reco.Ind0NHits > 2 and reco.Ind1NHits > 2:
                data["RecoY"][i] = (reco.Ind0RecoY + reco.Ind1RecoY)/2
                data["Matching"][i] = 2
            elif reco.Ind0NHits > 2 and reco.Ind1NHits <= 2:
                data["RecoY"][i] = reco.Ind0RecoY
                data["Matching"][i] = 0
            elif reco.Ind0NHits <= 2 and reco.Ind1NHits > 2:
                data["RecoY"][i] = reco.Ind1RecoY
                data["Matching"][i] = 1
            else:
                data["RecoY"][i] = (reco.Ind0RecoY + reco.Ind1RecoY)/2
                # print_colored("WARNING: No ind matching found for event %d"%i,"WARNING")
    
        if workflow in ["ANALYSIS","FULL"]:
            ############################
            # Flash Matching Computation
            ############################
            for j in range(len(reco.AdjOpFlashR)):
                if reco.AdjOpFlashR.at(j) > params["MAX_FLASH_R"][0]: continue
                if reco.AdjOpFlashPE.at(j) < params["MIN_FLASH_PE"][0]: continue
                if reco.AdjOpFlashPE.at(j)/reco.AdjOpFlashMaxPE.at(j) < reco.AdjOpFlashR.at(j)*params["RATIO_FLASH_PEvsR"][0]: continue
                data["FlashMatched"][i] = True
                if data["MatchedOpFlashPE"][i] < reco.AdjOpFlashPE.at(j): 
                    data["AssFlashIdx"][i] = j
                    data["MatchedOpFlashR"][i] = reco.AdjOpFlashR.at(j)
                    data["MatchedOpFlashPE"][i] = reco.AdjOpFlashPE.at(j)
                    data["MatchedOpFlashTime"][i] = reco.AdjOpFlashTime.at(j)  
            data["DriftTime"][i] = reco.Time - 2*data["MatchedOpFlashTime"][i]
            
            ############################
            # RecoX Computation
            ############################
            if info["GEOMETRY"][0] == "hd":
                if  reco.TPC%2 == 0:
                    try: data["RecoX"][i] = abs(data[params["DEFAULT_RECOX_TIME"][0]][i])*(info["DETECTOR_SIZE_X"][0]/2)/info["EVENT_TICKS"][0]
                    except KeyError: data["RecoX"][i] = abs(reco.Time)*(info["DETECTOR_SIZE_X"][0]/2)/info["EVENT_TICKS"][0]
                else:
                    try: data["RecoX"][i] = -abs(data[params["DEFAULT_RECOX_TIME"][0]][i])*(info["DETECTOR_SIZE_X"][0]/2)/info["EVENT_TICKS"][0]
                    except KeyError: data["RecoX"][i] = -abs(reco.Time)*(info["DETECTOR_SIZE_X"][0]/2)/info["EVENT_TICKS"][0]

            if info["GEOMETRY"][0] == "vd":
                try: data["RecoX"][i] = -abs(data[params["DEFAULT_RECOX_TIME"][0]][i])*info["DETECTOR_SIZE_X"][0]/info["EVENT_TICKS"][0] + info["DETECTOR_SIZE_X"][0]/2
                except KeyError: data["RecoX"][i] = -abs(reco.Time)*info["DETECTOR_SIZE_X"][0]/info["EVENT_TICKS"][0] + info["DETECTOR_SIZE_X"][0]/2
    
        if workflow in ["CALIBRATION","SMEARING","FULL"]:
            ############################
            # Energy Computation
            ############################
            try: data["Correction"][i]  = np.exp(np.abs(data[params["DEFAULT_ENERGY_TIME"][0]][i])/corr_popt[1])
            except KeyError: data["Correction"][i]  = np.exp(np.abs(reco.Time)/corr_popt[1])
            data["Energy"][i] = reco.Charge*data["Correction"][i]/corr_popt[0]
            for z in range(len(reco.AdjClR)):
                try: adj_cl_correction = np.exp(np.abs(data[params["DEFAULT_ADJCL_ENERGY_TIME"][0]][i])/corr_popt[1])
                except KeyError: adj_cl_correction = np.exp(np.abs(reco.AdjClTime.at(z))/corr_popt[1])
                adj_cl_energy = reco.AdjClCharge.at(z)*adj_cl_correction/corr_popt[0]
                if adj_cl_energy > data["MaxAdjClEnergy"][i]: data["MaxAdjClEnergy"][i] = adj_cl_energy
                data["TotalAdjClEnergy"][i] += adj_cl_energy
            data["TotalEnergy"][i] = data["TotalAdjClEnergy"][i] + data["Energy"][i] + 1.9
            
        if workflow in ["SMEARING","FULL"]:
            ############################
            # Reco Energy Computation
            ############################
            if data["RecoEnergy"][i] > 1.5: data["Energy"][i]*calib_info["ENERGY_AMP"][0] + calib_info["INTERSECTION"][0]
            if data["RecoEnergy"][i] < 1.5: data["Energy"][i]*calib_info["ENERGY_AMP"][0] + calib_info["INTERSECTION"][0] + 2.5

        ############################
        # Reco Filter Computation
        ############################
        try:
            if data["Generator"][i] > data_filter["generator"]: continue
        except KeyError: pass
        try:
            if data["TNuE"][i] < data_filter["min_energy"]: continue
        except KeyError: pass
        try:
            if data["TNuE"][i] > data_filter["max_energy"]: continue
        except KeyError: pass
        try:
            if data["NHits"][i] < data_filter["pre_nhits"]: continue
        except KeyError: pass
        try:
            if data_filter["primary"] and data["Primary"][i] == False: continue
        except KeyError: pass
        
        try:
            if data_filter["neutron"]:
                for j in range(len(reco.TMarleyPDG)):
                    if reco.TMarleyPDG[j] == 2112: continue
        except KeyError: pass

        # Fill data
        filter_idx = np.append(filter_idx,i)
    print_colored("-> Finished computing data","SUCCESS")
    return data, filter_idx