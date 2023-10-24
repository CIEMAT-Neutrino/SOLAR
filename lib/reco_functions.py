import gc
import numpy as np
from .io_functions import read_input_file,print_colored
from .fit_functions import exp


def get_param_dict(info,in_params,debug=False):
    '''
    Get the parameters for the reco workflow from the input files
    '''
    params = {
        # "MAX_ADJCL_R":None,
        # "MAX_ADJCL_TIME":None,
        "MAX_FLASH_R":None,
        "MIN_FLASH_PE":None,
        "RATIO_FLASH_PEvsR":None,
        "MAX_DRIFT_FACTOR": None,
        "DEFAULT_RECOX_TIME":None,
        "DEFAULT_ENERGY_TIME":None,
        "DEFAULT_ADJCL_ENERGY_TIME":None,
        # "MIN_BKG_R":None,
        # "MAX_BKG_CHARGE":None,
        # "COMPUTE_MATCHING":None,
        # "MAX_ENERGY_PER_HIT":None,
        # "MIN_ENERGY_PER_HIT":None,
        "FIDUTIAL_FACTOR":None,
        # "MAIN_PDG":None,
        # "MAX_MAIN_E":None,
        # "MIN_MAIN_E":None,
        "MAX_CL_E":None,
        "MIN_CL_E":None,
        # "MAX_ADJCL_E":None,
        # "MIN_ADJCL_E":None,
        }
    
    for param in params.keys(): 
        try:
            params[param] = in_params[param]
            if debug: print_colored("-> Using "+param+" from the input dictionary","WARNING")
        
        except KeyError:
            params[param] = info[param][0]
            if debug: print_colored("Using default "+param, "INFO")

    # Debug by printing the params dictionary (keys & vlaues)
    if debug: print_colored("Parameters used for the reco workflow:","INFO")
    if debug: print_colored("Loaded parameters: %s"%str(params),"INFO")
    return params

def compute_reco_workflow(run,config_files,params={},workflow="ANALYSIS",rm_branches=True,debug=False):
    '''
    Compute the reco variables for the events in the TTree
    - run: dictionary containing the TTree branches
    - config_files: dictionary containing the path to the configuration files for each geoemtry
    - params: dictionary containing the parameters for the reco functions
    - debug: print debug information
    '''
    # Compute reco variables
    if debug: print_colored("\nComputing reco workflow of type %s"%workflow,"INFO")

    if workflow == "BASIC":
        run = compute_primary_cluster(run,config_files,params,rm_branches=rm_branches,debug=debug)
        # run = compute_recoy(run,config_files,params,rm_branches=rm_branches,debug=debug)
    
    if workflow == "CALIBRATION":
        run = compute_primary_cluster(run,config_files,params,rm_branches=rm_branches,debug=debug)
        run = compute_recoy(run,config_files,params,rm_branches=rm_branches,debug=debug)

    if workflow == "VERTEXING":
        run = compute_primary_cluster(run,config_files,params,rm_branches=rm_branches,debug=debug)
        run = compute_recoy(run,config_files,params,rm_branches=rm_branches,debug=debug)
        run = compute_opflash_matching(run,config_files,params,rm_branches=rm_branches,debug=debug)
        run = compute_recox(run,config_files,params,rm_branches=rm_branches,debug=debug)

    if workflow == "ANALYSIS":
        run = compute_primary_cluster(run,config_files,params,rm_branches=rm_branches,debug=debug)
        run = compute_recoy(run,config_files,params,rm_branches=rm_branches,debug=debug)
        run = compute_opflash_matching(run,config_files,params,rm_branches=rm_branches,debug=debug)
        run = compute_recox(run,config_files,params,rm_branches=rm_branches,debug=debug)
        run = compute_cluster_energy(run,config_files,params,rm_branches=rm_branches,debug=debug)
    
    if workflow == "FULL":
        run = compute_primary_cluster(run,config_files,params,rm_branches=rm_branches,debug=debug)
        run = compute_opflash_matching(run,config_files,params,rm_branches=rm_branches,debug=debug)
        run = compute_recox(run,config_files,params,rm_branches=rm_branches,debug=debug)
        run = compute_cluster_energy(run,config_files,params,rm_branches=rm_branches,debug=debug)
        run = compute_reco_energy(run,config_files,params,rm_branches=rm_branches,debug=debug)
        run = compute_opflash_variables(run,config_files,params,rm_branches=rm_branches,debug=debug)
        run = compute_adjcl_variables(run,config_files,params,rm_branches=rm_branches,debug=debug)

    print_colored("\nReco workflow \t-> Done!\n", "SUCCESS")
    return run

def compute_primary_cluster(run, config_files, params={}, rm_branches=False, debug=False):
    '''
    Compute the primary cluster of the events in the TTree. This primary cluster is the one with the highest charge in the event.
    - run: dictionary containing the TTree branches
    - config_files: dictionary containing the path to the configuration files for each geoemtry
    - params: dictionary containing the parameters for the reco functions
    - debug: print debug information
    '''
    # New branches
    run["Reco"]["MaxAdjClCharge"] = np.zeros(len(run["Reco"]["Event"]),dtype=float)
    run["Reco"]["Primary"] = np.zeros(len(run["Reco"]["Event"]),dtype=bool)

    for config in config_files:
        info = read_input_file(config_files[config],path="../config/"+config+"/",debug=False)
        idx = np.where((np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"][0])*(np.asarray(run["Reco"]["Version"] )== info["VERSION"][0]))
        run["Reco"]["MaxAdjClCharge"][idx] = np.max(run["Reco"]["AdjClCharge"][idx],axis=1,initial=0)
        run["Reco"]["Primary"][idx] = run["Reco"]["Charge"][idx] > run["Reco"]["MaxAdjClCharge"][idx]

    print_colored("Primary cluster computation \t-> Done!", "SUCCESS")
    run = remove_branches(run,rm_branches,["MaxAdjClCharge"],debug=debug)
    return run

def compute_recoy(run,config_files,params={},rm_branches=False,debug=False):
    '''
    Compute the reconstructed Y position of the events in the TTree
    - run: dictionary containing the TTree
    - config_files: dictionary containing the path to the configuration files for each geoemtry
    - params: dictionary containing the parameters for the reco functions
        - "COMPUTE_MATCHING": if True, compute the matching between the colection and induction planes
    - debug: print debug information
    '''
    # New branches
    run["Reco"]["RecoY"] = 1e-6*np.ones(len(run["Reco"]["Event"]),dtype=float)
    run["Reco"]["Matching"] = -np.ones(len(run["Reco"]["Event"]),dtype=int)
    
    for config in config_files:
        info = read_input_file(config_files[config],path="../config/"+config+"/",debug=False)
        # Get values from the configuration file or use the ones given as input
        params = get_param_dict(info,params,debug=False)
        idx_2match = np.where((np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"][0])*(np.asarray(run["Reco"]["Version"] )== info["VERSION"][0])*(run["Reco"]["Ind0NHits"] > 2)*(run["Reco"]["Ind1NHits"] > 2))
        idx_1match = np.where((np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"][0])*(np.asarray(run["Reco"]["Version"] )== info["VERSION"][0])*(run["Reco"]["Ind0NHits"] <= 2)*(run["Reco"]["Ind1NHits"] > 2))
        idx_0match = np.where((np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"][0])*(np.asarray(run["Reco"]["Version"] )== info["VERSION"][0])*(run["Reco"]["Ind0NHits"] > 2)*(run["Reco"]["Ind1NHits"] <= 2))
        
        run["Reco"]["RecoY"][idx_2match] = (run["Reco"]["Ind0RecoY"][idx_2match]+run["Reco"]["Ind1RecoY"][idx_2match])/2
        run["Reco"]["RecoY"][idx_1match] = run["Reco"]["Ind1RecoY"][idx_1match]
        run["Reco"]["RecoY"][idx_0match] = run["Reco"]["Ind0RecoY"][idx_0match]
        
        if rm_branches == False:
            run["Reco"]["Matching"][idx_2match] = 2*np.ones(len(idx_2match[0]))
            run["Reco"]["Matching"][idx_1match] = 1*np.ones(len(idx_1match[0]))
            run["Reco"]["Matching"][idx_0match] = 0*np.ones(len(idx_0match[0]))

    print_colored("RecoY computation \t\t-> Done!", "SUCCESS")
    run = remove_branches(run,rm_branches,["Ind0RecoY","Ind1RecoY","Matching"],debug=debug)
    return run

def compute_opflash_matching(run,config_files,params={"MAX_FLASH_R":None,"MIN_FLASH_PE":None,"RATIO_FLASH_PEvsR":None},rm_branches=False,debug=False):
    '''
    Match the reconstructed events with selected OpFlash
    - run: dictionary containing the TTree branches
    - config_files: dictionary containing the path to the configuration files for each geoemtry
    - params: dictionary containing the parameters for the reco functions
        - "MAX_FLASH_R": maximum distance between the reconstructed vertex from cluster and the OpFlash
        - "MIN_FLASH_PE": minimum PE of the OpFlash
        - "RATIO_FLASH_PE": maximum ratio between the maximum PE of the OpFlash and the PE of the OpFlash
    - debug: print debug information
    '''
    # New branches
    run["Reco"]["FlashMathedIdx"]     = np.zeros((len(run["Reco"]["Event"]),len(run["Reco"]["AdjOpFlashR"][0])),dtype=bool)
    run["Reco"]["FlashMatched"]       = np.zeros(len(run["Reco"]["Event"]),dtype=bool)
    run["Reco"]["AssFlashIdx"]        = np.zeros(len(run["Reco"]["Event"]),dtype=int)
    run["Reco"]["MatchedOpFlashTime"] = np.zeros(len(run["Reco"]["Event"]),dtype=float)
    run["Reco"]["MatchedOpFlashPE"]   = np.zeros(len(run["Reco"]["Event"]),dtype=float)
    run["Reco"]["MatchedOpFlashR"]    = np.zeros(len(run["Reco"]["Event"]),dtype=float)
    run["Reco"]["DriftTime"]          = np.zeros(len(run["Reco"]["Event"]),dtype=float)
    run["Reco"]["AdjClDriftTime"]     = np.zeros((len(run["Reco"]["Event"]),len(run["Reco"]["AdjClTime"][0])),dtype=float)

    for config in config_files:
        info = read_input_file(config_files[config],path="../config/"+config+"/",debug=False)
        idx = np.where((np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"][0])*(np.asarray(run["Reco"]["Version"] )== info["VERSION"][0]))
        # Get values from the configuration file or use the ones given as input
        params = get_param_dict(info,params,debug=False)

        # Select all FlashMatch candidates
        max_r_filter = run["Reco"]["AdjOpFlashR"][idx] < params["MAX_FLASH_R"]
        min_pe_filter = run["Reco"]["AdjOpFlashPE"][idx] > params["MIN_FLASH_PE"]
        max_ratio_filter = (run["Reco"]["AdjOpFlashPE"][idx]/run["Reco"]["AdjOpFlashMaxPE"][idx]) > run["Reco"]["AdjOpFlashR"][idx]*params["RATIO_FLASH_PEvsR"]
        
        repeated_array = np.repeat(run["Reco"]["Time"][idx], len(run["Reco"]["AdjOpFlashTime"][idx][0]))
        converted_array = np.reshape(repeated_array, (-1, len(run["Reco"]["AdjOpFlashTime"][idx][0])))
        max_drift_filter = np.abs(converted_array - 2*run["Reco"]["AdjOpFlashTime"][idx]) < params["MAX_DRIFT_FACTOR"]*info["EVENT_TICKS"][0]
        run["Reco"]["FlashMathedIdx"][idx] = (max_r_filter)*(min_pe_filter)*(max_ratio_filter)*(max_drift_filter)
        
        # If at least one candidate is found, mark the event as matched and select the best candidate
        run["Reco"]["FlashMatched"][idx]   = np.sum(run["Reco"]["FlashMathedIdx"][idx],axis=1) > 0
        run["Reco"]["AssFlashIdx"][idx]    = np.argmax(run["Reco"]["AdjOpFlashPE"][idx]*run["Reco"]["FlashMathedIdx"][idx],axis=1)
        
        # Compute the drift time and the matched PE
        run["Reco"]["MatchedOpFlashTime"][idx] = run["Reco"]["AdjOpFlashTime"][idx[0], run["Reco"]["AssFlashIdx"][idx]]
        run["Reco"]["MatchedOpFlashPE"][idx]   = run["Reco"]["AdjOpFlashPE"][idx[0], run["Reco"]["AssFlashIdx"][idx]]
        run["Reco"]["MatchedOpFlashR"][idx]    = run["Reco"]["AdjOpFlashR"][idx[0], run["Reco"]["AssFlashIdx"][idx]]
        run["Reco"]["DriftTime"][idx]          = run["Reco"]["Time"][idx] - 2*run["Reco"]["MatchedOpFlashTime"][idx]
        run["Reco"]["AdjClDriftTime"][idx]     = run["Reco"]["AdjClTime"][idx] - 2*run["Reco"]["MatchedOpFlashTime"][idx][:,np.newaxis]

    print_colored("OpFlash matching \t\t-> Done!","SUCCESS")
    run = remove_branches(run,rm_branches,["FlashMathedIdx","AssFlashIdx"],debug=debug)
    return run

def compute_recox(run,config_files,params={"DEFAULT_RECOX_TIME":None},rm_branches=False,debug=False):
    '''
    Compute the reconstructed X position of the events in the TTree based on the drift time calculated from the OpFlash matching
    - run: dictionary containing the TTree
    - config_files: dictionary containing the path to the configuration files for each geoemtry
    - debug: print debug information
    '''
    
    run["Reco"]["RecoX"] = np.zeros(len(run["Reco"]["Event"]))
    run["Reco"]["AdjCldT"] = np.zeros((len(run["Reco"]["Event"]),len(run["Reco"]["AdjClTime"][0])),dtype=float)
    # run["Reco"]["AdjCl3DR"] = np.zeros((len(run["Reco"]["Event"]),len(run["Reco"]["AdjClR"][0])),dtype=float)

    for config in config_files:
        info = read_input_file(config_files[config],path="../config/"+config+"/",debug=False)
        idx = np.where((np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"][0])*(np.asarray(run["Reco"]["Version"] )== info["VERSION"][0]))
        # Get values from the configuration file or use the ones given as input
        params = get_param_dict(info,params,debug=False)

        repeated_array = np.repeat(run["Reco"]["Time"][idx], len(run["Reco"]["AdjClTime"][idx][0]))
        converted_array = np.reshape(repeated_array, (-1, len(run["Reco"]["AdjClTime"][idx][0])))
        run["Reco"]["AdjCldT"][idx] = np.abs(converted_array - run["Reco"]["AdjClTime"][idx])
        
        if info["GEOMETRY"][0] == "hd":
            tpc_filter = (run["Reco"]["TPC"])%2 == 0
            plus_idx = np.where((np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"][0])*(np.asarray(run["Reco"]["Version"] )== info["VERSION"][0])*(tpc_filter))
            run["Reco"]["RecoX"][plus_idx] = abs(run["Reco"][params["DEFAULT_RECOX_TIME"]][plus_idx])*(info["DETECTOR_SIZE_X"][0]/2)/info["EVENT_TICKS"][0]
            
            tpc_filter = (run["Reco"]["TPC"])%2 == 1
            mins_idx = np.where((np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"][0])*(np.asarray(run["Reco"]["Version"] )== info["VERSION"][0])*(tpc_filter))
            run["Reco"]["RecoX"][mins_idx] = -abs(run["Reco"][params["DEFAULT_RECOX_TIME"]][mins_idx])*(info["DETECTOR_SIZE_X"][0]/2)/info["EVENT_TICKS"][0]
        
        if info["GEOMETRY"][0] == "vd":
            run["Reco"]["RecoX"][idx] = -abs(run["Reco"][params["DEFAULT_RECOX_TIME"]][idx])*info["DETECTOR_SIZE_X"][0]/info["EVENT_TICKS"][0] + info["DETECTOR_SIZE_X"][0]/2

    print_colored("Computed RecoX \t\t\t-> Done!", "SUCCESS")
    run = remove_branches(run,rm_branches,["AdjCldT"],debug=debug)
    return run

def compute_cluster_energy(run,config_files,params={"DEFAULT_ENERGY_TIME":None,"DEFAULT_ADJCL_ENERGY_TIME":None},rm_branches=False,debug=False):
    '''
    Correct the charge of the events in the TTree according to the correction file.
    - run: dictionary containing the TTree
    - config_files: dictionary containing the path to the configuration files for each geoemtry
    - params: dictionary containing the parameters for the reco functions
    - debug: print debug information
    '''
    run["Reco"]["Correction"]      = np.ones(len(run["Reco"]["Event"]))
    run["Reco"]["Energy"]          = np.zeros(len(run["Reco"]["Event"]))
    run["Reco"]["AdjClCorrection"] = np.ones((len(run["Reco"]["Event"]),len(run["Reco"]["AdjClCharge"][0])),dtype=float)
    run["Reco"]["AdjClEnergy"]     = np.zeros((len(run["Reco"]["Event"]),len(run["Reco"]["AdjClCharge"][0])),dtype=float)

    for config in config_files:
        info = read_input_file(config_files[config],path="../config/"+config+"/",debug=False)
        idx = np.where((np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"][0])*(np.asarray(run["Reco"]["Version"] )== info["VERSION"][0]))
        
        params = get_param_dict(info,params,debug=False)
        corr_info = read_input_file(config+"_corr_config",path="../config/"+config+"/"+config+"_calib/",DOUBLES=["CHARGE_AMP","ELECTRON_TAU"],debug=False)
        corr_popt = [corr_info["CHARGE_AMP"][0],corr_info["ELECTRON_TAU"][0]]
        
        run["Reco"]["Correction"][idx]  = np.exp(np.abs(run["Reco"][params["DEFAULT_ENERGY_TIME"]][idx])/corr_popt[1])
        run["Reco"]["AdjClCorrection"][idx] = np.exp(np.abs(run["Reco"][params["DEFAULT_ADJCL_ENERGY_TIME"]][idx])/corr_popt[1])
        run["Reco"]["Energy"][idx]      = run["Reco"]["Charge"][idx]*run["Reco"]["Correction"][idx]/corr_popt[0]
        run["Reco"]["AdjClEnergy"][idx] = run["Reco"]["AdjClCharge"][idx]*run["Reco"]["Correction"][idx][:,np.newaxis]/corr_popt[0]

    if debug: print_colored("Clutser energy computation\t-> Done!", "SUCCESS")
    run = remove_branches(run,rm_branches,["Correction","AdjClCorrection"],debug=debug)
    return run

def compute_reco_energy(run,config_files,params={},rm_branches=False,debug=False):
    '''
    Compute the total energy of the events in the TTree
    - run: dictionary containing the TTree
    - config_files: dictionary containing the path to the configuration files for each geoemtry
    - params: dictionary containing the parameters for the reco functions
    - debug: print debug information
    '''    
    run["Reco"]["RecoEnergy"] = np.zeros(len(run["Reco"]["Event"]))
    run["Reco"]["TotalEnergy"] = np.zeros(len(run["Reco"]["Event"]))
    run["Reco"]["TotalAdjClEnergy"] = np.zeros(len(run["Reco"]["Event"]))
    for config in config_files:
        info = read_input_file(config_files[config],path="../config/"+config+"/",debug=False)
        idx = np.where((np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"][0])*(np.asarray(run["Reco"]["Version"] )== info["VERSION"][0]))
        params = get_param_dict(info,params,debug=False)
        calib_info = read_input_file(config+"_calib_config",path="../config/"+config+"/"+config+"_calib/",DOUBLES=["ENERGY_AMP","INTERSECTION"],debug=debug)
        
        run["Reco"]["TotalAdjClEnergy"][idx] = np.sum(run["Reco"]["AdjClEnergy"][idx],axis=1)
        run["Reco"]["TotalEnergy"][idx] = run["Reco"]["Energy"][idx] + run["Reco"]["TotalAdjClEnergy"][idx] + 1.9
        
        top_idx = np.where((np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"][0])*(np.asarray(run["Reco"]["Version"] )== info["VERSION"][0])*(run["Reco"]["TotalAdjClEnergy"] < 1.5))
        bot_idx = np.where((np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"][0])*(np.asarray(run["Reco"]["Version"] )== info["VERSION"][0])*(run["Reco"]["TotalAdjClEnergy"] >= 1.5))
        run["Reco"]["RecoEnergy"][top_idx] = run["Reco"]["Energy"][top_idx]*calib_info["ENERGY_AMP"][0] + calib_info["INTERSECTION"][0]
        run["Reco"]["RecoEnergy"][bot_idx] = run["Reco"]["Energy"][bot_idx]*calib_info["ENERGY_AMP"][0] + calib_info["INTERSECTION"][0] + 2.5

    print_colored("Total energy computation \t-> Done!", "SUCCESS")
    run = remove_branches(run,rm_branches,["TotalAdjClEnergy"],debug=debug)
    return run

def compute_opflash_variables(run,config_files,params={},rm_branches=False,debug=False):
    '''
    Compute the OpFlash variables for the events in the TTree
    - run: dictionary containing the TTree branches
    - config_files: dictionary containing the path to the configuration files for each geoemtry
    - params: dictionary containing the parameters for the reco functions
    - debug: print debug information
    '''
    # New branches
    run["Reco"]["AdjOpFlashNum"] = np.sum(run["Reco"]["AdjOpFlashR"] != 0,axis=1)

    print_colored("OpFlash variables computation \t-> Done!", "SUCCESS")
    return run

def compute_adjcl_variables(run,config_files,params={},rm_branches=False,debug=False):
    '''
    Compute the energy of the individual adjacent clusters based on the main calibration.
    - run: dictionary containing the TTree
    - config_files: dictionary containing the path to the configuration files for each geoemtry
    - params: dictionary containing the parameters for the reco functions
    - debug: print debug information
    '''
    # New branches
    run["Reco"]["AdjClNum"] = np.sum(run["Reco"]["AdjClCharge"] != 0,axis=1)
    run["Reco"]["TotalAdjClEnergy"] = np.zeros(len(run["Reco"]["Event"]),dtype=float)
    run["Reco"]["MaxAdjClEnergy"] = np.zeros(len(run["Reco"]["Event"]),dtype=float)
    for config in config_files:
        info = read_input_file(config_files[config],path="../config/"+config+"/",debug=False)
        idx = np.where((np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"][0])*(np.asarray(run["Reco"]["Version"] )== info["VERSION"][0]))
        params = get_param_dict(info,params,debug=False)
        run["Reco"]["TotalAdjClEnergy"][idx] = np.sum(run["Reco"]["AdjClEnergy"][idx],axis=1)
        run["Reco"]["MaxAdjClEnergy"][idx] = np.max(run["Reco"]["AdjClEnergy"][idx],axis=1)
        
    print_colored("AdjCl energy computation \t-> Done!", "SUCCESS")
    run = remove_branches(run,rm_branches,[],debug=debug)
    return run

def add_filter(filters,labels,this_filter,this_label,cummulative,debug):
    if cummulative:
        labels.append("All+"+this_label)
        filters.append((filters[-1])*(this_filter))
    else:
        labels.append(this_label)
        filters.append((filters[0])*this_filter)
    
    if debug: print("Filter "+this_label+" added -> Done!")
    return filters,labels

def compute_solarnuana_filters(run,config_files,config,name,gen,filter_list,params={},cummulative=True,debug=False):
    filters = []; labels = []
    info = read_input_file(config_files[config],path="../config/"+config+"/",debug=debug)
    
    # Select filters to be applied to the data
    geo_filter = np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"][0] 
    version_filter = np.asarray(run["Reco"]["Version"]) == info["VERSION"][0] 
    name_filter = np.asarray(run["Reco"]["Name"]) == name
    gen_filter = np.asarray(run["Reco"]["Generator"]) == gen
    base_filter = (geo_filter)*(version_filter)*(name_filter)*(gen_filter)
    
    labels.append("All")
    filters.append(base_filter)
    
    params = get_param_dict(info,params,debug=debug)
    for this_filter in filter_list:
        if this_filter == "Primary":
            primary_filter = run["Reco"]["Primary"] == True
            filters,labels = add_filter(filters,labels,(primary_filter),"Primary",cummulative,debug)

        if this_filter == "OpFlash":
            flash_filter = run["Reco"]["FlashMatched"] == True
            filters,labels = add_filter(filters,labels,flash_filter,"OpFlash",cummulative,debug)

        if this_filter == "AdjCl":
            adjcl_filter = np.sum(((run["Reco"]["AdjClR"] > params["MIN_BKG_R"])*(run["Reco"]["AdjClCharge"] < params["MAX_BKG_CHARGE"]) == False)*(run["Reco"]["AdjClCharge"] > 0),axis=1) > 0
            filters,labels = add_filter(filters,labels,adjcl_filter,"AdjCl",cummulative,debug)

        if this_filter == "EnergyPerHit":
            max_eperhit_filter = run["Reco"]["EnergyPerHit"] < params["MAX_ENERGY_PER_HIT"]
            min_eperhit_filter = run["Reco"]["EnergyPerHit"] > params["MIN_ENERGY_PER_HIT"]
            eperhit_filter = (max_eperhit_filter)*(min_eperhit_filter)
            filters,labels = add_filter(filters,labels,eperhit_filter,"EnergyPerHit",cummulative,debug)

        if this_filter == "Fiducial" or this_filter == "RecoX" or this_filter == "RecoY" or this_filter == "RecoZ":
            max_recox_filter = np.abs(run["Reco"]["RecoX"]) < (1-params["FIDUTIAL_FACTOR"])*(info["DETECTOR_SIZE_X"][0])/2
            min_recox_filter = np.abs(run["Reco"]["RecoX"]) > (params["FIDUTIAL_FACTOR"])*(info["DETECTOR_SIZE_X"][0])/2
            recox_filter = (max_recox_filter)*(min_recox_filter)
            recoy_filter = np.abs(run["Reco"]["RecoY"]) < (1-params["FIDUTIAL_FACTOR"])*(info["DETECTOR_SIZE_Y"][0])/2
            recoz_filter = ((run["Reco"]["RecoZ"]) < (1-params["FIDUTIAL_FACTOR"])*(info["DETECTOR_SIZE_Z"][0]))*(run["Reco"]["RecoZ"] > (params["FIDUTIAL_FACTOR"])*(info["DETECTOR_SIZE_Z"][0]))
            
            if this_filter == "Fiducial":
                filters,labels = add_filter(filters,labels,recox_filter*recoy_filter*recoz_filter,"Fiducial",cummulative,debug)
            elif this_filter == "RecoX":
                filters,labels = add_filter(filters,labels,recox_filter,"RecoX",cummulative,debug)
            elif this_filter == "RecoY":
                filters,labels = add_filter(filters,labels,recoy_filter,"RecoY",cummulative,debug)
            elif this_filter == "RecoZ":
                filters,labels = add_filter(filters,labels,recoz_filter,"RecoZ",cummulative,debug)

        if this_filter == "MainParticle":
            # mainpdg_filter = run["Reco"]["MainPDG"] != params["MAIN_PDG"]
            max_main_e_filter = run["Reco"]["MainE"] < params["MAX_MAIN_E"]
            min_main_e_filter = run["Reco"]["MainE"] > params["MIN_MAIN_E"]
            main_filter = (max_main_e_filter)*(min_main_e_filter)
            filters,labels = add_filter(filters,labels,main_filter,"MainParticle",cummulative,debug)
            
        if this_filter == "MainClEnergy":
            # mainpdg_filter = run["Reco"]["MainPDG"] != params["MAIN_PDG"]
            max_main_e_filter = run["Reco"]["Energy"] < params["MAX_CL_E"]
            min_main_e_filter = run["Reco"]["Energy"] > params["MIN_CL_E"]
            main_filter = (max_main_e_filter)*(min_main_e_filter)
            filters,labels = add_filter(filters,labels,main_filter,"MainClEnergy",cummulative,debug)
        
        if this_filter == "AdjClEnergy":
            # mainpdg_filter = run["Reco"]["MainPDG"] != params["MAIN_PDG"]
            max_main_e_filter = run["Reco"]["TotalAdjClEnergy"] < params["MAX_ADJCL_E"]
            min_main_e_filter = run["Reco"]["TotalAdjClEnergy"] > params["MIN_ADJCL_E"]
            main_filter = (max_main_e_filter)*(min_main_e_filter)
            filters,labels = add_filter(filters,labels,main_filter,"AdjClEnergy",cummulative,debug)

    if debug: print("Filters computation -> Done!")
    return filters,labels

def remove_branches(run, remove, branches, debug=False):
    if remove:
        if debug: print_colored("-> Removing branches: %s"%(branches),"WARNING")
        for branch in branches:
            run["Reco"].pop(branch)
            gc.collect()
    else:
        pass

    return run
        
# def old_compute_primary_cluster(run,config_files,params={"MAX_ADJCL_R":None,"MAX_ADJCL_TIME":None},rm_branches=False,debug=False):
#     '''
#     Compute the primary cluster of the events in the TTree
#     - run: dictionary containing the TTree branches
#     - config_files: dictionary containing the path to the configuration files for each geoemtry
#     - params: dictionary containing the parameters for the reco functions
#         - "MAX_ADJCL_R": maximum distance between the reconstructed vertex from cluster and the background clusters
#         - "MAX_ADJCL_TIME": maximum time between the reconstructed vertex from cluster and the background clusters
#     - debug: print debug information
#     '''
#     this_event = 0
#     this_flag = 0
#     run["Reco"]["Primary"] = np.ones(len(run["Reco"]["Event"]),dtype=bool)

#     for config in config_files:
#         info = read_input_file(config_files[config],path="../config/"+config+"/",debug=debug)
#         idx = np.where((np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"][0])*(np.asarray(run["Reco"]["Version"] )== info["VERSION"][0]))
#         # Get values from the configuration file or use the ones given as input
#         params = get_param_dict(info,params,debug=debug)
#         # Select primary cluster
#         for i in idx[0]:
#             if run["Reco"]["Event"][i] == this_event and run["Reco"]["Flag"][i] == this_flag:
#                 if (np.sqrt(np.power(run["Reco"]["RecoY"][i] - run["Reco"]["RecoY"][i-1],2) + np.power(run["Reco"]["RecoZ"][i] - run["Reco"]["RecoZ"][i-1],2)) < params["MAX_ADJCL_R"] and
#                     abs(run["Reco"]["Time"][i] - run["Reco"]["Time"][i-1]) < params["MAX_ADJCL_TIME"]):
#                     if run["Reco"]["Charge"][i] > run["Reco"]["Charge"][i-1]:
#                         run["Reco"]["Primary"][i-1] = False
#                     else:
#                         run["Reco"]["Primary"][i] = False
#             else:
#                 pass
#             this_event = run["Reco"]["Event"][i]
#             this_flag = run["Reco"]["Flag"][i]

#     print_colored("Primary cluster computation \t-> Done!", "SUCCESS")
#     return run

# def compute_charge_calibration(run,config_files,params={"DEFAULT_RECOCHARGE_TIME":None},rm_branches=False,debug=False):
#     '''
#     Calibrate the charge of the events in the TTree according to the calibration file
#     - run: dictionary containing the TTree
#     - config_files: dictionary containing the path to the configuration files for each geoemtry
#     - params: dictionary containing the parameters for the reco functions
#         - "DEFAULT_RECOCHARGE_TIME": default time to be used for the reco charge
#     - debug: print debug information
#     '''
#     run["Reco"]["Calibration"] = np.ones(len(run["Reco"]["Event"]))
#     for config in config_files:
#         info = read_input_file(config_files[config],path="../config/"+config+"/",debug=False)
#         idx = np.where((np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"][0])*(np.asarray(run["Reco"]["Version"] )== info["VERSION"][0]))
        
#         # Get values from the configuration file or use the ones given as input
#         params = get_param_dict(info,params,debug=False)
#         calib_file = config.split("config")[0]+"_calibration"
#         calib_info = read_input_file(calib_file,path="../config/"+config+"/",DOUBLES = ["CHARGE_AMP","E0_CORRECTION","MEAN_TAU"],debug=debug)
#         run["Reco"]["Calibration"][idx] = np.exp(np.abs(run["Reco"][params["DEFAULT_RECOCHARGE_TIME"]][idx])/calib_info["MEAN_TAU"][0])/calib_info["CHARGE_AMP"][0]
#         print_colored("-> Using "+params["DEFAULT_RECOCHARGE_TIME"]+" for the reco charge in config "+config, "WARNING")
    
#     print_colored("Computing charge corecction \t-> Done!", "SUCCESS")
#     return run

# def compute_total_charge(run,config_files,params={},rm_branches=False,debug=False):
#     '''
#     Compute the total charge of the events in the TTree
#     - run: dictionary containing the TTree
#     - config_files: dictionary containing the path to the configuration files for each geoemtry
#     - params: dictionary containing the parameters for the reco functions
#         - "MIN_BKG_R": minimum distance between the reconstructed vertex from cluster and the background clusters
#         - "MAX_BKG_CHARGE": maximum charge of the background clusters
#     - debug: print debug information
#     '''
#     run["Reco"]["AdjClMatched"] = np.zeros(len(run["Reco"]["Event"]),dtype=bool)
#     run["Reco"]["TotalAdjClCharge"] = np.zeros(len(run["Reco"]["Event"]),dtype=float)
#     run["Reco"]["TotalCharge"] = np.zeros(len(run["Reco"]["Event"]),dtype=float)
#     run["Reco"]["TotalNHits"] = np.zeros(len(run["Reco"]["Event"]),dtype=int)
#     for config in config_files:
#         if debug: print("Computing total charge for "+config+"...")
#         info = read_input_file(config_files[config],path="../config/"+config+"/",debug=False)
#         idx = np.where((np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"][0])*(np.asarray(run["Reco"]["Version"] )== info["VERSION"][0]))
#         # Get values from the configuration file or use the ones given as input
#         params = get_param_dict(info,params,debug=False)
#         bkg_r_filter = run["Reco"]["AdjClR"][idx] > params["MIN_BKG_R"]
#         bkg_charge_filter = run["Reco"]["AdjClCharge"][idx] < params["MAX_BKG_CHARGE"]
        
#         run["Reco"]["AdjClMatched"][idx] = np.sum((((bkg_r_filter)*(bkg_charge_filter)) == False)*(run["Reco"]["AdjClCharge"][idx] > 0),axis=1) > 0
#         run["Reco"]["TotalAdjClCharge"][idx] = np.sum(run["Reco"]["AdjClCharge"][idx]*(((bkg_r_filter)*(bkg_charge_filter)) == False)*(run["Reco"]["AdjClCharge"][idx] > 0),axis=1)
#         run["Reco"]["TotalCharge"][idx] = run["Reco"]["Charge"][idx] + run["Reco"]["TotalAdjClCharge"][idx]
#         run["Reco"]["TotalNHits"][idx] = run["Reco"]["NHits"][idx] + np.sum(run["Reco"]["AdjClNHit"][idx],axis=1)
    
#     print_colored("Total charge computation \t-> Done!", "SUCCESS")
#     return run

# def compute_total_energy(run,config_files,params={"DEFAULT_CHARGE_VARIABLE":None},rm_branches=False,debug=False):
#     '''
#     Compute the total energy of the events in the TTree
#     - run: dictionary containing the TTree
#     - config_files: dictionary containing the path to the configuration files for each geoemtry
#     - params: dictionary containing the parameters for the reco functions
#     - debug: print debug information
#     '''    
#     run["Reco"]["Energy"]      = np.zeros(len(run["Reco"]["Event"]))
#     run["Reco"]["TotalEnergy"] = np.zeros(len(run["Reco"]["Event"]))
#     run["Reco"]["EnergyPerHit"] = np.zeros(len(run["Reco"]["Event"]))
#     for config in config_files:
#         info = read_input_file(config_files[config],path="../config/"+config+"/",debug=False)
#         idx = np.where((np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"][0])*(np.asarray(run["Reco"]["Version"] )== info["VERSION"][0]))
#         # Get values from the configuration file or use the ones given as input
#         params = get_param_dict(info,params,debug=False)
#         run["Reco"]["Energy"][idx]      = 1.9 + run["Reco"]["Calibration"][idx]*run["Reco"][params["DEFAULT_CHARGE_VARIABLE"]][idx]
#         run["Reco"]["TotalEnergy"][idx] = 1.9 + run["Reco"]["Calibration"][idx]*run["Reco"]["TotalCharge"][idx]
#         run["Reco"]["EnergyPerHit"][idx] = run["Reco"]["TotalEnergy"][idx]/run["Reco"]["TotalNHits"][idx]
#         print_colored("-> Using "+params["DEFAULT_CHARGE_VARIABLE"]+" for the energy in config "+config, "WARNING")
    
#     print_colored("Total energy computation \t-> Done!", "SUCCESS")
#     return run