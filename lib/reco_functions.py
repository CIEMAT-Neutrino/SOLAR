import gc, numba, json
import numpy as np
from icecream import ic
from .io_functions import read_input_file,print_colored

def compute_reco_workflow(run, configs, params={}, workflow="ANALYSIS", rm_branches=True, debug=False):
    '''
    Compute the reco variables for the events in the run.
    All functions are called in the order they are defined in this file.
    All functions get the same arguments.

    Args:
        run: dictionary containing the run data.
        configs: dictionary containing the path to the configuration files for each geoemtry.
        params: dictionary containing the parameters for the reco functions.
        workflow: string containing the reco workflow to be used.
        rm_branches: boolean to remove the branches used in the reco workflow.
        debug: print debug information.

    Returns:
        run (dict): dictionary containing the TTree with the new branches.
    '''
    # Compute reco variables
    if debug: print_colored("\nComputing reco workflow of type %s"%workflow,"INFO")

    if workflow == "BASIC":
        run = compute_primary_cluster(run,configs,params,rm_branches=rm_branches,debug=debug)
        run = compute_true_efficiency(run,configs,params,rm_branches=rm_branches,debug=debug)
        # run = compute_recoy(run,configs,params,rm_branches=rm_branches,debug=debug)
    
    if workflow == "ADJCL":
        run = compute_primary_cluster(run,configs,params,rm_branches=rm_branches,debug=debug)
        run = compute_true_efficiency(run,configs,params,rm_branches=rm_branches,debug=debug)
        run = compute_adjcl_basics(run,configs,params,rm_branches=rm_branches,debug=debug)
    
    if workflow == "CALIBRATION":
        run = compute_primary_cluster(run,configs,params,rm_branches=rm_branches,debug=debug)
        run = compute_true_efficiency(run,configs,params,rm_branches=rm_branches,debug=debug)
        run = compute_recoy(run,configs,params,rm_branches=rm_branches,debug=debug)

    if workflow == "VERTEXING":
        run = compute_primary_cluster(run,configs,params,rm_branches=rm_branches,debug=debug)
        run = compute_true_efficiency(run,configs,params,rm_branches=rm_branches,debug=debug)
        run = compute_recoy(run,configs,params,rm_branches=rm_branches,debug=debug)
        run = compute_opflash_matching(run,configs,params,rm_branches=rm_branches,debug=debug)
        run = compute_recox(run,configs,params,rm_branches=rm_branches,debug=debug)

    if workflow == "ANALYSIS":
        run = compute_primary_cluster(run,configs,params,rm_branches=rm_branches,debug=debug)
        run = compute_true_efficiency(run,configs,params,rm_branches=rm_branches,debug=debug)
        run = compute_recoy(run,configs,params,rm_branches=rm_branches,debug=debug)
        run = compute_opflash_matching(run,configs,params,rm_branches=rm_branches,debug=debug)
        run = compute_recox(run,configs,params,rm_branches=rm_branches,debug=debug)
        run = compute_cluster_energy(run,configs,params,rm_branches=rm_branches,debug=debug)
    
    if workflow == "FULL":
        run = compute_primary_cluster(run,configs,params,rm_branches=rm_branches,debug=debug)
        run = compute_true_efficiency(run,configs,params,rm_branches=rm_branches,debug=debug)
        run = compute_opflash_matching(run,configs,params,rm_branches=rm_branches,debug=debug)
        run = compute_recox(run,configs,params,rm_branches=rm_branches,debug=debug)
        run = compute_cluster_energy(run,configs,params,rm_branches=rm_branches,debug=debug)
        run = compute_reco_energy(run,configs,params,rm_branches=rm_branches,debug=debug)
        run = compute_opflash_variables(run,configs,params,rm_branches=rm_branches,debug=debug)
        run = compute_adjcl_variables(run,configs,params,rm_branches=rm_branches,debug=debug)

    print_colored("\nReco workflow \t-> Done!\n", "SUCCESS")
    return run

def compute_primary_cluster(run, configs, params={}, rm_branches=False, debug=False):
    '''
    Compute the primary cluster of the events in the run.
    This primary cluster is the one with the highest charge in the event.
    '''
    # New branches
    run["Reco"]["MaxAdjClCharge"] = np.zeros(len(run["Reco"]["Event"]),dtype=float)
    run["Reco"]["Primary"] = np.zeros(len(run["Reco"]["Event"]),dtype=bool)

    for config in configs:
        info = read_input_file(config+'/'+configs[config],debug=False)
        idx = np.where((np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"][0])*(np.asarray(run["Reco"]["Version"] )== info["VERSION"][0]))
        run["Reco"]["MaxAdjClCharge"][idx] = np.max(run["Reco"]["AdjClCharge"][idx],axis=1,initial=0)
        run["Reco"]["Primary"][idx] = run["Reco"]["Charge"][idx] > run["Reco"]["MaxAdjClCharge"][idx]

    print_colored("Primary cluster computation \t-> Done!", "SUCCESS")
    run = remove_branches(run,rm_branches,["MaxAdjClCharge"],debug=debug)
    return run

def compute_true_efficiency(run, configs, params={}, rm_branches=False, debug=False):
    run["Truth"]["RecoIndex"] = np.zeros(len(run["Truth"]["Event"]),dtype=int)
    run["Truth"]["RecoMatch"] = np.zeros(len(run["Truth"]["Event"]),dtype=bool)
    run["Truth"]["ClCount"] = np.zeros(len(run["Truth"]["Event"]),dtype=int)
    run["Truth"]["HitCount"] = np.zeros(len(run["Truth"]["Event"]),dtype=int)
    run["Reco"]["TrueIndex"] = np.zeros(len(run["Reco"]["Event"]),dtype=int)

    for config in configs:
        info = read_input_file(config+'/'+configs[config],debug=False)
        idx = np.where((np.asarray(run["Truth"]["Geometry"]) == info["GEOMETRY"][0])*(np.asarray(run["Truth"]["Version"]) == info["VERSION"][0]))
        
        result = generate_index(run["Truth"]["Event"][idx], run["Truth"]["Flag"][idx], run["Reco"]["Event"][idx], run["Reco"]["Flag"][idx], run["Reco"]["NHits"][idx], debug=debug)
        run["Truth"]["RecoIndex"][idx] = np.asarray(result[0])
        run["Truth"]["RecoMatch"][idx] = np.asarray(result[1])
        run["Truth"]["ClCount"][idx]   = np.asarray(result[2])
        run["Truth"]["HitCount"][idx]  = np.asarray(result[3])
        run["Reco"]["TrueIndex"][idx]  = np.asarray(result[4])
    
    print_colored("True efficiency computation \t-> Done!", "SUCCESS")
    run = remove_branches(run,rm_branches,[],debug=debug)
    return run

def compute_recoy(run, configs, params={}, rm_branches=False, debug=False):
    '''
    Compute the reconstructed Y position of the events in the run.   
    '''
    # New branches
    run["Reco"]["RecoY"] = 1e-6*np.ones(len(run["Reco"]["Event"]),dtype=float)
    run["Reco"]["Matching"] = -np.ones(len(run["Reco"]["Event"]),dtype=int)
    
    for config in configs:
        info = read_input_file(config+'/'+configs[config],debug=False)
        # Get values from the configuration file or use the ones given as input
        params = get_param_dict(config+"/"+configs[config],params,debug=debug)
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

def compute_opflash_matching(run,configs,params={"MAX_FLASH_R":None,"MIN_FLASH_PE":None,"RATIO_FLASH_PEvsR":None},rm_branches=False,debug=False):
    '''
    Match the reconstructed events with selected OpFlash candidates.
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

    for config in configs:
        info = read_input_file(config+'/'+configs[config],debug=False)
        idx = np.where((np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"][0])*(np.asarray(run["Reco"]["Version"] )== info["VERSION"][0]))
        # Get values from the configuration file or use the ones given as input
        params = get_param_dict(config+"/"+configs[config],params,debug=debug)

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

def compute_recox(run,configs,params={"DEFAULT_RECOX_TIME":None},rm_branches=False,debug=False):
    '''
    Compute the reconstructed X position of the events in the run.
    '''
    
    run["Reco"]["RecoX"] = np.zeros(len(run["Reco"]["Event"]))
    run["Reco"]["AdjCldT"] = np.zeros((len(run["Reco"]["Event"]),len(run["Reco"]["AdjClTime"][0])),dtype=float)
    # run["Reco"]["AdjCl3DR"] = np.zeros((len(run["Reco"]["Event"]),len(run["Reco"]["AdjClR"][0])),dtype=float)

    for config in configs:
        info = read_input_file(config+'/'+configs[config],debug=False)
        idx = np.where((np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"][0])*(np.asarray(run["Reco"]["Version"] )== info["VERSION"][0]))
        # Get values from the configuration file or use the ones given as input
        params = get_param_dict(config+"/"+configs[config],params,debug=debug)

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

def compute_cluster_energy(run,configs,params={"DEFAULT_ENERGY_TIME":None,"DEFAULT_ADJCL_ENERGY_TIME":None},rm_branches=False,debug=False):
    '''
    Correct the charge of the events in the run according to the correction file.
    '''
    run["Reco"]["Correction"]      = np.ones(len(run["Reco"]["Event"]))
    run["Reco"]["Energy"]          = np.zeros(len(run["Reco"]["Event"]))
    run["Reco"]["AdjClCorrection"] = np.ones((len(run["Reco"]["Event"]),len(run["Reco"]["AdjClCharge"][0])),dtype=float)
    run["Reco"]["AdjClEnergy"]     = np.zeros((len(run["Reco"]["Event"]),len(run["Reco"]["AdjClCharge"][0])),dtype=float)

    for config in configs:
        info = read_input_file(config+'/'+configs[config],debug=False)
        idx = np.where((np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"][0])*(np.asarray(run["Reco"]["Version"] )== info["VERSION"][0]))
        
        params = get_param_dict(config+"/"+configs[config],params,debug=debug)
        corr_info = read_input_file(config+"_charge_correction",path="../config/"+config+"/"+config+"_calib/",DOUBLES=["CHARGE_AMP","ELECTRON_TAU"],debug=False)
        corr_popt = [corr_info["CHARGE_AMP"][0],corr_info["ELECTRON_TAU"][0]]
        
        run["Reco"]["Correction"][idx]  = np.exp(np.abs(run["Reco"][params["DEFAULT_ENERGY_TIME"]][idx])/corr_popt[1])
        run["Reco"]["AdjClCorrection"][idx] = np.exp(np.abs(run["Reco"][params["DEFAULT_ADJCL_ENERGY_TIME"]][idx])/corr_popt[1])
        run["Reco"]["Energy"][idx]      = run["Reco"]["Charge"][idx]*run["Reco"]["Correction"][idx]/corr_popt[0]
        run["Reco"]["AdjClEnergy"][idx] = run["Reco"]["AdjClCharge"][idx]*run["Reco"]["Correction"][idx][:,np.newaxis]/corr_popt[0]

    if debug: print_colored("Clutser energy computation\t-> Done!", "SUCCESS")
    run = remove_branches(run,rm_branches,["Correction","AdjClCorrection"],debug=debug)
    return run

def compute_reco_energy(run,configs,params={},rm_branches=False,debug=False):
    '''
    Compute the total energy of the events in the run.
    '''    
    run["Reco"]["RecoEnergy"] = np.zeros(len(run["Reco"]["Event"]))
    run["Reco"]["TotalEnergy"] = np.zeros(len(run["Reco"]["Event"]))
    run["Reco"]["TotalAdjClEnergy"] = np.zeros(len(run["Reco"]["Event"]))
    for config in configs:
        info = read_input_file(config+'/'+configs[config],debug=False)
        idx = np.where((np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"][0])*(np.asarray(run["Reco"]["Version"] )== info["VERSION"][0]))
        params = get_param_dict(config+"/"+configs[config],params,debug=debug)
        calib_info = read_input_file(config+"_energy_calibration",path="../config/"+config+"/"+config+"_calib/",DOUBLES=["ENERGY_AMP","INTERSECTION"],debug=debug)
        
        run["Reco"]["TotalAdjClEnergy"][idx] = np.sum(run["Reco"]["AdjClEnergy"][idx],axis=1)
        run["Reco"]["TotalEnergy"][idx] = run["Reco"]["Energy"][idx] + run["Reco"]["TotalAdjClEnergy"][idx] + 1.9
        
        top_idx = np.where((np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"][0])*(np.asarray(run["Reco"]["Version"] )== info["VERSION"][0])*(run["Reco"]["TotalAdjClEnergy"] < 1.5))
        bot_idx = np.where((np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"][0])*(np.asarray(run["Reco"]["Version"] )== info["VERSION"][0])*(run["Reco"]["TotalAdjClEnergy"] >= 1.5))
        run["Reco"]["RecoEnergy"][top_idx] = run["Reco"]["Energy"][top_idx]*calib_info["ENERGY_AMP"][0] + calib_info["INTERSECTION"][0]
        run["Reco"]["RecoEnergy"][bot_idx] = run["Reco"]["Energy"][bot_idx]*calib_info["ENERGY_AMP"][0] + calib_info["INTERSECTION"][0] + 2.5

    print_colored("Total energy computation \t-> Done!", "SUCCESS")
    run = remove_branches(run,rm_branches,["TotalAdjClEnergy"],debug=debug)
    return run

def compute_opflash_variables(run,configs,params={},rm_branches=False,debug=False):
    '''
    Compute the OpFlash variables for the events in the run.
    '''
    # New branches
    run["Reco"]["AdjOpFlashNum"] = np.sum(run["Reco"]["AdjOpFlashR"] != 0,axis=1)

    print_colored("OpFlash variables computation \t-> Done!", "SUCCESS")
    return run

def compute_adjcl_basics(run,configs,params={},rm_branches=False,debug=False):
    '''
    Compute basic variables for the adjacent clusters

    Args:
        run: dictionary containing the TTree
        configs: dictionary containing the path to the configuration files for each geoemtry
        params: dictionary containing the parameters for the reco functions
        debug: print debug information
    '''

    # New branches
    run["Reco"]["AdjClNum"] = np.sum(run["Reco"]["AdjClCharge"] != 0,axis=1)
    run["Reco"]["TotalAdjClCharge"] = np.zeros(len(run["Reco"]["Event"]),dtype=float)
    run["Reco"]["MaxAdjClCharge"] = np.zeros(len(run["Reco"]["Event"]),dtype=float)
    run["Reco"]["MeanAdjClCharge"] = np.zeros(len(run["Reco"]["Event"]),dtype=float)
    run["Reco"]["MeanAdjClR"] = np.zeros(len(run["Reco"]["Event"]),dtype=float)
    run["Reco"]["MeanAdjClTime"] = np.zeros(len(run["Reco"]["Event"]),dtype=float)
    for config in configs:
        info = read_input_file(config+'/'+configs[config],debug=False)
        idx = np.where((np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"][0])*(np.asarray(run["Reco"]["Version"] )== info["VERSION"][0]))
        params = get_param_dict(config+"/"+configs[config],params,debug=debug)
        run["Reco"]["TotalAdjClCharge"][idx] = np.sum(run["Reco"]["AdjClCharge"][idx],axis=1)
        run["Reco"]["MaxAdjClCharge"][idx] = np.max(run["Reco"]["AdjClCharge"][idx],axis=1)
        run["Reco"]["MeanAdjClCharge"][idx] = np.mean(run["Reco"]["AdjClCharge"][idx],axis=1)
        run["Reco"]["MeanAdjClR"][idx] = np.mean(run["Reco"]["AdjClR"][idx],axis=1)
        run["Reco"]["MeanAdjClTime"][idx] = np.mean(run["Reco"]["AdjClTime"][idx],axis=1)

    print_colored("AdjCl basic computation \t-> Done!", "SUCCESS")
    run = remove_branches(run,rm_branches,[],debug=debug)
    return run

def compute_adjcl_variables(run,configs,params={},rm_branches=False,debug=False):
    '''
    Compute the energy of the individual adjacent clusters based on the main calibration.

    Args:
        run: dictionary containing the TTree
        configs: dictionary containing the path to the configuration files for each geoemtry
        params: dictionary containing the parameters for the reco functions
        debug: print debug information
    '''
    # New branches
    run["Reco"]["TotalAdjClEnergy"] = np.zeros(len(run["Reco"]["Event"]),dtype=float)
    run["Reco"]["MaxAdjClEnergy"] = np.zeros(len(run["Reco"]["Event"]),dtype=float)
    for config in configs:
        info = read_input_file(config+'/'+configs[config],debug=False)
        idx = np.where((np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"][0])*(np.asarray(run["Reco"]["Version"] )== info["VERSION"][0]))
        params = get_param_dict(config+"/"+configs[config],params,debug=debug)
        run["Reco"]["TotalAdjClEnergy"][idx] = np.sum(run["Reco"]["AdjClEnergy"][idx],axis=1)
        run["Reco"]["MaxAdjClEnergy"][idx] = np.max(run["Reco"]["AdjClEnergy"][idx],axis=1)
        
    print_colored("AdjCl energy computation \t-> Done!", "SUCCESS")
    run = remove_branches(run,rm_branches,[],debug=debug)
    return run

def get_param_dict(config_file, in_params, debug=False):
    '''
    Get the parameters for the reco workflow from the input files.
    '''
    params = read_input_file(config_file, preset="params_input", debug=False)
    for param in params.keys(): 
        try:
            params[param] = in_params[param]
            print_colored("-> Using %s: %s from the input dictionary"%(param,in_params[param]),"WARNING")
        except KeyError:
            pass

    # if debug: ic(params)
    return params

def add_filter(filters, labels, this_filter, this_label, cummulative, debug=False):
    if cummulative:
        labels.append("All+"+this_label)
        filters.append((filters[-1])*(this_filter))
    else:
        labels.append(this_label)
        filters.append((filters[0])*this_filter)
    
    if debug: print("Filter "+this_label+" added -> Done!")
    return filters,labels

def compute_solarnuana_filters(run,configs,config,name,gen,filter_list,params={},cummulative=True,debug=False):
    '''
    Compute the filters for the solar workflow computation.

    Args:
        run: dictionary containing the data.
        configs: dictionary containing the path to the configuration files for each geoemtry.
        config: string containing the name of the configuration.
        name: string containing the name of the reco.
        gen: string containing the name of the generator.
        filter_list: list of filters to be applied.
        params: dictionary containing the parameters for the reco functions.
        cummulative: boolean to apply the filters cummulative or not.
        debug: print debug information.
    
    Returns:
        filters: list of filters to be applied (each filter is a list of bools).
    '''
    filters = []; labels = []
    info = read_input_file(config+'/'+configs[config],debug=debug)
    
    # Select filters to be applied to the data
    geo_filter = np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"][0] 
    version_filter = np.asarray(run["Reco"]["Version"]) == info["VERSION"][0] 
    name_filter = np.asarray(run["Reco"]["Name"]) == name
    gen_filter = np.asarray(run["Reco"]["Generator"]) == gen
    base_filter = (geo_filter)*(version_filter)*(name_filter)*(gen_filter)
    
    labels.append("All")
    filters.append(base_filter)
    
    params = get_param_dict(config+"/"+configs[config],params,debug=debug)
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
    '''
    Remove branches from the run dictionary

    Args:
        run (dict): dictionary containing the TTree
        remove (bool): if True, remove the branches
        branches (list): list of branches to be removed
        debug (bool): print debug information

    Returns:
        run (dict): dictionary containing the TTree with the new branches
    '''
    if remove:
        if debug: print_colored("-> Removing branches: %s"%(branches),"WARNING")
        for branch in branches:
            run["Reco"].pop(branch)
            gc.collect()
    else: pass
    
    if debug: print_colored("-> Branches removed!","WARNING")
    return run

@numba.njit
def generate_index(true_event, true_flag, reco_event, reco_flag, reco_nhits, debug=False):
    true_index  = np.arange(len(true_event), dtype=np.int32)
    true_result = np.zeros(len(true_event), dtype=np.int32)-1
    true_match  = np.zeros(len(true_event), dtype=np.bool_)
    true_counts = np.zeros(len(true_event), dtype=np.int32)
    true_nhits  = np.zeros(len(true_event), dtype=np.int32)
    reco_result = np.zeros(len(reco_event), dtype=np.int32)
    end_j = 0
    for i in range(len(reco_event)):
        start_j = reco_result[i]
        j = 0
        for z in range(true_index[end_j], true_index[-1]+1): 
            if reco_event[i+1] != true_event[z] and reco_flag[i+1] != true_flag[z]:
                j = j+1
            else:
                start_j = end_j
                end_j = end_j+j
                break
        for k in range(start_j, end_j+1):
            if reco_event[i] == true_event[k] and reco_flag[i] == true_flag[k]:
                reco_result[i] = int(k)
                true_result[k] = int(i)
                true_match[k] = True
                true_counts[k] = true_counts[k]+1
                true_nhits[k] = true_nhits[k]+reco_nhits[i]
                break
    return true_result, true_match, true_counts, true_nhits, reco_result

def compute_event_idx(run, configs, params={}, rm_branches=False, debug=False):
    '''
    Compute real event index for the events in the run.
    '''
    # New branches
    run["Reco"]["EventIdx"] = np.zeros(len(run["Reco"]["Event"]),dtype=int)
    for config in configs:
        info = read_input_file(config+'/'+configs[config],debug=False)
        idx = np.where((np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"][0])*(np.asarray(run["Reco"]["Version"]) == info["VERSION"][0]))
        @numba.njit
        def generate_index(event, flag, index):
            for i in range(1, len(event)):
                if event[i] != event[i - 1] or flag[i] != flag[i - 1]:
                    index[i] = index[i - 1] + 1
                else:
                    index[i] = index[i - 1]
            return index
        run["Reco"]["EventIdx"][idx] = generate_index(run["Reco"]["Event"][idx], run["Reco"]["Flag"][idx], run["Reco"]["EventIdx"][idx])

    print_colored("Event index computation \t-> Done!", "SUCCESS")
    run = remove_branches(run,rm_branches,[],debug=debug)
    return run