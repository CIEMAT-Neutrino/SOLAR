import copy
import numpy as np

from lib.io_functions import resize_subarrays

def process_hit_run(hit_run, tree, debug=False):
    '''
    This function takes a hit_run dictionary and returns a processed run dictionary.
    The processed run dictionary is designed to change the hit-based structure into a cluster-based structure.

    Args:
        hit_run (dict): Dictionary with the hit-based data.
        tree (str): Name of the tree to process.
        debug (bool): If True, the debug mode is activated.

    Returns:
        run (dict): Dictionary with the processed data.
    '''
    if debug: print("Processing run -> START")
    run = copy.deepcopy(hit_run)

    branches2keep = ["Event","Flag","Index","Main","Geometry","Version"]
    branches2rename = {}
    branches2max = {"TPC":"TPC","Ind0TPC":"Ind0TPC","Ind1TPC":"Ind1TPC","Ind0Time":"Ind0Time","Ind1Time":"Ind1Time","MainPDG":"MotherPDG"}
    branches2sum = {"Charge":"Charge","Ind0Charge":"Ind0Charge","Ind1Charge":"Ind1Charge"}
    branches2ave = {"Ind0RecoY":"Ind0Y","Ind1RecoY":"Ind1Y","Ind0RecoZ":"Ind0Z","Ind1RecoZ":"Ind1Z"}
    branches2mean = {"Time":"Time","RecoY":"Y", "RecoZ":"Z", "MainX":"MotherX", "MainY":"MotherY", "MainZ":"MotherZ","MainE":"MotherE"}
    branches2compute = ["NHits","Ind0NHits","Ind1NHits","AdjElectronClCharge","AdjGammaClCharge","AdjElectronClNHits","AdjGammaClNHits","AdjClCharge","AdjClTime","AdjClR"]
    opflash = ["AdjOpFlashR","AdjOpFlashPE","AdjOpFlashMaxPE","AdjOpFlashPur","AdjOpFlashX","AdjOpFlashY","AdjOpFlashZ","AdjOpFlashTime"]
    
    for branch in branches2rename:
        run[tree][branch] = hit_run[tree][branches2rename[branch]]
        if branch != branches2rename[branch]:
            del run[tree][branches2rename[branch]]

    for branch in branches2max:
        # Pick the value of the brach with the index corresponding to the maximum charge value
        run[tree][branch] = np.array([hit_run[tree][branches2max[branch]][idx][np.argmax(hit_run[tree]["Charge"][idx])] for idx in range(len(hit_run[tree][branches2max[branch]]))])
        if branch != branches2max[branch]:
            del run[tree][branches2max[branch]]
    
    for branch in branches2sum:
        run[tree][branch] = np.sum(hit_run[tree][branches2sum[branch]], axis=1)
        if branch != branches2sum[branch]:
            del run[tree][branches2sum[branch]]

    for branch in branches2ave:
        run[tree][branch] = np.mean(hit_run[tree][branches2ave[branch]], axis=1)
        if branch != branches2ave[branch]:
            del run[tree][branches2ave[branch]]

    for branch in branches2mean:
        run[tree][branch] = np.sum(hit_run[tree][branches2mean[branch]] * hit_run[tree]["Charge"], axis=1) / run[tree]["Charge"]
        if branch != branches2mean[branch]:
            del run[tree][branches2mean[branch]]

    for branch in branches2compute+opflash:
        run[tree][branch] = []

    this_event, this_flag, this_index = -1, -1, -1
    
    for idx,event in enumerate(hit_run[tree]["Event"]):
        for branch in branches2compute+opflash:
            run[tree][branch].append([])
        
        run[tree]["NHits"][idx].append(np.sum(hit_run[tree]["Charge"][idx] != 0))
        run[tree]["Ind0NHits"][idx].append(np.sum(hit_run[tree]["Ind0Charge"][idx] != 0))
        run[tree]["Ind1NHits"][idx].append(np.sum(hit_run[tree]["Ind1Charge"][idx] != 0))

        if hit_run[tree]["Main"][idx] == True:
            run[tree]["AdjOpFlashR"][idx] = [np.sqrt((hit_run[tree]["OpFlashY"][idx][i] - run[tree]["RecoY"][idx])**2 + (hit_run[tree]["OpFlashZ"][idx][i] - run[tree]["RecoZ"][idx])**2) for i in range(len(hit_run[tree]["OpFlashY"][idx]))]
            run[tree]["AdjOpFlashPE"][idx] = [hit_run[tree]["OpFlashPE"][idx][i] for i in range(len(hit_run[tree]["OpFlashPE"][idx]))]
            run[tree]["AdjOpFlashMaxPE"][idx] = [hit_run[tree]["OpFlashMaxPE"][idx][i] for i in range(len(hit_run[tree]["OpFlashMaxPE"][idx]))]
            run[tree]["AdjOpFlashPur"][idx] = [hit_run[tree]["OpFlashPur"][idx][i] for i in range(len(hit_run[tree]["OpFlashPur"][idx]))]
            run[tree]["AdjOpFlashX"][idx] = [hit_run[tree]["OpFlashX"][idx][i] for i in range(len(hit_run[tree]["OpFlashX"][idx]))]
            run[tree]["AdjOpFlashY"][idx] = [hit_run[tree]["OpFlashY"][idx][i] for i in range(len(hit_run[tree]["OpFlashY"][idx]))]
            run[tree]["AdjOpFlashZ"][idx] = [hit_run[tree]["OpFlashZ"][idx][i] for i in range(len(hit_run[tree]["OpFlashZ"][idx]))]
            run[tree]["AdjOpFlashTime"][idx] = [hit_run[tree]["OpFlashT"][idx][i] for i in range(len(hit_run[tree]["OpFlashT"][idx]))]

            this_event = event
            this_flag = hit_run[tree]["Flag"][idx]
            this_index = hit_run[tree]["Index"][idx]
            this_main_idx = idx
            # print("Main idx found: ", this_event, this_flag, this_index)

        if hit_run[tree]["Event"][idx] == this_event and hit_run[tree]["Flag"][idx] == this_flag and hit_run[tree]["Index"][idx] == this_index and hit_run[tree]["Main"][idx] == False:
            # Append to the AdjElectronCharge only the charge of the hits with corresponding MotherPDG == 11
            run[tree]["AdjElectronClCharge"][this_main_idx].append(run[tree]["Charge"][idx]*(run[tree]["MainPDG"][idx] == 11))
            run[tree]["AdjGammaClCharge"][this_main_idx].append(run[tree]["Charge"][idx]*(run[tree]["MainPDG"][idx] == 22))
            
            if run[tree]["MainPDG"][idx] == 11:
                run[tree]["AdjElectronClNHits"][this_main_idx].append(run[tree]["NHits"][idx][0])
            if run[tree]["MainPDG"][idx] == 22:
                run[tree]["AdjGammaClNHits"][this_main_idx].append(run[tree]["NHits"][idx][0])

            run[tree]["AdjClCharge"][this_main_idx].append(run[tree]["Charge"][idx])
            run[tree]["AdjClTime"][this_main_idx].append(run[tree]["Time"][idx])
            run[tree]["AdjClR"][this_main_idx].append(np.sqrt((run[tree]["RecoY"][idx] - run[tree]["RecoY"][this_main_idx])**2 + (run[tree]["RecoZ"][idx] - run[tree]["RecoZ"][this_main_idx])**2))
            # print("Adding idx charge: ", np.sum(hit_run[tree]["Charge"][idx], axis=0), " to main idx: ", this_main_idx)    

    old_branches = []
    for branch in run[tree].keys():
        if branch in list(branches2max.keys())+list(branches2sum.keys())+list(branches2mean.keys())+branches2compute+opflash:
            resized_array = resize_subarrays(run[tree][branch], branch, 0, debug=debug)
            run[tree][branch] = resized_array
        elif branch not in branches2keep:
            old_branches.append(branch)
    
    for branch in old_branches:
        del run[tree][branch]
    
    if debug: print("Finished processing run -> DONE!")
    return run