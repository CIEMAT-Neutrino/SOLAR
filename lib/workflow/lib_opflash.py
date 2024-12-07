import numpy as np

from typing import Optional
from rich import print as rprint
from .functions import reshape_array, get_param_dict, remove_branches

from src.utils import get_project_root
root = get_project_root()


def compute_opflash_basic(run, configs, params: Optional[dict] = None, trees: Optional[list[str]] = None, rm_branches: bool = False, output: Optional[str] = None, debug=False):
    """
    Compute the basic OpFlash variables for the events in the run.
    """
    if output is None:
        output = ""
    # New branches
    for tree in trees:
        new_int_branches = ["OpFlashNum", "OpFlashSignalNum", "OpFlashBkgNum"]
        new_float_branches = ["OpFlashR", "OpFlashErrorY", "OpFlashErrorZ"]
        new_bool_branches = ["OpFlashSignal"]
        if tree == "Truth":
            prefix = ""
            sufix = ""
        elif tree == "Reco":
            prefix = "Adj"
            sufix = "Reco"
            new_int_branches = [prefix+branch for branch in new_int_branches]
            new_float_branches = [prefix+branch for branch in new_float_branches]
            new_bool_branches = [prefix+branch for branch in new_bool_branches]
        else:
            rprint(f"[red][ERROR] Invalid tree {tree} in compute_opflash_basic[/red]")
        
        for branch_list, branch_type in zip([new_float_branches, new_bool_branches], [np.float32, np.bool]):
            for branch in branch_list:    
                run[tree][branch] = np.zeros(
                    (len(run[tree]["Event"]), len(run[tree][f"{prefix}OpFlashPE"][0])), dtype=branch_type)
        
        for branch in new_int_branches:
                run[tree][branch] = np.zeros(len(run[tree]["Event"]), dtype=int)
    
        for config in configs:
            info, params, output = get_param_dict(
                f"{root}/config/{config}/{config}", params, output, debug=debug)
            idx = np.where(
                (np.asarray(run[tree]["Geometry"]) == info["GEOMETRY"])
                * (np.asarray(run[tree]["Version"]) == info["VERSION"])
            )
            run[tree][f"{prefix}OpFlashSignal"][idx] = run[tree][f"{prefix}OpFlashPur"][idx] > 0

            run[tree][f"{prefix}OpFlashNum"][idx] = np.sum(
                run[tree][f"{prefix}OpFlashPE"][idx] != 0, axis=1)
            
            run[tree][f"{prefix}OpFlashSignalNum"][idx] = np.sum(
                (run[tree][f"{prefix}OpFlashPE"][idx] != 0) * (run[tree][f"{prefix}OpFlashSignal"][idx] == True), axis=1)
            
            run[tree][f"{prefix}OpFlashBkgNum"][idx] = np.sum(
                (run[tree][f"{prefix}OpFlashPE"][idx] != 0) * (run[tree][f"{prefix}OpFlashSignal"][idx] == False), axis=1)

            run[tree][f"{prefix}OpFlashErrorY"][idx] = np.absolute(run[tree][f"{prefix}OpFlash{sufix}Y"][idx] - \
                reshape_array(run[tree]["TNuY"][idx], len(
                    run[tree][f"{prefix}OpFlash{sufix}Y"][idx][0])))
            
            run[tree][f"{prefix}OpFlashErrorZ"][idx] = np.absolute(run[tree][f"{prefix}OpFlash{sufix}Z"][idx] - \
                reshape_array(run[tree]["TNuZ"][idx], len(
                    run[tree][f"{prefix}OpFlash{sufix}Z"][idx][0])))
            
            run[tree][f"{prefix}OpFlashR"][idx] = np.sqrt(np.power(
                run[tree][f"{prefix}OpFlashErrorY"][idx], 2) + np.power(run[tree][f"{prefix}OpFlashErrorZ"][idx], 2), dtype=np.float32)

    output += f"\tOpFlash basic variable computation \t-> Done!\n"
    return run, output, new_int_branches+new_float_branches


def compute_opflash_event(run, configs, params: Optional[dict] = None, trees: Optional[list[str]] = None, rm_branches: bool = False, output: Optional[str] = None, debug=False):
    """
    Compute the OpFlash variables for the events in the run.
    """
    # New branches
    new_branches = ["OpFlashMeanNHits", "OpFlashMeanR", "OpFlashMeanTime"]
    for branch in new_branches:
        run["Truth"][branch] = np.zeros(    
            len(run["Truth"]["Event"]), dtype=np.float32)
    
    for var in ["NHits", "R", "Time"]:
        # Weight the mean R by the PE
        run["Truth"][f"OpFlashMean{var}"] = np.sum(
            run["Truth"][f"OpFlash{var}"] * run["Truth"]["OpFlashPE"], axis=1) / np.sum(run["Truth"]["OpFlashPE"], axis=1)
        # In mean R, replace nans with -1e6
        run["Truth"][f"OpFlashMean{var}"] = np.nan_to_num(
            run["Truth"][f"OpFlashMean{var}"], nan=0, posinf=0, neginf=0)

    output += f"\tEvent OpFlashStat variables computation \t-> Done!\n"
    return run, output, new_branches


def compute_opflash_main(run, configs, params: Optional[dict] = None, trees: Optional[list[str]] = None, rm_branches: bool = False, output: Optional[str] = None, debug=False):
    """
    Compute the main OpFlash variables for the OpFlash with more PE.
    """
    # New branches
    for tree in trees:
        new_branches = ["MainOpFlashPur", "MainOpFlashNHits", "MainOpFlashTime", "MainOpFlashMaxPE", "MainOpFlashPE", "MainOpFlashR", "MainOpFlashErrorY", "MainOpFlashErrorZ"]
        new_bool_branches = ["MainOpFlashSignal"]
        prefix = ""
        if tree == "Reco":
            new_branches = ["MainAdjOpFlashPur", "MainAdjOpFlashNHits", "MainAdjOpFlashTime", "MainAdjOpFlashMaxPE", "MainAdjOpFlashPE", "MainAdjOpFlashR", "MainAdjOpFlashErrorY", "MainAdjOpFlashErrorZ"]
            new_bool_branches = ["MainAdjOpFlashSignal"]
            prefix = "Adj"
        
        for branch in new_branches:
            for tree in trees:
                run[tree][branch] = np.zeros(len(run[tree]["Event"]), dtype=np.float32)
        for branch in new_bool_branches:
            for tree in trees:
                run[tree][branch] = np.zeros(len(run[tree]["Event"]), dtype=np.bool)
        
        run[tree][f"Main{prefix}OpFlashIdx"] = np.zeros(len(run[tree]["Event"]), dtype=int)
        
        for config in configs:
            info, params, output = get_param_dict(
                f"{root}/config/{config}/{config}", params, output, debug=debug)
            idx = np.where(
                (np.asarray(run[tree]["Geometry"]) == info["GEOMETRY"])
                * (np.asarray(run[tree]["Version"]) == info["VERSION"])
            )

            run[tree][f"Main{prefix}OpFlashIdx"][idx] = np.argmax(run[tree][f"{prefix}OpFlashPE"][idx], axis=1)
            
            run[tree][f"Main{prefix}OpFlashPur"][idx] = run[tree][f"{prefix}OpFlashPur"][idx][np.arange(run[tree][f"{prefix}OpFlashPur"][idx].shape[0]), run[tree][f"Main{prefix}OpFlashIdx"][idx]]
            run[tree][f"Main{prefix}OpFlashSignal"][idx] = run[tree][f"Main{prefix}OpFlashPur"][idx] > 0

            run[tree][f"Main{prefix}OpFlashNHits"][idx] = run[tree][f"{prefix}OpFlashNHits"][idx][np.arange(run[tree][f"{prefix}OpFlashNHits"][idx].shape[0]), run[tree][f"Main{prefix}OpFlashIdx"][idx]]
            run[tree][f"Main{prefix}OpFlashTime"][idx] = run[tree][f"{prefix}OpFlashTime"][idx][np.arange(run[tree][f"{prefix}OpFlashTime"][idx].shape[0]), run[tree][f"Main{prefix}OpFlashIdx"][idx]]
            run[tree][f"Main{prefix}OpFlashPE"][idx] = run[tree][f"{prefix}OpFlashPE"][idx][np.arange(run[tree][f"{prefix}OpFlashPE"][idx].shape[0]), run[tree][f"Main{prefix}OpFlashIdx"][idx]]
            run[tree][f"Main{prefix}OpFlashMaxPE"][idx] = run[tree][f"{prefix}OpFlashMaxPE"][idx][np.arange(run[tree][f"{prefix}OpFlashMaxPE"][idx].shape[0]), run[tree][f"Main{prefix}OpFlashIdx"][idx]]


            run[tree][f"Main{prefix}OpFlashErrorY"][idx] = run[tree][f"{prefix}OpFlashErrorY"][idx][np.arange(run[tree][f"{prefix}OpFlashErrorY"][idx].shape[0]), run[tree][f"Main{prefix}OpFlashIdx"][idx]]
            run[tree][f"Main{prefix}OpFlashErrorZ"][idx] = run[tree][f"{prefix}OpFlashErrorZ"][idx][np.arange(run[tree][f"{prefix}OpFlashErrorZ"][idx].shape[0]), run[tree][f"Main{prefix}OpFlashIdx"][idx]]
            run[tree][f"Main{prefix}OpFlashR"][idx] = np.sqrt(np.power(
                run[tree][f"Main{prefix}OpFlashErrorY"][idx], 2) + np.power(run[tree][f"Main{prefix}OpFlashErrorZ"][idx], 2), dtype=np.float32)

    output += f"\tOpFlash main variables computation \t-> Done!\n"
    return run, output, new_branches+new_bool_branches


def compute_opflash_advanced(run, configs, params: Optional[dict] = None, rm_branches: bool = False, output: Optional[str] = None, debug=False):
    """
    Compute the OpFlash variables for the events in the run.
    """
    # Required branches
    required_branches = ["AdjOpFlashMaxPE", "AdjOpFlashPE"] 
    
    # New branches
    new_float_branches = ["AdjOpFlashRatio"]
    
    for branch_list, branch_type in zip([new_float_branches], [np.float32]):
        for branch in branch_list:
            run["Reco"][branch] = np.zeros(
                (len(run["Reco"]["Event"]), len(run["Reco"]["AdjOpFlashPE"][0])), dtype=branch_type)

    for config in configs:
        info, params, output = get_param_dict(
            f"{root}/config/{config}/{config}", params, output, debug=debug)
        idx = np.where(
            (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
            * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
        )

        run["Reco"]["AdjOpFlashRatio"][idx] = run["Reco"]["AdjOpFlashMaxPE"][idx] / \
            run["Reco"]["AdjOpFlashPE"][idx]
        # If AdjOpFlashRatio is 0 set it to Nan
        run["Reco"]["AdjOpFlashRatio"][idx] = np.where(
            run["Reco"]["AdjOpFlashRatio"][idx] == 0, np.nan, run["Reco"]["AdjOpFlashRatio"][idx])
    
    output += f"\tOpFlash variables computation \t-> Done!\n"
    return run, output, new_float_branches


def compute_opflash_matching(run, configs, params: Optional[dict] = None, rm_branches: bool = False, output: Optional[str] = None, debug=False):
    """
    Compute the OpFlash variables for the events in the run.
    """
    # New branches
    new_branches = ["MatchedOpFlashSignal", "MatchedOpFlashErrorY", "MatchedOpFlashErrorZ"]
    new_braches_types = [np.bool, np.float32, np.float32]
    for branch, branch_type in zip(new_branches, new_braches_types):
        run["Reco"][branch] = np.zeros(len(run["Reco"]["Event"]), dtype=branch_type)

    for config in configs:
        info, params, output = get_param_dict(
            f"{root}/config/{config}/{config}", params, output, debug=debug)
        idx = np.where(
            (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
            * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
        )
        run["Reco"]["MatchedOpFlashSignal"][idx] = run["Reco"]["MatchedOpFlashPur"][idx] > 0
        
        run["Reco"]["MatchedOpFlashErrorY"][idx] = np.absolute(run["Reco"]["MatchedOpFlashRecoY"][idx] - \
            run["Reco"]["TNuY"][idx])
        
        run["Reco"]["MatchedOpFlashErrorZ"][idx] = np.absolute(run["Reco"]["MatchedOpFlashRecoZ"][idx] - \
            run["Reco"]["TNuZ"][idx])
    
    output += f"\tMatchedOpFlash Signal Defined \t\t-> Done!\n"
    return run, output, new_branches


# def compute_opflash_matching(
#     run,
#     configs,
#     params: Optional[dict] = None,
#     rm_branches: bool = False,
#     output: Optional[str] = None,
#     debug=False,
# ) -> tuple[dict, str, list[str]]:
#     """
#     Match the reconstructed events with selected OpFlash candidates.
#     """
#     # New branches
#     new_branches = [
#         "FlashMathedIdx",
#         "FlashMatched",
#         "AssFlashIdx",
#         "MatchedOpFlashTime",
#         "MatchedOpFlashPE",
#         "MatchedOpFlashR",
#         "DriftTime",
#         "AdjClDriftTime",
#     ]
#     run["Reco"][new_branches[0]] = np.zeros(
#         (len(run["Reco"]["Event"]), len(run["Reco"]["AdjOpFlashR"][0])), dtype=bool
#     )
#     run["Reco"][new_branches[1]] = np.zeros(
#         len(run["Reco"]["Event"]), dtype=bool)
#     run["Reco"][new_branches[2]] = np.zeros(
#         len(run["Reco"]["Event"]), dtype=int)
#     run["Reco"][new_branches[3]] = np.zeros(
#         len(run["Reco"]["Event"]), dtype=np.float32)
#     run["Reco"][new_branches[4]] = np.zeros(
#         len(run["Reco"]["Event"]), dtype=np.float32)
#     run["Reco"][new_branches[5]] = np.zeros(
#         len(run["Reco"]["Event"]), dtype=np.float32)
#     run["Reco"][new_branches[6]] = np.zeros(
#         len(run["Reco"]["Event"]), dtype=np.float32)
#     run["Reco"][new_branches[7]] = np.zeros(
#         (len(run["Reco"]["Event"]), len(run["Reco"]["AdjClTime"][0])), dtype=np.float32
#     )
#     for config in configs:
#         info, params, output = get_param_dict(
#             f"{root}/config/{config}/{config}", params, output, debug=debug)
#         idx = np.where(
#             (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
#             * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
#         )
#         # If run["Reco"]["AdjClTime"][idx][0] empty skip the config:
#         if run["Reco"]["AdjOpFlashTime"][idx].sum() == 0:
#             continue

#         # Select all FlashMatch candidates
#         max_r_filter = run["Reco"]["AdjOpFlashR"][idx] < params["MAX_FLASH_R"]
#         min_pe_filter = run["Reco"]["AdjOpFlashPE"][idx] > params["MIN_FLASH_PE"]
#         signal_nan_filter = (run["Reco"]["AdjOpFlashSignal"][idx]
#                              > 0) * (run["Reco"]["AdjOpFlashSignal"][idx] < np.inf)

#         converted_array = reshape_array(
#             run["Reco"]["Time"][idx], len(
#                 run["Reco"]["AdjOpFlashTime"][idx][0])
#         )

#         max_drift_filter = (
#             np.abs(converted_array - 2 * run["Reco"]["AdjOpFlashTime"][idx])
#             < params["MAX_DRIFT_FACTOR"] * info["EVENT_TICKS"]
#         )
#         run["Reco"]["FlashMathedIdx"][idx] = (
#             (max_r_filter) * (min_pe_filter) *
#             (max_drift_filter) * (signal_nan_filter)
#         )

#         # If at least one candidate is found, mark the event as matched and select the best candidate
#         run["Reco"]["FlashMatched"][idx] = (
#             np.sum(run["Reco"]["FlashMathedIdx"][idx], axis=1) > 0
#         )
#         run["Reco"]["AssFlashIdx"][idx] = np.argmax(
#             run["Reco"]["AdjOpFlashSignal"][idx] *
#             run["Reco"]["FlashMathedIdx"][idx],
#             axis=1,
#         )

#         # Compute the drift time and the matched PE
#         run["Reco"]["MatchedOpFlashTime"][idx] = run["Reco"]["AdjOpFlashTime"][
#             idx[0], run["Reco"]["AssFlashIdx"][idx]
#         ]
#         run["Reco"]["MatchedOpFlashPE"][idx] = run["Reco"]["AdjOpFlashPE"][
#             idx[0], run["Reco"]["AssFlashIdx"][idx]
#         ]
#         run["Reco"]["MatchedOpFlashR"][idx] = run["Reco"]["AdjOpFlashR"][
#             idx[0], run["Reco"]["AssFlashIdx"][idx]
#         ]
#         run["Reco"]["DriftTime"][idx] = (
#             run["Reco"]["Time"][idx] - 2 *
#             run["Reco"]["MatchedOpFlashTime"][idx]
#         )
#         run["Reco"]["AdjClDriftTime"][idx] = (
#             run["Reco"]["AdjClTime"][idx]
#             - 2 * run["Reco"]["MatchedOpFlashTime"][idx][:, np.newaxis]
#         )

#     run = remove_branches(
#         run, rm_branches, ["FlashMathedIdx", "AssFlashIdx"], debug=debug
#     )
#     output += f"\tOpFlash matching \t\t-> Done!\n"
#     return run, output, new_branches