import numpy as np

from typing import Optional
from .functions import reshape_array, get_param_dict, remove_branches


def compute_opflash_basic(run, configs, params: Optional[dict] = None, trees: Optional[list[str]] = None, rm_branches: bool = False, output: Optional[str] = None, debug=False):
    """
    Compute the basic OpFlash variables for the events in the run.
    """
    # New branches
    new_branches = ["OpFlashNum", "OpFlashR", "OpFlashErrorY", "OpFlashErrorZ"]
    for branch in new_branches:
        for tree in trees:
            run[tree][branch] = np.zeros(
                (len(run[tree]["Event"]), len(run[tree]["OpFlashPE"][0])), dtype=np.float32)
    for tree in trees:
        run[tree]["OpFlashNum"] = np.sum(
            run[tree]["OpFlashPE"] != 0, axis=1)

        run[tree]["OpFlashSignal"] = (run[tree]["OpFlashTime"] > 0) * \
            (run[tree]["OpFlashTime"] < 5)

        run[tree]["OpFlashErrorY"] = np.absolute(run[tree]["OpFlashY"] - \
            reshape_array(run[tree]["TNuY"], len(
                run[tree]["OpFlashY"][0])))
        
        run[tree]["OpFlashErrorZ"] = np.absolute(run[tree]["OpFlashZ"] - \
            reshape_array(run[tree]["TNuZ"], len(
                run[tree]["OpFlashZ"][0])))
        
        run[tree]["OpFlashR"] = np.sqrt(np.power(
            run[tree]["OpFlashErrorY"], 2) + np.power(run[tree]["OpFlashErrorZ"], 2), dtype=np.float32)

    output += f"\tOpFlash basic variable computation \t-> Done!\n"
    return run, output, new_branches


def compute_opflash_main(run, configs, params: Optional[dict] = None, trees: Optional[list[str]] = None, rm_branches: bool = False, output: Optional[str] = None, debug=False):
    """
    Compute the main OpFlash variables for the OpFlash with more PE.
    """
    # New branches
    new_branches = ["MainOpFlashPE", "MainOpFlashR", "MainOpFlashErrorY", "MainOpFlashErrorZ"]
    for branch in new_branches:
        for tree in trees:
            run[tree][branch] = np.zeros(len(run[tree]["Event"]), dtype=np.float32)
    
    for tree in trees:
        run[tree]["MainOpFlashIdx"] = np.argmax(run[tree]["OpFlashPE"], axis=1)
        run[tree]["MainOpFlashPE"] = run[tree]["OpFlashPE"][np.arange(run[tree]["OpFlashPE"].shape[0]), run[tree]["MainOpFlashIdx"]]

        run[tree]["MainOpFlashErrorY"] = run[tree]["OpFlashErrorY"][np.arange(run[tree]["OpFlashErrorY"].shape[0]), run[tree]["MainOpFlashIdx"]]
        run[tree]["MainOpFlashErrorZ"] = run[tree]["OpFlashErrorZ"][np.arange(run[tree]["OpFlashErrorZ"].shape[0]), run[tree]["MainOpFlashIdx"]]
        
        run[tree]["MainOpFlashR"] = np.sqrt(np.power(
            run[tree]["MainOpFlashErrorY"], 2) + np.power(run[tree]["MainOpFlashErrorZ"], 2), dtype=np.float32)

    output += f"\tOpFlash main variables computation \t-> Done!\n"
    return run, output, new_branches


def compute_opflash_advanced(run, configs, params: Optional[dict] = None, rm_branches: bool = False, output: Optional[str] = None, debug=False):
    """
    Compute the OpFlash variables for the events in the run.
    """
    # New branches
    new_branches = ["AdjOpFlashMaxPE", "AdjOpFlashNHit", "AdjOpFlashPE", "AdjOpFlashR", "AdjOpFlashRecoY",
                    "AdjOpFlashRecoZ", "AdjOpFlashSignal", "AdjOpFlashRatio", "AdjOpFlashErrorY", "AdjOpFlashErrorZ"]

    run["Reco"]["AdjOpFlashNum"] = np.sum(
        run["Reco"]["AdjOpFlashR"] != 0, axis=1)
    
    run["Reco"]["AdjOpFlashErrorY"] = run["Reco"]["AdjOpFlashRecoY"] - \
        reshape_array(run["Reco"]["TNuY"], len(
            run["Reco"]["AdjOpFlashRecoY"][0]))
    
    run["Reco"]["AdjOpFlashErrorZ"] = run["Reco"]["AdjOpFlashRecoZ"] - \
        reshape_array(run["Reco"]["TNuZ"], len(
            run["Reco"]["AdjOpFlashRecoZ"][0]))
        
    run["Reco"]["AdjOpFlashSignal"] = (run["Reco"]["AdjOpFlashTime"] > 0) * \
        (run["Reco"]["AdjOpFlashTime"] < 5)

    run["Reco"]["AdjOpFlashRatio"] = run["Reco"]["AdjOpFlashMaxPE"] / \
        run["Reco"]["AdjOpFlashPE"]
    # If AdjOpFlashRatio is 0 set it to Nan
    run["Reco"]["AdjOpFlashRatio"] = np.where(
        run["Reco"]["AdjOpFlashRatio"] == 0, np.nan, run["Reco"]["AdjOpFlashRatio"])
    
    output += f"\tOpFlash variables computation \t-> Done!\n"
    return run, output, new_branches


def compute_opflash_matching(run, configs, params: Optional[dict] = None, rm_branches: bool = False, output: Optional[str] = None, debug=False):
    """
    Compute the OpFlash variables for the events in the run.
    """
    # New branches
    new_branches = ["MatchedOpFlashSignal"]

    run["Reco"]["MatchedOpFlashSignal"] = (run["Reco"]["MatchedOpFlashTime"] > 0) * \
        (run["Reco"]["MatchedOpFlashTime"] < 5)
    
    output += f"\tOpFlash matching \t\t-> Done!\n"
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