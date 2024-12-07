import numpy as np

from typing import Optional
from .functions import reshape_array

def compute_ophit_basic(run, configs, params: Optional[dict] = None, rm_branches: bool = False, output: Optional[str] = None, debug=False):
    """
    Compute the OpFlash variables for the events in the run.
    """
    # Generate repeated arrays of TNuX, TNuY, TNuZ
    converted_y = reshape_array(
        run["Truth"]["TNuY"], len(run["Truth"]["OpHitY"][0]))
    converted_z = reshape_array(
        run["Truth"]["TNuZ"], len(run["Truth"]["OpHitZ"][0]))

    # New branches
    new_branches = ["OpHitDY", "OpHitDZ", "OpHitR"]
    for branch in new_branches:
        run["Truth"][branch] = np.zeros(
            (len(run["Truth"]["Event"]), len(run["Truth"]["OpHitT"])), dtype=np.float32)

    # Create OpHitR array
    run["Truth"]["OpHitDY"] = np.absolute(run["Truth"]["OpHitY"] - converted_y)
    run["Truth"]["OpHitDZ"] = np.absolute(run["Truth"]["OpHitZ"] - converted_z)
    run["Truth"]["OpHitR"] = np.sqrt(np.power(
        converted_y-run["Truth"]["OpHitY"], 2) + np.power(converted_z-run["Truth"]["OpHitZ"], 2), dtype=np.float32)

    output += f"\tBasic OpHit variables computation \t-> Done!\n"
    return run, output, new_branches


def compute_ophit_event(run, configs, params: Optional[dict] = None, rm_branches: bool = False, output: Optional[str] = None, debug=False):
    """
    Compute the OpFlash variables for the events in the run.
    """
    # New branches
    new_branches = ["OpHitMaxPE", "OpHitMeanPE", "OpHitMaxPEY", "OpHitMaxPEZ", "OpHitMaxPER", "OpHitMeanR", "OpHitMeanT"]
    for branch in new_branches:
        run["Truth"][branch] = np.zeros(    
            len(run["Truth"]["Event"]), dtype=np.float32)

    # Create OpHitR array
    run["Truth"]["OpHitMaxPE"] = np.max(run["Truth"]["OpHitPE"], axis=1)
    run["Truth"]["OpHitTotalPE"] = np.sum(run["Truth"]["OpHitPE"], axis=1)
    run["Truth"]["OpHitMeanPE"] = run["Truth"]["OpHitTotalPE"] / \
        np.sum(run["Truth"]["OpHitPE"] > 0, 1)
    # Create variable for the idx of the max PE
    run["Truth"]["OpHitMaxPEIdx"] = np.argmax(run["Truth"]["OpHitPE"], axis=1)
    run["Truth"]["OpHitMaxPEY"] = run["Truth"]["OpHitDY"][np.arange(run["Truth"]["OpHitDY"].shape[0]), run["Truth"]["OpHitMaxPEIdx"]]
    run["Truth"]["OpHitMaxPEZ"] = run["Truth"]["OpHitDZ"][np.arange(run["Truth"]["OpHitDZ"].shape[0]), run["Truth"]["OpHitMaxPEIdx"]]
    run["Truth"]["OpHitMaxPER"] = run["Truth"]["OpHitR"][np.arange(run["Truth"]["OpHitR"].shape[0]), run["Truth"]["OpHitMaxPEIdx"]]
    
    for var in ["R", "T"]:
        # Weight the mean R by the PE
        run["Truth"][f"OpHitMean{var}"] = np.sum(
            run["Truth"][f"OpHit{var}"] * run["Truth"]["OpHitPE"], axis=1) / np.sum(run["Truth"]["OpHitPE"], axis=1)
        # In mean R, replace nans with -1e6
        run["Truth"][f"OpHitMean{var}"] = np.nan_to_num(
            run["Truth"][f"OpHitMean{var}"], nan=0, posinf=0, neginf=0)

    output += f"\tEvent OpHitStat variables computation \t-> Done!\n"
    return run, output, new_branches


def compute_ophit_advanced(run, configs, params: Optional[dict] = None, rm_branches: bool = False, output: Optional[str] = None, debug=False):
    """
    Compute the OpFlash variables for the events in the run.
    """
    # Generate repeated arrays of TNuX, TNuY, TNuZ
    converted_x = reshape_array(
        run["Truth"]["TNuX"], len(run["Truth"]["OpHitX"][0]))

    converted_y = reshape_array(
        run["Truth"]["TNuY"], len(run["Truth"]["OpHitY"][0]))

    converted_z = reshape_array(
        run["Truth"]["TNuZ"], len(run["Truth"]["OpHitZ"][0]))

    flash_id_list = np.unique(run["Truth"]["OpHitFlashID"])
    # New branches
    new_branches = ["OpFlashRefPE", "OpFlashResidual", "OpFlashTime",
                    "OpFlashPur", "OpFlashSignal", "OpFlashNHit", "OpFlashPE"]
    for branch in new_branches:
        run["Truth"][branch] = np.zeros(
            (len(run["Truth"]["Event"]), len(flash_id_list)), dtype=np.float32)

    # Make empty array for OpHitResidual
    run["Truth"]["OpHitResidual"] = np.zeros(
        (len(run["Truth"]["Event"]), len(run["Truth"]["OpHitPE"][0])), dtype=np.float32)

    # Make each entry in the OpFlashID equal to its idx
    run["Truth"]["OpFlashID"] = np.ones(
        (len(run["Truth"]["Event"]), len(flash_id_list)), dtype=int)
    run["Truth"]["OpFlashID"] = run["Truth"]["OpFlashID"] * \
        np.arange(len(flash_id_list))

    # Change all nans in run["Truth"]["OpHitPur"] for 0
    run["Truth"]["OpHitPur"] = np.nan_to_num(
        run["Truth"]["OpHitPur"], nan=0.0, posinf=0.0, neginf=0.0)
    run["Truth"]["OpHitDecay"] = np.power(converted_x, 2) / (np.power(converted_x, 2) + np.power(
        converted_y - run["Truth"]["OpHitY"], 2) + np.power(converted_z - run["Truth"]["OpHitZ"], 2))
    run["Truth"]["OpHitRefPE"] = np.array(
        run["Truth"]["OpHitPE"] * run["Truth"]["OpHitPur"])

    for flash_id in flash_id_list:
        jdx = np.where(run["Truth"]["OpFlashID"] == flash_id)
        flash_id_filter = np.asarray(run["Truth"]["OpHitFlashID"] == flash_id)
        event_id_count = np.sum(flash_id_filter, axis=1)
        run["Truth"]["OpFlashRefPE"][jdx] = np.sum(
            run["Truth"]["OpHitRefPE"] * flash_id_filter, axis=1) / event_id_count
        run["Truth"]["OpHitRefPE"][flash_id_filter] = np.repeat(
            run["Truth"]["OpFlashRefPE"][jdx], event_id_count)
        run["Truth"]["OpFlashNHit"][jdx] = np.sum(flash_id_filter, axis=1)
        run["Truth"]["OpFlashPE"][jdx] = np.sum(
            run["Truth"]["OpHitPE"] * flash_id_filter, axis=1) / event_id_count

        run["Truth"]["OpFlashResidual"][jdx] = np.mean(np.power((run["Truth"]["OpHitPE"] - run["Truth"]["OpHitRefPE"] * run["Truth"]
                                                                 ["OpHitDecay"]) * flash_id_filter, 2), axis=1)/run["Truth"]["OpFlashNHit"][jdx]/run["Truth"]["OpFlashPE"][jdx]
        run["Truth"]["OpHitResidual"][flash_id_filter] = np.repeat(
            run["Truth"]["OpFlashResidual"][jdx], event_id_count)
        run["Truth"]["OpFlashTime"][jdx] = np.sum((run["Truth"]["OpHitT"] * run["Truth"]["OpHitPE"])
                                                  * flash_id_filter, axis=1) / np.sum(run["Truth"]["OpHitPE"] * flash_id_filter, axis=1)
        # run["Truth"]["OpFlashPur"][jdx] = np.sum(run["Truth"]["OpHitPur"][idx] * run["Truth"]["OpHitPE"][idx], axis=1) / np.sum(run["Truth"]["OpHitPE"][idx], axis=1)
    run["Truth"]["OpFlashSignal"] = (
        abs(run["Truth"]["OpFlashTime"]) < 5) == True

    output += f"\tAdvanced OpHit variables computation \t-> Done!\n"
    return run, output, new_branches