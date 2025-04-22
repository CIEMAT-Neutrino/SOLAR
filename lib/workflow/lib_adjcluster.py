import numba
import numpy as np

from typing import Optional
from itertools import product
from lib.workflow.functions import remove_branches, get_param_dict, reshape_array

from src.utils import get_project_root
root = get_project_root()


def compute_adjcl_basics(run, configs, params: Optional[dict] = None, rm_branches=False, output: Optional[str] = None, debug=False):
    """
    Compute basic variables for the adjacent clusters

    Args:
        run: dictionary containing the TTree
        configs: dictionary containing the path to the configuration files for each geoemtry
        params: dictionary containing the parameters for the reco functions
        debug: print debug information
    """
    @ numba.njit
    def count_occurrences(arr, length):
        """
        Count the occurrences of each element in the array.

        Args:
            arr: array containing the elements
            length: length of the array
        """
        return [np.sum(arr == i) for i in range(length)]

    # New branches
    new_int_branches = ["AdjClNum", "AdjClSameGenNum"]
    
    new_branches = ["TotalAdjClCharge", "MaxAdjClCharge",
                    "MeanAdjClCharge", "MeanAdjClR", "MeanAdjClTime", "TotalAdjClSameGenCharge", "MaxAdjClSameGenCharge", "MeanAdjClSameGenCharge"]

    radial_limits = np.arange(0, 120, 20)
    new_radial_int_branches = ["TotalAdjClNum", "TotalAdjClSameGenNum"]
    new_radial_float_branches = ["TotalAdjClCharge", "TotalAdjClSameGenCharge"]

    
    for branch in new_int_branches:
        run["Reco"][branch] = np.zeros(
            len(run["Reco"]["Event"]), dtype=int)
    for branch in new_branches:
        run["Reco"][branch] = np.zeros(
            len(run["Reco"]["Event"]), dtype=np.float32)

    for limit, branch in product(radial_limits, new_radial_int_branches):
        run["Reco"][f"{branch}{limit}"] = np.zeros(
            (len(run["Reco"]["Event"])), dtype=int)
    
    for limit, branch in product(radial_limits, new_radial_float_branches):
        run["Reco"][f"{branch}{limit}"] = np.zeros(
            (len(run["Reco"]["Event"])), dtype=np.float32)
    
    run["Reco"]["AdjClGenNum"] = np.apply_along_axis(
        count_occurrences,
        arr=run["Reco"]["AdjClGen"],
        length=len(run["Reco"]["TruthPart"][0]) + 1,
        axis=1,
    )

    run["Reco"]["AdjClNum"] = np.sum(run["Reco"]["AdjClCharge"] != 0, axis=1)
    # Compute the number of adjacent clusters per radial limit
    for limit in radial_limits:
        run["Reco"][f"{new_radial_int_branches[0]}{limit}"] = np.sum(
            (run["Reco"]["AdjClR"] < limit) * (run["Reco"]["AdjClR"] > 0), axis=1
        )

    for config in configs:
        info, params, output = get_param_dict(
            f"{root}/config/{config}/{config}", params, output, debug=debug)
        idx = np.where(
            (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
            * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
        )
        run["Reco"]["TotalAdjClCharge"][idx] = np.sum(
            run["Reco"]["AdjClCharge"][idx], axis=1
        )
        run["Reco"]["MaxAdjClCharge"][idx] = np.max(
            run["Reco"]["AdjClCharge"][idx], axis=1
        )
        run["Reco"]["MeanAdjClCharge"][idx] = np.mean(
            run["Reco"]["AdjClCharge"][idx], axis=1
        )
        run["Reco"]["MeanAdjClR"][idx] = np.mean(
            run["Reco"]["AdjClR"][idx], axis=1
        )
        run["Reco"]["MeanAdjClTime"][idx] = np.mean(
            run["Reco"]["AdjClTime"][idx], axis=1
        )

        converted_array = reshape_array(
            run["Reco"]["Generator"][idx], len(run["Reco"]["AdjClGen"][idx][0]))

        gen_idx = converted_array == run["Reco"]["AdjClGen"][idx]
        
        run["Reco"]["AdjClSameGenNum"][idx] = np.sum(gen_idx, axis=1)
        for limit in radial_limits:
            run["Reco"][f"{new_radial_int_branches[1]}{limit}"][idx] = np.sum(
                (run["Reco"]["AdjClR"][idx] < limit) * (run["Reco"]["AdjClR"][idx] > 0) * gen_idx, axis=1
            )

        run["Reco"]["TotalAdjClSameGenCharge"][idx] = np.sum(
            run["Reco"]["AdjClCharge"][idx] * gen_idx, axis=1
        )
        for limit in radial_limits:
            rad_idx = (run["Reco"]["AdjClR"][idx] < limit) * (run["Reco"]["AdjClR"][idx] > 0)

            run["Reco"][f"{new_radial_float_branches[1]}{limit}"][idx] = np.sum(
                run["Reco"]["AdjClCharge"][idx] * gen_idx * rad_idx, axis=1
            )
            run["Reco"][f"{new_radial_float_branches[0]}{limit}"][idx] = np.sum(
                run["Reco"]["AdjClCharge"][idx] * rad_idx, axis=1
            )

        run["Reco"]["MaxAdjClSameGenCharge"][idx] = np.max(
            run["Reco"]["AdjClCharge"][idx] * gen_idx, axis=1
        )
        run["Reco"]["MeanAdjClSameGenCharge"][idx] = np.mean(
            run["Reco"]["AdjClCharge"][idx] * gen_idx, axis=1
        )

    run = remove_branches(run, rm_branches, [], debug=debug)
    output += f"\tAdjCl basic computation \t-> Done!\n"
    return run, output, new_branches


def compute_adjcl_advanced(run, configs, params: Optional[dict] = None, rm_branches=False, output: Optional[str] = None, debug=False):
    """
    Compute the energy of the individual adjacent clusters based on the main calibration.

    Args:
        run: dictionary containing the TTree
        configs: dictionary containing the path to the configuration files for each geoemtry
        params: dictionary containing the parameters for the reco functions
        debug: print debug information
    """
    # New branches
    new_branches = ["TotalAdjClEnergy", "TotalAdjClMainE", "MaxAdjClEnergy"]
    new_vector_branches = ["AdjCldTime", "AdjClRelCharge", "AdjClChargePerHit"]
    for branch in new_branches:
        run["Reco"][branch] = np.zeros(
            len(run["Reco"]["Event"]), dtype=np.float32)

    for branch in new_vector_branches:
        run["Reco"][branch] = np.zeros(
            (len(run["Reco"]["Event"]), len(run["Reco"]["AdjClTime"][0])), dtype=np.float32)

    for config in configs:
        info, params, output = get_param_dict(
            f"{root}/config/{config}/{config}", params, output, debug=debug)
        idx = np.where(
            (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
            * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
        )
        run["Reco"]["TotalAdjClEnergy"][idx] = np.sum(
            run["Reco"]["AdjClEnergy"][idx], axis=1
        )
        run["Reco"]["TotalAdjClMainE"][idx] = np.sum(
            run["Reco"]["AdjClMainE"][idx], axis=1
        )
        run["Reco"]["MaxAdjClEnergy"][idx] = np.max(
            run["Reco"]["AdjClEnergy"][idx], axis=1
        )
        converted_array_time = reshape_array(
            run["Reco"]["Time"][idx], len(run["Reco"]["AdjClTime"][idx][0]))

        converted_array_charge = reshape_array(
            run["Reco"]["Charge"][idx], len(run["Reco"]["AdjClCharge"][idx][0]))

        run["Reco"]["AdjCldTime"][idx] = run["Reco"]["AdjClTime"][idx] - \
            converted_array_time
        run["Reco"]["AdjClRelCharge"] = run["Reco"]["AdjClCharge"][idx] / \
            converted_array_charge
        run["Reco"]["AdjClChargePerHit"] = run["Reco"]["AdjClCharge"][idx] / \
            run["Reco"]["AdjClNHits"][idx]

    run = remove_branches(run, rm_branches, [], debug=debug)
    output += f"\tAdjCl energy computation \t-> Done!\n"
    return run, output, new_branches+new_vector_branches