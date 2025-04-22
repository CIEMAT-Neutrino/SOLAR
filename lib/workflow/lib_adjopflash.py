import numba
import numpy as np

from typing import Optional
from itertools import product
from lib.workflow.functions import remove_branches, get_param_dict, reshape_array

from src.utils import get_project_root

root = get_project_root()


def compute_adjopflash_basics(
    run,
    configs,
    params: Optional[dict] = None,
    rm_branches=False,
    output: Optional[str] = None,
    debug=False,
):
    """
    Compute basic variables for the adjacent clusters

    Args:
        run: dictionary containing the TTree
        configs: dictionary containing the path to the configuration files for each geoemtry
        params: dictionary containing the parameters for the reco functions
        debug: print debug information
    """

    @numba.njit
    def count_occurrences(arr, length):
        """
        Count the occurrences of each element in the array.

        Args:
            arr: array containing the elements
            length: length of the array
        """
        return [np.sum(arr == i) for i in range(length)]

    # New branches
    new_int_branches = ["AdjOpFlashNum", "AdjOpFlashSameGenNum"]

    new_branches = [
        "TotalAdjOpFlashPE",
        "MaxAdjOpFlashPE",
        "MeanAdjOpFlashPE",
        "MeanAdjOpFlashR",
        "MeanAdjOpFlashTime",
        "TotalAdjOpFlashSameGenPE",
        "MaxAdjOpFlashSameGenPE",
        "MeanAdjOpFlashSameGenPE",
    ]

    radial_limits = np.arange(40, 200, 40)
    new_radial_int_branches = ["TotalAdjOpFlashNum", "TotalAdjOpFlashSameGenNum"]
    new_radial_float_branches = ["TotalAdjOpFlashPE", "TotalAdjOpFlashSameGenPE"]

    for branch in new_int_branches:
        run["Reco"][branch] = np.zeros(len(run["Reco"]["Event"]), dtype=int)
    for branch in new_branches:
        run["Reco"][branch] = np.zeros(len(run["Reco"]["Event"]), dtype=np.float32)

    for limit, branch in product(radial_limits, new_radial_int_branches):
        run["Reco"][f"{branch}{limit}"] = np.zeros(
            (len(run["Reco"]["Event"])), dtype=int
        )

    for limit, branch in product(radial_limits, new_radial_float_branches):
        run["Reco"][f"{branch}{limit}"] = np.zeros(
            (len(run["Reco"]["Event"])), dtype=np.float32
        )

    run["Reco"]["AdjOpFlashGenNum"] = np.apply_along_axis(
        count_occurrences,
        arr=(run["Reco"]["AdjOpFlashPur"] > 0).astype(int),
        length=2,
        axis=1,
    )

    run["Reco"]["AdjOpFlashNum"] = np.sum(run["Reco"]["AdjOpFlashPE"] > 0, axis=1)
    # Compute the number of adjacent clusters per radial limit
    for limit in radial_limits:
        run["Reco"][f"{new_radial_int_branches[0]}{limit}"] = np.sum(
            (np.absolute(run["Reco"]["AdjOpFlashR"]) < limit)
            * (np.absolute(run["Reco"]["AdjOpFlashR"]) > 0),
            axis=1,
        )

    for config in configs:
        info, params, output = get_param_dict(
            f"{root}/config/{config}/{config}", params, output, debug=debug
        )
        idx = np.where(
            (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
            * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
        )
        run["Reco"]["TotalAdjOpFlashPE"][idx] = np.sum(
            run["Reco"]["AdjOpFlashPE"][idx],
            axis=1,
            where=run["Reco"]["AdjOpFlashPE"][idx] > 0,
        )
        run["Reco"]["MaxAdjOpFlashPE"][idx] = np.max(
            run["Reco"]["AdjOpFlashPE"][idx],
            axis=1,
        )
        run["Reco"]["MeanAdjOpFlashPE"][idx] = np.mean(
            run["Reco"]["AdjOpFlashPE"][idx],
            axis=1,
            where=run["Reco"]["AdjOpFlashPE"][idx] > 0,
        )
        run["Reco"]["MeanAdjOpFlashR"][idx] = np.mean(
            np.absolute(run["Reco"]["AdjOpFlashR"][idx]),
            axis=1,
            where=run["Reco"]["AdjOpFlashPE"][idx] > 0,
        )
        run["Reco"]["MeanAdjOpFlashTime"][idx] = np.mean(
            run["Reco"]["AdjOpFlashTime"][idx],
            axis=1,
            where=run["Reco"]["AdjOpFlashTime"][idx] > 0,
        )

        converted_array = reshape_array(
            run["Reco"]["Generator"][idx], len(run["Reco"]["AdjOpFlashPur"][idx][0])
        )

        gen_idx = converted_array == (run["Reco"]["AdjOpFlashPur"][idx] > 0).astype(int)

        run["Reco"]["AdjOpFlashSameGenNum"][idx] = np.sum(gen_idx, axis=1)
        for limit in radial_limits:
            run["Reco"][f"{new_radial_int_branches[1]}{limit}"][idx] = np.sum(
                (np.absolute(run["Reco"]["AdjOpFlashR"][idx]) < limit)
                * (np.absolute(run["Reco"]["AdjOpFlashR"][idx]) > 0)
                * gen_idx,
                axis=1,
            )

        run["Reco"]["TotalAdjOpFlashSameGenPE"][idx] = np.sum(
            run["Reco"]["AdjOpFlashPE"][idx] * gen_idx, axis=1
        )
        for limit in radial_limits:
            rad_idx = (np.absolute(run["Reco"]["AdjOpFlashR"][idx]) < limit) * (
                np.absolute(run["Reco"]["AdjOpFlashR"][idx]) > 0
            )

            run["Reco"][f"{new_radial_float_branches[1]}{limit}"][idx] = np.sum(
                run["Reco"]["AdjOpFlashPE"][idx] * gen_idx * rad_idx, axis=1
            )
            run["Reco"][f"{new_radial_float_branches[0]}{limit}"][idx] = np.sum(
                run["Reco"]["AdjOpFlashPE"][idx] * rad_idx, axis=1
            )

        run["Reco"]["MaxAdjOpFlashSameGenPE"][idx] = np.max(
            run["Reco"]["AdjOpFlashPE"][idx] * gen_idx, axis=1
        )
        run["Reco"]["MeanAdjOpFlashSameGenPE"][idx] = np.mean(
            run["Reco"]["AdjOpFlashPE"][idx] * gen_idx, axis=1
        )

    run = remove_branches(run, rm_branches, [], debug=debug)
    output += f"\tAdjOpFlash basic computation \t-> Done!\n"
    return run, output, new_branches
