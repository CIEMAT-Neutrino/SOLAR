import numba
import numpy as np

from typing import Optional
from itertools import product
from lib.workflow.functions import remove_branches, get_param_dict, reshape_array

from src.utils import get_project_root

root = get_project_root()


def compute_adjopflash_basic_variable(
    run,
    idx,
    branch: str,
    variable_interlabel: str = "",
    variable_extension: str = "",
    reference: Optional[np.ndarray] = None,
    compute: list[str] = ["mean", "max", "total"],
    params: dict = None,
    debug: bool = False,
):
    """
    Compute basic variables for the adjacent clusters

    Args:
        run: dictionary containing the TTree
        idx: index of the geometry - version - selection
        variable: variable to compute the basic variables for
        reference_variable: variable to use as a reference to exclude empty entries
        params: dictionary containing the parameters for the reco functions
        debug: print debug information
    """
    variable = branch.split("AdjOpFlash")[-1]
    if "mean" in compute:
        run["Reco"][
            f"MeanAdjOpFlash{variable_interlabel}{variable}{variable_extension}"
        ][idx] = np.mean(
            run["Reco"][branch][idx],
            axis=1,
            where=reference,
        )
        run["Reco"][
            f"STDAdjOpFlash{variable_interlabel}{variable}{variable_extension}"
        ][idx] = np.std(
            run["Reco"][branch][idx],
            axis=1,
            where=reference,
        )

    if "max" in compute:
        run["Reco"][
            f"MaxAdjOpFlash{variable_interlabel}{variable}{variable_extension}"
        ][idx] = np.max(
            run["Reco"][branch][idx],
            axis=1,
        )

    if "total" in compute:
        run["Reco"][
            f"TotalAdjOpFlash{variable_interlabel}{variable}{variable_extension}"
        ][idx] = np.sum(
            run["Reco"][branch][idx],
            axis=1,
            where=reference,
        )

    return run


def compute_adjopflash_basics(
    run,
    configs,
    params: Optional[dict] = None,
    rm_branches: bool = False,
    output: Optional[str] = None,
    debug: bool = False,
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

    int_branches = ["AdjOpFlashNum"]
    new_int_branches = []
    float_branches = [
        "AdjOpFlashPE",
        "AdjOpFlashR",
        "AdjOpFlashTime",
    ]
    new_float_branches = []

    max_planes = 0
    max_radius = 0
    for config in configs:
        info, this_params, output = get_param_dict(
            f"{root}/config/{config}/{config}", params, output, debug=debug
        )
        pds_planes = [int(x) for x in list(info["OPFLASH_PLANES"].keys())]
        # OPFLASH_PLANES is a dict with the planes for each geometry, where the key is an int and the value is a name
        if np.max(pds_planes) > max_planes:
            max_planes = np.max(pds_planes)
        if np.max(info["OPFLASH_RADIUS"]) > max_radius:
            max_radius = np.max(info["OPFLASH_RADIUS"])

    planes = np.arange(-1, max_planes + 1, 1)
    radial_limits = np.arange(20, max_radius + 20, 20)

    for branch in int_branches:
        variable = branch.split("AdjOpFlash")[-1]
        for prefix, interlabel in product(
            ["Mean", "STD", "Max", "Total"], ["", "SameGen", "Bkg"]
        ):
            run["Reco"][f"{prefix}AdjOpFlash{interlabel}{variable}"] = np.zeros(
                len(run["Reco"]["Event"]), dtype=int
            )
            new_int_branches.append(f"{prefix}AdjOpFlash{interlabel}{variable}")
            run["Reco"][branch] = np.zeros(len(run["Reco"]["Event"]), dtype=int)
            for plane in planes:
                run["Reco"][f"{prefix}AdjOpFlash{interlabel}{variable}Plane{plane}"] = (
                    np.zeros(len(run["Reco"]["Event"]), dtype=np.float32)
                )
                new_int_branches.append(
                    f"{prefix}AdjOpFlash{interlabel}{variable}Plane{plane}"
                )
            for limit in radial_limits:
                run["Reco"][
                    f"{prefix}AdjOpFlash{interlabel}{variable}Radius{limit}"
                ] = np.zeros((len(run["Reco"]["Event"])), dtype=int)
                new_int_branches.append(
                    f"{prefix}AdjOpFlash{interlabel}{variable}Radius{limit}"
                )

    for branch in float_branches:
        variable = branch.split("AdjOpFlash")[-1]
        for prefix, interlabel in product(
            ["Mean", "STD", "Max", "Total"], ["", "SameGen", "Bkg"]
        ):
            run["Reco"][f"{prefix}AdjOpFlash{interlabel}{variable}"] = np.zeros(
                len(run["Reco"]["Event"]), dtype=np.float32
            )
            new_float_branches.append(f"{prefix}AdjOpFlash{interlabel}{variable}")
            for plane in planes:
                run["Reco"][f"{prefix}AdjOpFlash{interlabel}{variable}Plane{plane}"] = (
                    np.zeros(len(run["Reco"]["Event"]), dtype=np.float32)
                )
                new_float_branches.append(
                    f"{prefix}AdjOpFlash{interlabel}{variable}Plane{plane}"
                )
            for limit in radial_limits:
                run["Reco"][
                    f"{prefix}AdjOpFlash{interlabel}{variable}Radius{limit}"
                ] = np.zeros((len(run["Reco"]["Event"])), dtype=np.float32)
                new_float_branches.append(
                    f"{prefix}AdjOpFlash{interlabel}{variable}Radius{limit}"
                )

    run["Reco"]["TotalAdjOpFlashNum"] = np.sum(run["Reco"]["AdjOpFlashPE"] > 0, axis=1)
    # Compute the number of adjacent clusters per plane
    for plane in planes:
        run["Reco"][f"TotalAdjOpFlashNumPlane{plane}"] = np.sum(
            (run["Reco"]["AdjOpFlashPE"] > 0)
            * (run["Reco"]["AdjOpFlashPlane"] == plane),
            axis=1,
        )

    for limit in radial_limits:
        run["Reco"][f"TotalAdjOpFlashNumRadius{limit}"] = np.sum(
            (np.absolute(run["Reco"]["AdjOpFlashR"]) < limit)
            * (np.absolute(run["Reco"]["AdjOpFlashR"]) > 0),
            axis=1,
        )

    for config in configs:
        info, this_params, output = get_param_dict(
            f"{root}/config/{config}/{config}", params, output, debug=debug
        )

        idx = np.where(
            (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
            * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
        )

        run["Reco"]["TotalAdjOpFlashSameGenNum"][idx] = np.sum(
            (run["Reco"]["AdjOpFlashPur"][idx] > 0), axis=1
        )
        run["Reco"]["TotalAdjOpFlashBkgNum"][idx] = np.sum(
            (run["Reco"]["AdjOpFlashPur"][idx] == 0)
            * (run["Reco"]["AdjOpFlashPE"][idx] > 0),
            axis=1,
        )
        for plane in planes:
            run["Reco"][f"TotalAdjOpFlashSameGenNumPlane{plane}"][idx] = np.sum(
                (run["Reco"]["AdjOpFlashPlane"][idx] == plane)
                * (run["Reco"]["AdjOpFlashPur"][idx] > 0)
                * (run["Reco"]["AdjOpFlashPE"][idx] > 0),
                axis=1,
            )

            run["Reco"][f"TotalAdjOpFlashBkgNumPlane{plane}"][idx] = np.sum(
                (run["Reco"]["AdjOpFlashPlane"][idx] == plane)
                * (run["Reco"]["AdjOpFlashPur"][idx] == 0)
                * (run["Reco"]["AdjOpFlashPE"][idx] > 0),
                axis=1,
            )

        for limit in radial_limits:
            run["Reco"][f"TotalAdjOpFlashSameGenNumRadius{limit}"][idx] = np.sum(
                (np.absolute(run["Reco"]["AdjOpFlashR"][idx]) < limit)
                * (np.absolute(run["Reco"]["AdjOpFlashR"][idx]) > 0)
                * (run["Reco"]["AdjOpFlashPur"][idx] > 0),
                axis=1,
            )

            run["Reco"][f"TotalAdjOpFlashBkgNumRadius{limit}"][idx] = np.sum(
                (np.absolute(run["Reco"]["AdjOpFlashR"][idx]) < limit)
                * (np.absolute(run["Reco"]["AdjOpFlashR"][idx]) > 0)
                * (run["Reco"]["AdjOpFlashPur"][idx] == 0),
                axis=1,
            )

        run["Reco"]["TotalAdjOpFlashSameGenPE"][idx] = np.sum(
            run["Reco"]["AdjOpFlashPE"][idx] * (run["Reco"]["AdjOpFlashPur"][idx] > 0),
            axis=1,
        )
        run["Reco"]["TotalAdjOpFlashBkgPE"][idx] = np.sum(
            run["Reco"]["AdjOpFlashPE"][idx] * (run["Reco"]["AdjOpFlashPur"][idx] == 0),
            axis=1,
        )

        for signal_idx, signal_label in zip(
            [
                (run["Reco"]["AdjOpFlashPur"][idx] >= 0),
                (run["Reco"]["AdjOpFlashPur"][idx] > 0),
                (run["Reco"]["AdjOpFlashPur"][idx] == 0),
            ],
            ["", "SameGen", "Bkg"],
        ):
            for plane in [None] + pds_planes:
                for branch, compute in zip(
                    ["AdjOpFlashPE", "AdjOpFlashR", "AdjOpFlashTime"],
                    [["mean", "max", "total"], ["mean"], ["mean"]],
                ):
                    if plane == None:
                        variable_extension = ""
                        reference = (run["Reco"]["AdjOpFlashPE"] > 0) * signal_idx

                    else:
                        variable_extension = f"Plane{plane}"
                        reference = (
                            (run["Reco"]["AdjOpFlashPE"] > 0)
                            * (run["Reco"]["AdjOpFlashPlane"] == plane)
                            * signal_idx
                        )

                    run = compute_adjopflash_basic_variable(
                        run,
                        idx,
                        branch=branch,
                        variable_interlabel=signal_label,
                        variable_extension=variable_extension,
                        reference=reference,
                        compute=compute,
                        params=this_params,
                        debug=debug,
                    )

            for limit in [None] + info["OPFLASH_RADIUS"]:
                for branch, compute in zip(
                    ["AdjOpFlashPE", "AdjOpFlashR", "AdjOpFlashTime"],
                    [["mean", "max", "total"], ["mean"], ["mean"]],
                ):
                    if limit == None:
                        variable_extension = ""
                        reference = (
                            (run["Reco"]["AdjOpFlashPE"] > 0)
                            * (np.absolute(run["Reco"]["AdjOpFlashPlane"]) >= 0)
                            * signal_idx
                        )

                    else:
                        variable_extension = f"Radius{limit}"
                        reference = (
                            (run["Reco"]["AdjOpFlashPE"] > 0)
                            * (run["Reco"]["AdjOpFlashPlane"] >= 0)
                            * (np.absolute(run["Reco"]["AdjOpFlashR"]) < limit)
                            * signal_idx
                        )

                    run = compute_adjopflash_basic_variable(
                        run,
                        idx,
                        branch=branch,
                        variable_interlabel=signal_label,
                        variable_extension=variable_extension,
                        reference=reference,
                        compute=compute,
                        params=this_params,
                        debug=debug,
                    )

    new_branches = []
    branches_to_remove = []
    for branch in new_int_branches + new_float_branches:
        if np.sum(run["Reco"][branch]) == 0:
            branches_to_remove.append(branch)
        else:
            new_branches.append(branch)

    run = remove_branches(run, rm_branches, branches_to_remove, debug=debug)
    output += f"\tAdjOpFlash basic computation \t-> Done!\n"
    return run, output, new_branches
