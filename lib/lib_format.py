import json
import pickle
import numba
import yaml
import numpy as np

from typing import Optional
from rich import print as rprint
from itertools import product
from particle import Particle

from .lib_df import npy2df
from .lib_fit import calibration_func
from .lib_default import get_default_info

from src.utils import get_project_root

root = get_project_root()


def get_param_dict(
    config_file: str,
    in_params: Optional[dict] = None,
    output: str = "",
    debug: bool = False,
):
    """
    Get the parameters for the reco workflow from the input files.
    """
    info = json.load(open(f"{config_file}_config.json", "r"))
    params = json.load(open(f"{config_file}_params.json", "r"))
    out_params = params.copy()

    if in_params is None:
        return info, out_params, output

    for param in params.keys():
        if param in in_params.keys():
            if in_params[param] is None:
                if debug:
                    info_string = f"\tApplying {param}: {params[param]} from the config file\n"
                    # Check if info string is already in the output
                    if info_string not in output:
                        output += info_string

            elif in_params[param] is not None:
                # params[param] = in_params[param]
                out_params[param] = in_params[param]
                warning_string = f"\tApplying {param}: {in_params[param]} from the input dictionary.\n"
                # Check if warning string is already in the output
                if warning_string not in output:
                    output += warning_string

        else:
            pass
    
    for param in in_params.keys():
        if param not in out_params.keys():
            out_params[param] = in_params[param]
            warning_string = f"\t{param}: {in_params[param]} is a new parameter in the config file.\n"
            # Check if warning string is already in the output
            if warning_string not in output:
                output += warning_string

    return info, out_params, output


def remove_branches(run: dict[str, dict], remove: bool, branches: list[str], tree: str = "Reco", debug: bool = False):
    """
    Remove branches from the run dictionary

    Args:
        run (dict): dictionary containing the TTree
        remove (bool): if True, remove the branches
        branches (list): list of branches to be removed
        tree (str): name of the TTree
        debug (bool): print debug information

    Returns:
        run (dict): dictionary containing the TTree with the new branches
    """
    if remove:
        if debug:
            print(f"-> Removing branches: {branches}")
        for branch in branches:
            run[tree].pop(branch)
    else:
        pass

    return run


def reshape_array(array: np.ndarray, length: int, debug: bool = False):
    """
    Reshape the array to the desired length.
    """
    repeated_array = np.repeat(array, length)
    return np.reshape(repeated_array, (-1, length))


@numba.njit
def expand_variables(branch):
    x = branch[:, 0]
    y = branch[:, 1]
    z = branch[:, 2]
    return x, y, z
