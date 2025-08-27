import json
import pickle
import numba
import yaml
import numpy as np

from typing import Optional
from rich import print as rprint
from itertools import product
from particle import Particle

from lib.df_functions import npy2df
from lib.fit_functions import calibration_func
from lib.workflow.lib_default import get_default_info

from src.utils import get_project_root

root = get_project_root()


def get_param_dict(
    config_file: dict,
    in_params: Optional[dict] = None,
    output: Optional[str] = None,
    debug: bool = False,
):
    """
    Get the parameters for the reco workflow from the input files.
    """
    info = json.load(open(f"{config_file}_config.json", "r"))
    params = json.load(open(f"{config_file}_params.json", "r"))

    if output is None:
        output = ""

    if in_params is None:
        return info, params, output

    for param in params.keys():
        try:
            if in_params[param] is None:
                if debug:
                    info_string = f"\t[cyan]***[INFO] Applying {param}: {params[param]} from the config file[/cyan]\n"
                    # Check if info string is already in the output
                    if info_string not in output:
                        output += info_string
            
            elif in_params[param] is not None:
                params[param] = in_params[param]
                warning_string = f"\t[yellow]***[WARNING][/yellow] Applying {param}: {in_params[param]} from the input dictionary\n"
                # Check if warning string is already in the output
                if warning_string not in output:
                    output += warning_string
        
        except KeyError:
            pass

    return info, params, output


def remove_branches(run, remove, branches, tree: str = "Reco", debug=False):
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


def reshape_array(array: np.array, length: int, debug: bool = False):
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
