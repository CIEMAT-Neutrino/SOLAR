import numpy as np

from typing import Optional
from .functions import expand_variables

def compute_main_variables(run: dict[dict], configs: dict[str, list[str]], params: Optional[dict] = None, rm_branches: bool = False, output: Optional[str] = None, debug=False):
    """
    Compute the main (backtracked) variables of main particle correspondong to the cluster in the run.
    """
    if output is None:
        output = ""
    required_branches = ["MainVertex",
                         "MainParentVertex", "RecoX", "RecoY", "RecoZ"]

    new_braches = ["MainX", "MainY", "MainZ", "MainParentX", "MainParentY",
                   "MainParentZ", "ErrorX", "ErrorY", "ErrorZ", "2DError", "3DError"]

    for branch in ["MainVertex", "MainParentVertex"]:
        x, y, z = expand_variables(run["Reco"][branch])
        main_branch = branch.split("Vertex")[0]
        run["Reco"][f"{main_branch}X"] = x
        run["Reco"][f"{main_branch}Y"] = y
        run["Reco"][f"{main_branch}Z"] = z

    run["Reco"]["ErrorY"] = abs(run["Reco"]["MainY"] - run["Reco"]["RecoY"])
    run["Reco"]["ErrorZ"] = abs(run["Reco"]["MainZ"] - run["Reco"]["RecoZ"])
    run["Reco"]["2DError"] = np.sqrt(
        np.power(run["Reco"]["ErrorZ"], 2) + np.power(run["Reco"]["ErrorY"], 2))
    try:
        run["Reco"]["ErrorX"] = abs(
            abs(run["Reco"]["MainX"]) - abs(run["Reco"]["RecoX"]))
        run["Reco"]["3DError"] = np.sqrt(
            np.power(run["Reco"]["ErrorZ"], 2) + np.power(run["Reco"]["ErrorY"], 2) + np.power(run["Reco"]["ErrorX"], 2))
    except KeyError:
        if debug:
            output += f"\t[red][ERROR] Missing RecoX: ErrorX and 3DError branches not computed[/red]\n"
        pass

    output += f"\tMain variables computation \t-> Done!\n"
    return run, output, new_braches