import numpy as np

from typing import Optional
from .functions import expand_variables
from lib.workflow.functions import remove_branches, get_param_dict

from src.utils import get_project_root

root = get_project_root()


def update_default_values(
    run: dict[dict],
    configs: dict[str, list[str]],
    params: Optional[dict] = None,
    rm_branches: bool = False,
    output: Optional[str] = None,
    debug=False,
):
    """
    Update the default values of non set variables in the run.
    """
    if output is None:
        output = ""

    for config in configs:
        info, params, output = get_param_dict(
            f"{root}/config/{config}/{config}", params, output, debug=debug
        )
        idx = np.where(
            (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
            * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
        )
        # Check if "RecoX" is in run["Reco"] to avoid KeyError
        if "RecoX" in run["Reco"]:
            # Substitute the values that are -1e6 with the central x coordinate point
            run["Reco"]["RecoX"][idx] = np.where(
                (run["Reco"]["RecoX"][idx] == -1e6)
                * (run["Reco"]["SignalParticleX"][idx] > 0),
                (
                    info["DETECTOR_SIZE_X"] / 4
                    if info["GEOMETRY"] == "hd"
                    else info["DETECTOR_SIZE_X"] / 2
                ),
                run["Reco"]["RecoX"][idx],
            )
            run["Reco"]["RecoX"][idx] = np.where(
                (run["Reco"]["RecoX"][idx] == -1e6)
                * (run["Reco"]["SignalParticleX"][idx] < 0),
                (
                    -info["DETECTOR_SIZE_X"] / 4
                    if info["GEOMETRY"] == "hd"
                    else -info["DETECTOR_SIZE_X"] / 2
                ),
                run["Reco"]["RecoX"][idx],
            )
    output += f"\tUpdate Default Values (RecoX) \t-> Done!\n"

    return run, output, []


def compute_main_variables(
    run: dict[dict],
    configs: dict[str, list[str]],
    params: Optional[dict] = None,
    rm_branches: bool = False,
    output: Optional[str] = None,
    debug=False,
):
    """
    Compute the main (backtracked) variables of main particle correspondong to the cluster in the run.
    """
    if output is None:
        output = ""
    required_branches = [
        "MainVertex",
        "EndVertex" "MainParentVertex",
        "RecoX",
        "RecoY",
        "RecoZ",
    ]

    new_braches = [
        "MainX",
        "MainY",
        "MainZ",
        "EndX",
        "EndY",
        "EndZ",
        "MainParentX",
        "MainParentY",
        "MainParentZ",
        "ErrorX",
        "ErrorY",
        "ErrorZ",
        "2DError",
        "3DError",
    ]

    for branch in ["MainVertex", "EndVertex", "MainParentVertex"]:
        x, y, z = expand_variables(run["Reco"][branch])
        main_branch = branch.split("Vertex")[0]
        run["Reco"][f"{main_branch}X"] = x
        run["Reco"][f"{main_branch}Y"] = y
        run["Reco"][f"{main_branch}Z"] = z

    run["Reco"]["ErrorY"] = abs(run["Reco"]["MainY"] - run["Reco"]["RecoY"])
    run["Reco"]["ErrorZ"] = abs(run["Reco"]["MainZ"] - run["Reco"]["RecoZ"])
    run["Reco"]["2DError"] = np.sqrt(
        np.power(run["Reco"]["ErrorZ"], 2) + np.power(run["Reco"]["ErrorY"], 2)
    )
    try:
        run["Reco"]["ErrorX"] = abs(
            abs(run["Reco"]["MainX"]) - abs(run["Reco"]["RecoX"])
        )
        run["Reco"]["3DError"] = np.sqrt(
            np.power(run["Reco"]["ErrorZ"], 2)
            + np.power(run["Reco"]["ErrorY"], 2)
            + np.power(run["Reco"]["ErrorX"], 2)
        )
    except KeyError:
        if debug:
            output += f"\t[red][ERROR] Missing RecoX: ErrorX and 3DError branches not computed[/red]\n"
        pass

    output += f"\tMain variables computation \t-> Done!\n"
    return run, output, new_braches
