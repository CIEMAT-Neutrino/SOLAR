import numpy as np

from typing import Optional
from lib.workflow.functions import remove_branches, get_param_dict

from src.utils import get_project_root

root = get_project_root()


def compute_true_drift(
    run,
    configs,
    params: Optional[dict] = None,
    rm_branches: bool = False,
    output: Optional[str] = None,
    debug=False,
):
    """
    Compute the true drift time of the events in the run.
    """
    # New branches
    new_branches = [
        "TruthX",
        "TruthDriftTime",
    ]
    new_vector_branches = ["TruthAdjClX", "TruthAdjClDriftTime"]
    for branch in new_branches:
        run["Reco"][branch] = np.zeros(len(run["Reco"]["Event"]), dtype=np.float32)
    for branch in new_vector_branches:
        run["Reco"][branch] = np.zeros(
            (len(run["Reco"]["Event"]), len(run["Reco"]["AdjClTime"][0])),
            dtype=np.float32,
        )

    for config in configs:
        info, params, output = get_param_dict(
            f"{root}/config/{config}/{config}", params, output, debug=debug
        )
        idx = np.where(
            (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
            * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
        )
        if info["GEOMETRY"] == "hd":
            run["Reco"]["TruthX"][idx] = abs(run["Reco"]["MainVertex"][idx, 0])
            run["Reco"]["TruthDriftTime"][idx] = (
                run["Reco"]["TruthX"][idx]
                * info["TIMEWINDOW"]
                * 1e6  # convert to us
                / info["DETECTOR_MAX_X"]
            )

            run["Reco"]["TruthAdjClX"][idx] = abs(run["Reco"]["AdjClMainX"][idx])
            run["Reco"]["TruthAdjClDriftTime"][idx] = (
                run["Reco"]["TruthAdjClX"][idx]
                * info["TIMEWINDOW"]
                * 1e6  # convert to us
                / info["DETECTOR_MAX_X"]
            )

        elif info["GEOMETRY"] == "vd":
            run["Reco"]["TruthX"][idx] = run["Reco"]["MainVertex"][idx, 0]
            run["Reco"]["TruthDriftTime"][idx] = (
                (info["DETECTOR_SIZE_X"] / 2 - run["Reco"]["TruthX"][idx])
                * info["TIMEWINDOW"]
                * 1e6  # convert to us
                / info["DETECTOR_SIZE_X"]
            )

            run["Reco"]["TruthAdjClX"][idx] = run["Reco"]["AdjClMainX"][idx]
            run["Reco"]["TruthAdjClDriftTime"][idx] = (
                (info["DETECTOR_SIZE_X"] / 2 - run["Reco"]["TruthAdjClX"][idx])
                * info["TIMEWINDOW"]
                * 1e6  # convert to us
                / info["DETECTOR_SIZE_X"]
            )

        # Select all values bigger than 1e6 or smaller than 0 and set them to 0
        run["Reco"]["TruthDriftTime"] = np.where(
            (run["Reco"]["TruthDriftTime"] > info["TIMEWINDOW"] * 1e6)
            | (run["Reco"]["TruthDriftTime"] < 0),
            info["TIMEWINDOW"] * 5e5,
            run["Reco"]["TruthDriftTime"],
        )
        run["Reco"]["TruthAdjClDriftTime"] = np.where(
            (run["Reco"]["TruthAdjClDriftTime"] > info["TIMEWINDOW"] * 1e6)
            | (run["Reco"]["TruthAdjClDriftTime"] < 0),
            info["TIMEWINDOW"] * 5e5,
            run["Reco"]["TruthAdjClDriftTime"],
        )
    output += f"\tTrue drift time computation \t-> Done!\n"
    return run, output, new_branches + new_vector_branches
