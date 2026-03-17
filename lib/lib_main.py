import numpy as np

from typing import Optional
from itertools import product
from .lib_format import expand_variables
from .lib_format import remove_branches, get_param_dict

from src.utils import get_project_root

root = get_project_root()


def update_default_values(
    run: dict[str, dict],
    configs: dict[str, list[str]],
    params: Optional[dict] = None,
    trees: Optional[list[str]] = None,
    output: str = "",
    debug=False,
):
    """
    Update the default values of non set variables in the run.
    """

    if output is None:
        output = ""

    output += f"\t[magenta][LOG][/magenta] Updating default values...\n"

    new_branches = []
    if trees is not None:
        new_branches = [f"ReferenceVariable{var}" for var in ["P", "X", "Y", "Z"]]
        for branch in new_branches:
            run["Truth"][branch] = np.full_like(
                run["Truth"]["SignalParticleP"], -1e6, dtype=np.float32
            )
            run["Reco"][branch] = np.full_like(
                run["Reco"]["SignalParticleP"], -1e6, dtype=np.float32
            )

    for config in configs:
        info, this_params, output = get_param_dict(
            f"{root}/config/{config}/{config}", params, output, debug=debug
        )
        for name in configs[config]:
            idx = np.where(
                (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
                * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
                * (np.asarray(run["Reco"]["Name"]) == name)
            )
            jdx = np.where(
                (np.asarray(run["Truth"]["Geometry"]) == info["GEOMETRY"])
                * (np.asarray(run["Truth"]["Version"]) == info["VERSION"])
                * (np.asarray(run["Truth"]["Name"]) == name)
            )
            if trees is not None:
                ref_variable = "Main" if "gamma" in name else "MainParent"
                backup_variable = "Main"
                for var in ["P", "X", "Y", "Z"]:
                    run["Truth"][f"ReferenceVariable{var}"][jdx] = run["Reco"][
                        f"{ref_variable}{var}"
                    ][idx][run["Truth"]["RecoIndex"][jdx]]
                    # Substitute the values that are 0 or -1e6 with the backup variable values
                    run["Truth"][f"ReferenceVariable{var}"][jdx] = np.where(
                        (run["Truth"][f"ReferenceVariable{var}"][jdx] == 0)
                        + (run["Truth"][f"ReferenceVariable{var}"][jdx] == -1e6),
                        run["Reco"][f"{backup_variable}{var}"][idx][
                            run["Truth"]["RecoIndex"][jdx]
                        ],
                        run["Truth"][f"ReferenceVariable{var}"][jdx],
                    )
                    # Prepare the Reco reference variable with the same values as the backup variable
                    run["Reco"][f"ReferenceVariable{var}"][idx] = run["Reco"][
                        f"{ref_variable}{var}"
                    ][idx]
                    run["Reco"][f"ReferenceVariable{var}"][idx] = np.where(
                        (run["Reco"][f"ReferenceVariable{var}"][idx] == 0)
                        + (run["Reco"][f"ReferenceVariable{var}"][idx] == -1e6),
                        run["Reco"][f"{backup_variable}{var}"][idx],
                        run["Reco"][f"ReferenceVariable{var}"][idx],
                    )

                for tree, variable, (kdx, var) in product(
                    trees, ["SignalParticle"], enumerate(["P", "X", "Y", "Z"])
                ):
                    if tree == "Truth" and name.split("_")[0] in [
                        "gamma",
                        "electron",
                        "neutron",
                        "proton",
                        "alpha",
                    ]:
                        if kdx == 0:
                            output += f"\t\t[cyan][INFO][/cyan] Update Default {tree} {variable} Values by Setting {100*((run[tree][f'{variable}{var}'][jdx] == 0).sum() + (run[tree][f'{variable}{var}'][jdx] == -1e6).sum())/ len(run[tree][f'{variable}{var}'][jdx]):.2f}% to ReferenceVariable\n"
                        run[tree][f"{variable}{var}"][jdx] = np.where(
                            (run[tree][f"{variable}{var}"][jdx] == 0)
                            + (run[tree][f"{variable}{var}"][jdx] == -1e6),
                            run[tree][f"ReferenceVariable{var}"][jdx],
                            run[tree][f"{variable}{var}"][jdx],
                        )
                    if tree == "Reco" and name.split("_")[0] in [
                        "gamma",
                        "electron",
                        "neutron",
                        "proton",
                        "alpha",
                    ]:
                        if kdx == 0:
                            output += f"\t\t[cyan][INFO][/cyan] Update Default {tree} {variable} Values by Setting {100*((run[tree][f'{variable}{var}'][idx] == 0).sum() + (run[tree][f'{variable}{var}'][idx] == -1e6).sum())/ len(run[tree][f'{variable}{var}'][idx]):.2f}% to ReferenceVariable\n"
                        run[tree][f"{variable}{var}"][idx] = np.where(
                            (run[tree][f"{variable}{var}"][idx] == 0)
                            + (run[tree][f"{variable}{var}"][idx] == -1e6),
                            run[tree][f"ReferenceVariable{var}"][idx],
                            run[tree][f"{variable}{var}"][idx],
                        )

            # Substitute the values that are -1e6 with the central x coordinate point
            output += f"\t\t[cyan][INFO][/cyan] Update Default RecoX Values by Setting {100 * np.sum(run['Reco']['RecoX'][idx] == -1e6) / len(run['Reco']['RecoX'][idx]):.2f}% to Default ([{info['DETECTOR_MIN_X']}, {info['DETECTOR_MAX_X']}])\n"
            run["Reco"]["RecoX"][idx] = np.where(
                (run["Reco"]["RecoX"][idx] == -1e6)
                * (run["Reco"]["SignalParticleX"][idx] >= 0),
                info["DETECTOR_MAX_X"],
                run["Reco"]["RecoX"][idx],
            )
            run["Reco"]["RecoX"][idx] = np.where(
                (run["Reco"]["RecoX"][idx] == -1e6)
                * (run["Reco"]["SignalParticleX"][idx] < 0),
                info["DETECTOR_MIN_X"],
                run["Reco"]["RecoX"][idx],
            )

            # Print percentage of values that are out of the detector range
            output += f"\t\t[cyan][INFO][/cyan] Update Default RecoY Values by Setting {100 * np.sum((run['Reco']['RecoY'][idx] < info['DETECTOR_MIN_Y']) + (run['Reco']['RecoY'][idx] > info['DETECTOR_MAX_Y'])) / len(run['Reco']['RecoY'][idx]):.2f}% to Default ([{info['DETECTOR_MIN_Y']}, {info['DETECTOR_MAX_Y']}])\n"
            run["Reco"]["RecoY"][idx] = np.where(
                (run["Reco"]["RecoY"][idx] < info["DETECTOR_MIN_Y"]),
                info["DETECTOR_MIN_Y"],
                run["Reco"]["RecoY"][idx],
            )
            run["Reco"]["RecoY"][idx] = np.where(
                (run["Reco"]["RecoY"][idx] > info["DETECTOR_MAX_Y"]),
                info["DETECTOR_MAX_Y"],
                run["Reco"]["RecoY"][idx],
            )
    output += f"\tUpdate default values \t\t-> Done!\n\n"

    return run, output, new_branches


def split_vector_branches(
    run: dict[str, dict],
    configs: dict[str, list[str]],
    params: Optional[dict] = None,
    rm_branches: bool = False,
    output: str = "",
    debug=False,
):
    if output is None:
        output = ""
    required_branches = [
        "MainVertex",
        "EndVertex",
        "MainParentVertex",
    ]
    for branch in required_branches:
        if branch not in run["Reco"]:
            if debug:
                output += f"\t[red][ERROR] Missing {branch} branch in Reco tree: Main variables computation skipped[/red]\n"
            return run, output, []

    new_braches = [
        f"{branch.split('Vertex')[0]}{coord}"
        for branch in required_branches
        for coord in ["X", "Y", "Z"]
    ]
    for branch in required_branches:
        x, y, z = expand_variables(run["Reco"][branch])
        main_branch = branch.split("Vertex")[0]
        run["Reco"][f"{main_branch}X"] = x
        run["Reco"][f"{main_branch}Y"] = y
        run["Reco"][f"{main_branch}Z"] = z

    output += f"\tVector variable splitting \t-> Done!\n\n"
    if rm_branches:
        run = remove_branches(run, rm_branches, required_branches, "Reco")
        output += f"\tVector variable splitting \t-> Original branches removed!\n\n"
    return run, output, new_braches


def compute_main_variables(
    run: dict[str, dict],
    configs: dict[str, list[str]],
    params: Optional[dict] = None,
    rm_branches: bool = False,
    output: str = "",
    debug=False,
):
    """
    Compute the main (backtracked) variables of main particle correspondong to the cluster in the run.
    """
    if output is None:
        output = ""

    new_braches = [
        "ErrorX",
        "ErrorY",
        "ErrorZ",
        "2DError",
        "3DError",
    ]

    run["Reco"]["ErrorY"] = run["Reco"]["MainY"] - run["Reco"]["RecoY"]
    run["Reco"]["ErrorZ"] = run["Reco"]["MainZ"] - run["Reco"]["RecoZ"]
    run["Reco"]["2DError"] = np.sqrt(
        np.power(run["Reco"]["ErrorZ"], 2) + np.power(run["Reco"]["ErrorY"], 2)
    )
    try:
        run["Reco"]["ErrorX"] = run["Reco"]["MainX"] - run["Reco"]["RecoX"]
        run["Reco"]["3DError"] = np.sqrt(
            np.power(run["Reco"]["ErrorZ"], 2)
            + np.power(run["Reco"]["ErrorY"], 2)
            + np.power(run["Reco"]["ErrorX"], 2)
        )
    except KeyError:
        if debug:
            output += f"\t[red][ERROR] Missing RecoX: ErrorX and 3DError branches not computed[/red]\n"
        pass

    output += f"\tMain variables computation \t-> Done!\n\n"
    return run, output, new_braches
