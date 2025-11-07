import numpy as np

from typing import Optional
from rich import print as rprint
from .functions import reshape_array, get_param_dict, remove_branches

from src.utils import get_project_root

root = get_project_root()


def compute_opflash_basic(
    run,
    configs,
    params: Optional[dict] = None,
    trees: Optional[list[str]] = None,
    rm_branches: bool = False,
    output: Optional[str] = None,
    debug=False,
):
    """
    Compute the basic OpFlash variables for the events in the run.
    """
    if output is None:
        output = ""
    # New branches
    for tree in trees:
        new_int_branches = ["OpFlashNum", "OpFlashSignalNum", "OpFlashBkgNum"]
        new_float_branches = [
            "OpFlashR",
            "OpFlashdT",
            "OpFlashErrorX",
            "OpFlashErrorY",
            "OpFlashErrorZ",
        ]
        new_bool_branches = ["OpFlashSignal"]
        if tree == "Truth":
            prefix = ""
            sufix = ""
        elif tree == "Reco":
            prefix = "Adj"
            sufix = "Reco"
            new_int_branches = [prefix + branch for branch in new_int_branches]
            new_float_branches = [prefix + branch for branch in new_float_branches]
            new_bool_branches = [prefix + branch for branch in new_bool_branches]
        else:
            rprint(f"[red][ERROR] Invalid tree {tree} in compute_opflash_basic[/red]")

        for branch_list, branch_type in zip(
            [new_float_branches, new_bool_branches], [np.float32, np.bool]
        ):
            for branch in branch_list:
                run[tree][branch] = np.zeros(
                    (len(run[tree]["Event"]), len(run[tree][f"{prefix}OpFlashPE"][0])),
                    dtype=branch_type,
                )

        for branch in new_int_branches:
            run[tree][branch] = np.zeros(len(run[tree]["Event"]), dtype=int)

        for config in configs:
            info, this_params, output = get_param_dict(
                f"{root}/config/{config}/{config}", params, output, debug=debug
            )
            idx = np.where(
                (np.asarray(run[tree]["Geometry"]) == info["GEOMETRY"])
                * (np.asarray(run[tree]["Version"]) == info["VERSION"])
            )
            run[tree][f"{prefix}OpFlashSignal"][idx] = (
                run[tree][f"{prefix}OpFlashPur"][idx] > 0
            )

            run[tree][f"{prefix}OpFlashNum"][idx] = np.sum(
                run[tree][f"{prefix}OpFlashPE"][idx] > 0, axis=1
            )

            run[tree][f"{prefix}OpFlashSignalNum"][idx] = np.sum(
                (run[tree][f"{prefix}OpFlashPE"][idx] > 0)
                * (run[tree][f"{prefix}OpFlashSignal"][idx] == True),
                axis=1,
            )

            run[tree][f"{prefix}OpFlashBkgNum"][idx] = np.sum(
                (run[tree][f"{prefix}OpFlashPE"][idx] > 0)
                * (run[tree][f"{prefix}OpFlashSignal"][idx] == False),
                axis=1,
            )
            run[tree][f"{prefix}OpFlashdT"][idx] = run[tree][f"{prefix}OpFlashTime"][
                idx
            ] - reshape_array(
                run[tree]["SignalParticleTime"][idx],
                len(run[tree][f"{prefix}OpFlashTime"][idx][0]),
            )

            run[tree][f"{prefix}OpFlashErrorX"][idx] = run[tree][
                f"{prefix}OpFlash{sufix}Y"
            ][idx] - reshape_array(
                run[tree]["SignalParticleX"][idx],
                len(run[tree][f"{prefix}OpFlash{sufix}X"][idx][0]),
            )

            run[tree][f"{prefix}OpFlashErrorY"][idx] = run[tree][
                f"{prefix}OpFlash{sufix}Y"
            ][idx] - reshape_array(
                run[tree]["SignalParticleY"][idx],
                len(run[tree][f"{prefix}OpFlash{sufix}Y"][idx][0]),
            )

            run[tree][f"{prefix}OpFlashErrorZ"][idx] = run[tree][
                f"{prefix}OpFlash{sufix}Z"
            ][idx] - reshape_array(
                run[tree]["SignalParticleZ"][idx],
                len(run[tree][f"{prefix}OpFlash{sufix}Z"][idx][0]),
            )
            if info["GEOMETRY"] == "vd":
                idx_cathode = np.where(
                    (run[tree][f"{prefix}OpFlashPlane"][idx] == 0)
                    * (run[tree][f"{prefix}OpFlashPE"][idx] > 0)
                )
                idx_membrane = np.where(
                    (
                        (run[tree][f"{prefix}OpFlashPlane"][idx] == 1)
                        + (run[tree][f"{prefix}OpFlashPlane"][idx] == 2)
                    )
                    * (run[tree][f"{prefix}OpFlashPE"][idx] > 0)
                )
                idx_endcap = np.where(
                    (
                        (run[tree][f"{prefix}OpFlashPlane"][idx] == 3)
                        + (run[tree][f"{prefix}OpFlashPlane"][idx] == 4)
                    )
                    * (run[tree][f"{prefix}OpFlashPE"][idx] > 0)
                )
                # Calculate OpFlashR based on the plane
                run[tree][f"{prefix}OpFlashR"][idx_cathode] = np.sqrt(
                    run[tree][f"{prefix}OpFlashErrorY"][idx_cathode] ** 2
                    + run[tree][f"{prefix}OpFlashErrorZ"][idx_cathode] ** 2,
                )
                # membranes at y=-743.302 and y=743.302
                run[tree][f"{prefix}OpFlashR"][idx_membrane] = np.sqrt(
                    run[tree][f"{prefix}OpFlashErrorX"][idx_membrane] ** 2
                    + run[tree][f"{prefix}OpFlashErrorZ"][idx_membrane] ** 2
                )
                # end-caps at z=-96.5 and z=2188.38
                run[tree][f"{prefix}OpFlashR"][idx_endcap] = np.sqrt(
                    run[tree][f"{prefix}OpFlashErrorX"][idx_endcap] ** 2
                    + run[tree][f"{prefix}OpFlashErrorY"][idx_endcap] ** 2
                )
            elif info["GEOMETRY"] == "hd":
                run[tree][f"{prefix}OpFlashR"][idx] = np.sqrt(
                    run[tree][f"{prefix}OpFlashErrorY"][idx] ** 2
                    + run[tree][f"{prefix}OpFlashErrorZ"][idx] ** 2,
                )
            else:
                rprint(f"[red][ERROR] Invalid geometry {info['GEOMETRY']}[/red]")
                continue

    output += f"\tOpFlash basic variable computation \t-> Done!\n"
    return run, output, new_int_branches + new_float_branches


def compute_opflash_event(
    run,
    configs,
    params: Optional[dict] = None,
    trees: Optional[list[str]] = None,
    rm_branches: bool = False,
    output: Optional[str] = None,
    debug=False,
):
    """
    Compute the OpFlash variables for the events in the run.
    """
    computations = ["Total", "Mean", "Max", "Min"]
    new_branches = []
    # New branches
    for comp in computations:
        this_branches = [
            f"{comp}OpFlashNHits",
            f"{comp}OpFlashR",
            f"{comp}OpFlashTime",
            f"{comp}OpFlashPE",
        ]
        for branch in this_branches:
            run["Truth"][branch] = np.zeros(
                len(run["Truth"]["Event"]), dtype=np.float32
            )
        new_branches += this_branches

    for var in ["NHits", "PE", "R", "Time"]:
        run["Truth"][f"TotalOpFlash{var}"] = np.sum(
            run["Truth"][f"OpFlash{var}"], axis=1
        )
        run["Truth"][f"MeanOpFlash{var}"] = np.sum(
            run["Truth"][f"OpFlash{var}"] * run["Truth"]["OpFlashPE"],
            axis=1,
            where=run["Truth"]["OpFlashPE"] > 0,
        ) / np.sum(run["Truth"]["OpFlashPE"], axis=1)
        run["Truth"][f"MaxOpFlash{var}"] = np.max(run["Truth"][f"OpFlash{var}"], axis=1)
        # Find the index of the min variable where OpFlashPE > 0
        run["Truth"][f"MinOpFlash{var}"] = np.min(
            run["Truth"][f"OpFlash{var}"],
            axis=1,
            initial=1e6,
            where=run["Truth"]["OpFlashPE"] > 0,
        )
    # If OpFlashMeanR is Nan set it to 0
    for branch in new_branches:
        run["Truth"][branch] = np.where(
            np.isnan(run["Truth"][branch]), 0, run["Truth"][branch]
        )

    output += f"\tEvent OpFlashStat variables computation -> Done!\n"
    return run, output, new_branches


def compute_opflash_main(
    run,
    configs,
    params: Optional[dict] = None,
    trees: Optional[list[str]] = None,
    rm_branches: bool = False,
    output: Optional[str] = None,
    debug=False,
):
    """
    Compute the main OpFlash variables for the OpFlash with more PE.
    """
    # New branches
    for tree in trees:
        new_branches = [
            "MainOpFlashPur",
            "MainOpFlashNHits",
            "MainOpFlashTime",
            "MainOpFlashdT",
            "MainOpFlashMaxPE",
            "MainOpFlashPE",
            "MainOpFlashR",
            "MainOpFlashErrorY",
            "MainOpFlashErrorZ",
        ]
        new_bool_branches = ["MainOpFlashSignal"]
        prefix = ""
        if tree == "Reco":
            new_branches = [
                "MainAdjOpFlashPur",
                "MainAdjOpFlashNHits",
                "MainAdjOpFlashTime",
                "MainAdjOpFlashdT",
                "MainAdjOpFlashMaxPE",
                "MainAdjOpFlashPE",
                "MainAdjOpFlashR",
                "MainAdjOpFlashErrorY",
                "MainAdjOpFlashErrorZ",
            ]
            new_bool_branches = ["MainAdjOpFlashSignal"]
            prefix = "Adj"

        for branch in new_branches:
            for tree in trees:
                run[tree][branch] = np.zeros(len(run[tree]["Event"]), dtype=np.float32)
        for branch in new_bool_branches:
            for tree in trees:
                run[tree][branch] = np.zeros(len(run[tree]["Event"]), dtype=np.bool)

        run[tree][f"Main{prefix}OpFlashIdx"] = np.zeros(
            len(run[tree]["Event"]), dtype=int
        )

        for config in configs:
            info, this_params, output = get_param_dict(
                f"{root}/config/{config}/{config}", params, output, debug=debug
            )
            idx = np.where(
                (np.asarray(run[tree]["Geometry"]) == info["GEOMETRY"])
                * (np.asarray(run[tree]["Version"]) == info["VERSION"])
            )

            run[tree][f"Main{prefix}OpFlashIdx"][idx] = np.argmax(
                run[tree][f"{prefix}OpFlashPE"][idx], axis=1
            )

            run[tree][f"Main{prefix}OpFlashPur"][idx] = run[tree][
                f"{prefix}OpFlashPur"
            ][idx][
                np.arange(run[tree][f"{prefix}OpFlashPur"][idx].shape[0]),
                run[tree][f"Main{prefix}OpFlashIdx"][idx],
            ]
            run[tree][f"Main{prefix}OpFlashSignal"][idx] = (
                run[tree][f"Main{prefix}OpFlashPur"][idx] > 0
            )

            run[tree][f"Main{prefix}OpFlashNHits"][idx] = run[tree][
                f"{prefix}OpFlashNHits"
            ][idx][
                np.arange(run[tree][f"{prefix}OpFlashNHits"][idx].shape[0]),
                run[tree][f"Main{prefix}OpFlashIdx"][idx],
            ]
            run[tree][f"Main{prefix}OpFlashTime"][idx] = run[tree][
                f"{prefix}OpFlashTime"
            ][idx][
                np.arange(run[tree][f"{prefix}OpFlashTime"][idx].shape[0]),
                run[tree][f"Main{prefix}OpFlashIdx"][idx],
            ]
            run[tree][f"Main{prefix}OpFlashdT"][idx] = run[tree][f"{prefix}OpFlashdT"][
                idx
            ][
                np.arange(run[tree][f"{prefix}OpFlashdT"][idx].shape[0]),
                run[tree][f"Main{prefix}OpFlashIdx"][idx],
            ]
            run[tree][f"Main{prefix}OpFlashPE"][idx] = run[tree][f"{prefix}OpFlashPE"][
                idx
            ][
                np.arange(run[tree][f"{prefix}OpFlashPE"][idx].shape[0]),
                run[tree][f"Main{prefix}OpFlashIdx"][idx],
            ]
            run[tree][f"Main{prefix}OpFlashMaxPE"][idx] = run[tree][
                f"{prefix}OpFlashMaxPE"
            ][idx][
                np.arange(run[tree][f"{prefix}OpFlashMaxPE"][idx].shape[0]),
                run[tree][f"Main{prefix}OpFlashIdx"][idx],
            ]

            run[tree][f"Main{prefix}OpFlashErrorY"][idx] = run[tree][
                f"{prefix}OpFlashErrorY"
            ][idx][
                np.arange(run[tree][f"{prefix}OpFlashErrorY"][idx].shape[0]),
                run[tree][f"Main{prefix}OpFlashIdx"][idx],
            ]
            run[tree][f"Main{prefix}OpFlashErrorZ"][idx] = run[tree][
                f"{prefix}OpFlashErrorZ"
            ][idx][
                np.arange(run[tree][f"{prefix}OpFlashErrorZ"][idx].shape[0]),
                run[tree][f"Main{prefix}OpFlashIdx"][idx],
            ]
            run[tree][f"Main{prefix}OpFlashR"][idx] = np.sqrt(
                np.power(run[tree][f"Main{prefix}OpFlashErrorY"][idx], 2)
                + np.power(run[tree][f"Main{prefix}OpFlashErrorZ"][idx], 2),
                dtype=np.float32,
            )

    output += f"\tOpFlash main variables computation \t-> Done!\n"
    return run, output, new_branches + new_bool_branches


def compute_opflash_advanced(
    run,
    configs,
    params: Optional[dict] = None,
    rm_branches: bool = False,
    output: Optional[str] = None,
    debug=False,
):
    """
    Compute the OpFlash variables for the events in the run.
    """
    # Required branches
    required_branches = ["AdjOpFlashMaxPE", "AdjOpFlashPE"]

    # New branches
    new_float_branches = ["AdjOpFlashRatio"]

    for branch_list, branch_type in zip([new_float_branches], [np.float32]):
        for branch in branch_list:
            run["Reco"][branch] = np.zeros(
                (len(run["Reco"]["Event"]), len(run["Reco"]["AdjOpFlashPE"][0])),
                dtype=branch_type,
            )

    for config in configs:
        info, this_params, output = get_param_dict(
            f"{root}/config/{config}/{config}", params, output, debug=debug
        )
        idx = np.where(
            (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
            * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
        )

        run["Reco"]["AdjOpFlashRatio"][idx] = (
            run["Reco"]["AdjOpFlashMaxPE"][idx] / run["Reco"]["AdjOpFlashPE"][idx]
        )
        # If AdjOpFlashRatio is 0 set it to Nan
        run["Reco"]["AdjOpFlashRatio"][idx] = np.where(
            run["Reco"]["AdjOpFlashRatio"][idx] == 0,
            np.nan,
            run["Reco"]["AdjOpFlashRatio"][idx],
        )

    output += f"\tOpFlash variables computation \t-> Done!\n"
    return run, output, new_float_branches


def compute_opflash_matching(
    run,
    configs,
    params: Optional[dict] = None,
    rm_branches: bool = False,
    output: Optional[str] = None,
    debug=False,
):
    """
    Compute the OpFlash variables for the events in the run.
    """
    # New branches
    new_branches = [
        "MatchedOpFlashSignal",
        "MatchedOpFlashErrorY",
        "MatchedOpFlashErrorZ",
    ]
    new_braches_types = [np.bool, np.float32, np.float32]
    for branch, branch_type in zip(new_branches, new_braches_types):
        run["Reco"][branch] = np.zeros(len(run["Reco"]["Event"]), dtype=branch_type)

    for config in configs:
        info, this_params, output = get_param_dict(
            f"{root}/config/{config}/{config}", params, output, debug=debug
        )
        idx = np.where(
            (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
            * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
        )
        run["Reco"]["MatchedOpFlashSignal"][idx] = (
            run["Reco"]["MatchedOpFlashPur"][idx] > 0
        )

        run["Reco"]["MatchedOpFlashErrorY"][idx] = (
            run["Reco"]["MatchedOpFlashRecoY"][idx]
            - run["Reco"]["SignalParticleY"][idx]
        )

        run["Reco"]["MatchedOpFlashErrorZ"][idx] = (
            run["Reco"]["MatchedOpFlashRecoZ"][idx]
            - run["Reco"]["SignalParticleZ"][idx]
        )

    output += f"\tMatchedOpFlash Signal Defined \t-> Done!\n"
    return run, output, new_branches
