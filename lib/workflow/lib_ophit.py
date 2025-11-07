import numpy as np

from typing import Optional
from .functions import reshape_array


def compute_ophit_basic(
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
    # Generate repeated arrays of SignalParticleX, SignalParticleY, SignalParticleZ
    converted_x = reshape_array(
        run["Truth"]["SignalParticleX"], len(run["Truth"]["OpHitX"][0])
    )
    converted_y = reshape_array(
        run["Truth"]["SignalParticleY"], len(run["Truth"]["OpHitY"][0])
    )
    converted_z = reshape_array(
        run["Truth"]["SignalParticleZ"], len(run["Truth"]["OpHitZ"][0])
    )

    # New branches
    new_branches = ["OpHitDX", "OpHitDY", "OpHitDZ", "OpHitR"]
    for branch in new_branches:
        run["Truth"][branch] = np.zeros(
            (len(run["Truth"]["Event"]), len(run["Truth"]["OpHitTime"][0])),
            dtype=np.float32,
        )

    # Create OpHitR array
    run["Truth"]["OpHitDX"] = np.absolute(run["Truth"]["OpHitX"] - converted_x)
    run["Truth"]["OpHitDY"] = np.absolute(run["Truth"]["OpHitY"] - converted_y)
    run["Truth"]["OpHitDZ"] = np.absolute(run["Truth"]["OpHitZ"] - converted_z)

    if "OpHitPlane" not in run["Truth"].keys():
        idx_cathode = np.where(run["Truth"]["OpHitX"] == -327.5)
        idx_membrane = np.where(np.absolute(run["Truth"]["OpHitY"]) == 743.3024292)
        idx_end_cap = np.where(
            (run["Truth"]["OpHitZ"] == -96.5)
            | (run["Truth"]["OpHitZ"] == 2188.37988281)
        )

        run["Truth"]["OpHitR"][idx_cathode] = np.sqrt(
            run["Truth"]["OpHitDY"][idx_cathode] ** 2
            + run["Truth"]["OpHitDZ"][idx_cathode] ** 2
        )
        run["Truth"]["OpHitR"][idx_membrane] = np.sqrt(
            run["Truth"]["OpHitDX"][idx_membrane] ** 2
            + run["Truth"]["OpHitDZ"][idx_membrane] ** 2
        )
        run["Truth"]["OpHitR"][idx_end_cap] = np.sqrt(
            run["Truth"]["OpHitDX"][idx_end_cap] ** 2
            + run["Truth"]["OpHitDY"][idx_end_cap] ** 2
        )

    else:
        run["Truth"]["OpHitR"] = np.sqrt(
            np.power(converted_y - run["Truth"]["OpHitY"], 2)
            + np.power(converted_z - run["Truth"]["OpHitZ"], 2),
            dtype=np.float32,
            where=(run["Truth"]["OpHitPlane"] == 0),
        )
        run["Truth"]["OpHitR"] = np.sqrt(
            np.power(converted_x - run["Truth"]["OpHitX"], 2)
            + np.power(converted_z - run["Truth"]["OpHitZ"], 2),
            dtype=np.float32,
            where=(run["Truth"]["OpHitPlane"] == 1) + (run["Truth"]["OpHitPlane"] == 2),
        )
        run["Truth"]["OpHitR"] = np.sqrt(
            np.power(converted_x - run["Truth"]["OpHitX"], 2)
            + np.power(converted_y - run["Truth"]["OpHitY"], 2),
            dtype=np.float32,
            where=(run["Truth"]["OpHitPlane"] == 3) + (run["Truth"]["OpHitPlane"] == 4),
        )

    output += f"\tBasic OpHit variables computation \t-> Done!\n"
    return run, output, new_branches


def compute_ophit_event(
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
        "OpHitMaxPE",
        "OpHitMeanPE",
        "OpHitMaxPEY",
        "OpHitMaxPEZ",
        "OpHitMaxPER",
        "OpHitMeanR",
        "OpHitMeanTime",
    ]
    for branch in new_branches:
        run["Truth"][branch] = np.zeros(len(run["Truth"]["Event"]), dtype=np.float32)

    # Create OpHitR array
    run["Truth"]["OpHitMaxPE"] = np.max(run["Truth"]["OpHitPE"], axis=1)
    run["Truth"]["OpHitTotalPE"] = np.sum(run["Truth"]["OpHitPE"], axis=1)
    run["Truth"]["OpHitMeanPE"] = run["Truth"]["OpHitTotalPE"] / np.sum(
        run["Truth"]["OpHitPE"] > 0, 1
    )
    # Create variable for the idx of the max PE
    run["Truth"]["OpHitMaxPEIdx"] = np.argmax(run["Truth"]["OpHitPE"], axis=1)
    run["Truth"]["OpHitMaxPEX"] = run["Truth"]["OpHitDX"][
        np.arange(run["Truth"]["OpHitDX"].shape[0]), run["Truth"]["OpHitMaxPEIdx"]
    ]
    run["Truth"]["OpHitMaxPEY"] = run["Truth"]["OpHitDY"][
        np.arange(run["Truth"]["OpHitDY"].shape[0]), run["Truth"]["OpHitMaxPEIdx"]
    ]
    run["Truth"]["OpHitMaxPEZ"] = run["Truth"]["OpHitDZ"][
        np.arange(run["Truth"]["OpHitDZ"].shape[0]), run["Truth"]["OpHitMaxPEIdx"]
    ]
    run["Truth"]["OpHitMaxPER"] = run["Truth"]["OpHitR"][
        np.arange(run["Truth"]["OpHitR"].shape[0]), run["Truth"]["OpHitMaxPEIdx"]
    ]

    for var in ["R", "Time"]:
        # Weight the mean R by the PE
        run["Truth"][f"OpHitMean{var}"] = np.sum(
            run["Truth"][f"OpHit{var}"] * run["Truth"]["OpHitPE"],
            axis=1,
            where=(run["Truth"]["OpHitPE"] > 0),
        ) / np.sum(run["Truth"]["OpHitPE"], axis=1)
        # In mean R, replace nans with -1e6
        run["Truth"][f"OpHitMean{var}"] = np.nan_to_num(
            run["Truth"][f"OpHitMean{var}"], nan=-1e6, posinf=-1e6, neginf=-1e6
        )

    output += f"\tEvent OpHitStat variables computation \t-> Done!\n"
    return run, output, new_branches


def compute_ophit_advanced(
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
    # Generate repeated arrays of SignalParticleX, SignalParticleY, SignalParticleZ
    converted_x = reshape_array(
        run["Truth"]["SignalParticleX"], len(run["Truth"]["OpHitX"][0])
    )

    converted_y = reshape_array(
        run["Truth"]["SignalParticleY"], len(run["Truth"]["OpHitY"][0])
    )

    converted_z = reshape_array(
        run["Truth"]["SignalParticleZ"], len(run["Truth"]["OpHitZ"][0])
    )

    flash_id_list = np.unique(run["Truth"]["OpHitFlashID"])
    # New branches
    new_branches = [
        "OpFlashRefPE",
        "OpFlashResidual",
        "OpFlashTime",
        "OpFlashPur",
        "OpFlashSignal",
        "OpFlashNHits",
        "OpFlashPE",
    ]
    for branch in new_branches:
        run["Truth"][branch] = np.zeros(
            (len(run["Truth"]["Event"]), len(flash_id_list)), dtype=np.float32
        )

    # Make empty array for OpHitResidual
    run["Truth"]["OpHitResidual"] = np.zeros(
        (len(run["Truth"]["Event"]), len(run["Truth"]["OpHitPE"][0])), dtype=np.float32
    )

    # Make each entry in the OpFlashID equal to its idx
    run["Truth"]["OpFlashID"] = np.ones(
        (len(run["Truth"]["Event"]), len(flash_id_list)), dtype=int
    )
    run["Truth"]["OpFlashID"] = run["Truth"]["OpFlashID"] * np.arange(
        len(flash_id_list)
    )

    # Change all nans in run["Truth"]["OpHitPur"] for 0
    run["Truth"]["OpHitPur"] = np.nan_to_num(
        run["Truth"]["OpHitPur"], nan=0.0, posinf=0.0, neginf=0.0
    )
    run["Truth"]["OpHitDecay"] = np.power(converted_x, 2) / (
        np.power(converted_x, 2)
        + np.power(converted_y - run["Truth"]["OpHitY"], 2)
        + np.power(converted_z - run["Truth"]["OpHitZ"], 2)
    )
    run["Truth"]["OpHitRefPE"] = np.array(
        run["Truth"]["OpHitPE"] * run["Truth"]["OpHitPur"]
    )

    for flash_id in flash_id_list:
        jdx = np.where(run["Truth"]["OpFlashID"] == flash_id)
        flash_id_filter = np.asarray(run["Truth"]["OpHitFlashID"] == flash_id)
        event_id_count = np.sum(flash_id_filter, axis=1)
        run["Truth"]["OpFlashRefPE"][jdx] = (
            np.sum(run["Truth"]["OpHitRefPE"] * flash_id_filter, axis=1)
            / event_id_count
        )
        run["Truth"]["OpHitRefPE"][flash_id_filter] = np.repeat(
            run["Truth"]["OpFlashRefPE"][jdx], event_id_count
        )
        run["Truth"]["OpFlashNHits"][jdx] = np.sum(flash_id_filter, axis=1)
        run["Truth"]["OpFlashPE"][jdx] = (
            np.sum(run["Truth"]["OpHitPE"] * flash_id_filter, axis=1) / event_id_count
        )

        run["Truth"]["OpFlashResidual"][jdx] = (
            np.mean(
                np.power(
                    (
                        run["Truth"]["OpHitPE"]
                        - run["Truth"]["OpHitRefPE"] * run["Truth"]["OpHitDecay"]
                    )
                    * flash_id_filter,
                    2,
                ),
                axis=1,
            )
            / run["Truth"]["OpFlashNHits"][jdx]
            / run["Truth"]["OpFlashPE"][jdx]
        )
        run["Truth"]["OpHitResidual"][flash_id_filter] = np.repeat(
            run["Truth"]["OpFlashResidual"][jdx], event_id_count
        )
        run["Truth"]["OpFlashTime"][jdx] = np.sum(
            (run["Truth"]["OpHitTime"] * run["Truth"]["OpHitPE"]) * flash_id_filter,
            axis=1,
        ) / np.sum(run["Truth"]["OpHitPE"] * flash_id_filter, axis=1)
        # run["Truth"]["OpFlashPur"][jdx] = np.sum(run["Truth"]["OpHitPur"][idx] * run["Truth"]["OpHitPE"][idx], axis=1) / np.sum(run["Truth"]["OpHitPE"][idx], axis=1)
    run["Truth"]["OpFlashSignal"] = (abs(run["Truth"]["OpFlashTime"]) < 5) == True

    output += f"\tAdvanced OpHit variables computation \t-> Done!\n"
    return run, output, new_branches
