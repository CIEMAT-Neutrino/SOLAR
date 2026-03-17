import json
import numba
import pickle
import numpy as np

from typing import Optional
from itertools import product
from rich import print as rprint
from lib.lib_format import remove_branches, get_param_dict

from src.utils import get_project_root

root = get_project_root()


def compute_true_efficiency(
    run: dict[dict],
    configs: dict[str, list[str]],
    params: Optional[dict] = None,
    rm_branches: bool = False,
    output: str = "",
    debug: bool = False,
) -> tuple[dict[dict], str, list[str]]:
    """
    Compute the true efficiency of the events in the run.
    """
    required_branches = {
        "Truth": ["Event", "Flag", "Geometry", "Version"],
        "Reco": [
            "Event",
            "Flag",
            "Geometry",
            "Version",
            "NHits",
            "Charge",
            "Generator",
        ],
    }
    # New branches
    new_branches = [
        "RecoIndex",
        "RecoMatch",
        "PDSMatch",
        "PDSPlane",
        "PDSPE",
        "ClusterCount",
        "HitCount",
        "TrueIndex",
        "TrueMain",
        "PDSPlane",
        "PDSPE",
    ]
    run["Truth"][new_branches[0]] = np.zeros(len(run["Truth"]["Event"]), dtype=np.int32)
    run["Truth"][new_branches[1]] = np.zeros(len(run["Truth"]["Event"]), dtype=np.bool_)
    run["Truth"][new_branches[2]] = np.zeros(len(run["Truth"]["Event"]), dtype=np.bool_)
    run["Truth"][new_branches[3]] = np.zeros(len(run["Truth"]["Event"]), dtype=np.int32)
    run["Truth"][new_branches[4]] = np.zeros(
        len(run["Truth"]["Event"]), dtype=np.float32
    )
    run["Truth"][new_branches[5]] = np.zeros(len(run["Truth"]["Event"]), dtype=np.int16)
    run["Truth"][new_branches[6]] = np.zeros(len(run["Truth"]["Event"]), dtype=np.int32)
    run["Reco"][new_branches[7]] = np.zeros(len(run["Reco"]["Event"]), dtype=np.int32)
    run["Reco"][new_branches[8]] = np.ones(len(run["Reco"]["Event"]), dtype=np.bool_)
    run["Reco"][new_branches[9]] = np.zeros(len(run["Reco"]["Event"]), dtype=np.int32)
    run["Reco"][new_branches[10]] = np.zeros(
        len(run["Reco"]["Event"]), dtype=np.float32
    )

    for config in configs:
        info, params, output = get_param_dict(
            f"{root}/config/{config}/{config}", params, output, debug=debug
        )
        idx = np.where(
            (np.asarray(run["Truth"]["Geometry"]) == info["GEOMETRY"])
            * (np.asarray(run["Truth"]["Version"]) == info["VERSION"])
        )
        jdx = np.where(
            (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
            * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
        )

        filter_gen = True
        if info["VERSION"] in ["hd_1x2x6", "vd_1x8x14_3view_30deg"]:
            rprint(
                f"[yellow][WARNING][/yellow] Not filtering generator for detection efficiency computation in version {info['VERSION']}"
            )
            filter_gen = False

        result = generate_index(
            run["Truth"]["Event"][idx],
            run["Truth"]["Flag"][idx],
            (
                run["Truth"]["OpFlashPur"][idx]
                if (
                    "OpFlashPur" in run["Truth"]
                    and len(run["Truth"]["OpFlashPur"][0]) > 0
                )
                else np.zeros((len(run["Truth"]["Event"][idx]), 1), dtype=np.float32)
            ),
            (
                run["Truth"]["OpFlashPlane"][idx]
                if (
                    "OpFlashPlane" in run["Truth"]
                    and len(run["Truth"]["OpFlashPlane"][0]) > 0
                )
                else -1 * np.ones((len(run["Truth"]["Event"][idx]), 1), dtype=np.int8)
            ),
            (
                run["Truth"]["OpFlashPE"][idx]
                if (
                    "OpFlashPE" in run["Truth"]
                    and len(run["Truth"]["OpFlashPE"][0]) > 0
                )
                else -1e6
                * np.ones((len(run["Truth"]["Event"][idx]), 1), dtype=np.float32)
            ),
            run["Reco"]["Event"][jdx],
            run["Reco"]["Flag"][jdx],
            run["Reco"]["NHits"][jdx],
            run["Reco"]["Charge"][jdx],
            run["Reco"]["Generator"][jdx],
            (
                run["Reco"]["MatchedOpFlashPur"][jdx]
                if "MatchedOpFlashPur" in run["Reco"]
                else np.zeros(len(run["Reco"]["Event"][jdx]), dtype=np.float32)
            ),
            (
                run["Reco"]["MatchedOpFlashPlane"][jdx]
                if "MatchedOpFlashPlane" in run["Reco"]
                else -1 * np.ones(len(run["Reco"]["Event"][jdx]), dtype=np.int8)
            ),
            (
                run["Reco"]["MatchedOpFlashPE"][jdx]
                if "MatchedOpFlashPE" in run["Reco"]
                else -1e6 * np.ones(len(run["Reco"]["Event"][jdx]), dtype=np.float32)
            ),
            filter_gen=filter_gen,
            debug=debug,
        )
        run["Truth"]["RecoIndex"][idx] = np.asarray(result[0])
        run["Truth"]["RecoMatch"][idx] = np.asarray(result[1])
        run["Truth"]["PDSMatch"][idx] = np.asarray(result[2])
        run["Truth"]["PDSPlane"][idx] = np.asarray(result[3])
        run["Truth"]["PDSPE"][idx] = np.asarray(result[4])
        run["Truth"]["ClusterCount"][idx] = np.asarray(result[5])
        run["Truth"]["HitCount"][idx] = np.asarray(result[6])
        run["Reco"]["TrueIndex"][jdx] = np.asarray(result[7])
        run["Reco"]["TrueMain"][jdx] = np.asarray(result[8])
        run["Reco"]["PDSPlane"] = np.asarray(result[9])
        run["Reco"]["PDSPE"] = np.asarray(result[10])

    run = remove_branches(run, rm_branches, [], debug=debug)
    output += f"\tTrue efficiency computation \t-> Done!\n\n"
    return run, output, new_branches


@numba.njit
def generate_index(
    true_event,
    true_flag,
    true_flash,
    true_plane,
    true_PE,
    reco_event,
    reco_flag,
    reco_nhits,
    reco_charge,
    reco_gen,
    reco_flash,
    reco_plane,
    reco_PE,
    filter_gen: bool = True,
    debug: bool = False,
):
    """
    Generate the event index for the true and reco events.
    """
    true_index = np.arange(len(true_event), dtype=np.int32)
    true_result = np.zeros(len(true_event), dtype=np.int32) - 1
    true_TPC_match = np.zeros(len(true_event), dtype=np.bool_)
    true_PDS_match = np.zeros(len(true_event), dtype=np.bool_)
    true_PDS_plane = -1 * np.ones(len(true_event), dtype=np.int8)
    true_PDS_PE = np.zeros(len(true_event), dtype=np.float32)
    true_counts = np.zeros(len(true_event), dtype=np.int32)
    true_nhits = np.zeros(len(true_event), dtype=np.int32)
    reco_result = np.zeros(len(reco_event), dtype=np.int32)
    reco_main = np.ones(len(reco_event), dtype=np.bool_)
    reco_PDS_plane = -1 * np.ones(len(reco_event), dtype=np.int8)
    reco_PDS_PE = np.zeros(len(reco_event), dtype=np.float32)

    end_j = 0
    start_j = 0
    assigned_true = []
    for i in range(0, len(reco_event)):
        # print(f"Reco Event {i-1} of {len(reco_event)}")
        if i == 0:
            start_j = 0
        else:
            start_j = reco_result[i - 1]
        j = 0
        for z in range(true_index[end_j], true_index[-1] + 1):
            # print(f"Lopping Over True Event {z} of {len(true_index)}")
            if reco_event[i + 1] != true_event[z] or reco_flag[i + 1] != true_flag[z]:
                j = j + 1
            else:
                # print(f"Match Found at Reco Event {i+1} True Event {z}: {reco_event[i + 1]}, {true_event[z]} and {reco_flag[i + 1]}, {true_flag[z]}")
                start_j = end_j
                end_j = end_j + j
                break

        for k in range(start_j, end_j + 1):
            # print(f"Second Looping Over True Event {k} of {len(true_index)}")
            if (
                (reco_event[i] == true_event[k])
                * (reco_flag[i] == true_flag[k])
                * (reco_gen[i] == 1 if filter_gen else True)
            ):
                reco_result[i] = int(k)
                # Find entry j in true_plane[k] for which true_flash[k][j] > 0 and PE is maximum in true_PE[k]
                if sum(true_flash[k]) > 0:
                    reco_PDS_PE[i] = max(true_PE[k][true_flash[k] > 0])
                    reco_PDS_plane[i] = true_plane[k][
                        np.argmax(true_PE[k][true_flash[k] > 0])
                    ]
                else:
                    reco_PDS_PE[i] = -1e6
                    reco_PDS_plane[i] = -1

                if k in assigned_true:
                    if reco_charge[i] > reco_charge[true_result[reco_result[i]]]:
                        reco_main[true_result[reco_result[i]]] = False
                    if reco_charge[i] < reco_charge[true_result[reco_result[i]]]:
                        reco_main[i] = False

                assigned_true.append(k)
                true_result[k] = int(i)
                true_TPC_match[k] = True
                true_counts[k] += 1
                true_nhits[k] = true_nhits[k] + reco_nhits[i]

                if reco_flash[i] > 0:
                    true_PDS_match[k] = True
                    true_PDS_plane[k] = reco_plane[i]
                    true_PDS_PE[k] = reco_PE[i]
                break

    return (
        true_result,
        true_TPC_match,
        true_PDS_match,
        true_PDS_plane,
        true_PDS_PE,
        true_counts,
        true_nhits,
        reco_result,
        reco_main,
        reco_PDS_plane,
        reco_PDS_PE,
    )
