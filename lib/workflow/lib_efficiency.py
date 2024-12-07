import numba
import numpy as np

from typing import Optional
from lib.workflow.functions import remove_branches, get_param_dict

def compute_true_efficiency(run: dict[dict], configs: dict[str, list[str]], params: Optional[dict] = None, rm_branches: bool = False, output: Optional[str] = None, debug: bool = False):
    """
    Compute the true efficiency of the events in the run.
    """
    required_branches = {"Truth": ["Event", "Flag", "Geometry", "Version"],
                         "Reco": ["Event", "Flag", "Geometry", "Version", "NHits", "Charge", "Generator"]}
    # New branches
    new_branches = ["RecoIndex", "RecoMatch",
                    "ClCount", "HitCount", "TrueIndex"]
    run["Truth"][new_branches[0]] = np.zeros(
        len(run["Truth"]["Event"]), dtype=int)
    run["Truth"][new_branches[1]] = np.zeros(
        len(run["Truth"]["Event"]), dtype=bool)
    run["Truth"][new_branches[2]] = np.zeros(
        len(run["Truth"]["Event"]), dtype=int)
    run["Truth"][new_branches[3]] = np.zeros(
        len(run["Truth"]["Event"]), dtype=int)
    run["Reco"][new_branches[4]] = np.zeros(
        len(run["Reco"]["Event"]), dtype=int)

    for config in configs:
        info, params, output = get_param_dict(
            f"{root}/config/{config}/{config}", params, output, debug=debug)
        idx = np.where(
            (np.asarray(run["Truth"]["Geometry"]) == info["GEOMETRY"])
            * (np.asarray(run["Truth"]["Version"]) == info["VERSION"])
        )
        jdx = np.where(
            (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
            * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
        )

        result = generate_index(
            run["Truth"]["Event"][idx],
            run["Truth"]["Flag"][idx],
            run["Reco"]["Event"][jdx],
            run["Reco"]["Flag"][jdx],
            run["Reco"]["NHits"][jdx],
            run["Reco"]["Charge"][jdx],
            run["Reco"]["Generator"][jdx],
            debug=debug,
        )
        run["Truth"]["RecoIndex"][idx] = np.asarray(result[0])
        run["Truth"]["RecoMatch"][idx] = np.asarray(result[1])
        run["Truth"]["ClCount"][idx] = np.asarray(result[2])
        run["Truth"]["HitCount"][idx] = np.asarray(result[3])
        run["Reco"]["TrueIndex"][jdx] = np.asarray(result[4])

    run = remove_branches(run, rm_branches, [], debug=debug)
    output += f"\tTrue efficiency computation \t-> Done!\n"
    return run, output, new_branches

@ numba.njit
def generate_index(
    true_event,
    true_flag,
    reco_event,
    reco_flag,
    reco_nhits,
    reco_charge,
    reco_gen,
    debug=False,
):
    """
    Generate the event index for the true and reco events.
    """
    true_index = np.arange(len(true_event), dtype=np.int32)
    true_result = np.zeros(len(true_event), dtype=np.int32) - 1
    true_match = np.zeros(len(true_event), dtype=np.bool_)
    true_counts = np.zeros(len(true_event), dtype=np.int32)
    true_nhits = np.zeros(len(true_event), dtype=np.int32)
    reco_result = np.zeros(len(reco_event), dtype=np.int32)

    end_j = 0
    for i in range(1, len(reco_event)):
        start_j = reco_result[i - 1]
        j = 0
        for z in range(true_index[end_j], true_index[-1] + 1):
            if reco_event[i + 1] != true_event[z] and reco_flag[i + 1] != true_flag[z]:
                j = j + 1
            else:
                start_j = end_j
                end_j = end_j + j
                break

        for k in range(start_j, end_j + 1):
            if (
                reco_event[i] == true_event[k]
                and reco_flag[i] == true_flag[k]
                and reco_gen[k] == 1
            ):
                reco_result[i] = int(k)
                if reco_charge[i] > reco_charge[true_result[k]]:
                    true_result[k] = int(i)
                true_match[k] = True
                true_counts[k] += 1
                true_nhits[k] = true_nhits[k] + reco_nhits[i]
                break
    return true_result, true_match, true_counts, true_nhits, reco_result