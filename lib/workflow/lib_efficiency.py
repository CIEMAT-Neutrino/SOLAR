import numba
import pickle
import numpy as np

from typing import Optional
from itertools import product
from rich import print as rprint
from lib.workflow.functions import remove_branches, get_param_dict

from src.utils import get_project_root
root = get_project_root()


def compute_particle_weights(run: dict[dict], configs: dict[str, list[str]], params: Optional[dict] = None, rm_branches: bool = False, output: Optional[str] = None, debug: bool = False):
    
    # Define a function to evaluate the joint PDF
    def evaluate_pdf(kde, p_values, x_values, y_values, z_values):
        """Evaluate the PDF at the given (p, x, y, z) points."""
        query_points = np.vstack([p_values, x_values, y_values, z_values]).T
        log_density = kde.score_samples(query_points)  # Log density
        return np.exp(log_density)  # Convert to actual density
    
    def evaluate_1d_pdf(kde, values):
        """Evaluate 1D PDF from a KDE."""
        log_density = kde.score_samples(values[:, None])  # [:, None] ensures 2D input
        return np.exp(log_density)

    # Define the joint PDF using dimensional independence
    def evaluate_joint_pdf(name, p_values, x_values, y_values, z_values, kde_p, kde_x, kde_y, kde_z):
        """Evaluate the joint PDF under the assumption of dimensional independence."""
        pdf_p = evaluate_1d_pdf(kde_p, p_values)
        pdf_x = evaluate_1d_pdf(kde_x, x_values)
        pdf_y = evaluate_1d_pdf(kde_y, y_values)
        pdf_z = evaluate_1d_pdf(kde_z, z_values)
        
        rprint(f"[cyan][INFO] Eveluated PDF for {name} with {len(p_values)} events[/cyan]")
        return pdf_p * pdf_x * pdf_y * pdf_z

    '''
    Compute the single weights for the events in the run.
    '''
    new_branches = ["SignalParticleWeight"]
    for tree, branch in product(["Truth", "Reco"], new_branches):
        run[tree][branch] = np.zeros(len(run[tree]["Event"]), dtype=np.float64)
    
    for config in configs:
        info, params, output = get_param_dict(
            f"{root}/config/{config}/{config}", params, output, debug=debug)
        

        for name in configs[config]:
            if name.lower().startswith("marley"):
                continue
            
            exposure = pickle.load(open(f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/background/truth/{config}/{config}_{name.split('_')[0]}_exposure.pkl", "rb"))
            kde_p = pickle.load(open(f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/background/truth/{config}/{config}_{name.split('_')[0]}_kde_p.pkl", "rb"))
            kde_x = pickle.load(open(f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/background/truth/{config}/{config}_{name.split('_')[0]}_kde_x.pkl", "rb"))
            kde_y = pickle.load(open(f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/background/truth/{config}/{config}_{name.split('_')[0]}_kde_y.pkl", "rb"))
            kde_z = pickle.load(open(f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/background/truth/{config}/{config}_{name.split('_')[0]}_kde_z.pkl", "rb"))
            
            idx = np.where(
                (np.asarray(run["Truth"]["Geometry"]) == info["GEOMETRY"])
                * (np.asarray(run["Truth"]["Version"]) == info["VERSION"])
                * (np.asarray(run["Truth"]["Name"]) == name)
            )

            jdx = np.where(
                (np.asarray(run["Config"]["Version"]) == info["VERSION"])
                * (np.asarray(run["Config"]["Name"]) == name)
            )

            config_tree_weight = 10
            if name.startswith("neutron"):
                config_tree_weight = 8*100
            if name.startswith("gamma"):
                config_tree_weight = 12*10

            run["Truth"]["SignalParticleWeight"][idx] = evaluate_joint_pdf(name, run["Truth"]["SignalParticleP"][idx], run["Truth"]["SignalParticleX"][idx], run["Truth"]["SignalParticleY"][idx], run["Truth"]["SignalParticleZ"][idx], kde_p, kde_x, kde_y, kde_z)
            run["Truth"]["SignalParticleWeight"][idx] = exposure["counts"] * len(run["Truth"]["Event"][idx]) / (config_tree_weight*len(run["Config"]["Geometry"][jdx])) * run["Truth"]["SignalParticleWeight"][idx] / (np.sum(run["Truth"]["SignalParticleWeight"][idx]) * exposure["exposure"])
            
 
    run["Reco"]["SignalParticleWeight"] = run["Truth"]["SignalParticleWeight"][run["Reco"]["TrueIndex"]]


    rprint(f"Particle weights computation \t-> Done!")
    run = remove_branches(run, rm_branches, [], debug=debug)
    output += f"\tTruth weights computation \t-> Done!\n"
    return run, output, new_branches


def compute_true_efficiency(run: dict[dict], configs: dict[str, list[str]], params: Optional[dict] = None, rm_branches: bool = False, output: Optional[str] = None, debug: bool = False):
    """
    Compute the true efficiency of the events in the run.
    """
    required_branches = {"Truth": ["Event", "Flag", "Geometry", "Version"],
                         "Reco": ["Event", "Flag", "Geometry", "Version", "NHits", "Charge", "Generator"]}
    # New branches
    new_branches = ["RecoIndex", "RecoMatch", "PDSMatch",
                    "ClusterCount", "HitCount", "TrueIndex"]
    run["Truth"][new_branches[0]] = np.zeros(
        len(run["Truth"]["Event"]), dtype=int)
    run["Truth"][new_branches[1]] = np.zeros(
        len(run["Truth"]["Event"]), dtype=bool)
    run["Truth"][new_branches[2]] = np.zeros(
        len(run["Truth"]["Event"]), dtype=bool)
    run["Truth"][new_branches[3]] = np.zeros(
        len(run["Truth"]["Event"]), dtype=int)
    run["Truth"][new_branches[4]] = np.zeros(
        len(run["Truth"]["Event"]), dtype=int)
    run["Reco"][new_branches[5]] = np.zeros(
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
            run["Reco"]["MatchedOpFlashPur"][jdx],
            debug=debug,
        )
        run["Truth"]["RecoIndex"][idx]    = np.asarray(result[0])
        run["Truth"]["RecoMatch"][idx]    = np.asarray(result[1])
        run["Truth"]["PDSMatch"][idx]     = np.asarray(result[2])
        run["Truth"]["ClusterCount"][idx] = np.asarray(result[3])
        run["Truth"]["HitCount"][idx]     = np.asarray(result[4])
        run["Reco"]["TrueIndex"][jdx]     = np.asarray(result[5])

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
    reco_flash,
    debug=False,
):
    """
    Generate the event index for the true and reco events.
    """
    true_index = np.arange(len(true_event), dtype=np.int32)
    true_result = np.zeros(len(true_event), dtype=np.int32) - 1
    true_TPC_match = np.zeros(len(true_event), dtype=np.bool_)
    true_PDS_match = np.zeros(len(true_event), dtype=np.bool_)
    true_counts = np.zeros(len(true_event), dtype=np.int32)
    true_nhits = np.zeros(len(true_event), dtype=np.int32)
    reco_result = np.zeros(len(reco_event), dtype=np.int32)

    end_j = 0
    start_j = 0
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

        for k in range(start_j , end_j + 1):
            # print(f"Second Looping Over True Event {k} of {len(true_index)}")
            if (
                (reco_event[i] == true_event[k]) \
                * (reco_flag[i] == true_flag[k]) \
                * (reco_gen[i] == 1)
            ):
                reco_result[i] = int(k)
                true_result[k] = int(i)

                # if reco_charge[i] > reco_charge[true_result[k]]:
                    # true_result[k] = int(i)
                
                true_TPC_match[k] = True
                true_counts[k] += 1
                true_nhits[k] = true_nhits[k] + reco_nhits[i]
                
                if reco_flash[i] > 0:
                    true_PDS_match[k] = True
                break
    
    return true_result, true_TPC_match, true_PDS_match, true_counts, true_nhits, reco_result