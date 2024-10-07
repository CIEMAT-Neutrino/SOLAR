import json
import pickle
import numba
import numpy as np

from typing import Optional
from rich import print as rprint
from itertools import product
from src.utils import get_project_root
from particle import Particle

from lib.df_functions import npy2df
from lib.ana_functions import get_default_info

root = get_project_root()


def compute_reco_workflow(
    run: dict[dict], configs: dict[str, list[str]], params: Optional[dict] = None, workflow: str = "BASIC", rm_branches: bool = True, debug: bool = False
) -> dict[dict]:
    """
    Compute the reco variables for the events in the run.
    All functions are called in the order they are defined in this file.
    All functions get the same arguments.

    Args:
        run: dictionary containing the run data.
        configs: dictionary containing the path to the configuration files for each geoemtry.
        params: dictionary containing the parameters for the reco functions.
        workflow: string containing the reco workflow to be used.
        rm_branches: boolean to remove the branches used in the reco workflow.
        debug: print debug information.

    Returns:
        run (dict): dictionary containing the TTree with the new branches.
    """
    # Compute reco variables
    terminal_output = f"[yellow]\nComputing {workflow} workflow\n[/yellow]"

    if workflow == "BASIC":
        terminal_output += f"Selected basic workflow. Returning loaded variables.\n"
        pass

    elif workflow == "TRUTH":
        run, output = compute_true_efficiency(
            run, configs, params, rm_branches=rm_branches, debug=debug
        )
        terminal_output += output
        run, output = compute_marley_directions(
            run, configs, params, trees=["Truth"], rm_branches=rm_branches, debug=debug
        )
        terminal_output += output

    elif workflow == "MARLEY":
        run, output = compute_true_efficiency(
            run, configs, params, rm_branches=rm_branches, debug=debug
        )
        terminal_output += output
        run, output = compute_marley_energies(
            run, configs, params, trees=["Truth"], rm_branches=rm_branches, debug=debug
        )
        terminal_output += output
        run, output = compute_marley_directions(
            run, configs, params, trees=["Truth"], rm_branches=rm_branches, debug=debug
        )
        terminal_output += output
        run, output = compute_particle_energies(
            run, configs, params, trees=["Truth"], rm_branches=rm_branches, debug=debug
        )
        terminal_output += output

    elif workflow == "RAW":
        run, output = compute_marley_energies(
            run, configs, params, trees=["Truth"], rm_branches=rm_branches, debug=debug
        )
        terminal_output += output
        run, output = compute_particle_energies(
            run, configs, params, trees=["Truth"], rm_branches=rm_branches, debug=debug
        )
        terminal_output += output

    elif workflow == "TRACK":
        run, output = compute_marley_directions(
            run, configs, params, trees=["Reco"], rm_branches=rm_branches, debug=debug
        )
        terminal_output += output
        run, output = compute_particle_directions(
            run, configs, params, rm_branches=rm_branches, debug=debug
        )
        terminal_output += output

    elif workflow == "ADJCL":
        run, output = compute_adjcl_basics(
            run, configs, params, rm_branches=rm_branches, debug=debug
        )
        terminal_output += output
        run, output = compute_cluster_energy(
            run, configs, params, rm_branches=rm_branches, debug=debug
        )
        terminal_output += output
        run, output = compute_adjcl_advanced(
            run, configs, params, rm_branches=rm_branches, debug=debug
        )
        terminal_output += output

    elif workflow in ["ADJFLASH", "OPHIT"]:
        run, output = compute_opflash_advanced(
            run, configs, params, rm_branches=rm_branches, debug=debug
        )
        terminal_output += output
        if workflow == "OPHIT":
            run, output = compute_ophit_basic(
                run, configs, params, rm_branches=rm_branches, debug=debug
            )
            terminal_output += output

    elif workflow in ["CORRECTION", "CALIBRATION", "DISCRIMINATION", "RECONSTRUCTION", "SMEARING", "ANALYSIS"]:
        if workflow == "ANALYSIS":
            run, output = compute_main_variables(
                run, configs, params, rm_branches=rm_branches, debug=debug
            )
            run, output = compute_true_drift(
                run, configs, params, rm_branches=rm_branches, debug=debug
            )
            terminal_output += output
        if workflow == "CORRECTION":
            run, output = compute_adjcl_basics(
                run, configs, params, rm_branches=rm_branches, debug=debug
            )
            run, output = compute_cluster_charge(
                run, configs, params, rm_branches=rm_branches, debug=debug
            )
            terminal_output += output
        if workflow in ["CORRECTION", "CALIBRATION", "DISCRIMINATION", "RECONSTRUCTION", "SMEARING", "ANALYSIS"]:
            run, output = compute_particle_energies(
                run, configs, params, trees=["Reco"], rm_branches=rm_branches, debug=debug
            )
            terminal_output += output
        if workflow in ["CALIBRATION", "DISCRIMINATION", "RECONSTRUCTION", "SMEARING", "ANALYSIS"]:
            run, output = compute_cluster_energy(
                run, configs, params, rm_branches=rm_branches, debug=debug
            )
            terminal_output += output
        if workflow in ["DISCRIMINATION", "RECONSTRUCTION", "SMEARING", "ANALYSIS"]:
            run, output = compute_cluster_calibration(
                run, configs, params, rm_branches=rm_branches, debug=debug
            )
            terminal_output += output
            run, output = compute_adjcl_basics(
                run, configs, params, rm_branches=rm_branches, debug=debug
            )
            terminal_output += output
            run, output = compute_adjcl_advanced(
                run, configs, params, rm_branches=rm_branches, debug=debug
            )
            terminal_output += output
            run, output = compute_total_energy(
                run, configs, params, rm_branches=rm_branches, debug=debug
            )
            terminal_output += output
        if workflow in ["RECONSTRUCTION", "SMEARING", "ANALYSIS"]:
            run, output = compute_reco_energy(
                run, configs, params, rm_branches=rm_branches, debug=debug
            )
            terminal_output += output
        if workflow in ["SMEARING", "ANALYSIS"]:
            run, output = compute_energy_calibration(
                run, configs, params, rm_branches=rm_branches, debug=debug
            )
            terminal_output += output

    elif workflow == "VERTEXING":
        default_workflow_params = {"MAX_FLASH_R": None, "MIN_FLASH_PE": None,
                                   "RATIO_FLASH_PEvsR": None}
        for key in default_workflow_params:
            if params is None:
                params = {}
            if key not in params:
                params[key] = default_workflow_params[key]

        run, output = compute_main_variables(
            run, configs, params, rm_branches=rm_branches, debug=debug
        )
        terminal_output += output
        run, output = compute_opflash_advanced(
            run, configs, params, rm_branches=rm_branches, debug=debug
        )
        terminal_output += output
        run, output = compute_opflash_matching(
            run, configs, params, rm_branches=rm_branches, debug=debug
        )
        terminal_output += output
        run, output = compute_cluster_drift(
            run, configs, params, rm_branches=rm_branches, debug=debug
        )
        terminal_output += output

    rprint(terminal_output +
           f"[green]{workflow} workflow completed!\n[/green]")
    return run


def compute_main_variables(run: dict[dict], configs: dict[str, list[str]], params: Optional[dict] = None, rm_branches: bool = False, output: Optional[str] = None, debug=False):
    """
    Compute the main (backtracked) variables of main particle correspondong to the cluster in the run.
    """
    @numba.njit
    def expand_variables(branch):
        x = branch[:, 0]
        y = branch[:, 1]
        z = branch[:, 2]
        return x, y, z

    required_branches = ["MainVertex",
                         "MainParentVertex", "RecoX", "RecoY", "RecoZ"]

    for branch in ["MainVertex", "MainParentVertex"]:
        x, y, z = expand_variables(run["Reco"][branch])
        main_branch = branch.split("Vertex")[0]
        run["Reco"][f"{main_branch}X"] = x
        run["Reco"][f"{main_branch}Y"] = y
        run["Reco"][f"{main_branch}Z"] = z

    # run["Reco"]["ErrorX"] = abs(run["Reco"]["MainX"] - run["Reco"]["RecoX"])
    run["Reco"]["ErrorY"] = abs(run["Reco"]["MainY"] - run["Reco"]["RecoY"])
    run["Reco"]["ErrorZ"] = abs(run["Reco"]["MainZ"] - run["Reco"]["RecoZ"])
    run["Reco"]["2DError"] = np.sqrt(
        np.power(run["Reco"]["ErrorZ"], 2) + np.power(run["Reco"]["ErrorY"], 2))
    # run["Reco"]["3DError"] = np.sqrt(
    #     np.power(run["Reco"]["ErrorZ"], 2) + np.power(run["Reco"]["ErrorY"], 2) + np.power(run["Reco"]["ErrorX"], 2))

    run["Reco"]["Neutrino"] = run["Reco"]["Generator"] == 1

    run["Reco"]["Electron"] = (
        run["Reco"]["Generator"] == 1) * (run["Reco"]["MarleyFrac"][:, 0] > 0.5)

    run["Reco"]["Gamma"] = (run["Reco"]["Generator"] == 1) * \
        (run["Reco"]["MarleyFrac"][:, 1] > 0.5)

    run["Reco"]["Neutron"] = (run["Reco"]["Generator"]
                              == 1) * (run["Reco"]["MarleyFrac"][:, 2] > 0.5)

    output = f"\tMain variables computation \t-> Done!\n"
    return run, output


def compute_true_efficiency(run: dict[dict], configs: dict[str, list[str]], params: Optional[dict] = None, rm_branches: bool = False, debug: bool = False):
    """
    Compute the true efficiency of the events in the run.
    """
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
    output = f"\tTrue efficiency computation \t-> Done! ({new_branches})\n"
    return run, output


def compute_marley_directions(run, configs, params={}, trees=["Truth", "Reco"], rm_branches: bool = False, output: Optional[str] = None, debug=False):
    """
    This functions loops over the Marley particles and computes the direction of the particles, returning variables with the same structure as TMarleyPDG.
    """
    new_branches = ["TMarleyTheta", "TMarleyPhi", "TMarleyDirectionX",
                    "TMarleyDirectionY", "TMarleyDirectionZ", "TMarleyDirectionMod"]
    for tree in trees:
        for branch in new_branches:
            run[tree][branch] = np.zeros(
                (len(run[tree]["Event"]), len(run[tree]["TMarleyPDG"][0])), dtype=np.float32)

        for config in configs:
            info, params, output = get_param_dict(
                f"{root}/config/{config}/{config}", params, debug=debug)
            idx = np.where(
                (np.asarray(run[tree]["Geometry"]) == info["GEOMETRY"])
                * (np.asarray(run[tree]["Version"]) == info["VERSION"])
            )
            for direction, start, end in zip(["TMarleyDirectionX", "TMarleyDirectionY", "TMarleyDirectionZ"], ["TNuX", "TNuY", "TNuZ"], ["TMarleyEndX", "TMarleyEndY", "TMarleyEndZ"]):
                run[tree][direction][idx] = run[tree][start][:,
                                                             None][idx] - run[tree][end][idx]

            run[tree]["TMarleyDirectionMod"][idx] = np.sqrt(np.power(run[tree]["TMarleyDirectionX"][idx], 2) + np.power(
                run[tree]["TMarleyDirectionY"][idx], 2) + np.power(run[tree]["TMarleyDirectionZ"][idx], 2))
            for coord in ["X", "Y", "Z"]:
                run[tree][f"TMarleyDirection{coord}"][idx] = run[tree][f"TMarleyDirection{coord}"][idx] / \
                    run[tree]["TMarleyDirectionMod"][idx]

            run[tree]["TMarleyTheta"][idx] = np.arccos(
                run[tree]["TMarleyDirectionZ"][idx])
            run[tree]["TMarleyPhi"][idx] = np.arctan2(
                run[tree]["TMarleyDirectionY"][idx], run[tree]["TMarleyDirectionX"][idx])

            run = remove_branches(
                run, rm_branches, ["TMarleyDirectionMod"], tree=tree, debug=debug)

    output = f"\tMarley direction computation \t-> Done! ({new_branches})\n"
    return run, output


def compute_particle_directions(run: dict, configs: dict, params: Optional[dict] = None, trees: list[str] = ["Reco"], rm_branches: bool = False, output: Optional[str] = None, debug: bool = False):
    """
    This functions loops over the Marley particles and computes the direction of the particles, returning variables with the same structure as TMarleyPDG.
    """
    new_branches = ["MTrackTheta", "MTrackPhi", "MTrackDirectionX",
                    "MTrackDirectionY", "MTrackDirectionZ", "MTrackDirectionMod"]
    for tree in trees:
        run[tree]["MTrackDirection"] = np.zeros(
            (len(run[tree]["Event"]), 3), dtype=np.float32)
        for branch in new_branches:
            run[tree][branch] = np.zeros(
                len(run[tree]["Event"]), dtype=np.float32)
        for config in configs:
            info, params, output = get_param_dict(
                f"{root}/config/{config}/{config}", params, debug=debug)
            idx = np.where(
                (np.asarray(run[tree]["Geometry"]) == info["GEOMETRY"])
                * (np.asarray(run[tree]["Version"]) == info["VERSION"])
            )
            run[tree]["MTrackDirection"][idx] = np.subtract(
                run[tree]["MTrackEnd"][idx], run[tree]["MTrackStart"][idx])
            # run[tree]["MTrackDirectionMod"][idx] = np.linalg.norm(run[tree]["MTrackDirection"][idx], axis=1)
            # run[tree]["MTrackDirection"][idx] = run[tree]["MTrackDirection"][idx]/run[tree]["MTrackDirectionMod"][:,None][idx]
            for coord_idx, coord in enumerate(["MTrackDirectionX", "MTrackDirectionY", "MTrackDirectionZ"]):
                rprint(f"Computing {coord} direction")
                run[tree][coord][idx] = run[tree]["MTrackDirection"][:, coord_idx][idx]

            run[tree]["MTrackTheta"][idx] = np.arccos(
                run[tree]["MTrackDirectionZ"][idx])
            run[tree]["MTrackPhi"][idx] = np.arctan2(
                run[tree]["MTrackDirectionY"][idx], run[tree]["MTrackDirectionX"][idx])

        run = remove_branches(
            run, rm_branches, ["MTrackDirectionMod"], tree=tree, debug=debug)

    output = f"\tMarley direction computation \t-> Done! ({new_branches})\n"
    return run, output


def compute_marley_energies(run, configs, params: Optional[dict] = None, trees=["Truth", "Reco"], rm_branches: bool = False, output: Optional[str] = None, debug=False):
    if params is None:
        params = {"NORM_TO_NUE": False}
    for tree in trees:
        pdg_list = np.unique(run[tree]["TMarleyPDG"]
                             [np.where(run[tree]["TMarleyMother"] == 0)])
        pdg_list = pdg_list[pdg_list != 0]
        mass_list = [Particle.from_pdgid(pdg).mass for pdg in pdg_list]
        new_branches = ["TMarleySumE", "TMarleySumP",
                        "TMarleySumK", "TMarleyK", "TMarleyMass"]
        run[tree][new_branches[0]] = np.zeros(
            (len(run[tree]["Event"]), len(pdg_list)), dtype=np.float32)
        run[tree][new_branches[1]] = np.zeros(
            (len(run[tree]["Event"]), len(pdg_list)), dtype=np.float32)
        run[tree][new_branches[2]] = np.zeros(
            (len(run[tree]["Event"]), len(pdg_list)), dtype=np.float32)
        run[tree][new_branches[3]] = np.zeros(
            (len(run[tree]["Event"]), len(run[tree]["TMarleyPDG"][0])), dtype=np.float32)
        run[tree][new_branches[4]] = np.zeros(
            (len(run[tree]["Event"]), len(run[tree]["TMarleyPDG"][0])), dtype=np.float32)

        full_pdg_list = np.unique(run[tree]["TMarleyPDG"])
        for non_pdg in [0, 1000120249, 1000140289, 1000190419, 1000210499, 1000220489, 1000130279, 1000360809, 1000360829]:
            full_pdg_list = full_pdg_list[full_pdg_list != non_pdg]
        full_mass_dict = {pdg: Particle.from_pdgid(
            pdg).mass for pdg in full_pdg_list}

        # Gnearte branch for the mass of the particles frmo the TMarleyPDG m times n array and store it in the TMarleyMass branch
        run[tree]["TMarleyMass"] = np.vectorize(
            full_mass_dict.get)(run[tree]["TMarleyPDG"])
        run[tree]["TMarleyMass"] = np.nan_to_num(
            run[tree]["TMarleyMass"], nan=0.0, posinf=0.0, neginf=0.0)
        run[tree]["TMarleyK"] = np.subtract(
            run[tree]["TMarleyE"], run[tree]["TMarleyMass"])

        for idx, pdg in enumerate(pdg_list):
            run[tree][new_branches[0]][:, idx] = np.sum(
                run[tree]["TMarleyE"]*(run[tree]["TMarleyPDG"] == pdg)*(run[tree]["TMarleyMother"] == 0), axis=1)
            run[tree][new_branches[1]][:, idx] = np.sum(
                run[tree]["TMarleyP"]*(run[tree]["TMarleyPDG"] == pdg)*(run[tree]["TMarleyMother"] == 0), axis=1)
            run[tree][new_branches[2]][:, idx] = np.sum(
                run[tree]["TMarleyK"]*(run[tree]["TMarleyPDG"] == pdg)*(run[tree]["TMarleyMother"] == 0), axis=1)

        if params["NORM_TO_NUE"]:
            # Divide by the energy of the neutrino
            run[tree][new_branches[0]] = run[tree][new_branches[0]] / \
                run[tree]["TNuE"][:, None]
            run[tree][new_branches[1]] = run[tree][new_branches[1]] / \
                run[tree]["TNuE"][:, None]
            run[tree][new_branches[2]] = run[tree][new_branches[2]] / \
                run[tree]["TNuE"][:, None]

        pdg_list = np.repeat(pdg_list, len(run[tree]["Event"])).reshape(
            len(pdg_list), len(run[tree]["Event"])).T
        run[tree]["TMarleySumPDG"] = pdg_list

        run = remove_branches(
            run, rm_branches, ["TMarleyMass"], tree=tree, debug=debug)

    output = f"\tMarley energy computation \t-> Done! ({new_branches})\n"
    return run, output


def compute_particle_energies(run, configs, params: Optional[dict] = {"NORM_TO_NUE", False}, trees: list[str] = ["Truth", "Reco"], rm_branches: bool = False, debug: bool = False):
    """
    This functions looks into "TMarleyPDG" branch and combines the corresponding "TMarleyE" entries to get a total energy for each daughter particle.
    """
    output = ""
    particles_pdg = {"Electron": 11, "Gamma": 22,
                     "Neutron": 2112, "Neutrino": 12, "Proton": 2212}
    particles_mass = {particle: values for particle, values in zip(particles_pdg.keys(
    ), [Particle.from_pdgid(particles_pdg[particle]).mass for particle in particles_pdg])}
    particles_mass["Neutrino"] = 0
    prticles_pdg_mass = {particles_pdg[particle]: particles_mass[particle]
                         for particle in particles_pdg}

    new_branches = list(particles_pdg.keys())
    for tree, particle in product(trees, particles_pdg):
        run[tree][f"{particle}E"] = np.zeros(
            len(run[tree]["Event"]), dtype=np.float32)
        run[tree][f"{particle}K"] = np.zeros(
            len(run[tree]["Event"]), dtype=np.float32)
        if len(run[tree]["TMarleyPDG"][0]) > 0:
            run[tree]["MarleyMass"] = np.vectorize(
                prticles_pdg_mass.get)(run[tree]["TMarleyPDG"])
        else:
            run[tree]["MarleyMass"] = np.zeros(
                (len(run[tree]["TMarleyPDG"]), len(run[tree]["TMarleyPDG"][0])), dtype=bool)
        run[tree]["MarleyMass"] = np.nan_to_num(
            run[tree]["MarleyMass"], nan=0.0, posinf=0.0, neginf=0.0)

    for config, tree in product(configs, trees):
        info, params, output = get_param_dict(
            f"{root}/config/{config}/{config}", params, output, debug=debug)
        idx = np.where(
            (np.asarray(run[tree]["Geometry"]) == info["GEOMETRY"])
            * (np.asarray(run[tree]["Version"]) == info["VERSION"])
        )
        for particle in particles_pdg:
            # Create the MarleyMass branch from the TMarleyPDG branch according to the particles_mass dictionary
            run[tree][f"{particle}E"][idx] = np.sum(
                run[tree]["TMarleyE"][idx]
                * np.array(run[tree]["TMarleyPDG"][idx]
                           == particles_pdg[particle]) * np.array(run[tree]["TMarleyMother"][idx] == 0),
                axis=1,
            )

            run[tree][f"{particle}K"][idx] = np.sum(
                np.subtract(run[tree]["TMarleyE"][idx],
                            run[tree]["MarleyMass"][idx])
                * np.array(run[tree]["TMarleyPDG"][idx]
                           == particles_pdg[particle]) * np.array(run[tree]["TMarleyMother"][idx] == 0),
                axis=1,
            )

            if params["NORM_TO_NUE"]:
                for particle in particles_pdg:
                    run[tree][f"{particle}E"][idx] = run[tree][f"{particle}E"][idx] / \
                        run[tree]["TNuE"][idx]
                    run[tree][f"{particle}K"][idx] = run[tree][f"{particle}K"][idx] / \
                        run[tree]["TNuE"][idx]

    run = remove_branches(run, rm_branches, [], debug=debug)
    output = f"\tParticle energy combination \t-> Done! ({new_branches})\n"
    return run, output


def compute_opflash_matching(
    run,
    configs,
    params: Optional[dict] = None,
    rm_branches: bool = False,
    output: Optional[str] = None,
    debug=False,
):
    """
    Match the reconstructed events with selected OpFlash candidates.
    """
    # New branches
    new_branches = [
        "FlashMathedIdx",
        "FlashMatched",
        "AssFlashIdx",
        "MatchedOpFlashTime",
        "MatchedOpFlashPE",
        "MatchedOpFlashR",
        "DriftTime",
        "AdjClDriftTime",
    ]
    run["Reco"][new_branches[0]] = np.zeros(
        (len(run["Reco"]["Event"]), len(run["Reco"]["AdjOpFlashR"][0])), dtype=bool
    )
    run["Reco"][new_branches[1]] = np.zeros(
        len(run["Reco"]["Event"]), dtype=bool)
    run["Reco"][new_branches[2]] = np.zeros(
        len(run["Reco"]["Event"]), dtype=int)
    run["Reco"][new_branches[3]] = np.zeros(
        len(run["Reco"]["Event"]), dtype=np.float32)
    run["Reco"][new_branches[4]] = np.zeros(
        len(run["Reco"]["Event"]), dtype=np.float32)
    run["Reco"][new_branches[5]] = np.zeros(
        len(run["Reco"]["Event"]), dtype=np.float32)
    run["Reco"][new_branches[6]] = np.zeros(
        len(run["Reco"]["Event"]), dtype=np.float32)
    run["Reco"][new_branches[7]] = np.zeros(
        (len(run["Reco"]["Event"]), len(run["Reco"]["AdjClTime"][0])), dtype=np.float32
    )
    for config in configs:
        info, params, output = get_param_dict(
            f"{root}/config/{config}/{config}", params, output, debug=debug)
        idx = np.where(
            (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
            * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
        )
        # If run["Reco"]["AdjClTime"][idx][0] empty skip the config:
        if run["Reco"]["AdjOpFlashTime"][idx].sum() == 0:
            continue

        # Select all FlashMatch candidates
        max_r_filter = run["Reco"]["AdjOpFlashR"][idx] < params["MAX_FLASH_R"]
        min_pe_filter = run["Reco"]["AdjOpFlashPE"][idx] > params["MIN_FLASH_PE"]
        signal_nan_filter = (run["Reco"]["AdjOpFlashSignal"][idx]
                             > 0) * (run["Reco"]["AdjOpFlashSignal"][idx] < np.inf)
        # max_ratio_filter = run["Reco"]["AdjOpFlashPE"][idx] > 3000 * run["Reco"]["AdjOpFlashMaxPE"][idx] / run["Reco"]["AdjOpFlashPE"][idx]

        converted_array = reshape_array(
            run["Reco"]["Time"][idx], len(
                run["Reco"]["AdjOpFlashTime"][idx][0])
        )
        # repeated_array = np.repeat(
        #     run["Reco"]["Time"][idx], len(
        #         run["Reco"]["AdjOpFlashTime"][idx][0])
        # )
        # converted_array = np.reshape(
        #     repeated_array, (-1, len(run["Reco"]["AdjOpFlashTime"][idx][0]))
        # )
        max_drift_filter = (
            np.abs(converted_array - 2 * run["Reco"]["AdjOpFlashTime"][idx])
            < params["MAX_DRIFT_FACTOR"] * info["EVENT_TICKS"]
        )
        run["Reco"]["FlashMathedIdx"][idx] = (
            (max_r_filter) * (min_pe_filter) *
            (max_drift_filter) * (signal_nan_filter)
        )

        # If at least one candidate is found, mark the event as matched and select the best candidate
        run["Reco"]["FlashMatched"][idx] = (
            np.sum(run["Reco"]["FlashMathedIdx"][idx], axis=1) > 0
        )
        run["Reco"]["AssFlashIdx"][idx] = np.argmax(
            run["Reco"]["AdjOpFlashSignal"][idx] *
            run["Reco"]["FlashMathedIdx"][idx],
            axis=1,
        )

        # Compute the drift time and the matched PE
        run["Reco"]["MatchedOpFlashTime"][idx] = run["Reco"]["AdjOpFlashTime"][
            idx[0], run["Reco"]["AssFlashIdx"][idx]
        ]
        run["Reco"]["MatchedOpFlashPE"][idx] = run["Reco"]["AdjOpFlashPE"][
            idx[0], run["Reco"]["AssFlashIdx"][idx]
        ]
        run["Reco"]["MatchedOpFlashR"][idx] = run["Reco"]["AdjOpFlashR"][
            idx[0], run["Reco"]["AssFlashIdx"][idx]
        ]
        run["Reco"]["DriftTime"][idx] = (
            run["Reco"]["Time"][idx] - 2 *
            run["Reco"]["MatchedOpFlashTime"][idx]
        )
        run["Reco"]["AdjClDriftTime"][idx] = (
            run["Reco"]["AdjClTime"][idx]
            - 2 * run["Reco"]["MatchedOpFlashTime"][idx][:, np.newaxis]
        )

    run = remove_branches(
        run, rm_branches, ["FlashMathedIdx", "AssFlashIdx"], debug=debug
    )
    output = f"\tOpFlash matching \t\t-> Done! ({new_branches})\n"
    return run, output


def compute_cluter_drift(
    run: dict, configs: dict[str, list[str]], params: Optional[dict] = None, rm_branches: bool = False, debug: bool = False
):
    """
    Compute the reconstructed X position of the events in the run.
    """
    new_branches = ["RecoX", "AdjCldT", "AdjClRecoX"]
    run["Reco"][new_branches[0]] = np.zeros(len(run["Reco"]["Event"]))
    run["Reco"][new_branches[1]] = np.zeros(
        (len(run["Reco"]["Event"]), len(run["Reco"]["AdjClTime"][0])), dtype=np.float32
    )
    run["Reco"][new_branches[2]] = np.zeros(
        (len(run["Reco"]["Event"]), len(run["Reco"]["AdjClTime"][0])), dtype=np.float32
    )

    for config in configs:
        info, params, output = get_param_dict(
            f"{root}/config/{config}/{config}", params, output, debug=debug)
        idx = np.where(
            (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
            * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
        )

        converted_array = reshape_array(
            run["Reco"]["Time"][idx], len(run["Reco"]["AdjClTime"][idx][0]))

        # repeated_array = np.repeat(
        #     run["Reco"]["Time"][idx], len(run["Reco"]["AdjClTime"][idx][0])
        # )
        # converted_array = np.reshape(
        #     repeated_array, (-1, len(run["Reco"]["AdjClTime"][idx][0]))
        # )
        run["Reco"]["AdjCldT"][idx] = run["Reco"]["AdjClTime"][idx] - converted_array

        if info["GEOMETRY"] == "hd":
            tpc_filter = (run["Reco"]["TPC"]) % 2 == 0
            plus_idx = np.where(
                (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
                * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
                * (tpc_filter)
            )
            run["Reco"]["RecoX"][plus_idx] = (
                abs(run["Reco"][params["DEFAULT_RECOX_TIME"]][plus_idx])
                * (info["DETECTOR_SIZE_X"] / 2)
                / info["EVENT_TICKS"]
            )

            tpc_filter = (run["Reco"]["TPC"]) % 2 == 1
            mins_idx = np.where(
                (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
                * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
                * (tpc_filter)
            )
            run["Reco"]["RecoX"][mins_idx] = (
                -abs(run["Reco"][params["DEFAULT_RECOX_TIME"]][mins_idx])
                * (info["DETECTOR_SIZE_X"] / 2)
                / info["EVENT_TICKS"]
            )

            converted_array = reshape_array(
                run["Reco"]["RecoX"], len(run["Reco"]["AdjClTime"][0]))

            # repeated_array = np.repeat(
            #     run["Reco"]["RecoX"], len(run["Reco"]["AdjClTime"][0])
            # )
            # converted_array = np.reshape(
            #     repeated_array, (-1, len(run["Reco"]["AdjClTime"][0]))
            # )
            run["Reco"]["AdjClRecoX"][plus_idx] = (
                run["Reco"]["AdjCldT"][plus_idx]
                * (info["DETECTOR_SIZE_X"] / 2)
                / info["EVENT_TICKS"]
                + converted_array[plus_idx]
            )
            run["Reco"]["AdjClRecoX"][mins_idx] = (
                -run["Reco"]["AdjCldT"][mins_idx]
                * (info["DETECTOR_SIZE_X"] / 2)
                / info["EVENT_TICKS"]
                + converted_array[mins_idx]
            )

        if info["GEOMETRY"] == "vd":
            run["Reco"]["RecoX"][idx] = (
                -abs(run["Reco"][params["DEFAULT_RECOX_TIME"]][idx])
                * info["DETECTOR_SIZE_X"]
                / info["EVENT_TICKS"]
                + info["DETECTOR_SIZE_X"] / 2
            )

            converted_array = reshape_array(
                run["Reco"]["RecoX"][idx], len(run["Reco"]["AdjClTime"][idx][0]))

            run["Reco"]["AdjClRecoX"][idx] = (
                run["Reco"]["AdjCldT"][idx]
                * info["DETECTOR_SIZE_X"]
                / info["EVENT_TICKS"]
            ) + converted_array

    run = remove_branches(run, rm_branches, ["AdjCldT"], debug=debug)
    output = f"\tComputed RecoX \t\t\t-> Done! ({new_branches})\n"
    return run, output


def compute_cluster_charge(
    run,
    configs: dict[str, list[str]],
    params: Optional[dict] = None,
    rm_branches: bool = False,
    output: Optional[str] = None,
    debug=False,
):
    """
    Correct the charge of the events in the run according to the correction file.
    """
    # New branches
    new_branches = ["ElectronCharge", "ElectronTime"]
    for branch in new_branches:
        run["Reco"][branch] = np.ones(len(run["Reco"]["Event"]))

    run["Reco"]["AdjClNum"] = np.sum(run["Reco"]["AdjClCharge"] != 0, axis=1)

    for config in configs:
        info, params, output = get_param_dict(
            f"{root}/config/{config}/{config}", params, output, debug=debug)
        idx = np.where(
            (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
            * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
        )
        run["Reco"]["ElectronCharge"][idx] = run["Reco"]["Charge"][idx] + np.sum(
            np.where((run["Reco"]["AdjClMainPDG"][idx] == 11),
                     run["Reco"]["AdjClCharge"][idx], 0),
            axis=1
        )
        # Compute the average weighted time of the clusters
        run["Reco"]["ElectronTime"][idx] = ((run["Reco"]["Charge"][idx] *
                                             run["Reco"]["Time"][idx]) + np.sum(
            np.where((run["Reco"]["AdjClMainPDG"][idx] == 11),
                     run["Reco"]["AdjClCharge"][idx] * run["Reco"]["AdjClTime"][idx], 0),
            axis=1
        )) / run["Reco"]["ElectronCharge"][idx]

    run = remove_branches(
        run, rm_branches, [], debug=debug
    )
    output = f"\tClutser charge computation\t-> Done! ({new_branches})\n"
    return run, output


def compute_cluster_energy(
    run: dict,
    configs: dict,
    params: Optional[dict] = None,
    rm_branches: bool = False,
    output: Optional[str] = None,
    debug: bool = False,
) -> tuple[dict, str]:
    """
    Correct the charge of the events in the run according to the correction file.
    """
    def calibration_func(x, a, b, c, d):
        return a*np.exp(-b*x)+c/(1+np.exp(-d*x))

    # New branches
    new_branches = ["Correction",
                    "CorrectedCharge", "CorrectionFactor", "Energy"]
    for branch in new_branches:
        run["Reco"][branch] = np.zeros(
            len(run["Reco"]["Event"]), dtype=np.float32)

    new_vector_branches = [
        "AdjClCorrection", "AdjClCorrectedCharge", "AdjClCorrectionFactor", "AdjClEnergy"]

    for branch in new_vector_branches:
        run["Reco"][branch] = np.ones(
            (len(run["Reco"]["Event"]), len(run["Reco"]["AdjClCharge"][0])), dtype=np.float32
        )

    for config in configs:
        info, params, output = get_param_dict(
            f"{root}/config/{config}/{config}", params, output, debug=debug)
        idx = np.where(
            (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
            * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
        )

        corr_info = json.load(open(
            f"{root}/config/{config}/Marley/{config}_calib/{config}_electroncharge_correction.json", "r"))

        drift_popt = [corr_info["CHARGE_AMP"], corr_info["ELECTRON_TAU"]]
        corr_popt = [corr_info["CORRECTION_AMP"],
                     corr_info["CORRECTION_DECAY"], corr_info["CORRECTION_CONST"], corr_info["CORRECTION_SIGMOID"]]

        # Divide the filter idx into two groups, one with the random correction and one without
        random_idx = idx[0][:int(
            len(idx[0])*params["RANDOM_CORRECTION_RATIO"])]
        default_idx = idx[0][int(
            len(idx[0])*params["RANDOM_CORRECTION_RATIO"]):]

        run["Reco"]["RandomEnergyTime"] = np.random.normal(
            0, info["EVENT_TICKS"], len(run["Reco"]["Event"]))
        run["Reco"]["RandomAdjClEnergyTime"] = np.random.normal(
            0, info["EVENT_TICKS"], (len(run["Reco"]["Event"]), len(
                run["Reco"]["AdjClCharge"][0]))
        )
        if params["RANDOM_CORRECTION_RATIO"] > 0:
            rprint(
                f"[yellow]--> Applying random correction {100*params['RANDOM_CORRECTION_RATIO']:.1f}% to {branch}[/yellow]")

        for branch, rand_branch, default_param in zip(["Correction", "AdjClCorrection"], ["RandomEnergyTime", "RandomAdjClEnergyTime"], [params["DEFAULT_ENERGY_TIME"], params["DEFAULT_ADJCL_ENERGY_TIME"]]):

            run["Reco"][branch][default_idx] = np.exp(
                np.abs(run["Reco"][default_param]
                       [default_idx]) / drift_popt[1]
            )

            run["Reco"][branch][random_idx] = np.exp(
                np.abs(run["Reco"][rand_branch]
                       [random_idx]) / drift_popt[1]
            )

        run["Reco"]["CorrectedCharge"][idx] = run["Reco"]["Charge"][idx] * \
            run["Reco"]["Correction"][idx]

        run["Reco"]["CorrectionFactor"][idx] = calibration_func(
            run["Reco"]["NHits"][idx], *corr_popt)

        run["Reco"]["Energy"][idx] = run["Reco"]["CorrectedCharge"][idx] / \
            run["Reco"]["CorrectionFactor"][idx]

        run["Reco"]["AdjClCorrectedCharge"][idx] = run["Reco"]["AdjClCharge"][idx] * \
            run["Reco"]["AdjClCorrection"][idx]

        run["Reco"]["AdjClCorrectionFactor"][idx] = corr_popt[0] * np.exp(
            -run["Reco"]["AdjClNHit"][idx] / corr_popt[1]) + corr_popt[2]

        run["Reco"]["AdjClEnergy"][idx] = run["Reco"]["AdjClCorrectedCharge"][idx] / \
            run["Reco"]["AdjClCorrectionFactor"][idx]

    run = remove_branches(
        run, rm_branches, new_branches[:-1]+new_vector_branches[:-1], debug=debug
    )
    output = f"\tClutser energy computation\t-> Done! ({new_branches+new_vector_branches})\n"
    return run, output


def compute_cluster_calibration(run, configs: dict[str, list[str]], params: Optional[dict] = None, rm_branches: bool = False, output: Optional[str] = None, debug=False):
    """
    Correct the charge of the events in the run according to the correction file.
    """
    new_branches = ["Energy"]
    for config in configs:
        info, params, output = get_param_dict(
            f"{root}/config/{config}/{config}", params, output, debug=debug)
        idx = np.where(
            (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
            * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
        )

        corr_info = json.load(open(
            f"{root}/config/{config}/Marley/{config}_calib/{config}_charge_calibration.json", "r"))

        corr_popt = [corr_info["SLOPE"], corr_info["INTERCEPT"]]
        run["Reco"]["Energy"][idx] = run["Reco"]["Energy"][idx] - corr_popt[1] * \
            corr_popt[0]

    run = remove_branches(
        run, rm_branches, [], debug=debug
    )
    output = f"\tClutser energy computation\t-> Done! ({new_branches})\n"
    return run, output


def compute_total_energy(run: dict, configs: dict[str, list[str]], params: Optional[dict] = None, rm_branches: bool = False, output: Optional[str] = None, debug=False):
    """
    Compute the total energy of the events in the run.
    """
    output = ""
    new_branches = ["SelectedAdjClNum", "RecoEnergy", "TotalEnergy", "SelectedEnergy", "TotalAdjClEnergy",
                    "SelectedAdjClEnergy", "SelectedMaxAdjClEnergy", "SelectedAdjClEnergyRatio", "Discriminant",]
    for branch in new_branches:
        run["Reco"][branch] = np.zeros(len(run["Reco"]["Event"]))

    for config in configs:
        info, params, output = get_param_dict(
            f"{root}/config/{config}/{config}", params, output, debug=debug)
        idx = np.where(
            (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
            * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
        )

        run["Reco"]["TotalAdjClEnergy"][idx] = np.sum(
            run["Reco"]["AdjClEnergy"][idx], axis=1
        )

        selected_filter = (run["Reco"]["AdjClR"][idx] < params["MIN_BKG_R"]) + \
            (run["Reco"]["AdjClCharge"][idx] > params["MAX_BKG_CHARGE"])

        if debug:
            output += \
                f"[yellow]\t***Selected filter for energy computation excludes {100*((np.sum(run['Reco']['AdjClNum'][idx])-np.sum(~selected_filter))/np.sum(run['Reco']['AdjClNum'][idx])):.1f}%\n[/yellow]"

        run["Reco"]["SelectedAdjClNum"][idx] = np.sum(selected_filter, axis=1)
        run["Reco"]["SelectedAdjClEnergy"][idx] = np.sum(
            run["Reco"]["AdjClEnergy"][idx], where=selected_filter, axis=1)
        run["Reco"]["SelectedMaxAdjClEnergy"][idx] = np.max(
            run["Reco"]["AdjClEnergy"][idx], where=selected_filter, axis=1, initial=0)

        run["Reco"]["TotalEnergy"][idx] = run["Reco"]["Energy"][idx] + \
            run["Reco"]["TotalAdjClEnergy"][idx]
        run["Reco"]["SelectedEnergy"][idx] = run["Reco"]["Energy"][idx] + \
            run["Reco"]["SelectedAdjClEnergy"][idx]
        run["Reco"]["SelectedAdjClEnergyRatio"][idx] = run["Reco"]["SelectedAdjClEnergy"][idx] / \
            run["Reco"]["Energy"][idx]

    run = remove_branches(
        run, rm_branches, ["TotalAdjClEnergy", "SelectedAdjClEnergy"], debug=debug)
    output += f"\tTotal energy computation \t-> Done! ({new_branches})\n"
    return run, output


def compute_reco_energy(run, configs, params: Optional[dict] = None, rm_branches: bool = False, output: Optional[str] = None, debug=False):
    features = get_default_info(root, "ML_FEATURES")
    new_branches = ["RecoEnergy", "Upper", "Lower"]
    for branch in new_branches:
        run["Reco"][branch] = np.zeros(len(run["Reco"]["Event"]))

    for config in configs:
        info, params, output = get_param_dict(
            f"{root}/config/{config}/{config}", params, output, debug=debug)
        for name in configs[config]:
            idx = np.where(
                (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
                * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
                * (np.asarray(run["Reco"]["Name"]) == name)
            )

            # Save the trained model to a file so it can be used later using pickle
            if name != "Marley":
                name = "wbkg"
            with open(f"{root}/config/{config}/{name}/models/{config}_{name}_random_forest_discriminant.pkl", 'rb') as model_file:
                rf_classifier = pickle.load(model_file)

            discriminant_info = json.load(
                open(
                    f"{root}/config/{config}/{name}/{config}_calib/{config}_discriminant_calibration.json",
                    "r",
                )
            )
            def upper_func(x): return x - discriminant_info["UPPER"]["OFFSET"]
            def lower_func(x): return x - discriminant_info["LOWER"]["OFFSET"]

            thld = discriminant_info["DISCRIMINANT_THRESHOLD"]

            run["Reco"]["Upper"][idx] = np.asarray(
                run["Reco"]["ElectronK"][idx] > run["Reco"]["TNuE"][idx] + thld, dtype=bool)
            run["Reco"]["Lower"][idx] = np.asarray(
                run["Reco"]["ElectronK"][idx] < run["Reco"]["TNuE"][idx] + thld, dtype=bool)

            df = npy2df(run, "Reco", branches=features+["Primary", "Generator", "TNuE", "NHits", "Upper", "Lower"],
                        debug=debug)

            df['ML'] = rf_classifier.predict(df[features])
            df['Discriminant'] = rf_classifier.predict_proba(df[features])[
                :, 1]
            upper_idx = df["Discriminant"] >= discriminant_info["ML_THRESHOLD"]
            lower_idx = df["Discriminant"] < discriminant_info["ML_THRESHOLD"]

            df.loc[upper_idx, "RecoEnergy"] = upper_func(
                df.loc[upper_idx, "Energy"])
            df.loc[lower_idx, "RecoEnergy"] = lower_func(
                df.loc[lower_idx, "Energy"])
            run["Reco"]["RecoEnergy"][idx] = df["RecoEnergy"].values

    run = remove_branches(
        run, rm_branches, [], debug=debug)
    output = f"\tReco energy computation \t-> Done! ({new_branches})\n"
    return run, output


def compute_energy_calibration(run, configs, params: Optional[dict] = None, rm_branches: bool = False, output: Optional[str] = None, debug=False):
    output = ""
    new_branches = ["RecoEnergy",
                    "SelectedEnergy", "TotalEnergy"]

    for config in configs:
        info, params, output = get_param_dict(
            f"{root}/config/{config}/{config}", params, output, debug=debug)
        for name in configs[config]:
            idx = np.where(
                (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
                * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
                * (np.asarray(run["Reco"]["Name"]) == name)
            )

            try:
                reco_info = json.load(
                    open(
                        f"{root}/config/{config}/{name}/{config}_calib/{config}_{name}_energy_calibration.json",
                        "r",
                    )
                )
                if debug:
                    output += f"[yellow]\t***Applying energy calibration from {name}[/yellow]\n"

            except FileNotFoundError:
                reco_info = json.load(
                    open(
                        f"{root}/config/{config}/Marley/{config}_calib/{config}_Marley_energy_calibration.json",
                        "r",
                    )
                )
                output += f"[yellow]\t***Applying default energy calibration from Marley[/yellow]\n"

            for energy in ["Reco", "Selected", "Total"]:
                run["Reco"][f"{energy}Energy"][idx] = (run["Reco"][f"{energy}Energy"][idx] -
                                                       reco_info[energy.upper()]["INTERSECTION"]) / reco_info[energy.upper()]["ENERGY_AMP"]

    run = remove_branches(
        run, rm_branches, [], debug=debug)
    output += f"\tReco energy calibration \t-> Done! ({new_branches})\n"
    return run, output


def compute_opflash_advanced(run, configs, params: Optional[dict] = None, rm_branches: bool = False, output: Optional[str] = None, debug=False):
    """
    Compute the OpFlash variables for the events in the run.
    """
    # New branches
    new_branches = ["AdjOpFlashMaxPE", "AdjOpFlashNHit", "AdjOpFlashPE", "AdjOpFlashR", "AdjOpFlashRecoY",
                    "AdjOpFlashRecoZ", "AdjOpFlashSignal", "AdjOpFlashRatio", "AdjOpFlashErrorY", "AdjOpFlashErrorZ"]

    run["Reco"]["AdjOpFlashNum"] = np.sum(
        run["Reco"]["AdjOpFlashR"] != 0, axis=1)
    # run["Reco"]["AdjOpFlashR"] = np.where(run["Reco"]["AdjOpFlashR"] == 0, np.nan, run["Reco"]["AdjOpFlashR"])
    run["Reco"]["AdjOpFlashRatio"] = run["Reco"]["AdjOpFlashMaxPE"] / \
        run["Reco"]["AdjOpFlashPE"]
    run["Reco"]["AdjOpFlashSignal"] = run["Reco"]["AdjOpFlashNHit"] * \
        run["Reco"]["AdjOpFlashPE"]/run["Reco"]["AdjOpFlashR"]
    # Set AdjOpFlashSignal to 0 if it is Nan
    run["Reco"]["AdjOpFlashSignal"] = np.where(
        np.isnan(run["Reco"]["AdjOpFlashSignal"]), 0, run["Reco"]["AdjOpFlashSignal"])
    # If AdjOpFlashRatio is 0 set it to Nan
    run["Reco"]["AdjOpFlashRatio"] = np.where(
        run["Reco"]["AdjOpFlashRatio"] == 0, np.nan, run["Reco"]["AdjOpFlashRatio"])
    run["Reco"]["AdjOpFlashErrorY"] = run["Reco"]["AdjOpFlashRecoY"] - \
        reshape_array(run["Reco"]["TNuY"], len(
            run["Reco"]["AdjOpFlashRecoY"][0]))
    run["Reco"]["AdjOpFlashErrorZ"] = run["Reco"]["AdjOpFlashRecoZ"] - \
        reshape_array(run["Reco"]["TNuZ"], len(
            run["Reco"]["AdjOpFlashRecoZ"][0]))

    output = f"\tOpFlash variables computation \t-> Done! ({new_branches})\n"
    return run, output


def compute_ophit_basic(run, configs, params: Optional[dict] = None, rm_branches: bool = False, output: Optional[str] = None, debug=False):
    """
    Compute the OpFlash variables for the events in the run.
    """
    # Generate repeated arrays of TNuX, TNuY, TNuZ
    converted_y = reshape_array(
        run["Truth"]["TNuY"], len(run["Truth"]["OpHitY"][0]))
    # repeated_y = np.repeat(run["Truth"]["TNuY"],
    #                        len(run["Truth"]["OpHitY"][0]))
    # converted_y = np.reshape(repeated_y, (-1, len(run["Truth"]["OpHitY"][0])))
    conveterd_z = reshape_array(
        run["Truth"]["TNuZ"], len(run["Truth"]["OpHitZ"][0]))
    # repeated_z = np.repeat(run["Truth"]["TNuZ"],
    #                        len(run["Truth"]["OpHitZ"][0]))
    # converted_z = np.reshape(repeated_z, (-1, len(run["Truth"]["OpHitZ"][0])))

    # New branches
    new_branches = ["OpHitR"]
    for branch in new_branches:
        run["Truth"][branch] = np.zeros(
            (len(run["Truth"]["Event"]), len(run["Truth"]["OpHitT"])), dtype=np.float32)

    # Create OpHitR array
    run["Truth"]["OpHitR"] = np.sqrt(np.power(
        converted_y-run["Truth"]["OpHitY"], 2) + np.power(converted_z-run["Truth"]["OpHitZ"], 2), dtype=np.float32)

    output = f"\tBasic OpHit variables computation \t-> Done! ({new_branches})\n"
    return run, output


def compute_ophit_advanced(run, configs, params: Optional[dict] = None, rm_branches: bool = False, output: Optional[str] = None, debug=False):
    """
    Compute the OpFlash variables for the events in the run.
    """
    # Generate repeated arrays of TNuX, TNuY, TNuZ
    converted_x = reshape_array(
        run["Truth"]["TNuX"], len(run["Truth"]["OpHitX"][0]))
    # repeated_x = np.repeat(run["Truth"]["TNuX"],
    #                        len(run["Truth"]["OpHitX"][0]))
    # converted_x = np.reshape(repeated_x, (-1, len(run["Truth"]["OpHitX"][0])))
    converted_y = reshape_array(
        run["Truth"]["TNuY"], len(run["Truth"]["OpHitY"][0]))
    # repeated_y = np.repeat(run["Truth"]["TNuY"],
    #                        len(run["Truth"]["OpHitY"][0]))
    # converted_y = np.reshape(repeated_y, (-1, len(run["Truth"]["OpHitY"][0])))
    converted_z = reshape_array(
        run["Truth"]["TNuZ"], len(run["Truth"]["OpHitZ"][0]))
    # repeated_z = np.repeat(run["Truth"]["TNuZ"],
    #                        len(run["Truth"]["OpHitZ"][0]))
    # converted_z = np.reshape(repeated_z, (-1, len(run["Truth"]["OpHitZ"][0])))

    flash_id_list = np.unique(run["Truth"]["OpHitFlashID"])
    # New branches
    new_branches = ["OpFlashRefPE", "OpFlashResidual", "OpFlashTime",
                    "OpFlashPur", "OpFlashSignal", "OpFlashNHit", "OpFlashPE"]
    for branch in new_branches:
        run["Truth"][branch] = np.zeros(
            (len(run["Truth"]["Event"]), len(flash_id_list)), dtype=np.float32)

    # Make empty array for OpHitResidual
    run["Truth"]["OpHitResidual"] = np.zeros(
        (len(run["Truth"]["Event"]), len(run["Truth"]["OpHitPE"][0])), dtype=np.float32)

    # Make each entry in the OpFlashID equal to its idx
    run["Truth"]["OpFlashID"] = np.ones(
        (len(run["Truth"]["Event"]), len(flash_id_list)), dtype=int)
    run["Truth"]["OpFlashID"] = run["Truth"]["OpFlashID"] * \
        np.arange(len(flash_id_list))
    # print(run["Truth"]["OpFlashID"])

    # Change all nans in run["Truth"]["OpHitPur"] for 0
    run["Truth"]["OpHitPur"] = np.nan_to_num(
        run["Truth"]["OpHitPur"], nan=0.0, posinf=0.0, neginf=0.0)
    run["Truth"]["OpHitDecay"] = np.power(converted_x, 2) / (np.power(converted_x, 2) + np.power(
        converted_y - run["Truth"]["OpHitY"], 2) + np.power(converted_z - run["Truth"]["OpHitZ"], 2))
    run["Truth"]["OpHitRefPE"] = np.array(
        run["Truth"]["OpHitPE"] * run["Truth"]["OpHitPur"])

    for flash_id in flash_id_list:
        jdx = np.where(run["Truth"]["OpFlashID"] == flash_id)
        flash_id_filter = np.asarray(run["Truth"]["OpHitFlashID"] == flash_id)
        event_id_count = np.sum(flash_id_filter, axis=1)
        run["Truth"]["OpFlashRefPE"][jdx] = np.sum(
            run["Truth"]["OpHitRefPE"] * flash_id_filter, axis=1) / event_id_count
        run["Truth"]["OpHitRefPE"][flash_id_filter] = np.repeat(
            run["Truth"]["OpFlashRefPE"][jdx], event_id_count)
        run["Truth"]["OpFlashNHit"][jdx] = np.sum(flash_id_filter, axis=1)
        run["Truth"]["OpFlashPE"][jdx] = np.sum(
            run["Truth"]["OpHitPE"] * flash_id_filter, axis=1) / event_id_count

        run["Truth"]["OpFlashResidual"][jdx] = np.mean(np.power((run["Truth"]["OpHitPE"] - run["Truth"]["OpHitRefPE"] * run["Truth"]
                                                                 ["OpHitDecay"]) * flash_id_filter, 2), axis=1)/run["Truth"]["OpFlashNHit"][jdx]/run["Truth"]["OpFlashPE"][jdx]
        run["Truth"]["OpHitResidual"][flash_id_filter] = np.repeat(
            run["Truth"]["OpFlashResidual"][jdx], event_id_count)
        run["Truth"]["OpFlashTime"][jdx] = np.sum((run["Truth"]["OpHitT"] * run["Truth"]["OpHitPE"])
                                                  * flash_id_filter, axis=1) / np.sum(run["Truth"]["OpHitPE"] * flash_id_filter, axis=1)
        # run["Truth"]["OpFlashPur"][jdx] = np.sum(run["Truth"]["OpHitPur"][idx] * run["Truth"]["OpHitPE"][idx], axis=1) / np.sum(run["Truth"]["OpHitPE"][idx], axis=1)
    run["Truth"]["OpFlashSignal"] = (
        abs(run["Truth"]["OpFlashTime"]) < 5) == True

    output = f"\tAdvanced OpHit variables computation \t-> Done! ({new_branches})\n"
    return run, output


def compute_true_drift(run, configs, params: Optional[dict] = None, rm_branches: bool = False, output: Optional[str] = None, debug=False):
    """
    Compute the true drift time of the events in the run.
    """
    # New branches
    new_branches = ["TrueDriftTime", "AdjClTrueDriftTime"]
    run["Reco"]["TrueDriftTime"] = np.zeros(
        len(run["Reco"]["Event"]), dtype=np.float32)
    run["Reco"]["AdjClTrueDriftTime"] = np.zeros(
        (len(run["Reco"]["Event"]), len(run["Reco"]["AdjClTime"][0])), dtype=np.float32
    )

    for config in configs:
        info, params, output = get_param_dict(
            f"{root}/config/{config}/{config}", params, output, debug=debug)
        idx = np.where(
            (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
            * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
        )
        if info["GEOMETRY"] == "hd":
            run["Reco"]["TrueDriftTime"][idx] = abs(
                run["Reco"]["MainVertex"][idx, 0]
            ) * 2 * info["EVENT_TICKS"]/info["DETECTOR_SIZE_X"]
            run["Reco"]["AdjClTrueDriftTime"][idx] = abs(
                run["Reco"]["AdjClMainX"][idx]
            ) * 2 * info["EVENT_TICKS"]/info["DETECTOR_SIZE_X"]

        if info["GEOMETRY"] == "vd":
            run["Reco"]["TrueDriftTime"][idx] = (
                info["DETECTOR_SIZE_X"] - run["Reco"]["MainVertex"][idx, 0]) * 0.5 * info["EVENT_TICKS"] / info["DETECTOR_SIZE_X"]
            run["Reco"]["AdjClTrueDriftTime"][idx] = (
                info["DETECTOR_SIZE_X"] - run["Reco"]["AdjClMainX"][idx]) * 0.5 * info["EVENT_TICKS"] / info["DETECTOR_SIZE_X"]

    # Select all values bigger than 1e6 or smaller than 0 and set them to 0
    run["Reco"]["TrueDriftTime"] = np.where(
        (run["Reco"]["TrueDriftTime"] > 1e6) | (
            run["Reco"]["TrueDriftTime"] < 0),
        0,
        run["Reco"]["TrueDriftTime"],
    )
    run["Reco"]["AdjClTrueDriftTime"] = np.where(
        (run["Reco"]["AdjClTrueDriftTime"] > 1e6) | (
            run["Reco"]["AdjClTrueDriftTime"] < 0),
        0,
        run["Reco"]["AdjClTrueDriftTime"],
    )
    output = f"\tTrue drift time computation \t-> Done!\n"
    return run, output


def compute_adjcl_basics(run, configs, params: Optional[dict] = None, rm_branches=False, output: Optional[str] = None, debug=False):
    """
    Compute basic variables for the adjacent clusters

    Args:
        run: dictionary containing the TTree
        configs: dictionary containing the path to the configuration files for each geoemtry
        params: dictionary containing the parameters for the reco functions
        debug: print debug information
    """
    @ numba.njit
    def count_occurrences(arr, length):
        """
        Count the occurrences of each element in the array.

        Args:
            arr: array containing the elements
            length: length of the array
        """
        return [np.sum(arr == i) for i in range(length)]

    # New branches
    new_branches = ["AdjClNum", "TotalAdjClCharge", "MaxAdjClCharge",
                    "MeanAdjClCharge", "MeanAdjClR", "MeanAdjClTime"]

    for branch in new_branches:
        run["Reco"][branch] = np.zeros(
            len(run["Reco"]["Event"]), dtype=np.float32)

    new_vector_branches = ["AdjClSameGenNum",
                           "TotalAdjClSameGenCharge", "MaxAdjClSameGenCharge", "MeanAdjClSameGenCharge"]
    run["Reco"][new_vector_branches[0]] = np.zeros(
        len(run["Reco"]["Event"]), dtype=int)
    for branch in new_vector_branches[1:]:
        run["Reco"][branch] = np.zeros(
            len(run["Reco"]["Event"]), dtype=np.float32
        )

    run["Reco"]["AdjClGenNum"] = np.apply_along_axis(
        count_occurrences,
        arr=run["Reco"]["AdjClGen"],
        length=len(run["Reco"]["TruthPart"][0]) + 1,
        axis=1,
    )

    run["Reco"]["AdjClNum"] = np.sum(run["Reco"]["AdjClCharge"] != 0, axis=1)

    for config in configs:
        info, params, output = get_param_dict(
            f"{root}/config/{config}/{config}", params, output, debug=debug)
        idx = np.where(
            (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
            * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
        )
        run["Reco"]["TotalAdjClCharge"][idx] = np.sum(
            run["Reco"]["AdjClCharge"][idx], axis=1
        )
        run["Reco"]["MaxAdjClCharge"][idx] = np.max(
            run["Reco"]["AdjClCharge"][idx], axis=1
        )
        run["Reco"]["MeanAdjClCharge"][idx] = np.mean(
            run["Reco"]["AdjClCharge"][idx], axis=1
        )
        run["Reco"]["MeanAdjClR"][idx] = np.mean(
            run["Reco"]["AdjClR"][idx], axis=1
        )
        run["Reco"]["MeanAdjClTime"][idx] = np.mean(
            run["Reco"]["AdjClTime"][idx], axis=1
        )
        converted_array = reshape_array(
            run["Reco"]["Generator"][idx], len(run["Reco"]["AdjClGen"][idx][0]))

        # repeated_array = np.repeat(
        #     run["Reco"]["Generator"][idx], len(run["Reco"]["AdjClGen"][idx][0])
        # )
        # converted_array = np.reshape(
        #     repeated_array, (-1, len(run["Reco"]["AdjClGen"][idx][0]))
        # )

        gen_idx = converted_array == run["Reco"]["AdjClGen"][idx]
        run["Reco"]["AdjClSameGenNum"][idx] = np.sum(gen_idx, axis=1)
        run["Reco"]["TotalAdjClSameGenCharge"][idx] = np.sum(
            run["Reco"]["AdjClCharge"][idx] * gen_idx, axis=1
        )
        run["Reco"]["MaxAdjClSameGenCharge"][idx] = np.max(
            run["Reco"]["AdjClCharge"][idx] * gen_idx, axis=1
        )
        run["Reco"]["MeanAdjClSameGenCharge"][idx] = np.mean(
            run["Reco"]["AdjClCharge"][idx] * gen_idx, axis=1
        )

    run = remove_branches(run, rm_branches, [], debug=debug)
    output = f"\tAdjCl basic computation \t-> Done!\n"
    return run, output


def compute_adjcl_advanced(run, configs, params: Optional[dict] = None, rm_branches=False, output: Optional[str] = None, debug=False):
    """
    Compute the energy of the individual adjacent clusters based on the main calibration.

    Args:
        run: dictionary containing the TTree
        configs: dictionary containing the path to the configuration files for each geoemtry
        params: dictionary containing the parameters for the reco functions
        debug: print debug information
    """
    # New branches
    new_branches = ["TotalAdjClEnergy", "MaxAdjClEnergy"]
    for branch in new_branches:
        run["Reco"][branch] = np.zeros(
            len(run["Reco"]["Event"]), dtype=np.float32)

    run["Reco"]["AdjCldT"] = np.zeros(
        (len(run["Reco"]["Event"]), len(run["Reco"]["AdjClTime"][0])), dtype=np.float32)
    run["Reco"]["AdjClRelCharge"] = np.zeros(
        (len(run["Reco"]["Event"]), len(run["Reco"]["AdjClCharge"][0])), dtype=np.float32)
    run["Reco"]["AdjClChargePerHit"] = np.zeros(
        (len(run["Reco"]["Event"]), len(run["Reco"]["AdjClCharge"][0])), dtype=np.float32)

    for config in configs:
        info, params, output = get_param_dict(
            f"{root}/config/{config}/{config}", params, output, debug=debug)
        idx = np.where(
            (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
            * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
        )
        run["Reco"]["TotalAdjClEnergy"][idx] = np.sum(
            run["Reco"]["AdjClEnergy"][idx], axis=1
        )
        run["Reco"]["MaxAdjClEnergy"][idx] = np.max(
            run["Reco"]["AdjClEnergy"][idx], axis=1
        )
        converted_array_time = reshape_array(
            run["Reco"]["Time"][idx], len(run["Reco"]["AdjClTime"][idx][0]))
        # repeated_array_time = np.repeat(
        #     run["Reco"]["Time"][idx], len(run["Reco"]["AdjClTime"][idx][0])
        # )
        # converted_array_time = np.reshape(
        #     repeated_array_time, (-1, len(run["Reco"]["AdjClTime"][idx][0]))
        # )
        converted_array_nhits = reshape_array(
            run["Reco"]["NHits"][idx], len(run["Reco"]["AdjClNHit"][idx][0]))
        # repeated_array_nhits = np.repeat(
        #     run["Reco"]["NHits"][idx], len(run["Reco"]["AdjClNHit"][idx][0])
        # )
        # converted_array_nhits = np.reshape(
        #     repeated_array_nhits, (-1, len(run["Reco"]["AdjClNHit"][idx][0]))
        # )
        converted_array_charge = reshape_array(
            run["Reco"]["Charge"][idx], len(run["Reco"]["AdjClCharge"][idx][0]))
        # repeated_array_charge = np.repeat(
        #     run["Reco"]["Charge"][idx], len(run["Reco"]["AdjClCharge"][idx][0])
        # )
        # converted_array_charge = np.reshape(
        #     repeated_array_charge, (-1,
        #                             len(run["Reco"]["AdjClCharge"][idx][0]))
        # )
        run["Reco"]["AdjCldT"][idx] = run["Reco"]["AdjClTime"][idx] - \
            converted_array_time
        run["Reco"]["AdjClRelCharge"] = run["Reco"]["AdjClCharge"][idx] / \
            converted_array_charge
        run["Reco"]["AdjClChargePerHit"] = run["Reco"]["AdjClCharge"][idx] / \
            run["Reco"]["AdjClNHit"][idx]

    run = remove_branches(run, rm_branches, [], debug=debug)
    output += f"\tAdjCl energy computation \t-> Done! ({new_branches})\n"
    return run, output


def compute_filtered_run(run: dict, configs: dict[str, list[str]], params: Optional[dict] = None, debug: bool = False):
    """
    Function to filter all events in the run according to the filters defined in the params dictionary.
    """
    output = ""
    new_run = {}
    if type(params) != dict and params != None:
        rprint(f"[red]Params must be a dictionary![/red]")
        return run

    if params == None:
        if debug:
            rprint("[yellow]No filter applied![/yellow]")
        return run

    new_trees = run.keys()
    for tree in new_trees:
        new_run[tree] = {}
        branch_list = list(run[tree].keys())
        idx = np.zeros(len(run[tree]["Event"]), dtype=bool)

        # Make sure that only the entries that correspond to the correct geometry, version and name are selected
        for config in configs:
            info = json.load(
                open(f"{root}/config/{config}/{config}_config.json", "r"))
            for name in configs[config]:
                config_filter = (run[tree]["Geometry"] == info["GEOMETRY"]) & (
                    run[tree]["Version"] == info["VERSION"]) & (run[tree]["Name"] == name)
                idx = idx + config_filter
        if debug:
            output += f"From {len(run[tree]['Event'])} events, {len(idx)} have been selected by configs for {tree} tree.\n"
        for param in params:
            if param[0] != tree:
                continue

            if type(param) != tuple or len(param) != 2:
                rprint(
                    f"[red]ERROR: Filter must be a tuple or list of length 2![/red]")
                return run
            if type(params[param]) != tuple or len(params[param]) != 2:
                rprint(
                    f"[red]ERROR: Filter must be a tuple or list of length 2![/red]")
                return run

            if param[1] not in run[param[0]].keys():
                rprint(
                    f"[red]ERROR: Branch {param[1]} not found in the run![/red]")
                return run

            if params[param][0] == "bigger":
                idx = idx & (run[param[0]][param[1]] > params[param][1])
            elif params[param][0] == "smaller":
                idx = idx & (run[param[0]][param[1]] < params[param][1])
            elif params[param][0] == "equal":
                idx = idx & (run[param[0]][param[1]] == params[param][1])
            elif params[param][0] == "different":
                idx = idx & (run[param[0]][param[1]] != params[param][1])
            elif params[param][0] == "between":
                idx = idx & (run[param[0]][param[1]] > params[param][1][0]) & (
                    run[param[0]][param[1]] < params[param][1][1])
            elif params[param][0] == "outside":
                idx = idx & ((run[param[0]][param[1]] < params[param][1]
                              [0]) + (run[param[0]][param[1]] > params[param][1][1]))
            elif params[param][0] == "contains":
                idx = idx & np.array(
                    [params[param][1] in item for item in run[param[0]][param[1]]])
            if debug:
                output = output + \
                    f"-> {param[1]}: {params[param][0]} {params[param][1]}:\t{np.sum(idx):.1E} ({100*np.sum(idx)/len(run[tree]['Event']):.1f}%) events\n"

        jdx = np.where(idx == True)
        for branch in branch_list:
            try:
                new_run[tree][branch] = np.asarray(run[tree][branch])[jdx]
            except Exception as e:
                rprint(f"Error filtering {branch}: {e}")

    if output != "":
        rprint(output)
    return new_run


def get_param_dict(config_file: dict, in_params: Optional[dict] = None, output: Optional[str] = None, debug: bool = False):
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
                    output += (
                        output
                        + f"\t***Applying {param}: {params[param]} from the config file\n"
                    )
            else:
                params[param] = in_params[param]
                output += (
                    output
                    + f"\t***Applying {param}: {in_params[param]} from the input dictionary\n"
                )
        except KeyError:
            pass

    return info, params, output


def add_filter(filters, labels, this_filter, this_label, cummulative, debug=False):
    if cummulative:
        labels.append("All+" + this_label)
        filters.append((filters[-1]) * (this_filter))
    else:
        labels.append(this_label)
        filters.append((filters[0]) * this_filter)

    if debug:
        print("Filter " + this_label + " added -> Done!")
    return filters, labels


def compute_solarnuana_filters(
    run,
    configs,
    config,
    name,
    gen,
    filter_list,
    params={},
    cummulative=True,
    debug=False,
):
    """
    Compute the filters for the solar workflow computation.

    Args:
        run: dictionary containing the data.
        configs: dictionary containing the path to the configuration files for each geoemtry.
        config: string containing the name of the configuration.
        name: string containing the name of the reco.
        gen: string containing the name of the generator.
        filter_list: list of filters to be applied.
        params: dictionary containing the parameters for the reco functions.
        cummulative: boolean to apply the filters cummulative or not.
        debug: print debug information.

    Returns:
        filters: list of filters to be applied (each filter is a list of bools).
    """
    labels = []
    filters = []
    info = json.load(open(f"{root}/config/{config}/{config}_config.json", "r"))

    # Select filters to be applied to the data
    geo_filter = np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"]
    version_filter = np.asarray(run["Reco"]["Version"]) == info["VERSION"]
    name_filter = np.asarray(run["Reco"]["Name"]) == name
    gen_filter = np.asarray(run["Reco"]["Generator"]) == gen
    base_filter = (geo_filter) * (version_filter) * \
        (name_filter) * (gen_filter)

    labels.append("All")
    filters.append(base_filter)

    info, params, output = get_param_dict(
        f"{root}/config/{config}/{config}", params, output, debug=debug)
    for this_filter in filter_list:
        if this_filter == "Primary":
            primary_filter = run["Reco"]["Primary"] == True
            filters, labels = add_filter(
                filters, labels, (primary_filter), "Primary", cummulative, debug
            )
        if this_filter == "NHits":
            hit_filter = run["Reco"]["NHits"] >= params["PRESELECTION_NHITS"]
            filters, labels = add_filter(
                filters, labels, (hit_filter), "NHits", cummulative, debug
            )
        if this_filter == "OpFlash":
            flash_filter = run["Reco"]["FlashMatched"] == True
            filters, labels = add_filter(
                filters, labels, flash_filter, "OpFlash", cummulative, debug
            )

        if this_filter == "AdjCl":
            adjcl_filter = (
                np.sum(
                    (
                        (run["Reco"]["AdjClR"] > params["MIN_BKG_R"])
                        * (run["Reco"]["AdjClCharge"] < params["MAX_BKG_CHARGE"])
                        == False
                    )
                    * (run["Reco"]["AdjClCharge"] > 0),
                    axis=1,
                )
                > 0
            )
            filters, labels = add_filter(
                filters, labels, adjcl_filter, "AdjCl", cummulative, debug
            )

        if this_filter == "EnergyPerHit":
            max_eperhit_filter = (
                run["Reco"]["EnergyPerHit"] < params["MAX_ENERGY_PER_HIT"]
            )
            min_eperhit_filter = (
                run["Reco"]["EnergyPerHit"] > params["MIN_ENERGY_PER_HIT"]
            )
            eperhit_filter = (max_eperhit_filter) * (min_eperhit_filter)
            filters, labels = add_filter(
                filters, labels, eperhit_filter, "EnergyPerHit", cummulative, debug
            )

        if (
            this_filter == "Fiducial"
            or this_filter == "RecoX"
            or this_filter == "RecoY"
            or this_filter == "RecoZ"
        ):
            max_recox_filter = (
                np.abs(run["Reco"]["RecoX"])
                < (1 - params["FIDUTIAL_FACTOR"]) * (info["DETECTOR_SIZE_X"]) / 2
            )
            min_recox_filter = (
                np.abs(run["Reco"]["RecoX"])
                > (params["FIDUTIAL_FACTOR"]) * (info["DETECTOR_SIZE_X"]) / 2
            )
            recox_filter = (max_recox_filter) * (min_recox_filter)
            recoy_filter = (
                np.abs(run["Reco"]["RecoY"])
                < (1 - params["FIDUTIAL_FACTOR"]) * (info["DETECTOR_SIZE_Y"]) / 2
            )
            recoz_filter = (
                (run["Reco"]["RecoZ"])
                < (1 - params["FIDUTIAL_FACTOR"]) * (info["DETECTOR_SIZE_Z"])
            ) * (
                run["Reco"]["RecoZ"]
                > (params["FIDUTIAL_FACTOR"]) * (info["DETECTOR_SIZE_Z"])
            )

            if this_filter == "Fiducial":
                filters, labels = add_filter(
                    filters,
                    labels,
                    recox_filter * recoy_filter * recoz_filter,
                    "Fiducial",
                    cummulative,
                    debug,
                )
            elif this_filter == "RecoX":
                filters, labels = add_filter(
                    filters, labels, recox_filter, "RecoX", cummulative, debug
                )
            elif this_filter == "RecoY":
                filters, labels = add_filter(
                    filters, labels, recoy_filter, "RecoY", cummulative, debug
                )
            elif this_filter == "RecoZ":
                filters, labels = add_filter(
                    filters, labels, recoz_filter, "RecoZ", cummulative, debug
                )

        if this_filter == "MainParticle":
            # mainpdg_filter = run["Reco"]["MainPDG"] != params["MAIN_PDG"]
            max_main_e_filter = run["Reco"]["MainE"] < params["MAX_MAIN_E"]
            min_main_e_filter = run["Reco"]["MainE"] > params["MIN_MAIN_E"]
            main_filter = (max_main_e_filter) * (min_main_e_filter)
            filters, labels = add_filter(
                filters, labels, main_filter, "MainParticle", cummulative, debug
            )

        if this_filter == "MainClEnergy":
            # mainpdg_filter = run["Reco"]["MainPDG"] != params["MAIN_PDG"]
            max_main_e_filter = run["Reco"]["Energy"] < params["MAX_CL_E"]
            min_main_e_filter = run["Reco"]["Energy"] > params["MIN_CL_E"]
            main_filter = (max_main_e_filter) * (min_main_e_filter)
            filters, labels = add_filter(
                filters, labels, main_filter, "MainClEnergy", cummulative, debug
            )

        if this_filter == "AdjClEnergy":
            # mainpdg_filter = run["Reco"]["MainPDG"] != params["MAIN_PDG"]
            max_main_e_filter = run["Reco"]["TotalAdjClEnergy"] < params["MAX_ADJCL_E"]
            min_main_e_filter = run["Reco"]["TotalAdjClEnergy"] > params["MIN_ADJCL_E"]
            main_filter = (max_main_e_filter) * (min_main_e_filter)
            filters, labels = add_filter(
                filters, labels, main_filter, "AdjClEnergy", cummulative, debug
            )

    if debug:
        print("Filters computation -> Done!")
    return filters, labels


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


def reshape_array(array: np.array, length: int, debug: bool = False):
    """
    Reshape the array to the desired length.
    """
    repeated_array = np.repeat(array, length)
    return np.reshape(repeated_array, (-1, length))
