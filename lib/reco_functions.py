import gc, numba, json
import numpy as np

from rich import print as rprint
from itertools import product
from src.utils import get_project_root
from particle import Particle

root = get_project_root()

def compute_reco_workflow(
    run, configs, params={}, workflow="ANALYSIS", rm_branches=True, debug=False
):
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
    if debug:
        rprint("[blue]\nComputing %s workflow[/blue]" % workflow)

    if workflow == "BASIC":
        pass

    if workflow == "TRUTH":
        run = compute_true_efficiency(
            run, configs, params, rm_branches=rm_branches, debug=debug
        )
        run = compute_marley_directions(
            run, configs, params, trees=["Truth"], rm_branches=rm_branches, debug=debug
        )
        
    if workflow == "MARLEY":
        run = compute_true_efficiency(
            run, configs, params, rm_branches=rm_branches, debug=debug
        )
        # Check if key in params and set to False if not
        if "NORM_TO_NUE" not in params:
            params["NORM_TO_NUE"] = False
        run = compute_marley_energies(
            run, configs, params, trees=["Truth"], rm_branches=rm_branches, debug=debug
        )
        run = compute_marley_directions(
            run, configs, params, trees=["Truth"], rm_branches=rm_branches, debug=debug
        )
        run = compute_particle_energies(
            run, configs, params, trees=["Truth"], rm_branches=rm_branches, debug=debug
        )
    
    if workflow == "RAW":
        # Check if key in params and set to False if not
        if "NORM_TO_NUE" not in params:
            params["NORM_TO_NUE"] = False
        run = compute_marley_energies(
            run, configs, params, trees=["Truth"], rm_branches=rm_branches, debug=debug
        )
        run = compute_particle_energies(
            run, configs, params, trees=["Truth"], rm_branches=rm_branches, debug=debug
        )

    if workflow == "TRACK":
        run = compute_marley_directions(
            run, configs, params, trees=["Reco"], rm_branches=rm_branches, debug=debug
        )
        run = compute_particle_directions(
            run, configs, params, rm_branches=rm_branches, debug=debug
        )

    if workflow == "ADJCL":
        run = compute_adjcl_basics(
            run, configs, params, rm_branches=rm_branches, debug=debug
        )
        run = compute_adjcl_advanced(
            run, configs, params, rm_branches=rm_branches, debug=debug
        )

    if workflow == "ADJFLASH":
        run = compute_opflash_advanced(
            run, configs, params, rm_branches=rm_branches, debug=debug
        )
        run = compute_opflash_matching(
            run, configs, params, rm_branches=rm_branches, debug=debug
        )
        run = compute_recox(
            run, configs, params, rm_branches=rm_branches, debug=debug
        )

    if workflow == "CALIBRATION":
        run = compute_main_variables(
            run, configs, params, rm_branches=rm_branches, debug=debug
        )

    if workflow == "VERTEXING":
        run = compute_main_variables(
            run, configs, params, rm_branches=rm_branches, debug=debug
        )
        run = compute_opflash_advanced(
            run, configs, params, rm_branches=rm_branches, debug=debug
        )
        run = compute_opflash_matching(
            run, configs, params, rm_branches=rm_branches, debug=debug
        )
        run = compute_recox(
            run, configs, params, rm_branches=rm_branches, debug=debug
        )

    if workflow == "ANALYSIS":
        run = compute_opflash_advanced(
            run, configs, params, rm_branches=rm_branches, debug=debug
        )
        run = compute_opflash_matching(
            run, configs, params, rm_branches=rm_branches, debug=debug
        )
        run = compute_recox(
            run, configs, params, rm_branches=rm_branches, debug=debug
        )
        run = compute_cluster_energy(
            run, configs, params, rm_branches=rm_branches, debug=debug
        )
    
    if workflow == "ENERGY":
        run = compute_particle_energies(
            run, configs, debug=debug
        )
        run = compute_true_drift(
            run, configs, debug=debug
        )
        run = compute_cluster_energy(
            run, configs, params={"DEFAULT_ENERGY_TIME":"TrueDriftTime", "DEFAULT_ADJCL_ENERGY_TIME":"AdjClTrueDriftTime"},debug=debug
        )
        run = compute_adjcl_basics(
            run, configs, debug=debug
        )
        run = compute_adjcl_advanced(
            run, configs, debug=debug
        )
        run = compute_reco_energy(
            run, configs, debug=debug
        )

    if workflow == "SOLAR":
        run = compute_filtered_run(
            run, configs, params={('Reco','Primary'):('equal',True)}, debug=debug
        )
        run = compute_opflash_advanced(
            run, configs, params, rm_branches=rm_branches, debug=debug
        )
        run = compute_opflash_matching(
            run, configs, params, rm_branches=rm_branches, debug=debug
        )
        run = compute_recox(
            run, configs, params, rm_branches=rm_branches, debug=debug
        )
        run = compute_main_variables(
            run, configs, rm_branches=rm_branches, debug=debug
        )
        run = compute_true_drift(
            run, configs, rm_branches=rm_branches, debug=debug
        )
        run = compute_cluster_energy(
            run, configs, params=params, rm_branches=rm_branches, debug=debug
        )
        run = compute_adjcl_basics(
            run, configs, rm_branches=rm_branches, debug=debug
        )
        run = compute_adjcl_advanced(
            run, configs, rm_branches=rm_branches, debug=debug
        )
        run = compute_reco_energy(
            run, configs, rm_branches=rm_branches, debug=debug
        )
    
    if workflow == "FULL":
        run = compute_adjcl_basics(
            run, configs, params, rm_branches=rm_branches, debug=debug
        )
        run = compute_opflash_matching(
            run, configs, params, rm_branches=rm_branches, debug=debug
        )
        run = compute_adjcl_advanced(
            run, configs, params, rm_branches=rm_branches, debug=debug
        )
        run = compute_recox(
            run, configs, params, rm_branches=rm_branches, debug=debug
        )
        run = compute_cluster_energy(
            run, configs, params, rm_branches=rm_branches, debug=debug
        )
        run = compute_reco_energy(
            run, configs, params, rm_branches=rm_branches, debug=debug
        )
        run = compute_opflash_advanced(
            run, configs, params, rm_branches=rm_branches, debug=debug
        )

    rprint("[green]Reco workflow \t-> Done![/green]")
    return run


def compute_main_variables(run, configs, params={}, rm_branches=False, debug=False):
    """
    Compute the main (backtracked) variables of main particle correspondong to the cluster in the run.
    """
    run["Reco"]["MainX"]  = run["Reco"]["MainVertex"][:,0]
    run["Reco"]["MainY"]  = run["Reco"]["MainVertex"][:,1]
    run["Reco"]["MainZ"]  = run["Reco"]["MainVertex"][:,2]

    run["Reco"]["ParentX"]  = run["Reco"]["MainParentVertex"][:,0]
    run["Reco"]["ParentY"]  = run["Reco"]["MainParentVertex"][:,1]
    run["Reco"]["ParentZ"]  = run["Reco"]["MainParentVertex"][:,2]

    run["Reco"]["ErrorY"]    = abs(run["Reco"]["MainY"] - run["Reco"]["RecoY"]) 
    run["Reco"]["ErrorZ"]    = abs(run["Reco"]["MainZ"] - run["Reco"]["RecoZ"])
    run["Reco"]["ErrorTot"]  = np.sqrt(np.power(run["Reco"]["ErrorZ"],2) + np.power(run["Reco"]["ErrorY"],2))

    run["Reco"]["Neutrino"] = run["Reco"]["Generator"] == 1
    run["Reco"]["Electron"] = (run["Reco"]["Generator"] == 1) * (run["Reco"]["MarleyFrac"][:,0] > 0.5)
    run["Reco"]["Gamma"]    = (run["Reco"]["Generator"] == 1) * (run["Reco"]["MarleyFrac"][:,1] > 0.5)
    run["Reco"]["Neutron"]  = (run["Reco"]["Generator"] == 1) * (run["Reco"]["MarleyFrac"][:,2] > 0.5)
    return run


def compute_true_efficiency(run, configs, params={}, rm_branches=False, debug=False):
    """
    Compute the true efficiency of the events in the run.
    """
    # New branches
    new_branches = ["RecoIndex", "RecoMatch", "ClCount", "HitCount", "TrueIndex"]
    run["Truth"][new_branches[0]] = np.zeros(len(run["Truth"]["Event"]), dtype=int)
    run["Truth"][new_branches[1]] = np.zeros(len(run["Truth"]["Event"]), dtype=bool)
    run["Truth"][new_branches[2]] = np.zeros(len(run["Truth"]["Event"]), dtype=int)
    run["Truth"][new_branches[3]] = np.zeros(len(run["Truth"]["Event"]), dtype=int)
    run["Reco"][new_branches[4]] = np.zeros(len(run["Reco"]["Event"]), dtype=int)

    for config in configs:
        info = json.load(open(f"{root}/config/{config}/{config}_config.json", "r"))
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

    rprint(f"True efficiency computation \t-> Done! ({new_branches})")
    run = remove_branches(run, rm_branches, [], debug=debug)
    return run

def compute_marley_directions(run, configs, params={}, trees=["Truth","Reco"], rm_branches=False, debug=False):
    """
    This functions loops over the Marley particles and computes the direction of the particles, returning variables with the same structure as TMarleyPDG.
    """
    new_branches = ["TMarleyTheta","TMarleyPhi","TMarleyDirectionX","TMarleyDirectionY","TMarleyDirectionZ","TMarleyDirectionMod"]
    for tree in trees:
        for branch in new_branches:
            run[tree][branch] = np.zeros((len(run[tree]["Event"]),len(run[tree]["TMarleyPDG"][0])),dtype=float)

        for config in configs:
            info = json.load(open(f"{root}/config/{config}/{config}_config.json", "r"))
            idx = np.where(
                (np.asarray(run[tree]["Geometry"]) == info["GEOMETRY"])
                * (np.asarray(run[tree]["Version"]) == info["VERSION"])
            )
            for direction,start,end in zip(["TMarleyDirectionX","TMarleyDirectionY","TMarleyDirectionZ"],["TNuX","TNuY","TNuZ"],["TMarleyEndX","TMarleyEndY","TMarleyEndZ"]):
                run[tree][direction][idx] = run[tree][start][:,None][idx] - run[tree][end][idx]
            
            run[tree]["TMarleyDirectionMod"][idx] = np.sqrt(np.power(run[tree]["TMarleyDirectionX"][idx],2) + np.power(run[tree]["TMarleyDirectionY"][idx],2) + np.power(run[tree]["TMarleyDirectionZ"][idx],2))
            for coord in ["X","Y","Z"]:
                run[tree][f"TMarleyDirection{coord}"][idx] = run[tree][f"TMarleyDirection{coord}"][idx]/run[tree]["TMarleyDirectionMod"][idx]

            run[tree]["TMarleyTheta"][idx] = np.arccos(run[tree]["TMarleyDirectionZ"][idx])
            run[tree]["TMarleyPhi"][idx] = np.arctan2(run[tree]["TMarleyDirectionY"][idx],run[tree]["TMarleyDirectionX"][idx])

            run = remove_branches(run, rm_branches, ["TMarleyDirectionMod"], tree=tree, debug=debug)
    rprint(f"Marley direction computation \t-> Done! ({new_branches})")
    return run


def compute_particle_directions(run, configs, params={}, trees=["Reco"], rm_branches=False, debug=False):
    """
    This functions loops over the Marley particles and computes the direction of the particles, returning variables with the same structure as TMarleyPDG.
    """
    new_branches = ["MTrackTheta","MTrackPhi","MTrackDirectionX","MTrackDirectionY","MTrackDirectionZ","MTrackDirectionMod"]
    for tree in trees:
        run[tree]["MTrackDirection"] = np.zeros((len(run[tree]["Event"]),3),dtype=float)
        for branch in new_branches:
            run[tree][branch] = np.zeros(len(run[tree]["Event"]),dtype=float)
        for config in configs:  
            info = json.load(open(f"{root}/config/{config}/{config}_config.json", "r"))
            idx = np.where(
                (np.asarray(run[tree]["Geometry"]) == info["GEOMETRY"])
                * (np.asarray(run[tree]["Version"]) == info["VERSION"])
            )
            run[tree]["MTrackDirection"][idx] = np.subtract(run[tree]["MTrackEnd"][idx], run[tree]["MTrackStart"][idx])
            # run[tree]["MTrackDirectionMod"][idx] = np.linalg.norm(run[tree]["MTrackDirection"][idx], axis=1)
            # run[tree]["MTrackDirection"][idx] = run[tree]["MTrackDirection"][idx]/run[tree]["MTrackDirectionMod"][:,None][idx]
            for coord_idx, coord in enumerate(["MTrackDirectionX","MTrackDirectionY","MTrackDirectionZ"]):
                rprint(f"Computing {coord} direction")
                run[tree][coord][idx] = run[tree]["MTrackDirection"][:,coord_idx][idx]

            run[tree]["MTrackTheta"][idx] = np.arccos(run[tree]["MTrackDirectionZ"][idx])
            run[tree]["MTrackPhi"][idx] = np.arctan2(run[tree]["MTrackDirectionY"][idx],run[tree]["MTrackDirectionX"][idx])

        run = remove_branches(run, rm_branches, ["MTrackDirectionMod"], tree=tree, debug=debug)
    rprint(f"Marley direction computation \t-> Done! ({new_branches})")
    return run

def compute_marley_energies(run, configs, params={"NORM_TO_NUE":False}, trees=["Truth","Reco"], rm_branches=False, debug=False):
    for tree in trees:
        pdg_list = np.unique(run[tree]["TMarleyPDG"][np.where(run[tree]["TMarleyMother"] == 0)])
        pdg_list = pdg_list[pdg_list != 0]
        mass_list = [Particle.from_pdgid(pdg).mass for pdg in pdg_list]
        new_branches = ["TMarleySumE","TMarleySumP","TMarleySumK","TMarleyK","TMarleyMass"]
        run[tree][new_branches[0]] = np.zeros((len(run[tree]["Event"]),len(pdg_list)),dtype=float)
        run[tree][new_branches[1]] = np.zeros((len(run[tree]["Event"]),len(pdg_list)),dtype=float)
        run[tree][new_branches[2]] = np.zeros((len(run[tree]["Event"]),len(pdg_list)),dtype=float)
        run[tree][new_branches[3]] = np.zeros((len(run[tree]["Event"]),len(run[tree]["TMarleyPDG"][0])),dtype=float)
        run[tree][new_branches[4]] = np.zeros((len(run[tree]["Event"]),len(run[tree]["TMarleyPDG"][0])),dtype=float)

        full_pdg_list = np.unique(run[tree]["TMarleyPDG"])
        for non_pdg in [0,1000120249,1000140289,1000190419,1000210499,1000220489,1000130279,1000360809,1000360829]:
            full_pdg_list = full_pdg_list[full_pdg_list != non_pdg]
        full_mass_dict = {pdg:Particle.from_pdgid(pdg).mass for pdg in full_pdg_list}

        # Gnearte branch for the mass of the particles frmo the TMarleyPDG m times n array and store it in the TMarleyMass branch
        run[tree]["TMarleyMass"] = np.vectorize(full_mass_dict.get)(run[tree]["TMarleyPDG"])
        run[tree]["TMarleyMass"] = np.nan_to_num(run[tree]["TMarleyMass"],nan=0.0,posinf=0.0,neginf=0.0)
        run[tree]["TMarleyK"] = np.subtract(run[tree]["TMarleyE"], run[tree]["TMarleyMass"])

        for idx,pdg in enumerate(pdg_list):
            run[tree][new_branches[0]][:,idx] = np.sum(run[tree]["TMarleyE"]*(run[tree]["TMarleyPDG"] == pdg)*(run[tree]["TMarleyMother"] == 0),axis=1)
            run[tree][new_branches[1]][:,idx] = np.sum(run[tree]["TMarleyP"]*(run[tree]["TMarleyPDG"] == pdg)*(run[tree]["TMarleyMother"] == 0),axis=1)
            run[tree][new_branches[2]][:,idx] = np.sum(run[tree]["TMarleyK"]*(run[tree]["TMarleyPDG"] == pdg)*(run[tree]["TMarleyMother"] == 0),axis=1)
        
        if params["NORM_TO_NUE"]:
            # Divide by the energy of the neutrino
            run[tree][new_branches[0]] = run[tree][new_branches[0]]/run[tree]["TNuE"][:,None]
            run[tree][new_branches[1]] = run[tree][new_branches[1]]/run[tree]["TNuE"][:,None]
            run[tree][new_branches[2]] = run[tree][new_branches[2]]/run[tree]["TNuE"][:,None]
        
        pdg_list = np.repeat(pdg_list, len(run[tree]["Event"])).reshape(len(pdg_list),len(run[tree]["Event"])).T
        run[tree]["TMarleySumPDG"] = pdg_list
        
        run = remove_branches(run, rm_branches, ["TMarleyMass"], tree=tree, debug=debug)
    rprint(f"Marley energy computation \t-> Done! ({new_branches})")
    return run

def compute_particle_energies(run, configs, params={"NORM_TO_NUE":False}, trees=["Truth","Reco"], rm_branches=False, debug=False):
    """
    This functions looks into "TMarleyPDG" branch and combines the corresponding "TMarleyE" entries to get a total energy for each daughter particle.
    """
    particles = {"Electron": 11, "Gamma": 22, "Neutron": 2112, "Neutrino": 12, "Proton": 2212}
    particles_mass = { particle:values for particle,values in zip(particles.keys(),[Particle.from_pdgid(particles[particle]).mass for particle in particles])}
    particles_mass["Neutrino"] = 0
    new_branches = list(particles.keys())
    for tree, particle in product(trees,particles):
        run[tree][f"{particle}E"] = np.zeros(len(run[tree]["Event"]), dtype=float)
        # run[tree][f"{particle}P"] = np.zeros(len(run[tree]["Event"]), dtype=float)
        run[tree][f"{particle}K"] = np.zeros(len(run[tree]["Event"]), dtype=float)

    for config, tree in product(configs,trees):
        info = json.load(open(f"{root}/config/{config}/{config}_config.json", "r"))
        idx = np.where(
            (np.asarray(run[tree]["Geometry"]) == info["GEOMETRY"])
            * (np.asarray(run[tree]["Version"]) == info["VERSION"])
        )

        for particle in particles:
            run[tree][f"{particle}E"][idx] = np.sum(
                run[tree]["TMarleyE"][idx]
                * np.array(run[tree]["TMarleyPDG"][idx] == particles[particle]),
                axis=1,
            )
            # run[tree][f"{particle}P"][idx] = np.sum(
            #     run[tree]["TMarleyP"][idx]
            #     * np.array(run[tree]["TMarleyPDG"][idx] == particles[particle]),
            #     axis=1,
            # )
            run[tree][f"{particle}K"][idx] = np.sum(
                (run[tree]["TMarleyE"][idx] - particles_mass[particle])
                * np.array(run[tree]["TMarleyPDG"][idx] == particles[particle]),
                axis=1,
            )
            
            if params["NORM_TO_NUE"]:
                for particle in particles:
                    run[tree][f"{particle}E"] = run[tree][f"{particle}E"]/run[tree]["TNuE"]
                    # run[tree][f"{particle}P"] = run[tree][f"{particle}P"]/run[tree]["TNuE"]
                    run[tree][f"{particle}K"] = run[tree][f"{particle}K"]/run[tree]["TNuE"]

    rprint(f"Particle energy combination \t-> Done! ({new_branches})")
    run = remove_branches(run, rm_branches, [], debug=debug)
    return run


def compute_opflash_matching(
    run,
    configs,
    params={"MAX_FLASH_R": None, "MIN_FLASH_PE": None, "RATIO_FLASH_PEvsR": None},
    rm_branches=False,
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
    run["Reco"][new_branches[1]] = np.zeros(len(run["Reco"]["Event"]), dtype=bool)
    run["Reco"][new_branches[2]] = np.zeros(len(run["Reco"]["Event"]), dtype=int)
    run["Reco"][new_branches[3]] = np.zeros(len(run["Reco"]["Event"]), dtype=float)
    run["Reco"][new_branches[4]] = np.zeros(len(run["Reco"]["Event"]), dtype=float)
    run["Reco"][new_branches[5]] = np.zeros(len(run["Reco"]["Event"]), dtype=float)
    run["Reco"][new_branches[6]] = np.zeros(len(run["Reco"]["Event"]), dtype=float)
    run["Reco"][new_branches[7]] = np.zeros(
        (len(run["Reco"]["Event"]), len(run["Reco"]["AdjClTime"][0])), dtype=float
    )
    for config in configs:
        info = json.load(open(f"{root}/config/{config}/{config}_config.json", "r"))
        idx = np.where(
            (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
            * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
        )
        # If run["Reco"]["AdjClTime"][idx][0] empty skip the config:
        if run["Reco"]["AdjOpFlashTime"][idx].sum() == 0:
            continue

        # Get values from the configuration file or use the ones given as input
        params = get_param_dict(f"{root}/config/{config}/{config}_config.json", params, debug=debug)

        # Select all FlashMatch candidates
        max_r_filter = run["Reco"]["AdjOpFlashR"][idx] < params["MAX_FLASH_R"]
        min_pe_filter = run["Reco"]["AdjOpFlashPE"][idx] > params["MIN_FLASH_PE"]
        signal_nan_filter = (run["Reco"]["AdjOpFlashSignal"][idx] > 0) * (run["Reco"]["AdjOpFlashSignal"][idx] < np.inf)
        # max_ratio_filter = run["Reco"]["AdjOpFlashPE"][idx] > 3000 * run["Reco"]["AdjOpFlashMaxPE"][idx] / run["Reco"]["AdjOpFlashPE"][idx]

        repeated_array = np.repeat(
            run["Reco"]["Time"][idx], len(run["Reco"]["AdjOpFlashTime"][idx][0])
        )
        converted_array = np.reshape(
            repeated_array, (-1, len(run["Reco"]["AdjOpFlashTime"][idx][0]))
        )
        max_drift_filter = (
            np.abs(converted_array - 2 * run["Reco"]["AdjOpFlashTime"][idx])
            < params["MAX_DRIFT_FACTOR"] * info["EVENT_TICKS"]
        )
        run["Reco"]["FlashMathedIdx"][idx] = (
            (max_r_filter) * (min_pe_filter) * (max_drift_filter) * (signal_nan_filter)
        )

        # If at least one candidate is found, mark the event as matched and select the best candidate
        run["Reco"]["FlashMatched"][idx] = (
            np.sum(run["Reco"]["FlashMathedIdx"][idx], axis=1) > 0
        )
        run["Reco"]["AssFlashIdx"][idx] = np.argmax(
            run["Reco"]["AdjOpFlashSignal"][idx] * run["Reco"]["FlashMathedIdx"][idx],
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
            run["Reco"]["Time"][idx] - 2 * run["Reco"]["MatchedOpFlashTime"][idx]
        )
        run["Reco"]["AdjClDriftTime"][idx] = (
            run["Reco"]["AdjClTime"][idx]
            - 2 * run["Reco"]["MatchedOpFlashTime"][idx][:, np.newaxis]
        )

    rprint("[green]OpFlash matching \t\t-> Done! (%s)[/green]" % new_branches)
    run = remove_branches(
        run, rm_branches, ["FlashMathedIdx", "AssFlashIdx"], debug=debug
    )
    return run


def compute_recox(
    run, configs, params={"DEFAULT_RECOX_TIME": None}, rm_branches=False, debug=False
):
    """
    Compute the reconstructed X position of the events in the run.
    """
    new_branches = ["RecoX", "AdjCldT", "AdjClRecoX"]
    run["Reco"][new_branches[0]] = np.zeros(len(run["Reco"]["Event"]))
    run["Reco"][new_branches[1]] = np.zeros(
        (len(run["Reco"]["Event"]), len(run["Reco"]["AdjClTime"][0])), dtype=float
    )
    run["Reco"][new_branches[2]] = np.zeros(
        (len(run["Reco"]["Event"]), len(run["Reco"]["AdjClTime"][0])), dtype=float
    )

    for config in configs:
        info = json.load(open(f"{root}/config/{config}/{config}_config.json", "r"))
        idx = np.where(
            (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
            * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
        )
        # Get values from the configuration file or use the ones given as input
        params = get_param_dict(f"{root}/config/{config}/{config}_config.json", params, debug=debug)

        repeated_array = np.repeat(
            run["Reco"]["Time"][idx], len(run["Reco"]["AdjClTime"][idx][0])
        )
        converted_array = np.reshape(
            repeated_array, (-1, len(run["Reco"]["AdjClTime"][idx][0]))
        )
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

            repeated_array = np.repeat(
                run["Reco"]["RecoX"], len(run["Reco"]["AdjClTime"][0])
            )
            converted_array = np.reshape(
                repeated_array, (-1, len(run["Reco"]["AdjClTime"][0]))
            )
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
                        
            converted_array = reshape_array(run["Reco"]["RecoX"][idx], len(run["Reco"]["AdjClTime"][idx][0]))
            # repeated_array = np.repeat(
            #     run["Reco"]["RecoX"][idx], len(run["Reco"]["AdjClTime"][idx][0])
            # )
            # converted_array = np.reshape(
            #     repeated_array, (-1, len(run["Reco"]["AdjClTime"][idx][0]))
            # )
            run["Reco"]["AdjClRecoX"][idx] = (
                run["Reco"]["AdjCldT"][idx]
                * info["DETECTOR_SIZE_X"]
                / info["EVENT_TICKS"]
            ) + converted_array

    rprint(f"Computed RecoX \t\t\t-> Done! ({new_branches})")
    run = remove_branches(run, rm_branches, ["AdjCldT"], debug=debug)
    return run


def compute_cluster_energy(
    run,
    configs,
    params={"DEFAULT_ENERGY_TIME": None, "DEFAULT_ADJCL_ENERGY_TIME": None},
    rm_branches=False,
    debug=False,
):
    """
    Correct the charge of the events in the run according to the correction file.
    """
    # New branches
    new_branches = ["Correction", "Energy", "AdjClCorrection", "AdjClEnergy"]
    run["Reco"][new_branches[0]] = np.ones(len(run["Reco"]["Event"]))
    run["Reco"][new_branches[1]] = np.zeros(len(run["Reco"]["Event"]))
    run["Reco"][new_branches[2]] = np.ones(
        (len(run["Reco"]["Event"]), len(run["Reco"]["AdjClCharge"][0])), dtype=float
    )
    run["Reco"][new_branches[3]] = np.zeros(
        (len(run["Reco"]["Event"]), len(run["Reco"]["AdjClCharge"][0])), dtype=float
    )

    for config in configs:
        info = json.load(open(f"{root}/config/{config}/{config}_config.json", "r"))
        idx = np.where(
            (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
            * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
        )

        params = get_param_dict(f"{root}/config/{config}/{config}_config.json", params, debug=debug)
        corr_info = json.load(open(f"{root}/config/{config}/wbkg/{config}_calib/{config}_charge_correction.json","r"))
        corr_popt = [corr_info["CHARGE_AMP"], corr_info["ELECTRON_TAU"]]
        reco_popt = [corr_info["SLOPE"], corr_info["INTERCEPT"]]
        
        run["Reco"]["Correction"][idx] = np.exp(
            np.abs(run["Reco"][params["DEFAULT_ENERGY_TIME"]][idx]) / corr_popt[1]
        )
        run["Reco"]["AdjClCorrection"][idx] = np.exp(
            np.abs(run["Reco"][params["DEFAULT_ADJCL_ENERGY_TIME"]][idx]) / corr_popt[1]
        )
        run["Reco"]["Energy"][idx] = (
            run["Reco"]["Charge"][idx] * run["Reco"]["Correction"][idx] / corr_popt[0]
        )
        run["Reco"]["AdjClEnergy"][idx] = (
            run["Reco"]["AdjClCharge"][idx]
            * run["Reco"]["Correction"][idx][:, np.newaxis]
            / corr_popt[0]
        )
        # Fine tuning of the energy calibration
        run["Reco"]["Energy"][idx] = (run["Reco"]["Energy"][idx] - reco_popt[1]) / reco_popt[0]

    rprint(f"[green]Clutser energy computation\t-> Done! ({new_branches})[/green]")
    run = remove_branches(
        run, rm_branches, ["Correction", "AdjClCorrection"], debug=debug
    )
    return run


def compute_reco_energy(run, configs, params={}, rm_branches=False, debug=False):
    """
    Compute the total energy of the events in the run.
    """
    run["Reco"]["Discriminant"]        = np.zeros(len(run["Reco"]["Event"]))
    run["Reco"]["RecoEnergy"]          = np.zeros(len(run["Reco"]["Event"]))
    run["Reco"]["TotalEnergy"]         = np.zeros(len(run["Reco"]["Event"]))
    run["Reco"]["SelectedEnergy"]      = np.zeros(len(run["Reco"]["Event"]))
    run["Reco"]["TotalAdjClEnergy"]    = np.zeros(len(run["Reco"]["Event"]))
    run["Reco"]["SelectedAdjClEnergy"] = np.zeros(len(run["Reco"]["Event"]))
    
    for config in configs:
        info = json.load(open(f"{root}/config/{config}/{config}_config.json", "r"))
        idx = np.where(
            (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
            * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
        )

        discriminant_info = json.load(
            open(
                f"{root}/config/{config}/wbkg/{config}_calib/{config}_discriminant_calibration.json",
                "r",
            )
        )
        
        calib_info = json.load(
            open(
                f"{root}/config/{config}/wbkg/{config}_calib/{config}_energy_calibration.json",
                "r",
            )
        )

        run["Reco"]["TotalAdjClEnergy"][idx] = np.sum(
            run["Reco"]["AdjClEnergy"][idx], axis=1
        )

        run["Reco"]["SelectedAdjClEnergy"][idx] = np.sum(
            np.where((run["Reco"]["AdjClR"][idx] < info["MIN_BKG_R"]) + (run["Reco"]["AdjCldT"][idx] < info["MIN_BKG_DT"]) + (run["Reco"]["AdjClCharge"][idx] > info["MAX_BKG_CHARGE"]), run["Reco"]["AdjClEnergy"][idx], 0),
            axis=1
        )

        run["Reco"]["TotalEnergy"][idx] = run["Reco"]["Energy"][idx] + run["Reco"]["TotalAdjClEnergy"][idx] 
        run["Reco"]["SelectedEnergy"][idx] = run["Reco"]["Energy"][idx] + run["Reco"]["SelectedAdjClEnergy"][idx] 

        run["Reco"]["Discriminant"][idx] = (
            run["Reco"]["MaxAdjClEnergy"][idx] + run["Reco"]["AdjClNum"][idx]
        )

        bot_idx = np.where(
            (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
            * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
            * (run["Reco"]["Discriminant"] >= discriminant_info["DISCRIMINANT_THRESHOLD"])
        )
        top_idx = np.where(
            (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
            * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
            * (run["Reco"]["Discriminant"] < discriminant_info["DISCRIMINANT_THRESHOLD"])
        )
        run["Reco"]["RecoEnergy"][bot_idx] = (
            run["Reco"]["Energy"][bot_idx] / discriminant_info["LOWER"]["ENERGY_AMP"]
            - discriminant_info["LOWER"]["INTERSECTION"]
        )
        run["Reco"]["RecoEnergy"][top_idx] = (
            run["Reco"]["Energy"][top_idx] / discriminant_info["UPPER"]["ENERGY_AMP"]
            - discriminant_info["UPPER"]["INTERSECTION"]
        )
        for energy_label, energy_key in zip(["TotalEnergy","RecoEnergy"],["TOTAL","RECO"]):
            run["Reco"][energy_label][idx] = (run["Reco"][energy_label][idx] - calib_info[energy_key]["INTERSECTION"]) / calib_info[energy_key]["ENERGY_AMP"]


    rprint("[green]Total energy computation \t-> Done![/green]")
    run = remove_branches(run, rm_branches, ["TotalAdjClEnergy", "SelectedAdjClEnergy"], debug=debug)
    return run


def compute_opflash_advanced(run, configs, params={}, rm_branches=False, debug=False):
    """
    Compute the OpFlash variables for the events in the run.
    """
    # New branches
    run["Reco"]["AdjOpFlashNum"] = np.sum(run["Reco"]["AdjOpFlashR"] != 0, axis=1)
    # run["Reco"]["AdjOpFlashR"] = np.where(run["Reco"]["AdjOpFlashR"] == 0, np.nan, run["Reco"]["AdjOpFlashR"])
    run["Reco"]["AdjOpFlashRatio"] = run["Reco"]["AdjOpFlashMaxPE"]/run["Reco"]["AdjOpFlashPE"]
    run["Reco"]["AdjOpFlashSignal"] = run["Reco"]["AdjOpFlashNHit"]*run["Reco"]["AdjOpFlashPE"]/run["Reco"]["AdjOpFlashR"]
    # Set AdjOpFlashSignal to 0 if it is Nan
    run["Reco"]["AdjOpFlashSignal"] = np.where(np.isnan(run["Reco"]["AdjOpFlashSignal"]), 0, run["Reco"]["AdjOpFlashSignal"])
    # If AdjOpFlashRatio is 0 set it to Nan
    run["Reco"]["AdjOpFlashRatio"] = np.where(run["Reco"]["AdjOpFlashRatio"] == 0, np.nan, run["Reco"]["AdjOpFlashRatio"])
    run["Reco"]["AdjOpFlashErrorY"] = run["Reco"]["AdjOpFlashRecoY"] - reshape_array(run["Reco"]["TNuY"], len(run["Reco"]["AdjOpFlashRecoY"][0]))
    run["Reco"]["AdjOpFlashErrorZ"] = run["Reco"]["AdjOpFlashRecoZ"] - reshape_array(run["Reco"]["TNuZ"], len(run["Reco"]["AdjOpFlashRecoZ"][0]))

    rprint("[green]OpFlash variables computation \t-> Done![/green]")
    return run


def compute_true_drift(run, configs, params={}, rm_branches=False, debug=False):
    """
    Compute the true drift time of the events in the run.
    """
    # New branches
    run["Reco"]["TrueDriftTime"] = np.zeros(len(run["Reco"]["Event"]), dtype=float)
    run["Reco"]["AdjClTrueDriftTime"] = np.zeros(
        (len(run["Reco"]["Event"]), len(run["Reco"]["AdjClTime"][0])), dtype=float
    )

    for config in configs:
        info = json.load(open(f"{root}/config/{config}/{config}_config.json", "r"))
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
            run["Reco"]["TrueDriftTime"][idx] = (info["DETECTOR_SIZE_X"] - run["Reco"]["MainVertex"][idx, 0]) * 0.5 * info["EVENT_TICKS"] / info["DETECTOR_SIZE_X"]
            run["Reco"]["AdjClTrueDriftTime"][idx] = (info["DETECTOR_SIZE_X"] - run["Reco"]["AdjClMainX"][idx]) * 0.5 * info["EVENT_TICKS"] / info["DETECTOR_SIZE_X"]

    # Select all values bigger than 1e6 or smaller than 0 and set them to 0
    run["Reco"]["TrueDriftTime"] = np.where(
        (run["Reco"]["TrueDriftTime"] > 1e6) | (run["Reco"]["TrueDriftTime"] < 0),
        0,
        run["Reco"]["TrueDriftTime"],
    )
    run["Reco"]["AdjClTrueDriftTime"] = np.where(
        (run["Reco"]["AdjClTrueDriftTime"] > 1e6) | (run["Reco"]["AdjClTrueDriftTime"] < 0),
        0,
        run["Reco"]["AdjClTrueDriftTime"],
    )
    rprint("True drift time computation \t-> Done!")
    return run


def compute_adjcl_basics(run, configs, params={}, rm_branches=False, debug=False):
    """
    Compute basic variables for the adjacent clusters

    Args:
        run: dictionary containing the TTree
        configs: dictionary containing the path to the configuration files for each geoemtry
        params: dictionary containing the parameters for the reco functions
        debug: print debug information
    """
    @numba.njit
    def count_occurrences(arr, length):
        """
        Count the occurrences of each element in the array.

        Args:
            arr: array containing the elements
            length: length of the array
        """
        return [np.sum(arr == i) for i in range(length)]

    # New branches
    run["Reco"]["AdjClNum"] = np.sum(run["Reco"]["AdjClCharge"] != 0, axis=1)
    run["Reco"]["TotalAdjClCharge"] = np.zeros(len(run["Reco"]["Event"]), dtype=float)
    run["Reco"]["MaxAdjClCharge"] = np.zeros(len(run["Reco"]["Event"]), dtype=float)
    run["Reco"]["MeanAdjClCharge"] = np.zeros(len(run["Reco"]["Event"]), dtype=float)
    run["Reco"]["MeanAdjClR"] = np.zeros(len(run["Reco"]["Event"]), dtype=float)
    run["Reco"]["MeanAdjClTime"] = np.zeros(len(run["Reco"]["Event"]), dtype=float)

    run["Reco"]["AdjClSameGenNum"] = np.zeros(len(run["Reco"]["Event"]), dtype=int)
    run["Reco"]["TotalAdjClSameGenCharge"] = np.zeros(
        len(run["Reco"]["Event"]), dtype=float
    )
    run["Reco"]["MaxAdjClSameGenCharge"] = np.zeros(
        len(run["Reco"]["Event"]), dtype=float
    )
    run["Reco"]["MeanAdjClSameGenCharge"] = np.zeros(
        len(run["Reco"]["Event"]), dtype=float
    )

    run["Reco"]["AdjClGenNum"] = np.zeros(
        (len(run["Reco"]["Event"]), len(run["Truth"]["TruthPart"][0]) + 1), dtype=int
    )

    for config in configs:
        info = json.load(open(f"{root}/config/{config}/{config}_config.json", "r"))
        idx = np.where(
            (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
            * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
        )
        params = get_param_dict(f"{root}/config/{config}/{config}_config.json", params, debug=debug)
        run["Reco"]["TotalAdjClCharge"][idx] = np.sum(
            run["Reco"]["AdjClCharge"][idx], axis=1
        )
        run["Reco"]["MaxAdjClCharge"][idx] = np.max(
            run["Reco"]["AdjClCharge"][idx], axis=1
        )
        run["Reco"]["MeanAdjClCharge"][idx] = np.mean(
            run["Reco"]["AdjClCharge"][idx], axis=1
        )
        run["Reco"]["MeanAdjClR"][idx] = np.mean(run["Reco"]["AdjClR"][idx], axis=1)
        run["Reco"]["MeanAdjClTime"][idx] = np.mean(
            run["Reco"]["AdjClTime"][idx], axis=1
        )
        run["Reco"]["AdjClGenNum"][idx] = np.apply_along_axis(
            count_occurrences,
            arr=run["Reco"]["AdjClGen"][idx],
            length=len(run["Reco"]["TruthPart"][idx][0]) + 1,
            axis=1,
        )
        converted_array = reshape_array(run["Reco"]["AdjClGen"][idx], len(run["Reco"]["AdjClGen"][idx][0]))
        repeated_array = np.repeat(
            run["Reco"]["Generator"][idx], len(run["Reco"]["AdjClGen"][idx][0])
        )
        converted_array = np.reshape(
            repeated_array, (-1, len(run["Reco"]["AdjClGen"][idx][0]))
        )
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

    rprint("[green]AdjCl basic computation \t-> Done![/green]")
    run = remove_branches(run, rm_branches, [], debug=debug)
    return run


def compute_adjcl_advanced(run, configs, params={}, rm_branches=False, debug=False):
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
        run["Reco"][branch] = np.zeros(len(run["Reco"]["Event"]), dtype=float)
    
    run["Reco"]["AdjCldT"] = np.zeros((len(run["Reco"]["Event"]), len(run["Reco"]["AdjClTime"][0])), dtype=float)
    run["Reco"]["AdjClRelCharge"] = np.zeros((len(run["Reco"]["Event"]), len(run["Reco"]["AdjClCharge"][0])), dtype=float)
    run["Reco"]["AdjClChargePerHit"] = np.zeros((len(run["Reco"]["Event"]), len(run["Reco"]["AdjClCharge"][0])), dtype=float)
    
    for config in configs:
        info = json.load(open(f"{root}/config/{config}/{config}_config.json", "r"))
        idx = np.where(
            (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
            * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
        )
        params = get_param_dict(f"{root}/config/{config}/{config}_config.json", params, debug=debug)
        run["Reco"]["TotalAdjClEnergy"][idx] = np.sum(
            run["Reco"]["AdjClEnergy"][idx], axis=1
        )
        run["Reco"]["MaxAdjClEnergy"][idx] = np.max(
            run["Reco"]["AdjClEnergy"][idx], axis=1
        )
        repeated_array_time = np.repeat(
            run["Reco"]["Time"][idx], len(run["Reco"]["AdjClTime"][idx][0])
        )
        repeated_array_nhits = np.repeat(
            run["Reco"]["NHits"][idx], len(run["Reco"]["AdjClNHit"][idx][0])
        )
        repeated_array_charge = np.repeat(
            run["Reco"]["Charge"][idx], len(run["Reco"]["AdjClCharge"][idx][0])
        )
        converted_array_time = np.reshape(
            repeated_array_time, (-1, len(run["Reco"]["AdjClTime"][idx][0]))
        )
        converted_array_nhits = np.reshape(
            repeated_array_nhits, (-1, len(run["Reco"]["AdjClNHit"][idx][0]))
        )
        converted_array_charge = np.reshape(
            repeated_array_charge, (-1, len(run["Reco"]["AdjClCharge"][idx][0]))
        )
        run["Reco"]["AdjCldT"][idx] = run["Reco"]["AdjClTime"][idx] - converted_array_time
        run["Reco"]["AdjClRelCharge"] = run["Reco"]["AdjClCharge"][idx] / converted_array_charge
        run["Reco"]["AdjClChargePerHit"] = run["Reco"]["AdjClCharge"][idx] / run["Reco"]["AdjClNHit"][idx]

    rprint(f"AdjCl energy computation \t-> Done! ({new_branches})")
    run = remove_branches(run, rm_branches, [], debug=debug)
    return run


def compute_filtered_run(run, configs, params: dict = {}, debug=False):
    """
    Function to filter all events in the run according to the filters defined in the params dictionary.
    """
    new_run = {}
    if type(params) != dict:
        rprint(f"[red]Params must be a dictionary![/red]")
        return run

    new_trees = run.keys()
    for tree in new_trees:
        new_run[tree] = {}  
        branch_list = list(run[tree].keys())
        idx = np.ones(len(run[tree]["Event"]), dtype=bool)
        filter_output = f""
        for param in params:
            if param[0] != tree:
                continue

            if type(param) != tuple or len(param) != 2:
                rprint(f"[red]ERROR: Filter must be a tuple or list of length 2![/red]")
                return run
            if type(params[param]) != tuple or len(params[param]) != 2:
                rprint(f"[red]ERROR: Filter must be a tuple or list of length 2![/red]")
                return run

            if param[1] not in run[param[0]].keys():
                rprint(f"[red]ERROR: Branch {param[1]} not found in the run![/red]")
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
                idx = idx & (run[param[0]][param[1]] > params[param][1][0]) & (run[param[0]][param[1]] < params[param][1][1])
            elif params[param][0] == "outside":
                idx = idx & ((run[param[0]][param[1]] < params[param][1][0]) + (run[param[0]][param[1]] > params[param][1][1]))
            elif params[param][0] == "contains":
                idx = idx & np.array([params[param][1] in item for item in run[param[0]][param[1]]])
        
        jdx = np.where(idx == True)
        for branch in branch_list:
            try:
                new_run[tree][branch] = np.asarray(run[tree][branch])[jdx]
            except Exception as e:
                rprint(f"Error filtering {branch}: {e}")
        rprint(filter_output)

    # rprint("[green]Filtered run \t\t\t-> Done![/green]")
    return new_run


def get_param_dict(config_file, in_params, debug=False):
    """
    Get the parameters for the reco workflow from the input files.
    """
    params = json.load(open(config_file, "r"))
    terminal_print = ""
    for param in params.keys():
        try:
            if in_params[param] != None:
                params[param] = in_params[param]
                terminal_print = (
                    terminal_print
                    + "-> Using %s: %s from the input dictionary\n"
                    % (param, in_params[param])
                )
            else:
                terminal_print = (
                    terminal_print
                    + "-> Using %s: %s from the config file\n" % (param, params[param])
                )
        except KeyError:
            pass
    if debug:
        if terminal_print != "":
            rprint(terminal_print)
    return params


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
    base_filter = (geo_filter) * (version_filter) * (name_filter) * (gen_filter)

    labels.append("All")
    filters.append(base_filter)

    params = get_param_dict(f"{root}/config/{config}/{config}_config.json", params, debug=debug)
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


def remove_branches(run, remove, branches, tree:str="Reco", debug=False):
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
            rprint(f"[cyan]-> Removing branches: {branches}[/cyan]")
        for branch in branches:
            run[tree].pop(branch)
            gc.collect()
    else:
        pass

    return run


@numba.njit
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


def reshape_array(array, length):
    repeated_array = np.repeat(array, length)
    return np.reshape(repeated_array, (-1, length))