import numpy as np

from typing import Optional
from itertools import product
from rich import print as rprint
from particle import Particle

from lib.workflow.functions import get_param_dict, remove_branches

from src.utils import get_project_root
root = get_project_root()

def compute_marley_particle(run: dict[dict], configs: dict[str, list[str]], params: Optional[dict] = None, rm_branches: bool = False, output: Optional[str] = None, debug=False):
    '''
    Compute the Signal particle type for the events in the run.
    '''
    signal = "Signal"
    if output is None:
        output = ""
    required_branches = ["Generator", f"TSignalFrac"]
    new_branches = ["Neutrino", "Electron", "Gamma", "Neutron"]

    run["Reco"]["Neutrino"] = run["Reco"]["Generator"] == 1
    for idx, particle in enumerate(new_branches[1:]):
        run["Reco"][particle] = (run["Reco"]["Generator"] == 1) * \
            (run["Reco"][f"TSignalFrac"][:, idx] > 0.5)

    output += f"\tMarley particle computation \t-> Done!\n"
    return run, output, new_branches


def compute_signal_directions(run, configs, params:Optional[dict] = None, trees=["Truth", "Reco"], rm_branches: bool = False, output: Optional[str] = None, debug=False):
    """
    This functions loops over the Signal particles and computes the direction of the particles, returning variables with the same structure as TMarleyPDG.
    """

    required_branches = {"Truth": ["Event", "Flag", "Geometry", "Version", f"TSignalPDG", f"TSignalMother", f"TSignalE", f"TSignalP", f"TSignalEnd", f"TSignalDirection"],
                         "Reco": ["Event", "Flag", "Geometry", "Version", "MTrackEnd", "MTrackStart", "MTrackDirection"]}
    new_branches = [f"TSignalTheta", f"TSignalPhi", f"TSignalDirectionX",
                    f"TSignalDirectionY", f"TSignalDirectionZ", f"TSignalDirectionMod"]
    for tree in trees:
        for branch in new_branches:
            run[tree][branch] = np.zeros(
                (len(run[tree]["Event"]), len(run[tree][f"TSignalPDG"][0])), dtype=np.float32)

        for config in configs:
            info, params, output = get_param_dict(
                f"{root}/config/{config}/{config}", params, debug=debug)
            idx = np.where(
                (np.asarray(run[tree]["Geometry"]) == info["GEOMETRY"])
                * (np.asarray(run[tree]["Version"]) == info["VERSION"])
            )
            for direction, start, end in zip([f"TSignalDirectionX", f"TSignalDirectionY", f"TSignalDirectionZ"], ["SignalParticleX", "SignalParticleY", "SignalParticleZ"], [f"TSignalEndX", f"TSignalEndY", f"TSignalEndZ"]):
                run[tree][direction][idx] = run[tree][start][:,
                                                             None][idx] - run[tree][end][idx]

            run[tree][f"TSignalDirectionMod"][idx] = np.sqrt(np.power(run[tree][f"TSignalDirectionX"][idx], 2) + np.power(
                run[tree][f"TSignalDirectionY"][idx], 2) + np.power(run[tree][f"TSignalDirectionZ"][idx], 2))
            for coord in ["X", "Y", "Z"]:
                run[tree][f"TSignalDirection{coord}"][idx] = run[tree][f"TSignalDirection{coord}"][idx] / \
                    run[tree][f"TSignalDirectionMod"][idx]

            run[tree][f"TSignalTheta"][idx] = np.arccos(
                run[tree][f"TSignalDirectionZ"][idx])
            run[tree][f"TSignalPhi"][idx] = np.arctan2(
                run[tree][f"TSignalDirectionY"][idx], run[tree][f"TSignalDirectionX"][idx])

            run = remove_branches(
                run, rm_branches, [f"TSignalDirectionMod"], tree=tree, debug=debug)

    output += f"\tMarley direction computation \t-> Done!\n"
    return run, output, new_branches


def compute_signal_energies(run, configs, params: Optional[dict] = None, trees=["Truth", "Reco"], rm_branches: bool = False, output: Optional[str] = None, debug=False):
    '''
    This function computes the total energy of the Signal particles in the event.
    '''

    required_branches = {"Truth": ["Event", "Flag", "Geometry", "Version", "SignalParticleE", f"TSignalPDG", f"TSignalMother", f"TSignalE", f"TSignalP", f"TSignalK"],
                         "Reco": ["Event", "Flag", "Geometry", "Version", "SignalParticleE", f"TSignalPDG", f"TSignalMother", f"TSignalE", f"TSignalP", f"TSignalK"]}
    
    new_branches = [f"TSignalSumE", f"TSignalSumP",f"TSignalSumK"]
    
    if params is None:
        params = {"NORM_TO_NUE": False}
    
    for tree in trees:
        pdg_list = np.unique(run[tree][f"TSignalPDG"]
                             [np.where(run[tree][f"TSignalMother"] == 0)])
        
        pdg_list = pdg_list[pdg_list != 0]
        mass_list = [Particle.from_pdgid(pdg).mass for pdg in pdg_list]
        run[tree][new_branches[0]] = np.zeros(
            (len(run[tree]["Event"]), len(pdg_list)), dtype=np.float32)
        run[tree][new_branches[1]] = np.zeros(
            (len(run[tree]["Event"]), len(pdg_list)), dtype=np.float32)
        run[tree][new_branches[2]] = np.zeros(
            (len(run[tree]["Event"]), len(pdg_list)), dtype=np.float32)

        for idx, pdg in enumerate(pdg_list):
            run[tree][new_branches[0]][:, idx] = np.sum(
                run[tree][f"TSignalE"]*(run[tree][f"TSignalPDG"] == pdg)*(run[tree][f"TSignalMother"] == 0), axis=1)
            run[tree][new_branches[1]][:, idx] = np.sum(
                run[tree][f"TSignalP"]*(run[tree][f"TSignalPDG"] == pdg)*(run[tree][f"TSignalMother"] == 0), axis=1)
            run[tree][new_branches[2]][:, idx] = np.sum(
                run[tree][f"TSignalK"]*(run[tree][f"TSignalPDG"] == pdg)*(run[tree][f"TSignalMother"] == 0), axis=1)

        if params["NORM_TO_NUE"]:
            # Divide by the energy of the neutrino
            run[tree][new_branches[0]] = run[tree][new_branches[0]] / \
                run[tree]["SignalParticleE"][:, None]
            run[tree][new_branches[1]] = run[tree][new_branches[1]] / \
                run[tree]["SignalParticleP"][:, None]
            run[tree][new_branches[2]] = run[tree][new_branches[2]] / \
                run[tree]["SignalParticleK"][:, None]

        run = remove_branches(
            run, rm_branches, [], tree=tree, debug=debug)

    output += f"\tSignal energy computation \t-> Done!\n"
    return run, output, new_branches


def compute_particle_energies(run, configs, params: Optional[dict] = None, trees: list[str] = ["Truth", "Reco"], rm_branches: bool = False, output: Optional[str] = None, debug: bool = False):
    """
    This functions looks into "TMarleyPDG" branch and combines the corresponding "TMarleyE" entries to get a total energy for each daughter particle.
    """

    required_branches = {"Truth": ["Event", "Flag", "Geometry", "Version", f"TSignalPDG", f"TSignalMother", f"TSignalK"],
                         "Reco": ["Event", "Flag", "Geometry", "Version", f"TSignalPDG", f"TSignalMother", f"TSignalK"]}
    
    particles_pdg = {"Electron": 11, "Gamma": 22, "Alpha": 1000020040,
                     "Neutron": 2112, "Neutrino": 12, "Proton": 2212}

    new_branches = list(particles_pdg.keys())
    for tree, particle in product(trees, particles_pdg):
        run[tree][f"{particle}K"] = np.zeros(
            len(run[tree]["Event"]), dtype=np.float32)

    for config, tree in product(configs, trees):
        info, params, output = get_param_dict(
            f"{root}/config/{config}/{config}", params, output, debug=debug)
        idx = np.where(
            (np.asarray(run[tree]["Geometry"]) == info["GEOMETRY"])
            * (np.asarray(run[tree]["Version"]) == info["VERSION"])
        )
        for particle in particles_pdg:
            run[tree][f"{particle}K"][idx] = np.sum(
                run[tree][f"TSignalK"][idx]
                * np.array(run[tree][f"TSignalPDG"][idx]
                           == particles_pdg[particle]) * np.array(run[tree][f"TSignalMother"][idx] == 0),
                axis=1,
            )

            if params["NORM_TO_NUE"]:
                for particle in particles_pdg:
                    run[tree][f"{particle}K"][idx] = run[tree][f"{particle}K"][idx] / \
                        run[tree]["SignalParticleE"][idx]

    run = remove_branches(run, rm_branches, [], debug=debug)
    output += f"\tParticle energy combination \t-> Done!\n"
    return run, output, new_branches