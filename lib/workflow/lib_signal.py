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
    Compute the Marley particle type for the events in the run.
    '''
    signal = "Signal"
    if output is None:
        output = ""
    required_branches = ["Generator", f"T{signal}Frac"]
    new_branches = ["Neutrino", "Electron", "Gamma", "Neutron"]

    run["Reco"]["Neutrino"] = run["Reco"]["Generator"] == 1
    for idx, particle in enumerate(new_branches[1:]):
        run["Reco"][particle] = (run["Reco"]["Generator"] == 1) * \
            (run["Reco"][f"T{signal}Frac"][:, idx] > 0.5)

    output += f"\tMarley particle computation \t-> Done!\n"
    return run, output, new_branches


def compute_signal_directions(run, configs, params:Optional[dict] = None, trees=["Truth", "Reco"], rm_branches: bool = False, output: Optional[str] = None, debug=False):
    """
    This functions loops over the Marley particles and computes the direction of the particles, returning variables with the same structure as TMarleyPDG.
    """
    signal = "Signal"
    required_branches = {"Truth": ["Event", "Flag", "Geometry", "Version", f"T{signal}PDG", f"T{signal}Mother", f"T{signal}E", f"T{signal}P", f"T{signal}End", f"T{signal}Direction"],
                         "Reco": ["Event", "Flag", "Geometry", "Version", "MTrackEnd", "MTrackStart", "MTrackDirection"]}
    new_branches = [f"T{signal}Theta", f"T{signal}Phi", f"T{signal}DirectionX",
                    f"T{signal}DirectionY", f"T{signal}DirectionZ", f"T{signal}DirectionMod"]
    for tree in trees:
        for branch in new_branches:
            run[tree][branch] = np.zeros(
                (len(run[tree]["Event"]), len(run[tree][f"T{signal}PDG"][0])), dtype=np.float32)

        for config in configs:
            info, params, output = get_param_dict(
                f"{root}/config/{config}/{config}", params, debug=debug)
            idx = np.where(
                (np.asarray(run[tree]["Geometry"]) == info["GEOMETRY"])
                * (np.asarray(run[tree]["Version"]) == info["VERSION"])
            )
            for direction, start, end in zip([f"T{signal}DirectionX", f"T{signal}DirectionY", f"T{signal}DirectionZ"], ["SignalParticleX", "SignalParticleY", "SignalParticleZ"], [f"T{signal}EndX", f"T{signal}EndY", f"T{signal}EndZ"]):
                run[tree][direction][idx] = run[tree][start][:,
                                                             None][idx] - run[tree][end][idx]

            run[tree][f"T{signal}DirectionMod"][idx] = np.sqrt(np.power(run[tree][f"T{signal}DirectionX"][idx], 2) + np.power(
                run[tree][f"T{signal}DirectionY"][idx], 2) + np.power(run[tree][f"T{signal}DirectionZ"][idx], 2))
            for coord in ["X", "Y", "Z"]:
                run[tree][f"T{signal}Direction{coord}"][idx] = run[tree][f"T{signal}Direction{coord}"][idx] / \
                    run[tree][f"T{signal}DirectionMod"][idx]

            run[tree][f"T{signal}Theta"][idx] = np.arccos(
                run[tree][f"T{signal}DirectionZ"][idx])
            run[tree][f"T{signal}Phi"][idx] = np.arctan2(
                run[tree][f"T{signal}DirectionY"][idx], run[tree][f"T{signal}DirectionX"][idx])

            run = remove_branches(
                run, rm_branches, [f"T{signal}DirectionMod"], tree=tree, debug=debug)

    output += f"\tMarley direction computation \t-> Done!\n"
    return run, output, new_branches


def compute_signal_energies(run, configs, params: Optional[dict] = None, trees=["Truth", "Reco"], rm_branches: bool = False, output: Optional[str] = None, debug=False):
    signal = "Signal"
    required_branches = {"Truth": ["Event", "Flag", "Geometry", "Version", "SignalParticleE", f"T{signal}PDG", f"T{signal}Mother", f"T{signal}E", f"T{signal}P", f"T{signal}K"],
                         "Reco": ["Event", "Flag", "Geometry", "Version", "SignalParticleE", f"T{signal}PDG", f"T{signal}Mother", f"T{signal}E", f"T{signal}P", f"T{signal}K"]}
    new_branches = [f"T{signal}SumE", f"T{signal}SumP",
                    f"T{signal}SumK", f"T{signal}K", f"T{signal}Mass"]
    if params is None:
        params = {"NORM_TO_NUE": False}
    for tree in trees:
        pdg_list = np.unique(run[tree][f"T{signal}PDG"]
                             [np.where(run[tree][f"T{signal}Mother"] == 0)])
        pdg_list = pdg_list[pdg_list != 0]
        mass_list = [Particle.from_pdgid(pdg).mass for pdg in pdg_list]
        run[tree][new_branches[0]] = np.zeros(
            (len(run[tree]["Event"]), len(pdg_list)), dtype=np.float32)
        run[tree][new_branches[1]] = np.zeros(
            (len(run[tree]["Event"]), len(pdg_list)), dtype=np.float32)
        run[tree][new_branches[2]] = np.zeros(
            (len(run[tree]["Event"]), len(pdg_list)), dtype=np.float32)
        run[tree][new_branches[3]] = np.zeros(
            (len(run[tree]["Event"]), len(run[tree][f"T{signal}PDG"][0])), dtype=np.float32)
        run[tree][new_branches[4]] = np.zeros(
            (len(run[tree]["Event"]), len(run[tree][f"T{signal}PDG"][0])), dtype=np.float32)

        full_pdg_list = np.unique(run[tree][f"T{signal}PDG"])
        for non_pdg in [0, 1000120249, 1000140289, 1000190419, 1000210499, 1000220489, 1000130279, 1000360809, 1000360829]:
            full_pdg_list = full_pdg_list[full_pdg_list != non_pdg]
        full_mass_dict = {pdg: Particle.from_pdgid(
            pdg).mass for pdg in full_pdg_list}

        # Gnearte branch for the mass of the particles frmo the T{signal}PDG m times n array and store it in the T{signal}Mass branch
        run[tree][f"T{signal}Mass"] = np.vectorize(
            full_mass_dict.get)(run[tree][f"T{signal}PDG"])
        run[tree][f"T{signal}Mass"] = np.nan_to_num(
            run[tree][f"T{signal}Mass"], nan=0.0, posinf=0.0, neginf=0.0)
        run[tree][f"T{signal}K"] = np.subtract(
            run[tree][f"T{signal}E"], run[tree][f"T{signal}Mass"])

        for idx, pdg in enumerate(pdg_list):
            run[tree][new_branches[0]][:, idx] = np.sum(
                run[tree][f"T{signal}E"]*(run[tree][f"T{signal}PDG"] == pdg)*(run[tree][f"T{signal}Mother"] == 0), axis=1)
            run[tree][new_branches[1]][:, idx] = np.sum(
                run[tree][f"T{signal}P"]*(run[tree][f"T{signal}PDG"] == pdg)*(run[tree][f"T{signal}Mother"] == 0), axis=1)
            run[tree][new_branches[2]][:, idx] = np.sum(
                run[tree][f"T{signal}K"]*(run[tree][f"T{signal}PDG"] == pdg)*(run[tree][f"T{signal}Mother"] == 0), axis=1)

        if params["NORM_TO_NUE"]:
            # Divide by the energy of the neutrino
            run[tree][new_branches[0]] = run[tree][new_branches[0]] / \
                run[tree]["SignalParticleE"][:, None]
            run[tree][new_branches[1]] = run[tree][new_branches[1]] / \
                run[tree]["SignalParticleE"][:, None]
            run[tree][new_branches[2]] = run[tree][new_branches[2]] / \
                run[tree]["SignalParticleE"][:, None]

        pdg_list = np.repeat(pdg_list, len(run[tree]["Event"])).reshape(
            len(pdg_list), len(run[tree]["Event"])).T
        run[tree][f"T{signal}SumPDG"] = pdg_list

        run = remove_branches(
            run, rm_branches, [f"T{signal}Mass"], tree=tree, debug=debug)

    output += f"\tSignal energy computation \t-> Done!\n"
    return run, output, new_branches


def compute_particle_energies(run, configs, params: Optional[dict] = None, trees: list[str] = ["Truth", "Reco"], rm_branches: bool = False, output: Optional[str] = None, debug: bool = False):
    """
    This functions looks into "TMarleyPDG" branch and combines the corresponding "TMarleyE" entries to get a total energy for each daughter particle.
    """
    signal = "Signal"
    required_branches = {"Truth": ["Event", "Flag", "Geometry", "Version", f"T{signal}PDG", f"T{signal}Mother", f"T{signal}E"],
                         "Reco": ["Event", "Flag", "Geometry", "Version", f"T{signal}PDG", f"T{signal}Mother", f"T{signal}E"]}
    
    particles_pdg = {"Electron": 11, "Gamma": 22,
                     "Neutron": 2112, "Neutrino": 12, "Proton": 2212}
    
    particles_mass = {particle: values for particle, values in zip(particles_pdg.keys(
    ), [Particle.from_pdgid(particles_pdg[particle]).mass for particle in particles_pdg])}
    
    particles_mass["Neutrino"] = 0
    particles_pdg_mass = {particles_pdg[particle]: particles_mass[particle]
                         for particle in particles_pdg}

    new_branches = list(particles_pdg.keys())
    for tree, particle in product(trees, particles_pdg):
        run[tree][f"{particle}E"] = np.zeros(
            len(run[tree]["Event"]), dtype=np.float32)
        run[tree][f"{particle}K"] = np.zeros(
            len(run[tree]["Event"]), dtype=np.float32)
        if len(run[tree][f"T{signal}PDG"][0]) > 0:
            try:
                run[tree][f"{signal}Mass"] = np.vectorize(
                    particles_pdg_mass.get)(run[tree][f"T{signal}PDG"])
            except TypeError:
                rprint(f"Failed pdg-mass computation: {particles_pdg_mass}")
                return run, output, new_branches
                
        else:
            run[tree][f"{signal}Mass"] = np.zeros(
                (len(run[tree][f"T{signal}PDG"]), len(run[tree][f"T{signal}PDG"][0])), dtype=bool)
        
        run[tree][f"{signal}Mass"] = np.nan_to_num(
            run[tree][f"{signal}Mass"], nan=0.0, posinf=0.0, neginf=0.0)

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
                run[tree][f"T{signal}E"][idx]
                * np.array(run[tree][f"T{signal}PDG"][idx]
                           == particles_pdg[particle]) * np.array(run[tree][f"T{signal}Mother"][idx] == 0),
                axis=1,
            )

            run[tree][f"{particle}K"][idx] = np.sum(
                np.subtract(run[tree][f"T{signal}E"][idx],
                            run[tree][f"{signal}Mass"][idx])
                * np.array(run[tree][f"T{signal}PDG"][idx]
                           == particles_pdg[particle]) * np.array(run[tree][f"T{signal}Mother"][idx] == 0),
                axis=1,
            )

            if params["NORM_TO_NUE"]:
                for particle in particles_pdg:
                    run[tree][f"{particle}E"][idx] = run[tree][f"{particle}E"][idx] / \
                        run[tree]["SignalParticleE"][idx]
                    run[tree][f"{particle}K"][idx] = run[tree][f"{particle}K"][idx] / \
                        run[tree]["SignalParticleE"][idx]

    run = remove_branches(run, rm_branches, [], debug=debug)
    output += f"\tParticle energy combination \t-> Done!\n"
    return run, output, new_branches