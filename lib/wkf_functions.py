import json
import pickle
import numpy as np
import pandas as pd

from ROOT import RDataFrame, TFile
from rich.progress import track
from rich import print as rprint
from particle import Particle

from .workflow.lib_default import get_default_info
from .workflow.functions import get_param_dict
from .workflow.lib_efficiency import generate_index

from src.utils import get_project_root
root = get_project_root()


def compute_root_workflow(
    user_input, info, data_filter, workflow="BASICS", debug=False
):
    root = get_project_root()
    config = user_input["config_file"].split("/")[0]
    all_conf, all_true, all_reco = {}, {}, {}
    for name_idx, name in enumerate(user_input["root_file"]):
        debug_str = []
        debug_str.append(f"\nLoading {name} data...")
        filename = f"{root}/{info['PATH']}{info['NAME']}{name}.root"
        input_file = TFile(filename, "READ")
        folder_name = input_file.GetListOfKeys()[0].GetName()
        conf_tree = input_file.Get(folder_name + "/" + "ConfigTree")
        true_tree = input_file.Get(folder_name + "/" + "MCTruthTree")
        reco_tree = input_file.Get(folder_name + "/" + "SolarNuAnaTree")

        params = get_param_dict(
            f"{root}/config/{config}/{config}_config.json", in_params={}, debug=user_input["debug"]
        )

        conf, true, reco = {}, {}, {}
        filter_idx = np.array([], dtype=int)

        reco_branches = []
        new_reco_branches = []
        true_branches = []
        new_true_branches = []
        conf_branches = []
        new_conf_branches = []

        object_types = {
            "INTEGERS": int,
            "DOUBLES": float,
            "STRINGS": str,
            "BOOLS": bool,
        }

        data_config = json.load(
            open(f"{root}/lib/workflow/{workflow}.json", "r"))
        for object_type in object_types.keys():
            new_conf_branches = new_conf_branches + list(
                data_config["Config"][object_type].keys()
            )
            new_true_branches = new_true_branches + list(
                data_config["Truth"][object_type].keys()
            )
            new_reco_branches = new_reco_branches + list(
                data_config["Reco"][object_type].keys()
            )

        for branch in new_conf_branches:
            if branch in conf_tree.GetListOfBranches():
                conf_branches.append(branch)

        for branch in new_true_branches:
            if branch in true_tree.GetListOfBranches():
                true_branches.append(branch)

        for branch in new_reco_branches:
            if branch in reco_tree.GetListOfBranches():
                reco_branches.append(branch)

        # try:
        #     root = RDataFrame(conf_tree).AsNumpy(conf_branches)
        # except ValueError:
        #     pass

        # debug_str.append("\nCONFIG:")
        # for object_type in object_types.keys():
        #     for key in data_config["CONFIG"][object_type]:
        #         conf[key] = data_config["CONFIG"][object_type][key] * np.asarray(
        #             conf_tree.GetBranch(key).GetLeaf(key).GetValue(0),
        #             dtype=object_types[object_type],
        #         )
        #         debug_str.append(
        #             "\n\t-> Found %s \t %s" % (key, object_types[object_type])
        #         )

        # try:
        #     truth_rdf = RDataFrame(true_tree).AsNumpy(true_branches)
        # except ValueError:
        #     pass

        debug_str.append("\nTruth:")
        for object_type in object_types.keys():
            for key in data_config["Truth"][object_type]:
                # try:
                #     true[key] = data_config["Truth"][object_type][key] * truth_rdf[key]
                #     if type(true[key][0]) == np.ndarray:
                #         true[key] = np.vstack(true[key])
                #     debug_str.append(
                #         "\n\t-> Found %s \t %s" % (key, object_types[object_type])
                #     )

                # try:
                true[key] = data_config["Truth"][object_type][key] * np.ones(
                    true_tree.GetEntries(), dtype=object_types[object_type]
                )
                debug_str.append(
                    f"\n\t-> Created {key} \t {object_types[object_type]} with factor {data_config['Truth'][object_type][key]}")

                # except UnboundLocalError:
                #     true[key] = data_config["Truth"][object_type][key] * np.ones(
                #         true_tree.GetEntries(), dtype=object_types[object_type]
                #     )
                #     debug_str.append(f"\n\t-> WARNING: Key {key} not found in true tree!")
                #     continue

        try:
            reco_rdf = RDataFrame(reco_tree).AsNumpy(reco_branches)
        except ValueError:
            rprint("[yellow]WARNING: reco_rdf not generated![/yellow]")
            pass

        debug_str.append("\nReco:")
        for object_type in object_types.keys():
            for key in data_config["Reco"][object_type]:
                try:
                    reco[key] = data_config["Reco"][object_type][key] * np.asarray(
                        reco_rdf[key], dtype=object_types[object_type]
                    )
                    debug_str.append(
                        f"\n\t-> Found {key} \t {object_types[object_type]}"
                    )

                except ValueError:
                    reco[key] = data_config["Reco"][object_type][key] * np.ones(
                        reco_tree.GetEntries(), dtype=object_types[object_type]
                    )
                    debug_str.append(
                        f"\n\t-> Created {key} \t {object_types[object_type]} with factor {data_config['Reco'][object_type][key]}"
                    )

                except KeyError:
                    reco[key] = data_config["Reco"][object_type][key] * np.ones(
                        reco_tree.GetEntries(), dtype=object_types[object_type]
                    )
                    debug_str.append(
                        f"\n\t-> WARNING: Key {key} not found in reco tree!")
                    continue
        if debug:
            rprint("[magenta]" + "".join(debug_str) + "[/magenta]")

        if workflow in ["CALIBRATION", "DISCRIMINATION", "RECONSTRUCTION", "SMEARING", "ANALYSIS"]:
            correction_info = json.load(
                open(
                    f"{root}/config/{config}/{name}/{config}_calib/{config}_charge_correction.json",
                    "r",
                )
            )
            corr_popt = [
                correction_info["CHARGE_AMP"],
                correction_info["ELECTRON_TAU"]
            ]

            def correction_amp(nhits):
                return correction_info["SLOPE"] * nhits + correction_info["INTERCEPT"]

        if workflow in ["DISCRIMINATION", "RECONSTRUCTION", "SMEARING", "ANALYSIS"]:
            calibration_info = json.load(
                open(
                    f"{root}/config/{config}/{name}/{config}_calib/{config}_charge_calibration.json",
                    "r",
                )
            )

            calib_popt = [
                calibration_info["CHARGE_AMP"],
                calibration_info["ELECTRON_TAU"]
            ]

            reco_popt = [
                calibration_info["SLOPE"],
                calibration_info["INTERCEPT"]
            ]

        if workflow in ["SMEARING", "VERTEXING", "ANALYSIS"]:
            calib_info = json.load(open(
                f"{root}/config/{config}/{name}/{config}_calib/{config}_energy_calibration.json", "r"))

        for i in track(
            range(true_tree.GetEntries()),
            description="Getting Truth %s data..." % (name),
        ):
            true_tree.GetEntry(i)
            true["Event"][i] = true_tree.Event
            true["Flag"][i] = true_tree.Flag
            if workflow in ["SMEARING"]:
                true["TNuE"][i] = true_tree.TNuE

        for i in track(
            range(reco_tree.GetEntries()),
            description=f"Getting Reco {name} data...",
        ):
            reco_tree.GetEntry(i)
            try:
                reco["Primary"][i] = reco_tree.Charge > max(
                    reco_tree.AdjClCharge)
            except ValueError:
                reco["Primary"][i] = False

            if workflow in ["CORRECTION", "CALIBRATION", "DISCRIMINATION", "RECONSTRUCTION", "SMEARING", "ANALYSIS"]:
                ############################
                # Primary Computation
                ############################
                reco["AdjClNum"][i] = len(reco_tree.AdjClR)
                try:
                    reco["AdjOpFlashNum"][i] = len(reco_tree.AdjOpFlashR)
                except AttributeError:
                    pass
                    ############################
                    # True Computation
                    ############################
                particles = {"Electron": 11, "Gamma": 22,
                             "Neutron": 2112, "Neutrino": 12, "Proton": 2212}
                particles_mass = {particle: values for particle, values in zip(particles.keys(
                ), [Particle.from_pdgid(particles[particle]).mass for particle in particles])}
                particles_mass["Neutrino"] = 0
                for j in range(len(reco_tree.TMarleyPDG)):
                    for particle in particles:
                        if (reco_tree.TMarleyPDG[j] == particles[particle]
                            and reco_tree.TMarleyMother[j] == 0
                            ):
                            try:
                                reco[particle + "E"][i] += reco_tree.TMarleyE[j]
                            except KeyError:
                                pass
                            try:
                                reco[particle + "P"][i] += reco_tree.TMarleyP[j]
                            except KeyError:
                                pass
                            try:
                                reco[particle + "K"][i] += reco_tree.TMarleyE[j] - \
                                    particles_mass[particle]
                            except KeyError:
                                pass

                reco["VisEnergy"][i] = reco["ElectronK"][i] + reco["GammaK"][i]
                reco["ElectronCharge"][i] = reco_tree.Charge
                for z in range(len(reco_tree.AdjClR)):
                    if reco_tree.AdjClGen.at(z) == 1 and reco_tree.AdjClMainPDG.at(z) == 11:
                        reco["ElectronCharge"][i] += reco_tree.AdjClCharge.at(
                            z)

            if workflow in ["ANALYSIS"]:
                ############################
                # Flash Matching Computation
                ############################
                if info["GEOMETRY"] == "hd":
                    for j in range(len(reco_tree.AdjOpFlashR)):
                        if reco_tree.AdjOpFlashR.at(j) > params["MAX_FLASH_R"]:
                            continue
                        if reco_tree.AdjOpFlashPE.at(j) < params["MIN_FLASH_PE"]:
                            continue
                        if reco_tree.AdjOpFlashPE.at(j) < 3000 * reco_tree.AdjOpFlashMaxPE.at(j) / reco_tree.AdjOpFlashPE.at(j):
                            continue
                        # max_ratio_filter = run["Reco"]["AdjOpFlashPE"][idx] > 3000 * run["Reco"]["AdjOpFlashMaxPE"][idx] / run["Reco"]["AdjOpFlashPE"][idx]

                        reco["FlashMatched"][i] = True
                        if reco["MatchedOpFlashPE"][i] < reco_tree.AdjOpFlashPE.at(j):
                            reco["AssFlashIdx"][i] = j
                            reco["MatchedOpFlashR"][i] = reco_tree.AdjOpFlashR.at(
                                j)
                            reco["MatchedOpFlashPE"][i] = reco_tree.AdjOpFlashPE.at(
                                j)
                            reco["MatchedOpFlashTime"][i] = reco_tree.AdjOpFlashTime.at(
                                j)
                    reco["DriftTime"][i] = (
                        reco_tree.Time - 2 * reco["MatchedOpFlashTime"][i]
                    )

                ############################
                # RecoX Computation
                ############################
                if info["GEOMETRY"] == "hd":
                    if reco_tree.TPC % 2 == 0:
                        try:
                            reco["RecoX"][i] = (
                                abs(reco[params["DEFAULT_RECOX_TIME"]][i])
                                * (info["DETECTOR_SIZE_X"] / 2)
                                / info["EVENT_TICKS"]
                            )
                        except KeyError:
                            reco["RecoX"][i] = (
                                abs(reco_tree.Time)
                                * (info["DETECTOR_SIZE_X"][0] / 2)
                                / info["EVENT_TICKS"]
                            )
                    else:
                        try:
                            reco["RecoX"][i] = (
                                -abs(reco[params["DEFAULT_RECOX_TIME"]][i])
                                * (info["DETECTOR_SIZE_X"] / 2)
                                / info["EVENT_TICKS"]
                            )
                        except KeyError:
                            reco["RecoX"][i] = (
                                -abs(reco_tree.Time)
                                * (info["DETECTOR_SIZE_X"] / 2)
                                / info["EVENT_TICKS"]
                            )

                if info["GEOMETRY"][0] == "vd":
                    try:
                        reco["RecoX"][i] = (
                            -abs(reco[params["DEFAULT_RECOX_TIME"][0]][i])
                            * info["DETECTOR_SIZE_X"]
                            / info["EVENT_TICKS"]
                            + info["DETECTOR_SIZE_X"] / 2
                        )
                    except KeyError:
                        reco["RecoX"][i] = (
                            -abs(reco_tree.Time)
                            * info["DETECTOR_SIZE_X"]
                            / info["EVENT_TICKS"]
                            + info["DETECTOR_SIZE_X"] / 2
                        )

            if workflow in ["CALIBRATION", "DISCRIMINATION", "RECONSTRUCTION", "SMEARING", "ANALYSIS"]:
                ############################
                # Energy Computation
                ############################
                try:
                    reco["Correction"][i] = np.exp(
                        np.abs(reco[params["DEFAULT_ENERGY_TIME"][0]]
                               [i]) / corr_popt[1]
                    )
                except KeyError:
                    reco["Correction"][i] = np.exp(
                        np.abs(reco_tree.Time) / corr_popt[1]
                    )
                reco["CorrectedCharge"][i] = reco_tree.Charge * \
                    reco["Correction"][i]

                reco["CorrectedChargePerMeV"][i] = reco["CorrectedCharge"][i] / \
                    reco["ElectronE"][i]

                reco["CorrectionFactor"][i] = correction_amp(reco_tree.NHits)

                reco["Energy"][i] = reco["CorrectedCharge"][i] / \
                    reco["CorrectionFactor"][i]

            if workflow in ["DISCRIMINATION", "RECONSTRUCTION", "SMEARING", "ANALYSIS"]:
                # reco["Energy"][i] = (reco["Energy"][i] -
                #                      reco_popt[1]) / reco_popt[0]

                for z in range(len(reco_tree.AdjClR)):
                    try:
                        adj_cl_correction = np.exp(
                            np.abs(
                                reco[params["DEFAULT_ADJCL_ENERGY_TIME"][0]][i])
                            / corr_popt[1]
                        )
                    except KeyError:
                        adj_cl_correction = np.exp(
                            np.abs(reco_tree.AdjClTime.at(z)) / corr_popt[1]
                        )
                    adj_cl_energy = (
                        reco_tree.AdjClCharge.at(
                            z) * adj_cl_correction / correction_amp(reco_tree.AdjClNHits.at(z))
                    )

                    adj_cl_r = reco_tree.AdjClR.at(z)
                    adj_cl_charge = reco_tree.AdjClCharge.at(z)
                    adj_cl_dt = abs(reco_tree.Time - reco_tree.AdjClTime.at(z))

                    if adj_cl_energy > reco["MaxAdjClEnergy"][i]:
                        reco["MaxAdjClEnergy"][i] = adj_cl_energy
                        reco["MaxAdjClR"][i] = adj_cl_r

                    reco["TotalAdjClEnergy"][i] += adj_cl_energy
                    reco["TotalAdjClR"][i] += adj_cl_r

                    if adj_cl_r > info["MIN_BKG_R"] and adj_cl_dt > info["MIN_BKG_DT"] and adj_cl_charge < info["MAX_BKG_CHARGE"]:
                        continue
                    else:
                        reco["SelectedAdjClNum"][i] += 1
                        reco["SelectedAdjClEnergy"][i] += adj_cl_energy
                        reco["SelectedAdjClR"][i] += adj_cl_r

                        if adj_cl_energy > reco["SelectedMaxAdjClEnergy"][i]:
                            reco["SelectedMaxAdjClEnergy"][i] = adj_cl_energy
                            reco["SelectedMaxAdjClR"][i] = adj_cl_r

                reco["TotalEnergy"][i] = (
                    reco["TotalAdjClEnergy"][i] + reco["Energy"][i])
                reco["SelectedEnergy"][i] = (
                    reco["SelectedAdjClEnergy"][i] + reco["Energy"][i])
                reco["SelectedAdjClEnergyRatio"][i] = reco["SelectedAdjClEnergy"][i] / \
                    reco["Energy"][i]

            if workflow in ["SMEARING", "ANALYSIS"]:
                ###########################
                # Reco Energy Computation
                ###########################
                for energy_label, energy_key in zip(["TotalEnergy", "SolarEnergy", "SelectedEnergy"], ["TOTAL", "RECO", "SELECTED"]):
                    reco[energy_label][i] = (
                        (reco[energy_label][i] - calib_info[energy_key]
                         ["INTERSECTION"]) / calib_info[energy_key]["ENERGY_AMP"]
                    )

            ############################
            # Reco Filter Computation
            ############################
            try:
                if reco["Generator"][i] != data_filter["generator"]:
                    continue
            except KeyError:
                pass
            try:
                if reco["TNuE"][i] < data_filter["min_energy"]:
                    continue
                if reco["ElectronE"][i] < data_filter["min_energy"]:
                    continue
                if data_filter["min_charge_per_energy"]:
                    if reco["Charge"][i] / reco["ElectronE"][i] < correction_info["CHARGE_PER_ENERGY_TRIMM"]:
                        continue
            except KeyError:
                pass
            try:
                if reco["TNuE"][i] > data_filter["max_energy"]:
                    continue
                if reco["ElectronE"][i] > data_filter["max_energy"]:
                    continue
            except KeyError:
                pass
            try:
                if reco["NHits"][i] < data_filter["pre_nhits"]:
                    continue
            except KeyError:
                pass
            try:
                if reco["Primary"][i] == False and data_filter["primary"]:
                    continue
            except KeyError:
                pass
            try:
                if data_filter["nan"]:
                    for variable in reco_branches:
                        if np.isnan(reco[variable][i]) or np.isinf(reco[variable][i]):
                            continue
            except KeyError:
                pass
            try:
                if reco["FlashMatched"][i] == False and data_filter["flash-matched"]:
                    continue
            except KeyError:
                pass
            try:
                if data_filter["neutron"]:
                    for j in range(len(reco_tree.TMarleyPDG)):
                        if reco_tree.TMarleyPDG[j] == 2112:
                            continue
            except KeyError:
                pass

            # Fill data
            filter_idx = np.append(filter_idx, i)

        rprint("[green]-> Finished computing data![/green]")
        for tree, data in zip(
            [conf_tree, true_tree, reco_tree], [conf, true, reco]
        ):
            data["Geometry"], data["Version"], data["Name"] = (
                np.asarray([info["GEOMETRY"]] * tree.GetEntries()),
                np.asarray([info["VERSION"]] * tree.GetEntries()),
                np.asarray([name] * tree.GetEntries()),
            )

        if name_idx == 0:
            print(f"First data file {name} computed!")
            all_conf, all_true, all_reco = conf, true, reco

        else:
            print(f"Additional data file {name} computed!")
            for key in conf.keys():
                all_conf[key] = np.append(all_conf[key], conf[key])
            for key in true.keys():
                all_true[key] = np.append(all_true[key], true[key])
            for key in reco.keys():
                all_reco[key] = np.append(all_reco[key], reco[key])

    return all_true, all_reco, filter_idx


def compute_event_matching(true, reco, debug=False):
    '''
    Use the event and flag keys to match the true and reco datadicts.
    '''
    # Check if true and reco have the keys event, flag
    (
        true["RecoIndex"],
        true["RecoMatch"],
        true["ClCount"],
        true["HitCount"],
        reco["TrueIndex"],
    ) = generate_index(
        true["Event"],
        true["Flag"],
        reco["Event"],
        reco["Flag"],
        reco["NHits"],
        reco["Charge"],
        reco["Generator"],
        debug=debug,
    )
    return true, reco
