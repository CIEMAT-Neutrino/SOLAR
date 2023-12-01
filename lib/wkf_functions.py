import json
import numpy as np

from ROOT import RDataFrame, TFile
from rich.progress import track
from rich import print as rprint
from lib.reco_functions import get_param_dict, generate_index


def compute_root_workflow(
    user_input, info, data_filter, workflow="BASICS", debug=False
):
    config = user_input["config_file"].split("/")[0]
    all_true, all_reco = {}, {}
    for name_idx, name in enumerate(user_input["root_file"]):
        debug_str = []
        debug_str.append("\nLoading %s data..." % (name))
        input_file = TFile(
            info["PATH"] + info["NAME"] + user_input["root_file"][0] + ".root"
        )
        folder_name = input_file.GetListOfKeys()[0].GetName()
        true_tree = input_file.Get(folder_name + "/" + "MCTruthTree")
        reco_tree = input_file.Get(folder_name + "/" + "SolarNuAnaTree")

        params = get_param_dict(
            user_input["config_file"], in_params={}, debug=user_input["debug"]
        )

        true, reco = {}, {}
        filter_idx = np.array([], dtype=int)
        reco_branches, new_reco_branches, true_branches, new_true_branches = (
            [],
            [],
            [],
            [],
        )
        object_types = {
            "INTEGERS": int,
            "DOUBLES": float,
            "STRINGS": str,
            "BOOLS": bool,
        }

        data_config = json.load(open("../lib/workflow/" + workflow + ".json", "r"))
        for object_type in object_types.keys():
            new_true_branches = new_true_branches + list(
                data_config["TRUE"][object_type].keys()
            )
            new_reco_branches = new_reco_branches + list(
                data_config["RECO"][object_type].keys()
            )

        for branch in new_true_branches:
            if branch in true_tree.GetListOfBranches():
                true_branches.append(branch)

        for branch in new_reco_branches:
            if branch in reco_tree.GetListOfBranches():
                reco_branches.append(branch)

        # try: root = root_numpy.tree2array(true_tree, branches=true_branches)
        try:
            root = RDataFrame(true_tree).AsNumpy(true_branches)
        except ValueError:
            pass
        debug_str.append("\nTRUTH:")
        for object_type in object_types.keys():
            for key in data_config["TRUE"][object_type]:
                true[key] = data_config["TRUE"][object_type][key] * root[key]
                if type(true[key][0]) == np.ndarray:
                    true[key] = np.vstack(true[key])
                debug_str.append(
                    "\n\t-> Found %s \t %s" % (key, object_types[object_type])
                )

        try:
            root = RDataFrame(reco_tree).AsNumpy(reco_branches)
        except ValueError:
            pass
        debug_str.append("\nRECO:")
        for object_type in object_types.keys():
            for key in data_config["RECO"][object_type]:
                try:
                    reco[key] = data_config["RECO"][object_type][key] * np.asarray(
                        root[key], dtype=object_types[object_type]
                    )
                    debug_str.append(
                        "\n\t-> Found %s \t %s" % (key, object_types[object_type])
                    )
                except ValueError:
                    reco[key] = data_config["RECO"][object_type][key] * np.ones(
                        reco_tree.GetEntries(), dtype=object_types[object_type]
                    )
                    branch_info = (
                        key,
                        object_types[object_type],
                        data_config["RECO"][object_type][key],
                    )
                    debug_str.append(
                        "\n\t-> Created %s \t %s with factor %.2e" % branch_info
                    )
                except KeyError:
                    reco[key] = data_config["RECO"][object_type][key] * np.ones(
                        reco_tree.GetEntries(), dtype=object_types[object_type]
                    )
                    branch_info = (
                        key,
                        object_types[object_type],
                        data_config["RECO"][object_type][key],
                    )
                    debug_str.append(
                        "\n\t-> WARNING: Key %s not found in reco tree!" % key
                    )
                    continue
        if debug:
            rprint("[magenta]" + "".join(debug_str) + "[/magenta]")

        if workflow in ["RECONSTRUCTION", "SMEARING", "VERTEXING", "ANALYSIS", "FULL"]:
            # calibration_info = read_input_file(config+"_charge_correction",path="../config/"+config+"/"+config+"_calib/",DOUBLES=["CHARGE_AMP","ELECTRON_TAU"],debug=user_input["debug"])
            calibration_info = json.load(
                open(
                    "../config/"
                    + config
                    + "/"
                    + config
                    + "_calib/"
                    + config
                    + "_charge_correction.json",
                    "r",
                )
            )
            corr_popt = [
                calibration_info["CHARGE_AMP"],
                calibration_info["ELECTRON_TAU"],
            ]

        if workflow in ["SMEARING", "VERTEXING", "ANALYSIS", "FULL"]:
            # calib_info = read_input_file(config+"_energy_calibration",path="../config/"+config+"/"+config+"_calib/",DOUBLES=["ENERGY_AMP","INTERSECTION"],debug=False)
            calib_info = json.load(
                open(
                    "../config/"
                    + config
                    + "/"
                    + config
                    + "_calib/"
                    + config
                    + "_energy_calibration.json",
                    "r",
                )
            )

        # debug_str.append("\n--> Computing reco efficiency...")
        true, reco = compute_event_matching(true, reco, debug=debug)
        for i in track(
            range(reco_tree.GetEntries()), description="Computing %s data..." % (name)
        ):
            reco_tree.GetEntry(i)
            try:
                reco["Primary"][i] = reco_tree.Charge > max(reco_tree.AdjClCharge)
            except ValueError:
                reco["Primary"][i] = False
            if workflow in [
                "CALIBRATION",
                "RECONSTRUCTION",
                "SMEARING",
                "VERTEXING",
                "ANALYSIS",
                "FULL",
            ]:
                ############################
                # Primary Computation
                ############################
                reco["AdjClNum"][i] = len(reco_tree.AdjClR)
                reco["AdjOpFlashNum"][i] = len(reco_tree.AdjOpFlashR)
                # print("Primary: ",reco["Primary"][i])

                ############################
                # True Computation
                ############################
                for j in range(len(reco_tree.TMarleyPDG)):
                    if (
                        reco_tree.TMarleyPDG[j] == 11
                        and 1e3 * reco_tree.TMarleyE[j] > reco["ElectronE"][i]
                    ):
                        reco["ElectronE"][i] = 1e3 * reco_tree.TMarleyE[j]
                    if reco_tree.TMarleyPDG[j] == 22:
                        reco["GammaE"][i] += 1e3 * reco_tree.TMarleyE[j]
                    if reco_tree.TMarleyPDG[j] == 2112:
                        reco["NeutronP"][i] += 1e3 * reco_tree.TMarleyP[j]
                reco["VisEnergy"][i] = reco["ElectronE"][i] + reco["GammaE"][i]

            if workflow in ["VERTEXING", "ANALYSIS", "FULL"]:
                ############################
                # RecoY Computation
                ############################
                if reco_tree.Ind0NHits > 2 and reco_tree.Ind1NHits > 2:
                    reco["RecoY"][i] = (reco_tree.Ind0RecoY + reco_tree.Ind1RecoY) / 2
                    reco["Matching"][i] = 2
                elif reco_tree.Ind0NHits > 2 and reco_tree.Ind1NHits <= 2:
                    reco["RecoY"][i] = reco_tree.Ind0RecoY
                    reco["Matching"][i] = 0
                elif reco_tree.Ind0NHits <= 2 and reco_tree.Ind1NHits > 2:
                    reco["RecoY"][i] = reco_tree.Ind1RecoY
                    reco["Matching"][i] = 1
                else:
                    reco["RecoY"][i] = (reco_tree.Ind0RecoY + reco_tree.Ind1RecoY) / 2

            if workflow in ["ANALYSIS", "FULL"]:
                ############################
                # Flash Matching Computation
                ############################
                for j in range(len(reco_tree.AdjOpFlashR)):
                    if reco_tree.AdjOpFlashR.at(j) > params["MAX_FLASH_R"]:
                        continue
                    if reco_tree.AdjOpFlashPE.at(j) < params["MIN_FLASH_PE"]:
                        continue
                    if (
                        reco_tree.AdjOpFlashPE.at(j) / reco_tree.AdjOpFlashMaxPE.at(j)
                        < reco_tree.AdjOpFlashR.at(j) * params["RATIO_FLASH_PEvsR"]
                    ):
                        continue
                    reco["FlashMatched"][i] = True
                    if reco["MatchedOpFlashPE"][i] < reco_tree.AdjOpFlashPE.at(j):
                        reco["AssFlashIdx"][i] = j
                        reco["MatchedOpFlashR"][i] = reco_tree.AdjOpFlashR.at(j)
                        reco["MatchedOpFlashPE"][i] = reco_tree.AdjOpFlashPE.at(j)
                        reco["MatchedOpFlashTime"][i] = reco_tree.AdjOpFlashTime.at(j)
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

            if workflow in ["RECONSTRUCTION", "SMEARING", "FULL"]:
                ############################
                # Energy Computation
                ############################
                try:
                    reco["Correction"][i] = np.exp(
                        np.abs(reco[params["DEFAULT_ENERGY_TIME"][0]][i]) / corr_popt[1]
                    )
                except KeyError:
                    reco["Correction"][i] = np.exp(
                        np.abs(reco_tree.Time) / corr_popt[1]
                    )
                reco["Energy"][i] = (
                    reco_tree.Charge * reco["Correction"][i] / corr_popt[0]
                )
                for z in range(len(reco_tree.AdjClR)):
                    try:
                        adj_cl_correction = np.exp(
                            np.abs(reco[params["DEFAULT_ADJCL_ENERGY_TIME"][0]][i])
                            / corr_popt[1]
                        )
                    except KeyError:
                        adj_cl_correction = np.exp(
                            np.abs(reco_tree.AdjClTime.at(z)) / corr_popt[1]
                        )
                    adj_cl_energy = (
                        reco_tree.AdjClCharge.at(z) * adj_cl_correction / corr_popt[0]
                    )
                    if adj_cl_energy > reco["MaxAdjClEnergy"][i]:
                        reco["MaxAdjClEnergy"][i] = adj_cl_energy
                    reco["TotalAdjClEnergy"][i] += adj_cl_energy
                reco["TotalEnergy"][i] = (
                    reco["TotalAdjClEnergy"][i] + reco["Energy"][i] + 1.9
                )

            if workflow in ["SMEARING", "FULL"]:
                ############################
                # Reco Energy Computation
                ############################
                if reco["RecoEnergy"][i] > 1.5:
                    (
                        reco["Energy"][i] * calib_info["ENERGY_AMP"]
                        + calib_info["INTERSECTION"]
                    )
                if reco["RecoEnergy"][i] < 1.5:
                    (
                        reco["Energy"][i] * calib_info["ENERGY_AMP"]
                        + calib_info["INTERSECTION"]
                        + 2.5
                    )

            ############################
            # Reco Filter Computation
            ############################
            try:
                if reco["Generator"][i] > data_filter["generator"]:
                    continue
            except KeyError:
                pass
            try:
                if reco["TNuE"][i] < data_filter["min_energy"]:
                    continue
                if reco["ElectronE"][i] < data_filter["min_energy"]:
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
                if data_filter["neutron"]:
                    for j in range(len(reco_tree.TMarleyPDG)):
                        if reco_tree.TMarleyPDG[j] == 2112:
                            continue
            except KeyError:
                pass

            # Fill data
            filter_idx = np.append(filter_idx, i)

        rprint("[green]-> Finished computing data![/green]")
        true["Geometry"], true["Version"], true["Name"] = (
            np.asarray([info["GEOMETRY"][0]] * len(true["Event"])),
            np.asarray([info["VERSION"][0]] * len(true["Event"])),
            np.asarray([name] * len(true["Event"])),
        )
        reco["Geometry"], reco["Version"], reco["Name"] = (
            np.asarray([info["GEOMETRY"][0]] * len(reco["Event"])),
            np.asarray([info["VERSION"][0]] * len(reco["Event"])),
            np.asarray([name] * len(reco["Event"])),
        )

        if name_idx == 0:
            all_true = true
            all_reco = reco
        else:
            for key in true.keys():
                all_true[key] = np.append(all_true[key], true[key])
            for key in reco.keys():
                all_reco[key] = np.append(all_reco[key], reco[key])

    return all_true, all_reco, filter_idx


def compute_event_matching(true, reco, debug=False):
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
