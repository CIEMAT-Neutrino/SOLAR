# ln -s /pc/choozdsk01/palomare/DUNE/SOLAR/data/ .

# Load function libraries and set up the environment
import sys

sys.path.insert(0, "../")
from lib.__init__ import *
from ROOT import TFile, TTree, TList

np.seterr(divide="ignore", invalid="ignore")
plt.rcParams.update({"font.size": 15})

# Load macro configuration
default_dict = {}
user_input = initialize_macro(
    "04Computing",
    ["config_file", "root_file", "rewrite", "debug"],
    default_dict=default_dict,
    debug=True,
)
user_input = check_macro_config(user_input, debug=user_input["debug"])

# Compute data: root -> numpy #
info = json.load(open("../config/" + f"{config}/{config}_config" + ".json", "r"))
config = user_input["config_file"].split("/")[-1].split("_config")[0]
params = get_param_dict(
    user_input["config_file"], in_params={}, debug=user_input["debug"]
)

# Start by opening the input file
print_colored("\nLoading data...", "DEBUG")
input_file = TFile(
    info["PATH"][0] + info["NAME"][0] + user_input["root_file"][0] + ".root"
)
folder_name = input_file.GetListOfKeys()[0].GetName()
tree = input_file.Get(folder_name + "/" + "SolarNuAnaTree")
print_colored("-> Found tree: %s" % tree.GetName(), "SUCCESS")

# Print list of branches
# print_colored("-> Branches:","SUCCESS")
# for branch in tree.GetListOfBranches():
#     print_colored("   - %s"%branch.GetName(),"INFO")

# Load calibration parameters
corr_info = read_input_file(
    config + "_charge_correction",
    path="../config/" + config + "/" + config + "_calib/",
    DOUBLES=["CHARGE_AMP", "ELECTRON_TAU"],
    debug=False,
)
corr_popt = [corr_info["CHARGE_AMP"][0], corr_info["ELECTRON_TAU"][0]]
calib_info = read_input_file(
    config + "_energy_calibration",
    path="../config/" + config + "/" + config + "_calib/",
    DOUBLES=["ENERGY_AMP", "INTERSECTION"],
    debug=False,
)

data = {
    "AdjClNum": np.zeros(tree.GetEntries(), dtype=np.int16),
    "AdjOpFlashNum": np.zeros(tree.GetEntries(), dtype=np.int16),
    "Primary": np.zeros(tree.GetEntries(), dtype=bool),
    "RecoY": -1e6 * np.ones(tree.GetEntries(), dtype=float),
    "Matching": -1 * np.ones(tree.GetEntries(), dtype=np.int8),
    "FlashMatched": np.zeros(tree.GetEntries(), dtype=bool),
    "AssFlashIdx": np.zeros(tree.GetEntries(), dtype=int),
    "MatchedOpFlashTime": np.zeros(tree.GetEntries(), dtype=float),
    "MatchedOpFlashPE": np.zeros(tree.GetEntries(), dtype=float),
    "MatchedOpFlashR": np.zeros(tree.GetEntries(), dtype=float),
    "DriftTime": np.zeros(tree.GetEntries(), dtype=float),
    "RecoX": -1e6 * np.zeros(tree.GetEntries(), dtype=float),
    "Correction": np.zeros(tree.GetEntries(), dtype=float),
    "Energy": np.zeros(tree.GetEntries(), dtype=float),
    "MaxAdjClEnergy": np.zeros(tree.GetEntries(), dtype=float),
    "TotalAdjClEnergy": np.zeros(tree.GetEntries(), dtype=float),
    "TotalEnergy": np.zeros(tree.GetEntries(), dtype=float),
    "RecoEnergy": np.zeros(tree.GetEntries(), dtype=float),
}


for i in track(range(tree.GetEntries()), description="Computing data..."):
    tree.GetEntry(i)
    # Basic Topology Computation
    data["AdjClNum"][i] = len(tree.AdjClR)
    data["AdjOpFlashNum"][i] = len(tree.AdjOpFlashR)

    ############################
    # Primary Computation
    ############################
    try:
        data["Primary"][i] = tree.Charge > max(tree.AdjClCharge)
    except ValueError:
        data["Primary"][i] = False

    ############################
    # RecoY Computation
    ############################
    if tree.Ind0NHits > 2 and tree.Ind1NHits > 2:
        data["RecoY"][i] = (tree.Ind0RecoY + tree.Ind1RecoY) / 2
        data["Matching"][i] = 2
    elif tree.Ind0NHits > 2 and tree.Ind1NHits <= 2:
        data["RecoY"][i] = tree.Ind0RecoY
        data["Matching"][i] = 0
    elif tree.Ind0NHits <= 2 and tree.Ind1NHits > 2:
        data["RecoY"][i] = tree.Ind1RecoY
        data["Matching"][i] = 1
    else:
        data["RecoY"][i] = (tree.Ind0RecoY + tree.Ind1RecoY) / 2
        print_colored("WARNING: No ind matching found for event %d" % i, "WARNING")

    ############################
    # Flash Matching Computation
    ############################
    for j in range(len(tree.AdjOpFlashR)):
        if tree.AdjOpFlashR.at(j) > params["MAX_FLASH_R"]:
            continue
        if tree.AdjOpFlashPE.at(j) < params["MIN_FLASH_PE"]:
            continue
        if (
            tree.AdjOpFlashPE.at(j) / tree.AdjOpFlashMaxPE.at(j)
            < tree.AdjOpFlashR.at(j) * params["RATIO_FLASH_PEvsR"]
        ):
            continue
        data["FlashMatched"][i] = True
        if data["MatchedOpFlashPE"][i] < tree.AdjOpFlashPE.at(j):
            data["AssFlashIdx"][i] = j
            data["MatchedOpFlashR"][i] = tree.AdjOpFlashR.at(j)
            data["MatchedOpFlashPE"][i] = tree.AdjOpFlashPE.at(j)
            data["MatchedOpFlashTime"][i] = tree.AdjOpFlashTime.at(j)
    data["DriftTime"][i] = tree.Time - 2 * data["MatchedOpFlashTime"][i]

    ############################
    # RecoX Computation
    ############################
    if info["GEOMETRY"][0] == "hd":
        if tree.TPC % 2 == 0:
            try:
                data["RecoX"][i] = (
                    abs(data[params["DEFAULT_RECOX_TIME"]][i])
                    * (info["DETECTOR_SIZE_X"][0] / 2)
                    / info["EVENT_TICKS"][0]
                )
            except KeyError:
                data["RecoX"][i] = (
                    abs(tree.Time)
                    * (info["DETECTOR_SIZE_X"][0] / 2)
                    / info["EVENT_TICKS"][0]
                )
        else:
            try:
                data["RecoX"][i] = (
                    -abs(data[params["DEFAULT_RECOX_TIME"]][i])
                    * (info["DETECTOR_SIZE_X"][0] / 2)
                    / info["EVENT_TICKS"][0]
                )
            except KeyError:
                data["RecoX"][i] = (
                    -abs(tree.Time)
                    * (info["DETECTOR_SIZE_X"][0] / 2)
                    / info["EVENT_TICKS"][0]
                )

    if info["GEOMETRY"][0] == "vd":
        try:
            data["RecoX"][i] = (
                -abs(data[params["DEFAULT_RECOX_TIME"]][i])
                * info["DETECTOR_SIZE_X"][0]
                / info["EVENT_TICKS"][0]
                + info["DETECTOR_SIZE_X"][0] / 2
            )
        except KeyError:
            data["RecoX"][i] = (
                -abs(tree.Time) * info["DETECTOR_SIZE_X"][0] / info["EVENT_TICKS"][0]
                + info["DETECTOR_SIZE_X"][0] / 2
            )

    ############################
    # Energy Computation
    ############################
    try:
        data["Correction"][i] = np.exp(
            np.abs(data[params["DEFAULT_ENERGY_TIME"]][i]) / corr_popt[1]
        )
    except KeyError:
        data["Correction"][i] = np.exp(np.abs(tree.Time) / corr_popt[1])
    data["Energy"][i] = tree.Charge * data["Correction"][i] / corr_popt[0]

    ############################
    # Reco Energy Computation
    ############################
    for z in range(len(tree.AdjClR)):
        try:
            adj_cl_correction = np.exp(
                np.abs(data[params["DEFAULT_ADJCL_ENERGY_TIME"]][i]) / corr_popt[1]
            )
        except KeyError:
            adj_cl_correction = np.exp(np.abs(tree.AdjClTime.at(z)) / corr_popt[1])
        adj_cl_energy = tree.AdjClCharge.at(z) * adj_cl_correction / corr_popt[0]
        if adj_cl_energy > data["MaxAdjClEnergy"][i]:
            data["MaxAdjClEnergy"][i] = adj_cl_energy
        data["TotalAdjClEnergy"][i] += adj_cl_energy
    data["TotalEnergy"][i] = data["TotalAdjClEnergy"][i] + data["Energy"][i] + 1.9
    if data["RecoEnergy"][i] > 1.5:
        data["Energy"][i] * calib_info["ENERGY_AMP"][0] + calib_info["INTERSECTION"][0]
    if data["RecoEnergy"][i] < 1.5:
        (
            data["Energy"][i] * calib_info["ENERGY_AMP"][0]
            + calib_info["INTERSECTION"][0]
            + 2.5
        )

# Save the data
print_colored("\nSaving data...", "DEBUG")
for key in data.keys():
    # Check if the output file already exists
    if (
        os.path.isfile(
            info["PATH"][0]
            + info["NAME"][0]
            + user_input["root_file"][0]
            + "/Reco/"
            + key
            + ".npy"
        )
        and not user_input["rewrite"]
    ):
        print_colored("File %s already exists, skipping..." % key, "WARNING")
        continue

    np.save(
        info["PATH"][0]
        + info["NAME"][0]
        + user_input["root_file"][0]
        + "/Reco/"
        + key
        + ".npy",
        data[key],
    )
    print_colored("--> %s saved" % key, "SUCCESS")
