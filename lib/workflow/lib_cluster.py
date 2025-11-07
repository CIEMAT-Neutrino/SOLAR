import json
import pickle
import numpy as np

from typing import Optional
from itertools import product
from rich import print as rprint
from lib.workflow.functions import remove_branches, get_param_dict, reshape_array
from lib.workflow.lib_default import get_default_info
from lib.fit_functions import calibration_func
from lib.df_functions import npy2df

from src.utils import get_project_root

root = get_project_root()


def compute_electron_cluster(
    run: dict,
    configs: dict[str, list[str]],
    params: Optional[dict] = None,
    rm_branches: bool = False,
    output: Optional[str] = None,
    debug: bool = False,
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
        info, this_params, output = get_param_dict(
            f"{root}/config/{config}/{config}", params, output, debug=debug
        )
        idx = np.where(
            (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
            * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
        )
        run["Reco"]["ElectronCharge"][idx] = run["Reco"]["Charge"][idx] + np.sum(
            np.where(
                (run["Reco"]["AdjClMainPDG"][idx] == 11),
                run["Reco"]["AdjClCharge"][idx],
                0,
            ),
            axis=1,
        )
        # Compute the average weighted time of the clusters
        run["Reco"]["ElectronTime"][idx] = (
            (run["Reco"]["Charge"][idx] * run["Reco"]["Time"][idx])
            + np.sum(
                np.where(
                    (run["Reco"]["AdjClMainPDG"][idx] == 11),
                    run["Reco"]["AdjClCharge"][idx] * run["Reco"]["AdjClTime"][idx],
                    0,
                ),
                axis=1,
            )
        ) / run["Reco"]["ElectronCharge"][idx]

    run = remove_branches(run, rm_branches, [], debug=debug)
    output += f"\tCluster charge computation\t-> Done!\n"
    return run, output, new_branches


def compute_cluster_energy(
    run: dict,
    configs: dict,
    params: Optional[dict] = None,
    clusters: list[str] = [""],
    rm_branches: bool = False,
    output: Optional[str] = None,
    debug: bool = False,
) -> tuple[dict, str]:
    """
    Correct the charge of the events in the run according to the correction file.
    """
    default_sample = "marley"
    # New branches
    new_branches = ["Correction", "CorrectionFactor"]
    for cluster in clusters:
        new_branches.append(f"Corrected{cluster}Charge")
        new_branches.append(f"{cluster}Energy")
    new_vector_branches = [
        "AdjClCorrection",
        "AdjClCorrectedCharge",
        "AdjClCorrectionFactor",
        "AdjClEnergy",
    ]

    for branch in new_branches:
        run["Reco"][branch] = np.zeros(len(run["Reco"]["Event"]), dtype=np.float32)
    for branch in new_vector_branches:
        run["Reco"][branch] = np.ones(
            (len(run["Reco"]["Event"]), len(run["Reco"]["AdjClCharge"][0])),
            dtype=np.float32,
        )

    for config in configs:
        info, this_params, output = get_param_dict(
            f"{root}/config/{config}/{config}", params, output, debug=debug
        )

        for name in configs[config]:
            idx = np.where(
                (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
                * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
                * (np.asarray(run["Reco"]["Name"]) == name)
            )

            try:
                corr_info = json.load(
                    open(
                        f"{root}/config/{config}/{name}/{config}_calib/{config}_electroncharge_correction.json",
                        "r",
                    )
                )

            except FileNotFoundError:
                output += f"\t[yellow]***[WARNING] Correction file not found for {config} {name}. Defaulting to {default_sample}![/yellow]\n"
                corr_info = json.load(
                    open(
                        f"{root}/config/{config}/{default_sample}/{config}_calib/{config}_electroncharge_correction.json",
                        "r",
                    )
                )

            drift_popt = [corr_info["CHARGE_AMP"], corr_info["ELECTRON_TAU"]]
            corr_popt = [
                corr_info["CORRECTION_AMP"],
                corr_info["CORRECTION_DECAY"],
                corr_info["CORRECTION_CONST"],
                corr_info["CORRECTION_SIGMOID"],
            ]
            print(
                f'Using correction variables: {this_params["DEFAULT_ENERGY_TIME"]}, {this_params["DEFAULT_ADJCL_ENERGY_TIME"]}'
            )
            for branch, default_branch in zip(
                ["Correction", "AdjClCorrection"],
                [
                    this_params["DEFAULT_ENERGY_TIME"],
                    this_params["DEFAULT_ADJCL_ENERGY_TIME"],
                ],
            ):

                run["Reco"][branch][idx] = np.exp(
                    np.abs(run["Reco"][default_branch][idx]) / drift_popt[1]
                )

            run["Reco"]["CorrectionFactor"][idx] = calibration_func(
                run["Reco"]["NHits"][idx], *corr_popt
            )

            for cluster in clusters:
                run["Reco"][f"Corrected{cluster}Charge"][idx] = (
                    run["Reco"][f"{cluster}Charge"][idx]
                    * run["Reco"]["Correction"][idx]
                )

                run["Reco"][f"{cluster}Energy"][idx] = (
                    run["Reco"][f"Corrected{cluster}Charge"][idx]
                    / run["Reco"]["CorrectionFactor"][idx]
                )

            run["Reco"]["AdjClCorrectedCharge"][idx] = (
                run["Reco"]["AdjClCharge"][idx] * run["Reco"]["AdjClCorrection"][idx]
            )

            run["Reco"]["AdjClEnergy"][idx] = (
                run["Reco"]["AdjClCorrectedCharge"][idx] / corr_info["CHARGE_AMP"]
            )

    run = remove_branches(
        run, rm_branches, new_branches[:-1] + new_vector_branches[:-1], debug=debug
    )
    output += f"\tCluster energy computation\t-> Done!\n"
    return run, output, new_branches + new_vector_branches


def compute_cluster_calibration(
    run,
    configs: dict[str, list[str]],
    params: Optional[dict] = None,
    clusters: list[str] = ["", "Electron"],
    rm_branches: bool = False,
    output: Optional[str] = None,
    debug=False,
):
    """
    Correct the charge of the events in the run according to the correction file.
    """
    default_sample = "marley"
    new_branches = ["EnergySlope", "EnergyIntercept"]
    for cluster, branch in product(clusters, new_branches):
        run["Reco"][f"{cluster}{branch}"] = np.zeros(len(run["Reco"]["Event"]))

    for config in configs:
        info, this_params, output = get_param_dict(
            f"{root}/config/{config}/{config}", params, output, debug=debug
        )

        for name in configs[config]:
            idx = np.where(
                (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
                * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
                * (np.asarray(run["Reco"]["Name"]) == name)
            )

            for cluster in clusters:
                try:
                    corr_slope = pickle.load(
                        open(
                            f"{root}/config/{config}/{name}/{config}_calib/{config}_{cluster.lower()}charge_slope_calibration.pkl",
                            "rb",
                        )
                    )
                    corr_intercept = pickle.load(
                        open(
                            f"{root}/config/{config}/{name}/{config}_calib/{config}_{cluster.lower()}charge_intercept_calibration.pkl",
                            "rb",
                        )
                    )

                except FileNotFoundError:
                    rprint(
                        f"[red][ERROR][/red] Calibration file {root}/config/{config}/{name}/{config}_calib/{config}_{cluster.lower()}charge_calibration.pkl not found. Defaulting to {default_sample}!"
                    )
                    corr_slope = pickle.load(
                        open(
                            f"{root}/config/{config}/{default_sample}/{config}_calib/{config}_{cluster.lower()}charge_slope_calibration.pkl",
                            "rb",
                        )
                    )
                    corr_intercept = pickle.load(
                        open(
                            f"{root}/config/{config}/{default_sample}/{config}_calib/{config}_{cluster.lower()}charge_intercept_calibration.pkl",
                            "rb",
                        )
                    )

                run["Reco"][f"{cluster}EnergySlope"][idx] = corr_slope(
                    run["Reco"]["NHits"][idx]
                )
                run["Reco"][f"{cluster}EnergyIntercept"][idx] = corr_intercept(
                    run["Reco"]["NHits"][idx]
                )

                run["Reco"][f"{cluster}Energy"][idx] = (
                    run["Reco"][f"{cluster}Energy"][idx]
                    - run["Reco"][f"{cluster}EnergyIntercept"][idx]
                ) / run["Reco"][f"{cluster}EnergySlope"][idx]

    run = remove_branches(run, rm_branches, [], debug=debug)
    output += f"\tClutser energy computation\t-> Done!\n"
    return run, output, new_branches


def compute_total_energy(
    run: dict,
    configs: dict[str, list[str]],
    params: Optional[dict] = None,
    rm_branches: bool = False,
    output: Optional[str] = None,
    debug=False,
):
    """
    Compute the total energy of the events in the run.
    """
    new_branches = [
        "SelectedAdjClNum",
        "SolarEnergy",
        "TotalEnergy",
        "SelectedEnergy",
        "TotalAdjClEnergy",
        "SelectedAdjClEnergy",
        "SelectedMaxAdjClEnergy",
        "SelectedAdjClEnergyRatio",
        "Discriminant",
    ]
    for branch in new_branches:
        run["Reco"][branch] = np.zeros(len(run["Reco"]["Event"]))

    for config in configs:
        info, this_params, output = get_param_dict(
            f"{root}/config/{config}/{config}", params, output, debug=debug
        )
        idx = np.where(
            (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
            * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
        )

        run["Reco"]["TotalAdjClEnergy"][idx] = np.sum(
            run["Reco"]["AdjClEnergy"][idx], axis=1
        )

        selected_filter = (run["Reco"]["AdjClR"][idx] < this_params["MIN_BKG_R"]) + (
            run["Reco"]["AdjClCharge"][idx] > this_params["MAX_BKG_CHARGE"]
        )

        if debug:
            output += f"\t***[cyan][INFO][/cyan] Selected filter for energy computation excludes {100*((np.sum(run['Reco']['AdjClNum'][idx])-np.sum(~selected_filter))/np.sum(run['Reco']['AdjClNum'][idx])):.1f}% of Adj. clusters\n"

        run["Reco"]["SelectedAdjClNum"][idx] = np.sum(selected_filter, axis=1)
        run["Reco"]["SelectedAdjClEnergy"][idx] = np.sum(
            run["Reco"]["AdjClEnergy"][idx], where=selected_filter, axis=1
        )
        run["Reco"]["SelectedMaxAdjClEnergy"][idx] = np.max(
            run["Reco"]["AdjClEnergy"][idx], where=selected_filter, axis=1, initial=0
        )

        run["Reco"]["TotalEnergy"][idx] = (
            run["Reco"]["Energy"][idx] + run["Reco"]["TotalAdjClEnergy"][idx]
        )
        run["Reco"]["SelectedEnergy"][idx] = (
            run["Reco"]["Energy"][idx] + run["Reco"]["SelectedAdjClEnergy"][idx]
        )
        run["Reco"]["SelectedAdjClEnergyRatio"][idx] = (
            run["Reco"]["SelectedAdjClEnergy"][idx] / run["Reco"]["Energy"][idx]
        )

    run = remove_branches(
        run, rm_branches, ["TotalAdjClEnergy", "SelectedAdjClEnergy"], debug=debug
    )
    output += f"\tTotal energy computation \t-> Done!\n"
    return run, output, new_branches


def compute_reco_energy(
    run,
    configs,
    params: Optional[dict] = None,
    rm_branches: bool = False,
    output: Optional[str] = None,
    debug=False,
):
    features = get_default_info(root, "ML_FEATURES")
    new_branches = ["SolarEnergy", "Upper", "Lower"]
    for branch in new_branches:
        run["Reco"][branch] = np.zeros(len(run["Reco"]["Event"]))

    for config in configs:
        info, this_params, output = get_param_dict(
            f"{root}/config/{config}/{config}", params, output, debug=debug
        )
        for name in configs[config]:
            idx = np.where(
                (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
                * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
                * (np.asarray(run["Reco"]["Name"]) == name)
            )

            # Save the trained model to a file so it can be used later using pickle
            default_sample = "marley"
            try:
                path = f"{root}/config/{config}/{name}/models/{config}_{name}_random_forest_discriminant.pkl"
                with open(path, "rb") as model_file:
                    rf_classifier = pickle.load(model_file)

            except FileNotFoundError:
                rprint(
                    f"[red][ERROR][/red] ML model file {root}/config/{config}/{name}/models/{config}_{name}_random_forest_discriminant.pkl not found. Defaulting to {default_sample}!\n"
                )
                path = f"{root}/config/{config}/{default_sample}/models/{config}_{default_sample}_random_forest_discriminant.pkl"

                with open(path, "rb") as model_file:
                    rf_classifier = pickle.load(model_file)

            try:
                discriminant_info = json.load(
                    open(
                        f"{root}/config/{config}/{name}/{config}_calib/{config}_discriminant_calibration.json",
                        "r",
                    )
                )
                output += f"\t***[cyan][INFO][/cyan] Loading model for {name}\n"

            except FileNotFoundError:
                discriminant_info = json.load(
                    open(
                        f"{root}/config/{config}/{default_sample}/{config}_calib/{config}_discriminant_calibration.json",
                        "r",
                    )
                )
                output += f"\t[yellow]***[WARNING][/yellow] Loading model for {default_sample}\n"

            def upper_func(x):
                return x - discriminant_info["UPPER"]["OFFSET"]

            def lower_func(x):
                return x - discriminant_info["LOWER"]["OFFSET"]

            thld = discriminant_info["DISCRIMINANT_THRESHOLD"]
            try:
                run["Reco"]["Upper"][idx] = np.asarray(
                    run["Reco"]["ElectronK"][idx]
                    > run["Reco"]["SignalParticleK"][idx] + thld,
                    dtype=bool,
                )
                run["Reco"]["Lower"][idx] = np.asarray(
                    run["Reco"]["ElectronK"][idx]
                    < run["Reco"]["SignalParticleK"][idx] + thld,
                    dtype=bool,
                )
            except KeyError:
                pass

            df = npy2df(
                run,
                "Reco",
                branches=features
                + [
                    "Primary",
                    "Generator",
                    "SignalParticleK",
                    "NHits",
                    "Upper",
                    "Lower",
                ],
                debug=debug,
            )

            try:
                df["ML"] = rf_classifier.predict(df[features])
                df["Discriminant"] = rf_classifier.predict_proba(df[features])[:, 1]
            except ValueError:
                print(df[features])
                raise ValueError

            upper_idx = df["Discriminant"] >= discriminant_info["ML_THRESHOLD"]
            lower_idx = df["Discriminant"] < discriminant_info["ML_THRESHOLD"]

            df.loc[upper_idx, "SolarEnergy"] = upper_func(df.loc[upper_idx, "Energy"])
            df.loc[lower_idx, "SolarEnergy"] = lower_func(df.loc[lower_idx, "Energy"])
            run["Reco"]["SolarEnergy"][idx] = df["SolarEnergy"].values

    run = remove_branches(run, rm_branches, [], debug=debug)
    output += f"\tReco energy computation \t-> Done!\n"
    return run, output, new_branches


def compute_energy_calibration(
    run,
    configs,
    params: Optional[dict] = None,
    rm_branches: bool = False,
    output: Optional[str] = None,
    debug=False,
):
    # Inverse function of polinomyal with degree 2
    def inverse_quadratic(x, a, b, c):
        return (-b + np.sqrt(b**2 - 4 * a * (c - x))) / (2 * a)

    default_sample = "marley"
    new_branches = ["ClusterEnergy", "SolarEnergy", "SelectedEnergy", "TotalEnergy"]

    run["Reco"]["ClusterEnergy"] = np.zeros(len(run["Reco"]["Event"]))
    for config in configs:
        info, this_params, output = get_param_dict(
            f"{root}/config/{config}/{config}", params, output, debug=debug
        )

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
                    output += f"\t***[cyan][INFO][/cyan] Applying energy calibration from {name}\n"

            except FileNotFoundError:
                reco_info = json.load(
                    open(
                        f"{root}/config/{config}/{default_sample}/{config}_calib/{config}_{default_sample}_energy_calibration.json",
                        "r",
                    )
                )
                output += f"\t[yellow]***[WARNING] Applying default energy calibration from {default_sample}[/yellow]\n"

            for energy, ref_energy in zip(
                ["Solar", "Selected", "Total", "Cluster"],
                ["Solar", "Selected", "Total", ""],
            ):
                if this_params["SAMPLE_FIT"][f"{energy}Energy"] == "linear":
                    run["Reco"][f"{energy}Energy"][idx] = (
                        run["Reco"][f"{ref_energy}Energy"][idx]
                        - reco_info[energy.upper()]["INTERSECTION"]
                    ) / reco_info[energy.upper()]["ENERGY_AMP"]
                    output += f"\t[cyan]***[cyan][INFO][/cyan] Applying linear energy calibration for {energy}Energy\n"

                elif this_params["SAMPLE_FIT"][f"{energy}Energy"] == "quadratic":
                    run["Reco"][f"{energy}Energy"][idx] = inverse_quadratic(
                        run["Reco"][f"{ref_energy}Energy"][idx],
                        reco_info[energy.upper()]["ENERGY_CURVATURE"],
                        reco_info[energy.upper()]["ENERGY_AMP"],
                        reco_info[energy.upper()]["INTERSECTION"],
                    )
                    output += f"\t***[cyan][INFO][/cyan] Applying quadratic energy calibration for {energy}Energy\n"

    run = remove_branches(run, rm_branches, [], debug=debug)
    output += f"\tReco energy calibration \t-> Done!\n"
    return run, output, new_branches


def compute_cluster_time(
    run: dict,
    configs: dict[str, list[str]],
    params: Optional[dict] = None,
    rm_branches: bool = False,
    output: Optional[str] = None,
    debug: bool = False,
):
    """
    Correct the charge of the events in the run according to the correction file.
    """
    # New branches
    new_branches = ["RecoDriftTime", "AverageDriftTime"]
    new_vector_branches = ["AdjCldTime", "AdjClRecoDriftTime", "AdjClAverageDriftTime"]

    for branch in new_branches:
        # Check if the branch exists
        if branch in run["Reco"]:
            output += f"\t[yellow]***[WARNING] Branch {branch} already exists! Overwriting... [/yellow]\n"

        if branch == "AverageDriftTime":
            run["Reco"][branch] = np.ones(len(run["Reco"]["Event"]), dtype=np.float32)
        else:
            run["Reco"][branch] = np.zeros(len(run["Reco"]["Event"]), dtype=np.float32)

    for branch in new_vector_branches:
        # Check if the branch exists
        if branch in run["Reco"]:
            output += f"\t[yellow]***[WARNING] Branch {branch} already exists! Overwriting... [/yellow]\n"

        if branch == "AdjClAverageDriftTime":
            run["Reco"][branch] = np.ones(
                (len(run["Reco"]["Event"]), len(run["Reco"]["AdjClCharge"][0])),
                dtype=np.float32,
            )
        else:
            run["Reco"][branch] = np.zeros(
                (len(run["Reco"]["Event"]), len(run["Reco"]["AdjClCharge"][0])),
                dtype=np.float32,
            )

    for config in configs:
        info, this_params, output = get_param_dict(
            f"{root}/config/{config}/{config}", params, output, debug=debug
        )

        idx = np.where(
            (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
            * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
        )

        # Check if idx is empty
        if len(idx[0]) == 0:
            output += f"\t[yellow]***[WARNING] No events found with flash-match information! Skipping drift-time computation.[/yellow]\n"
            return run, output, new_branches + new_vector_branches

        reco_time_array = reshape_array(
            run["Reco"]["Time"][idx], len(run["Reco"]["AdjClTime"][idx][0])
        )

        run["Reco"]["AdjCldTime"][idx] = run["Reco"]["AdjClTime"][idx] - reco_time_array

        if info["GEOMETRY"] == "hd":
            run["Reco"]["RecoDriftTime"][idx] = (
                np.absolute(run["Reco"]["RecoX"][idx])
                * info["TIMEWINDOW"]
                * 1e6
                / info["DETECTOR_MAX_X"]
            )

            reco_drift_array = reshape_array(
                run["Reco"]["RecoDriftTime"][idx], len(run["Reco"]["AdjClTime"][idx][0])
            )

            run["Reco"]["AdjClRecoDriftTime"][idx] = (
                run["Reco"]["AdjCldTime"][idx] + reco_drift_array
            )

        if info["GEOMETRY"] == "vd":
            run["Reco"]["RecoDriftTime"][idx] = (
                (info["DETECTOR_SIZE_X"] / 2 - run["Reco"]["RecoX"][idx])
                * info["TIMEWINDOW"]
                * 1e6
                / info["DETECTOR_SIZE_X"]
            )

            reco_drift_array = reshape_array(
                run["Reco"]["RecoDriftTime"][idx], len(run["Reco"]["AdjClTime"][idx][0])
            )

            run["Reco"]["AdjClRecoDriftTime"][idx] = (
                run["Reco"]["AdjCldTime"][idx] + reco_drift_array
            )

        run["Reco"]["AverageDriftTime"][idx] *= 0.5 * info["TIMEWINDOW"] * 1e6
        run["Reco"]["AdjClAverageDriftTime"][idx] *= 0.5 * info["TIMEWINDOW"] * 1e6

    run = remove_branches(
        run, rm_branches, new_branches[:-1] + new_vector_branches[:-1], debug=debug
    )
    output += f"\tClutser time computation\t-> Done!\n"
    return run, output, new_branches + new_vector_branches


def compute_cluster_estimated_time(
    run: dict,
    configs: dict[str, list[str]],
    params: Optional[dict] = None,
    rm_branches: bool = False,
    output: Optional[str] = None,
    debug: bool = False,
):
    """
    Correct the charge of the events in the run according to the correction file.
    """
    # New branches
    new_branches = [
        "RandomX",
        "EstimatedRecoX",
        "RandomDriftTime",
        "EstimatedRecoDriftTime",
    ]
    new_vector_branches = [
        "RandomAdjClX",
        "EstimatedRecoAdjClX",
        "RandomAdjClDriftTime",
        "EstimatedRecoAdjClDriftTime",
    ]

    for branch in new_branches:
        # Check if the branch exists
        if branch in run["Reco"]:
            output += f"\t[yellow]***[WARNING] Branch {branch} already exists! Overwriting... [/yellow]\n"

        run["Reco"][branch] = np.zeros(len(run["Reco"]["Event"]), dtype=np.float32)

    for branch in new_vector_branches:
        # Check if the branch exists
        if branch in run["Reco"]:
            output += f"\t[yellow]***[WARNING] Branch {branch} already exists! Overwriting... [/yellow]\n"

        run["Reco"][branch] = np.ones(
            (len(run["Reco"]["Event"]), len(run["Reco"]["AdjClCharge"][0])),
            dtype=np.float32,
        )

    for config in configs:
        info, this_params, output = get_param_dict(
            f"{root}/config/{config}/{config}", params, output, debug=debug
        )
        idx = np.where(
            (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
            * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
        )

        # Divide the filter idx into two groups, one with the random correction and one without
        random_idx = idx[0][: int(len(idx[0]) * this_params["RANDOM_CORRECTION_RATIO"])]

        default_idx = idx[0][
            int(len(idx[0]) * this_params["RANDOM_CORRECTION_RATIO"]) :
        ]

        if info["GEOMETRY"] == "hd":
            # Create array of flat random values between -DETECTOR_SIZE_X/2 and DETECTOR_SIZE_X/2
            run["Reco"]["RandomX"][idx] = (
                np.random.rand(len(run["Reco"]["Event"][idx])) * info["DETECTOR_SIZE_X"]
                - info["DETECTOR_SIZE_X"] / 2
            )

            # Create a 2D array of random values between -DETECTOR_SIZE_X/2 and DETECTOR_SIZE_X/2
            run["Reco"]["RandomAdjClX"][idx] = (
                np.random.rand(
                    len(run["Reco"]["Event"][idx]), len(run["Reco"]["AdjClCharge"][0])
                )
                * info["DETECTOR_SIZE_X"]
                - info["DETECTOR_SIZE_X"] / 2
            )

        if info["GEOMETRY"] == "vd":
            run["Reco"]["RandomX"][idx] = (
                np.random.rand(len(run["Reco"]["Event"][idx])) * info["DETECTOR_SIZE_X"]
            )

            run["Reco"]["RandomAdjClX"][idx] = (
                np.random.rand(
                    len(run["Reco"]["Event"][idx]), len(run["Reco"]["AdjClCharge"][0])
                )
                * info["DETECTOR_SIZE_X"]
            )

        run["Reco"]["EstimatedRecoX"][random_idx] = run["Reco"]["RandomX"][random_idx]
        run["Reco"]["EstimatedRecoAdjClX"][random_idx] = run["Reco"]["RandomAdjClX"][
            random_idx
        ]
        run["Reco"]["EstimatedRecoX"][default_idx] = run["Reco"]["MainVertex"][
            default_idx, 0
        ]
        run["Reco"]["EstimatedRecoAdjClX"][default_idx] = run["Reco"]["AdjClMainX"][
            default_idx
        ]

        if info["GEOMETRY"] == "hd":
            run["Reco"]["RandomDriftTime"][idx] = (
                run["Reco"]["RandomX"][idx]
                * 2
                * info["EVENT_TICKS"]
                / info["DETECTOR_SIZE_X"]
            )

            run["Reco"]["RandomAdjClDriftTime"][idx] = (
                run["Reco"]["RandomAdjClX"][idx]
                * 2
                * info["EVENT_TICKS"]
                / info["DETECTOR_SIZE_X"]
            )

        if info["GEOMETRY"] == "vd":
            run["Reco"]["RandomDriftTime"][idx] = (
                (info["DETECTOR_SIZE_X"] - run["Reco"]["RandomX"][idx])
                * 0.5
                * info["EVENT_TICKS"]
                / info["DETECTOR_SIZE_X"]
            )

            run["Reco"]["RandomAdjClDriftTime"][idx] = (
                (info["DETECTOR_SIZE_X"] - run["Reco"]["RandomAdjClX"][idx])
                * 0.5
                * info["EVENT_TICKS"]
                / info["DETECTOR_SIZE_X"]
            )

        if this_params["RANDOM_CORRECTION_RATIO"] > 0:
            output += f"[yellow]--> Applying random correction {100*this_params['RANDOM_CORRECTION_RATIO']:.1f}% to {branch}[/yellow]"

        for branch, (ref, jdx) in product(
            ["DriftTime", "AdjClDriftTime"],
            zip(["Random", "Truth"], [random_idx, default_idx]),
        ):
            run["Reco"][f"EstimatedReco{branch}"][jdx] = run["Reco"][f"{ref}{branch}"][
                jdx
            ]

    run = remove_branches(
        run, rm_branches, new_branches[:-1] + new_vector_branches[:-1], debug=debug
    )
    output += f"\tClutser time computation\t-> Done!\n"
    return run, output, new_branches + new_vector_branches


def compute_cluster_recox(
    run: dict,
    configs: dict[str, list[str]],
    params: Optional[dict] = None,
    rm_branches: bool = False,
    output: Optional[str] = None,
    debug: bool = False,
):
    """
    Compute the reconstructed X position of the events in the run.
    """
    new_branches = ["RecoX"]
    new_vector_branches = ["AdjCldT", "AdjClRecoX"]
    for branch in new_branches:
        run["Reco"][branch] = np.zeros(len(run["Reco"]["Event"]), dtype=np.float32)
    for branch in new_vector_branches:
        run["Reco"][branch] = np.zeros(
            (len(run["Reco"]["Event"]), len(run["Reco"]["AdjClTime"][0])),
            dtype=np.float32,
        )

    for config in configs:
        info, this_params, output = get_param_dict(
            f"{root}/config/{config}/{config}", params, output, debug=debug
        )
        idx = np.where(
            (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
            * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
        )

        converted_array = reshape_array(
            run["Reco"]["Time"][idx], len(run["Reco"]["AdjClTime"][idx][0])
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
                abs(run["Reco"][this_params["DEFAULT_RECOX_TIME"]][plus_idx])
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
                -abs(run["Reco"][this_params["DEFAULT_RECOX_TIME"]][mins_idx])
                * (info["DETECTOR_SIZE_X"] / 2)
                / info["EVENT_TICKS"]
            )

            converted_array = reshape_array(
                run["Reco"]["RecoX"], len(run["Reco"]["AdjClTime"][0])
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
                -abs(run["Reco"][this_params["DEFAULT_RECOX_TIME"]][idx])
                * info["DETECTOR_SIZE_X"]
                / info["EVENT_TICKS"]
                + info["DETECTOR_SIZE_X"] / 2
            )

            converted_array = reshape_array(
                run["Reco"]["RecoX"][idx], len(run["Reco"]["AdjClTime"][idx][0])
            )

            run["Reco"]["AdjClRecoX"][idx] = (
                run["Reco"]["AdjCldT"][idx]
                * info["DETECTOR_SIZE_X"]
                / info["EVENT_TICKS"]
            ) + converted_array

    run = remove_branches(run, rm_branches, ["AdjCldT"], debug=debug)
    output += f"\tComputed RecoX \t\t\t-> Done!\n"
    return run, output, new_branches + new_vector_branches


def compute_cluster_adjrecox(
    run: dict,
    configs: dict[str, list[str]],
    params: Optional[dict] = None,
    rm_branches: bool = False,
    output: Optional[str] = None,
    debug: bool = False,
):
    """
    Compute the reconstructed X position of the events in the run.
    """
    new_vector_branches = ["AdjCldT", "AdjClRecoX"]
    for branch in new_vector_branches:
        run["Reco"][branch] = np.zeros(
            (len(run["Reco"]["Event"]), len(run["Reco"]["AdjClTime"][0])),
            dtype=np.float32,
        )

    for config in configs:
        info, this_params, output = get_param_dict(
            f"{root}/config/{config}/{config}", params, output, debug=debug
        )
        idx = np.where(
            (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
            * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
        )

        converted_array = reshape_array(
            run["Reco"]["Time"][idx], len(run["Reco"]["AdjClTime"][idx][0])
        )

        run["Reco"]["AdjCldT"][idx] = run["Reco"]["AdjClTime"][idx] - converted_array

        if info["GEOMETRY"] == "hd":
            tpc_filter = (run["Reco"]["TPC"]) % 2 == 0
            plus_idx = np.where(
                (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
                * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
                * (tpc_filter)
            )

            tpc_filter = (run["Reco"]["TPC"]) % 2 == 1
            mins_idx = np.where(
                (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
                * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
                * (tpc_filter)
            )

            converted_array = reshape_array(
                run["Reco"]["RecoX"], len(run["Reco"]["AdjClTime"][0])
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
            converted_array = reshape_array(
                run["Reco"]["RecoX"][idx], len(run["Reco"]["AdjClTime"][idx][0])
            )

            run["Reco"]["AdjClRecoX"][idx] = (
                run["Reco"]["AdjCldT"][idx]
                * info["DETECTOR_SIZE_X"]
                / info["EVENT_TICKS"]
            ) + converted_array

    run = remove_branches(run, rm_branches, ["AdjCldT"], debug=debug)
    output += f"\tComputed AdjRecoX \t\t-> Done!\n"
    return run, output, new_vector_branches
