import os
import json
import pickle
import numpy as np

from typing import Optional
from itertools import product
from .formatting import remove_branches, get_param_dict, reshape_array
from .defaults import get_default_info, load_analysis_info
from .fitting import calibration_func
from .dataframe import npy2df

from src.utils import get_project_root

root = get_project_root()
_pnfs_calib = load_analysis_info(str(root)).get("DATA_PATHS", {}).get(
    "CALIBRATION", "/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/calibration"
)


def _calib_fallback(name: str, default: str) -> str:
    """Calibration sample used when `name` has no calibration of its own.

    Samples of the final-defaults production (gamma_final, neutron_final, radiological_final, ...)
    share the reconstruction configuration of marley_final, so their charge correction, cluster
    calibration, discriminant model and energy calibration must come from marley_final, not from
    the legacy marley_official default.
    """
    return "marley_final" if str(name).endswith("_final") and str(name) != "marley_final" else default




def _signal_background_flags(run, config, name, idx, this_params):
    """First SolarEnergy stage: True for clusters the signal/background classifier calls signal-like.

    Uses the two fold models of 03b_signal_background.py for the calibration sample of `name`
    (`name` itself or _calib_fallback). A cluster in fold f = (Flag + Event) % 2 is scored by the
    model trained on the other fold, so training clusters are never scored by their own model.
    Clusters below the training MIN_ENERGY are left signal-like. Returns None (no change) if the
    classifier is switched off (config.json SIGNAL_BACKGROUND_CLASSIFIER, params SB_CLASSIFIER) or absent.
    """
    if not load_analysis_info(str(root)).get("SIGNAL_BACKGROUND_CLASSIFIER", False):
        return None
    if this_params is not None and this_params.get("SB_CLASSIFIER", True) is False:
        return None
    for sample in dict.fromkeys([name, _calib_fallback(name, "")]):
        if not sample:
            continue
        jpath = f"{root}/config/{config}/{sample}/{config}_calib/{config}_signal_background.json"
        mpaths = [f"{_pnfs_calib}/{config}/{sample}/models/{config}_{sample}_signal_background_fold{k}.pkl" for k in (0, 1)]
        if os.path.isfile(jpath) and all(os.path.isfile(p) for p in mpaths):
            break
    else:
        rprint(f"[red][WARNING][/red] SIGNAL_BACKGROUND_CLASSIFIER is on but no classifier was found for {config} {name}: "
               "SolarEnergy keeps the offset for every cluster of this sample.")
        return None
    info = json.load(open(jpath))
    models = [pickle.load(open(p, "rb")) for p in mpaths]
    reco = run["Reco"]
    missing = [feat for feat in info["FEATURES"] if feat not in reco]
    if missing:
        raise KeyError(f"Signal/background classifier for {config} {name}: features {missing} not computed by this workflow")
    X = np.column_stack([np.asarray(reco[feat])[idx].astype(float) for feat in info["FEATURES"]])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    fold = ((np.asarray(reco["Flag"])[idx].astype(np.int64) + np.asarray(reco["Event"])[idx].astype(np.int64)) % 2).astype(int)
    score = np.ones(len(fold))
    for k in (0, 1):
        m = fold == 1 - k  # model k was trained on fold k
        if np.any(m):
            score[m] = models[k].predict_proba(X[m])[:, 1]
    energy = np.asarray(reco["Energy"])[idx].astype(float)
    score = np.where(energy >= info.get("MIN_ENERGY", 0.0), score, 1.0)
    if "SignalBackgroundScore" not in reco:
        reco["SignalBackgroundScore"] = np.ones(len(reco["Event"]))
    reco["SignalBackgroundScore"][idx] = score
    return score >= info["SB_THRESHOLD"]

def _adjcl_slots(run, idx):
    """Mask of the real adjacent clusters in the zero-padded AdjCl* arrays.

    The per-cluster AdjCl vectors are padded with 0 to the longest vector of the loaded sample
    (lib/io.resize_subarrays). A padded slot has AdjClR = 0 and would pass any AdjClR < limit cut,
    so every count over these arrays must be restricted to the first AdjClNum slots.
    """
    width = np.asarray(run["Reco"]["AdjClR"][idx]).shape[1]
    if "AdjClNum" in run["Reco"]:
        num = np.asarray(run["Reco"]["AdjClNum"][idx]).astype(np.int64)
        return np.arange(width)[None, :] < num[:, None]
    return np.asarray(run["Reco"]["AdjClR"][idx]) > 0

def _selected_limit_arrays(run, config, info, names, idx, this_params):
    """Per-row adjacent-cluster selection limits (MIN_BKG_R, MAX_BKG_CHARGE).

    Each sample uses config/{config}/{sample}/{config}_calib/{config}_selected_limits.json written by
    03_discrimination.py (falling back to the calibration sample of _calib_fallback, then to the
    per-config params.json), so the limits no longer depend on which sample was calibrated last.
    """
    rows = np.asarray(run["Reco"]["Name"])[idx]
    rlim = np.full(len(rows), float(this_params["MIN_BKG_R"]))
    qlim = np.full(len(rows), float(this_params["MAX_BKG_CHARGE"]))
    for name in names:
        for sample in dict.fromkeys([name, _calib_fallback(name, "")]):
            path = f"{root}/config/{config}/{sample}/{config}_calib/{config}_selected_limits.json"
            if sample and os.path.isfile(path):
                lim = json.load(open(path))
                rlim[rows == name] = float(lim["MIN_BKG_R"])
                qlim[rows == name] = float(lim["MAX_BKG_CHARGE"])
                break
    return rlim, qlim

def compute_electron_cluster(
    run: dict,
    configs: dict[str, list[str]],
    params: Optional[dict] = None,
    rm_branches: bool = False,
    output: str = "",
    debug: bool = False,
) -> tuple[dict, str, list[str]]:
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
    output += f"\tCluster charge computation\t-> Done!\n\n"
    return run, output, new_branches


def compute_cluster_energy(
    run: dict,
    configs: dict,
    params: Optional[dict] = None,
    clusters: list[str] = [""],
    rm_branches: bool = False,
    output: str = "",
    debug: bool = False,
) -> tuple[dict, str, list[str]]:
    """
    Correct the charge of the events in the run according to the correction file.
    """
    default_sample = "marley_official"
    # New branches
    new_branches = ["Correction", "CorrectionFactor"]
    for cluster in clusters:
        new_branches.append(f"Corrected{cluster}Charge")
        new_branches.append(f"{cluster}Energy")
        new_branches.append(f"Raw{cluster}Energy")

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

    output += f"\t[magenta][LOG][/magenta] Computing primary energy...\n"
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
                output += f"\t\t[yellow][WARNING][/yellow] Correction file not found for {name}. Defaulting to {_calib_fallback(name, default_sample)}!\n"
                corr_info = json.load(
                    open(
                        f"{root}/config/{config}/{_calib_fallback(name, default_sample)}/{config}_calib/{config}_electroncharge_correction.json",
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
            output += f'\t\t[cyan][INFO][/cyan] Using correction variables: {this_params["DEFAULT_ENERGY_TIME"]}, {this_params["DEFAULT_ADJCL_ENERGY_TIME"]}\n'

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

            if "CORRECTION_TABLE_NHITS" in corr_info:
                # Measured F0 per NHits (01_correction --f0_from table): linear between measured values,
                # nearest measured value outside the measured range (np.interp clamps at both ends).
                run["Reco"]["CorrectionFactor"][idx] = np.interp(
                    np.asarray(run["Reco"]["NHits"][idx], dtype=float),
                    np.asarray(corr_info["CORRECTION_TABLE_NHITS"], dtype=float),
                    np.asarray(corr_info["CORRECTION_TABLE_FACTOR"], dtype=float),
                )
            else:
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

                run["Reco"][f"Raw{cluster}Energy"][idx] = (
                    run["Reco"][f"{cluster}Charge"][idx] / corr_info["CHARGE_AMP"]
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
    output += f"\tPrimary energy computation\t-> Done!\n\n"
    return run, output, new_branches + new_vector_branches


def compute_cluster_calibration(
    run,
    configs: dict[str, list[str]],
    params: Optional[dict] = None,
    clusters: list[str] = ["", "Electron"],
    rm_branches: bool = False,
    output: str = "",
    debug=False,
) -> tuple[dict, str, list[str]]:
    """
    Correct the charge of the events in the run according to the correction file.
    """

    default_sample = "marley_official"
    new_branches = ["EnergySlope", "EnergyIntercept"]
    for cluster, branch in product(clusters, new_branches):
        run["Reco"][f"{cluster}{branch}"] = np.zeros(len(run["Reco"]["Event"]))

    if output is None:
        output = ""
    output += f"\t[magenta][LOG][/magenta] Computing cluster energy calibration...\n"

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
                            f"{_pnfs_calib}/{config}/{name}/calib/{config}_{cluster.lower()}charge_slope_calibration.pkl",
                            "rb",
                        )
                    )
                    corr_intercept = pickle.load(
                        open(
                            f"{_pnfs_calib}/{config}/{name}/calib/{config}_{cluster.lower()}charge_intercept_calibration.pkl",
                            "rb",
                        )
                    )

                except FileNotFoundError:
                    output += f"\t\t[yellow][WARNING][/yellow] Calibration file not found. Defaulting to {_calib_fallback(name, default_sample)}!\n"

                    corr_slope = pickle.load(
                        open(
                            f"{_pnfs_calib}/{config}/{_calib_fallback(name, default_sample)}/calib/{config}_{cluster.lower()}charge_slope_calibration.pkl",
                            "rb",
                        )
                    )
                    corr_intercept = pickle.load(
                        open(
                            f"{_pnfs_calib}/{config}/{_calib_fallback(name, default_sample)}/calib/{config}_{cluster.lower()}charge_intercept_calibration.pkl",
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
    output += f"\tCluster energy calibration\t-> Done!\n\n"
    return run, output, new_branches


def compute_total_energy(
    run: dict,
    configs: dict[str, list[str]],
    params: Optional[dict] = None,
    rm_branches: bool = False,
    output: str = "",
    debug=False,
) -> tuple[dict, str, list[str]]:
    """
    Compute the total energy of the events in the run.
    """

    if output is None:
        output = ""
    output += f"\t[magenta][LOG][/magenta] Computing event energies...\n"

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
        _rlim, _qlim = _selected_limit_arrays(run, config, info, configs[config], idx, this_params)

        run["Reco"]["TotalAdjClEnergy"][idx] = np.sum(
            run["Reco"]["AdjClEnergy"][idx], axis=1
        )

        selected_filter = (
            (run["Reco"]["AdjClR"][idx] < _rlim[:, None])
            + (run["Reco"]["AdjClCharge"][idx] > _qlim[:, None])
        ) & _adjcl_slots(run, idx)

        if debug:
            output += f"\t\t[cyan][INFO][/cyan] Selected filter for energy computation excludes {100*((np.sum(run['Reco']['AdjClNum'][idx])-np.sum(~selected_filter))/np.sum(run['Reco']['AdjClNum'][idx])):.1f}% of Adj. clusters\n"

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
    output += f"\tClusterEvent energy computation -> Done!\n\n"
    return run, output, new_branches


def recompute_reco_energy_with_charge_threshold(
    run: dict,
    configs: dict,
    charge_threshold: float,
    params: Optional[dict] = None,
    output: str = "",
    debug: bool = False,
) -> tuple[dict, str]:
    """
    Recompute all AdjCl-based energy features with a minimum charge floor.

    Applies AdjClCharge > charge_threshold to every AdjCl-based sum — both the
    Selected* features (which already carry per-config R and MAX_BKG_CHARGE cuts)
    and the Total* / Max features (which normally sum unconditionally).  No BDT is
    involved; the charge study uses SelectedEnergy as its analysis metric, which is a
    direct sum (Energy + SelectedAdjClEnergy) and requires no retraining.

    Called from 03_analysis.py after compute_reco_workflow when --charge_threshold > 0.
    The charge variants in run_studies.py pass energy_override="SelectedEnergy" so
    that the Rebin pkls written afterwards use the charge-corrected SelectedEnergy axis.
    """
    if output is None:
        output = ""
    output += f"\t[magenta][LOG][/magenta] Recomputing AdjCl energy features with Q>{charge_threshold:.0f} ADC...\n"

    for config in configs:
        info, this_params, output = get_param_dict(
            f"{root}/config/{config}/{config}", params, output, debug=debug
        )
        idx = np.where(
            (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
            * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
        )
        _rlim, _qlim = _selected_limit_arrays(run, config, info, configs[config], idx, this_params)

        adj_charge = run["Reco"]["AdjClCharge"][idx]
        adj_energy = run["Reco"]["AdjClEnergy"][idx]
        charge_floor = adj_charge > charge_threshold

        # Total* features: sum / max over all AdjCl above the charge floor.
        run["Reco"]["TotalAdjClEnergy"][idx] = np.sum(adj_energy, where=charge_floor, axis=1)
        run["Reco"]["MaxAdjClEnergy"][idx]   = np.max(adj_energy, where=charge_floor, axis=1, initial=0)
        run["Reco"]["TotalEnergy"][idx]      = run["Reco"]["Energy"][idx] + run["Reco"]["TotalAdjClEnergy"][idx]

        # Selected* features: existing R + MAX_BKG_CHARGE filter, then charge floor.
        selected_filter = (
            (run["Reco"]["AdjClR"][idx] < _rlim[:, None]) + (adj_charge > _qlim[:, None])
        ) & _adjcl_slots(run, idx)
        selected_filter_Q = selected_filter & charge_floor

        run["Reco"]["SelectedAdjClNum"][idx]         = np.sum(selected_filter_Q, axis=1)
        run["Reco"]["SelectedAdjClEnergy"][idx]      = np.sum(adj_energy, where=selected_filter_Q, axis=1)
        run["Reco"]["SelectedMaxAdjClEnergy"][idx]   = np.max(adj_energy, where=selected_filter_Q, axis=1, initial=0)
        run["Reco"]["SelectedEnergy"][idx]           = (
            run["Reco"]["Energy"][idx] + run["Reco"]["SelectedAdjClEnergy"][idx]
        )
        run["Reco"]["SelectedAdjClEnergyRatio"][idx] = np.where(
            run["Reco"]["Energy"][idx] != 0,
            run["Reco"]["SelectedAdjClEnergy"][idx] / run["Reco"]["Energy"][idx],
            0.0,
        )

    output += f"\tAdjCl energy features recomputed with Q>{charge_threshold:.0f} ADC\t-> Done!\n\n"
    return run, output


def compute_reco_energy(
    run,
    configs,
    params: Optional[dict] = None,
    rm_branches: bool = False,
    output: str = "",
    debug=False,
) -> tuple[dict, str, list[str]]:
    if output is None:
        output = ""

    output += f"\t[magenta][LOG][/magenta] Computing SolarEnergy...\n"

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
            default_sample = "marley_official"
            try:
                path = f"{_pnfs_calib}/{config}/{name}/models/{config}_{name}_random_forest_discriminant.pkl"
                with open(path, "rb") as model_file:
                    rf_classifier = pickle.load(model_file)

            except FileNotFoundError:
                output += f"\t\t[yellow][WARNING][/yellow] ML model file not found. Defaulting to {_calib_fallback(name, default_sample)}!\n"
                path = f"{_pnfs_calib}/{config}/{_calib_fallback(name, default_sample)}/models/{config}_{_calib_fallback(name, default_sample)}_random_forest_discriminant.pkl"

                with open(path, "rb") as model_file:
                    rf_classifier = pickle.load(model_file)

            try:
                discriminant_info = json.load(
                    open(
                        f"{root}/config/{config}/{name}/{config}_calib/{config}_discriminant_calibration.json",
                        "r",
                    )
                )
                output += f"\t\t[cyan][INFO][/cyan] Loading model for {name}\n"

            except FileNotFoundError:
                discriminant_info = json.load(
                    open(
                        f"{root}/config/{config}/{_calib_fallback(name, default_sample)}/{config}_calib/{config}_discriminant_calibration.json",
                        "r",
                    )
                )
                output += f"\t\t[yellow][WARNING][/yellow] Loading model for {_calib_fallback(name, default_sample)}\n"

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

            model_features = list(getattr(rf_classifier, "feature_names_in_", features))
            missing = [feat for feat in model_features if feat not in run["Reco"]]
            if missing:
                raise KeyError(f"Offset classifier for {config} {name}: features {missing} not loaded by this workflow")
            df = npy2df(
                run,
                "Reco",
                branches=model_features
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
                df[model_features] = df[model_features].replace([np.inf, -np.inf], np.nan).fillna(0)
                df[model_features] = np.clip(
                    df[model_features], a_min=-1e6, a_max=1e6
                )  # Adjust the clipping values as necessary
                df["ML"] = rf_classifier.predict(df[model_features])
                df["Discriminant"] = rf_classifier.predict_proba(df[model_features])[:, 1]
            except ValueError:
                print(df[model_features])

            upper_idx = df["Discriminant"] >= discriminant_info["ML_THRESHOLD"]
            lower_idx = df["Discriminant"] < discriminant_info["ML_THRESHOLD"]

            df.loc[upper_idx, "SolarEnergy"] = upper_func(df.loc[upper_idx, "Energy"])
            df.loc[lower_idx, "SolarEnergy"] = lower_func(df.loc[lower_idx, "Energy"])
            # The signal/background first stage is applied in compute_energy_calibration, once
            # TotalEnergy/SelectedEnergy carry their final calibration (the values it was trained on).
            run["Reco"]["SolarEnergy"][idx] = df["SolarEnergy"].values

    run = remove_branches(run, rm_branches, [], debug=debug)
    output += f"\tSolarEvent energy computation \t-> Done!\n\n"
    return run, output, new_branches


def compute_energy_calibration(
    run,
    configs,
    params: Optional[dict] = None,
    rm_branches: bool = False,
    output: str = "",
    debug=False,
) -> tuple[dict, str, list[str]]:
    # Inverse function of polinomyal with degree 2
    def inverse_quadratic(x, a, b, c):
        return (-b + np.sqrt(b**2 - 4 * a * (c - x))) / (2 * a)

    output += f"\t[magenta][LOG][/magenta] Computing particle energy calibration...\n"
    default_sample = "marley_official"
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
                    output += f"\t\t[cyan][INFO][/cyan] Applying energy calibration from {name}\n"

            except FileNotFoundError:
                reco_info = json.load(
                    open(
                        f"{root}/config/{config}/{_calib_fallback(name, default_sample)}/{config}_calib/{config}_{_calib_fallback(name, default_sample)}_energy_calibration.json",
                        "r",
                    )
                )
                output += f"\t\t[yellow][WARNING][/yellow] Applying default energy calibration from {_calib_fallback(name, default_sample)}\n"

            for energy, ref_energy in zip(
                ["Solar", "Selected", "Total", "Cluster"],
                ["Solar", "Selected", "Total", ""],
            ):
                if this_params["SAMPLE_FIT"][f"{energy}Energy"] == "linear":
                    run["Reco"][f"{energy}Energy"][idx] = (
                        run["Reco"][f"{ref_energy}Energy"][idx]
                        - reco_info[energy.upper()]["INTERSECTION"]
                    ) / reco_info[energy.upper()]["ENERGY_AMP"]
                    output += f"\t\t[cyan][INFO][/cyan] Applying linear energy calibration for {energy}Energy\n"

                elif this_params["SAMPLE_FIT"][f"{energy}Energy"] == "quadratic":
                    run["Reco"][f"{energy}Energy"][idx] = inverse_quadratic(
                        run["Reco"][f"{ref_energy}Energy"][idx],
                        reco_info[energy.upper()]["ENERGY_CURVATURE"],
                        reco_info[energy.upper()]["ENERGY_AMP"],
                        reco_info[energy.upper()]["INTERSECTION"],
                    )
                    output += f"\t\t[cyan][INFO][/cyan] Applying quadratic energy calibration for {energy}Energy\n"

            # First SolarEnergy stage: clusters the signal/background classifier calls background-like
            # get the SolarEnergy calibration applied to their cluster energy, i.e. no de-excitation offset.
            # Applied here because the classifier was trained on these end-of-workflow features.
            signal_like = _signal_background_flags(run, config, name, idx, this_params)
            if signal_like is not None:
                rows = np.asarray(idx[0])
                if this_params["SAMPLE_FIT"]["SolarEnergy"] == "linear":
                    no_offset = (np.asarray(run["Reco"]["Energy"])[rows] - reco_info["SOLAR"]["INTERSECTION"]) / reco_info["SOLAR"]["ENERGY_AMP"]
                else:
                    no_offset = inverse_quadratic(
                        np.asarray(run["Reco"]["Energy"])[rows],
                        reco_info["SOLAR"]["ENERGY_CURVATURE"],
                        reco_info["SOLAR"]["ENERGY_AMP"],
                        reco_info["SOLAR"]["INTERSECTION"],
                    )
                # Both SolarEnergy versions are kept so that the threshold scan
                # (03d_signal_background_threshold.py) can rebuild SolarEnergy for any threshold.
                for branch in ["SolarEnergyOffset", "SolarEnergyNoOffset"]:
                    if branch not in run["Reco"]:
                        run["Reco"][branch] = np.zeros(len(run["Reco"]["Event"]))
                run["Reco"]["SolarEnergyOffset"][rows] = run["Reco"]["SolarEnergy"][rows]
                run["Reco"]["SolarEnergyNoOffset"][rows] = no_offset
                run["Reco"]["SolarEnergy"][rows[~signal_like]] = no_offset[~signal_like]
                output += f"\t\t[cyan][INFO][/cyan] {name}: {np.mean(~signal_like):.1%} of clusters background-like (no offset)\n"

    run = remove_branches(run, rm_branches, [], debug=debug)
    output += f"\tEvent energy calibration \t-> Done!\n\n"
    return run, output, new_branches


def compute_cluster_time(
    run: dict,
    configs: dict[str, list[str]],
    params: Optional[dict] = None,
    rm_branches: bool = False,
    output: str = "",
    debug: bool = False,
) -> tuple[dict, str, list[str]]:
    """
    Correct the charge of the events in the run according to the correction file.
    """
    output += f"\t[magenta][LOG][/magenta] Computing cluster times...\n"

    # New branches
    new_branches = ["RecoDriftTime", "AverageDriftTime"]
    new_vector_branches = ["AdjCldTime", "AdjClRecoDriftTime", "AdjClAverageDriftTime"]

    for branch in new_branches:
        # Check if the branch exists
        if branch in run["Reco"]:
            output += f"\t\t[yellow][WARNING][/yellow] Branch {branch} already exists! Overwriting...\n"

        if branch == "AverageDriftTime":
            run["Reco"][branch] = np.ones(len(run["Reco"]["Event"]), dtype=np.float32)
        else:
            run["Reco"][branch] = np.zeros(len(run["Reco"]["Event"]), dtype=np.float32)

    for branch in new_vector_branches:
        # Check if the branch exists
        if branch in run["Reco"]:
            output += f"\t\t[yellow][WARNING][/yellow] Branch {branch} already exists! Overwriting...\n"

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
            output += f"\t\t[yellow][WARNING][/yellow] No events found with flash-match information! Skipping drift-time computation.\n"
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

    run = remove_branches(run, rm_branches, [], debug=debug)
    output += f"\tCluster time computation\t-> Done!\n\n"
    return run, output, new_branches + new_vector_branches


def compute_cluster_estimated_time(
    run: dict,
    configs: dict[str, list[str]],
    params: Optional[dict] = None,
    rm_branches: bool = False,
    output: str = "",
    debug: bool = False,
) -> tuple[dict, str, list[str]]:
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
            output += f"\t\t[yellow][WARNING][/yellow] Branch {branch} already exists! Overwriting...\n"

        run["Reco"][branch] = np.zeros(len(run["Reco"]["Event"]), dtype=np.float32)

    for branch in new_vector_branches:
        # Check if the branch exists
        if branch in run["Reco"]:
            output += f"\t\t[yellow][WARNING][/yellow] Branch {branch} already exists! Overwriting...\n"

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
    output += f"\tClutser time computation\t-> Done!\n\n"
    return run, output, new_branches + new_vector_branches


# def compute_cluster_recox(
#     run: dict,
#     configs: dict[str, list[str]],
#     params: Optional[dict] = None,
#     rm_branches: bool = False,
#     output: str = "",
#     debug: bool = False,
# ):
#     """
#     Compute the reconstructed X position of the events in the run.
#     """
#     new_branches = ["RecoX"]
#     new_vector_branches = ["AdjCldT", "AdjClRecoX"]
#     for branch in new_branches:
#         run["Reco"][branch] = np.zeros(len(run["Reco"]["Event"]), dtype=np.float32)

#     for branch in new_vector_branches:
#         run["Reco"][branch] = np.zeros(
#             (len(run["Reco"]["Event"]), len(run["Reco"]["AdjClTime"][0])),
#             dtype=np.float32,
#         )

#     for config in configs:
#         info, this_params, output = get_param_dict(
#             f"{root}/config/{config}/{config}", params, output, debug=debug
#         )
#         idx = np.where(
#             (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
#             * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
#         )

#         converted_array = reshape_array(
#             run["Reco"]["Time"][idx], len(run["Reco"]["AdjClTime"][idx][0])
#         )

#         run["Reco"]["AdjCldT"][idx] = run["Reco"]["AdjClTime"][idx] - converted_array

#         if info["GEOMETRY"] == "hd":
#             tpc_filter = (run["Reco"]["TPC"]) % 2 == 0
#             plus_idx = np.where(
#                 (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
#                 * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
#                 * (tpc_filter)
#             )
#             run["Reco"]["RecoX"][plus_idx] = (
#                 abs(run["Reco"][this_params["DEFAULT_RECOX_TIME"]][plus_idx])
#                 * (info["DETECTOR_SIZE_X"] / 2)
#                 / info["EVENT_TICKS"]
#             )

#             tpc_filter = (run["Reco"]["TPC"]) % 2 == 1
#             mins_idx = np.where(
#                 (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
#                 * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
#                 * (tpc_filter)
#             )
#             run["Reco"]["RecoX"][mins_idx] = (
#                 -abs(run["Reco"][this_params["DEFAULT_RECOX_TIME"]][mins_idx])
#                 * (info["DETECTOR_SIZE_X"] / 2)
#                 / info["EVENT_TICKS"]
#             )

#             converted_array = reshape_array(
#                 run["Reco"]["RecoX"], len(run["Reco"]["AdjClTime"][0])
#             )

#             run["Reco"]["AdjClRecoX"][plus_idx] = (
#                 run["Reco"]["AdjCldT"][plus_idx]
#                 * (info["DETECTOR_SIZE_X"] / 2)
#                 / info["EVENT_TICKS"]
#                 + converted_array[plus_idx]
#             )
#             run["Reco"]["AdjClRecoX"][mins_idx] = (
#                 -run["Reco"]["AdjCldT"][mins_idx]
#                 * (info["DETECTOR_SIZE_X"] / 2)
#                 / info["EVENT_TICKS"]
#                 + converted_array[mins_idx]
#             )

#         if info["GEOMETRY"] == "vd":
#             run["Reco"]["RecoX"][idx] = (
#                 -abs(run["Reco"][this_params["DEFAULT_RECOX_TIME"]][idx])
#                 * info["DETECTOR_SIZE_X"]
#                 / info["EVENT_TICKS"]
#                 + info["DETECTOR_SIZE_X"] / 2
#             )

#             converted_array = reshape_array(
#                 run["Reco"]["RecoX"][idx], len(run["Reco"]["AdjClTime"][idx][0])
#             )

#             run["Reco"]["AdjClRecoX"][idx] = (
#                 run["Reco"]["AdjCldT"][idx]
#                 * info["DETECTOR_SIZE_X"]
#                 / info["EVENT_TICKS"]
#             ) + converted_array

#     run = remove_branches(run, rm_branches, ["AdjCldT"], debug=debug)
#     output += f"\tComputed RecoX \t\t\t-> Done!\n\n"
#     return run, output, new_branches + new_vector_branches


# def compute_cluster_adjrecox(
#     run: dict,
#     configs: dict[str, list[str]],
#     params: Optional[dict] = None,
#     rm_branches: bool = False,
#     output: str = "",
#     debug: bool = False,
# ):
#     """
#     Compute the reconstructed X position of the events in the run.
#     """
#     new_vector_branches = ["AdjCldT", "AdjClRecoX"]
#     for branch in new_vector_branches:
#         run["Reco"][branch] = np.zeros(
#             (len(run["Reco"]["Event"]), len(run["Reco"]["AdjClTime"][0])),
#             dtype=np.float32,
#         )

#     for config in configs:
#         info, this_params, output = get_param_dict(
#             f"{root}/config/{config}/{config}", params, output, debug=debug
#         )
#         idx = np.where(
#             (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
#             * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
#         )

#         converted_array = reshape_array(
#             run["Reco"]["Time"][idx], len(run["Reco"]["AdjClTime"][idx][0])
#         )

#         run["Reco"]["AdjCldT"][idx] = run["Reco"]["AdjClTime"][idx] - converted_array

#         if info["GEOMETRY"] == "hd":
#             tpc_filter = (run["Reco"]["TPC"]) % 2 == 0
#             plus_idx = np.where(
#                 (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
#                 * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
#                 * (tpc_filter)
#             )

#             tpc_filter = (run["Reco"]["TPC"]) % 2 == 1
#             mins_idx = np.where(
#                 (np.asarray(run["Reco"]["Geometry"]) == info["GEOMETRY"])
#                 * (np.asarray(run["Reco"]["Version"]) == info["VERSION"])
#                 * (tpc_filter)
#             )

#             converted_array = reshape_array(
#                 run["Reco"]["RecoX"], len(run["Reco"]["AdjClTime"][0])
#             )

#             run["Reco"]["AdjClRecoX"][plus_idx] = (
#                 run["Reco"]["AdjCldT"][plus_idx]
#                 * (info["DETECTOR_SIZE_X"] / 2)
#                 / info["EVENT_TICKS"]
#                 + converted_array[plus_idx]
#             )
#             run["Reco"]["AdjClRecoX"][mins_idx] = (
#                 -run["Reco"]["AdjCldT"][mins_idx]
#                 * (info["DETECTOR_SIZE_X"] / 2)
#                 / info["EVENT_TICKS"]
#                 + converted_array[mins_idx]
#             )

#         if info["GEOMETRY"] == "vd":
#             converted_array = reshape_array(
#                 run["Reco"]["RecoX"][idx], len(run["Reco"]["AdjClTime"][idx][0])
#             )

#             run["Reco"]["AdjClRecoX"][idx] = (
#                 run["Reco"]["AdjCldT"][idx]
#                 * info["DETECTOR_SIZE_X"]
#                 / info["EVENT_TICKS"]
#             ) + converted_array

#     run = remove_branches(run, rm_branches, ["AdjCldT"], debug=debug)
#     output += f"\tComputed AdjRecoX \t\t-> Done!\n\n"
#     return run, output, new_vector_branches
