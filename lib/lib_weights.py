import json
import pickle
import numpy as np

from typing import Optional
from itertools import product
from rich import print as rprint
from lib.lib_format import remove_branches, get_param_dict

from src.utils import get_project_root

root = get_project_root()
osc_names = {"mean": "Mean", "day": "Day", "night": "Night"}


def evaluate_1d_pdf(kde, values):
    """Evaluate 1D PDF from a KDE."""
    log_density = kde.score_samples(values[:, None])  # [:, None] ensures 2D input
    return np.exp(log_density)


# Define the joint PDF using dimensional independence
def evaluate_joint_pdf(p, x, y, z, kde_p, kde_x, kde_y, kde_z):
    """Evaluate the joint PDF under the assumption of dimensional independence."""
    pdf_p = evaluate_1d_pdf(kde_p, p)
    pdf_x = evaluate_1d_pdf(kde_x, x)
    pdf_y = evaluate_1d_pdf(kde_y, y)
    pdf_z = evaluate_1d_pdf(kde_z, z)

    # rprint(f"[cyan][INFO] Eveluated PDF for {name} with {len(p)} events[/cyan]")
    return pdf_p * pdf_x * pdf_y * pdf_z


def evaluate_surface_pdf(info, name, A, alpha, p, x, y, z, s, kde_p, kde_uv):
    """Evaluate the surface PDF under the assumption of dimensional independence."""
    # surface_weights = json.load(open(f"{root}/import/surface_weights.json"))
    # surface_production = json.load(open(f"{root}/import/surface_production.json"))
    w = np.empty_like(p)
    N = 0
    # Combine u and v into a 2D array for kde_uv
    for surface in range(0, N):
        mask = s == surface
        if sum(mask) == 0:
            continue

        N += 1
        if surface in [0]:
            u, v = y, z
        elif surface in [1, 2]:
            u, v = x, z
        elif surface in [3, 4]:
            u, v = x, y
        else:
            raise ValueError(f"Invalid surface_id: {surface}")

        log_p = kde_p[surface].score_samples(p[mask, None])
        log_uv = kde_uv[surface].score_samples(np.column_stack([u[mask], v[mask]]))
        w[mask] = (
            alpha[surface]
            * A[str(surface)]
            * np.exp(log_p)
            * np.exp(log_uv)
            # * surface_production[info["VERSION"]][str(surface)]
            # / surface_weights[info["VERSION"]][name][str(surface)]
        )

    # Take the 1% highest weights and set the to the median of the rest to avoid extreme outliers
    threshold = np.percentile(w, 99)
    median = np.median(w[w < threshold])
    w[w > threshold] = median

    # If there are nan values in w, set them to median
    if np.isnan(w).any():
        w = np.where(np.isnan(w), median, w)

    # Return the surface production weights
    return N * np.absolute(w)


def evaluate_background_pdf(info, name, A, alpha, p, s, hists, bins):
    """Evaluate the background PDF using histogram PDFs for each surface."""
    # surface_weights = json.load(open(f"{root}/import/surface_weights.json"))
    # surface_production = json.load(open(f"{root}/import/surface_production.json"))
    N = 0
    w = np.zeros_like(p)
    for surface_id in np.unique(s):
        mask = s == surface_id
        if surface_id < 0 or len(hists[surface_id]) == 0:
            continue

        N += 1
        p_s = p[mask]
        bin_idx_s = np.digitize(p_s, bins[surface_id]) - 1
        bin_idx_s = np.clip(bin_idx_s, 0, len(hists[surface_id]) - 1)
        w[mask] = (
            A[str(surface_id)]
            * alpha[surface_id]
            * hists[surface_id][bin_idx_s]
            # * surface_production[info["VERSION"]][str(surface_id)]
            # / surface_weights[info["VERSION"]][name][str(surface_id)]
        )

    return N * w


def compute_particle_weights(
    run: dict[str, dict],
    configs: dict[str, list[str]],
    params: dict = {},
    trees: list[str] = ["Truth", "Reco"],
    rm_branches: bool = False,
    output: str = "",
    debug: bool = False,
):
    """
    Compute the single weights for the events in the run.
    """
    if output is None:
        output = ""

    output += f"\t[magenta][LOG][/magenta] Computing particle weights...\n"
    new_weights = [""]
    new_branches = ["SignalParticleWeight"]
    for tree, branch in product(trees, new_branches):
        run[tree][branch] = np.zeros(len(run[tree]["Event"]), dtype=np.float64)

    output += f"\t\t[cyan][INFO][/cyan] Computing truth particle weights...\n"
    run, output, exposure, new_branches_true = compute_true_weights(
        run, configs, params, trees, None, output, debug=debug
    )
    new_branches += new_branches_true
    output += f"\t\t[cyan][INFO][/cyan] Renormalizing truth particle weights...\n"
    run, output, new_weights = normalize_true_weights(
        run, configs, params, trees, output, exposure, debug=debug
    )
    # output += f"\t\t[cyan][INFO][/cyan] Computing reco particle weights...\n"
    # run, output = compute_reco_weights(run, configs, params, new_weights, output, debug)

    run = remove_branches(run, rm_branches, [], debug=debug)
    output += f"\tParticle weights computation \t-> Done!\n\n"
    return run, output, new_branches


def compute_particle_surface(
    run: dict[str, dict],
    configs: dict[str, list[str]],
    params: Optional[dict] = None,
    trees: list[str] = ["Reco"],
    variable: Optional[str] = None,
    rm_branches: bool = False,
    output: str = "",
    debug: bool = False,
):
    """
    Evaluate the particle's production surface and generate "Truth" and "Reco" branches in the run.
    """
    if output is None:
        output = ""

    output += f"\t[magenta][LOG][/magenta] Computing particle surfaces...\n"

    surfaces = json.load(open(f"{root}/import/surface_positions.json"))

    new_branches = ["SignalParticleSurface"]
    for tree in trees:
        run[tree]["SignalParticleSurface"] = -1 * np.ones(
            len(run[tree]["Event"]), dtype=np.int8
        )

    for config in configs:
        info, params, output = get_param_dict(
            f"{root}/config/{config}/{config}", params, output, debug=debug
        )

        for name, tree in product(configs[config], trees):
            if variable is None:
                if (
                    "gamma" in name.lower()
                    or "marley" in name.lower()
                    or "alpha" in name.lower()
                    or "electron" in name.lower()
                    or "neutron" in name.lower()
                ):
                    variable = "SignalParticle"
                else:
                    rprint(
                        f"[red][ERROR][/red] Particle name {name} not recognized for surface computation..."
                    )
                    continue

            idx = np.where(
                (np.asarray(run[tree]["Geometry"]) == info["GEOMETRY"])
                * (np.asarray(run[tree]["Version"]) == info["VERSION"])
                * (np.asarray(run[tree]["Name"]) == name)
            )
            # If marley in name, production is flat so surface is -1 (unknown)
            if "marley" in name.lower():
                run[tree]["SignalParticleSurface"][idx] = -1.0

            elif "gamma" in name.lower() or "neutron" in name.lower():
                # surface_weights = {}
                main_coords = {
                    "X": run[tree][f"{variable}X"][idx],
                    "Y": run[tree][f"{variable}Y"][idx],
                    "Z": run[tree][f"{variable}Z"][idx],
                }

                output += (
                    f"\t\t[cyan][INFO][/cyan] Computed {tree} particles per surface: "
                )
                for surface_name, [surface_value, surface_id] in surfaces[
                    info["GEOMETRY"]
                ].items():
                    coord = (
                        main_coords["X"]
                        if surface_name.lower() in ["anode", "cathode", "apa", "cpa"]
                        else (
                            main_coords["Y"]
                            if "membrane" in surface_name.lower()
                            or surface_name.lower() in ["top", "bottom"]
                            else main_coords["Z"]
                        )
                    )
                    mask = np.abs(coord - surface_value) <= 1
                    output += (
                        f"{surface_name}: {100 * mask.sum() / len(idx[0]):.1f}% / "
                    )
                    # surface_weights[str(surface_id)] = mask.sum() / len(idx[0])
                    run[tree]["SignalParticleSurface"][idx[0][mask]] = surface_id
                output = output[:-3] + "\n"
                # Print the number of particles that are not within 1 cm of any surface if more than 1%
                if (
                    (run[tree]["SignalParticleSurface"][idx] == -1).sum()
                    / len(run[tree]["SignalParticleSurface"][idx])
                ) > 0.01:
                    output += f"[yellow][WARNING][/yellow] {100 * (run[tree]['SignalParticleSurface'][idx] == -1).sum() / len(run[tree]['SignalParticleSurface'][idx]):.2f}% of particles not within 1 cm of any surface for {name} in {config}...\n"

                # Update the surface weights in the import json file
                # surface_weights_file = f"{root}/import/surface_weights.json"
                # surface_weights_data = json.load(open(surface_weights_file))
                # if info["VERSION"] in surface_weights_data:
                # surface_weights_data[info["VERSION"]][name] = surface_weights
                #     with open(surface_weights_file, "w") as f:
                #         json.dump(surface_weights_data, f, indent=4)
                # else:
                #     rprint(
                #         f"[red][ERROR][/red] Version {info['VERSION']} not found in surface weights json file for updating surface weights for {name} in {config}..."
                #     )

            else:
                rprint(
                    f"[red][ERROR][/red] Particle name {name} not recognized for surface computation[/red]"
                )

    output += f"\tParticle surface computation \t-> Done!\n\n"
    run = remove_branches(run, rm_branches, [], debug=debug)
    return run, output, new_branches


def compute_true_weights(
    run: dict[str, dict],
    configs: dict[str, list[str]],
    params: dict = {},
    trees: list[str] = ["Truth", "Reco"],
    new_branches: Optional[list[str]] = None,
    output: str = "",
    debug: bool = False,
):

    # Load json with number of trees per configuration
    exposure = {}
    surfaces = json.load(open(f"{root}/import/surface_positions.json"))
    if new_branches is None:
        new_branches = []

    for config in configs:
        info, params, output = get_param_dict(
            f"{root}/config/{config}/{config}", params, output, debug=debug
        )

        for name, tree in product(configs[config], trees):
            exposure = {}
            idx = np.where(
                (np.asarray(run[tree]["Geometry"]) == info["GEOMETRY"])
                * (np.asarray(run[tree]["Version"]) == info["VERSION"])
                * (np.asarray(run[tree]["Name"]) == name)
            )

            if "marley" in name.lower():
                new_weights = ["", "b8", "hep"]
                for new_weight, osc in product(
                    new_weights, params["DEFAULT_SIGNAL_WEIGHT"]
                ):
                    run[tree][f"SignalParticleWeight{new_weight}"] = run[tree][
                        "SignalParticleWeight"
                    ].copy()

                    if new_weight != "":
                        new_branches.append(f"SignalParticleWeight{new_weight}")

                    if osc == "osc":
                        for osc_name in params["DEFAULT_SIGNAL_AZIMUTH"]:
                            run[tree][
                                f"SignalParticleWeight{new_weight}Osc{osc_names[osc_name]}"
                            ] = run[tree]["SignalParticleWeight"].copy()
                            new_branches.append(
                                f"SignalParticleWeight{new_weight}Osc{osc_names[osc_name]}"
                            )

            if "PARTICLE_TYPE" not in params or params["PARTICLE_TYPE"] is None:
                output += f"[red][ERROR][/red] PARTICLE_TYPE not specified in config for weight computation..."
                continue

            elif params["PARTICLE_TYPE"] == "signal":
                for (comp, label), osc in product(
                    zip(["comb", "b8", "hep"], ["", "b8", "hep"]),
                    params["DEFAULT_SIGNAL_WEIGHT"],
                ):
                    exposure[(label, osc)] = dict()
                    if osc == "osc":
                        dm2 = f"{params['DEFAULT_SIGNAL_DM2']:.3e}"
                        sin13 = f"{params['DEFAULT_SIGNAL_SIN13']:.3e}"
                        sin12 = f"{params['DEFAULT_SIGNAL_SIN12']:.3e}"
                        for azimuth in params["DEFAULT_SIGNAL_AZIMUTH"]:
                            exposure[(label, osc)][osc_names[azimuth]] = pickle.load(
                                open(
                                    f"{info['PATH']}/signal/osc/azimuth_{azimuth}/{config}/{config}_{name.split('_')[0]}_{comp}_exposure_azimuth_{azimuth}_dm2_{dm2}_sin13_{sin13}_sin12_{sin12}.pkl",
                                    "rb",
                                )
                            )
                            kde_p = pickle.load(
                                open(
                                    f"{info['PATH']}/signal/osc/azimuth_{azimuth}/{config}/{config}_{name.split('_')[0]}_{comp}_kde_p_azimuth_{azimuth}_dm2_{dm2}_sin13_{sin13}_sin12_{sin12}.pkl",
                                    "rb",
                                )
                            )
                            run[tree][
                                f"SignalParticleWeight{label}Osc{osc_names[azimuth]}"
                            ][idx] = evaluate_1d_pdf(
                                kde_p, run[tree]["SignalParticleP"][idx]
                            )

                    if osc == "truth":
                        exposure[(label, osc)]["truth"] = pickle.load(
                            open(
                                f"{info['PATH']}/signal/truth/{config}/{config}_{name.split('_')[0]}_{comp}_exposure.pkl",
                                "rb",
                            )
                        )
                        kde_p = pickle.load(
                            open(
                                f"{info['PATH']}/signal/truth/{config}/{config}_{name.split('_')[0]}_{comp}_kde_p.pkl",
                                "rb",
                            )
                        )
                        run[tree][f"SignalParticleWeight{label}"][idx] = (
                            evaluate_1d_pdf(kde_p, run[tree]["SignalParticleP"][idx])
                        )

            elif params["PARTICLE_TYPE"] == "background":
                if params["PARTICLE_WEIGHTING"] in ["histogram", "surface"]:
                    alpha_truth = []
                    areas = json.load(open(f"{root}/import/surface_areas.json", "r"))
                    for geometry, surfaces in areas.items():
                        for surface, area in surfaces.items():
                            areas[geometry][surface] = eval(area)
                    A = areas[info["GEOMETRY"].lower()]
                    exposure = pickle.load(
                        open(
                            f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/background/truth/{config}/{config}_{name.split('_')[0]}_exposure.pkl",
                            "rb",
                        )
                    )

                    # rprint(exposure)
                    total_counts = sum(
                        [this_exposure["counts"] for this_exposure in exposure.values()]
                    )
                    for this_exposure in exposure.values():
                        alpha_truth.append(this_exposure["counts"] / total_counts)
                    # rprint(f"Alpha truth: {alpha_truth}")
                    # for surface_label, (surface_value, surface_id) in surfaces[
                    #     info["GEOMETRY"]
                    # ].items():
                    #     if surface_id < 0:
                    #         continue
                    #     if surface_id == 0:
                    #         A.append(
                    #             (info["PRODUCTION_SIZE_Y"])
                    #             * (info["PRODUCTION_SIZE_Z"])
                    #         )
                    #     elif surface_id in [1, 2]:
                    #         if info["VERSION"] == "hd_1x2x6_lateralAPA":
                    #             A.append(
                    #                 (info["PRODUCTION_SIZE_X"])
                    #                 / 2
                    #                 * (info["PRODUCTION_SIZE_Z"])
                    #             )
                    #         else:
                    #             A.append(
                    #                 (info["PRODUCTION_SIZE_X"])
                    #                 * (info["PRODUCTION_SIZE_Z"])
                    #             )
                    #     elif surface_id in [3, 4]:
                    #         if info["VERSION"] == "hd_1x2x6_lateralAPA":
                    #             A.append(
                    #                 (info["PRODUCTION_SIZE_X"])
                    #                 / 2
                    #                 * (info["PRODUCTION_SIZE_Y"])
                    #             )
                    #         else:
                    #             A.append(
                    #                 (info["PRODUCTION_SIZE_X"])
                    #                 * (info["PRODUCTION_SIZE_Y"])
                    #             )
                    #     else:
                    #         raise ValueError(f"Invalid surface_id: {surface_id}")

                    if params["PARTICLE_WEIGHTING"] == "histogram":
                        pdf_hist, pdf_bins = pickle.load(
                            open(
                                f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/background/truth/{config}/{config}_{name.split('_')[0]}_pdf.pkl",
                                "rb",
                            )
                        )

                        run[tree]["SignalParticleWeight"][idx] = (
                            evaluate_background_pdf(
                                info,
                                name,
                                A,
                                alpha_truth,
                                run[tree]["SignalParticleP"][idx],
                                run[tree]["SignalParticleSurface"][idx],
                                pdf_hist,
                                pdf_bins,
                            )
                        )

                    elif params["PARTICLE_WEIGHTING"] == "surface":
                        kde_files = {"p": "kde_p", "uv": "kde_uv"}

                        kdes = {
                            key: pickle.load(
                                open(
                                    f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/background/truth/{config}/{config}_{name.split('_')[0]}_{val}.pkl",
                                    "rb",
                                )
                            )
                            for key, val in kde_files.items()
                        }

                        # rprint(f"Surface areas: {A}")
                        run[tree]["SignalParticleWeight"][idx] = evaluate_surface_pdf(
                            info,
                            name,
                            A,
                            alpha_truth,
                            run[tree]["SignalParticleP"][idx],
                            run[tree]["SignalParticleX"][idx],
                            run[tree]["SignalParticleY"][idx],
                            run[tree]["SignalParticleZ"][idx],
                            run[tree]["SignalParticleSurface"][idx],
                            kdes["p"],
                            kdes["uv"],
                        )
                    else:
                        rprint(
                            f"[red][ERROR][/red] PARTICLE_WEIGHTING should be 'histogram', 'surface' or 'volume' in config for background weight computation..."
                        )

                elif params["PARTICLE_WEIGHTING"] == "volume":
                    rprint(
                        f"\t\t[yellow][WARNING][/yellow] Using volume weighting for background particle {name}...\n"
                    )
                    kde_files = {"p": "kde_p", "x": "kde_x", "y": "kde_y", "z": "kde_z"}
                    kdes = {
                        key: pickle.load(
                            open(
                                f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/background/truth/{config}/{config}_{name.split('_')[0]}_{val}.pkl",
                                "rb",
                            )
                        )
                        for key, val in kde_files.items()
                    }

                    exposure[("", "truth")] = {
                        "truth": pickle.load(
                            open(
                                f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/background/truth/{config}/{config}_{name.split('_')[0]}_exposure.pkl",
                                "rb",
                            )
                        )
                    }

                    run[tree]["SignalParticleWeight"][idx] = evaluate_joint_pdf(
                        run[tree]["SignalParticleP"][idx],
                        run[tree]["SignalParticleX"][idx],
                        run[tree]["SignalParticleY"][idx],
                        run[tree]["SignalParticleZ"][idx],
                        kdes["p"],
                        kdes["x"],
                        kdes["y"],
                        kdes["z"],
                    )
                else:
                    rprint(
                        f"[red][ERROR][/red] PARTICLE_WEIGHTING should be 'histogram', 'surface' or 'volume' in config for background weight computation..."
                    )
            else:
                rprint(
                    f"[red][ERROR][/red] Particle type {params['PARTICLE_TYPE']} not recognized for weight computation..."
                )
    return run, output, exposure, new_branches


def normalize_true_weights(
    run: dict[str, dict],
    configs: dict[str, list[str]],
    params: dict = {},
    trees: list[str] = ["Truth", "Reco"],
    output: str = "",
    exposure: dict = {},
    debug: bool = False,
):
    if output is None:
        output = ""

    new_weights = [""]
    surfaces = json.load(open(f"{root}/import/surface_positions.json"))
    # surface_weights = json.load(open(f"{root}/import/surface_weights.json"))
    # surface_production = json.load(open(f"{root}/import/surface_production.json"))
    config_tree_weight = json.load(open(f"{root}/import/productions.json"))
    # Planes * productions * events per production
    for config in configs:
        info, params, output = get_param_dict(
            f"{root}/config/{config}/{config}", params, output, debug=debug
        )
        for name, tree in product(configs[config], trees):
            new_weights = [""]
            if name.lower().startswith("marley"):
                new_weights = ["", "b8", "hep"]
            idx = np.where(
                (np.asarray(run[tree]["Geometry"]) == info["GEOMETRY"])
                * (np.asarray(run[tree]["Version"]) == info["VERSION"])
                * (np.asarray(run[tree]["Name"]) == name)
            )
            jdx = np.where(
                (np.asarray(run["Config"]["Version"]) == info["VERSION"])
                * (np.asarray(run["Config"]["Name"]) == name)
            )

            for weight in new_weights:
                for osc in params["DEFAULT_SIGNAL_WEIGHT"]:
                    this_weight_name = None
                    if osc == "osc":
                        if params["PARTICLE_TYPE"] == "signal":
                            for osc_name in params["DEFAULT_SIGNAL_AZIMUTH"]:
                                this_weight_name = f"SignalParticleWeight{weight}Osc{osc_names[osc_name]}"
                                this_exposure = exposure[(weight, "osc")][
                                    osc_names[osc_name]
                                ]
                                this_event_count = (
                                    config_tree_weight[config][name]
                                    * len(run["Config"]["Geometry"][jdx])
                                    if (
                                        config in config_tree_weight
                                        and name in config_tree_weight[config]
                                    )
                                    else np.sum(run["Config"]["AnalyzedEvents"][jdx])
                                )
                                run[tree][this_weight_name][idx] = (
                                    this_exposure["counts"]
                                    * run[tree][this_weight_name][idx]
                                    / np.sum(run[tree][this_weight_name][idx])
                                    / this_exposure["exposure"]
                                )
                                output += f"\t\t[cyan][INFO][/cyan] Normalized {this_weight_name}\twith {this_exposure['counts']:.0f} / {this_exposure['exposure']:.0f} counts / kT·y for {this_event_count} / {len(run[tree]['Event'][idx])} Produced / {tree} events...\n"
                        else:
                            this_weight_name = None
                            rprint(
                                f"[red][ERROR][/red] PARTICLE_TYPE {params['PARTICLE_TYPE']} not recognized for weight normalization..."
                            )

                    elif osc == "truth":
                        # Compute truth weights by normalizing to exposure
                        if params["PARTICLE_TYPE"] == "background":
                            # Renormalize surface weights
                            for surface_label, (surface_value, surface_id) in surfaces[
                                info["GEOMETRY"]
                            ].items():
                                if surface_id < 0:
                                    continue

                                this_weight_name = f"SignalParticleWeight{weight}"
                                this_exposure = exposure[surface_id]
                                this_event_count = config_tree_weight[config][
                                    name
                                ] * len(run["Config"]["Geometry"][jdx])

                                mask = (
                                    run[tree]["SignalParticleSurface"][idx]
                                    == surface_id
                                )
                                run[tree][this_weight_name][idx[0][mask]] = (
                                    this_exposure["counts"]
                                    * len(run[tree]["Event"][idx[0][mask]])
                                    * run[tree][this_weight_name][idx[0][mask]]
                                    # * surface_production[info["VERSION"]][
                                    #     str(surface_id)
                                    # ]
                                    # / surface_weights[info["VERSION"]][name][
                                    #     str(surface_id)
                                    # ]
                                    / this_event_count
                                    / np.sum(run[tree][this_weight_name][idx[0][mask]])
                                    / this_exposure["exposure"]
                                )

                        elif params["PARTICLE_TYPE"] == "signal":
                            this_weight_name = f"SignalParticleWeight{weight}"
                            this_exposure = exposure[(weight, "truth")]["truth"]
                            this_event_count = (
                                config_tree_weight[config][name]
                                * len(run["Config"]["Geometry"][jdx])
                                if (
                                    config in config_tree_weight
                                    and name in config_tree_weight[config]
                                )
                                else np.sum(run["Config"]["AnalyzedEvents"][jdx])
                            )
                            run[tree][this_weight_name][idx] = (
                                this_exposure["counts"]
                                * len(run[tree]["Event"][idx])
                                / this_event_count
                                * run[tree][this_weight_name][idx]
                                / (
                                    np.sum(run[tree][this_weight_name][idx])
                                    * this_exposure["exposure"]
                                )
                            )
                            output += f"\t\t[cyan][INFO][/cyan] Normalized {this_weight_name}\twith {this_exposure['counts']:.0f} / {this_exposure['exposure']:.0f} counts / kT·y for {this_event_count} / {len(run[tree]['Event'][idx])} Produced / {tree} events...\n"

                        else:
                            this_weight_name = None
                            rprint(
                                f"[red][ERROR][/red] PARTICLE_TYPE {params['PARTICLE_TYPE']} not recognized for weight normalization..."
                            )

                    else:
                        this_weight_name = None
                        rprint(
                            f"[red]ERROR: Sigles weight should be 'osc' or 'truth'[/red]"
                        )
                    # If there are any nan values in the weights, set them to 0 and print a warning if more than 1% of the weights are nan
                    if (
                        this_weight_name is not None
                        and np.isnan(run[tree][this_weight_name][idx]).any()
                    ):
                        nan_fraction = np.isnan(
                            run[tree][this_weight_name][idx]
                        ).sum() / len(run[tree][this_weight_name][idx])
                        if nan_fraction > 0.01:
                            output += f"[yellow][WARNING][/yellow] {100 * nan_fraction:.2f}% of weights are NaN for {this_weight_name} in {name} for {tree} in {config} after normalization...\n"
                        run[tree][this_weight_name][idx] = np.where(
                            np.isnan(run[tree][this_weight_name][idx]),
                            0,
                            run[tree][this_weight_name][idx],
                        )

    return run, output, new_weights


# def compute_reco_weights(
#     run: dict[str, dict],
#     configs: dict[str, list[str]],
#     params: Optional[dict] = None,
#     new_weights: Optional[list[str]] = None,
#     output: str = "",
#     debug: bool = False,
# ):
#     if new_weights is None:
#         new_weights = [""]

#     for config in configs:
#         info, params, output = get_param_dict(
#             f"{root}/config/{config}/{config}", params, output, debug=debug
#         )

#         for weight in new_weights:
#             run["Reco"][f"SignalParticleWeight{weight}"] = run["Truth"][
#                 f"SignalParticleWeight{weight}"
#             ][run["Reco"]["TrueIndex"]]

#             if params is not None:
#                 for osc in params["DEFAULT_SIGNAL_WEIGHT"]:
#                     if osc == "osc":
#                         for osc_name in params["DEFAULT_SIGNAL_AZIMUTH"]:
#                             run["Reco"][
#                                 f"SignalParticleWeight{weight}Osc{osc_names[osc_name]}"
#                             ] = run["Truth"][
#                                 f"SignalParticleWeight{weight}Osc{osc_names[osc_name]}"
#                             ][
#                                 run["Reco"]["TrueIndex"]
#                             ]
#             else:
#                 rprint(
#                     f"[red][ERROR][/red] No parameters found for config {config} during reco weight computation..."
#                 )
#     return run, output
