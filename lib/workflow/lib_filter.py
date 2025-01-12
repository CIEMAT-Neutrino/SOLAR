import yaml
import json
import numpy as np

from typing import Optional

from src.utils import get_project_root
root = get_project_root()

def import_filter_preset(params:dict, config: str, presets: Optional[list[str]], output: Optional[str] = None, debug: bool = False)-> dict:
    """
    Import the filter configuration from the yml file.

    Args:
        params: dictionary containing the filter configuration
        config: path to the configuration file
        preset: name of the preset to be imported
        output: output string
        debug: print debug information

    Returns:
        params: dictionary containing the filter configuration
    """
    if output is None:
        output = ""

    if presets is None:
        return params, output

    if presets is not None:
        # Load yml config file
        with open(f"{root}/config/{config}/{config}_filter.yml", "r") as file:
            filter_config = yaml.load(file, Loader=yaml.FullLoader)
            for preset in presets:
                try:
                    filter_config[preset]
                except KeyError:
                    output += f"[red][ERROR]Preset {preset} not found in the filter configuration![/red]"
                    continue
                for tree_dict in filter_config[preset]:
                    tree_name = list(tree_dict.keys())[0]
                    if list(tree_dict.values())[0] is None:
                        continue
                    else:
                        for branch_dict in list(tree_dict.values())[0]: 
                            branch_name = list(branch_dict.keys())[0]
                            filter_key = (tree_name, branch_name)
                            filter_cut = (list(branch_dict[branch_name][0].keys())[0], list(branch_dict[branch_name][0].values())[0]) 
                            params[filter_key] = filter_cut
    
    return params, output   


def compute_filtered_run(run: dict, configs: dict[str, list[str]], params: Optional[dict] = None, presets: Optional[list[str]] = None, output: Optional[str] = None, debug: bool = False):
    """
    Function to filter all events in the run according to the filters defined in the params dictionary.
    """
    new_run = {}
    if type(params) != dict and params != None:
        output += f"[red][ERROR]Params must be a dictionary![/red]"
        return run, output

    if output is None:
        output = ""

    if params == None and presets == None:
        if debug:
            output += "[yellow][WARNING] No filter applied![/yellow]"
        return run, output
    
    elif params == None and presets != None:
        params = {}

    elif params != None and presets != None:
        output += f"[cyan][INFO] Combining preset {presets} with custom filters[/cyan]"

    new_trees = run.keys()
    branch_ref = {"Config": "Geometry", "Truth": "Event", "Reco": "Event"}
    for tree in new_trees:
        new_run[tree] = {}
        branch_list = list(run[tree].keys())
        idx = np.zeros(len(run[tree][branch_ref[tree]]), dtype=bool)
        kdx = np.zeros(len(run[tree][branch_ref[tree]]), dtype=bool)

        # Make sure that only the entries that correspond to the correct geometry, version and name are selected
        for config in configs:
            info = json.load(
                open(f"{root}/config/{config}/{config}_config.json", "r"))
            params, output = import_filter_preset(params, config, presets, output, debug)                 
            for name in configs[config]:
                config_filter = (run[tree]["Geometry"] == info["GEOMETRY"]) & (
                    run[tree]["Version"] == info["VERSION"]) & (run[tree]["Name"] == name)
                jdx = idx + config_filter
        
                if debug:
                    output += f"From {len(run[tree][branch_ref[tree]])} events, {len(jdx)} have been selected by configs for {tree} tree.\n"
                
                for param in params:
                    if param[0] != tree:
                        continue

                    if not isinstance(params[param], tuple) and not isinstance(params[param], list):
                        output += f"[red][ERROR]: Filter must be tuple or list, but found {type(params[param])}[/red]"
                        if debug: rprint(f"{param}: {params[param]}")
                        return run, output
                    
                    if len(params[param]) != 2:
                        output += f"[red][ERROR]: Filter must be of length 2![/red]"
                        if debug: rprint(f"{param}: {params[param]}")
                        return run, output

                    if param[1] not in run[param[0]].keys():
                        output += f"[red][ERROR]: Branch {param[1]} not found in the run![/red]"
                        if debug: rprint(f"{param}: {params[param]}")
                        return run, output

                    if params[param][0] == "bigger":
                        jdx = jdx & (run[param[0]][param[1]] > params[param][1])
                    elif params[param][0] == "smaller":
                        jdx = jdx & (run[param[0]][param[1]] < params[param][1])
                    elif params[param][0] == "equal":
                        jdx = jdx & (run[param[0]][param[1]] == params[param][1])
                    elif params[param][0] == "different":
                        jdx = jdx & (run[param[0]][param[1]] != params[param][1])
                    elif params[param][0] == "between":
                        jdx = jdx & (run[param[0]][param[1]] > params[param][1][0]) & \
                                    (run[param[0]][param[1]] < params[param][1][1])
                    elif params[param][0] == "absbetween":
                        jdx = jdx & (abs(run[param[0]][param[1]]) > params[param][1][0]) & \
                                    (abs(run[param[0]][param[1]]) < params[param][1][1])
                    elif params[param][0] == "outside":
                        jdx = jdx & ((run[param[0]][param[1]] < params[param][1][0]) + \
                                    (run[param[0]][param[1]] > params[param][1][1]))
                    elif params[param][0] == "absoutside":
                        jdx = jdx & ((abs(run[param[0]][param[1]]) < params[param][1][0]) + \
                                    (abs(run[param[0]][param[1]]) > params[param][1][1]))
                    elif params[param][0] == "contains":
                        jdx = jdx & np.array(
                            [params[param][1] in item for item in run[param[0]][param[1]]])
                    if debug:
                        try:
                            output += f"-> {param[1]}: {params[param][0]} ({params[param][1][0]:.1f}, {params[param][1][1]:.1f}) -> {np.sum(jdx):.1E} ({100*np.sum(jdx)/len(run[tree][branch_ref[tree]]):.1f}%) events\n"
                        except:
                            output += f"-> {param[1]}: {params[param][0]} {params[param][1]} -> {np.sum(jdx):.1E} ({100*np.sum(jdx)/len(run[tree][branch_ref[tree]]):.1f}%) events\n"
                
                kdx = kdx + jdx

        combined_filter = np.where(kdx == True)
        for branch in branch_list:
            try:
                new_run[tree][branch] = np.asarray(run[tree][branch])[combined_filter]
            except Exception as e:
                output += f"[red][ERROR] Couldn't filter branch {branch}: {e}[/red]"

    return new_run, output


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
    params:Optional[dict]=None,
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