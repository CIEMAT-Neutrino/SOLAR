import uproot
import os
import copy
import stat
import json
import numpy as np
import pandas as pd
import awkward as ak
import plotly
import plotly.express as px
import plotly.graph_objs as go

from typing import Optional
from rich import print as rprint
from src.utils import get_project_root
from matplotlib import pyplot as plt

root = get_project_root()


def save_figure(fig, path: str, rm: bool = False, output: str = "png", debug: bool = False) -> None:
    """
    Save the figure in the path

    Args:
        fig (plotly.graph_objs._figure.Figure): figure to save
        path (str): path to save the figure
        debug (bool): if True, the debug mode is activated (default: False)
    """
    folder_path = "/".join(path.split("/")[:-1])
    try:
        # Make the directory recursively
        os.makedirs(folder_path)
    except FileExistsError:
        if debug:
            print_colored("DATA STRUCTURE ALREADY EXISTS", "DEBUG")

    # Check if figure already exists
    if os.path.isfile(f"{path}.{output}"):
        if rm:
            os.remove(f"{path}.{output}")
            if debug:
                rprint(f"Removed existing figure in: {path}.{output}")
        else:
            if debug:
                rprint("File already exists. Skipping...", "WARNING")
            return 0

    # Check type of figure to select the correct saving method
    if type(fig) == go._figure.Figure or type(fig) == plotly.graph_objs._figure.Figure:
        fig.write_image(f"{path}.{output}")
        if debug:
            rprint(f"Saved figure in: {path}.{output}")

    elif type(fig) == plt.Figure:
        fig.savefig(f"{path}.{output}")
        if debug:
            rprint(f"Saved figure in: {path}.{output}")
    else:
        rprint("The input figure is not a known type: ", type(fig))
        # fig.savefig(path + ".{output}")


def print_colored(string, color, bold=False, italic=False, debug=False):
    """
    Print a string in a specific color

    Args:
        string (str): string to be printed
        color (str): color to be used
        bold (bool): if True, the bold mode is activated (default: False)
    """
    colors = {
        "DEBUG": "purple",
        "ERROR": "red",
        "SUCCESS": "green",
        "WARNING": "yellow",
        "INFO": "blue",
    }
    if color in list(colors.values()):
        color = colors[color]

    if bold == False and italic == False:
        output = f"[{colors[color]}]{string}[/{colors[color]}]"
    elif bold == True and italic == False:
        output = f"[bold {colors[color]}]{string}[/bold {colors[color]}]"
    elif bold == False and italic == True:
        output = (
            "["
            + "italic "
            + colors[color]
            + "]"
            + string
            + "["
            + "/italic "
            + colors[color]
            + "]"
        )
    elif bold == True and italic == True:
        output = (
            "["
            + "bold italic "
            + colors[color]
            + "]"
            + string
            + "["
            + "/bold italic "
            + colors[color]
            + "]"
        )
    else:
        output = string

    rprint(output)
    return 0


def root2npy(root_info, user_input, trim=False, debug=False):
    """
    Dumper from .root format to npy files. Input are root input file, path and npy outputfile as strings

    Args:
        root_info (dict): dictionary with the information of the root file
        trim (bool): if True, trim the array to the selected size (default: False)
        debug (bool): if True, the debug mode is activated (default: False)

    Returns:
        0 (int): if the function is executed correctly
    """
    path = root_info["Path"]
    name = root_info["Name"]
    rprint("Converting from: " +
           root_info["Path"] + root_info["Name"] + ".root")
    with uproot.open(root_info["Path"] + root_info["Name"] + ".root") as f:
        for tree in root_info["TreeNames"]:
            if root_info["TreeNames"][tree].lower() == "test":
                rprint(f"[red]Skipping Test tree[/red]")
                continue

            done_root_info = []
            out_folder = root_info["TreeNames"][tree]

            if debug:
                rprint("----------------------")
                rprint("Dumping file:" + str(path + name))

            for branch in root_info[tree]:
                if branch not in done_root_info:
                    # To avoid repeating branches
                    done_root_info.append(branch)
                print_colored(
                    "\n" + tree + " ---> " + out_folder + ": " + str(branch),
                    "SUCCESS",
                )

                # if "Map" not in branch:
                this_array = f[root_info["Folder"] +
                               "/" + tree][branch].array()
                if trim != False and debug:
                    print("Selected trimming value: ", trim)
                resized_array = resize_subarrays(
                    this_array, 0, trim=trim, debug=debug)
                save2pnfs(
                    path + name + "/" + out_folder + "/" + branch + ".npy",
                    user_input,
                    resized_array,
                    debug
                )
                if debug:
                    print(resized_array)
                del resized_array
                del this_array

                print_colored(
                    "\nSaved data in:" + str(path + name + "/" + out_folder),
                    "SUCCESS",
                )
                rprint(f"[green]----------------------\n[/green]")

    if debug:
        rprint(f"[green]-> Finished dumping root file to npy files![/green]")
    return 0


def resize_subarrays(array, value, trim=False, debug=False):
    """
    Resize the arrays so that the have the same lenght and numpy can handle them
    The arrays with len < max_len are filled with 0 until they have max_len

    Args:
        array (list): array to resize
        value (int): value to fill the array
        trim (bool): if True, trim the array to the selected size (default: False)
        debug (bool): if True, the debug mode is activated (default: False)

    Returns:
        tot_array (np.array): resized array
    """
    array = array2list(array, debug=debug)
    check_expand = False

    try:
        np.asarray(array, dtype=float)
        check_expand = False
        if debug:
            rprint(f"[green]-> Array can be converted to numpy array![/green]")
    except ValueError:
        check_expand = True
        if debug:
            rprint(
                f"[yellow]-> Array needs resizing for numpy conversion![/yellow]")

    if check_expand:
        expand = False
        if type(trim) == bool:
            max_len = max(map(len, array))
            mean_len = sum(map(len, array)) / len(array)
            if debug:
                rprint(
                    f"[cyan]-> Max/Mean length of subarrays {max_len}/{mean_len:.2f}[/cyan]")

            if max_len != mean_len:
                expand = True
                if trim:
                    std_len = np.std(list(map(len, array)))
                    max_len = int(mean_len + std_len)

        elif type(trim) == int:
            max_len = trim
            expand = True

        if expand:
            tot_array = resize_subarrays_fixed(
                array, value, max_len, debug=debug)
        else:
            tot_array = array
    else:
        tot_array = np.asarray(array)

    if debug:
        rprint(f"[green]-> Returning array as type: {type(tot_array)}[/green]")
    return np.asarray(tot_array)


def resize_subarrays_fixed(array, value, max_len, debug=False):
    """
    Resize the arrays so that the have the same lenght and numpy can handle them
    The arrays with len < size are filled with 0 until they have selected size

    Args:
        array (list): array to resize
        value (int): value to fill the array
        max_len (int): size of the array
        debug (bool): if True, the debug mode is activated (default: False)

    Returns:
        return_array (np.array): resized array
    """
    try:
        tot_array = [
            this_array.tolist()[:max_len]
            if len(this_array.tolist()) > max_len
            else this_array.tolist() + [value] * (max_len - len(this_array))
            for this_array in array
        ]
        if debug:
            rprint(
                f"[green]-> Successfully resized array to {max_len}[/green]")
    except AttributeError:
        tot_array = [
            this_array[:max_len]
            if len(this_array) > max_len
            else this_array + [value] * (max_len - len(this_array))
            for this_array in array
        ]
        if debug:
            rprint(
                f"[green]-> Successfully resized array to {max_len}[/green]")
    except TypeError:
        tot_array = array
        if debug:
            rprint(f"[yellow]-> Array is not a list of lists[/yellow]")

    return_array = np.asarray(tot_array)
    return return_array


def array2list(array, debug=False):
    """
    Check if the array is a list of lists or a list of arrays

    Args:
        array (list): array to check
        debug (bool): if True, the debug mode is activated (default: False)

    Returns:
        array (list): array converted to list
    """
    if type(array) == np.ndarray:
        array = array.tolist()

    elif type(array) == list:
        array = array

    elif type(array) == ak.highlevel.Array:
        array = array.to_list()

    else:
        print("Array type: ", type(array))
        if debug:
            rprint("[red]ERROR: Array type not recognized[/red]")
        raise TypeError

    if debug:
        rprint(f"--- Array lenght is {len(array)} ---")
    return array


def get_tree_info(root_file, debug=False):
    """
    From a root file (root_file = uproot.open(path+name+".root")) you get the two lists with:
    \n - directory: list of the directories in the root file
    \n - tree: list of the trees in the root file

    Args:
        root_file (uproot.rootio.ROOTDirectory): root file to analyze
        debug (bool): if True, the debug mode is activated (default: False)

    Returns:
        directory (list): list of the directories in the root file
    """

    directory = [
        i for i in root_file.classnames() if root_file.classnames()[i] == "TDirectory"
    ]
    tree = [
        i.split("/")[1]
        for i in root_file.classnames()
        if root_file.classnames()[i] == "TTree"
    ]
    print_colored(
        "The input root file has a TDirectory: " + str(directory), color="DEBUG"
    )
    print_colored(
        "The input root file has %i TTrees: " % len(tree) + str(tree), color="DEBUG"
    )

    return directory, tree


def get_root_info(name: str, path: str, user_input: dict, debug=False):
    """
    Function which returns a dictionary with the following structure:
    \n {"Path": path, "Name": name, "Folder": folder (from get_tree_info), "TreeNames": {"RootName_Tree1":YourName_Tree1, "RootName_Tree2":YourName_Tree2}, "RootName_Tree1":[BRANCH_LIST], "RootName_Tree2":[BRANCH_LIST]}

    Args:
        name (str): name of the root file.
        path (str): path of the root file.
        debug (bool): if True, the debug mode is activated (default: False)

    Returns:
        output (dict): dictionary with the information of the root file
    """

    f = uproot.open(path + name + ".root")
    folder, trees = get_tree_info(f, debug=debug)

    input_loop = True
    while input_loop:
        default = input("Continue with default processing? (y/n): ")
        if default.lower() in ["y", "yes"]:
            if len(trees) == 2:
                rename_default = input(
                    "Use default tree names (Truth - Reco)? (y/n): ")
                if rename_default.lower() in ["y", "yes"]:
                    out_folder = ["Truth", "Reco"]
                    input_loop = False

            elif len(trees) == 3:
                rename_default = input(
                    "Use default tree names (Config - Truth - Reco)? (y/n): "
                )
                if rename_default.lower() in ["y", "yes"]:
                    out_folder = ["Config", "Truth", "Reco"]
                    input_loop = False

            else:
                print_colored(
                    "The number of TTrees is not as expected. Please, select coustom names.",
                    "WARNING",
                )

        elif default.lower() in ["n", "no"]:
            print("Select custom names for the TTrees (name 'Test' will be skipped)")
            out_folder = input(
                "Write the new names of the TTrees separated by commas: "
            ).split(",")
            # Check input is correct
            print("You wrote: " + str(out_folder))
            check = input("Is this correct? (y/n): ")
            if check.lower() in ["y", "yes"]:
                input_loop = False

        else:
            print("Please, write 'y' or 'n'")

    output = dict()

    output["Path"] = path
    output["Name"] = name
    output["Folder"] = folder[0]
    output["TreeNames"] = dict()

    for i, tree in enumerate(trees):
        branches = []
        if debug:
            print_colored(
                "--- BRANCH #%i --> Original Tree Name:" % i
                + str(tree)
                + "... Renaming to: ... "
                + str(out_folder[i]),
                "DEBUG",
            )

        try:
            os.mkdir(path + name)
        except FileExistsError:
            print("DATA STRUCTURE ALREADY EXISTS")

        try:
            os.mkdir(path + name + "/" + out_folder[i])
        except FileExistsError:
            print(out_folder[i])

        # save branches in a list
        for branch in f[folder[0] + "/" + tree].keys():
            if "Map" in branch:
                branch = branch.split("_")[
                    0
                ]  # remove the "_keys" or "_values" from the branch name
            if branch not in branches:
                branches.append(branch)  # save the branch name in a list

        # Check if file already exists
        save2pnfs(f"{path}{name}/{out_folder[i]}/Branches.npy",
                  user_input, branches, user_input["debug"])
        output[tree] = np.asarray(branches, dtype=object)
        output["TreeNames"][tree] = out_folder[i]
        save2pnfs(f"{path}{name}/TTrees.npy", user_input,
                  output, user_input["debug"])

    if debug:
        rprint(output)
    return output


def save2pnfs(filename: str, user_input: dict, data, debug: bool) -> None:
    """
    Save the data in a .npy file to pnfs (dCache) storage.

    Args:
        filename (str): name of the file to save
        user_input (dict): dictionary with the user input
        data (np.array): data to save
        debug (bool): if True, the debug mode is activated (default: False)
    """
    # Check data filetype
    if debug:
        rprint(type(data))
    if type(data) != np.ndarray:
        data = np.asarray(data)

    if os.path.isfile(filename):
        if user_input["rewrite"]:
            # Delete the file
            os.remove(filename)
            # save the branches of each tree in a .npy file
            np.save(filename, data)
        else:
            rprint("File already exists. Skipping...")
    else:
        # save the branches of each tree in a .npy file
        np.save(filename, data)


def get_branches(name: str, path: str, debug=False):
    """
    Function which returns a dictionary with the following structure:
    \n {"YourName_Tree1":[BRANCH_LIST], "YourName_Tree2":[BRANCH_LIST]}

    Args:
        name (str): name of the root file.
        path (str): path of the root file.
        debug (bool): if True, the debug mode is activated (default: False)

    Returns:
        branch_dict (dict): dictionary with the branches of the root file
    """
    branch_dict = dict()
    tree_info = np.load(path + name + "/" + "TTrees.npy",
                        allow_pickle=True).item()

    for tree in tree_info["TreeNames"].keys():
        branch_dict[tree_info["TreeNames"][tree]] = tree_info[tree]

    if debug:
        rprint(branch_dict)
    return branch_dict


def get_branches2use(run, debug=False):
    """
    Function to get the branches of the TTree not in ['Name', 'Path', 'Labels', 'Colors']

    Args:
        run (dict): dictionary with the data to load
        debug (bool): if True, the debug mode is activated (default: False)
    """
    branches_raw = list(run.keys())
    branches = [
        i for i in branches_raw if i not in ["Name", "Path", "Labels", "Colors"]
    ]
    if debug:
        print_colored(
            "\nFounded keys " + str(branches) +
            " to construct the dictionaries.",
            "DEBUG",
        )
    return branches


def check_key(my_dict, key, debug=False):
    """
    Check if a given dict contains a key and return True or False
    """
    try:
        my_dict[key]
        return True
    except KeyError:
        return False


def delete_keys(run, keys, debug=False):
    """
    Delete the keys list introduced as 2nd variable
    """
    for key in keys:
        del run[key]
    return run


def remove_processed_branches(root_info, debug=False):
    """
    Removes the branches that have been already processed

    Args:
        root_info (dict): dictionary with the information of the root file
        debug (bool): if True, the debug mode is activated (default: False)

    Returns:
        root_info (dict): updated dictionary with the information of the root file
    """
    path = root_info["Path"] + root_info["Name"] + "/"
    for tree in root_info["TreeNames"]:
        # print(root_info[tree])
        dir_list = os.listdir(path + root_info["TreeNames"][tree] + "/")
        remove_idx = []
        for branch in dir_list:
            for i in range(len(root_info[tree])):
                if branch.split(".")[0] == root_info[tree][i]:
                    if debug:
                        print_colored(
                            "REMOVING " + branch + " from " + tree + "", color="WARNING"
                        )
                    remove_idx.append(i)
        root_info[tree] = np.delete(root_info[tree], remove_idx)
        if debug:
            print_colored(
                "New branch list to process for Tree %s: %s" % (
                    tree, root_info[tree]),
                color="SUCCESS",
            )
    return root_info


def load_multi(
    configs: dict,
    tree_labels: list = ["Config", "Truth", "Reco"],
    load_all: bool = False,
    preset=None,
    branches: Optional[dict] = None,
    generator_swap=False,
    debug=False,
):
    """
    Load multiple files with different configurations and merge them into a single dictionary

    Args:
        names (dict): dictionary with the names of the files to load, e.g. {"hd":["Marley","wbkg]}
        configs (dict): dictionary with the configurations of the files to load, e.g. {"hd":"hd_config"}
        load_all (bool): if True, load all the branches from the input file (default: False)
        preset (str): if not "", load the branches from the preset list (default: "")
        branches (dict): dictionary with the branches to load (default: {})
        generator_swap (bool): if True, swap the generator for the background files (default: False)
        debug (bool): if True, the debug mode is activated (default: False)

    Returns:
        run (dict): dictionary with the loaded data
    """
    out = ""
    run = dict()
    for idx, config in enumerate(configs):
        info = json.load(
            open(f"{root}/config/{config}/{config}_config.json", "r"))
        path = info["PATH"]
        name = info["NAME"]
        geo = info["GEOMETRY"]
        vers = info["VERSION"]
        filepath = f"{root}{path}{name}"
        bkg_dict, color_dict = get_bkg_config(info)
        inv_bkg_dict = {v: k for k, v in bkg_dict.items()}

        for jdx, name in enumerate(configs[config]):
            if load_all == True:
                branches_dict = get_branches(
                    name, path=filepath, debug=debug
                )  # Get ALL the branches
            elif preset != None:
                tree_branches = get_workflow_branches(
                    tree_list=tree_labels, workflow=preset, debug=False
                )  # Get PRESET branches
                # branches_dict = {"Truth": truth_labels, "Reco": reco_labels}
                branches_dict = {
                    tree_labels[i]: tree_branches[i] for i in range(len(tree_labels))
                }
                if debug:
                    rprint(branches_dict)
            else:
                branches_dict = branches  # Get CUSTOMIZED branches from the input

            for tree in branches_dict.keys():
                if branches_dict[tree] == []:
                    continue  # If the tree is empty, skip it
                else:
                    if (
                        idx == 0 and jdx == 0
                    ):  # If it is the first file, create the dictionary
                        run[tree] = dict()
                        for identifiyer_label, identifiyer in zip(
                            ["Name", "Geometry", "Version"], [name, geo, vers]
                        ):
                            run[tree][identifiyer_label] = [identifiyer] * len(
                                np.load(
                                    filepath + name + "/" + tree + "/Event.npy",
                                    allow_pickle=True,
                                )
                            )
                            run[tree][identifiyer_label] = np.asarray(
                                run[tree][identifiyer_label], dtype=str)
                    else:
                        run[tree]["Name"] = np.concatenate(
                            (
                                run[tree]["Name"],
                                [name]
                                * len(
                                    np.load(
                                        filepath + name + "/" + tree + "/Event.npy",
                                        allow_pickle=True,
                                    )
                                ),
                            ),
                            axis=0,
                        )
                        run[tree]["Geometry"] = np.concatenate(
                            (
                                run[tree]["Geometry"],
                                [geo]
                                * len(
                                    np.load(
                                        filepath + name + "/" + tree + "/Event.npy",
                                        allow_pickle=True,
                                    )
                                ),
                            ),
                            axis=0,
                        )
                        run[tree]["Version"] = np.concatenate(
                            (
                                run[tree]["Version"],
                                [vers]
                                * len(
                                    np.load(
                                        f"{filepath}{name}/{tree}/Event.npy",
                                        allow_pickle=True,
                                    )
                                ),
                            ),
                            axis=0,
                        )

                    for key in branches_dict[tree]:
                        try:
                            branch = np.load(
                                filepath + name + "/" + tree + "/" + key + ".npy",
                                allow_pickle=True,
                            )
                            if key == "Generator":
                                # Create a new branch with the background names according to the bkg_dict
                                label_branch = np.asarray(
                                    [bkg_dict[gen] for gen in branch], dtype=str
                                )
                                if idx == 0 and jdx == 0:
                                    run[tree]["GeneratorLabel"] = label_branch
                                else:
                                    run[tree]["GeneratorLabel"] = np.concatenate(
                                        (run[tree]["GeneratorLabel"],
                                         label_branch), axis=0
                                    )
                            if generator_swap == True:
                                if key == "Generator":
                                    if name in bkg_dict.values():
                                        out = (
                                            out
                                            + f"-> Changing the generator for {name} to {inv_bkg_dict[name]}"
                                        )
                                        mapped_gen = inv_bkg_dict[name]
                                        # branch[branch == 2] = mapped_gen # Map the generator to the correct background
                                        branch[
                                            :
                                        ] = mapped_gen  # Map the generator to the correct background

                                if key == "TruthPart" and name in bkg_dict.values():
                                    mapped_gen = inv_bkg_dict[name]
                                    branch = resize_subarrays_fixed(
                                        branch,
                                        0,
                                        max_len=len(bkg_dict.keys()),
                                        debug=debug,
                                    )
                                    branch[:, mapped_gen - 1] = branch[:, 1]
                                    if mapped_gen != 2:
                                        branch[:, 1] = 0

                            if idx == 0 and jdx == 0:
                                run[tree][key] = branch
                            else:
                                try:
                                    run[tree][key] = np.concatenate(
                                        (run[tree][key], branch), axis=0
                                    )
                                except ValueError:
                                    run[tree][key] = (
                                        run[tree][key].tolist() +
                                        branch.tolist()
                                    )
                                    run[tree][key] = resize_subarrays(
                                        run[tree][key], 0, trim=False, debug=debug
                                    )
                            if key == "Event":
                                out = (
                                    out
                                    + f"\nLoaded {config} events:\t{len(branch)}\t from {tree} -> {name}"
                                )
                        except FileNotFoundError:
                            if debug:
                                out = out + \
                                    f"\n[red]File {key} not found![/red]"

    if debug:
        for tree in branches_dict.keys():
            try:
                out = out + f"\nKeys extracted from the {tree} tree:\n"
                out = out + str(run[tree].keys())
                out = out + \
                    "\n-> Total reco clusters: %i\n" % len(run[tree]["Event"])
            except KeyError:
                out = out + "- No reco tree found!\n"

    rprint(out)
    return run


def save_proccesed_variables(run, info={}, force=False, debug=False):
    """
    Save a copy of run with all modifications.

    Args:
        run (dict): dictionary with the data to save (delete the keys that you don't want to save or set Force = False to save only the new ones)
        info (dict): dictionary with the information of the input file (default: {})
        force (bool): if True, overwrite the existing files (default: False)
        debug (bool): if True, the debug mode is activated (default: False)

    Returns:
        0 (int): if the function is executed correctly
    """

    aux = copy.deepcopy(run)
    path = info["PATH"] + info["NAME"] + "/"
    branches = get_branches2use(
        run, debug=debug
    )  # Load the branches of the TTree not in ['Name', 'Path', 'Labels', 'Colors']
    for branch in branches:
        key_list = run[branch].keys()
        files = os.listdir(path + branch)
        for key in key_list:
            key = key.replace(".npy", "")

            if key + ".npy" in files and force == False:
                print_colored("File (%s.npy) alredy exists" % key, "SUCCESS")
                continue  # If the file already exists, skip it

            elif (
                key + ".npy" in files or key + ".npy" in files
            ) and force == True:  # If the file already exists and force is True, overwrite it
                os.remove(path + branch + "/" + key + ".npy")
                np.save(path + branch + "/" + key + ".npy", aux[branch][key])
                os.chmod(
                    path + branch + "/" + key + ".npy",
                    stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO,
                )
                print_colored("File (%s.npy) OVERWRITTEN " % key, "WARNING")

            else:  # If the file does not exist, create it
                np.save(path + branch + "/" + key + ".npy", aux[branch][key])
                os.chmod(
                    path + branch + "/" + key + ".npy",
                    stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO,
                )
                print_colored("Saving NEW file: %s.npy" % key, "SUCCESS")
                print_colored(path + branch + "/" + key + ".npy", "SUCCESS")

    del run
    if debug:
        print_colored("Saved data in: " + path + branch, "DEBUG")
    return 0


def get_bkg_config(info, debug=False):
    """
    This function returns a dictionary of background names according to the input file.
    Each key of the dictionary should be a tuple of the form (geometry,version) and each value should be a list of background names.

    Args:
        info (dict): dictionary with the information of the input file
        debug (bool): if True, the debug mode is activated (default: False)

    Returns:
        bkg_dict (dict): dictionary with the background names
    """
    bkg_dict = {}
    color_dict = {}
    f = json.load(open(f"{root}/lib/import/generator_order.json", "r"))
    bkg_list = f[info["GEOMETRY"]][info["VERSION"]].keys()

    color_ass = get_bkg_color(bkg_list)
    for idx, bkg in enumerate(bkg_list):
        bkg_dict[idx] = bkg
        color_dict[idx] = color_ass[bkg]

    if debug:
        print_colored("Loaded background dictionary: %s" %
                      str(bkg_dict), "INFO")
    return bkg_dict, color_dict


def get_gen_label(configs, debug=False):
    """
    Get the generator label from configuration.
    """
    gen_dict = dict()
    for idx, config in enumerate(configs):
        info = json.load(
            open(f"{root}/config/{config}/{config}_config.json", "r"))
        geo = info["GEOMETRY"]
        version = info["VERSION"]
        for idx, gen in enumerate(get_bkg_config(info, debug)[0].values()):
            gen_dict[(geo, version, idx)] = gen
    return gen_dict


def weight_lists(mean_truth_df, count_truth_df, count_reco_df, config, debug=False):
    """ """
    info = json.load(open(f"{root}/config/{config}/{config}_config.json", "r"))
    weight_list = get_bkg_weights(info)
    truth_values = []
    reco_values = []
    for bkg_idx, bkg in enumerate(mean_truth_df.index):
        truth_values.append(
            mean_truth_df.values[bkg_idx]
            * np.power(info["TIMEWINDOW"] * weight_list[bkg], -1)
        )
        reco_values.append(
            count_reco_df.values[0][bkg_idx]
            * np.power(count_truth_df.values[bkg_idx] * info["TIMEWINDOW"], -1)
        )
        if debug:
            print(count_truth_df.values[bkg_idx], reco_values)
    return truth_values, reco_values


def get_bkg_color(name_list, debug=False):
    """
    Get the color for each background according to its "simple" name.

    Args:
        name_list (list): list of the background names
        debug (bool): if True, the debug mode is activated (default: False)

    Returns:
        color_dict (dict): dictionary with the colors of the backgrounds
    """
    color_dict = dict()
    simple_name_list = get_simple_name(name_list, debug=debug)
    colors = px.colors.qualitative.Prism
    for name in name_list:
        if simple_name_list[name] == "Unknown":
            color_dict[name] = "black"
        elif simple_name_list[name] == "Marley":
            color_dict[name] = colors[6]
        elif simple_name_list[name] == "APA":
            color_dict[name] = colors[8]
        elif simple_name_list[name] == "Neutron":
            color_dict[name] = colors[3]
        elif simple_name_list[name] == "CPA":
            color_dict[name] = colors[9]
        elif simple_name_list[name] == "Ar42":
            color_dict[name] = colors[1]
        elif simple_name_list[name] == "K42":
            color_dict[name] = colors[1]
        elif simple_name_list[name] == "Kr85":
            color_dict[name] = "pink"
        elif simple_name_list[name] == "Ar39":
            color_dict[name] = colors[10]
        elif simple_name_list[name] == "Rn22":
            color_dict[name] = colors[5]
        elif simple_name_list[name] == "Po210":
            color_dict[name] = "brown"
        elif simple_name_list[name] == "PDS":
            color_dict[name] = colors[7]
        else:
            color_dict[name] = "black"

    return color_dict


def get_bkg_weights(info, names, debug=False):
    bkg_dict, color_dict = get_bkg_config(info, debug=False)
    weights_dict = dict()
    for bkg in bkg_dict.values():
        weights_dict[bkg] = 1
    if "wbkg" in names:
        weights_dict["wbkg"] = 1
        return weights_dict
    else:
        if info["GEOMETRY"] == "hd" and info["VERSION"] == "hd_1x2x6":
            custom_weights = {"NeutronsInCavernwall": 1e3}
        if info["GEOMETRY"] == "hd" and info["VERSION"] == "hd_1x2x6_legacy":
            custom_weights = {
                "Po210": 1e4,
                "APA": 1e4,
                "CPA": 1e2,
                "Ar42": 1e4,
                "Neutron": 1e3,
                "Rn222": 1e4,
            }
        if info["GEOMETRY"] == "vd" and info["VERSION"] == "vd_1x8x14_3view_30deg":
            custom_weights = {"Neutron": 1e2}

        for bkg in custom_weights:
            weights_dict[bkg] = custom_weights[bkg]
        return weights_dict


def get_gen_weights(configs, names, debug=False):
    weights_dict = dict()
    for idx, config in enumerate(configs):
        info = json.load(
            open(f"{root}/config/{config}/{name}/{config}_config.json", "r"))
        # Write a function that returns a dictionary of background names according to the input file. Each key of the dictionary should be a tuple of the form (geometry,version) and each value should be a list of background names.
        geo = info["GEOMETRY"]
        name_list = names[config]
        geo_weights_dict = get_bkg_weights(info, name_list)
        for idx, name in enumerate(name_list):
            weights_dict[(geo, name)] = geo_weights_dict[name]
    return weights_dict


def get_simple_name(name_list, debug=False):
    simple_name = dict()
    basic_names = ["Ar42", "Ar39", "Kr85", "Po210", "Rn22"]
    for name in name_list:
        if "LAr" in name:
            for basic_name in basic_names:
                if basic_name in name:
                    simple_name[name] = basic_name
            if "K42" in name:
                simple_name[name] = "Ar42"

        elif "Gamma" in name:
            simple_name[name] = "Gamma"
        elif "Neutron" in name:
            simple_name[name] = "Neutron"
        elif "CPA" in name:
            simple_name[name] = "CPA"
        elif "Cathode" in name:
            simple_name[name] = "CPA"
        elif "CRP" in name:
            simple_name[name] = "APA"
        elif "APA" in name:
            simple_name[name] = "APA"
        elif "PDS" in name:
            simple_name[name] = "PDS"
        else:
            simple_name[name] = name

    if debug:
        print_colored("Loaded simple name dictionary: %s" %
                      str(simple_name), "INFO")
    return simple_name


def get_workflow_branches(tree_list, workflow: str = "BASIC", debug=False):
    """
    Get the workflow variables from the input file.

    Args:
        workflow (str): workflow (default: "BASIC")
        debug (bool): if True, print debug messages (default: False)

    Returns:
        truth_list (list): list of truth variables
    """
    f = json.load(open(f"{root}/lib/workflow/{workflow}.json", "r"))
    branch_lists = []
    for tree in tree_list:
        tree_branch_list = []
        if tree not in f.keys():
            raise KeyError(f"Tree {tree} not found in the workflow file")
        else:
            for branch_type in f[tree]:
                for branch in f[tree][branch_type]:
                    tree_branch_list.append(branch)
            branch_lists.append(tree_branch_list)
    if debug:
        rprint()
    return branch_lists
