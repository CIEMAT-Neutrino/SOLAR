import sys, inquirer, os, json

from rich import print as rprint
from typing import Optional

from src.utils import get_project_root

root = get_project_root()


def get_flag_dict(debug=False):
    """
    This function returns a dictionary with the available flags for the macros.

    Args:
        debug (bool): If True, the debug mode is activated.

    Returns:
        flag_dict (dict): Dictionary with the available flags for the macros.
    """
    flag_dict = {
        ("-c", "--config_file"): "config_file \t(hd, vd, etc.)",
        ("-d", "--debug"): "debug \t(True/False)",
        ("-r", "--root_file"): "root_file \t(Marley, wbkg, etc.)",
        ("-rw", "--rewrite"): "rewrite \t(True/False)",
        ("-s", "--show"): "show \t(Show resulting images)",
        ("-t", "--trim"): "trim \t(True/False)",
        ("-v", "--variable"): "variable \t(SolarEnergy, TotalEnergy, etc.)",
    }
    if debug:
        rprint(flag_dict)
    return flag_dict


def initialize_macro(
    macro: str,
    input_list: Optional[list[str]] = [
        "config_file",
        "root_prefix",
        "root_file",
        "debug",
    ],
    default_dict: Optional[dict] = None,
    debug: bool = False,
) -> dict[str, str]:
    """
    This function initializes the macro by reading the input file and the user input.

    Args:
        macro (str): Name of the macro to be executed.
        input_list (list(str)): List with the keys of the user input that need to be updated.
        default_dict (dict): Dictionary with the default values for the user input.
        debug (bool): If True, the debug mode is activated.

    Returns:
        user_input (dict): Dictionary with the user input.
    """

    flag_dict = get_flag_dict()
    user_input = dict()

    print_header()
    if len(sys.argv) > 1:
        for arg in sys.argv:
            if arg == "-h" or arg == "--help":
                for flag in flag_dict:
                    print_macro_info(macro)
                    rprint(
                        f"Usage: python3 {macro}.py -i config_file *--flags\nAvailable Flags:"
                    )
                    rprint(flag_dict)
                    exit()

            for flag in flag_dict:
                if arg == flag[0] or arg == flag[1]:
                    try:
                        user_input[flag[1].split("--")[1]] = sys.argv[
                            sys.argv.index(arg) + 1
                        ].split(",")
                        rprint(f"Using {flag_dict[flag]} from command line")

                    except IndexError:
                        print("Please provide argument for flag %s" % flag_dict[flag])
                        exit()

    user_input = select_input_file(user_input, debug=debug)
    config = user_input["config_file"][0].split("/")[0]
    rprint(f"Using config file {config}")
    info = json.load(open(f"{root}/config/{config}/{config}_config.json", "r"))
    default_dict["root_prefix"] = info["NAME"]
    user_input = update_user_input(user_input, input_list, default_dict, debug=debug)

    user_input["config_file"] = user_input["config_file"][0]
    for bool_key in ["debug", "rewrite", "show", "trim"]:
        if bool_key in user_input.keys():
            user_input[bool_key] = user_input[bool_key][0].lower() in [
                "true",
                "1",
                "t",
                "y",
                "yes",
            ]
    return user_input


def update_user_input(
    user_input,
    new_input_list: Optional[list[str]] = None,
    default_dict: Optional[dict] = None,
    debug: bool = False,
):
    """
    This function updates the user input by asking the user to provide the missing information.

    Args:
        user_input (dict): Dictionary with the user input.
        new_input_list (list(str)): List with the keys of the user input that need to be updated.
        debug (bool): If True, the debug mode is activated.

    Returns:
        new_user_input (dict): Dictionary with the updated user input.
    """

    new_user_input = user_input.copy()
    for key_label in new_input_list:
        if key_label not in user_input.keys():
            if key_label in default_dict.keys():
                # Use inquirer to ask user for input with default value
                q1 = [
                    inquirer.Text(
                        key_label,
                        message="Please select %s" % key_label,
                        default=default_dict[key_label],
                    )
                ]
                new_user_input[key_label] = inquirer.prompt(q1)[key_label].split(",")[0]
            else:
                # Use inquirer to ask user for input
                q1 = [inquirer.Text(key_label, message="Please select %s" % key_label)]
                new_user_input[key_label] = inquirer.prompt(q1)[key_label].split(",")[0]
        else:
            # if debug: print("Using %s from user input"%key_label)
            pass

    return new_user_input


def select_input_file(user_input, debug=False):
    """
    This function asks the user to select the input file.

    Args:
        user_input (dict): Dictionary with the user input.
        debug (bool): If True, the debug mode is activated.

    Returns:
        new_user_input (dict): Dictionary with the updated user input.
    """

    new_user_input = user_input.copy()
    if "config_file" not in user_input.keys():
        folder_names = [
            file_name.replace(".txt", "") for file_name in os.listdir(f"{root}/config/")
        ]
        q1 = [
            inquirer.List(
                "config_file",
                message="Please select input file",
                choices=folder_names,
                default="hd",
            )
        ]
        input_folder = inquirer.prompt(q1)["config_file"]
        # Check if config file exists in the selected folder
        if not os.path.exists(
            f"{root}/config/" + input_folder + "/" + input_folder + "_config.json"
        ):
            rprint(
                f"[yellow]WARNING[/yellow]: No config file found in folder {input_folder}"
            )
            file_names = [
                file_name.replace(".txt", "")
                for file_name in os.listdir(f"{root}/config/" + input_folder + "/")
            ]
            q2 = [
                inquirer.List(
                    "config_file",
                    message="Please select input file",
                    choices=file_names,
                    default="hd",
                )
            ]
            new_user_input["config_file"] = [
                input_folder + "/" + inquirer.prompt(q2)["config_file"]
            ]
        else:
            rprint(f"-> Using config file: {input_folder}_config")
            new_user_input["config_file"] = [
                input_folder + "/" + input_folder + "_config"
            ]
    return new_user_input


def check_macro_config(user_input, debug=False):
    """
    This function asks the user to confirm the macro configuration.

    Args:
        user_input (dict): Dictionary with the user input.
        debug (bool): If True, the debug mode is activated.

    Returns:
        user_input (dict): Dictionary with the user input.
    """
    # Print macro configuration
    rprint(f"\nMacro configuration:")
    for key in user_input.keys():
        rprint("  " + key + ": " + str(user_input[key]))

    proceed = input("Proceed? [y/n]: ")
    if proceed.lower() in ["n", "no", "f", "false"]:
        sys.exit()

    return user_input


def print_macro_info(macro, debug=False):
    """
    This function prints the information of the macro.

    Args:
        macro (str): Name of the macro to be executed.
        debug (bool): If True, the debug mode is activated.
    """
    f = open(f"{root}/lib/docs/{macro}.txt", "r")
    file_contents = f.read()
    rprint(file_contents + "\n")
    f.close
    if debug:
        rprint("----- Debug mode activated -----")


def print_header(debug=False):
    """
    This function prints the header of the macro.
    """
    f = open(f"{root}/lib/docs/header.txt", "r")
    file_contents = f.read()
    rprint(file_contents)
    f.close
    rprint("----- Starting macro -----")
