import sys, inquirer, os
import numpy as np
from icecream import ic

from .io_functions import print_colored

def get_flag_dict(debug = False):
    '''
    This function returns a dictionary with the available flags for the macros.
    
    Args:
        debug (bool): If True, the debug mode is activated.
    
    Returns:
        flag_dict (dict): Dictionary with the available flags for the macros.
    '''
    flag_dict = {("-c","--config_file"):"config_file \t(hd, vd, etc.)",
        ("-d","--debug"):"debug \t(True/False)",
        ("-r","--root_file"):"root_file \t(Marley, wbkg, etc.)",
        ("-rw","--rewrite"):"rewrite \t(True/False)",
        ("-s","--show"):"show \t(Show resulting plots)",
        ("-t","--trim"):"trim \t(True/False)",
        ("-v","--variable"):"variable \t(RecoEnergy, TotalEnergy, etc.)",
        }
    if debug: ic(flag_dict)
    return flag_dict

def initialize_macro(macro, input_list=["config_file","root_file","debug"], default_dict={}, debug=False):
    '''
    This function initializes the macro by reading the input file and the user input.
    
    Args:
        macro (str): Name of the macro to be executed.
        input_list (list(str)): List with the keys of the user input that need to be updated.
        default_dict (dict): Dictionary with the default values for the user input.
        debug (bool): If True, the debug mode is activated.
    
    Returns:
        user_input (dict): Dictionary with the user input.
    '''
    from .io_functions import print_colored

    flag_dict = get_flag_dict()
    user_input = dict()
    
    print_header()
    if len(sys.argv) > 1:
        for arg in sys.argv:
            if arg == "-h" or arg == "--help":
                for flag in flag_dict:
                    print_macro_info(macro)
                    print_colored("Usage: python3 %s.py -i config_file *--flags"%macro,color="white",bold=True)
                    print_colored("Available Flags:","INFO",bold=True)
                    for flag in flag_dict:
                        print_colored("%s: %s"%(flag[0],flag_dict[flag]),"INFO")
                    exit()

            for flag in flag_dict:
                if arg == flag[0] or arg == flag[1]:
                    try:
                        user_input[flag[1].split("--")[1]] = sys.argv[sys.argv.index(arg)+1].split(",")
                        print_colored("Using %s from command line"%flag_dict[flag],"INFO")
                    except IndexError:
                        print("Please provide argument for flag %s"%flag_dict[flag])
                        exit()

    user_input = select_input_file(user_input, debug=debug)
    user_input = update_user_input(user_input,input_list,debug=debug)
    
    user_input["config_file"] = user_input["config_file"][0]
    for bool_key in ["debug","rewrite","show","trim"]:
        if bool_key in user_input.keys():
            user_input[bool_key] = user_input[bool_key][0].lower() in ['true', '1', 't', 'y', 'yes']
    return user_input

def update_user_input(user_input, new_input_list, debug=False):
    '''
    This function updates the user input by asking the user to provide the missing information.

    Args:
        user_input (dict): Dictionary with the user input.
        new_input_list (list(str)): List with the keys of the user input that need to be updated.
        debug (bool): If True, the debug mode is activated.

    Returns:
        new_user_input (dict): Dictionary with the updated user input.
    '''
    from .io_functions import check_key

    new_user_input = user_input.copy()
    for key_label in new_input_list:
        if check_key(user_input, key_label) == False:
            new_user_input[key_label]= input("Please select %s (separated with commas): "%key_label).split(",")
        else:
            # if debug: print("Using %s from user input"%key_label)
            pass
    
    return new_user_input

def select_input_file(user_input, debug=False):
    '''
    This function asks the user to select the input file.
    **VARIABLES:**
    \n** - user_input:** Dictionary with the user input.
    \n** - debug:**      If True, the debug mode is activated.
    '''
    from .io_functions import check_key, print_colored
    
    new_user_input = user_input.copy()
    if check_key(user_input, "config_file") == False:
        folder_names = [file_name.replace(".txt", "") for file_name in os.listdir('../config/')]
        q1 = [ inquirer.List("config_file", message="Please select input file", choices=folder_names, default="hd") ]
        input_folder = inquirer.prompt(q1)["config_file"]
        # Check if config file exists in the selected folder
        if not os.path.exists('../config/'+input_folder+'/'+input_folder+'_config.txt'):
            print_colored("WARNING: No config file found in folder %s"%input_folder,"WARNING")
            file_names = [file_name.replace(".txt", "") for file_name in os.listdir('../config/'+input_folder+'/')]
            q2 = [ inquirer.List("config_file", message="Please select input file", choices=file_names, default="hd") ]
            new_user_input["config_file"] = [input_folder+"/"+inquirer.prompt(q2)["config_file"]]
        else:
            print_colored("-> Using config file %s"%input_folder+"_config","SUCCESS")
            new_user_input["config_file"] = [input_folder+"/"+input_folder+"_config"]
    # if debug: print_colored("Using config file %s"%new_user_input["config_file"][0],"INFO")
    return new_user_input

def use_default_input(user_input, default_dict, debug=False):
    '''
    This function updates the user input by asking the user to provide the missing information.
    **VARIABLES:**
    \n** - user_input:** Dictionary with the user input.
    \n** - info:**       Dictionary with the information from the input file.
    \n** - debug:**      If True, the debug mode is activated.
    '''
    from .io_functions import check_key, print_colored, read_input_file
    
    info = read_input_file(user_input["config_file"])
    new_user_input = user_input.copy()
    return new_user_input

def check_macro_config(user_input, debug=False):
    '''
    This function asks the user to confirm the macro configuration.
    **VARIABLES:**
    \n** - user_input:** Dictionary with the user input.
    '''
    # Print macro configuration
    print_colored("\nMacro configuration:","WARNING")
    for key in user_input.keys():
        print("  "+key+": "+str(user_input[key]))

    proceed = input("Proceed? [y/n]: ")
    if proceed.lower() in ["n","no","f","false"]: sys.exit()

    return user_input

def print_macro_info(macro, debug=False):
    f = open('../info/'+macro+'.txt', 'r')
    file_contents = f.read()
    print (file_contents+"\n")
    f.close
    if debug: print("----- Debug mode activated -----")

def print_header(debug = False):
    f = open('../info/header.txt', 'r')
    file_contents = f.read()
    print (file_contents)
    f.close
    print("----- Starting macro -----")