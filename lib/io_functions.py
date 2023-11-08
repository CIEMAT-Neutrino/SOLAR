import uproot, os, copy, stat, json
import numpy   as np
import pandas  as pd
import awkward as ak

from icecream import ic

def print_colored(string, color, bold=False, italic=False, debug=False):
    '''
    Print a string in a specific color

    Args:
        string (str): string to be printed
        color (str): color to be used
        bold (bool): if True, the bold mode is activated (default: False)
    '''
    from rich import print as rprint
    colors = {"DEBUG":'purple',"ERROR":'red',"SUCCESS":'green',"WARNING":'yellow',"INFO":'blue'}
    if color in list(colors.values()): color = colors[color]
    
    if bold == False and italic == False:   output = '['+colors[color]+']' + string + '['+colors[color]+']'
    elif bold == True  and italic == False: output = '['+'bold '+colors[color]+']' + string + '['+'/bold '+colors[color]+']'
    elif bold == False and italic == True:  output = '['+'italic '+colors[color]+']' + string + '['+'/italic '+colors[color]+']'
    elif bold == True  and italic == True:  output = '['+'bold italic '+colors[color]+']' + string + '['+'/bold italic '+colors[color]+']'
    else: output = string
    
    rprint(output)
    return 0
    
def read_input_file(input_file, path="../config/", preset="default_input", INTEGERS=[], DOUBLES=[], STRINGS=[], BOOLS=[], debug=False):
    '''
    Obtain the information stored in a .txt input file to load the runs and channels needed

    Args:
        input (str): name of the input file
        path (str): path of the input file (default: "../config/")
        INTEGERS (list(str)): list of the variables to be read as integers (default: [])
        DOUBLES (list(str)): list of the variables to be read as floats (default: [])
        STRINGS (list(str)): list of the variables to be read as strings (default: [])
        BOOLS (list(str)): list of the variables to be read as booleans (default: [])
        debug (bool): if True, the debug mode is activated (default: False)

    Returns:
        info (dict): dictionary with the information from the input file
    '''
    info = dict()
    if INTEGERS == [] and DOUBLES == [] and STRINGS == [] and BOOLS == []: 
        json_config = json.load(open(path+preset+".json","r"))
        DOUBLES  = json_config["DOUBLES"]
        INTEGERS = json_config["INTEGERS"]
        STRINGS  = json_config["STRINGS"]
        BOOLS    = json_config["BOOLS"]

    for key in INTEGERS+DOUBLES+STRINGS+BOOLS: info[key] = None
    with open(path+input_file+".txt", 'r') as txt_file:
        lines = txt_file.readlines()

        for line in lines:
            for key in DOUBLES:
                if line.startswith(key):
                    info[key] = []
                    doubles = line.split(" ")[1]
                    for i in doubles.split(","):
                        info[key].append(float(i))
                        # if debug: print_colored(string="Found %s: "%key+str(info[key]), color="DEBUG")
            for key in INTEGERS:
                if line.startswith(key):
                    info[key] = []
                    integers = line.split(" ")[1]
                    for i in integers.split(","):
                        info[key].append(int(i))
                        # if debug: print_colored(string="Found %s: "%key+str(info[key]), color="DEBUG")
            for key in STRINGS:
                if line.startswith(key):
                    info[key] = []
                    strings = line.split(" ")[1]
                    for i in strings.split(","):
                        info[key].append(i.strip('\n'))
                        # if debug: print_colored(string="Found %s: "%key+str(info[key]), color="DEBUG")
            for key in BOOLS:
                if line.startswith(key):
                    info[key] = []
                    bools = line.split(" ")[1]
                    for i in bools.split(","):
                        if i.strip('\n') == "True":  info[key].append(True)
                        if i.strip('\n') == "False": info[key].append(False)
                        # if debug: print_colored(string="Found %s: "%key+str(info[key]), color="DEBUG")
    if debug: ic(info)
    return info

def root2npy(root_info, trim=False, debug=False):
    '''
    Dumper from .root format to npy files. Input are root input file, path and npy outputfile as strings

    Args:
        root_info (dict): dictionary with the information of the root file
        trim (bool): if True, trim the array to the selected size (default: False)
        debug (bool): if True, the debug mode is activated (default: False)

    Returns:
        0 (int): if the function is executed correctly
    '''
    path = root_info["Path"]
    name = root_info["Name"]
    if debug: print_colored("Converting from: "+root_info["Path"]+root_info["Name"]+".root","DEBUG")
    with uproot.open(root_info["Path"]+root_info["Name"]+".root") as f:
        for tree in root_info["TreeNames"].keys():
            done_root_info = []
            out_folder = root_info["TreeNames"][tree]

            if debug:
                print_colored("----------------------","DEBUG")
                print_colored("Dumping file:"+ str(path+name),"DEBUG")

            for branch in root_info[tree]:
                if branch not in done_root_info: done_root_info.append(branch) # To avoid repeating branches
                if debug: print_colored("\n"+tree+" ---> "+out_folder+": " + str(branch),"SUCCESS")

                # if "Map" not in branch: 
                this_array    = f[root_info["Folder"]+"/"+tree][branch].array()
                if trim != False and debug: print("Selected trimming value: ", trim)
                resized_array = resize_subarrays(this_array, 0, trim=trim, debug=debug)
                np.save(path+name+"/"+out_folder+"/"+branch+".npy", resized_array)
                if debug: print(resized_array)
                del resized_array
                del this_array

                # else: 
                #     print("Branch is a map: ", branch)
                #     dicts = []
                #     this_keys   = f[root_info["Folder"]+"/"+tree][branch+"_keys"].array()
                #     this_values = f[root_info["Folder"]+"/"+tree][branch+"_values"].array()
                #     if debug: print_colored("Using keys: " + str(this_keys[0]),"DEBUG"); print_colored("Using values: " + str(this_values[0]),"DEBUG")
                #     for j in range(len(this_keys)): dicts.append(dict(zip(this_keys[j], this_values[j])))
                #     np.save(path+name+"/"+out_folder+"/"+branch+".npy", dicts)
                #     print(dicts[0])
                #     del dicts

                if debug: 
                    print_colored("\nSaved data in:" + str(path+name+"/"+out_folder),"SUCCESS")
                    print_colored("----------------------\n","SUCCESS")
    
    if debug: print_colored("-> Finished dumping root file to npy files!","SUCCESS")
    return 0
                    
def resize_subarrays(array, value, trim=False, debug=False):
    '''
    Resize the arrays so that the have the same lenght and numpy can handle them
    The arrays with len < max_len are filled with 0 until they have max_len

    Args:
        array (list): array to resize
        value (int): value to fill the array
        trim (bool): if True, trim the array to the selected size (default: False)
        debug (bool): if True, the debug mode is activated (default: False)

    Returns:
        tot_array (np.array): resized array
    '''    
    array = array2list(array,debug=debug)
    check_expand = False
    
    try:
        np.asarray(array, dtype=float)
        check_expand = False
        if debug: print_colored("-> Array can be converted to numpy array!","SUCCESS")
    except ValueError:
        check_expand = True
        if debug: print_colored("-> Array needs resizing for numpy conversion!","WARNING")
    
    if check_expand:
        expand = False
        if type(trim) == bool:
            max_len = max(map(len, array))
            mean_len = sum(map(len, array))/len(array)
            if debug: print_colored("Max/Mean length of subarrays are %s/%s: "%(str(max_len),str(mean_len)),"DEBUG")

            if max_len != mean_len:
                expand = True
                if trim: 
                    std_len = np.std(list(map(len, array))) 
                    max_len = int(mean_len+std_len)

        elif type(trim) == int:
            max_len = trim
            expand = True
        
        if expand:tot_array = resize_subarrays_fixed(array, value, max_len, debug=debug)
        else:tot_array = array
    else:
        tot_array = np.asarray(array)

    if debug: print_colored("-> Returning array as type: %s"%(type(tot_array)),"SUCCESS")
    return np.asarray(tot_array)

def resize_subarrays_fixed(array, value, max_len, debug=False):
    '''
    Resize the arrays so that the have the same lenght and numpy can handle them
    The arrays with len < size are filled with 0 until they have selected size

    Args:
        array (list): array to resize
        value (int): value to fill the array
        max_len (int): size of the array
        debug (bool): if True, the debug mode is activated (default: False)

    Returns:
        return_array (np.array): resized array
    '''
    try:
        tot_array = [this_array.tolist()[:max_len] if len(this_array.tolist()) > max_len else this_array.tolist()+[value]*(max_len-len(this_array)) for this_array in array]
        print_colored("-> Successfully resized array to %i"%max_len,"SUCCESS")
    except AttributeError:
        tot_array = [this_array[:max_len] if len(this_array) > max_len else this_array+[value]*(max_len-len(this_array)) for this_array in array]
        print_colored("-> Successfully resized array to %i"%max_len,"SUCCESS")
    except TypeError:
        tot_array = array
        print_colored("-> Array is not a list of lists","WARNING")

    return_array = np.asarray(tot_array)
    return return_array

def array2list(array, debug=False):
    '''
    Check if the array is a list of lists or a list of arrays

    Args:
        array (list): array to check
        debug (bool): if True, the debug mode is activated (default: False)

    Returns:
        array (list): array converted to list
    '''
    if type(array) == np.ndarray:
        array = array.tolist()
        if debug: print_colored("Array type is a numpy array","INFO")
        return array

    elif type(array) == list:
        array = array
        if debug: print_colored("Array type is a list","INFO")
        return array

    elif type(array) == ak.highlevel.Array:
        array = array.to_list()
        if debug: print_colored("Array type is a awkward array","INFO")
        return array
    
    else:
        print("Array type: ", type(array))
        if debug: print_colored("Array type not recognized","ERROR")
        raise TypeError

def get_tree_info(root_file, debug=False):
    '''
    From a root file (root_file = uproot.open(path+name+".root")) you get the two lists with:
    \n - directory: list of the directories in the root file
    \n - tree: list of the trees in the root file

    Args:
        root_file (uproot.rootio.ROOTDirectory): root file to analyze
        debug (bool): if True, the debug mode is activated (default: False)
    
    Returns:
        directory (list): list of the directories in the root file
    '''

    directory = [i for i in root_file.classnames() if root_file.classnames()[i]=="TDirectory"]
    tree      = [i.split("/")[1] for i in root_file.classnames() if root_file.classnames()[i]=="TTree"]
    if debug:
        print_colored("The input root file has a TDirectory: " + str(directory),  color="DEBUG")
        print_colored("The input root file has %i TTrees: "%len(tree) +str(tree), color="DEBUG")

    return directory,tree

def get_root_info(name:str, path:str, debug=False):
    '''
    Function which returns a dictionary with the following structure:
    \n {"Path": path, "Name": name, "Folder": folder (from get_tree_info), "TreeNames": {"RootName_Tree1":YourName_Tree1, "RootName_Tree2":YourName_Tree2}, "RootName_Tree1":[BRANCH_LIST], "RootName_Tree2":[BRANCH_LIST]}

    Args:
        name (str): name of the root file.
        path (str): path of the root file.
        debug (bool): if True, the debug mode is activated (default: False)

    Returns:
        output (dict): dictionary with the information of the root file
    '''

    f = uproot.open(path+name+".root")
    folder, trees = get_tree_info(f,debug=debug)
    
    input_loop = True
    while input_loop:
        rename = input("Rename the TTrees? (y/n): ")
        if rename.lower() in ["y","yes"]:
            if len(trees) == 1:
                rename_default = input("Use default tree names (Truth)? (y/n): ")
                if rename_default.lower() in ["y","yes"]:
                    out_folder = ["Truth"]
                    input_loop = False

            elif len(trees) == 2:
                rename_default = input("Use default tree names (Truth - Reco)? (y/n): ")
                if rename_default.lower() in ["y","yes"]:
                    out_folder = ["Truth", "Reco"]
                    input_loop = False
            
            elif len(trees) == 3:
                rename_default = input("Use default tree names (Truth - Interaction - Reco)? (y/n): ")
                if rename_default.lower() in ["y","yes"]:
                    out_folder = ["Truth", "Interaction", "Reco"]
                    input_loop = False

            else:
                print_colored("There are more than 3 trees in the root file. Please, rename them manually", "WARNING")
        
        elif rename.lower() in ["n","no"]: 
            out_folder = input("Write the new names of the TTrees separated by commas: ").split(",")
            # Check input is correct
            print("You wrote: " + str(out_folder))
            check = input("Is this correct? (y/n): ")
            if check.lower() in ["y","yes"]:
                input_loop = False
        
        else: print("Please, write 'y' or 'n'")

    output = dict()

    output["Path"]      = path
    output["Name"]      = name
    output["Folder"]    = folder[0]
    output["TreeNames"] = dict()

    for i,tree in enumerate(trees):
        branches = []
        if debug: print_colored("--- BRANCH #%i --> Original Tree Name:"%i + str(tree) + "... Renaming to: ... " + str(out_folder[i]), "DEBUG")

        try:   os.mkdir(path+name)
        except FileExistsError: print("DATA STRUCTURE ALREADY EXISTS") 
        
        try:   os.mkdir(path+name+"/"+out_folder[i])
        except FileExistsError: print(out_folder[i]) 

        # save branches in a list
        for branch in f[folder[0]+"/"+tree].keys(): 
            if "Map" in branch:        branch = branch.split("_")[0] # remove the "_keys" or "_values" from the branch name
            if branch not in branches: branches.append(branch)    # save the branch name in a list
        
        np.save(path+name+"/"+out_folder[i]+"/Branches.npy",np.asarray(branches)) # save the branches of each tree in a .npy file
        output[tree] = np.asarray(branches, dtype=object)
        output["TreeNames"][tree] = out_folder[i]
        np.save(path+name+"/"+"TTrees.npy", output) 

    if debug: print(output)

    return output

def get_branches(name:str, path:str, debug=False):
    '''
    Function which returns a dictionary with the following structure:
    \n {"YourName_Tree1":[BRANCH_LIST], "YourName_Tree2":[BRANCH_LIST]}

    Args:
        name (str): name of the root file.
        path (str): path of the root file.
        debug (bool): if True, the debug mode is activated (default: False)
    
    Returns:
        branch_dict (dict): dictionary with the branches of the root file
    '''
    branch_dict = dict()
    tree_info   = np.load(path+name+"/"+"TTrees.npy", allow_pickle=True).item()
    
    for tree in tree_info["TreeNames"].keys():
        branch_dict[tree_info["TreeNames"][tree]] = tree_info[tree]
    
    if debug: ic(branch_dict)
    return branch_dict

def get_branches2use(run, debug=False):
    '''
    Function to get the branches of the TTree not in ['Name', 'Path', 'Labels', 'Colors']

    Args:
        run (dict): dictionary with the data to load
        debug (bool): if True, the debug mode is activated (default: False)
    '''
    branches_raw = list(run.keys())
    branches     = [i for i in branches_raw if i not in ['Name', 'Path', 'Labels', 'Colors']]
    if debug: print_colored("\nFounded keys " + str(branches) + " to construct the dictionaries.", "DEBUG")
    return branches

def check_key(my_dict,key,debug=False):
    '''
    Check if a given dict contains a key and return True or False
    '''
    try: my_dict[key]; return True    
    except KeyError:   return False

def delete_keys(run,keys,debug=False):
    '''
    Delete the keys list introduced as 2nd variable
    '''
    for key in keys: del run[key]
    return run

def remove_processed_branches(root_info,debug=False):
    '''
    Removes the branches that have been already processed
    
    Args:
        root_info (dict): dictionary with the information of the root file
        debug (bool): if True, the debug mode is activated (default: False)
    
    Returns:
        root_info (dict): updated dictionary with the information of the root file
    '''    
    path = root_info["Path"] + root_info["Name"] + "/"
    for tree in root_info["TreeNames"]:
        # print(root_info[tree])
        dir_list = os.listdir(path + root_info["TreeNames"][tree] + "/")
        remove_idx = []
        for branch in dir_list:
            for i in range(len(root_info[tree])):
                if branch.split(".")[0] == root_info[tree][i]:
                    if debug: print_colored("REMOVING " + branch + " from " + tree + "",color="WARNING")
                    remove_idx.append(i)    
        root_info[tree] = np.delete(root_info[tree],remove_idx)    
        if debug: print_colored("New branch list to process for Tree %s: %s"%(tree,root_info[tree]),color="SUCCESS")
    return root_info

def load_multi(names:dict,configs:dict,load_all=False,preset="",branches={},generator_swap=False,debug=False):
    '''
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
    '''
    run = dict()
    for idx,config in enumerate(configs):
        if debug: print_colored("\nLoading config %s"%configs[config],"DEBUG")
        info = read_input_file(config+'/'+configs[config],debug=False)
        path = info["PATH"][0]+info["NAME"][0]
        geo  = info["GEOMETRY"][0]
        vers = info["VERSION"][0]
        bkg_dict,color_dict = get_bkg_config(info)
        inv_bkg_dict = {v: k for k, v in bkg_dict.items()}

        for jdx,name in enumerate(names[config]):
            if debug: print_colored("\nLoading file %s%s"%(path,name),"DEBUG")
            if load_all == True: branches_dict = get_branches(name,path=path,debug=debug)    # Get ALL the branches
            else: branches_dict = branches                                                   # Get CUSTOMIZED branches from the input

            for tree in branches_dict.keys(): 
                if branches_dict[tree] == []: continue # If the tree is empty, skip it
                else:
                    if idx == 0 and jdx == 0: # If it is the first file, create the dictionary
                        run[tree] = dict()
                        run[tree]["Name"]     = [name]*len(np.load(path+name+"/"+tree+"/Event.npy",allow_pickle=True)) 
                        run[tree]["Geometry"] = [geo]*len(np.load(path+name+"/"+tree+"/Event.npy",allow_pickle=True)) 
                        run[tree]["Version"]  = [vers]*len(np.load(path+name+"/"+tree+"/Event.npy",allow_pickle=True)) 
                    else:
                        run[tree]["Name"]     = np.concatenate((run[tree]["Name"],[name]*len(np.load(path+name+"/"+tree+"/Event.npy",allow_pickle=True))),axis=0)
                        run[tree]["Geometry"] = np.concatenate((run[tree]["Geometry"],[geo]*len(np.load(path+name+"/"+tree+"/Event.npy",allow_pickle=True))),axis=0)
                        run[tree]["Version"]  = np.concatenate((run[tree]["Version"],[vers]*len(np.load(path+name+"/"+tree+"/Event.npy",allow_pickle=True))),axis=0)

                    for key in branches_dict[tree]:
                        # if debug: print_colored("\nLoading %s/%s/%s"%(name,tree,key),"DEBUG")
                        # try: 
                        branch = np.load(path+name+"/"+tree+"/"+key+".npy",allow_pickle=True)
                        # print("WE ARE HERE")
                        if generator_swap == True:
                            if key == "Generator" and name in bkg_dict.values(): 
                                print_colored("-> Changing the generator for %s to %s"%(name,inv_bkg_dict[name]),"WARNING")
                                mapped_gen = inv_bkg_dict[name]
                                # branch[branch == 2] = mapped_gen # Map the generator to the correct background
                                branch[:] = mapped_gen # Map the generator to the correct background

                            if key == "TruthPart" and name in bkg_dict.values():
                                # print("HERE")
                                mapped_gen = inv_bkg_dict[name]                            
                                branch = resize_subarrays_fixed(branch, 0, max_len=len(bkg_dict.keys()),debug=debug)
                                branch[:,mapped_gen-1] = branch[:,1]
                                if mapped_gen != 2: branch[:,1] = 0
                            # except FileNotFoundError: print_colored("File not found!","ERROR"); continue

                        if idx == 0 and jdx == 0: run[tree][key] = branch
                        else:
                            try: run[tree][key] = np.concatenate((run[tree][key],branch),axis=0)
                            except ValueError:
                                run[tree][key] = run[tree][key].tolist()+branch.tolist()
                                run[tree][key] = resize_subarrays(run[tree][key], 0, trim=False, debug=debug)
                        if key == "Event": print_colored("Loaded %s events:\t%i\t from %s -> %s"%(config,len(branch),tree,name),"INFO")
    
    try:
        print("\n- Keys extracted from the truth tree:\n",run["Truth"].keys(),"\n") # Check that all keys from the original TTree are recovered!
        print("- Total events: ",len(run["Truth"]["Event"]),"\n") 
    except:
        print("\n- No truth tree found!\n")
    try:
        print("- Keys extracted from the reco tree:\n",run["Reco"].keys(),"\n")  
        print("- Total reco clusters: ",len(run["Reco"]["Event"]),"\n") 
    except KeyError:
        print("- No reco tree found!\n")

    print_colored("\nLoaded *%s* files with trees: %s\n"%(list(configs),list(run.keys())),"SUCCESS")
    return run

def save_proccesed_variables(run, info={}, force=False, debug=False):
    '''
    Save a copy of run with all modifications. 

    Args:
        run (dict): dictionary with the data to save (delete the keys that you don't want to save or set Force = False to save only the new ones)
        info (dict): dictionary with the information of the input file (default: {})
        force (bool): if True, overwrite the existing files (default: False)
        debug (bool): if True, the debug mode is activated (default: False)

    Returns:
        0 (int): if the function is executed correctly
    '''

    aux  = copy.deepcopy(run) 
    path = info["PATH"][0] + info["NAME"][0] + "/"
    branches = get_branches2use(run,debug=debug) # Load the branches of the TTree not in ['Name', 'Path', 'Labels', 'Colors']
    for branch in branches:
        key_list = run[branch].keys()
        files = os.listdir(path+branch)
        for key in key_list:
                key = key.replace(".npy","")

                if key+".npy" in files and force == False: print_colored("File (%s.npy) alredy exists"%key,"SUCCESS"); continue # If the file already exists, skip it
                
                elif (key+".npy" in files or key+".npy" in files) and force == True: # If the file already exists and force is True, overwrite it       
                    os.remove(path+branch+"/"+key+".npy")
                    np.save(path+branch+"/"+key+".npy",aux[branch][key])
                    os.chmod(path+branch+"/"+key+".npy", stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                    print_colored("File (%s.npy) OVERWRITTEN "%key, "WARNING")
                
                else: # If the file does not exist, create it
                    np.save(path+branch+"/"+key+".npy",aux[branch][key])
                    os.chmod(path+branch+"/"+key+".npy", stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                    print_colored("Saving NEW file: %s.npy"%key, "SUCCESS")
                    print_colored(path+branch+"/"+key+".npy", "SUCCESS")

    del run
    if debug: print_colored("Saved data in: "+path+branch, "DEBUG")
    return 0

def get_bkg_config(info,debug=False):
    '''
    This function returns a dictionary of background names according to the input file.
    Each key of the dictionary should be a tuple of the form (geometry,version) and each value should be a list of background names.

    Args:
        info (dict): dictionary with the information of the input file
        debug (bool): if True, the debug mode is activated (default: False)

    Returns:
        bkg_dict (dict): dictionary with the background names
    '''
    bkg_dict = {}; color_dict = {}
    if (info["GEOMETRY"][0] == "hd" and info["VERSION"][0] == "hd_1x2x6"):
        bkg_list = ["Unknown",
            "Marley",
            "Ar39GenInLAr",
            "Kr85GenInLAr",
            "Ar42GenInLAr",
            "K42From42ArGenInLAr",
            "Rn222ChainRn222GenInLAr",
            "Rn222ChainPo218GenInLAr",
            "Rn222ChainPb214GenInLAr",
            "Rn222ChainBi214GenInLAr",
            "Rn222ChainPb210GenInLAr",
            "Rn220ChainPb212GenInLAr",
            "K40GenInCPA",
            "U238ChainGenInCPA",
            "K42From42ArGenInCPA",
            "Rn222ChainPo218GenInCPA",
            "Rn222ChainPb214GenInCPA",
            "Rn222ChainBi214GenInCPA",
            "Rn222ChainPb210GenInCPA",
            "Rn222ChainFromBi210GenInCPA",
            "Rn220ChainFromPb212GenInCPA",
            "Co60GenInAPA",
            "U238ChainGenInAPA",
            "Th232ChainGenInAPA",
            "Rn222ChainGenInPDS",
            "GammasInCavernwall",
            "GammasInFoam",
            "NeutronsInCavernwall",
            "GammasInCryostat",
            "GammasInCavern"]
            
    elif info["GEOMETRY"][0] == "hd" and info["VERSION"][0] == "hd_1x2x6_legacy":
        bkg_list = ["Unknown",
            "Marley",
            "APA",
            "Neutron",
            "Po210InLAr",
            "CPA",
            "Ar42InLAr",
            "Kr85InLAr",
            "Ar39InLAr",
            "Rn222InLAr",
            ]

    elif info["GEOMETRY"][0] == "vd" and info["VERSION"][0] == "vd_1x8x14_3view_30deg":
        bkg_list = ["Unknown",
            "Marley",
            "Ar39InLAr",
            "Kr85InLAr",
            "Ar42InLAr",
            "K42-Ar42InLAr",
            "Rn222InLAr",
            "CPAK42-Ar42",
            "CPAK40",
            "CPAU238",
            "PDSRn222",
            "Neutron",
            "Gamma"]
        
    elif info["GEOMETRY"][0] == "vd" and info["VERSION"][0] == "vd_1x8x6_3view_30deg":
        bkg_list = ["Unknown",
            "marley",
            "Ar39GenInLAr",
            "Kr85GenInLAr",
            "Ar42GenInLAr",
            "K42From42ArGenInLAr",
            "Rn222ChainRn222GenInLAr",
            "Rn220ChainPb212GenInLAr",
            "K40GenInCathode",
            "U238ChainGenInCathode",
            "K42From42ArGenInCathode",
            "Rn220ChainFromPb212GenInCathode",
            "Rn222ChainGenInPDS",
            "GammasInCavernwall",
            "GammasInFoam",
            "NeutronsInCavernwall",
            "GammasInCryostat",
            "GammasInCavern"]
    
    color_ass = get_bkg_color(bkg_list)
    for idx,bkg in enumerate(bkg_list):
        bkg_dict[idx] = bkg
        color_dict[idx] = color_ass[bkg]
        
    if debug: print_colored("Loaded background dictionary: %s"%str(bkg_dict),"INFO")
    return bkg_dict,color_dict

def get_gen_label(configs, debug=False):
    '''
    Get the generator label from configuration.
    '''
    gen_dict = dict()
    for idx,config in enumerate(configs):
        info = read_input_file(config+'/'+configs[config],debug=debug)
        geo = info["GEOMETRY"][0]
        version = info["VERSION"][0]
        for idx,gen in enumerate(get_bkg_config(info,debug)[0].values()):
            gen_dict[(geo,version,idx)] = gen
    return gen_dict

def weight_lists(mean_truth_df, count_truth_df, count_reco_df, config, debug=False):
    '''
    
    '''
    info = read_input_file(config,path="../config/"+config+"/",debug=debug)
    weight_list = get_bkg_weights(info)
    truth_values = []; reco_values = []
    for bkg_idx,bkg in enumerate(mean_truth_df.index):
        truth_values.append(mean_truth_df.values[bkg_idx]*np.power(info["TIMEWINDOW"][0]*weight_list[bkg],-1))
        reco_values.append(count_reco_df.values[0][bkg_idx]*np.power(count_truth_df.values[bkg_idx]*info["TIMEWINDOW"][0],-1))
        if debug: print(count_truth_df.values[bkg_idx],reco_values)
    return truth_values,reco_values

def get_bkg_color(name_list, debug=False):
    '''
    Get the color for each background according to its "simple" name.

    Args:
        name_list (list): list of the background names
        debug (bool): if True, the debug mode is activated (default: False)

    Returns:
        color_dict (dict): dictionary with the colors of the backgrounds
    '''
    color_dict = dict()
    simple_name_list = get_simple_name(name_list,debug=debug)

    for name in name_list:
        if simple_name_list[name] == "Unknown":
            color_dict[name] = "black"
        elif simple_name_list[name] == "Marley":
            color_dict[name] = "orange"
        elif simple_name_list[name] == "APA":
            color_dict[name] = "violet"
        elif simple_name_list[name] == "Neutron":
            color_dict[name] = "green"
        elif simple_name_list[name] == "CPA":
            color_dict[name] = "purple"
        elif simple_name_list[name] == "Ar42":
            color_dict[name] = "blue"
        elif simple_name_list[name] == "K42":
            color_dict[name] = "blue"
        elif simple_name_list[name] == "Kr85":
            color_dict[name] = "pink"
        elif simple_name_list[name] == "Ar39":
            color_dict[name] = "grey"
        elif simple_name_list[name] == "Rn22":
            color_dict[name] = "yellow"
        elif simple_name_list[name] == "Po210":
            color_dict[name] = "brown"
        elif simple_name_list[name] == "PDS":
            color_dict[name] = "red"
        else:
            color_dict[name] = "black"

    return color_dict

def reorder_df(df,info, bkg_dict, color_dict, debug=False):
    '''
    Reorder the dataframe according to the background dictionary.

    Args:
        df (pd.DataFrame): dataframe to reorder
        info (dict): dictionary with the information of the input file
        bkg_dict (dict): dictionary with the background names
        color_dict (dict): dictionary with the colors of the backgrounds
        debug (bool): if True, the debug mode is activated (default: False)

    Returns:
        df (pd.DataFrame): reordered dataframe
        color_list (list): list of the colors of the backgrounds
    '''
    if info["GEOMETRY"][0] == "hd" and info["VERSION"][0] == "hd_1x2x6_legacy":
        order = ["Po210InLAr",
            "CPA",
            "APA",
            "Ar42InLAr",
            "Neutron",
            "Rn222InLAr",
            "Kr85InLAr",
            "Ar39InLAr"]
    
    elif info["GEOMETRY"][0] == "hd" and info["VERSION"][0] == "hd_1x2x6":
        order = ["Rn222ChainGenInPDS",
            "K40GenInCPA",
            "U238ChainGenInCPA",
            "K42From42ArGenInCPA",
            "Rn222ChainPo218GenInCPA",
            "Rn222ChainPb214GenInCPA",
            "Rn222ChainBi214GenInCPA",
            "Rn222ChainPb210GenInCPA",
            "Rn222ChainFromBi210GenInCPA",
            "Rn220ChainFromPb212GenInCPA",
            "Co60GenInAPA",
            "U238ChainGenInAPA",
            "Th232ChainGenInAPA",
            "Ar42GenInLAr",
            "K42From42ArGenInLAr",
            "NeutronsInCavernwall",
            "Rn222ChainRn222GenInLAr",
            "Rn222ChainPo218GenInLAr",
            "Rn222ChainPb214GenInLAr",
            "Rn222ChainBi214GenInLAr",
            "Rn222ChainPb210GenInLAr",
            "Rn220ChainPb212GenInLAr",
            "Kr85GenInLAr",
            "Ar39GenInLAr",
            "GammasInCryostat",
            "GammasInCavern",
            "GammasInFoam",
            "GammasInCavernwall",
            ]
            
    elif info["GEOMETRY"][0] == "vd" and info["VERSION"][0] == "vd_1x8x14_3view_30deg":
        order = ["CPAU238",
            "CPAK42-Ar42",
            "CPAK40",
            "Ar42",
            "K42-Ar42",
            "Neutron",
            "PDSRn222",
            "Rn222",
            "Kr85",
            "Ar39",
            "Gamma"]

    else: order = list(bkg_dict.values())[2:]

    df = df[order]
    color_list = []
    for bkg in order:
        color_list.append(color_dict[list(bkg_dict.values()).index(bkg)])
    
    if debug: print_colored("Reordered dataframe with columns: %s"%order,"INFO")
    return df,color_list

def get_simple_name(name_list, debug=False):
    simple_name = dict()
    basic_names = ["Ar42","Ar39","Kr85","Po210","Rn22"]
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
        elif "APA" in name:
            simple_name[name] = "APA"
        elif "PDS" in name:
            simple_name[name] = "PDS"
        else:
            simple_name[name] = name

    if debug: print_colored("Loaded simple name dictionary: %s"%str(simple_name),"INFO")
    return simple_name

def get_gen_weights(configs, names, debug=False):
    weights_dict = dict()
    for idx,config in enumerate(configs):
        info = read_input_file(config+'/'+configs[config],debug=debug)
        # Write a function that returns a dictionary of background names according to the input file. Each key of the dictionary should be a tuple of the form (geometry,version) and each value should be a list of background names.
        geo = info["GEOMETRY"][0]
        name_list = names[config]
        geo_weights_dict = get_bkg_weights(info,name_list)
        for idx,name in enumerate(name_list):
            weights_dict[(geo,name)] = geo_weights_dict[name]
    return weights_dict

def get_bkg_weights(info, names, debug = False):
    bkg_dict,color_dict = get_bkg_config(info,debug=False)
    weights_dict = dict()
    for bkg in bkg_dict.values():
        weights_dict[bkg] = 1
    if "wbkg" in names:
        weights_dict["wbkg"] = 1
        return weights_dict
    else:
        if info["GEOMETRY"][0] == "hd" and info["VERSION"][0] == "hd_1x2x6":
            custom_weights = {"NeutronsInCavernwall":1e3}
        if info["GEOMETRY"][0] == "hd" and info["VERSION"][0] == "hd_1x2x6_legacy":
            custom_weights = {"Po210":1e4,"APA":1e4,"CPA":1e2,"Ar42":1e4,"Neutron":1e2,"Rn222":1e4}
        if info["GEOMETRY"][0] == "vd" and info["VERSION"][0] == "vd_1x8x14_3view_30deg":
            custom_weights = {"Neutron":1e2}
        
        for bkg in custom_weights:
            weights_dict[bkg] = custom_weights[bkg]
        return weights_dict