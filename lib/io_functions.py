import uproot, os, copy, stat, inquirer
import numpy   as np
import pandas  as pd
import awkward as ak

def print_colored(string, color, bold=False):
    '''
    Print a string in a specific color.
    VARIABLES:
        \n - string: string to print
        \n - color: color of the string (see colors dictionary)
        \n - bold: if True, the string is printed in bold (default: False)
    '''
    colors = {
              "DEBUG":   '\033[35m', #PURPLE
              "ERROR":   '\033[91m', #RED
              "SUCCESS": '\033[92m', #GREEN
              "WARNING": '\033[93m', #YELLOW
              "INFO":    '\033[94m', #BLUE
              "magenta": '\033[95m',
              "cyan":    '\033[96m',
              "white":   '\033[97m',
              "black":   '\033[98m',
              "end":     '\033[0m'
              }
    
    if bold == False: output = colors[color] + string + colors["end"]
    if bold == True:  output = '\033[1m' + colors[color] + string + colors["end"]
    print(output)
    return output
    
def read_input_file(input, path="../config/", INTEGERS=[], DOUBLES=[], STRINGS=[], BOOLS=[], debug=False):
    '''
    Obtain the information stored in a .txt input file to load the runs and channels needed.
    **VARIABLES:**
    \n** - input:**    name of the input file.
    \n** - path:**     path of the input file.
    \n** - INTEGERS:** list of integers to read from the input file.
    \n** - DOUBLES:**  list of doubles to read from the input file.
    \n** - STRINGS:**  list of strings to read from the input file.
    \n** - BOOLS:**    list of booleans to read from the input file.
    \n** - debug:**    if True, the debug mode is activated.
    '''
    info = dict()
    if INTEGERS == []: INTEGERS = ["EVENT_TICKS","MAX_ADJCL_TIME","DETECTOR_SIZE_X","DETECTOR_SIZE_Y","DETECTOR_SIZE_Z","MAIN_PDG"]
    if DOUBLES  == []: DOUBLES  = ["FULL_DETECTOR_FACTOR","TIMEWINDOW","MAX_ADJCL_R","MAX_FLASH_R","MIN_FLASH_PE","RATIO_FLASH_PEvsR","MAX_DRIFT_FACTOR","MIN_BKG_R","MAX_BKG_CHARGE","MAX_ENERGY_PER_HIT","MIN_ENERGY_PER_HIT","FIDUTIAL_FACTOR","MAX_MAIN_E","MIN_MAIN_E","MAX_CL_E","MIN_CL_E","MAX_ADJCL_E","MIN_ADJCL_E"]
    if STRINGS  == []: STRINGS  = ["NAME","PATH","GEOMETRY","VERSION","DEFAULT_ANALYSIS_ENERGY","DEFAULT_RECOX_TIME","DEFAULT_ENERGY_TIME","DEFAULT_ADJCL_ENERGY_TIME","DEFAULT_CHARGE_VARIABLE"]
    if BOOLS    == []: BOOLS    = ["COMPUTE_MATCHING"]
    
    with open(path+input+".txt", 'r') as txt_file:
        lines = txt_file.readlines()

        for line in lines:
            for LABEL in DOUBLES:
                if line.startswith(LABEL):
                    info[LABEL] = []
                    doubles = line.split(" ")[1]
                    for i in doubles.split(","):
                        info[LABEL].append(float(i))
                        if debug: print_colored(string="Found %s: "%LABEL+str(info[LABEL]), color="DEBUG")
            for LABEL in INTEGERS:
                if line.startswith(LABEL):
                    info[LABEL] = []
                    integers = line.split(" ")[1]
                    for i in integers.split(","):
                        info[LABEL].append(int(i))
                        if debug: print_colored(string="Found %s: "%LABEL+str(info[LABEL]), color="DEBUG")
            for LABEL in STRINGS:
                if line.startswith(LABEL):
                    info[LABEL] = []
                    strings = line.split(" ")[1]
                    for i in strings.split(","):
                        info[LABEL].append(i.strip('\n'))
                        if debug: print_colored(string="Found %s: "%LABEL+str(info[LABEL]), color="DEBUG")
            for LABEL in BOOLS:
                if line.startswith(LABEL):
                    info[LABEL] = []
                    bools = line.split(" ")[1]
                    for i in bools.split(","):
                        if i.strip('\n') == "True":  info[LABEL].append(True)
                        if i.strip('\n') == "False": info[LABEL].append(False)
                        if debug: print_colored(string="Found %s: "%LABEL+str(info[LABEL]), color="DEBUG")

    if debug: print_colored("InputFile Info:"+str(info.keys()),"SUCCESS")
    return info

def root2npy(root_info, trim=False, debug=False):
    '''
    Dumper from .root format to npy files. Input are root input file, path and npy outputfile as strings.
    \n Depends on uproot, awkward and numpy.
    \n Size increases x2 times. 
    VARIABLES:
    \n root_info: output dictionary from get_root_info function.
    '''

    path = root_info["Path"]
    name = root_info["Name"]
    print(root_info["Path"]+root_info["Name"]+".root")

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

                if "Map" not in branch: 
                    this_array    = f[root_info["Folder"]+"/"+tree][branch].array()
                    if trim != False and debug: print("Selected trimming value: ", trim)
                    resized_array = resize_subarrays(this_array, 0, trim=trim, debug=debug)
                    np.save(path+name+"/"+out_folder+"/"+branch+".npy", resized_array)
                    if debug: print(resized_array)
                    del resized_array
                    del this_array

                else: 
                    print("Branch is a map: ", branch)
                    dicts = []
                    this_keys   = f[root_info["Folder"]+"/"+tree][branch+"_keys"].array()
                    this_values = f[root_info["Folder"]+"/"+tree][branch+"_values"].array()
                    if debug: print_colored("Using keys: " + str(this_keys[0]),"DEBUG"); print_colored("Using values: " + str(this_values[0]),"DEBUG")
                    for j in range(len(this_keys)): dicts.append(dict(zip(this_keys[j], this_values[j])))
                    np.save(path+name+"/"+out_folder+"/"+branch+".npy", dicts)
                    print(dicts[0])
                    del dicts

                if debug: 
                    print_colored("\nSaved data in:" + str(path+name+"/"+out_folder),"SUCCESS")
                    print_colored("----------------------\n","SUCCESS")
                    
def resize_subarrays(array, value, trim=False, debug=False):
    '''
    Resize the arrays so that the have the same lenght and numpy can handle them.
    The arrays with len < max_len are filled with 0 until they have max_len
    VARIABLES:
        \n - array: array to resize
        \n - branch: name of the branch
        \n - value: value to fill the array
        \n - trim: if True, trim the array to the selected size (default: False)
    '''    
    array = check_array_type(array,debug=debug)
    check_expand = False
    
    try:
        # Create a good check to determine if array needs to be expanded
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
        
        if expand:
            # if debug: print_colored("Expanding array to %i"%max_len,"DEBUG")
            tot_array = resize_subarrays_fixed(array, value, max_len, debug=debug)
        else:
            tot_array = array
    
    else:
        tot_array = np.asarray(array)

    if debug: print_colored("-> Returning array as type: %s"%(type(tot_array)),"SUCCESS")
    return np.asarray(tot_array)

def resize_subarrays_fixed(array, value, max_len, debug=False):
    '''
    Resize the arrays so that the have the same lenght and numpy can handle them.
    The arrays with len < size are filled with 0 until they have selected size
    VARIABLES:
        \n - array: array to resize
        \n - value: value to fill the array
        \n - size: size of the array
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

def check_array_type(array, debug=False):
    '''
    Check if the array is a list of lists or a list of arrays
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

#===========================================================================#
#************************* KEYS/BRANCHES ***********************************#
#===========================================================================#

def get_tree_info(root_file, debug=False):
    '''
    From a root file (root_file = uproot.open(path+name+".root")) you get the two lists with:
    VARIABLES
        \n - directory: ["solarnuana", "myanalysis;1"] with as many directories as you root file has
        \n - tree: ["Pandora_Outpur;1", "MCTruthTree;1", "SolarNuAnaTree;1"] with as many trees as you root file has
    '''

    directory = [i for i in root_file.classnames() if root_file.classnames()[i]=="TDirectory"]
    tree      = [i.split("/")[1] for i in root_file.classnames() if root_file.classnames()[i]=="TTree"]
    if debug:
        print_colored("The input root file has a TDirectory: " + str(directory),  color="DEBUG")
        print_colored("The input root file has %i TTrees: "%len(tree) +str(tree), color="DEBUG")

    return directory,tree

def get_root_info(name, path, debug=False):
    '''
    Function which returns a dictionary with the following structure:
    \n {"Path": path, "Name": name, "Folder": folder (from get_tree_info), "TreeNames": {"RootName_Tree1":YourName_Tree1, "RootName_Tree2":YourName_Tree2}, "RootName_Tree1":[BRANCH_LIST], "RootName_Tree2":[BRANCH_LIST]}
    VARIABLES:
        \n name: name of the root file
        \n path: path of the root file
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

def get_branches(name, path, debug=False):
    '''
    Function which returns a dictionary with the following structure:
    \n {"YourName_Tree1":[BRANCH_LIST], "YourName_Tree2":[BRANCH_LIST]}
    **VARIABLES:**
    \n **name:** name of the root file.
    \n **path:** path of the root file.
    '''

    branch_dict = dict()
    tree_info   = np.load(path+name+"/"+"TTrees.npy", allow_pickle=True).item()
    
    for tree in tree_info["TreeNames"].keys():
        branch_dict[tree_info["TreeNames"][tree]] = tree_info[tree]

    if debug: print(branch_dict)

    return branch_dict

def get_branches2use(run, debug=False):
    '''
    Function to get the branches of the TTree not in ['Name', 'Path', 'Labels', 'Colors']
    **VARIABLES:**
    \n** - run:**   Dictionary with the data to save (delete the keys that you don't want to save or set Force = False to save only the new ones).
    \n** - debug:** If True, the debug mode is activated.
    '''
    branches_raw = list(run.keys())
    branches     = [i for i in branches_raw if i not in ['Name', 'Path', 'Labels', 'Colors']]
    if debug: print_colored("\nFounded keys " + str(branches) + " to construct the dictionaries.", "DEBUG")

    return branches

def check_key(my_dict,key):
    '''
    Check if a given dict contains a key and return True or False
    '''
    try: my_dict[key]; return True    
    except KeyError:   return False

def delete_keys(run,keys):
    '''
    Delete the keys list introduced as 2nd variable
    '''
    for key in keys: del run[key]

def remove_processed_branches(root_info,debug=False):    
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

#===========================================================================#
#************************** LOAD/SAVE NPY **********************************#
#===========================================================================# 

def get_preset_list(name,path="../data",preset="ALL",debug=False):
    '''
    Return as output presets lists for load npy files.
    VARIABLES:
        \n - name: name of the file
        \n - path: loading path
        \n - preset: 
            (a) "ALL": all the existing keys/branches (default)
            (b) "ROOT": the ROOT branches (i.e the default branches given in the root file)
            (c) "NON_ROOT": the NEW branches (i.e the ones that are not in the root file)
    '''

    root_keys = get_branches(name, path, debug=False)
    npy_keys = dict()
    
    for tree in root_keys.keys(): # Truth, Reco, etc
        key_list = os.listdir(path+"/"+name+"/"+tree)

        if preset == "ALL": # Save/Load all keys for each branch
            remove = ["Branches.npy"]; aux = []
            for key in key_list:
                if key not in remove: 
                    aux.append(key.replace(".npy",""))
            key_list = aux 
        # Save Raw branches (i.e the default branches given in the root file)
        elif preset == "ROOT": key_list  = root_keys[tree] # i.e EventID, ADC, mcEnergy, etc

        elif preset == "NON_ROOT":  # Save NEW branches (i.e the ones that are not in the root file)
            remove = root_keys[tree]; aux=[]
            for key in key_list:
                if key not in remove: aux.append(key)
            key_list = aux

        else:     print_colored("Preset not found!","ERROR")
        if debug: print_colored("\nPreset key_list:" + str(key_list), "DEBUG")
        npy_keys[tree] = key_list

    return npy_keys

def load_multi(names,configs,load_all=False,preset="",branches={},generator_swap=False,debug=False):
    '''
    Load multiple files with different configurations and merge them into a single dictionary
    VARIABLES:
        \n - names: dict of lists with names of the files
        \n - configs: dict of configurations (i.e {geo1:"config1",geo2:"config2"})
        \n - load_all: if True, load all the branches and keys (default: True)
        \n - preset:
            (a) "ALL": all the existing keys/branches (default)
            (b) "ROOT": the ROOT branches
            (c) "NON_ROOT": the NEW branches
        \n - branches: dictionary of branches to load (i.e {"Tree1":["Branch1","Branch2"], "Tree2":["Branch1","Branch2"]})
    '''
    run = dict()
    for idx,config in enumerate(configs):
        if debug: print_colored("\nLoading config %s"%configs[config],"DEBUG")
        info = read_input_file(configs[config],path="../config/"+config+"/",debug=False)
        path = info["PATH"][0]+info["NAME"][0]
        geo  = info["GEOMETRY"][0]
        vers = info["VERSION"][0]
        bkg_dict,color_dict = get_bkg_config(info)
        inv_bkg_dict = {v: k for k, v in bkg_dict.items()}

        for jdx,name in enumerate(names[config]):
            if debug: print_colored("\nLoading file %s%s"%(path,name),"DEBUG")
            if load_all == True: branches_dict = get_branches(name,path=path,debug=debug)                   # Get ALL the branches
            elif preset == "":   branches_dict = branches                                                   # Get CUSTOMIZED branches from the input
            elif preset != "":   branches_dict = get_preset_list(name,path=path,preset=preset,debug=debug)  # Get PRESET branches

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
    VARIABLES:
         \n - run: dictionary with the data to save (delete the keys that you don't want to save or set Force = False to save only the new ones)
         \n - info: 
         \n - force: if True, overwrite the existing files
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
    print("Saved data in: ",path+branch)


#===========================================================================#
#*************************** DATA FRAMES ***********************************#
#===========================================================================# 

def npy2df(run,tree,branches=[],debug=False):
    '''
    Function to convert the dictionary of the TTree into a pandas Dataframe.
    VARIABLES:
        \n - run: dictionary with the data to save (delete the keys that you don't want to save or set Force = False to save only the new ones)
        \n - tree: name of the tree to convert to dataframe
        \n - branches: list of branches to convert to dataframe (default: [])
    '''
    reco_df = pd.DataFrame()
    if branches == []: branches = run[tree].keys()
    for branch in branches:
        if debug: print("Evaluated branch: ",branch)
        try:
            reco_df[branch] = run[tree][branch].tolist()
        except AttributeError:
            try: reco_df[branch] = run[tree][branch]
            except ValueError:
                print("ValueError: ",branch)
                continue
    # if debug: display(reco_df.info())
    if debug: print(reco_df.info())

    return reco_df

def dict2df(run, debug=False):
    '''
    Function to convert the dictionary of the TTree into a list of pandas Dataframes of len = len(branches)
    i.e. df_list = [df_truth, df_reco, ...]
    '''

    branches = get_branches2use(run,debug=debug) # Load the branches of the TTree not in ['Name', 'Path', 'Labels', 'Colors']
    df_list = []
    for branch in branches:
        df = pd.DataFrame()
        for key in run[branch].keys():
            try: df[key] = run[branch][key].tolist()
            except AttributeError: df[key] = run[branch][key]
            if debug: print_colored(" --- Dataframe for key %s created"%key, "DEBUG"); print("\n"); print_colored(df[key],"DEBUG"); print("\n")
        df_list.append(df)

        if debug: print_colored(" --- Dataframe for branch %s created"%branch, "DEBUG")

    return df_list

def merge_df(df1, df2, label1, label2, debug=False):
    '''
    Function to merge two dataframes in one adding an extra column to indicate its origin. 
    Also maintain the columns that are not in both df an include NaNs in the missing columns.
    VARIABLES:
        \n - df1: first dataframe
        \n - df2: second dataframe
        \n - label1: label of the first dataframe
        \n - label2: label of the second dataframe
    '''
    
    df1["Label"] = label1 # Add a column to indicate the origin of the event
    df2["Label"] = label2 # Add a column to indicate the origin of the event
    df = pd.concat([df1,df2], ignore_index=True) # Merge the two dataframes
    if debug: print_colored(" --- New dataframe from %s, %s created"%(label1,label2), "DEBUG")
    
    return df

# def add_labels(run, name, path, labels, label):
#     label_branches = []
#     try: 
#         os.mkdir(path+name+"/Label")
#     except FileExistsError:
#         print("Label") 
    
#     for l in range(len(label)):
#         np.save(path+name+"/Label/"+label[l]+".npy",np.asarray(labels[l]))
#         label_branches.append(label[l])
#     np.save(path+name+"/Label/Branches.npy",np.asarray(label_branches))
    
#     print("Adding labels -> Done!")
#     return run