import sys; sys.path.insert(0, '../'); from lib import *

# Check that waht I am getting from the root has sense 
# Clarify workflow


# ======================================================================================================== #
# ---------------------------------------- FORMATTING FUNCTIONS ------------------------------------------ #
# ======================================================================================================== #

def event_df(data_frame,event):
    '''
    This function creates a dataframe with the information of one event with the same structure as the original dataframe.
    VARIABLES:
        \n - data_frame:    dataframe with the information of the events [type: pandas dataframe]
        \n - event:         event number [type: int]
    '''

    columns = list(data_frame.columns)
    df = pd.DataFrame()
    for i, data_list in enumerate(np.array(data_frame.loc[event])): df[columns[i]] = [data_list]

    return df

def filter_array(df_evt,column,filter_value):
    '''
    This function filters the values of a column of a dataframe.
    VARIABLES:
        \n - data_frame:    df with info of one event (expected output of event_df). [type: pandas dataframe]
        \n - column:        column name                                              [type: string]
        \n - filter_value:  value to filter                                          [type: int]
    '''

    filtered = np.array(df_evt[column][0])[np.array(df_evt[column][0]) != filter_value]
    
    return filtered

def df2array(data_frame,column,filter_value=""):
    '''
    This function transforms a dataframe column into an array of arrays, where each sub-array are the values of the column for each event.
    VARIABLES:
        \n - data_frame:    dataframe with the information of the events    [type: pandas dataframe]
        \n - column:        column name                                     [type: string]
    '''
    
    filtered = []

    # Array of arrays with shape (n_events,n_particles)
    my_array = np.concatenate(data_frame[column][:].to_numpy()).reshape(len(data_frame[column]),len(data_frame[column][0])) 
    for evt in range(len(data_frame[column])):
        filtered = np.append(filtered,np.array(data_frame[column][evt])[np.array(data_frame[column][evt]) != filter_value])
    
    if filter_value == "": return my_array
    if filter_value != "": return np.array(filtered, dtype=object)


# ======================================================================================================== #
# ---------------------------------------- COMPUTATION FUNCTIONS ----------------------------------------- #
# ======================================================================================================== #

def track2idx(run, OPT,mode="idx2track",nan_value=None,debug=False):
    '''
    Function to "translate" from TrackID to index so you can use the information of the TrackID without missleading the indexes of your arrays.
    VARIABLES:
        \n - mode: 
            \n * "idx2track" --> dict_keys = index   and dict_values = trackID [type = str] DEFAULT
            \n * "track2idx" --> dict_keys = trackID and dict_values = index   [type = str]
    '''

    # Check if the OPT dictionary has the required keys
    if check_key(OPT,"EVENTID") and check_key(OPT,"TRACKID"): Event_label = OPT["EVENTID"]; TrackID = OPT["TRACKID"]
    else: print_colored(string="ERROR: Missing keys in OPT dict. Please check the documentation.",color="ERROR"); return

    branches = get_branches2use(run,debug=debug) # Load the branches of the TTree not in ['Name', 'Path', 'Labels', 'Colors']
    if debug: print_colored(string="Working with mode: "+str(mode),color="DEBUG")
    dict_track2idx= dict.fromkeys(branches)      # Different PDGCodes for the MCParticles for each event stored i.e run[branch][evt#] = [-14,-13,11,22] but not conserving the ammount of each one
    for branch in branches:
        track2idx = [[] for _ in range(len(run[branch][Event_label]))]
        for evt in range(len(run[branch][Event_label])):
            index = []; track = []
            for nt,tr in enumerate(run[branch][TrackID][evt][run[branch][TrackID][evt]!=nan_value]):
                if mode == "idx2track": index.append(nt); track.append(tr)
                if mode == "track2idx": index.append(tr); track.append(nt)
            
            track2idx[evt] = dict(zip(index,track))
        dict_track2idx[branch] = track2idx

    return dict_track2idx

def pdg_info(run,OPT,branches=[""],nan_value=None,debug=False):
    '''
    Once we load the runs we can obtain information of the particles in each event from the PdgCode_label branch information.
    VARIABLES:
        \n - run:       as loaded from load_npy               [type = dict]
        \n - OPT:       dictionary with the branches to use   [type = dict]
        \n - branches:  list of branches to use               [type = list]
        \n - debug:     print debug messages                  [type = bool]
    '''
    
    # Check if the OPT dictionary has the required keys
    if check_key(OPT,"EVENTID") and check_key(OPT,"TRACKID") and check_key(OPT,"PDG_LAB"): TrackID = OPT["TRACKID"]; Event_label = OPT["EVENTID"]; PdgCode_label = OPT["PDG_LAB"]
    else: print_colored(string="ERROR: Missing keys in OPT dict. Please check the documentation.",color="ERROR"); return

    if branches == [""]: branches = get_branches2use(run,debug=debug) # Load the branches of the TTree not in ['Name', 'Path', 'Labels', 'Colors']
    dict_pdg_num = dict.fromkeys(branches) # Different PDGCodes for the MCParticles for each event stored i.e run[branch][evt#] = [-14,-13,11,22] but not conserving the ammount of each one
    dict_pdg_nam = dict.fromkeys(branches) # Different PDGNames for the MCParticles for each event stored i.e run[branch][evt#] = [nu_mu,mu+,e-,gamma]
    dict_pdg_all = dict.fromkeys(branches) # PDGNames for all the MCParticles for each event and branch stored in dictionary run[branch][evt#] = [nu_mu,mu+,e-,gamma, ...]
    for branch in branches:
        if debug: print_colored(string="\n---- Processing PDG NAMES ----",  color="DEBUG"); print_colored(string="Loading branch: "+ str(branch),     color="DEBUG"); print_colored(string="Loading key: " + str(PdgCode_label),color="DEBUG")
        pdg_num = [[] for _ in range(len(run[branch][Event_label]))]
        pdg_nam = [[] for _ in range(len(run[branch][Event_label]))]
        pdg_all = [{} for _ in range(len(run[branch][Event_label]))]
        for evt in range(len(run[branch][Event_label])):
            [pdg_num[evt].append(x) for x in run[str(branch)][PdgCode_label][evt] if x not in pdg_num[evt] and x !=nan_value and x != 0] # Append pdg codes to the list if not already in it
            try:    [pdg_nam[evt].append(r"$"+str(Particle.from_pdgid(i).latex_name)+"$") for i in pdg_num[evt]]                   # Append pdg names to the list if not already in it
            except: [pdg_nam[evt].append(r"$"+"NONE"+"$") for i in pdg_num[evt]]

            trackID = []; pdgnames = [];
            for nt,tr in enumerate(run[branch][TrackID][evt][run[branch][TrackID][evt]!=nan_value]):
                x = run[branch][PdgCode_label][evt][nt]
                if x != 0: trackID.append(tr); pdgnames.append(r"$"+str(Particle.from_pdgid(x).latex_name)+"$") # Append trackIDs and pdg names to their lists 
                else:      trackID.append("null_"+str(nt)); pdgnames.append(x)                                  # Append null_* and 0 (WARNING: dictionary keys must be unique)

            pdg_all[evt] = dict(zip(trackID, pdgnames)) #if keys are repeated they are replaced !!
        if debug:
            print_colored(string="\n---- Processed %i events ----"%(len(run[branch][Event_label])),color="DEBUG")
            print_colored(string="Example of PDG codes stored (evt#0): " + str(pdg_num[0]),          color="DEBUG")
            print_colored(string="Example of PDG names stored (evt#0): " + str(pdg_nam[0]),          color="DEBUG")
            print_colored(string="Founded %i particles for the Event 0: "%(len(pdg_all[0])),         color="DEBUG")
            print_colored(string="with the following (TrackID, PDG Names): " + str(pdg_all[0]) ,     color="DEBUG")

        dict_pdg_num[branch] = pdg_num
        dict_pdg_nam[branch] = pdg_nam
        dict_pdg_all[branch] = pdg_all
    
    return dict_pdg_num, dict_pdg_nam, dict_pdg_all

def energy_computation(run,OPT,branch,debug=False):
    '''
    This function computes the energy of the particles in each event from the energy_key branch information.
    VARIABLES:
        \n - run:       as loaded from load_npy               [type = dict]
        \n - OPT:       dictionary with the branches to use   [type = dict]
        \n - branch:    name of the branch to be analyzed     [type = string]
        \n - debug:     if True, prints debug messages        [type = bool]        
    '''
    # Check if the OPT dictionary has the required keys
    if check_key(OPT,"EVENTID")  and check_key(OPT,"PDG_LAB") and check_key(OPT,"ENERGY"):
        Event_label = OPT["EVENTID"]; PdgCode_label = OPT["PDG_LAB"]; energy_label = OPT["ENERGY"]
    else: print_colored(string="ERROR: Missing keys in OPT dict. Please check the documentation.",color="ERROR"); return
   
    if debug: print_colored(string="Using the key " + str(energy_label) + "to compute the energy of the particles", color="DEBUG")
    PDG = pdg_info(run,OPT=OPT,debug=debug)
    energy_true = [[] for _ in range(len(run[branch][Event_label]))]
    for evt in range(len(run[branch][Event_label])):
        dict_energy_evt = dict()
        for mc, pdg_mc in enumerate(PDG[0][branch][evt]): dict_energy_evt[PDG[1][branch][evt][mc]] = run[branch][energy_label][evt][run[branch][PdgCode_label][evt] == pdg_mc]
        energy_true[evt] = dict_energy_evt
    if debug: print_colored(string="Completed computation of the energy", color="DEBUG")

    return energy_true

def energy2plot(df_evt,lenght,OPT,nan_value=None,debug=False):
    '''
    This function generates an array of energies for each event to use it in the plots to define the size of the markers.
    VARIABLES:
        \n - df_evt:    df of the event to be analyzed        [type = pandas dataframe]
        \n - OPT:       dictionary with the branches to use   [type = dict]
        \n - debug:     if True, prints debug messages        [type = bool]        
    '''

    # Check if the OPT dictionary has the required keys
    if check_key(OPT,"ENERGY"): energy_label = OPT["ENERGY"]
    else: E = np.ones(lenght); print_colored(string="ERROR: Missing keys in OPT dict. RETURNING ARRAY OF NP.ONES(LENGHT)",color="ERROR")

    if debug: print_colored(string="Using the key " + str(energy_label) + "to compute the energy of the particles", color="DEBUG")
    E_raw = np.array(df_evt[energy_label][0])[np.array(df_evt[energy_label][0]) != nan_value] # Get the energy of the particles in the event
    E = E_raw[:]/max(E_raw) # Normalize the energy to the maximum energy of the event
    E = np.append(1,E[1:]) # The primary particle is always the first one in the list, so we set the energy of the primary particle to 1
    #igual esto es mejor generalizando en la proxima version
    if debug: print_colored(string="Completed computation of the energy", color="DEBUG")
    if len(E) != lenght: print_colored(string="ERROR: The lenght of the energy array is not the same as the lenght of the particle array", color="ERROR"); return

    a = (E[0]/E[1])/10; E = int(a) * E

    return list(E)

def energy_conservation(my_df,evt,OPT):
    '''
    Function to compute the energy before and after the vertex to check it is the same and it is conserved.
    VARIABLES:
        \n - my_df:     dataframe to be analyzed            [type = pandas dataframe]
        \n - evt:       event to be analyzed or "ALL"       [type = int/str]
        \n - OPT:       dictionary with the keys to use     [type = dict]
    '''

    # Check if the OPT dictionary has the required keys
    if check_key(OPT,"EVENTID")  and check_key(OPT,"PDG_LAB") and check_key(OPT,"ENERGY"):
        Event_label = OPT["EVENTID"]; PdgCode_label = OPT["PDG_LAB"]; energy_label = OPT["ENERGY"]
    else: print_colored(string="ERROR: Missing keys in OPT dict. Please check the documentation.",color="ERROR"); return

    # Check if the dataframe has the Prim_idx column (to check the primary particle of the event)
    if check_key(my_df,"VertexPos") and check_key(my_df,"Energies"): vertex_info = my_df["VertexPos"]
    else: print_colored("WARNING: Missing keys in given dataframe. Please check mc_info.py output.","WARNING"); return
    if evt != "ALL": my_df = event_df(my_df,evt); events = [evt]
    else: events = range(len(my_df["VertexPos"]))
    
    energies = np.array(my_df[energy_label])
    energy_conservation = [{} for _ in events]
    for e,event in enumerate(events):
        vertex_idx = list(vertex_info[e].keys())
        energy_conservation[e]["Pre"] = np.nansum(np.array(energies[e][1:vertex_idx[0]],dtype=float))
        energy_conservation[e]["Pos"] = np.nansum(np.array(energies[e][vertex_idx[0]:], dtype=float))
        print_colored("-- VERTEX INFO (Event %i) -- Energy before (wout primary): %.2f (GeV); Energy after: %.2f (GeV)" %(event,energy_conservation[e]["Pre"],energy_conservation[e]["Pos"]),"blue")

    return energy_conservation

def primary_computation(run,OPT,branch,debug=False):
    '''
    This function computes the primary particle for each event in the run.
    VARIABLES:
        \n - run:        dictionary containing the data of the run  [type = dict]
        \n - OPT:        dictionary with the branches to use        [type = dict]
        \n - branch:     branch of the run to be processed          [type = string]
        \n - debug:      if True, prints the debug messages         [type = bool]
    '''

    # Check if the OPT dictionary has the required keys
    if check_key(OPT,"EVENTID")  and check_key(OPT,"PDG_LAB") and check_key(OPT,"PRIMARY"):
        Event_label = OPT["EVENTID"]; PdgCode_label = OPT["PDG_LAB"]; primary_label = OPT["PRIMARY"]
    else: print_colored(string="ERROR: Missing keys in OPT dict. Please check the documentation.",color="ERROR"); return

    if debug: print_colored(string="Using the key " + str(primary_label) + "to compute the primary particle", color="DEBUG")
    primary_idx = []
    for evt in range(len(run[branch][Event_label])): primary_idx.append(run[branch][PdgCode_label][evt][np.where(run[branch][primary_label][evt] == 1)][0])
    if debug: print_colored(string="Completed computation of the primary particle", color="DEBUG")
    
    return primary_idx

def find_eva(run,branch,event,trackID,ancestry_key,nan_value=None,debug=False):
    '''
    This function returns the trackID of the primary particle that originated the trackID given as input.
    VARIABLES:
        \n - run:            dictionary containing the data of the run  [type = dict]
        \n - branch:         branch of the run to be processed          [type = string]
        \n - event:          event to be processed                      [type = int]
        \n - trackID:        trackID of the particle to be processed    [type = int]
        \n - ancestry_key:   key to access the ancestry information     [type = string]
        \n - debug:          if True, prints the debug messages         [type = bool]
    '''

    idxs = track2idx(run, mode="track2idx")
    idx  = idxs[branch][event][trackID]
    if debug: print_colored(string="ArrayIndex: "+str(idx),color="DEBUG")

    Mother_trackID = run[branch][ancestry_key][event][idx]
    if trackID == 1 or trackID == nan_value: counter = 0
    else:
        counter = 1
        if debug: print_colored(string="("+str(counter)+") "+"Mother_trackID:",color="DEBUG")
        if Mother_trackID != 1 and Mother_trackID != nan_value:
            while counter < 20:
                idx = idxs[branch][event][Mother_trackID]
                Mother_trackID = run[branch][ancestry_key][event][idx]
                if Mother_trackID != 1 and Mother_trackID != nan_value: counter = counter + 1
                if debug: print_colored(string="("+str(counter)+") "+"Mother_trackID:",color="DEBUG")
                if Mother_trackID == 0: break
                if counter > 10: break
        if Mother_trackID == nan_value: counter = 0
    if debug: print_colored(string="Counter: "+str(counter),color="DEBUG")

    return counter

### DEPRECATED ### BETTER/OPTIMIZED TO OBTAIN IT DIRECTLY IN THE ROOT FILE :)
def ancestry_computation(run, OPT, branch, events, nan_value=None,debug=False): #cambiar por ascestry
    '''
    This function returns the ancestry of the particles in the event given as input.
    VARIABLES:
        \n - run:        dictionary containing the data of the run  [type = dict]
        \n - OPT:        dictionary with the branches to use        [type = dict]
        \n - branch:     branch of the run to be processed          [type = string]
        \n - events:     event to be processed                      [type = int]
        \n - debug:      if True, prints the debug messages         [type = bool]
    '''

    # Check if the OPT dictionary has the required keys
    if check_key(OPT,"")  and check_key(OPT,"TRACKID") and check_key(OPT,"ANCESTRY"):
        Event_label = OPT["EVENTID"]; TrackID = OPT["TRACKID"]; ancestry_label = OPT["ANCESTRY"]
    else: print_colored(string="ERROR: Missing keys in OPT dict. Please check the documentation.",color="ERROR"); return

    if debug: print_colored(string="Using the key " + str(ancestry_label) + "to compute the ancestry of the particles", color="DEBUG")
    if events == "ALL": ancestry_level = [[] for _ in range(len(run[branch][Event_label]))]; events2compute = range(len(run[branch][Event_label][:]))
    if events != "ALL": ancestry_level = []; events2compute = [events]
    
    for evt in events2compute:
        if debug: print("\n ------ EVENT %i ------ "%evt)
        trackID = []; ancestry = []
        for trID in run[branch][TrackID][evt][run[branch][TrackID][evt]!=nan_value]:
            if debug: print_colored(string="\nTrackID:"+str(trID),color="DEBUG");
            counter = find_eva(run,branch,event=evt,TrackID=trID,ancestry_key=ancestry_label,debug=debug)
            trackID.append(trID); ancestry.append(counter)
        if events == "ALL": ancestry_level[evt] = dict(zip(trackID, ancestry))
        if events != "ALL": ancestry_level = dict(zip(trackID, ancestry))
    if debug: print_colored(string="Completed computation of the ancestry", color="DEBUG")
    
    return ancestry_level

def particles_info(run,OPT,branches=["Reco1"],compute=["PRIMARY","ENERGY","ANCESTRY"],debug=False):
    '''
    Dictionaries with useful information for the particles of each branch and event. The keys of the labels dictionary are used to call 
    the expecific functions to compute the parameters.
    VARIABLES:
        \n - run:               as loaded from load_npy                 [type = dict]
        \n - OPT:               dictionary with the branches to use     [type = dict]
            \n * EVENTID:       key to access the Event information   [type = string]
            \n * PDG_LAB:       key to access the PDGCode information   [type = string]
            \n * PRIMARY:       key to access the primary information   [type = string]
            \n * ENERGY:        key to access the energy information    [type = string]
            \n * ANCESTRY:      key to access the ancestry information  [type = string]
        \n - branches:          branches of the run to be processed     [type = list]
        \n - compute:           list of the parameters to be computed   [type = list]
                                \n i.e. ["PRIMARY","ENERGY","ANCESTRY"]
        \n - debug:             if True, prints the debug messages      [type = bool]
    '''

    # Check if the OPT dictionary has the required keys
    if check_key(OPT,"EVENTID")  and check_key(OPT,"PDG_LAB") and check_key(OPT,"PRIMARY") and check_key(OPT,"ENERGY") and check_key(OPT,"ANCESTRY"):
        Event_label = OPT["EVENTID"]; PdgCode_label = OPT["PDG_LAB"]; primary_label = OPT["PRIMARY"]; energies_label = OPT["ENERGY"]; ancestry_label = OPT["ANCESTRY"]
    else: print_colored(string="ERROR: Missing keys in OPT dict. Please check the documentation.",color="ERROR"); return

    if "PRIMARY"  in compute: dict_primary_idx = dict.fromkeys(branches)
    if "ENERGY"   in compute: dict_energy_true = dict.fromkeys(branches) 
    if "ANCESTRY" in compute: dict_ancestry_mc = dict.fromkeys(branches)
    output = []

    if debug: print_colored(string="\n---- Processing information for Particles in all the Events ----", color="DEBUG")
    for i, branch in enumerate(branches):

        if "PRIMARY" in compute:
            primary_idx              = primary_computation(run, OPT, branch, debug=debug)
            dict_primary_idx[branch] = primary_idx
            output.append(dict_primary_idx)

        if "ENERGY" in compute:
            energy_true              = energy_computation(run, OPT, branch, debug=debug)
            dict_energy_true[branch] = energy_true
            output.append(dict_energy_true)

        if "ANCESTRY" in compute:
            ancestry_true            = ancestry_computation(run, OPT, branch, events="ALL", debug=debug)
            dict_ancestry_mc[branch] = ancestry_true
            output.append(dict_ancestry_mc)

    print_colored("Completed: information for the particles in your events. len(output) = "+str(len(output)),"SUCCESS")
    
    return output

def how_many(run,OPT,branches=[""],debug=False):
    '''
    Compute the number of particles of a given event.
    VARIABLES:
        \n - run:           as loaded from load_npy                 [type = dict].
        \n - OPT:           dictionary with the branches to use     [type = dict]
            \n * EVENTID:   Name of the branch with the event ID.
            \n * PDG_LAB:   Name of the branch with the PDG code of the particles.
    '''

    # Check if the OPT dictionary has the required keys
    if check_key(OPT,"EVENTID") and check_key(OPT,"PDG_LAB"): Event_label = OPT["EVENTID"]; pdg_label = OPT["PDG_LAB"]
    else: print_colored(string="ERROR: Missing keys in OPT dict. Please check the documentation.",color="ERROR"); return

    if branches == [""]: branches = get_branches2use(run,debug=debug) # Load the branches of the TTree not in ['Name', 'Path', 'Labels', 'Colors']
    dict_how_many = dict.fromkeys(branches)

    for branch in branches:    
        if check_key(run[branch],"PdgNum") and check_key(run[branch],"PdgNam") and check_key(run[branch],"PdgAll"):
            pdg_num = run[branch]["PdgNum"]; pdg_nam = run[branch]["PdgNam"]; pdg_all = run[branch]["PdgAll"]
        else: print_colored(string="WARNING: Missing keys in given dataframe. Please check mc_info.py output.",color="WARNING"); return

        counts = [[] for _ in range(len(run[branch][Event_label]))]
        for evt in range(len(run[branch][Event_label])):
            for mc, pdg_mc in enumerate(pdg_num[evt]):
                counts[evt].append(np.count_nonzero(run[branch][pdg_label][evt] == pdg_mc)) 
                if debug: print_colored(string="Event %d: %d %s particle(s)"%(evt,counts[evt][mc],pdg_nam[evt][mc]),color="DEBUG")
        dict_how_many[branch] = counts
    
    return dict_how_many

def particle_info_summed(my_df,keys_label,values_label,debug=False):
    '''
    Compute the number of particles for all the events. 
    (i.e Returns a dictionary with keys: "PdgName" and values: summed number of particles when you want to group (energies) by particles).
    VARIABLES:
        \n - my_df:         dataframe with the information of the events        [type = pandas dataframe]
        \n - keys_labels:   list of the keys of the dictionary to be returned   [type = list]
        \n - values_labels: list of the values of the dictionary to be summed [type = list]
    '''

    # Check keys in dataframe 
    if check_key(my_df,keys_label) and check_key(my_df,values_label): keys =  my_df[keys_label]; values = my_df[values_label]
    else: print_colored(string="WARNING: Missing keys in given dataframe. Please check mc_info.py output.",color="WARNING"); return
    
    keys_list = []; summed = {}
    for i in range(len(my_df[keys_label])):
        for key in keys.iloc[i]:
            if key not in keys_list: keys_list.append(key)

    for key in keys_list:
        summed[key] = 0
        for i in range(len(my_df[keys_label])):
            for j in range(len(keys.iloc[i])):
                if keys.iloc[i][j] == key: summed[key] += values.iloc[i][j]

    if debug: print_colored(string="-- Summed '%s' in dict: "%(values_label)+str(summed),color="DEBUG")

    return summed

def distance3D(x1,x2,y1,y2,z1,z2, debug=False):
    '''
    Compute the distance between two points in 3D space.
    VARIABLES:
        \n - xi: x coordinate of the ith point. [type = float]
        \n - yi: y coordinate of the ith point. [type = float]
        \n - zi: z coordinate of the ith point. [type = float]
    '''
    # Sanity check
    if x1 == None or x2 == None or y1 == None or y2 == None or z1 == None or z2 == None: return None
    else:
        x = pow((x2-x1),2); y = pow((y2-y1),2); z = pow((z2-z1),2)
        distance = pow((x+y+z),0.5)
        if debug: 
            print_colored(string="\nDistanceX = %0.2f with (X2=%0.2f,X1=%0.2f)"%(x,x2,x1),color="DEBUG")
            print_colored(string=  "DistanceY = %0.2f with (Y2=%0.2f,Y1=%0.2f)"%(y,y2,y1),color="DEBUG")
            print_colored(string=  "DistanceZ = %0.2f with (Z2=%0.2f,Z1=%0.2f)"%(z,z2,z1),color="DEBUG")

    return distance

def track_length(my_df,OPT,debug=False):
    '''
    Compute the track length of a given event returning the distance between the initial and end position of the primary particle.
    VARIABLES:
        \n - my_df:     DataFrame with the data to be processed. [type = pandas dataframe]
        \n - OPT:       dictionary with the branches to use      [type = dict]
            \n * BR_NAME: Name of the branch to be processed.
            \n * POS_INI: Name of the branch with the initial position of the particle.
            \n * POS_END: Name of the branch with the end position of the particle.
            \n * EVENTID: Name of the branch with the event ID.
    '''

    # Check if the OPT dictionary has the required keys
    if check_key(OPT,"BR_NAME") and check_key(OPT,"POS_INI") and check_key(OPT,"POS_END") and check_key(OPT,"EVENTID"): 
        branches_list = OPT["BR_NAME"]; pos_ini_label = OPT["POS_INI"]; pos_end_label = OPT["POS_END"]; Event_label = OPT["EVENTID"]
    else: print_colored(string="ERROR: Missing keys in OPT dict. Please check the documentation.",color="ERROR"); return

    dict_track = dict.fromkeys(branches_list)
    for branch in branches_list:
        if debug: print_colored(string="\n---- Processing TRACK LENGHT ----",color="DEBUG")
        track_len = []
        for evt in range(len(my_df[branch][Event_label])):
            distance = distance3D(x1=my_df[branch][pos_ini_label[0]][evt][0],x2=my_df[branch][pos_end_label[0]][evt][0],y1=my_df[branch][pos_ini_label[1]][evt][0],y2=my_df[branch][pos_end_label[1]][evt][0],z1=my_df[branch][pos_ini_label[2]][evt][0],z2=my_df[branch][pos_end_label[2]][evt][0])
            if debug: print_colored(string="Track lenght: %0.2f"%distance,color="DEBUG")
            track_len.append(distance)
        dict_track[branch] = track_len

    return dict_track

def vertex_reco(my_df,OPT,debug=True):
    '''
    Compute the particles of the vertex.
    VARIABLES:
        \n - my_df:     DataFrame with the data to be processed. [type = pandas dataframe]
        \n - OPT:       dictionary with the branches to use      [type = dict]
            \n * BR_NAME: Name of the branch to be processed.
            \n * POS_INI: Name of the branch with the initial position of the particle.
            \n * POS_END: Name of the branch with the end position of the particle.
            \n * EVENTID: Name of the branch with the event ID.
    '''

    # Check if the OPT dictionary has the required keys
    if check_key(OPT,"BR_NAME") and check_key(OPT,"POS_INI") and check_key(OPT,"POS_END") and check_key(OPT,"EVENTID"):
        branches_list = OPT["BR_NAME"]; pos_ini_label = OPT["POS_INI"]; pos_end_label = OPT["POS_END"]; Event_label = OPT["EVENTID"]
    else: print_colored(string="ERROR: Missing keys in OPT dict. Please check the documentation.",color="ERROR"); return

    # Check if the dataframe has the Prim_idx column (to check the primary particle of the event)
    if check_key(my_df,"TrackLen") and check_key(my_df,"PdgAll"): track_lengths = np.array(my_df["TrackLen"]) 
    else: print_colored("WARNING: Missing keys in given dataframe. Please check mc_info.py output.","WARNING"); return


    dict_vertex   = dict.fromkeys(branches_list); pos = ["before","at","after"]
    dict_dist2gen = dict.fromkeys(branches_list)
    for branch in branches_list:
        if debug: print_colored(string="\n---- Processing VERTEX RECO ----",color="DEBUG")
        vertex = [{} for _ in range(len(my_df[Event_label]))]; distances = [[] for _ in range(len(my_df[Event_label]))]
        for evt in range(len(my_df[Event_label])):
            df_evt = event_df(my_df, evt); df_evt["PdgAll"] = [list(df_evt["PdgAll"][0].values())]
            part_idx = []; part_name = [];
            for particle in range(len(my_df[pos_ini_label[0]][evt])):
                # Compute the distance between the initial of the particle and the inital of the primary particle
                distance = distance3D(x1=my_df[pos_ini_label[0]][evt][particle],x2=my_df[pos_ini_label[0]][evt][0],y1=my_df[pos_ini_label[1]][evt][particle],y2=my_df[pos_ini_label[1]][evt][0],z1=my_df[pos_ini_label[2]][evt][particle],z2=my_df[pos_ini_label[2]][evt][0])
                if distance != None:
                    [distances[evt].append(distance)]
                    if distance-track_lengths[evt] == 0: 
                        [part_idx.append(particle)]
                        [part_name.append(df_evt["PdgAll"][0][particle])]
                        if debug: print_colored(string="Event %i, particle %i in the vertex"%(evt,particle),color="DEBUG")
            vertex[evt] = dict(zip(part_idx, part_name)) #if keys are repeated they are replaced !!

        if debug: 
            evt = 0
            fig = px.line(x=np.arange(len(distances[evt])),y=distances[evt]-track_lengths[evt],title="Event %i"%evt)
            fig.add_traces(px.scatter(x=np.array(list(vertex[evt].keys())), y=np.zeros(len(list(vertex[evt].keys()))),title="Event %i"%evt,color=list(vertex[evt].values())).data)
            fig.update_layout( xaxis_title="Particle index", yaxis_title="Distance primary - Track Lenght [cm]",
                                font=dict(size=15),
                                legend = dict(font = dict(size = 30, color = "black"),orientation="h"),
                                legend_title_text='',
                                xaxis = dict( tickfont = dict(size=15)) )
            fig.show()

        dict_vertex[branch]   = vertex
        dict_dist2gen[branch] = distances

    return dict_vertex, dict_dist2gen


# ======================================================================================================== #
# --------------------------------------- VISUALIZATION FUNCTIONS ---------------------------------------- #
# ======================================================================================================== #

def vis_histogram(my_df,column,nbins=None,title="",xlabel="",ylabel="",html=False,save=False):
    '''
    Plot an histogram with plotly given a dataframe and a column.
    VARIABLES:
        \n - my_df:     dataframe with the information of the events    [type: pandas dataframe]
        \n - column:    column of the dataframe to plot                 [type: string]
        \n - nbins:     number of bins of the histogram                 [type: int]
        \n - title:     title of the histogram                          [type: string]
        \n - xlabel:    label of the x axis                             [type: string]
        \n - ylabel:    label of the y axis                             [type: string]
        \n - html:      if True, the plot is saved in an html file      [type: bool]
        \n - save:      if True, the plot is saved in "histogram.png"   [type: bool]
    '''

    fig_px = px.histogram(my_df, x=column, nbins=nbins)
    fig_px.update_layout( legend = dict(font = dict(size = 35, color = "black"),orientation="h"),
                          legend_title = dict(font = dict(size = 20, color = "black")) )
    
    if title  != "": fig_px.update_layout(title=title)
    if xlabel != "": fig_px.update_layout(xaxis_title=xlabel)
    if ylabel != "": fig_px.update_layout(yaxis_title=ylabel)
    if html: pyoff.plot(fig_px, include_mathjax='cdn')
    if save: fig_px.write_image("histogram.png")
    
    fig_px.show()

def vis_event_new(my_df,evt,OPT,pos_end=False,energy_size=False,nan_value=None,html=False,save=False):
    '''
    Visualize the event in 3D.
    VARIABLES:
        \n - my_df:         dataframe with the information of the events                [type: pandas dataframe]
        \n - evt:           event number                                                [type: int]
        \n - OPT:           dictionary with the branches to use                         [type = dict]
        \n - pos_end:       boolean to show the end position of the particles           [type: bool]
        \n - energy_size:   boolean to show the energy of the particles                 [type: bool]
        \n - counts:        boolean to compute the number of particles in the event     [type: bool]
        \n - html:          boolean to save the plot in html format                     [type: bool]
        \n - save:          boolean to save the plot in "vis_event.png"                 [type: bool]
    '''

    # Check if the OPT dictionary has the required keys
    if check_key(OPT,"BR_NAME") and check_key(OPT,"POS_INI") and check_key(OPT,"POS_END") and check_key(OPT,"EVENTID"): 
        Event_label = OPT["EVENTID"]; energy_label = OPT["ENERGY"]; Prim_label = OPT["PRIMARY"]
        x_i = OPT["POS_INI"][0]; y_i = OPT["POS_INI"][1]; z_i = OPT["POS_INI"][2]
        x_f = OPT["POS_END"][0]; y_f = OPT["POS_END"][1]; z_f = OPT["POS_END"][2]
    else: print_colored("ERROR: Missing arguments in OPT dict. Please check the documentation.","ERROR"); return
    
    if evt == "ALL":
        event_num = []; nums = np.linspace(1,len(np.array(my_df[Event_label])),len(np.array(my_df[Event_label])))
        for i in range(len(np.array(my_df[Event_label]))):
            # event_num = event_num + list((my_df[Event_label][i]*np.ones(len(np.array(my_df[x_i][i])[np.array(my_df[x_i][i])!=None]))))
            event_num = event_num + list((nums[i]*np.ones(len(np.array(my_df[x_i][i])[np.array(my_df[x_i][i])!=nan_value]))))

        X = df2array(my_df, x_i, filter_value=nan_value); Y = df2array(my_df, y_i, filter_value=nan_value); Z = df2array(my_df, z_i, filter_value=nan_value)
        E = np.ones(len(np.array(X)))

        if pos_end: print("Not developed. Showing what we have")

        fig_px = px.scatter_3d(x=X,y=Y,z=Z,color=event_num,opacity=0.7,title="All generated Events")
        fig_px.update_layout(legend = dict(font = dict(size = 35, color = "black"),orientation="h"), 
                             coloraxis_colorbar=dict(title="Events"))

    if evt != "ALL": 
        df_evt = event_df(my_df, evt); df_evt["PdgAll"] = [list(df_evt["PdgAll"][0].values())]
        X = filter_array(df_evt, x_i, nan_value); Y = filter_array(df_evt, y_i, nan_value); Z = filter_array(df_evt, z_i, nan_value)

        if energy_size: E = energy2plot(df_evt,len(X),OPT,nan_value=None,debug=False)
        else:           E = np.ones(len(np.array(X)[np.where(np.array(df_evt[z_i][0]) != nan_value)[0]])) * 25; E =list(E)

        pdg_names   = np.array(df_evt["PdgAll"][0])
        label_extra = np.array(df_evt["PdgAll"][0])[np.where(np.array(df_evt[Prim_label][0]) == 1)[0]][0]
        
        if pos_end:
            X = np.append(X,np.array(df_evt[x_f][0])[np.where(np.array(df_evt[Prim_label][0]) == 1)[0]][0])
            Y = np.append(Y,np.array(df_evt[y_f][0])[np.where(np.array(df_evt[Prim_label][0]) == 1)[0]][0])
            Z = np.append(Z,np.array(df_evt[z_f][0])[np.where(np.array(df_evt[Prim_label][0]) == 1)[0]][0])
            E.insert(len(E),E[0]);
            pdg_names = np.append(pdg_names,np.array([label_extra]))

        df2plot = pd.DataFrame({ "X": X, "Y": Y, "Z": Z, "PDG_Name": pdg_names })
        fig_px = px.scatter_3d(data_frame=df2plot,x="X",y="Y",z="Z",color="PDG_Name",opacity=0.7,title="Event %i"%evt)
        if pos_end==True: fig_px.add_traces(px.line_3d(x=[X[0], X[-1]], y=[Y[0], Y[-1]], z=[Z[0], Z[-1]]).data[0])

    fig_px.update_traces( marker={'size': E, 'line':dict(width=0, color='DarkSlateGrey')}, line={'width':15} )
    fig_px.update_layout( legend = dict(font = dict(size = 35, color = "black"),orientation="v"),
                          legend_title_text='',
                          legend_title = dict(font = dict(size = 20, color = "black")) )
    
    if html: pyoff.plot(fig_px, include_mathjax='cdn')
    if save: fig_px.write_html("vis_event.html", include_mathjax = 'cdn')

    fig_px.show()

def vis_particle_barplot_new(my_df,evt,OPT,pie=True,html=False,save=False):
    '''
    Visualize the particle distribution of a given event in a barplot and in a pie chart.
    VARIABLES:
        \n - my_df: Pandas DataFrame with the data to be plotted                    [type: pandas.DataFrame]
        \n - evt:   Event number to be plotted (i.e int or "ALL" for all events)    [type: int or str]
        \n - OPT:   dictionary with the branches to use                             [type = dict]
            \n * "BR_NAME": Name of the branch containing the particle names 
            \n * "EVENTID": Name of the branch containing the event ID
            \n * "PRIMARY": Name of the branch containing the primary particle information
        \n - html:  If True, the plot will be saved as an html file                 [type: bool]
        \n - save:  If True, the plot will be saved as a png file                   [type: bool]
    '''

    # Check if the OPT dictionary has the required keys
    if check_key(OPT,"PDG_LAB") and check_key(OPT,"PRIMARY"): Pdg_label = OPT["PDG_LAB"]; Prim_label = OPT["PRIMARY"]
    else: print_colored("ERROR: Missing arguments in OPT dict. Please check the documentation.","ERROR"); return

    # Check if the dataframe has the Prim_idx column (to check the primary particle of the event)
    if check_key(my_df,"Prim_idx") and check_key(my_df,"PdgNam") and check_key(my_df,"NumParts"): primary_name = np.array(my_df["Prim_idx"]) 
    else: print_colored("WARNING: Missing 'Prim_idx' in given dataframe. Please check mc_info.py output.","WARNING"); return

    if evt != "ALL":
        primary_evt = r"$"+str(Particle.from_pdgid(primary_name[evt]).latex_name)+"$"
        df_evt = event_df(my_df, evt)

        title = "Event %i - Particle distribution"%evt; annotate_all=str("")
        X = np.array(df_evt["PdgNam"][0]); Y = np.array(df_evt["NumParts"][0]); df2plot = pd.DataFrame({ "X": X, "Y": Y})

    if evt == "ALL":
        primary_evt = r"$"+str(Particle.from_pdgid(primary_name[0]).latex_name)+"$" # PARCHE
        total_particles = particle_info_summed(my_df,"PdgNam","NumParts",debug=True)
        X = list(total_particles.keys()); Y = list(total_particles.values()); df2plot = pd.DataFrame({ "X": np.array(X), "Y": np.array(Y)})
        title = "All events - Particle distribution"; annotate_all = ""
        # annotate_all = " ("+str(np.round(100*len(np.where(primary_name == primary_name[0])[0])/len(run[branch]["Event"]),2))+"%"+" evts) "

    fig_px = px.bar(df2plot, x="X", y="Y", title=title,text_auto=True) # X = PDGNames of MCPart of the branch + evt; Y = list number particles 
    fig_px.add_vline(x=primary_evt, line_dash="dash", line_color="red")
    fig_px.add_annotation(text="PRIMARY"+annotate_all, x=primary_evt, y=max(Y), arrowhead=1, showarrow=False, font_color="red")
    fig_px.update_layout( xaxis_title="Particle", yaxis_title="Counts",
                          font=dict(size=15),
                          xaxis = dict( tickfont = dict(size=30)) )
    if html: pyoff.plot(fig_px, include_mathjax='cdn')
    if save: fig_px.write_html("vis_particle_barplot.html", include_mathjax = 'cdn')
    fig_px.show()
    
    if pie:
        fig2_px = px.pie(df2plot,values="Y", names ="X",title=title)
        fig2_px.update_layout( font=dict(size=15), 
                               legend = dict(font = dict(size = 30, color = "black"),orientation="v") ) 
        if html: pyoff.plot(fig2_px, include_mathjax='cdn')
        if save: fig2_px.write_html("vis_particle_pie.html", include_mathjax = 'cdn')
        fig2_px.show()

def vis_particle_energy_new(my_df,evt,OPT,compute="sum",pie=True,html=False, save=False):
    '''
    Visualize the energy distribution of a given event in a barplot and in a pie chart.
    VARIABLES:
        \n - my_df: Pandas DataFrame with the data to be plotted                    [type: pandas.DataFrame]
        \n - evt:   Event number to be plotted (i.e int or "ALL" for all events)    [type: int or str]
        \n - OPT:   dictionary with the branches to use                             [type = dict]
            \n * "PDG_LAB": Name of the branch containing the particle names
            \n * "ENERGY": Name of the branch containing the energy information
        \n - *implementting*compute: If "sum", the total energy of each particle will be computed. If "mean", the mean energy of each particle will be computed. [type: str]
        \n - html:  If True, the plot will be saved as an html file                 [type: bool]
        \n - save:  If True, the plot will be saved as a png file                   [type: bool]
    '''
    # Check if the OPT dictionary has the required keys
    if check_key(OPT,"PDG_LAB") and check_key(OPT,"ENERGY"): Pdg_label = OPT["PDG_LAB"]; Ene_label = OPT["ENERGY"]
    else: print_colored("ERROR: Missing arguments in OPT dict. Please check the documentation.","ERROR"); return

    if evt != "ALL":
        df_evt = event_df(my_df, evt)

        if check_key(df_evt,"Energies") and check_key(df_evt,"PdgNam"): energies = df_evt["Energies"][0]
        else: print_colored("WARNING: Missing keys in given dataframe. Please check mc_info.py output.","WARNING"); return

        title = "Event %i - Energy distributions"%evt
        X = np.array(df_evt["PdgNam"][0]); Y = []; 
        [Y.append(np.sum(energies[x])) for x in X]; 
        df2plot = pd.DataFrame({ "X": X, "Y": Y}); df2plot.columns.name = 'MEAN ENERGY (Event %i)'%evt; print(df2plot)
        e = energy_conservation(my_df,evt,OPT)
        print("\nPrimary particle energy:",np.round(df2plot.iloc[0]["Y"],3) ,"GeV = descendants energy?:",np.round(sum(df2plot.iloc[1:]["Y"]),3), "GeV")

    if evt == "ALL":
        if check_key(my_df,"Energies") and check_key(my_df,"PdgNam"): energies_dict = np.array(my_df["Energies"])
        else: print_colored("WARNING: Missing keys in given dataframe. Please check mc_info.py output.","WARNING"); return

        energies = {}
        for evt in range(len(energies_dict)):
            df_evt = event_df(my_df, evt)
            for part in energies_dict[evt]:
                if part not in energies.keys(): energies[part] = [energies_dict[evt][part]]
                else: energies[part].append(energies_dict[evt][part])

        energies_sum = {}; 
        for key in energies.keys(): 
            energies[key] = (np.concatenate(energies[key]))
            energies_sum[key] = np.sum(energies[key])
        X = list(energies_sum.keys()); Y = list(energies_sum.values())
        df2plot = pd.DataFrame({ "X": X, "Y": Y}); df2plot.columns.name = 'MEAN ENERGY (ALL EVENTS)'; print(df2plot)

        print("\nPrimary particle energy:",np.round(np.array(df2plot.iloc[0]["Y"]),3)  , "GeV = descendants energy?:",np.round(sum(np.array(df2plot.iloc[1:]["Y"])),3), "GeV")
        e = energy_conservation(my_df,"ALL",OPT)
        title = "All events - Particle distribution"; annotate_all = ""
        
    fig_px = go.Figure()
    for x in X: fig_px.add_trace(go.Histogram(x=energies[x], name=x))
    fig_px.update_layout(barmode="overlay",title = title, xaxis_title="Energy [GeV]", yaxis_title="Counts",
                         font=dict(size=15),
                         xaxis = dict( tickfont = dict(size=15)) ,
                         legend = dict(font = dict(size = 30, color = "black"),orientation="v") ) 
    if html: pyoff.plot(fig_px, include_mathjax='cdn')
    if save: fig_px.write_html("vis_energy.html", include_mathjax = 'cdn')
    fig_px.show()

    if pie: 
        fig2_px = px.pie(df2plot,values="Y", names ="X",title=title)
        fig2_px.update_layout( font=dict(size=15), 
                               legend = dict(font = dict(size = 30, color = "black"),orientation="v") ) 
        if html: pyoff.plot(fig2_px, include_mathjax='cdn')
        if save: fig2_px.write_html("vis_energy_pie.html", include_mathjax = 'cdn')
        fig2_px.show()

def vis_particle_ancestry_new(my_df,evt,OPT,html=False,save=False):
    '''
    Visualize the particle ancestry distribution of a given event in a barplot and/or pie.
    VARIABLES:
        \n - my_df: Pandas DataFrame with the data to be plotted                    [type: pd.DataFrame]
        \n - evt:   Event number to be plotted (i.e int or "ALL" for all events)    [type: int or str]
        \n - OPT:  dictionary with the branches to use                              [type = dict]
            \n * "ANCESTOR": Name of the branch containing the ancestry information
    '''  

    # Check if the OPT dictionary has the required keys
    if check_key(OPT,"ANCESTOR"): ancestor_label = OPT["ANCESTOR"]
    else: print_colored("ERROR: Missing arguments in OPT dict. Please check the documentation.","ERROR"); return
    
    if evt == "ALL": print("Not available yet")

    # if evt != "ALL":
    df_evt = event_df(my_df, evt)

    if check_key(df_evt,"PdgAll"): pdg_names = list(df_evt["PdgAll"][0].values()); X = np.array(df_evt[ancestor_label][0].values())
    else: print_colored("WARNING: Missing keys in given dataframe. Please check mc_info.py output.","WARNING"); return

    df2plot = pd.DataFrame({ "X": X, "Particles": pdg_names}); title = "Event %i - Ancestry level"%evt


    fig_px = px.histogram(df2plot, x="X", color="Particles", title=title,text_auto=True) # X = PDGNames of MCPart of the branch + evt; Y = list ancestry level for each particles
    fig_px.update_layout( xaxis_title="Ancestry level", yaxis_title="Counts",
                          font=dict(size=15),
                          legend = dict(font = dict(size = 30, color = "black"),orientation="h"),
                          legend_title_text='',
                          xaxis = dict( tickfont = dict(size=15)) )
    
    if html: pyoff.plot(fig_px, include_mathjax='cdn')
    if save: fig_px.write_image("vis_ancestry.png")

    fig_px.show()

def vis_detector_response(my_df, evt, OPT, title="", subtitles=[""], xlabel="", ylabel="",energy_size=False,nan_value=None,html=False, save=False):
    '''
    Plot the detector response for a given event.
    VARIABLES:
    \n - my_df:     Pandas dataframe with the data          [type: pd.DataFrame]
    \n - evt:       Event number                            [type: int]
    \n - OPT:       dictionary with the branches to use     [type = dict]
        \n * CHS:   Label of the channel
        \n * ADC:   Label of the ADC
        \n * TDC:   Label of the TDC
        \n * VIEW:  Label of the view
    \n - title:     Title of the plot                       [type: str]
    \n - subtitles: List with the subtitles of the plot     [type: list]
    \n - xlabel:    Label of the x axis                     [type: str]
    \n - ylabel:    Label of the y axis                     [type: str]
    \n - energy_size: If True, points' size is porp to E    [type: bool]
    \n - nan_value: Value to be used as NaN                 [type: float]
    \n - html:  If True, the plot is saved as a html file   [type: bool]
    \n - save:  If True, the plot is saved as a png file    [type: bool]
    '''

    # Check if the OPT dictionary has the required keys
    if check_key(OPT,"CHS") and check_key(OPT,"ADC") and check_key(OPT,"TDC") and check_key(OPT,"VIEW"): 
        chs_label = OPT["CHS"]; adc_label = OPT["ADC"]; tdc_label = OPT["TDC"]; view_label = OPT["VIEW"]
    else: print_colored("ERROR: Missing arguments in OPT dict. Please check the documentation.","ERROR"); return

    # Get the data from the dataframe
    df_evt = event_df(my_df, evt)
    chs    = filter_array(df_evt, chs_label, nan_value)
    # adc    = filter_array(df_evt, adc_label, None)
    tdc    = filter_array(df_evt, tdc_label, nan_value)
    view   = filter_array(df_evt, view_label,nan_value)
    
    # Check if the dataframe has the PdgAll column. If it does, use the names of the particles to color the points
    if check_key(df_evt,"PdgAll"):   df_evt["PdgAll"] = [list(df_evt["PdgAll"][0].values())]; color = np.array(df_evt["PdgAll"][0])
    else: color = np.ones(len(chs)); print('\033[93m'+"WARNING: Missing 'PdgAll' in given dataframe. Please check mc_info.py output."+'\033[0m')
    fig_px = px.scatter(x=chs,y=tdc,facet_col=view,color=color, opacity=0.7)
    if title  != "": fig_px.update_layout(title=title + " (Event %d)"%evt)
    if subtitles  != [""]: 
        for a in fig_px.layout.annotations: a.text = subtitles[int(a.text.split("=")[1])]
    if xlabel != "": fig_px.update_layout(xaxis1_title=xlabel,xaxis2_title=xlabel,xaxis3_title=xlabel)
    if ylabel != "": fig_px.update_layout(yaxis_title=ylabel)

    if energy_size: E = energy2plot(df_evt,len(chs),OPT,nan_value=None,debug=False)
    else:           E = np.ones(len(np.array(chs)))*15; E = list(E)

    fig_px.update_traces( marker={'size': E, 'line':dict(width=0, color='DarkSlateGrey')} )
    fig_px.update_layout( legend = dict(title = "", font = dict(size = 35, color = "black"),orientation="h"),
                          legend_title = dict(font = dict(size = 20, color = "black")) )
    
    if html: pyoff.plot(fig_px, include_mathjax='cdn')
    if save: fig_px.write_html("vis_detector_response.html", include_mathjax='cdn')
    
    fig_px.show()