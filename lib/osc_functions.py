import os
import glob
import uproot
import numpy as np
import pandas as pd
import plotly.express as px

from scipy import stats as sps
from scipy.stats import chi2
from scipy import interpolate
from itertools import product

from lib.io_functions import print_colored, read_input_file
from lib.plt_functions import format_coustom_plotly, unicode

def get_nadir_angle(show=False,debug=False):
    '''
    This function can be used to obtain the nadir angle distribution for DUNE.
    VARIABLES:
        \n - show: if True, show the plot (default: False)
    '''
    with uproot.open("../data/OSCILLATION/nadir.root") as nadir:
        # Loas pdf histogram
        pdf = nadir["nadir;1"]
        pdf_array = pdf.to_hist().to_numpy()
        xbin_edges = pdf_array[1]
        xnadir_centers = 0.5*(xbin_edges[1:] + xbin_edges[:-1])
        ynadir_centers = pdf_array[0]

        # Create scatter plot
        if show: 
            fig = px.scatter(x=xnadir_centers, y=ynadir_centers, labels={'x':'Nadir Angle cos('+unicode("eta")+')', 'y':'PDF'})
            fig = format_coustom_plotly(fig,figsize=(800,600),tickformat=(".1f",".0e"),ranges=(None,[4e-4,1.2e-3]))
            fig.show()
            
    if debug: print_colored("Nadir angle loaded!","DEBUG")
    return (xnadir_centers,ynadir_centers)

def get_oscillation_datafiles(dm2="DEFAULT",sin13="DEFAULT",sin12="DEFAULT",path="../data/OSCILLATION/",ext="root",auto=False,debug=False):
    '''
    This function can be used to obtain the oscillation data files for DUNE's solar analysis.  
    VARIABLES:
        \n - dm2: list of dm2 values (default: [6e-5,7.4e-5]).
        \n - sin13: list of sin13 values (default: [0.021]).
        \n - sin12: list of sin12 values (default: [0.303]).
        \n - path: path to the data files (default: "../data/OSCILLATION/root/").
        \n - auto: if True, automatically find all the data files in the path (default: True).
    RETURNS:
        \n - (dm2,sin13,sin12): tuple containing the dm2, sin13 and sin12 values found in the path.
    '''
    if auto:
        data_files = glob.glob(path+'*_dm2_*_sin13_*_sin12_*')
        string_dm2, trash, string_sin13, trash, string_sin12 = zip(*[tuple(map(str,os.path.basename(osc_file).split('.'+ext)[0].split("_")[-5:])) for osc_file in data_files])
        found_dm2 = [float(i) for i in string_dm2]
        found_sin13 = [float(i) for i in string_sin13]
        found_sin12 = [float(i) for i in string_sin12]
        
    if auto == False:
        if type(dm2) == list and type(sin13) == list and type(sin12) == list:
            found_dm2, found_sin13, found_sin12 = [], [], []
            for this_dm2,this_sin13,this_sin12 in zip(dm2,sin13,sin12):
                    if os.path.isfile(path+'osc_probability_dm2_%.3e_sin13_%.3e_sin12_%.3e'%(this_dm2,this_sin13,this_sin12)+'.'+ext):
                        found_dm2.append(this_dm2) 
                        found_sin13.append(this_sin13)
                        found_sin12.append(this_sin12)
                    else:
                        print_colored("WARNING: file %sosc_probability_dm2_%.3e_sin13_%.3e_sin12_%.3e"%(path,this_dm2,this_sin13,this_sin12)+'.'+ext+" not found!","WARNING") 
        
        elif type(dm2) == float and type(sin13) == float and type(sin12) == float:
            if os.path.isfile(path+"osc_probability_dm2_%.3e_sin13_%.3e_sin12_%.3e"%(dm2,sin13,sin12)+'.'+ext):
                found_dm2 = [dm2]
                found_sin13 = [sin13]
                found_sin12 = [sin12]
            else:
                print_colored("WARNING: file %sosc_probability_dm2_%.3e_sin13_%.3e_sin12_%.3e"%(path,this_dm2,this_sin13,this_sin12)+'.'+ext+" not found! Returning default.","WARNING")
                dm2, sin13, sin12 = "DEFAULT", "DEFAULT", "DEFAULT"
        
        elif dm2 == "DEFAULT" and sin13 == "DEFAULT" and sin12 == "DEFAULT":
            analysis_info = read_input_file("analysis",DOUBLES=["SOLAR_DM2","SIN13","SIN12"],debug=debug)
            found_dm2, found_sin13, found_sin12 = analysis_info["SOLAR_DM2"], analysis_info["SIN13"], analysis_info["SIN12"]

        else:
            print_colored("ERROR: dm2 and sin12 must be both lists or floats!","FAIL")
            raise TypeError
        
    if type(auto) != bool:
        print_colored("ERROR: auto must be a boolean!","FAIL")
        raise TypeError
    
    if debug: print_colored("Found %d oscillation files!"%len(found_dm2),"INFO")
    return (found_dm2,found_sin13,found_sin12)

def get_oscillation_map(path="../data/OSCILLATION/",dm2="DEFAULT",sin13="DEFAULT",sin12="DEFAULT",auto=True,rebin=True,output="interp",save=False,show=False,ext="root",debug=False):
    '''
    This function can be used to obtain the oscillation correction for DUNE's solar analysis.
    VARIABLES:
        \n - nadir_centers: tuple containing the nadir angle and the PDF (output of get_nadir_angle)
        \n - show: if True, show the plot (default: False)
        \n - save: if True, save the oscillation correction to a numpy array (default: False)
    '''
    
    df_dict = {}
    interp_dict = {}
    nadir_data = get_nadir_angle()
    subfolder=''
    if ext == 'pkl': 
        if subfolder == True: subfolder = 'rebin/'
        else: subfolder = 'raw/'
    dm2,sin13,sin12 = get_oscillation_datafiles(dm2,sin13,sin12,path=path+ext+'/'+subfolder,ext=ext,auto=auto,debug=debug)
    analysis_info = read_input_file("analysis",INTEGERS=["ROOT_NADIR_RANGE","ROOT_NADIR_BINS"],debug=debug)

    root_nadir_edges = np.linspace(analysis_info["ROOT_NADIR_RANGE"][0],analysis_info["ROOT_NADIR_RANGE"][1],analysis_info["ROOT_NADIR_BINS"][0]+1)
    root_nadir_centers = (root_nadir_edges[1:]+root_nadir_edges[:-1])/2
    
    for dm2_value,sin13_value,sin12_value in zip(dm2,sin13,sin12):
        # Format dm2 and sin12 values to be used in the file name with appropriate precision
        dm2_value = float("%.3e"%dm2_value)
        sin12_value = float("%.3e"%sin12_value)
        if glob.glob(path+"/pkl/rebin/osc_probability_dm2_%.3e_sin13_%.3e_sin12_%.3e.pkl"%(dm2_value,sin13_value,sin12_value)) != []:
            if debug: print_colored("Loading rebin data from: "+path+"/pkl/rebin/osc_probability_dm2_%.3e_sin13_%.3e_sin12_%.3e.pkl"%(dm2_value,sin13_value,sin12_value),"DEBUG")
            df = pd.read_pickle(path+"/pkl/rebin/osc_probability_dm2_%.3e_sin13_%.3e_sin12_%.3e.pkl"%(dm2_value,sin13_value,sin12_value))
        
        if rebin == False and glob.glob(path+"/pkl/rebin/osc_probability_dm2_%.3e_sin13_%.3e_sin12_%.3e.pkl"%(dm2_value,sin13_value,sin12_value)) != []:
            if debug: print_colored("Loading rebin data from: "+path+"/pkl/rebin/osc_probability_dm2_%.3e_sin13_%.3e_sin12_%.3e.pkl"%(dm2_value,sin13_value,sin12_value),"DEBUG")
            df = pd.read_pickle(path+"/pkl/rebin/osc_probability_dm2_%.3e_sin13_%.3e_sin12_%.3e.pkl"%(dm2_value,sin13_value,sin12_value))

        else:
            if debug: print_colored("Loading raw data from: "+path+"/root/osc_probability_dm2_%.3e_sin13_%.3e_sin12_%.3e%s"%(dm2_value,sin13_value,sin12_value,'.'+ext),"DEBUG")
            data = uproot.open(path+"/root/osc_probability_dm2_%.3e_sin13_%.3e_sin12_%.3e%s"%(dm2_value,sin13_value,sin12_value,'.'+ext))
            # Convert the histogram to a pandas DataFrame
            # hist = data["hsurv;1"]["osc_probability_dm2_%.3e_sin13_%.3e_sin12_%.3e;1"%(dm2_value,sin13_value,sin12_value)] # Load the 2D histogram
            hist = data["hsurv;1"] # Load the 2D histogram
            data_array = hist.to_hist().to_numpy()
            
            # Data contains the bin edges and the bin contents
            data = data_array[0][:,:-1]
            root_energy_edges = data_array[1]
            root_energy_centers = 0.5*(root_energy_edges[1:] + root_energy_edges[:-1])
            root_nadir_edges = data_array[2][:-1]
            root_nadir = 0.5*(root_nadir_edges[1:] + root_nadir_edges[:-1])

            # Create a DataFrame with the bin contents
            df1 = pd.DataFrame(data, index=1e3*root_energy_centers, columns=root_nadir)
            df2 = pd.DataFrame(data_array[0][:,-1][:, np.newaxis]*np.ones((len(root_energy_centers),len(root_nadir))), index=1e3*root_energy_centers, columns=1+root_nadir)
            df = df1.join(df2).T
            
            # Interpolate nadir data to match ybins
            nadir = interpolate.interp1d(nadir_data[0],nadir_data[1],kind='linear',fill_value='extrapolate')
            nadir_y = nadir(x=root_nadir_centers)
            # normalize nadir distribution
            nadir_y = nadir_y/nadir_y.sum()
            df = df.mul(nadir_y,axis=0)
        
        if rebin:
            save_path = path+"/pkl/rebin/osc_probability_dm2_%.3e_sin13_%.3e_sin12_%.3e.pkl"%(dm2_value,sin13_value,sin12_value)
            if glob.glob(save_path) != []:
                if debug: print_colored("Loading rebinned data from %s"%save_path,"DEBUG")
                df = pd.read_pickle(save_path)
            else:
                if debug: print_colored("Rebinning data!","DEBUG")
                df = rebin_df(df,show=False,save=save,save_path=save_path,debug=debug)


        # if save and glob.glob(path+"/pkl/raw/osc_probability_dm2_%.3e_sin13_%.3e_sin12_%.3e.pkl"%(dm2_value,sin13_value,sin12_value)) == []:
        #     if debug: print_colored("Saving raw data to: "+path+"/pkl/raw/osc_probability_dm2_%.3e_sin13_%.3e_sin12_%.3e.pkl"%(dm2_value,sin13_value,sin12_value),"DEBUG")
        #     df.to_pickle("../data/OSCILLATION/pkl/raw/osc_probability_dm2_%.3e_sin13_%.3e_sin12_%.3e.pkl"%(dm2_value,sin13_value,sin12_value))

        if show:
            fig = px.imshow(df,color_continuous_scale='turbo', origin='lower', aspect='auto')
            fig = format_coustom_plotly(fig,figsize=(800,600))
            fig.show()
        
        if output == "interp":                
            osc_map_x = df.loc[df.index[0],:].keys().to_list()
            osc_map_y = df.loc[:,df.columns[0]].keys().to_list()
            oscillation_map = interpolate.RegularGridInterpolator((osc_map_x,osc_map_y),df.loc[:,:].to_numpy().T,method='linear',bounds_error=False,fill_value=None)
            interp_dict[(dm2_value,sin13_value,sin12_value)] = oscillation_map
        
        df_dict[(dm2_value,sin13_value,sin12_value)] = df
    
    if output == "interp":
        if debug: print_colored("Returning interpolation dictionary!","DEBUG")
        return interp_dict
    
    if output == "df":
        if debug: print_colored("Returning dataframe dictionary!","DEBUG")
        return df_dict
    
    else:
        print_colored("ERROR: output must be 'interp' or 'df'!","FAIL")
        return None
    
def rebin_df(df,xarray=[],yarray=[],show=False,save=True,save_path="../data/pkl/rebin/df.pkl",debug=False):
    '''
    This function can be used to rebin any dataframe that has a 2D index (like an imshow dataset).
    VARIABLES:
    \n - df: dataframe to rebin
    \n - xarray: array of xbins (default: [])
    \n - yarray: array of ybins (default: [])
    '''
    analysis_info = read_input_file("analysis",INTEGERS=["RECO_ENERGY_RANGE","RECO_ENERGY_BINS","NADIR_RANGE","NADIR_BINS"],debug=False)
    energy_edges = np.linspace(analysis_info["RECO_ENERGY_RANGE"][0],analysis_info["RECO_ENERGY_RANGE"][1],analysis_info["RECO_ENERGY_BINS"][0]+1)
    energy_centers = (energy_edges[1:]+energy_edges[:-1])/2
    
    df.index = df.index.astype(float)
    if xarray == [] and yarray == []:
        reduced_rows_edges = np.linspace(-1, 1,40+1,endpoint=True)
        reduced_rows = 0.5*(reduced_rows_edges[1:] + reduced_rows_edges[:-1])
        reduced_rows = np.round(reduced_rows,4)
        if debug: print_colored("Rebinning data with default parameters!","DEBUG")
    else:
        energy_centers = xarray
        reduced_rows = yarray
        if debug: print_colored("Rebinning data with custom parameters!","DEBUG")
        
    # Create an empty reduced data frame
    goal_int = df.sum().mean()
    small_df = pd.DataFrame(index=reduced_rows, columns=energy_centers)

    for col in energy_centers:
        for row in reduced_rows:
            # Calculate the average of the original data within the corresponding range
            step_col = energy_centers[1] - energy_centers[0]
            start_col = float(col) - step_col/2
            end_col = float(col) + step_col/2
            
            step_row = reduced_rows[1] - reduced_rows[0]
            start_row = round(float(row) - step_row/2, 4)
            end_row = round(float(row) + step_row/2, 4)

            small_df.loc[row, col] = df.loc(axis=1)[start_col:end_col].loc(axis=0)[start_row:end_row].sum().mean()
    # Substitute NaN values with 0
    small_df = small_df.fillna(0)
        
    # Print the reduced data frame
    if show:
        fig = px.imshow(small_df,
            aspect="auto",
            origin='lower',
            color_continuous_scale='turbo',
            title="Oscillation Correction Map",
            labels=dict(y="Nadir Angle (°)", x="TrueEnergy"))
        fig = format_coustom_plotly(fig,figsize=(800,600))
        fig.show()
    
    if save:
        small_df.to_pickle(save_path)    
    
    return small_df

def compute_log_likelihood(pred_df,fake_df,method="log-likelihood",debug=False):
    '''
    This function can be used to compute the log likelihood of a prediction given a fake data set.
    VARIABLES:
        \n - pred_df: prediction dataframe. Must have the same shape as fake_data_df.
        \n - fake_data_df: fake data dataframe. Must have the same shape as pred_df.
    RESULT:
        \n - ll: log likelihood of the prediction given the fake data.
    '''
    
    if method == "log-likelihood":
        ll = 0
        for col in fake_df.columns:
            for row in fake_df.index:
                if pred_df.loc[row,col] == 0: continue
                if fake_df.loc[row,col] == 0:
                    this_ll = pred_df.loc[row,col] - fake_df.loc[row,col]
                    if np.isnan(this_ll): this_ll = 0
                    ll = ll + this_ll
                else:
                    this_ll = pred_df.loc[row,col] - fake_df.loc[row,col] + fake_df.loc[row,col]*np.log(fake_df.loc[row,col]/pred_df.loc[row,col])
                    if np.isnan(this_ll): this_ll = 0
                    ll = ll + this_ll
        ll = 2*ll
        chi_square = ll
    
    if debug: print_colored("Chi-square computed!","DEBUG")
    return chi_square