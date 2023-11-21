import pandas as pd
import numpy as np
import itertools
import dask.dataframe as dd
from dask import delayed

from rich import print as rprint
from .io_functions import print_colored, get_branches2use

def explode(df, columns_to_explode, debug=False):
    '''
    Function to explode a list of columns of a dataframe.

    Args:
        df (pandas.DataFrame): Dataframe to explode.
        columns_to_explode (list(str)): List of columns to explode.
        debug (bool): If True, the debug mode is activated.

    Returns:
        result_df (pandas.DataFrame): Dataframe exploded.
    '''
    # Convert Pandas DataFrame to Dask DataFrame
    ddf = dd.from_pandas(df, npartitions=2)  # Adjust the number of partitions as needed

    # Define a function to explode a column
    @delayed
    def explode_column(column):
        return column.explode()

    # Explode each column in parallel
    exploded_columns = [explode_column(ddf[col]) for col in columns_to_explode]

    # Compute the results
    try:
        result_columns = dd.compute(*exploded_columns)
    # If TypeError: 'dict_values' object does not support indexing, try:
    except TypeError:
        result_columns = dd.compute(*list(exploded_columns))
    # Combine the results with regular columns
    result_df = pd.concat([df.drop(columns=columns_to_explode)] + list(result_columns), axis=1)

    if debug: print_colored("Dataframe exploded!","SUCCESS")
    return result_df

def npy2df(run, tree="", branches=[], debug=False):
    '''
    Function to convert the dictionary of the TTree into a pandas Dataframe.

    Args:
        run (dict): Dictionary with the data to save (delete the keys that you don't want to save or set Force = False to save only the new ones)
        tree (str): Name of the tree to convert to dataframe.
        branches (list(str)): List of branches to convert to dataframe.
        debug (bool): If True, the debug mode is activated.

    Returns:
        df (pandas.DataFrame): Dataframe with the data of the TTree.
    '''
    if tree == "" and branches == []:
        df = pd.DataFrame(run)
    else:
        if debug: print_colored("Converting tree %s to dataframe"%tree,"DEBUG")
        df = pd.DataFrame()
        if branches == []: branches = run[tree].keys()
        for branch in branches:
            try:
                df[branch] = run[tree][branch].tolist()
            except AttributeError:
                try: df[branch] = run[tree][branch]
                except ValueError:
                    print("ValueError: ",branch)
                    continue
    
    if debug: rprint(df.groupby(["Geometry","Version","Name"])["Event"].count())
    print_colored("Dataframe for tree %s created"%tree,"SUCCESS")
    return df

def dict2df(run, debug=False):
    '''
    Function to convert the dictionary of the TTree into a list of pandas Dataframes of len = len(branches)
    i.e. df_list = [df_truth, df_reco, ...]

    Args:
        run (dict): Dictionary with the data to save (delete the keys that you don't want to save or set Force = False to save only the new ones)
        debug (bool): If True, the debug mode is activated.

    Returns:
        df_list (list(pandas.DataFrame)): List of pandas Dataframes of len = len(branches)
    '''
    df_list = []
    branches = get_branches2use(run,debug=debug) # Load the branches of the TTree not in ['Name', 'Path', 'Labels', 'Colors']
    for branch in branches:
        df = pd.DataFrame()
        for key in run[branch].keys():
            try: df[key] = run[branch][key].tolist()
            except AttributeError: df[key] = run[branch][key]
            if debug: 
                print_colored(" --- Dataframe for key %s created\n"%key, "DEBUG")
                print_colored(df[key]+"\n","DEBUG")
        df_list.append(df)

    print_colored("DataFrame generated from dict!", "SUCCESS")
    return df_list

def merge_df(df1, df2, label1, label2, debug=False):
    '''
    Function to merge two dataframes in one adding an extra column to indicate its origin. 
    Also maintain the columns that are not in both df an include NaNs in the missing columns.

    Args:
        df1 (pandas.DataFrame): First dataframe to merge.
        df2 (pandas.DataFrame): Second dataframe to merge.
        label1 (str): Name of the first dataframe.
        label2 (str): Name of the second dataframe.
        debug (bool): If True, the debug mode is activated.

    Returns:
        df (pandas.DataFrame): Merged dataframe.
    '''
    df1["Label"] = label1 # Add a column to indicate the origin of the event
    df2["Label"] = label2 # Add a column to indicate the origin of the event
    df = pd.concat([df1,df2], ignore_index=True) # Merge the two dataframes
    if debug: print_colored(" --- New dataframe from %s, %s created"%(label1,label2), "DEBUG")
    
    return df