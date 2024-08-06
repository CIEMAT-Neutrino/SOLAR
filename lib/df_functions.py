from src.utils import get_project_root

import json
import pandas as pd
import numpy as np
import itertools
import dask.dataframe as dd
import plotly.graph_objects as go
import plotly.express as px

from dask import delayed
from rich import print as rprint
from .io_functions import (
    print_colored,
    get_branches2use,
    get_bkg_config,
    get_simple_name,
    save_figure,
)
from plotly.subplots import make_subplots
from .plt_functions import format_coustom_plotly

root = get_project_root()

def explode(df, explode, keep=None, debug=False):
    """
    Function to explode a list of columns of a dataframe.

    Args:
        df (pandas.DataFrame): Dataframe to explode.
        explode (list(str)): List of columns to explode.
        keep (list(str)): List of columns to keep.
        debug (bool): If True, the debug mode is activated.

    Returns:
        result_df (pandas.DataFrame): Dataframe exploded.
    """
    # Check that entries in explode and keep are colums of the dataframe. If not, remove and print a warning
    if keep is not None:
        old_keep = keep.copy()
        for col in old_keep:
            if col not in df.columns:
                print_colored(f"Column {col} not found in the dataframe", "WARNING")
                keep.remove(col)
    # make a copy of explode to avoid modifying the original list
    old_explode = explode.copy()
    for col in old_explode:
        if col not in df.columns:
            print_colored(f"Column {col} not found in the dataframe", "WARNING")
            explode.remove(col)

    try:
        # Explode the columns
        if keep is not None:
            # Make a copy of the dataframe but only with the columns to keep + the columns to explode
            result_df = df[keep + explode].explode(explode)
        else:
            result_df = df.explode(explode)
    
    except ValueError:
        # Convert Pandas DataFrame to Dask DataFrame but keep the columns in keep + explode
        ddf = dd.from_pandas(df[keep + explode], npartitions=2)
        # ddf = dd.from_pandas(df, npartitions=2)  # Adjust the number of partitions as needed

        # Define a function to explode a column
        @delayed
        def explode_column(column):
            return column.explode()

        # Explode each column in parallel
        exploded_columns = [explode_column(ddf[col]) for col in explode]

        # Compute the results
        try:
            result_columns = dd.compute(*exploded_columns)
        # If TypeError: 'dict_values' object does not support indexing, try:
        except TypeError:
            result_columns = dd.compute(*list(exploded_columns))
        # Combine the results with regular columns
        result_df = pd.concat(
            [df.drop(columns=explode)] + list(result_columns), axis=1
        )

    if debug:
        print_colored("Dataframe exploded!", "SUCCESS")
    return result_df


def npy2df(run, tree, branches=[], debug=False):
    """
    Function to convert the dictionary of the TTree into a pandas Dataframe.

    Args:
        run (dict): Dictionary with the data to save (delete the keys that you don't want to save or set Force = False to save only the new ones)
        tree (list(str)): Name of the tree to convert to dataframe.
        branches (list(str)): List of branches to convert to dataframe.
        debug (bool): If True, the debug mode is activated.

    Returns:
        df_dict (dict(pandas.DataFrame)): Dataframe dict with the data of the TTree.
    """
    if tree not in run.keys():
        rprint(f"[red]Tree {tree} not found in the dictionary[/red]")
        raise KeyError

    df = pd.DataFrame()
    if branches == []:
        branches = run[tree].keys()
    for branch in branches:
        try:
            df[branch] = run[tree][branch].tolist()
        except AttributeError:
            try:
                df[branch] = run[tree][branch]
            except ValueError:
                print("ValueError: ", branch)
                continue
        except ValueError:
            print("ValueError: ", branch)
            continue

    if debug:
        rprint(df.groupby(["Geometry", "Version", "Name"])["Event"].count())
    return df


def dict2df(run, debug=False):
    """
    Function to convert the dictionary of the TTree into a list of pandas Dataframes of len = len(branches)
    i.e. df_list = [df_truth, df_reco, ...]

    Args:
        run (dict): Dictionary with the data to save (delete the keys that you don't want to save or set Force = False to save only the new ones)
        debug (bool): If True, the debug mode is activated.

    Returns:
        df_list (list(pandas.DataFrame)): List of pandas Dataframes of len = len(branches)
    """
    df_list = []
    branches = get_branches2use(
        run, debug=debug
    )  # Load the branches of the TTree not in ['Name', 'Path', 'Labels', 'Colors']
    for branch in branches:
        df = pd.DataFrame()
        for key in run[branch].keys():
            try:
                df[key] = run[branch][key].tolist()
            except AttributeError:
                df[key] = run[branch][key]
            if debug:
                print_colored(" --- Dataframe for key %s created\n" % key, "DEBUG")
                print_colored(df[key] + "\n", "DEBUG")
        df_list.append(df)

    print_colored("DataFrame generated from dict!", "SUCCESS")
    return df_list


def merge_df(df1, df2, label1, label2, debug=False):
    """
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
    """
    df1["Label"] = label1  # Add a column to indicate the origin of the event
    df2["Label"] = label2  # Add a column to indicate the origin of the event
    df = pd.concat([df1, df2], ignore_index=True)  # Merge the two dataframes

    if debug:
        print_colored(
            " --- New dataframe from %s, %s created" % (label1, label2), "DEBUG"
        )
    return df


def reorder_df(df, info, bkg_dict, color_dict, debug=False):
    """
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
    """

    f = json.load(open(f"{root}/lib/import/generator_order.json", "r"))
    bkg_list = f[info["GEOMETRY"]][info["VERSION"]].keys()
    bkg_order = f[info["GEOMETRY"]][info["VERSION"]].values()

    # Reorder bkg_list according to bkg_order
    order = [x for _, x in sorted(zip(bkg_order, bkg_list))][2:]
    df = df[order]
    color_list = []
    for bkg in order:
        color_list.append(color_dict[list(bkg_dict.values()).index(bkg)])

    if debug:
        print_colored("Reordered dataframe with columns: %s" % order, "INFO")
    return df, color_list


def generate_truth_dataframe(run, info, fullname=True, debug=False):
    bkg_dict, color_dict = get_bkg_config(info)
    if fullname:
        columns = list(bkg_dict.values())[1:]
    else:
        name_dict = get_simple_name(list(bkg_dict.values())[1:])
        columns = [name_dict[name] for name in list(bkg_dict.values())[1:]]

    truth_gen_df = pd.DataFrame(
        np.asarray(run["Truth"]["TruthPart"])[:, 0 : len(list(bkg_dict.values())[1:])],
        columns=columns,
    )
    truth_gen_df["Geometry"] = run["Truth"]["Geometry"]
    truth_gen_df["Version"] = run["Truth"]["Version"]
    truth_gen_df["Name"] = run["Truth"]["Name"]
    truth_gen_df = truth_gen_df[
        (truth_gen_df["Version"] == info["VERSION"])
        & (truth_gen_df["Geometry"] == info["GEOMETRY"])
    ]
    if debug:
        rprint(truth_gen_df)
    return truth_gen_df


def calculate_mean_truth_df(truth_gen_df, debug=False):
    mean_truth_df = (
        truth_gen_df.drop(columns=["Geometry", "Version"]).groupby("Name").mean()
    )
    mean_truth_df = mean_truth_df.replace(0, np.nan).mean()
    return mean_truth_df


def calculate_pileup_df(mean_truth_df, info, factor=1, debug=False):
    mean_truth_df = mean_truth_df / info["TIMEWINDOW"]
    timewindow = (
        factor
        * 2
        * (info["EVENT_TICKS"] * 0.5e-6)
        / (2 * info["DETECTOR_SIZE_X"] / 100)
    )
    area = 1e-4 * info["DETECTOR_SIZE_Y"] * info["DETECTOR_SIZE_Z"] / np.pi
    data = timewindow * mean_truth_df.values
    names = mean_truth_df.index

    # Add neutrino
    data = np.append(data, timewindow * 1 / (info["EVENT_TICKS"] * 0.5e-6))
    names = np.append(names, "Neutrino")

    pileup_df = pd.DataFrame(
        data[:, None] * data[None, :] * factor**2 / area, index=names, columns=names
    )
    for i in range(len(names)):
        for j in range(len(names)):
            if i < j:
                pileup_df.iloc[i, j] = 0
    if debug:
        # Print summary information
        print("Time window: %.2e s" % timewindow)
        print("area: %.2e m³" % area)

        # rprint(pileup_df)
    return pileup_df


def calculate_pileup_df_dict(run, configs, factor=1, debug=False):
    pileup_df_dict = {}
    color_dict = {}
    for idx, config in enumerate(configs):
        info = json.load(open(f"{root}/config/{config}/{config}_config.json", "r"))
        bkg_dict, color_dict = get_bkg_config(info, debug=debug)
        truth_gen_df = generate_truth_dataframe(run, info, debug=debug)
        df = calculate_mean_truth_df(truth_gen_df, debug=debug)
        df, color_list = reorder_df(df, info, bkg_dict, color_dict, debug=debug)
        color_dict[config] = color_list
        pileup_df = calculate_pileup_df(df, info, factor=factor, debug=debug)
        pileup_df_dict[config] = pileup_df

    return pileup_df_dict, color_dict


def generate_pileup_matrix(run, configs, factor, save=False, show=False, debug=False):
    pileup_df_dict, color_dict = calculate_pileup_df_dict(
        run, configs, factor=factor, debug=debug
    )

    for idx, config in enumerate(configs):
        pileup_df = pileup_df_dict[config]
        fig = px.imshow(
            np.log10(pileup_df),
            color_continuous_scale="Turbo",
            labels=dict(color="log10"),
        )
        fig = format_coustom_plotly(fig, figsize=(1800, 1200))
        fig.update_layout(title="Pile-up probability per [s · m³]")

        fig.update_traces(
            text=pileup_df, texttemplate="<b>%{text:.2e}</b>", textfont_size=10
        )

        if save:
            fig.write_image(f"{root}/images/bkg/{config}_PileUp_Area.png", scale=1.5)
        if show:
            fig.show()
    return fig


def generate_background_distribution(
    run, configs, fullname=True, add_values=True, show=False, save=False, debug=False
):
    fig = make_subplots(
        rows=len(configs), cols=1, shared_yaxes=False, shared_xaxes=False
    )
    for idx, config in enumerate(configs):
        info = json.load(open(f"{root}/config/{config}/{config}_config.json", "r"))
        bkg_dict, color_dict = get_bkg_config(info)
        truth_gen_df = generate_truth_dataframe(run, info, fullname=fullname)
        mean_truth_df = calculate_mean_truth_df(truth_gen_df)
        mean_truth_df, color_list = reorder_df(
            mean_truth_df, info, bkg_dict, color_dict
        )
        mean_truth_df = mean_truth_df / info["TIMEWINDOW"]
        fig = plot_generator_distribution(
            fig, mean_truth_df, color_list, (1, 1), add_values=add_values, debug=debug
        )

    fig = format_generator_distribution(fig, info, debug=debug)
    if save:
        name = truth_gen_df["Name"].values[0]
        version = info["VERSION"]
        save_figure(fig,f"{root}/images/bkg/rates/{version}/{version}_{name}_generator_distribution", rm=save)
    if show:
        fig.show()
    return fig


def generate_pileup_distribution(fig, pileup_df_dict, color_dict, debug=False):
    for idx, config in enumerate(pileup_df_dict):
        info = json.load(
            open(f"{root}/config/{config}/{config}_config.json", "r")
        )
        pileup_df = pileup_df_dict[config]
        color_list = color_dict[config]
        x_values = pileup_df.columns.values[:-1]  #
        y_values = pileup_df.loc["Neutrino"].values[:-1]
        df = pd.DataFrame({"Counts": y_values}, index=x_values)["Counts"]
        if debug:
            rprint(df)
        fig = plot_generator_distribution(fig, df, color_list, idx=(1, 1), debug=debug)

    fig = format_generator_distribution(fig, info, debug=debug)
    return fig


def plot_generator_distribution(
    fig, df, color_list, idx=(1, 1), add_values=True, debug=False
):
    if debug:
        rprint(df)
    for i in range(len(df.values)):
        fig.add_trace(
            go.Bar(
                name=df.index[i],
                # text=[(df.values)[i]],
                x=[df.index[i]],
                y=[(df.values)[i]],
                marker_color=[color_list[i]],
            ),
            row=idx[0],
            col=idx[1],
        )
        if add_values:
            fig.update_traces(
                texttemplate="%{text:.2E}",
                textposition="inside",
                textfont_size=10,
                opacity=0.75,
            )
    return fig


def format_generator_distribution(fig, info, debug=False):
    fig = format_coustom_plotly(
        fig,
        title="Background generation %s" % (info["VERSION"]),
        legend=dict(orientation="v"),
        figsize=(1800, 1000),
        log=(False, True),
        tickformat=("", ""),
        margin={"auto": False, "margin": (100, 100, 100, 200)},
    )
    fig.update_layout(bargap=0.1)
    fig.update_yaxes(title="Frequency [Hz]")
    return fig
