import json
import numba
import pandas as pd
import numpy as np
import dask.dataframe as dd
import plotly.graph_objects as go
import plotly.express as px

from typing import Optional
from dask import delayed
from rich import print as rprint
from plotly.subplots import make_subplots

from .plt_functions import format_coustom_plotly
from .io_functions import (
    get_branches2use,
    get_bkg_config,
    save_figure
)

from src.utils import get_project_root
root = get_project_root()


def rebin_hist(x: np.ndarray, y: np.ndarray, y_error: np.ndarray, rebin: Optional[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(rebin, int):
        new_x = [np.mean(x[i:i+rebin]) for i in range(0, len(x), rebin)]
        new_y = [np.sum(y[i:i+rebin]) for i in range(0, len(y), rebin)]
        new_y_per_x = [np.sum(y[i:i+rebin])/rebin for i in range(0, len(y), rebin)]
        new_y_error = [np.sqrt(np.sum(y_error[i:i+rebin]**2)) for i in range(0, len(y_error), rebin)]
    
    elif isinstance(rebin, list) or isinstance(rebin, np.ndarray):
        rebin_centers = [np.mean([rebin[i], rebin[i+1]]) for i in range(len(rebin)-1)]
        rebin_widths = [rebin[i+1] - rebin[i] for i in range(len(rebin)-1)]
        new_y = np.zeros(len(rebin_centers))
        new_y_per_x = np.zeros(len(rebin_centers))
        new_y_error = np.zeros(len(rebin_centers))
        
        for i, value in enumerate(rebin_centers):
            idx = np.where((x >= rebin[i]) & (x < rebin[i+1]))[0]
            if rebin[i] > x[-1]:
                break
            new_y[i] = np.sum(y[idx[0]:idx[-1]+1])
            new_y_per_x[i] = np.sum(y[idx[0]:idx[-1]+1])/rebin_widths[i]
            new_y_error[i] = np.sqrt(np.sum(y_error[idx[0]:idx[-1]+1]**2))
        new_x = rebin_centers
    return new_x, new_y, new_y_per_x, new_y_error


def rebin_hist2d(x: np.ndarray, y: np.ndarray, z: np.ndarray, rebin: Optional[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    This function rebins a 2D histogram according to the rebin value/array.
    The rebin value can be an integer or a list/array with the bin edges.
        If the rebin value is an integer, the function will rebin the x and y axis of the 2D histogram.
        -> Changing the number of bins from n*m to n//rebin * m//rebin.
        If the rebin value is a list/array, the function will rebin the x axis of the 2D histogram according to the bin edges of the rebin list/array.
        -> Changing the number of bins from n*m to len(rebin) * m.
    '''
    if isinstance(rebin, int):
        new_x = x[:len(x)//rebin*rebin].reshape(-1, rebin).mean(axis=1)
        new_y = y[:len(y)//rebin*rebin].reshape(-1, rebin).mean(axis=1)
        new_z = z[:len(x)//rebin*rebin, :len(y)//rebin*rebin].reshape(len(x)//rebin, rebin, len(y)//rebin, rebin).sum(axis=(1, 3))
        new_z_per_x = new_z / rebin

    elif isinstance(rebin, list) or isinstance(rebin, np.ndarray):
        rebin_centers = [np.mean([rebin[i], rebin[i+1]]) for i in range(len(rebin)-1)]
        rebin_widths = [rebin[i+1] - rebin[i] for i in range(len(rebin)-1)]
        new_z = np.zeros((len(y), len(rebin_centers)))
        new_z_per_x = np.zeros((len(y), len(rebin_centers)))
        for i, value in enumerate(rebin_centers):
            mask = np.logical_and(x >= rebin[i], x < rebin[i+1])
            new_z[:, i] = np.sum(z[:, mask], axis=1)
            new_z_per_x = new_z / rebin_widths
        
        new_x = rebin_centers
        new_y = y
    return new_x, new_y, new_z, new_z_per_x


def rebin_df_columns(df: pd.DataFrame, rebin: Optional[int], bins: str = "Energy", counts: str = "Counts", counts_per_energy: str = "Counts/Energy", counts_error: str = "Error") -> pd.DataFrame:
    new_df = df.copy()
    if isinstance(rebin, int):
        bins_array = np.zeros((len(df), int(len(df[counts][0])/rebin)))
        count_array = np.zeros((len(df), int(len(df[counts][0])/rebin)))
        count_per_energy_array = np.zeros((len(df), int(len(df[counts][0])/rebin)))
        count_error_array = np.zeros((len(df), int(len(df[counts_error][0])/rebin)))
            
    elif isinstance(rebin, list) or isinstance(rebin, np.ndarray):
        bins_array = np.zeros((len(df), len(rebin)-1))
        count_array = np.zeros((len(df), len(rebin)-1))
        count_per_energy_array = np.zeros((len(df), len(rebin)-1))
        count_error_array = np.zeros((len(df), len(rebin)-1))
    
    for i in range(len(df)):
        bins_array[i], count_array[i], count_per_energy_array[i], count_error_array[i] = rebin_hist(np.asarray(df[bins][i]), np.asarray(df[counts][i]), np.asarray(df[counts_error][i]), rebin)

    new_df[bins] = pd.Series(bins_array.tolist())
    new_df[counts] = pd.Series(count_array.tolist())
    new_df[counts_per_energy] = pd.Series(count_per_energy_array.tolist())
    new_df[counts_error] = pd.Series(count_error_array.tolist())

    return new_df


def explode(df: pd.DataFrame, explode: list[str], keep: Optional[list] = None, debug: bool = False) -> pd.DataFrame:
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
                rprint(
                    f"[yellow]Column {col} not found in the dataframe[/yellow]")
                keep.remove(col)

    # make a copy of explode to avoid modifying the original list
    old_explode = explode.copy()
    for col in old_explode:
        if col not in df.columns:
            rprint(
                f"[yellow]Column {col} not found in the dataframe[/yellow]")
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
        rprint("[green]Dataframe exploded![/green]")
    return result_df


def npy2df(run: dict, tree: Optional[str] = None, branches: Optional[list[str]] = None, debug: bool = False) -> pd.DataFrame:
    """
    Function to convert the dictionary of the TTree into a pandas Dataframe.

    Args:
        run (dict): Dictionary with the data to save (delete the keys that you don't want to save or set Force = False to save only the new ones)
        tree (str): Name of the tree to convert to dataframe.
        branches (list(str)): List of branches to convert to dataframe.
        debug (bool): If True, the debug mode is activated.

    Returns:
        df_dict (dict(pandas.DataFrame)): Dataframe dict with the data of the TTree.
    """
    if tree is None:
        rprint(f"[yellow]Tree not defined. Returning df from root.[/yellow]")
        tree = "Data"
        run = {tree: run}

    elif tree not in run.keys():
        rprint(f"[red]Tree {tree} not found in the dictionary[/red]")
        raise KeyError

    df = pd.DataFrame()
    if branches is None:
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
        try:
            rprint(df.groupby(["Geometry", "Version", "Name"])[
                "Event"].count())
        except KeyError:
            pass
    return df


def dict2df(run: dict, debug: bool = False) -> list[pd.DataFrame]:
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
                rprint(
                    f"[magenta] --- Dataframe for key {key} created\n[/magenta]")
                rprint(f"{df[key]} \n")
        df_list.append(df)

    rprint(f"[green]DataFrame generated from dict![/green]")
    return df_list


def merge_df(df1: pd.DataFrame, df2: pd.DataFrame, label1: str, label2: str, debug: bool = False) -> pd.DataFrame:
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
        rprint(f" --- New dataframe from {label1}, {label2} created")
    return df


def reorder_df(df: pd.DataFrame, info: dict, bkg_dict: dict, color_dict: dict, debug: bool = False) -> tuple[pd.DataFrame, list]:
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
        rprint(f"[cyan]Reordered dataframe with columns: {order}[/cyan]")
    return df, color_list


def generate_truth_dataframe(run: dict, info: dict, fullname: bool = True, debug: bool = False) -> pd.DataFrame:
    """
    Generate a dataframe with the truth information of the run.
    """
    bkg_dict, color_dict = get_bkg_config(info)
    if fullname:
        columns = list(bkg_dict.values())[1:]
    else:
        name_dict = get_simple_name(list(bkg_dict.values())[1:])
        columns = [name_dict[name] for name in list(bkg_dict.values())[1:]]

    truth_gen_df = pd.DataFrame(
        np.asarray(run["Truth"]["TruthPart"])[
            :, 0: len(list(bkg_dict.values())[1:])],
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


def calculate_mean_truth_df(truth_gen_df: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    mean_truth_df = (
        truth_gen_df.drop(
            columns=["Geometry", "Version"]).groupby("Name").mean()
    )
    mean_truth_df = mean_truth_df.replace(0, np.nan).mean()
    return mean_truth_df


def calculate_pileup_df(mean_truth_df: pd.DataFrame, info: dict, factor: float = 1, debug: bool = False) -> pd.DataFrame:
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


def calculate_pileup_df_dict(run: dict[dict], configs: dict[str, list[str]], factor: float = 1, debug: bool = False):
    pileup_df_dict = {}
    color_dict = {}
    for idx, config in enumerate(configs):
        info = json.load(
            open(f"{root}/config/{config}/{config}_config.json", "r"))
        bkg_dict, color_dict = get_bkg_config(info, debug=debug)
        truth_gen_df = generate_truth_dataframe(run, info, debug=debug)
        df = calculate_mean_truth_df(truth_gen_df, debug=debug)
        df, color_list = reorder_df(
            df, info, bkg_dict, color_dict, debug=debug)
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
            fig.write_image(
                f"{root}/images/bkg/{config}_PileUp_Area.png", scale=1.5)
        if show:
            fig.show()
    return fig


def generate_background_distribution(
    run, configs, fullname=True, add_values=True, show=False, save:Optional[str]=None, debug=False
):
    fig = make_subplots(
        rows=len(configs), cols=1, shared_yaxes=False, shared_xaxes=False
    )
    for idx, config in enumerate(configs):
        info = json.load(
            open(f"{root}/config/{config}/{config}_config.json", "r"))
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
    if save is not None:
        name = truth_gen_df["Name"].values[0]
        save_figure(
            fig, save, config, name, f"generator_distribution", rm=True, debug=debug)
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
        fig = plot_generator_distribution(
            fig, df, color_list, idx=(1, 1), debug=debug)

    fig = format_generator_distribution(fig, info, debug=debug)
    return fig


def plot_generator_distribution(
    fig, df: pd.DataFrame, color_list: list, idx: tuple[int, int] = (1, 1), add_values: bool = True, debug: bool = False
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


def format_generator_distribution(fig, info: dict, debug: bool = False):
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
