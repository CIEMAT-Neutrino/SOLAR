import os, glob, uproot, json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from scipy import stats as sps
from scipy.stats import chi2
from scipy import interpolate
from itertools import product
from rich import print as rprint
from plotly.subplots import make_subplots

from lib.io_functions import print_colored
from lib.plt_functions import format_coustom_plotly, unicode


def get_nadir_angle(
    path: str = "../data/OSCILLATION/", show: bool = False, debug: bool = False
):
    """
    This function can be used to obtain the nadir angle distribution for DUNE.

    Args:
        path (str): Path to the nadir angle data file (default: "../data/OSCILLATION/")
        show (bool): If True, show the plot (default: False)
        debug (bool): If True, the debug mode is activated.

    Returns:
        (xnadir_centers,ynadir_centers): tuple containing the nadir angle and the PDF.
    """
    with uproot.open(path + "nadir.root") as nadir:
        # Loas pdf histogram
        pdf = nadir["nadir;1"]
        pdf_array = pdf.to_hist().to_numpy()
        xbin_edges = pdf_array[1]
        xnadir_centers = 0.5 * (xbin_edges[1:] + xbin_edges[:-1])
        ynadir_centers = pdf_array[0]

    return (xnadir_centers, ynadir_centers)


def plot_nadir_angle(fig, idx, debug: bool = False):
    """
    This function can be used to plot the nadir angle distribution for DUNE.

    Args:
        fig (plotly.graph_objects.Figure): Plotly figure.
        idx (tuple(int)): Index of the subplot.
        debug (bool): If True, the debug mode is activated.

    Returns:
        fig (plotly.graph_objects.Figure): Plotly figure.
    """
    analysis_info = json.load(open("../import/analysis.json", "r"))
    nadir_data = get_nadir_angle(show=False, debug=debug)
    fig.add_trace(
        px.scatter(
            x=nadir_data[0],
            y=nadir_data[1],
            labels={"x": "Nadir Angle cos(" + unicode("eta") + ")", "y": "PDF"},
        ).data[0],
        row=idx[0],
        col=idx[1],
    )
    fig.update_xaxes(
        title_text="Nadir Angle cos(" + unicode("eta") + ")", row=idx[0], col=idx[1]
    )
    fig.update_yaxes(title_text="PDF", row=idx[0], col=idx[1])
    # fig = format_coustom_plotly(
    #     fig,
    #     tickformat=(".1f", ".0e"),
    #     ranges=(None, [4e-4, 1.2e-3]),
    # )

    return fig


def get_oscillation_datafiles(
    dm2=None,
    sin13=None,
    sin12=None,
    path: str = "../data/OSCILLATION/",
    ext: str = "root",
    auto: bool = False,
    debug: bool = False,
):
    """
    This function can be used to obtain the oscillation data files for DUNE's solar analysis.

    Args:
        dm2 (float/list):   Solar mass squared difference (default: analysis["DM2"])
        sin13 (float/list): Solar mixing angle (default: analysis["SIN13"])
        sin12 (float/list): Solar mixing angle (default: analysis["SIN12"])
        path (str):   Path to the oscillation data files (default: "../data/OSCILLATION/")
        ext (str):    Extension of the oscillation data files (default: "root")
        auto (bool):  If True, the function will look for all the oscillation data files in the path (default: False)
        debug (bool): If True, the debug mode is activated.

    Returns:
        (found_dm2,found_sin13,found_sin12): tuple containing the dm2, sin13 and sin12 values of the found oscillation data files.
    """
    if auto:
        data_files = glob.glob(path + "*_dm2_*_sin13_*_sin12_*")
        string_dm2, trash, string_sin13, trash, string_sin12 = zip(
            *[
                tuple(
                    map(
                        str,
                        os.path.basename(osc_file).split("." + ext)[0].split("_")[-5:],
                    )
                )
                for osc_file in data_files
            ]
        )
        found_dm2 = [float(i) for i in string_dm2]
        found_sin13 = [float(i) for i in string_sin13]
        found_sin12 = [float(i) for i in string_sin12]

    if auto == False:
        if type(dm2) == list and type(sin13) == list and type(sin12) == list:
            found_dm2, found_sin13, found_sin12 = [], [], []
            for this_dm2, this_sin13, this_sin12 in zip(dm2, sin13, sin12):
                if os.path.isfile(
                    path
                    + "osc_probability_dm2_%.3e_sin13_%.3e_sin12_%.3e"
                    % (this_dm2, this_sin13, this_sin12)
                    + "."
                    + ext
                ):
                    found_dm2.append(this_dm2)
                    found_sin13.append(this_sin13)
                    found_sin12.append(this_sin12)
                else:
                    print_colored(
                        "WARNING: file %sosc_probability_dm2_%.3e_sin13_%.3e_sin12_%.3e"
                        % (path, this_dm2, this_sin13, this_sin12)
                        + "."
                        + ext
                        + " not found!",
                        "WARNING",
                    )

        elif type(dm2) == float and type(sin13) == float and type(sin12) == float:
            if os.path.isfile(
                path
                + "osc_probability_dm2_%.3e_sin13_%.3e_sin12_%.3e" % (dm2, sin13, sin12)
                + "."
                + ext
            ):
                found_dm2 = [dm2]
                found_sin13 = [sin13]
                found_sin12 = [sin12]
            else:
                print_colored(
                    "WARNING: file %sosc_probability_dm2_%.3e_sin13_%.3e_sin12_%.3e"
                    % (path, this_dm2, this_sin13, this_sin12)
                    + "."
                    + ext
                    + " not found! Returning default.",
                    "WARNING",
                )
                dm2, sin13, sin12 = None, None, None

        elif dm2 == None and sin13 == None and sin12 == None:
            analysis_info = json.load(open("../import/analysis.json", "r"))
            found_dm2, found_sin13, found_sin12 = (
                analysis_info["REACT_DM2"],
                analysis_info["SIN13"],
                analysis_info["SIN12"],
            )

        else:
            print_colored(
                "ERROR: oscillation parameters must be floats or lists!", "ERROR"
            )
            raise TypeError

    if type(auto) != bool:
        print_colored("ERROR: auto must be a boolean!", "FAIL")
        raise TypeError

    if type(found_dm2) == list:
        if debug:
            print_colored("Found %d oscillation files!" % len(found_dm2), "INFO")
    if type(found_dm2) == float:
        found_dm2, found_sin13, found_sin12 = [found_dm2], [found_sin13], [found_sin12]
        if debug:
            print_colored("Found 1 oscillation file!", "INFO")

    return (found_dm2, found_sin13, found_sin12)


def get_oscillation_map(
    path="../data/OSCILLATION/",
    dm2=None,
    sin13=None,
    sin12=None,
    ext="root",
    auto=False,
    rebin=False,
    output="df",
    save=False,
    debug=False,
):
    """
    This function can be used to obtain the oscillation correction for DUNE's solar analysis.

    Args:
        path (str): Path to the oscillation data files (default: "../data/OSCILLATION/")
        dm2 (float): Solar mass squared difference (default: "DEFAULT")
        sin13 (float): Solar mixing angle (default: "DEFAULT")
        sin12 (float): Solar mixing angle (default: "DEFAULT")
        auto (bool): If True, the function will look for all the oscillation data files in the path (default: True)
        rebin (bool): If True, rebin the oscillation map (default: True)
        output (str): If "interp", the function will return the interpolation dictionary. If "df", the function will return the dataframe dictionary (default: "interp")
        save (bool): If True, save the rebin dataframes (default: False)
        show (bool): If True, show the oscillation map (default: False)
        ext (str): Extension of the oscillation data files (default: "root")
        debug (bool): If True, the debug mode is activated.

    Returns:
        interp_dict (dict): Dictionary containing the interpolation functions for each dm2, sin13 and sin12 value.
    """

    df_dict = {}
    interp_dict = {}
    # nadir_data = get_nadir_angle(path=path, show=False, debug=debug)
    subfolder = ""
    if ext == "pkl":
        if rebin == True:
            subfolder = "rebin"
        else:
            subfolder = "raw"

    dm2, sin13, sin12 = get_oscillation_datafiles(
        dm2,
        sin13,
        sin12,
        path=path + ext + "/" + subfolder + "/",
        ext=ext,
        auto=auto,
        debug=debug,
    )
    analysis_info = json.load(open("../import/analysis.json", "r"))

    if type(dm2) == float and type(sin13) == float and type(sin12) == float:
        dm2, sin13, sin12 = [dm2], [sin13], [sin12]

    for dm2_value, sin13_value, sin12_value in zip(dm2, sin13, sin12):
        # Format dm2 and sin12 values to be used in the file name with appropriate precision
        dm2_value = float("%.3e" % dm2_value)
        sin12_value = float("%.3e" % sin12_value)

        if ext == "pkl":
            if (
                glob.glob(
                    path
                    + ext
                    + "/"
                    + subfolder
                    + "/"
                    + "osc_probability_dm2_%.3e_sin13_%.3e_sin12_%.3e.pkl"
                    % (dm2_value, sin13_value, sin12_value)
                )
                != []
            ):
                if debug:
                    print_colored(
                        "Loading data from: "
                        + path
                        + ext
                        + "/"
                        + subfolder
                        + "/"
                        + "osc_probability_dm2_%.3e_sin13_%.3e_sin12_%.3e.pkl"
                        % (dm2_value, sin13_value, sin12_value),
                        "DEBUG",
                    )
                df = pd.read_pickle(
                    path
                    + ext
                    + "/"
                    + subfolder
                    + "/"
                    + "osc_probability_dm2_%.3e_sin13_%.3e_sin12_%.3e.pkl"
                    % (dm2_value, sin13_value, sin12_value)
                )

                if rebin:
                    save_path = (
                        path
                        + ext
                        + "/rebin/osc_probability_dm2_%.3e_sin13_%.3e_sin12_%.3e.pkl"
                        % (dm2_value, sin13_value, sin12_value)
                    )
                    if glob.glob(save_path) != []:
                        if debug:
                            print_colored(
                                "Loading rebinned data from %s" % save_path, "DEBUG"
                            )
                        df = pd.read_pickle(save_path)
                    else:
                        df = rebin_df(
                            df, show=False, save=save, save_path=save_path, debug=debug
                        )

            else:
                print_colored(
                    "ERROR: file %sosc_probability_dm2_%.3e_sin13_%.3e_sin12_%.3e.pkl"
                    % (path, dm2_value, sin13_value, sin12_value)
                    + " not found!",
                    "FAIL",
                )
                return None
            # if rebin == False and glob.glob(path+ext+'/'+subfolder+'/'+"osc_probability_dm2_%.3e_sin13_%.3e_sin12_%.3e.pkl"%(dm2_value,sin13_value,sin12_value)) != []:
            #     if debug: print_colored("Loading rebin data from: "+path+ext+'/'+subfolder+'/'+"osc_probability_dm2_%.3e_sin13_%.3e_sin12_%.3e.pkl"%(dm2_value,sin13_value,sin12_value),"DEBUG")
            #     df = pd.read_pickle(path+ext+'/'+subfolder+'/'+"osc_probability_dm2_%.3e_sin13_%.3e_sin12_%.3e.pkl"%(dm2_value,sin13_value,sin12_value))

        elif ext == "root":
            if debug:
                print_colored(
                    "Loading raw data from: "
                    + path
                    + "/root/osc_probability_dm2_%.3e_sin13_%.3e_sin12_%.3e%s"
                    % (dm2_value, sin13_value, sin12_value, "." + ext),
                    "DEBUG",
                )

            df = process_oscillation_map(
                path,
                dm2_value,
                sin13_value,
                sin12_value,
                ext=ext,
                convolve=True,
                debug=debug,
            )

            if rebin:
                save_path = (
                    path
                    + "/pkl/rebin/osc_probability_dm2_%.3e_sin13_%.3e_sin12_%.3e.pkl"
                    % (dm2_value, sin13_value, sin12_value)
                )
                if glob.glob(save_path) != []:
                    if debug:
                        print_colored(
                            "Loading rebinned data from %s" % save_path, "DEBUG"
                        )
                    df = pd.read_pickle(save_path)
                else:
                    df = rebin_df(
                        df, show=False, save=save, save_path=save_path, debug=debug
                    )

        else:
            print_colored("ERROR: ext must be 'root' or 'pkl'!", "FAIL")
            return None

        if output in ["interp1d", "interp2d"]:
            osc_map_x = df.loc[df.index[0], :].keys().to_list()
            osc_map_y = df.loc[:, df.columns[0]].keys().to_list()
            if output == "interp1d":
                oscillation_map = interpolate.interp1d(
                    osc_map_x,
                    np.sum(df.values, axis=0),
                    kind="linear",
                    fill_value="extrapolate",
                )
            if output == "interp2d":
                oscillation_map = interpolate.RegularGridInterpolator(
                    (osc_map_x, osc_map_y),
                    df.loc[:, :].to_numpy().T,
                    method="linear",
                    bounds_error=False,
                    fill_value=1e-6,
                )
            interp_dict[(dm2_value, sin13_value, sin12_value)] = oscillation_map

        df_dict[(dm2_value, sin13_value, sin12_value)] = df

    if output == "figure":
        fig = px.imshow(
            df, color_continuous_scale="turbo", origin="lower", aspect="auto"
        )
        fig = format_coustom_plotly(fig, figsize=(800, 600))
        return fig

    elif output in ["interp1d", "interp2d"]:
        if debug:
            print_colored("Returning interpolation dictionary!", "DEBUG")
        return interp_dict

    elif output == "df":
        if debug:
            print_colored("Returning dataframe dictionary!", "DEBUG")
        return df_dict

    else:
        print_colored("ERROR: output must be 'interp' or 'df'!", "FAIL")
        return None


def process_oscillation_map(
    path="../data/OSCILLATION/",
    dm2_value=None,
    sin13_value=None,
    sin12_value=None,
    ext="root",
    convolve=True,
    debug=False,
):
    nadir_data = get_nadir_angle(path=path, debug=debug)
    analysis_info = json.load(open("../import/analysis.json", "r"))
    root_nadir_edges = np.linspace(
        analysis_info["ROOT_NADIR_RANGE"][0],
        analysis_info["ROOT_NADIR_RANGE"][1],
        analysis_info["ROOT_NADIR_BINS"] + 1,
    )
    root_nadir_centers = (root_nadir_edges[1:] + root_nadir_edges[:-1]) / 2
    if dm2_value == None and sin13_value == None and sin12_value == None:
        dm2_value, sin13_value, sin12_value = (
            analysis_info["REACT_DM2"],
            analysis_info["SIN13"],
            analysis_info["SIN12"],
        )
    data = uproot.open(
        path
        + "/root/osc_probability_dm2_%.3e_sin13_%.3e_sin12_%.3e%s"
        % (dm2_value, sin13_value, sin12_value, "." + ext)
    )
    # Convert the histogram to a pandas DataFrame
    hist = data["hsurv;1"]  # Load the 2D histogram
    data_array = hist.to_hist().to_numpy()

    # Data contains the bin edges and the bin contents
    data = data_array[0][:, :-1]
    root_energy_edges = data_array[1]
    root_energy_centers = 0.5 * (root_energy_edges[1:] + root_energy_edges[:-1])
    root_nadir_edges = data_array[2][:-1]
    root_nadir = 0.5 * (root_nadir_edges[1:] + root_nadir_edges[:-1])

    # Create a DataFrame with the bin contents
    df1 = pd.DataFrame(data, index=1e3 * root_energy_centers, columns=root_nadir)
    df2 = pd.DataFrame(
        data_array[0][:, -1][:, np.newaxis]
        * np.ones((len(root_energy_centers), len(root_nadir))),
        index=1e3 * root_energy_centers,
        columns=1 + root_nadir,
    )
    df = df1.join(df2).T

    if convolve:
        # Interpolate nadir data to match ybins
        nadir = interpolate.interp1d(
            nadir_data[0], nadir_data[1], kind="linear", fill_value="extrapolate"
        )
        nadir_y = nadir(x=root_nadir_centers)
        # normalize nadir distribution
        nadir_y = nadir_y / nadir_y.sum()
        df = df.mul(nadir_y, axis=0)

    return df


def plot_oscillation_map(fig, idx, dm2=None, sin13=None, sin12=None, debug=False):
    """
    This function can be used to plot the oscillation map for DUNE's solar analysis.

    Args:
        fig (plotly.graph_objects.Figure): Plotly figure.
        idx (tuple(int)): Index of the subplot.
        dm2 (float): Solar mass squared difference (default: None)
        sin13 (float): Solar mixing angle (default: None)
        sin12 (float): Solar mixing angle (default: None)
        debug (bool): If True, the debug mode is activated.

    Returns:
        fig (plotly.graph_objects.Figure): Plotly figure.
    """
    analysis_info = json.load(open("../import/analysis.json", "r"))
    oscillation_map = get_oscillation_map(
        dm2=dm2, sin13=sin13, sin12=sin12, debug=debug
    )
    for this_dm2, this_sin13, this_sin12 in oscillation_map.keys():
        df = oscillation_map[(this_dm2, this_sin13, this_sin12)]
        fig.add_trace(
            go.Heatmap(
                z=df,
                x=df.columns,
                y=df.index,
                colorscale="turbo",
                # coloraxis="coloraxis" + str(idx[1]),
                colorbar=dict(title="Osc. PDF"),
                colorbar_x=1,
            ),
            row=idx[0],
            col=idx[1],
        )

    if debug:
        rprint("Oscillation map plotted!")
    return fig


def rebin_df(
    df,
    save_path="../data/pkl/rebin/df.pkl",
    xarray=[],
    yarray=[],
    show=False,
    save=True,
    debug=False,
):
    """
    This function can be used to rebin any dataframe that has a 2D index (like an imshow dataset).

    Args:
        df (pandas.DataFrame): Dataframe to rebin.
        xarray (list): List of xbins to use for the rebinning.
        yarray (list): List of ybins to use for the rebinning.
        show (bool): If True, show the rebinning result (default: False)
        save (bool): If True, save the rebinning result (default: True)
        save_path (str): Path to save the rebinning result (default: "../data/pkl/rebin/df.pkl")
        debug (bool): If True, the debug mode is activated.

    Returns:
        small_df (pandas.DataFrame): Rebinning result.
    """
    analysis_info = json.load(open("../import/analysis.json", "r"))
    energy_edges = np.linspace(
        analysis_info["RECO_ENERGY_RANGE"][0],
        analysis_info["RECO_ENERGY_RANGE"][1],
        analysis_info["RECO_ENERGY_BINS"] + 1,
    )
    energy_centers = (energy_edges[1:] + energy_edges[:-1]) / 2

    df.index = df.index.astype(float)
    if xarray == [] and yarray == []:
        reduced_rows_edges = np.linspace(-1, 1, 40 + 1, endpoint=True)
        reduced_rows = 0.5 * (reduced_rows_edges[1:] + reduced_rows_edges[:-1])
        reduced_rows = np.round(reduced_rows, 4)
        if debug:
            print_colored("Rebinning data with default parameters!", "DEBUG")
    else:
        energy_centers = xarray
        reduced_rows = yarray
        if debug:
            print_colored("Rebinning data with custom parameters!", "DEBUG")

    # Create an empty reduced data frame
    goal_int = df.sum().mean()
    small_df = pd.DataFrame(index=reduced_rows, columns=energy_centers)

    for col in energy_centers:
        for row in reduced_rows:
            # Calculate the average of the original data within the corresponding range
            step_col = energy_centers[1] - energy_centers[0]
            start_col = float(col) - step_col / 2
            end_col = float(col) + step_col / 2

            step_row = reduced_rows[1] - reduced_rows[0]
            start_row = round(float(row) - step_row / 2, 4)
            end_row = round(float(row) + step_row / 2, 4)

            small_df.loc[row, col] = (
                df.loc(axis=1)[start_col:end_col]
                .loc(axis=0)[start_row:end_row]
                .sum()
                .mean()
            )
    # Substitute NaN values with 0
    small_df = small_df.fillna(0)

    # Print the reduced data frame
    if show:
        fig = px.imshow(
            small_df,
            aspect="auto",
            origin="lower",
            color_continuous_scale="turbo",
            title="Oscillation Correction Map",
            labels=dict(y="Nadir Angle (°)", x="TrueEnergy"),
        )
        fig = format_coustom_plotly(fig, figsize=(800, 600))
        fig.show()

    if save:
        small_df.to_pickle(save_path)

    return small_df


def compute_log_likelihood(pred_df, fake_df, method="log-likelihood", debug=False):
    """
    This function can be used to compute the log likelihood of a prediction given a fake data set.

    Args:
        pred_df (pandas.DataFrame): Prediction dataframe.
        fake_df (pandas.DataFrame): Fake dataframe.
        method (str): Method to compute the log likelihood (default: "log-likelihood")
        debug (bool): If True, the debug mode is activated.

    Returns:
        ll (float): Log likelihood.
    """

    if method == "log-likelihood":
        ll = 0
        for col in fake_df.columns:
            for row in fake_df.index:
                if pred_df.loc[row, col] == 0:
                    continue
                if fake_df.loc[row, col] == 0:
                    this_ll = pred_df.loc[row, col] - fake_df.loc[row, col]
                    if np.isnan(this_ll):
                        this_ll = 0
                    ll = ll + this_ll
                else:
                    this_ll = (
                        pred_df.loc[row, col]
                        - fake_df.loc[row, col]
                        + fake_df.loc[row, col]
                        * np.log(fake_df.loc[row, col] / pred_df.loc[row, col])
                    )
                    if np.isnan(this_ll):
                        this_ll = 0
                    ll = ll + this_ll
        ll = 2 * ll
        chi_square = ll

    if debug:
        print_colored("Chi-square computed!", "DEBUG")
    return chi_square


def make_oscillation_map_plot(dm2=None, sin13=None, sin12=None, debug=False):
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "DUNE's FD Yearly Nadir Angle",
            "Survival Probability",
            "Convolved Probability",
        ),
    )

    fig = plot_nadir_angle(fig, (1, 1), debug=True)
    fig.update_xaxes(title_text="Nadir Angle cos(" + unicode("eta") + ")", row=1, col=1)
    fig.update_yaxes(range=[0.0004, 0.0013], title_text="PDF", row=1, col=1)

    df = process_oscillation_map(
        dm2_value=dm2, sin13_value=sin13, sin12_value=sin12, convolve=False, debug=debug
    )
    fig.add_trace(
        go.Heatmap(
            z=df,
            y=df.index,
            x=df.columns,
            colorscale="turbo",
            colorbar=dict(title="Prob."),
            colorbar_x=0.645,
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="True Neutrino Energy [MeV]", row=1, col=2)
    fig.update_yaxes(title_text="Nadir Angle con(" + unicode("eta") + ")", row=1, col=2)

    fig = plot_oscillation_map(
        fig, (1, 3), dm2=dm2, sin13=sin13, sin12=sin12, debug=debug
    )
    fig.update_xaxes(title_text="True Neutrino Energy [MeV]", row=1, col=3)
    fig.update_coloraxes(colorbar=dict(title="Events"), colorbar_x=0.9, row=1, col=3)

    fig = format_coustom_plotly(
        fig, figsize=(2100, 600), matches=(None, None), tickformat=(None, None)
    )

    return fig
