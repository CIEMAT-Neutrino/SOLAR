import os
import plotly
import numba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from scipy import interpolate
from particle import Particle
from itertools import product
from rich.progress import track
from scipy import constants as const
from rich import print as rprint

from lib.io_functions import (
    print_colored,
    get_bkg_config,
    get_gen_label,
    get_gen_weights,
)
from lib.osc_functions import get_nadir_angle, get_oscillation_datafiles


def compute_solar_spectrum(
    run,
    info,
    configs,
    config,
    names,
    name,
    gen,
    energy_edges,
    int_time,
    filters,
    truth_filter,
    reco_filter,
    factor="SOLAR",
    input_dm2=None,
    input_sin13=None,
    input_sin12=None,
    auto=False,
    save=False,
    debug=False,
):
    """
    Get the weighted spectrum for a given background and configuration

    Args:
        run (dict): dictionary with the run data
        info (dict): dictionary with the run info
        configs (dict): dictionary with the config files
        config (str): name of the config
        names (dict): dictionary with the names
        name (str): name of the background
        gen (int): generator of the background
        energy_edges (np.array): energy edges
        int_time (int): integration time
        filters (list): list of filters
        truth_filter (str): truth filter
        reco_filter (str): reco filter
        factor (str): factor to scale the spectrum (default: "SOLAR")
        input_dm2 (float): input dm2 value (default: analysis["SOLAR_DM2"])
        input_sin13 (float): input sin13 value (default: analysis["SIN13"])
        input_sin12 (float): input sin12 value (default: analysis["SIN12"])
        auto (bool): if True, use the auto mode (default: False)
        save (bool): if True, save the output (default: False)
        debug (bool): if True, print debug messages (default: False)

    Returns:
        dict_array (list): list of dictionaries with the weighted spectrum
        weighted_df_dict (dict): dictionary with the weighted spectrum
    """
    dict_array = []
    weighted_df_dict = {}
    energy_centers = 0.5 * (energy_edges[1:] + energy_edges[:-1])
    gen_label_dict = get_gen_label(configs)
    gen_label = gen_label_dict[(info["GEOMETRY"], info["VERSION"], gen)]
    gen_weigths_dict = get_gen_weights(configs, names)
    nadir = get_nadir_angle(show=False, debug=debug)

    if factor == "SOLAR":
        factor = 40 * 60 * 60 * 24 * 365
    else:
        factor = 1

    smearing_df = pd.read_pickle(
        "../config/" + config + "/" + config + "_calib/" + config + "_smearing.pkl"
    )
    for ldx, this_filter in enumerate(filters[0]):
        if debug:
            print_colored("Filtering: %s" % (filters[1][ldx]), "DEBUG")
        if gen == 1:
            int_time = 1
            t_hist, bin_edges = np.histogram(
                run["Truth"]["TNuE"][(truth_filter)], bins=energy_edges
            )
            r_hist, bin_edges = np.histogram(
                run["Reco"]["TNuE"][this_filter], bins=energy_edges
            )
            efficient_flux = {A: B for A, B in zip(energy_centers, r_hist / t_hist)}

            eff_smearing_df = smearing_df.mul(efficient_flux) * factor
            eff_smearing_df = eff_smearing_df.replace(np.nan, 0)

            if not os.path.exists("../sensitivity/" + config + "/" + name + "/"):
                os.makedirs("../sensitivity/" + config + "/" + name + "/")
            eff_smearing_df.to_pickle(
                "../sensitivity/" + config + "/" + name + "/eff_smearing.pkl"
            )

            this_dm2, this_sin13, this_sin12 = None, None, None
            this_auto = False
            if ldx == len(filters[0]) - 1:
                this_dm2 = input_dm2
                this_sin13 = input_sin13
                this_sin12 = input_sin12
                this_auto = auto

            (dm2_list, sin13_list, sin12_list) = get_oscillation_datafiles(
                dm2=this_dm2,
                sin13=this_sin13,
                sin12=this_sin12,
                ext="pkl",
                auto=this_auto,
                debug=debug,
            )
            # for dm2,sin13,sin12 in zip(dm2_list,sin13_list,sin12_list):
            for i in track(
                range(len(dm2_list)), description="Computing oscillation maps..."
            ):
                dm2 = dm2_list[i]
                sin13 = sin13_list[i]
                sin12 = sin12_list[i]
                oscillation_df = pd.read_pickle(
                    "../data/OSCILLATION/pkl/rebin/osc_probability_dm2_%.3e_sin13_%.3e_sin12_%.3e.pkl"
                    % (dm2, sin13, sin12)
                )

                for column in eff_smearing_df.columns:
                    if column not in oscillation_df.columns:
                        eff_smearing_df = eff_smearing_df.drop(columns=column)
                        eff_smearing_df = eff_smearing_df.drop(index=column)

                weighted_df = pd.DataFrame(
                    columns=oscillation_df.columns, index=oscillation_df.index
                )
                for row in weighted_df.index:
                    for col in weighted_df.columns:
                        weighted_df.loc[row, col] = np.sum(
                            oscillation_df.loc[row, :].to_numpy()
                            * eff_smearing_df.T.loc[:, col].to_numpy()
                        )

                total_counts = weighted_df.sum().to_list()
                weighted_df_dict[
                    (np.around(dm2, 7), np.around(sin13, 5), np.around(sin12, 4))
                ] = weighted_df

                if save and ldx == len(filters[0]) - 1:
                    if not os.path.exists(
                        "../sensitivity/" + config + "/" + name + "/" + gen_label + "/"
                    ):
                        os.makedirs(
                            "../sensitivity/"
                            + config
                            + "/"
                            + name
                            + "/"
                            + gen_label
                            + "/"
                        )
                    weighted_df.to_pickle(
                        "../sensitivity/"
                        + config
                        + "/"
                        + name
                        + "/"
                        + gen_label
                        + "/solar_events_dm2_%.3e_sin13_%.3e_sin12_%.3e.pkl"
                        % (dm2, sin13, sin12)
                    )

        else:
            t_hist, bin_edges = np.histogram(
                run["Reco"][info["DEFAULT_ANALYSIS_ENERGY"]][(reco_filter)],
                bins=energy_edges,
            )
            r_hist, bin_edges = np.histogram(
                run["Reco"][info["DEFAULT_ANALYSIS_ENERGY"]][this_filter],
                bins=energy_edges,
            )
            efficient_flux = {A: B for A, B in zip(energy_centers, r_hist / t_hist)}
            raised_activity = gen_weigths_dict[(info["GEOMETRY"], name)]
            weight = np.ones(len(r_hist)) / (raised_activity * int_time)
            # if debug: print_colored("Weight for %s with filter %s: %.2e"%(name,filters[1][ldx],raised_activity),color="INFO")
            r_hist = r_hist * info["FULL_DETECTOR_FACTOR"] * factor * weight
            total_counts = r_hist.tolist()

            if ldx == len(filters[0]) - 1:
                reduced_rows_edges = np.linspace(-1, 1, 41, endpoint=True)
                reduced_rows = 0.5 * (reduced_rows_edges[1:] + reduced_rows_edges[:-1])
                reduced_rows = np.round(reduced_rows, 4)

                # Interpolate nadir data to match ybins
                interp_nadir = interpolate.interp1d(
                    nadir[0], nadir[1], kind="linear", fill_value=0
                )
                nadir_y = interp_nadir(reduced_rows)
                nadir_y = nadir_y / np.sum(nadir_y)
                # Make a list with n cpopies of reco_hist
                data = [total_counts] * len(reduced_rows)
                weighted_df = pd.DataFrame(
                    data, index=reduced_rows, columns=energy_centers
                ).mul(nadir_y, axis=0)
                # If output folder does not exist, create it
                if not os.path.exists(
                    "../sensitivity/" + config + "/" + name + "/" + gen_label + "/"
                ):
                    os.makedirs(
                        "../sensitivity/" + config + "/" + name + "/" + gen_label + "/"
                    )
                if save:
                    weighted_df.to_pickle(
                        "../sensitivity/"
                        + config
                        + "/"
                        + name
                        + "/"
                        + gen_label
                        + "/%s_events.pkl" % (gen_label)
                    )
                weighted_df_dict[(None, None, None)] = weighted_df

        # if debug:print_colored("Total counts for %s with filter %s: %.2e"%(gen_label,filters[1][ldx],np.sum(total_counts)),color="INFO")

        this_dict_array = {
            "Geometry": info["GEOMETRY"],
            "Version": info["VERSION"],
            "Name": name,
            "Generator": gen,
            "GenLabel": gen_label,
            "Time": int_time,
            "Filter": filters[1][ldx],
            "Efficiency": list(efficient_flux.values()),
            "TotalCounts": total_counts,
            "Energy": energy_centers.tolist(),
        }

        dict_array.append(this_dict_array)

    return dict_array, weighted_df_dict


def get_truth_count(run, info, config, names, debug=False):
    """
    Get the truth count for a given background and configuration
    """
    bkg_dict, color_dict = get_bkg_config(info)
    truth_gen_df = pd.DataFrame(
        np.asarray(run["Truth"]["TruthPart"])[:, 0 : len(run["Truth"]["TruthPart"][0])],
        columns=list(bkg_dict.values())[1 : len(run["Truth"]["TruthPart"][0]) + 1],
    )
    truth_gen_df["Geometry"] = info["GEOMETRY"]
    truth_gen_df["Version"] = info["VERSION"]
    truth_gen_df["Name"] = run["Truth"]["Name"]
    truth_gen_df = truth_gen_df[
        (truth_gen_df["Geometry"] == info["GEOMETRY"])
        & (truth_gen_df["Version"] == info["VERSION"])
    ]

    count_truth_df = truth_gen_df.groupby("Name").count().drop(columns=["Geometry"])
    if "wbkg" not in names[config]:
        mask = count_truth_df.index.values == count_truth_df.columns.values[:, None]
        count_truth_df = count_truth_df.where(mask.T).mean().replace(np.nan, 0)
    count_truth_df["Unknown"] = 0
    return count_truth_df


def get_pdg_name(unique_value_list, debug=False):
    """
    Get the name for each pdg.
    """
    pdg_dict = dict()
    for pdg in unique_value_list:
        pdg_dict[pdg] = Particle.from_pdgid(pdg).name
    return pdg_dict


def get_pdg_color(pdg_list, debug=False):
    """
    Get the color for each pdg.
    """
    default_color_dict = {11: "orange", 22: "blue", 2112: "green", 2212: "gray"}
    if type(pdg_list) == int:
        try:
            return default_color_dict[pdg_list]
        except:
            return "grey"

    else:
        color_dict = dict()
        for pdg in pdg_list:
            if pdg == 11:
                color_dict[pdg] = "blue"
            elif pdg == 12:
                color_dict[pdg] = "red"
            elif pdg == 22:
                color_dict[pdg] = "purple"
            elif pdg == 2112:
                color_dict[pdg] = "green"
            elif pdg == 2212:
                color_dict[pdg] = "orange"
            else:
                color_dict[pdg] = "grey"
        return color_dict


def get_solar_weigths(weights="BS05"):
    """
    Get the solar flux weights.
    """
    if weights == "BS05":
        weights_dict = {
            "pp": 5.991e00,
            "pep": 1e-10,
            "b7": 1e-10,
            "n13": 3.066e-02,
            "o15": 2.331e-02,
            "f17": 5.836e-04,
            "b8": 5.691e-04,
            "hep": 7.930e-07,
        }  # Flux amp of each component
        return weights_dict
    else:
        print("ERROR: Weights not defined, using BS05!")
        weights_dict = {
            "pp": 5.991e00,
            "pep": 1e-10,
            "b7": 1e-10,
            "n13": 3.066e-02,
            "o15": 2.331e-02,
            "f17": 5.836e-04,
            "b8": 5.691e-04,
            "hep": 7.930e-07,
        }  # Flux amp of each component
        return weights_dict


def get_solar_colors(source):
    """
    Use plotly colors to color the solar flux components.
    """
    colors = plotly.colors.qualitative.Prism
    # source_list = ["pp","pep","b7","n13","o15","f17","b8","hep"]    # Flux amp of each component
    source_list = list(get_solar_weigths().keys())
    for idx, this_source in enumerate(source_list):
        if this_source == source:
            return colors[idx]


def read_solar_data(in_path, source, weigths):
    """
    Read in the solar flux data and interpolate it to the desired energy bins.
    """
    data = get_solar_weigths(weigths)
    energy = []
    flux = []
    factor = 1e10 * data[source]
    text = open(in_path + source + ".dat", "r")
    lines = text.readlines()

    for j in range(len(lines)):
        values = lines[j].split()
        energy.append(float(values[0]))
        flux.append(float(values[1]) * factor)

    return np.array([energy, flux])


def get_solar_spectrum(
    components,
    bins,
    weigths="BS05",
    interpolation=(0, 0),
    show=False,
    out=False,
    in_path="../data/SOLAR/",
    out_path="../data/OUTPUT/",
    debug=False,
):
    """
    Read in the solar flux data and interpolate it to the desired energy bins.

    Args:
        components (list): list of components
        bins (np.array): energy bins
        weigths (str): weigths (default: "BS05")
        interpolation (Any): interpolation method from scipy.interpolate.interp1d (default: (0,0))
        show (bool): if True, show the plot (default: False)
        out (bool): if True, save the output (default: False)
        in_path (str): input path (default: "../data/SOLAR/")
        out_path (str): output path (default: "../data/OUTPUT/")
        debug (bool): if True, print debug messages (default: False)

    Returns:
        y (np.array): interpolated flux values
    """

    if out:
        output = open(out_path + "neutrino_flux.txt", "w")  # Output text file.
    x = bins
    y = np.zeros(len(x))  # Array that will host the interpolated flux values.

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    for idx, source in enumerate(components):
        array = read_solar_data(in_path, source, weigths)
        if source != "pep" and source != "b7":
            func = interpolate.interp1d(
                array[0], array[1], kind="cubic", bounds_error=False, fill_value=0
            )
            y = y + func(x)

        if show:
            if source != "b7":
                fig.add_trace(
                    go.Scatter(
                        x=array[0],
                        y=array[1],
                        name=source,
                        line=dict(color=get_solar_colors(source)),
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=array[0][:2],
                        y=array[1][:2],
                        name=source,
                        line=dict(color=get_solar_colors(source)),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=array[0][2:],
                        y=array[1][2:],
                        name=source,
                        line=dict(color=get_solar_colors(source)),
                    )
                )
            # plt.plot(array[0],array[1],label=source)

    if show:
        return fig

    return y


def get_neutrino_cs(bins, interpolation, path="../data/SOLAR", debug=False):
    """
    Read in marley data and return the neutrino cc spectrum.
    """
    # Get the neutrino cc spectrum from file
    data = np.loadtxt(path + "/neutrino_cc.txt")
    energies = data[:, 0]
    cc = data[:, 1]

    if debug:
        print_colored("Interpolation: %s" % interpolation, "DEBUG")
    func = interpolate.interp1d(
        energies, cc, kind="cubic", bounds_error=False, fill_value=interpolation
    )
    return func(bins)


def get_detected_solar_spectrum(
    bins, mass=10e9, components=[], interpolation=(0, 0), show=False, debug=False
):
    """
    Get the detected solar spectrum.

    Args:
        bins (np.array): energy bins
        mass (float): mass (default: 10e9)
        components (list): list of components (default: [])
        interpolation (str): interpolation method (default: "extrapolate")
        show (bool): if True, show the plot (default: False)
        debug (bool): if True, print debug messages (default: False)

    Returns:
        fig (plotly.graph_objects.Figure): plotly figure
    """
    # Prepare solar spectrum to convolve with marley signal
    # nbins = 60
    # CS    = 1e-42                 # Cross-section for CC interactions in Ar [cm²]
    CS = get_neutrino_cs(bins, interpolation=interpolation)
    mol = 39.948  # Molar mass [g/mol]
    # mass  = 10e9                # Mass in [g]
    # sec_to_year = 60*60*24*365  # Time factor [s/year]

    # Get the solar spectrum
    # components=["pp","f17","o15","n13","b8","hep"]
    if components == []:
        components = ["b8", "hep"]
    flux = get_solar_spectrum(
        components, bins, weigths="BS05", interpolation=interpolation, show=False
    )  # Flux [1/(cm²*s*Mev)]
    func = interpolate.interp1d(
        bins, flux, kind="cubic", bounds_error=False, fill_value=0
    )
    # Compute the effective flux by convolving with the cross-section and detcector prporties
    spectrum = func(bins)
    factor = (bins[1] - bins[0]) * mass * const.N_A / mol
    if show:
        fig = go.Figure()
        for source in components:
            array = read_solar_data("../data/SOLAR/", source, "BS05")
            func = interpolate.interp1d(
                array[0], array[1], kind="cubic", bounds_error=False, fill_value=0
            )
            this_spectrum = func(bins)
            fig.add_trace(
                go.Scatter(
                    x=bins,
                    y=CS * this_spectrum * factor,
                    name=source,
                    line=dict(color=get_solar_colors(source)),
                )
            )
            print_colored(
                "Total counts for %s: %.2e [Counts/10kt·s]"
                % (source, np.sum(CS * this_spectrum * factor)),
                color="INFO",
            )
            print_colored(
                "Total counts for %s: %.2e [Counts/70kt·year]"
                % (
                    source,
                    np.sum(CS * this_spectrum * factor) * 60 * 60 * 24 * 365 * 7,
                ),
                color="INFO",
            )

        fig.add_trace(
            go.Scatter(
                x=bins,
                y=CS * spectrum * factor,
                name="Spectrum",
                line=dict(color="blue", dash="dash"),
            )
        )
        print_colored(
            "Total counts for all sources: %.2e [Counts/10kt·s]"
            % (np.sum(CS * spectrum * factor)),
            color="INFO",
        )
        print_colored(
            "Total counts for all sources: %.2e [Counts/70kt·year]"
            % (np.sum(CS * spectrum * factor) * 60 * 60 * 24 * 365 * 7),
            color="INFO",
        )
        return fig
    else:
        return CS * spectrum * factor  # Flux [1/s]


@numba.njit
def get_marleyfrac_vectors(run, frac_name):
    """
    Fast function to get the marley particles from background data.
    """
    electron = run["Reco"][frac_name][np.where(run["Reco"]["Generator"] == 1)][:, 0]
    gamma = run["Reco"][frac_name][np.where(run["Reco"]["Generator"] == 1)][:, 1]
    neutron = run["Reco"][frac_name][np.where(run["Reco"]["Generator"] == 1)][:, 2]
    other = run["Reco"][frac_name][np.where(run["Reco"]["Generator"] == 1)][:, 3]
    return [electron, gamma, neutron, other], ["Electron", "Gamma", "Neutron", "Other"]


def compute_generator_df(reco_df, gen_labels, column_name="Generator", debug=False):
    """
    Compute the generator dataframe from the reco dataframe.
    """
    reco_gen_df = pd.DataFrame(reco_df[column_name].value_counts())
    reco_gen_df = reco_gen_df.reset_index()
    reco_gen_df = reco_gen_df.rename(
        columns={"index": column_name, column_name: "Value"}
    )
    if debug:
        rprint(reco_gen_df)
    # reco_gen_df.set_index(column_name,inplace=True)
    new_index = pd.Index(range(len(gen_labels)), name=column_name)
    reco_gen_df = reco_gen_df.reindex(new_index, fill_value=0, method=None)
    reco_gen_df.reset_index(inplace=True)
    reco_gen_df[column_name] = gen_labels
    reco_gen_df = reco_gen_df.set_index("Generator").T
    return reco_gen_df
