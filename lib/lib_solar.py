from src.utils import get_project_root

import os
import json
import plotly
import numba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from typing import Optional
from plotly.subplots import make_subplots
from scipy import interpolate
from particle import Particle
from rich.progress import track
from scipy import constants as const
from rich import print as rprint

from lib.lib_io import (
    get_bkg_config,
    get_gen_label,
)
from lib.lib_osc import (
    get_nadir_angle,
    get_oscillation_datafiles,
    get_oscillation_map,
)
from lib.lib_plt import format_coustom_plotly, unicode

root = get_project_root()


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
        try:
            pdg_dict[pdg] = Particle.from_pdgid(pdg).name
        except:
            pdg_dict[pdg] = -1
    return pdg_dict


def get_pdg_color(pdgs: list[str], debug: bool = False):
    """
    Get the color for each pdg.
    """
    # default_color_dict = {12: "red", 11: "orange", 22: "blue", 2112: "green", 2212: "purple", -12: "cyan",}
    output = ""
    default_color_dict = json.load(open(f"{root}/import/pdg_color.json"))
    if debug:
        rprint(f"[cyan]PDGs: {default_color_dict}[/cyan]")

    color_dict = dict()
    for pdg in pdgs:
        try:
            color_dict[str(pdg)] = default_color_dict[str(pdg)]
        except:
            if debug:
                output += f"[yellow][WARNING][/yellow] PDG {pdg} not found in default color dictionary!\n"
            color_dict[str(pdg)] = "grey"

    if output != "":
        rprint(output)
    return color_dict


def get_solar_weigths(weights="B16-GS98"):
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

    if weights == "B16-GS98":
        weights_dict = {
            "pp": 5.98e00,
            "pep": 1.44e-10,
            "b7": 4.93e-10,
            "n13": 2.78e-02,
            "o15": 2.05e-02,
            "f17": 5.29e-04,
            "b8": 5.46e-04,
            "hep": 7.98e-07,
        }  # Flux amp of each component
        return weights_dict

    else:
        rprint("[red][ERROR] Weights not defined, using B16-GS98![/red]")
        get_solar_weigths("B16-GS98")


def get_solar_colors(source):
    """
    Use plotly colors to color the solar flux components.
    """
    colors = plotly.colors.qualitative.Prism
    source_list = list(get_solar_weigths().keys())
    for idx, this_source in enumerate(source_list):
        if this_source == source:
            return colors[idx]


def read_solar_data(source, path: Optional[str] = None, weigths: str = "B16-GS98"):
    """
    Read in the solar flux data and interpolate it to the desired energy bins.
    """

    if path == None:
        path = f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/data/SOLAR/"

    data = get_solar_weigths(weigths)
    energy = []
    flux = []
    factor = 1e10 * data[source.split("_")[0]]
    text = open(path + source + ".dat", "r")
    lines = text.readlines()

    for j in range(len(lines)):
        values = lines[j].split()
        energy.append(float(values[0]))
        flux.append(float(values[1]) * factor)

    return np.array([energy, flux])


def get_solar_spectrum(
    components: list,
    bins,
    weigths="B16-GS98",
    path: Optional[str] = None,
    interpolation="linear",
    bounds=(0, 0),
    debug=False,
):
    """
    Read in the solar flux data and interpolate it to the desired energy bins.

    Args:
        components (list): list of components
        bins (np.array): energy bins
        weigths (str): weigths (default: "BS05")
        path (str): input path (default: "/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/data/SOLAR/")
        interpolation (Any): interpolation method from scipy.interpolate.interp1d (default: 'linear')
        bounds (Any): bounds for the interpolation (default: (0,0))
        debug (bool): if True, print debug messages (default: False)

    Returns:
        y (np.array): interpolated flux values
    """
    if path == None:
        path = f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/data/SOLAR/"

    x = bins
    y = np.zeros(len(x))  # Array that will host the interpolated flux values.

    for idx, source in enumerate(components):
        array = read_solar_data(source, path, weigths)
        if source.split("_")[0] != "pep" and source.split("_")[0] != "b7":
            func = interpolate_solar_data(
                array[0],
                array[1],
                source,
                interpolation=interpolation,
                bounds=bounds,
                debug=debug,
            )
            y = y + func(x)

    return y


def plot_solar_spectrum(
    fig,
    idx: int,
    components: list = ["pp", "pep", "b7", "f17", "o15", "n13", "b8", "hep"],
    weights: str = "B16-GS98",
    path: Optional[str] = None,
    debug: bool = False,
):
    """
    Plot the solar flux data.

    Args:
        components (list): list of components
        bins (np.array): energy bins
        weigths (str): weigths (default: "BS05")
        interpolation (Any): interpolation method from scipy.interpolate.interp1d (default: (0,0))
        in_path (str): input path (default: "../data/SOLAR/")
        show (bool): if True, show the plot (default: False)

    Returns:
        fig (plotly.graph_objects.Figure): the plotly figure object
    """
    data_list = []
    if path == None:
        path = f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/data/SOLAR/"
    for source in components:
        for jdx in range(1 if source.split("_")[0] != "b7" else 2):
            if source.split("_")[0] != "b7":
                array = read_solar_data(source, path, weights)
            else:
                array = read_solar_data(f"{source}_{jdx+1}", path, weights)

            data_list.append(
                {
                    "Flux": array[1],
                    "Energy": array[0],
                    "Source": source.split("_")[0],
                    "Plot": weights,
                }
            )

            fig.add_trace(
                go.Scatter(
                    legendgrouptitle_text="Solar Process",
                    legendgroup=idx,
                    x=array[0],
                    y=array[1],
                    name=source,
                    line=dict(color=get_solar_colors(source)),
                    showlegend=True if source.split("_")[0] != "b7" else jdx == 0,
                ),
                col=1 + idx,
                row=1,
            )

    df = pd.DataFrame(data_list)
    fig.update_layout(title=f"Solar Neutrino Spectrum ({weights})")
    return fig, df


def interpolate_solar_data(x, y, label, interpolation=None, bounds=None, debug=False):
    """
    Interpolate the solar flux data.
    """
    if debug:
        rprint(f"{label}\nInterpolation: {interpolation}\n Bounds: {bounds}")
    func = interpolate.interp1d(
        x, y, kind=interpolation, bounds_error=False, fill_value=bounds
    )
    return func


def get_neutrino_cs(
    bins,
    interpolation=None,
    bounds=None,
    path=f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/data/SOLAR/",
    label="neutrino_cc_final",
    debug=False,
):
    """
    Read in marley data and return the neutrino cc spectrum.
    """
    # Get the neutrino cc spectrum from file
    data = np.loadtxt(f"{path}{label}.txt")
    energies = data[:, 0]
    cc = data[:, 1]

    if interpolation != None:
        func = interpolate_solar_data(
            energies, cc, label, interpolation=interpolation, bounds=bounds, debug=debug
        )
    if interpolation == None:
        func = interpolate_solar_data(
            energies,
            cc,
            label,
            interpolation="linear",
            bounds="extrapolate",
            debug=debug,
        )
    return func(bins)


def get_detected_solar_spectrum(
    bins, mass=1e9, components=[], interpolation="linear", bounds=(0, 0), debug=False
):
    """
    Get the data for the detected solar spectrum.

    Args:
        bins (np.array): energy bins
        mass (float): mass (default: 10e9)
        components (list): list of components (default: [])
        interpolation (str): interpolation method (default: "extrapolate")
        debug (bool): if True, print debug messages (default: False)

    Returns:
        spectrum (np.array): detected solar spectrum data
    """
    # Prepare solar spectrum to convolve with marley signal
    CS = get_neutrino_cs(
        bins, interpolation=interpolation, bounds=bounds, debug=debug
    )  # Cross-section [cm²]
    mol = 39.948  # Molar mass [g/mol]

    # Get the solar spectrum
    if components == []:
        components = ["b8", "hep"]
    flux = get_solar_spectrum(components, bins)
    func = interpolate_solar_data(
        bins,
        flux,
        "solar_spectrum",
        interpolation=interpolation,
        bounds=bounds,
        debug=debug,
    )
    # Compute the effective flux by convolving with the cross-section and detector properties
    spectrum = func(bins)
    factor = (bins[1] - bins[0]) * mass * const.N_A / mol

    return CS * spectrum * factor  # Flux [1/s]


def plot_detected_solar_spectrum(
    fig,
    idx,
    bins,
    mass=10e9,
    components: list = ["b8", "hep"],
    interpolation="linear",
    bounds=(0, 0),
    osc=False,
    debug=False,
):
    """
    Plot the detected solar spectrum.

    Args:
        bins (np.array): energy bins
        mass (float): mass (default: 10e9)
        components (list): list of components (default: [])
        interpolation (str): interpolation method (default: "extrapolate")
        debug (bool): if True, print debug messages (default: False)

    Returns:
        fig (plotly.graph_objects.Figure): plotly figure
    """
    data = {}
    info = ""
    colors = plotly.colors.qualitative.Prism
    oscillation = get_oscillation_map(output="interp1d")
    osc_func = oscillation[list(oscillation.keys())[0]]
    spectrum = get_detected_solar_spectrum(
        bins,
        mass=mass,
        components=components,
        interpolation=interpolation,
        bounds=bounds,
        debug=debug,
    )
    data["Energy"] = bins
    for source in components:
        this_spectrum = get_detected_solar_spectrum(
            bins,
            mass=mass,
            components=[source],
            interpolation=interpolation,
            bounds=bounds,
            debug=debug,
        )
        # If value in spectrum smaller than 1e-10, set it to NaN
        this_spectrum[this_spectrum < 1e-10] = 0
        if osc:
            this_spectrum = osc_func(bins) * this_spectrum

        fig.add_trace(
            go.Scatter(
                legendgroup=str(idx),
                x=bins,
                y=this_spectrum / (bins[1] - bins[0]),
                name=f"CC {source}",
                line=dict(color=get_solar_colors(source)),
            ),
            col=1 + idx,
            row=1,
        )
        # Add spectrum to data
        data[source] = this_spectrum / (bins[1] - bins[0])
        info = f"{info}\nTotal counts for {source}:\t{np.sum(this_spectrum):.2e} [Counts/{mass:.1e}] kg·s]"
        # info = info + "\nTotal counts for %s:\t%.2e [Counts/70kt·year]" % (
        #     source,
        #     np.sum(this_spectrum) * 60 * 60 * 24 * 365 * 7,
        # )

    # If value in spectrum smaller than 1e-10, set it to NaN
    spectrum[spectrum < 1e-10] = 0
    if osc:
        spectrum = osc_func(bins) * spectrum

    fig.add_trace(
        go.Scatter(
            legendgrouptitle_text="Interacting Spectrum",
            legendgroup=idx,
            x=bins,
            y=spectrum / (bins[1] - bins[0]),
            name="Combined ",
            line=dict(color=colors[-1], dash="dash"),
        ),
        col=1 + idx,
        row=1,
    )
    # Add spectrum to data
    data["Combined"] = spectrum / (bins[1] - bins[0])
    info = f"{info}\nTotal counts for all:\t{np.sum(spectrum):.2e} [Counts/{mass:.1e} kg·s]"
    info = f"{info}\nTotal counts for all:\t{60 * 60 * 24 * 365 * np.sum(spectrum):.2e} [Counts/{mass:.1e} kg·year]"

    if debug:
        rprint(info)
    return fig, data


def make_true_solar_plot(
    bins,
    components=["b8", "hep"],
    mass=10e9,
    osc=True,
    interpolation="linear",
    bounds=(0, 0),
    debug=False,
):
    """
    Plot the detected solar spectrum.

    Args:
        bins (np.array): energy bins
        mass (float): mass (default: 10e9)
        components (list): list of components (default: [])
        interpolation (str): interpolation method (default: "extrapolate")
        debug (bool): if True, print debug messages (default: False)

    Returns:
        fig (plotly.graph_objects.Figure): plotly figure
    """
    colors = plotly.colors.qualitative.Prism
    cc_array = get_neutrino_cs(bins, interpolation, bounds, debug=debug)
    fig = make_subplots(
        rows=1,
        cols=3,
        horizontal_spacing=0.1,
        subplot_titles=(
            "True Solar Neutrino Spectrum",
            "X-Section for " + unicode("nu") + " + 40Ar Interaction",
            "Interacting Solar Neutrino Spectrum",
        ),
    )
    # 1st plot
    fig, df = plot_solar_spectrum(fig, 0)
    # 2nd plot
    cc_array_linear = get_neutrino_cs(bins, "linear", (0, 0), debug=debug)
    for cc, ll, ls, lc in zip(
        [cc_array_linear, cc_array],
        ["linear", "default"],
        ["solid", "dash"],
        [colors[-2], colors[-1]],
    ):
        fig.add_trace(
            go.Scatter(
                legendgrouptitle_text="Marley X-Section",
                legendgroup=1,
                x=bins,
                y=cc,
                mode="lines",
                name=f"NuE-Ar CC {ll}",
                line=dict(color=lc, dash=ls),
            ),
            row=1,
            col=2,
        )
    # 3rd plot
    fig, data = plot_detected_solar_spectrum(
        fig,
        2,
        bins,
        mass=mass,
        components=components,
        interpolation=interpolation,
        bounds=bounds,
        osc=osc,
        debug=debug,
    )
    fig = format_coustom_plotly(
        fig,
        log=(True, True),
        tickformat=(".0f", ".0e"),
        matches=(None, None),
        add_units=False,
    )

    fig.update_xaxes(range=[-1, 1.4], row=1, col=1)
    fig.update_yaxes(range=[1, 12], title_text="Flux (Hz/MeV·cm²)", row=1, col=1)
    fig.update_yaxes(range=[-44, -40], title_text="Cross Section (cm²)", row=1, col=2)
    fig.update_yaxes(
        range=[-8, -3], title_text="Counts/Energy·10kt·s (Hz/MeV)", row=1, col=3
    )
    fig.update_xaxes(title_text="Energy (MeV)")

    return fig, data


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
