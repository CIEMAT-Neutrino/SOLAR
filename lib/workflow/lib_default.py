import json
import numpy as np


def get_default_info(root: str, variable: str):
    """
    This function returns the default analysis information.
    """
    analysis_info = json.load(open(f"{root}/lib/import/analysis.json", "r"))
    return analysis_info[variable]


def get_default_nhits(root: str, variable: str = "NHITS"):
    """
    This function returns the default energy binning used in the analysis.
    """
    return get_default_info(root, variable)


def get_default_energies(root: str, variable: str = "ENERGY"):
    """
    This function returns the default energy binning used in the analysis.
    """
    analysis_info = json.load(open(f"{root}/lib/import/analysis.json", "r"))
    e_range = analysis_info[f"{variable}_RANGE"]
    e_bins = analysis_info[f"{variable}_BINS"]
    energy_edges = np.linspace(e_range[0], e_range[-1], e_bins + 1)
    energy_centers = (energy_edges[1:] + energy_edges[:-1]) / 2
    return energy_edges, energy_centers, energy_edges[1] - energy_edges[0]
