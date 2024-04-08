import json
import numpy as np

def get_default_nhits(root, variable:str = "NHITS"):
    '''
    This function returns the default energy binning used in the analysis.
    '''
    analysis_info = json.load(open(f"{root}/import/analysis.json",'r'))
    nhits = analysis_info[f"{variable}"]
    return nhits


def get_default_energies(root, variable:str = "RECO_ENERGY"):
    '''
    This function returns the default energy binning used in the analysis.
    '''
    analysis_info = json.load(open(f"{root}/import/analysis.json",'r'))
    e_range = analysis_info[f"{variable}_RANGE"]
    e_bins = analysis_info[f"{variable}_BINS"]
    energy_edges = np.linspace(e_range[0],e_range[-1],e_bins+1)
    energy_centers = (energy_edges[1:]+energy_edges[:-1])/2
    ebin = energy_edges[1]-energy_edges[0]
    return energy_edges, energy_centers, ebin