import os
import json
import pickle
import mplhep
import argparse
import plotly.io as pio

from hist import Hist, Stack
from itertools import product
from src.utils import get_project_root
from scipy.signal import find_peaks
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score
from scipy.interpolate import interp1d
from rich.progress import track
from rich import print as rprint

# Import all the local functions
from .reco_workflow import compute_reco_workflow
from .reco_workflow import compute_true_efficiency
from .workflow.lib_default import get_default_energies, get_default_nhits
from .workflow.lib_filter import compute_filtered_run
from .workflow.lib_efficiency import compute_particle_weights
from .df_functions import *
from .fit_functions import *
from .geo_functions import *
from .head_functions import *
from .hit_functions import *
from .io_functions import *
from .osc_functions import *
from .plt_functions import *
from .root_functions import *
from .sigma_functions import evaluate_significance
from .solar_functions import *
from .wkf_functions import *
from .evt_functions import *

# Config the external libraries
np.seterr(divide="ignore", invalid="ignore")
pd.options.mode.chained_assignment = None  # default='warn'
pd.options.display.float_format = "{:,.2f}".format
pd.set_option("future.no_silent_downcasting", True)
plt.rcParams.update({"font.size": 15})

# Load the default values
root = get_project_root()

nhits = get_default_nhits(root)

energy_edges, energy_centers, ebin = get_default_energies(root)

bkg_energy_edges, bkg_energy_centers, bkg_ebin = get_default_energies(
    root, "BKG_RECO_ENERGY"
)

red_energy_edges, red_energy_centers, red_ebin = get_default_energies(
    root, "REDUCED_RECO_ENERGY"
)

lowe_energy_edges, lowe_energy_centers, lowe_ebin = get_default_energies(
    root, "LOWE_RECO_ENERGY"
)

hep_rebin = np.arange(0, 10, 10)
hep_rebin = np.append(hep_rebin, np.arange(10, 20, 1))
hep_rebin = np.append(hep_rebin, np.arange(20, 32, 2))
hep_rebin_centers = (hep_rebin[1:] + hep_rebin[:-1]) / 2

sensitivity_rebin = np.arange(0, 10, 5)
sensitivity_rebin = np.append(sensitivity_rebin, np.arange(10, 16, 1))
sensitivity_rebin = np.append(sensitivity_rebin, np.arange(16, 20, 2))
sensitivity_rebin = np.append(sensitivity_rebin, np.arange(20, 30, 5))
sensitivity_rebin_centers = (sensitivity_rebin[1:] + sensitivity_rebin[:-1]) / 2

daynight_rebin = np.arange(0, 8, 8)
daynight_rebin = np.append(daynight_rebin, np.arange(8, 16, 0.5))
daynight_rebin = np.append(daynight_rebin, np.arange(16, 32, 1))
daynight_rebin_centers = (daynight_rebin[1:] + daynight_rebin[:-1]) / 2

pio.templates.default = "none"
colors = px.colors.qualitative.Prism
compare = px.colors.qualitative.Plotly
