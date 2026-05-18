import os
import json
import pickle
import mplhep
import argparse
import plotly.io as pio

# Import savgol_filter
from hist import Hist, Stack
from itertools import product
from src.utils import get_project_root
from scipy.signal import find_peaks
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from rich.progress import track
from rich import print as rprint

# Import all the local functions
from .lib_reco import compute_reco_workflow
from .lib_reco import compute_true_efficiency
from .lib_default import (
    get_default_energies,
    get_default_nhits,
    load_analysis_info,
    get_analysis_threshold,
)
from .lib_filter import (
    compute_filtered_run,
    update_yaml_file,
    update_json_file,
)
from .lib_weights import compute_particle_weights, compute_particle_surface
from .lib_cluster import compute_total_energy
from .lib_df import *
from .lib_fit import *
from .lib_geo import *
from .lib_head import *
from .lib_hit import *
from .lib_io import *
from .lib_osc import *
from .lib_plt import *
from .lib_root import *
from .lib_smooth import *
from .lib_sigma import evaluate_significance, evaluate_profile_likelihood_discovery
from .lib_solar import *
from .lib_wkf import *
from .lib_evt import *
from .lib_fiducial import *
from .lib_background import *
from .lib_log import configure_global_logging, get_global_logging_config

# Config the external libraries
np.seterr(divide="ignore", invalid="ignore")
pd.options.mode.chained_assignment = None  # default='warn'
pd.options.display.float_format = "{:,.2f}".format
pd.set_option("future.no_silent_downcasting", True)
plt.rcParams.update({"font.size": 15})

# Load the default values
root = get_project_root()


def get_default_acc(length: int) -> float:
    """
    Get the default accuracy value for plotting or analysis.

    Args:
        length (int): Length of the data array.

    Returns:
        float: Default accuracy value.
    """
    acc = int(length / 100)
    if acc > 100:
        acc = 100

    elif acc < 20:
        acc = 20

    return acc


nhits = get_default_nhits(str(root))

energy_edges, energy_centers, ebin = get_default_energies(str(root))

true_energy_edges, true_energy_centers, true_ebin = get_default_energies(
    str(root), "TRUE_ENERGY"
)

reco_energy_edges, reco_energy_centers, reco_ebin = get_default_energies(
    str(root), "RECO_ENERGY"
)

bkg_energy_edges, bkg_energy_centers, bkg_ebin = get_default_energies(
    str(root), "BKG_RECO_ENERGY"
)

red_energy_edges, red_energy_centers, red_ebin = get_default_energies(
    str(root), "REDUCED_RECO_ENERGY"
)

lowe_energy_edges, lowe_energy_centers, lowe_ebin = get_default_energies(
    str(root), "LOWE_RECO_ENERGY"
)

hep_rebin = np.arange(0, 31, 1)
hep_rebin_centers = (hep_rebin[1:] + hep_rebin[:-1]) / 2

sensitivity_rebin = np.arange(0, 31, 1)
sensitivity_rebin_centers = (sensitivity_rebin[1:] + sensitivity_rebin[:-1]) / 2

daynight_rebin = np.arange(0, 32, 2)
daynight_rebin_centers = (daynight_rebin[1:] + daynight_rebin[:-1]) / 2

pio.templates.default = "none"
default = px.colors.qualitative.D3
colors = px.colors.qualitative.Prism
compare = px.colors.qualitative.Plotly

# ---------------------------------------------------------------------------
# Kaleido / Chrome path
#
# kaleido v1+ requires Chrome for static image export.  The system container
# is read-only so Chrome is kept in <project>/.chrome/.  Set BROWSER_PATH if
# not already in the environment so kaleido picks it up automatically.
# ---------------------------------------------------------------------------
_chrome_exe = os.path.join(str(root), ".chrome", "chrome-linux64", "chrome")
if os.path.isfile(_chrome_exe) and "BROWSER_PATH" not in os.environ:
    os.environ["BROWSER_PATH"] = _chrome_exe

# ---------------------------------------------------------------------------
# Apptainer arrow conflict workaround
#
# The container carries both pyarrow's libarrow.so.2100 and the system
# libarrow.so.900.  They have different sonames so the dynamic linker loads
# both; the C++ ThreadPool vtable then resolves across incompatible versions
# and segfaults during normal Python teardown.  Registering os._exit(0) as an
# atexit handler (LIFO → runs last) lets all user handlers finish first, then
# terminates before C++ destructors are invoked.  Guard on the conflicting
# library so the workaround is inert everywhere else.
# ---------------------------------------------------------------------------
import atexit as _atexit
import sys as _sys

if os.path.exists("/lib64/libarrow.so.900"):
    def _exit_before_arrow_teardown():
        _sys.stdout.flush()
        _sys.stderr.flush()
        os._exit(0)
    _atexit.register(_exit_before_arrow_teardown)
