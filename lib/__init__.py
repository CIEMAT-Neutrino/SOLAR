from src.utils import get_project_root
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from scipy.interpolate import interp1d

from .ana_functions import *
from .df_functions import *
from .fit_functions import *
from .geo_functions import *
from .head_functions import *
from .hit_functions import *
from .io_functions import *
from .osc_functions import *
from .plt_functions import *
from .reco_functions import *
from .root_functions import *
from .solar_functions import *
from .wkf_functions import *
from .evt_functions import *

root = get_project_root()
np.seterr(divide="ignore", invalid="ignore")
pd.options.mode.chained_assignment = None  # default='warn'
plt.rcParams.update({"font.size": 15})

nhits = get_default_nhits(root)
energy_edges, energy_centers, ebin = get_default_energies(root)
red_energy_edges, red_energy_centers, red_ebin = get_default_energies(root,"REDUCED_RECO_ENERGY")