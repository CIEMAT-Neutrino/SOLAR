# PYTHON/GENERAL PACKAGES #
import os, copy, uproot, math, random, gc, ROOT
import numpy                 as np
import pandas                as pd
import awkward               as ak
import plotly.express        as px
import plotly.offline        as pyoff
# import chart_studio.plotly   as py
import plotly.graph_objects  as go
import matplotlib.cm         as cm
import matplotlib.colors     as colors
import matplotlib.pyplot     as plt
import datashader            as ds
from icecream import ic      as ic

from particle                import Particle
from itertools               import product
from scipy                   import interpolate
from scipy                   import special as sp
from scipy                   import constants as const
from scipy.optimize          import curve_fit
from plotly.tools            import FigureFactory
from matplotlib.colors       import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from plotly.subplots         import make_subplots
from matplotlib.colors       import Normalize
from rich.progress           import track
from ROOT                    import TFile, TTree, TList

# LARSOFT_CYTOOLS SELF-LIB FUNCTIONS #
from .bkg_functions       import *
from .df_functions        import *
from .fit_functions       import *
from .geo_functions       import *
from .head_functions      import *
from .hit_functions       import *
from .io_functions        import *
from .osc_functions       import *
from .plt_functions       import *
from .reco_functions      import *
from .root_functions      import *
from .solar_functions     import *