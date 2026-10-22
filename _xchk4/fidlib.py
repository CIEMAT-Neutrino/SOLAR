"""_xchk4 helper: reproduce 02_best_fiducial.py significance for single volumes (read-only).

The functions of 02_best_fiducial.py are loaded from its source (function definitions only, the
script's main part is not executed); the metric itself comes from lib.fiducial_significance.
"""
import ast
import json
import os
import sys

B = "/pc/choozdsk01/users/manthey/SOLAR-sb"
sys.path.insert(0, B)
from lib import *  # noqa
from lib.fiducial_significance import (GROUP_COLUMNS, FIDUCIAL_COLUMNS, combine_components,
                                       build_significance_payload, get_essential_background_components)

FID = "/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/FIDUCIAL/truncated"
CONFIGS = {"hd_1x2x6_centralAPA": "HD Central", "hd_1x2x6_lateralAPA": "HD Lateral",
           "vd_1x8x14_3view_30deg_nominal": "VD Top", "vd_1x8x14_3view_30deg_shielded": "VD Shielded"}
SAMPLES = ["marley_final", "gamma_final", "neutron_final", "radiological_final"]


def _load_02_functions():
    src = open(f"{B}/src/physics/signal/02_best_fiducial.py").read()
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom))]
    ns = dict(globals())
    exec(compile(ast.Module(body=keep, type_ignores=[]), "02_best_fiducial_functions", "exec"), ns)
    return ns


F02 = _load_02_functions()
_info = load_analysis_info(B)


def load_plot_df(config, energy="SolarEnergy", suffix="", name="marley_final"):
    """As 02_best_fiducial.py: concat signal + background scans and explode."""
    dfs = [pd.read_pickle(f"{FID}/{config}/{name}/{config}_{name}_{energy}_Fiducial_Scan{suffix}.pkl")]
    for b in get_background_samples(B):
        p = f"{FID}/{config}/{b}/{config}_{b}_{energy}_Fiducial_Scan{suffix}.pkl"
        if os.path.exists(p):
            dfs.append(pd.read_pickle(p))
    raw = pd.concat(dfs, ignore_index=True)
    df = explode(raw, ["Counts", "Error+", "Error-", "Energy", "MCCounts"], debug=False).copy()
    for c in ["Counts", "Error+", "Error-", "MCCounts"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["Energy"] = pd.to_numeric(df["Energy"], errors="coerce")
    return df


def analysis_setup(analysis):
    key = analysis.upper()
    acfg = get_fiducialization_config(B, key)
    ref = str(_info.get("BEST_SIGMA_SIGNIFICANCE_REFERENCE", {}).get(key, "")).lower()
    if ref in {"asimov", "gaussian"} and ref != acfg.get("significance_type", "gaussian").lower():
        acfg = dict(acfg); acfg["significance_type"] = ref
    scfg = get_smoothing_config(B, analysis_name=key, dimensions="1d", stage="fiducial")
    return acfg, scfg


def band_config(acfg, band):
    """band None = full window, else (emin, emax)."""
    if band is None:
        return acfg
    c = dict(acfg); c["energy_min"], c["energy_max"] = band
    return c


def volume_df(plot_df, vol):
    x, y, z = vol
    return plot_df.loc[(plot_df["FiducializedX"] == x) & (plot_df["FiducializedY"] == y) & (plot_df["FiducializedZ"] == z)]


def merged_for(df, acfg):
    s = combine_components(df, acfg.get("signal_components", []))
    b = combine_components(df, acfg.get("background_components", []))
    m = pd.merge(s, b, on=GROUP_COLUMNS, how="outer", suffixes=("Signal", "Background"))
    for c in ["MCCountsSignal", "CountsSignal", "Error+Signal", "Error-Signal",
              "MCCountsBackground", "CountsBackground", "Error+Background", "Error-Background"]:
        if c not in m.columns:
            m[c] = 0.0
    return m.fillna(0.0)


def payload(plot_df, vol, acfg, scfg, exposure=100.0):
    m = merged_for(volume_df(plot_df, vol), acfg)
    return build_significance_payload(m, acfg.get("significance_type", "gaussian"), acfg.get("combine_mode", "quadrature"),
                                      scfg, exposure, acfg.get("energy_min"), acfg.get("energy_max"))


def profile_path(config, energy="SolarEnergy", suffix=""):
    return f"{FID}/{config}/marley_final/{config}_marley_final_{energy}_Fiducial_Profiles{suffix}.pkl"


# ---- event level (tables written by s4_events.py, validated against the scan pkls) ----
BKG = ["gamma_final", "neutron_final", "radiological_final"]
COMP = {"gamma_final": "gamma", "neutron_final": "neutron", "radiological_final": "radiological"}


def config_info(config):
    return json.load(open(f"{B}/config/{config}/{config}_config.json"))


def load_events(config, sample):
    d = pd.read_pickle(f"{B}/_xchk4/events/{config}_{sample}.pkl")
    d["Bin"] = np.digitize(d.SolarEnergy.values, reco_energy_edges) - 1
    d["BinCenter"] = np.where((d.Bin >= 0) & (d.Bin < len(reco_energy_centers)),
                              np.asarray(reco_energy_centers)[np.clip(d.Bin, 0, len(reco_energy_centers) - 1)], np.nan)
    return d


def in_volume(d, config, vol):
    """Mask of 01_fiducialize.py (reco positions)."""
    info = config_info(config)
    fx, fy, fz = vol
    if fx == 0 and fy == 0 and fz == 0:
        return np.ones(len(d), bool)
    dx = info["DETECTOR_SIZE_X"] + 2 * info["DETECTOR_GAP_X"]; dy = info["DETECTOR_SIZE_Y"] + 2 * info["DETECTOR_GAP_Y"]
    x, y, z = d.RecoX.values, d.RecoY.values, d.RecoZ.values
    if config == "hd_1x2x6_lateralAPA":
        mx = np.abs(x) > fx
    elif config == "hd_1x2x6_centralAPA":
        mx = np.abs(x) < dx / 2 - fx
    else:
        mx = x < dx / 2 - fx
    return (mx & (np.abs(y) < dy / 2 - fy) & (z > fz - info["DETECTOR_GAP_Z"])
            & (z < info["DETECTOR_SIZE_Z"] + info["DETECTOR_GAP_Z"] - fz))
