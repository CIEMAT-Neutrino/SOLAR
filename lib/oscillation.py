from src.utils import get_project_root

import os, glob, uproot, json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from typing import Optional
from scipy import interpolate
from rich.progress import track
from rich import print as rprint
from plotly.subplots import make_subplots

from lib.io import print_colored
from lib.defaults import load_analysis_info
from lib.plotting import format_coustom_plotly, unicode

root = get_project_root()

def get_nadir_angle(
    path: str = f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/data/OSCILLATION/", debug: bool = False
) -> list[np.array]:
    """
    This function can be used to obtain the nadir angle distribution for DUNE.

    Args:
        path (str): Path to the nadir angle data file (default: f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/data/OSCILLATION/")
        show (bool): If True, show the plot (default: False)
        debug (bool): If True, the debug mode is activated.

    Returns:
        [xnadir_centers,ynadir_centers]: list containing the nadir angle and the PDF.
    """
    with uproot.open(path + "nadir.root") as root_file:
        # Loas pdf histogram
        pdf = root_file["nadir;1"]
        pdf_array = pdf.to_hist().to_numpy()
        xbin_edges = pdf_array[1]
        xnadir_centers = 0.5 * (xbin_edges[1:] + xbin_edges[:-1])
        ynadir_centers = pdf_array[0]

    if debug:
        print(f"Nadir angle data loaded: xnadir_centers = {xnadir_centers}, ynadir_centers = {ynadir_centers}")
        print(f"Check for PDF normalization: {np.sum(ynadir_centers)}")

    return [xnadir_centers, ynadir_centers]


def plot_nadir_angle(fig, idx, norm: Optional[float] = None, plot_type: str = "scatter", debug: bool = False):
    """
    This function can be used to plot the nadir angle distribution for DUNE.

    Args:
        fig (plotly.graph_objects.Figure): Plotly figure.
        idx (tuple(int)): Index of the subplot.
        debug (bool): If True, the debug mode is activated.

    Returns:
        fig (plotly.graph_objects.Figure): Plotly figure.
    """
    nadir_data = get_nadir_angle(debug = debug)
    name = "Nadir Angle PDF"
    
    if norm is not None: 
        nadir_data[1] = nadir_data[1] / (norm*np.max(nadir_data[1]))
        name = "DUNE Yearly Exposure (AU)"

    if plot_type == "scatter":
        fig.add_scatter(
            x=nadir_data[0].astype(float),
            y=nadir_data[1].astype(float),
            mode='markers',
            name=name,
            row=idx[0],
            col=idx[1],
        )
    if plot_type == "hist":
        fig.add_trace(
            go.Scatter(
                x=nadir_data[0].astype(float),
                y=nadir_data[1].astype(float),
                mode="lines",
                line_shape="hvh",
                name=name,
            ),
            row=idx[0],
            col=idx[1],
        )
    fig.update_xaxes(
        title_text="Azimuth Angle cos(" + unicode("eta") + ")", row=idx[0], col=idx[1]
    )
    fig.update_yaxes(title_text="PDF", row=idx[0], col=idx[1])

    return fig


def make_oscillation_grid(analysis_info: dict, dm2=None, sin13=None, sin12=None):
    """
    Build an inhomogeneous oscillation parameter grid from OSCILLATION_GRID in analysis_info.

    The grid is defined as the union of 2D planes and 1D lines (for denser 1D projections).
    No full Cartesian product is taken — only the specified slices are included.

    Config keys in OSCILLATION_GRID:
      ranges: global {axis: {min, max, scale}} — sets axis extent and scale
      planes: list of {axes: [a, b], fixed: {c: ref_key|float}, steps: [na, nb]}
              each plane is a 2D slice with the third parameter fixed
      lines:  list of {axis: a, fixed: {b: ref, c: ref}, steps: n}
              denser 1D scans at the intersection of two planes

    Fixed values accept string references resolved from analysis_info:
      "SOLAR_DM2", "REACT_DM2", "SIN13", "SIN12"

    Reference values (SOLAR_DM2, REACT_DM2, SIN13, SIN12) are always added
    to each axis so that reference points are guaranteed to exist in the grid.

    If dm2/sin13/sin12 are given, returns only that specific combination
    (used for targeted single-point queries).

    Returns:
        (dm2_list, sin13_list, sin12_list) — parallel flat lists of all grid points.
    """
    # Specific-point override: skip grid building
    if dm2 is not None or sin13 is not None or sin12 is not None:
        from itertools import product as _product
        if dm2   is None and "SOLAR_DM2" not in analysis_info:
            raise SystemExit("[oscillation] 'SOLAR_DM2' missing from analysis config (physics.json).")
        if sin13 is None and "SIN13"     not in analysis_info:
            raise SystemExit("[oscillation] 'SIN13' missing from analysis config (physics.json).")
        if sin12 is None and "SIN12"     not in analysis_info:
            raise SystemExit("[oscillation] 'SIN12' missing from analysis config (physics.json).")
        dv = [float(dm2)]   if dm2   is not None else [float(analysis_info["SOLAR_DM2"])]
        tv = [float(sin13)] if sin13 is not None else [float(analysis_info["SIN13"])]
        sv = [float(sin12)] if sin12 is not None else [float(analysis_info["SIN12"])]
        pts = list(_product(dv, tv, sv))
        return ([p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts])

    # Reference value lookup
    _ref_map = {
        "SOLAR_DM2": float(analysis_info.get("SOLAR_DM2", 6e-5)),
        "REACT_DM2": float(analysis_info.get("REACT_DM2", 7.4e-5)),
        "SIN13":     float(analysis_info.get("SIN13",     0.021)),
        "SIN12":     float(analysis_info.get("SIN12",     0.303)),
    }
    # Always-included values per axis
    _axis_pins = {
        "dm2":   [_ref_map["SOLAR_DM2"], _ref_map["REACT_DM2"]],
        "sin13": [_ref_map["SIN13"]],
        "sin12": [_ref_map["SIN12"]],
    }

    def _resolve(v):
        return _ref_map[v] if isinstance(v, str) else float(v)

    grid_cfg = analysis_info.get("OSCILLATION_GRID", {})

    # Default ranges if section absent
    _default_ranges = {
        "dm2":   {"min": 2e-5, "max": 2e-4, "scale": "log"},
        "sin13": {"min": 0.017, "max": 0.025, "scale": "linear"},
        "sin12": {"min": 0.10,  "max": 0.70,  "scale": "linear"},
    }
    ranges = grid_cfg.get("ranges", _default_ranges)

    def _make_axis(name: str, steps: int) -> list:
        r = ranges.get(name, _default_ranges.get(name, {}))
        mn    = float(r.get("min", _axis_pins[name][0]))
        mx    = float(r.get("max", _axis_pins[name][-1]))
        scale = r.get("scale", "linear")
        if steps <= 1:
            pts = [mn] if mn == mx else [mn, mx]
        elif scale == "log":
            pts = list(np.logspace(np.log10(mn), np.log10(mx), steps))
        else:
            pts = list(np.linspace(mn, mx, steps))
        return sorted(set(pts + _axis_pins[name]))

    # If no planes/lines defined, fall back to legacy full Cartesian product
    if "planes" not in grid_cfg and "lines" not in grid_cfg:
        from itertools import product as _product
        def _expand_legacy(spec, pins):
            if isinstance(spec, list):
                return sorted(set(float(v) for v in spec) | set(pins))
            if isinstance(spec, (int, float)):
                return [float(spec)]
            mn    = float(spec["min"])
            mx    = float(spec["max"])
            n     = int(spec.get("steps", 1))
            scale = spec.get("scale", "linear")
            incl  = spec.get("include_ref", True)
            pts   = ([mn] if mn == mx else [mn, mx]) if n <= 1 else (
                list(np.logspace(np.log10(mn), np.log10(mx), n)) if scale == "log"
                else list(np.linspace(mn, mx, n))
            )
            return sorted(set(pts + pins) if incl else pts)
        dv = _expand_legacy(grid_cfg.get("dm2",   _ref_map["SOLAR_DM2"]), _axis_pins["dm2"])
        tv = _expand_legacy(grid_cfg.get("sin13", _ref_map["SIN13"]),     _axis_pins["sin13"])
        sv = _expand_legacy(grid_cfg.get("sin12", _ref_map["SIN12"]),     _axis_pins["sin12"])
        pts = list(_product(dv, tv, sv))
        return ([p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts])

    triplets: set = set()

    # 2D planes
    for plane in grid_cfg.get("planes", []):
        a0, a1   = plane["axes"]
        s0, s1   = plane["steps"]
        fixed_kv = {k: _resolve(v) for k, v in plane["fixed"].items()}
        v0 = _make_axis(a0, s0)
        v1 = _make_axis(a1, s1)
        for x0 in v0:
            for x1 in v1:
                p = dict(fixed_kv, **{a0: x0, a1: x1})
                triplets.add((p["dm2"], p["sin13"], p["sin12"]))

    # 1D lines (denser, for 1D projections)
    for line in grid_cfg.get("lines", []):
        axis     = line["axis"]
        steps    = line["steps"]
        fixed_kv = {k: _resolve(v) for k, v in line["fixed"].items()}
        for x in _make_axis(axis, steps):
            p = dict(fixed_kv, **{axis: x})
            triplets.add((p["dm2"], p["sin13"], p["sin12"]))

    if not triplets:
        triplets.add((_ref_map["SOLAR_DM2"], _ref_map["SIN13"], _ref_map["SIN12"]))

    sorted_pts = sorted(triplets)
    return (
        [t[0] for t in sorted_pts],
        [t[1] for t in sorted_pts],
        [t[2] for t in sorted_pts],
    )


def get_oscillation_datafiles(
    dm2=None,
    sin13=None,
    sin12=None,
    path: str = f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/data/OSCILLATION/",
    ext: str = "root",
    auto: bool = False,
    debug: bool = False,
    backend: str = "file",
):
    """
    This function can be used to obtain the oscillation data files for DUNE's solar analysis.

    Args:
        dm2 (float/list):   Solar mass squared difference (default: analysis["DM2"])
        sin13 (float/list): Solar mixing angle (default: analysis["SIN13"])
        sin12 (float/list): Solar mixing angle (default: analysis["SIN12"])
        path (str):   Path to the oscillation data files (default: f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/data/OSCILLATION/")
        ext (str):    Extension of the oscillation data files (default: "root")
        auto (bool):  If True, the function will look for all the oscillation data files in the path (default: False)
        debug (bool): If True, the debug mode is activated.

    Returns:
        (found_dm2,found_sin13,found_sin12): tuple containing the dm2, sin13 and sin12 values of the found oscillation data files.
    """
    if backend != "file":
        analysis_info = load_analysis_info(str(root))
        result = make_oscillation_grid(analysis_info, dm2=dm2, sin13=sin13, sin12=sin12)
        if debug:
            rprint(f"[cyan]get_oscillation_datafiles backend={backend}: {len(result[0])} grid point(s) from OSCILLATION_GRID[/cyan]")
        return result

    data_files = glob.glob(f"{path}" + "*_dm2_*_sin13_*_sin12_*")
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
        if dm2 is None and sin13 is None and sin12 is None:
            analysis_info = load_analysis_info(str(root))
            found_dm2, found_sin13, found_sin12 = (
                analysis_info["SOLAR_DM2"],
                analysis_info["SIN13"],
                analysis_info["SIN12"],
            )
            return ([found_dm2], [found_sin13], [found_sin12])

        if dm2 is None: dm2 = found_dm2
        if sin13 is None: sin13 = found_sin13
        if sin12 is None: sin12 = found_sin12

        if isinstance(dm2, float): dm2 = [dm2]
        if isinstance(sin13, float): sin13 = [sin13]
        if isinstance(sin12, float): sin12 = [sin12]

        if type(dm2) == list and type(sin13) == list and type(sin12) == list:
            filtered_dm2, filtered_sin13, filtered_sin12 = [], [], []
            for this_dm2, this_sin13, this_sin12 in zip(found_dm2, found_sin13, found_sin12):
                if this_dm2 in dm2 and this_sin13 in sin13 and this_sin12 in sin12:
                    filtered_dm2.append(this_dm2)
                    filtered_sin13.append(this_sin13)
                    filtered_sin12.append(this_sin12)

            found_dm2, found_sin13, found_sin12 = filtered_dm2, filtered_sin13, filtered_sin12

        else:
            rprint(f"[red]ERROR: oscillation parameters must be floats or lists![/red]")
            raise TypeError

    if type(auto) != bool:
        rprint(f"[red]ERROR: auto must be a boolean![/red]")
        raise TypeError

    if type(found_dm2) == list:
        if debug: rprint(f"[cyan]Found {len(found_dm2)} oscillation files![/cyan]")

    return (found_dm2, found_sin13, found_sin12)


def _get_oscillation_map_computed(
    dm2=None, sin13=None, sin12=None,
    output="df", backend="prob3",
    separate_day_night=False, debug=False,
):
    """
    Compute oscillation maps on-the-fly using prob3 or nufast backend.
    Returns same dict structure as get_oscillation_map(backend="file").
    """
    from lib.oscillation_backends import (
        compute_prob3, compute_nufast,
        get_nadir_pdf_file, get_nadir_pdf_nufast,
        combine_day_night,
    )
    analysis_info = load_analysis_info(str(root))

    if dm2 is None:   dm2   = [analysis_info["SOLAR_DM2"]]
    if sin13 is None: sin13 = [analysis_info["SIN13"]]
    if sin12 is None: sin12 = [analysis_info["SIN12"]]
    if isinstance(dm2,   float): dm2   = [dm2]
    if isinstance(sin13, float): sin13 = [sin13]
    if isinstance(sin12, float): sin12 = [sin12]

    e_range = analysis_info.get("OSC_ENERGY_RANGE", analysis_info.get("RECO_ENERGY_RANGE", [0, 30]))
    e_bins  = analysis_info.get("OSC_ENERGY_BINS",   120)
    energy_edges = np.linspace(e_range[0], e_range[1], e_bins + 1)
    nadir_edges = np.linspace(-1.0, 1.0, analysis_info["NADIR_BINS"] + 1)
    nadir_centers = 0.5 * (nadir_edges[1:] + nadir_edges[:-1])

    latitude_deg = analysis_info.get("DUNE_LATITUDE_DEG", 44.35)

    # Use ROOT file nadir PDF for all backends to maintain backward compatibility
    # File backend interpolates 2000-bin ROOT histogram; nufast direct computation differs
    try:
        nadir_pdf = get_nadir_pdf_file(nadir_centers=nadir_centers)
    except Exception:
        # Fallback to nufast if ROOT not available
        if backend == "nufast":
            nadir_pdf = get_nadir_pdf_nufast(nadir_centers, latitude_deg)
        else:
            nadir_pdf = get_nadir_pdf_nufast(nadir_centers, latitude_deg)

    result_dict = {}
    for dm2_v, sin13_v, sin12_v in zip(dm2, sin13, sin12):
        key = (float("%.3e" % dm2_v), sin13_v, float("%.3e" % sin12_v))
        if debug:
            print_colored(f"Computing oscillogram [{backend}]: dm2={key[0]:.3e} sin13={key[1]:.3e} sin12={key[2]:.3e}", "DEBUG")

        if backend == "prob3":
            osc = compute_prob3(key[0], key[1], key[2], energy_edges, nadir_edges)
        else:
            osc = compute_nufast(key[0], key[1], key[2], energy_edges, nadir_edges,
                                 latitude_deg=latitude_deg)

        if separate_day_night:
            result_dict[key] = osc
        else:
            df = combine_day_night(osc, nadir_pdf)  # mirrors process_oscillation_map()
            if output in ("interp1d", "interp2d"):
                from scipy import interpolate as _interp
                osc_map_x = df.columns.to_list()
                osc_map_y = df.index.to_list()
                if output == "interp1d":
                    result_dict[key] = _interp.interp1d(
                        osc_map_x, np.sum(df.values, axis=0),
                        kind="linear", fill_value="extrapolate",
                    )
                else:
                    result_dict[key] = _interp.RegularGridInterpolator(
                        (osc_map_x, osc_map_y), df.to_numpy().T,
                        method="linear", bounds_error=False, fill_value=1e-6,
                    )
            else:
                result_dict[key] = df

    return result_dict


def get_oscillation_map(
    path: str = f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/data/OSCILLATION/",
    dm2: Optional[list[float]] = None,
    sin13: Optional[list[float]] = None,
    sin12: Optional[list[float]] = None,
    ext: Optional[str] = "root",
    auto: bool = False,
    rebin: bool = False,
    rw: bool = False,
    output: str = "df",
    save: bool = False,
    debug: bool = False,
    backend: str = "file",
    separate_day_night: bool = False,
):
    """
    This function can be used to obtain the oscillation correction for DUNE's solar analysis.

    Args:
        path (str): Path to the oscillation data files (default: f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/data/OSCILLATION/")
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

    # ── Non-file backends ────────────────────────────────────────────────────
    if backend in ("prob3", "nufast"):
        return _get_oscillation_map_computed(
            dm2=dm2, sin13=sin13, sin12=sin12,
            output=output, backend=backend,
            separate_day_night=separate_day_night,
            debug=debug,
        )

    df_dict = {}
    interp_dict = {}

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

    if type(dm2) == float and type(sin13) == float and type(sin12) == float:
        dm2, sin13, sin12 = [dm2], [sin13], [sin12]

    for dm2_value, sin13_value, sin12_value in zip(dm2, sin13, sin12):
        # Format dm2 and sin12 values to be used in the file name with appropriate precision
        dm2_value = float("%.3e" % dm2_value)
        sin12_value = float("%.3e" % sin12_value)

        if ext == "pkl":
            if (
                glob.glob(
                    f"{path}{ext}/{subfolder}/osc_probability_dm2_{dm2_value:.3e}_sin13_{sin13_value:.3e}_sin12_{sin12_value:.3e}.pkl"
                )
                != []
            ):
                if debug: rprint(f"Loading data from: {path}{ext}/{subfolder}/osc_probability_dm2_{dm2_value:.3e}_sin13_{sin13_value:.3e}_sin12_{sin12_value:.3e}.pkl")
                df = pd.read_pickle(f"{path}{ext}/{subfolder}/osc_probability_dm2_{dm2_value:.3e}_sin13_{sin13_value:.3e}_sin12_{sin12_value:.3e}.pkl")

                save_path = f"{path}{ext}/rebin/osc_probability_dm2_{dm2_value:.3e}_sin13_{sin13_value:.3e}_sin12_{sin12_value:.3e}.pkl"
                if rebin and glob.glob(save_path) != [] and rw == False:
                    if debug: rprint(f"Loading rebinned data from {save_path}")
                    df = pd.read_pickle(save_path)
                
                elif rebin:
                    if debug: rprint(f"[green]Saving rebinned oscillation data dm2 = {dm2_value:.3e}, sin13 = {sin13_value:.3e}, sin12 = {sin12_value:.3e}[/green]")
                    df = rebin_df(df, show=False, save=save, save_path=save_path, debug=debug)

            else:
                rprint(f"ERROR: file {path}osc_probability_dm2_{dm2_value:.3e}_sin13_{sin13_value:.3e}_sin12_{sin12_value:.3e}.pkl not found!")
                return None

        elif ext == "root":
            if debug: rprint(f"Loading raw data from: {path}/root/osc_probability_dm2_{dm2_value:.3e}_sin13_{sin13_value:.3e}_sin12_{sin12_value:.3e}.{ext}")

            df = process_oscillation_map(
                path,
                dm2_value,
                sin13_value,
                sin12_value,
                ext=ext,
                convolve=True,
                debug=debug,
            )

            save_path = (f"{path}/pkl/rebin/osc_probability_dm2_{dm2_value:.3e}_sin13_{sin13_value:.3e}_sin12_{sin12_value:.3e}.pkl")
            if rebin and glob.glob(save_path) != [] and rw == False:
                if debug: rprint(f"Loading rebinned data from {save_path}")
                df = pd.read_pickle(save_path)
            
            elif rebin:
                if debug: rprint(f"[green]Saving rebinned oscillation data dm2 = {dm2_value:.3e}, sin13 = {sin13_value:.3e}, sin12 = {sin12_value:.3e}[/green]") 
                df = rebin_df(df, show=False, save=save, save_path=save_path, debug=debug)

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

    if output in ["interp1d", "interp2d"]:
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
    path=f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/data/OSCILLATION/",
    dm2_value=None,
    sin13_value=None,
    sin12_value=None,
    ext="root",
    convolve=True,
    debug=False,
):
    nadir_data = get_nadir_angle(path=path, debug=debug)
    analysis_info = load_analysis_info(str(root))
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
        nadir_interp = interpolate.interp1d(
            nadir_data[0], nadir_data[1], kind="linear", fill_value="extrapolate"
        )
        nadir_y = nadir_interp(x=root_nadir_centers)
        # normalize nadir distribution
        nadir_y = nadir_y / nadir_y.sum()
        df = df.mul(nadir_y, axis=0)

    return df


def plot_oscillation_map(fig, idx, dm2=None, sin13=None, sin12=None, factor=1, debug=False):
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
    analysis_info = load_analysis_info(str(root))
    oscillation_map = get_oscillation_map(
        dm2=dm2, sin13=sin13, sin12=sin12, debug=debug
    )
    for this_dm2, this_sin13, this_sin12 in oscillation_map.keys():
        df = oscillation_map[(this_dm2, this_sin13, this_sin12)]
        fig.add_trace(
            go.Heatmap(
                z=df*factor,
                x=df.columns,
                y=df.index,
                colorscale="turbo",
                # coloraxis="coloraxis" + str(idx[1]),
                colorbar=dict(title="Osc. PDF"),
                colorbar_x=1,
                coloraxis="coloraxis",
            ),
            row=idx[0],
            col=idx[1],
        )

    if debug:
        rprint("Oscillation map plotted!")
    return fig


def rebin_df(
    df,
    save_path: Optional[str] = None,
    xarray=[],
    yarray=[],
    convolve=True,
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
        save_path (str): Path to save the rebinning result.
        debug (bool): If True, the debug mode is activated.

    Returns:
        small_df (pandas.DataFrame): Rebinning result.
    """
    analysis_info = load_analysis_info(str(root))
    energy_edges = np.linspace(
        analysis_info["OSC_ENERGY_RANGE"][0],
        analysis_info["OSC_ENERGY_RANGE"][1],
        analysis_info["OSC_ENERGY_BINS"] + 1,
    )
    energy_centers = (energy_edges[1:] + energy_edges[:-1]) / 2

    df.index = df.index.astype(float)
    if xarray == [] and yarray == []:
        reduced_rows_edges = np.linspace(-1, 1, analysis_info["NADIR_BINS"] + 1, endpoint=True)
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
            
            if convolve:
                small_df.loc[row, col] = (
                    df.loc(axis=1)[start_col:end_col]
                    .loc(axis=0)[start_row:end_row]
                    .sum()
                    .sum()
                )
            else:
                small_df.loc[row, col] = (
                    df.loc(axis=1)[start_col:end_col]
                    .loc(axis=0)[start_row:end_row]
                    .mean()
                    .mean()
                )
    # Substitute NaN values with 0
    small_df = small_df.fillna(0).infer_objects(copy=False)

    # Print the reduced data frame
    if show:
        fig = px.imshow(
            small_df,
            aspect="auto",
            origin="lower",
            color_continuous_scale="turbo",
            title="Oscillation Correction Map",
            labels=dict(y=f"Azimuth Angle {unicode('eta')}", x="TrueEnergy"),
        )
        fig = format_coustom_plotly(fig)
        fig.show()

    if save:
        # Check if the file already exists
        if os.path.exists(save_path):
            os.remove(save_path)
        small_df.to_pickle(save_path)
        # Give permissions to the file
        os.system(f"chmod 777 {save_path}")

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


def make_oscillation_map_plot(dm2=None, sin13=None, sin12=None, factor=1, debug=False):
    fig = make_subplots(
        rows=1,
        cols=3,
        horizontal_spacing=0.1,
        subplot_titles=(
            f"DUNE's FD Yearly Azimuth Angle",
            f"Survival Probability",
            f"Convolved Probability * {factor}",
        ),
    )

    fig = plot_nadir_angle(fig, (1, 1), norm=True, debug=debug)

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
            coloraxis="coloraxis",
        ),
        row=1,
        col=2,
    )

    fig = plot_oscillation_map(
        fig, (1, 3), dm2=dm2, sin13=sin13, sin12=sin12, factor=factor, debug=debug
    )
    fig.update_coloraxes(colorbar=dict(title="Events"), colorbar_x=0.9, row=1, col=3)

    fig = format_coustom_plotly(
        fig, matches=(None, None), tickformat=(None, None), add_units=False
    )
    fig.update_xaxes(title_text="Azimuth Angle cos(" + unicode("eta") + ")", row=1, col=1)
    fig.update_yaxes(range=[0.3, 1.09], title_text="Norm.", row=1, col=1)
    
    fig.update_xaxes(title_text="True Neutrino Energy (MeV)", row=1, col=2)
    fig.update_yaxes(title_text="Azimuth Angle cos(" + unicode("eta") + ")", row=1, col=2)
    
    fig.update_xaxes(title_text="True Neutrino Energy (MeV)", row=1, col=3)
    fig.update_yaxes(title_text="Azimuth Angle cos(" + unicode("eta") + ")", row=1, col=3)

    return fig