"""
Oscillogram backend abstraction layer.

Two backends produce P_ee(E, cos_nadir) from first principles:
  - "prob3"  : Prob3++ BargerPropagator (same algorithm as original ROOT library)
  - "nufast" : NuFast-Earth Probability_Engine (independent, faster)

Both expose an OscResult with separate day/night DataFrames.
combine_day_night() collapses them into the same format as process_oscillation_map()
so existing callers need zero changes.

Convention (identical in both backends and lib_osc.py):
  DataFrame index  = cos(η) centers  [nadir of neutrino direction]
  DataFrame columns = E_MeV centers
  cos(η) < 0 : night (Earth-traversing)
  cos(η) > 0 : day
"""

import sys
import os
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import interpolate

# Locate compiled binding modules
_PROB3_DIR   = os.path.join(os.path.dirname(__file__), '..', 'external', 'Prob3plusplus',  'python')
_NUFAST_DIR  = os.path.join(os.path.dirname(__file__), '..', 'external', 'NuFast-Earth',   'python')


def _load_prob3():
    if _PROB3_DIR not in sys.path:
        sys.path.insert(0, _PROB3_DIR)
    try:
        import _prob3_solar
        return _prob3_solar
    except ImportError as e:
        raise ImportError(
            f"Prob3++ binding not found. Build with:\n"
            f"  cd external/Prob3plusplus/python && make\n"
            f"Original error: {e}"
        ) from e


def _load_nufast():
    if _NUFAST_DIR not in sys.path:
        sys.path.insert(0, _NUFAST_DIR)
    try:
        import _nufast_earth
        return _nufast_earth
    except ImportError as e:
        raise ImportError(
            f"NuFast-Earth binding not found. Build with:\n"
            f"  cd external/NuFast-Earth/python && make\n"
            f"Original error: {e}"
        ) from e


@dataclass
class OscResult:
    """
    Separate day/night survival probability maps.

    day:   DataFrame  columns=E_MeV              P_ee for day (energy-only, uniform in nadir)
    night: DataFrame  index=cos(η), columns=E_MeV P_ee for night (Earth matter effect)

    Both have float64 values in [0, 1].
    """
    day:   pd.DataFrame
    night: pd.DataFrame


def _make_energy_nadir_grids(energy_edges_mev: np.ndarray, nadir_edges: np.ndarray):
    """Return (energy_centers, nadir_centers) from edges."""
    e_centers = 0.5 * (energy_edges_mev[1:] + energy_edges_mev[:-1])
    n_centers = 0.5 * (nadir_edges[1:]      + nadir_edges[:-1])
    return e_centers, n_centers


def compute_prob3(
    dm2:              float,
    sin13:            float,
    sin12:            float,
    energy_edges_mev: np.ndarray,
    nadir_edges:      np.ndarray,
) -> OscResult:
    """
    Compute P_ee using Prob3++ BargerPropagator.

    Replicates solar.cc: same Earth model, same production fractions,
    same parameter convention as the original ROOT template library.

    Parameters
    ----------
    dm2   : Δm²₂₁ in eV²
    sin13 : sin²(θ₁₃)
    sin12 : sin²(θ₁₂)
    energy_edges_mev : bin edges for energy axis in MeV
    nadir_edges      : bin edges for cos(η) axis

    Returns
    -------
    OscResult with day/night DataFrames
    """
    prob3 = _load_prob3()
    e_centers, n_centers = _make_energy_nadir_grids(energy_edges_mev, nadir_edges)

    pee_2d = prob3.compute_solar_oscillogram(
        sin12, sin13, dm2,
        e_centers * 1e-3,   # MeV → GeV
        n_centers,
    )
    # pee_2d shape: [n_energy, n_nadir]

    # Split on nadir sign: day = cos(η) > 0, night = cos(η) < 0
    day_mask   = n_centers >  0.0
    night_mask = n_centers <= 0.0

    # Day: average over day-nadir bins → energy-only Series
    day_pee   = pee_2d[:, day_mask].mean(axis=1) if day_mask.any() else pee_2d[:, 0]
    day_df    = pd.DataFrame(
        day_pee[np.newaxis, :],
        index=pd.Index([0.0], name='cos_nadir'),
        columns=pd.Index(e_centers, name='E_MeV'),
    )

    # Night: keep all nadir bins (including day bins for combined use downstream)
    night_df  = pd.DataFrame(
        pee_2d.T,
        index=pd.Index(n_centers, name='cos_nadir'),
        columns=pd.Index(e_centers, name='E_MeV'),
    )

    return OscResult(day=day_df, night=night_df)


def compute_nufast(
    dm2:              float,
    sin13:            float,
    sin12:            float,
    energy_edges_mev: np.ndarray,
    nadir_edges:      np.ndarray,
    latitude_deg:     float = 44.35,
    rhoYe_sun:        float = 90.0,
    detector_depth_km: float = 1.478,
) -> OscResult:
    """
    Compute P_ee using NuFast-Earth Probability_Engine.

    Parameters
    ----------
    dm2   : Δm²₂₁ in eV²
    sin13 : sin²(θ₁₃)
    sin12 : sin²(θ₁₂)
    energy_edges_mev : bin edges for energy axis in MeV
    nadir_edges      : bin edges for cos(η) axis
    latitude_deg     : detector latitude in degrees (DUNE FD SURF = 44.35°N)
    rhoYe_sun        : solar production density × Ye in g/cm³ (default 90.0 = Prob3++ match)
    detector_depth_km: detector depth in km (SURF ≈ 1.478 km)

    Returns
    -------
    OscResult with day/night DataFrames
    """
    nufast = _load_nufast()
    e_centers, n_centers = _make_energy_nadir_grids(energy_edges_mev, nadir_edges)
    e_gev = e_centers * 1e-3

    # Day: energy-only (no nadir dependence in vacuum+Sun)
    day_pee = nufast.compute_solar_day(sin12, sin13, dm2, e_gev, rhoYe_sun)
    day_df  = pd.DataFrame(
        day_pee[np.newaxis, :],
        index=pd.Index([0.0], name='cos_nadir'),
        columns=pd.Index(e_centers, name='E_MeV'),
    )

    # Night: Earth traversal for cosz < 0
    night_mask   = n_centers < 0.0
    night_coszs  = n_centers[night_mask]
    night_pee_2d = nufast.compute_solar_night(
        sin12, sin13, dm2, e_gev, night_coszs, rhoYe_sun, detector_depth_km
    )
    # night_pee_2d shape: [n_energy, n_night_cosz]

    # Build full [n_nadir, n_energy] DataFrame for all nadir bins.
    # Day bins (cosz > 0) filled from compute_solar_day (no Earth effect).
    full_pee = np.empty((len(n_centers), len(e_centers)))
    full_pee[night_mask, :] = night_pee_2d.T
    full_pee[~night_mask, :] = day_pee[np.newaxis, :]  # broadcast day to all day-bins

    night_df = pd.DataFrame(
        full_pee,
        index=pd.Index(n_centers, name='cos_nadir'),
        columns=pd.Index(e_centers, name='E_MeV'),
    )

    return OscResult(day=day_df, night=night_df)


def get_nadir_pdf_file(
    path: str = "/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/data/OSCILLATION/",
    nadir_centers: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Load nadir angle PDF from nadir.root and interpolate to nadir_centers.

    Used by 'file' and 'prob3' backends.
    """
    import uproot
    with uproot.open(path + "nadir.root") as f:
        pdf = f["nadir;1"]
        arr = pdf.to_hist().to_numpy()
        xedges = arr[1]
        xcenters = 0.5 * (xedges[1:] + xedges[:-1])
        yvalues = arr[0].astype(float)

    if nadir_centers is None:
        return yvalues / yvalues.sum()

    interp = interpolate.interp1d(xcenters, yvalues, kind='linear', fill_value=0.0, bounds_error=False)
    pdf_interp = interp(nadir_centers)
    pdf_interp = np.clip(pdf_interp, 0.0, None)
    total = pdf_interp.sum()
    return pdf_interp / total if total > 0 else pdf_interp


def get_nadir_pdf_nufast(
    nadir_centers: np.ndarray,
    latitude_deg:  float = 44.35,
) -> np.ndarray:
    """
    Compute full nadir angle PDF using NuFast-Earth Solar_Weight.

    Solar_Weight(eta, lat) covers only eta ∈ [0, π/2] (Sun above horizon).
    For night bins (cos_nadir < 0), the Sun is below the horizon; by symmetry
    of the yearly trajectory, Solar_Weight(arccos(|cos_nadir|), lat) is used.

    This yields a symmetric PDF matching the nadir.root distribution structure.

    Returns normalised weights over nadir_centers.
    """
    nufast = _load_nufast()
    lat_rad = latitude_deg * math.pi / 180.0
    # eta always in [0, π/2]: use |cos_nadir| so both day and night map to [0, π/2]
    etas = np.arccos(np.abs(nadir_centers))
    weights = nufast.solar_weight_array(etas, lat_rad)
    weights = np.clip(weights, 0.0, None)
    total = weights.sum()
    return weights / total if total > 0 else weights


def combine_day_night(
    result:    OscResult,
    nadir_pdf: np.ndarray,
) -> pd.DataFrame:
    """
    Combine OscResult into a nadir-weighted DataFrame matching process_oscillation_map().

    Mirrors process_oscillation_map(convolve=True) exactly:
      result.night contains P_ee(E, cos_nadir) for ALL nadir bins (both day and night).
      Simply multiply each nadir row by the normalised nadir PDF.

    The day/night distinction is already encoded in result.night:
      cos_nadir > 0: day P_ee (from Sun-matter MSW, no Earth)
      cos_nadir < 0: night P_ee (Sun-matter MSW + Earth regeneration)

    Parameters
    ----------
    result    : OscResult from compute_prob3 or compute_nufast
    nadir_pdf : 1D normalised PDF over nadir_centers (same ordering as result.night.index)

    Returns
    -------
    pd.DataFrame  index=cos_nadir, columns=E_MeV  (same structure as pkl files)
    """
    night_df  = result.night
    n_centers = night_df.index.values
    e_centers = night_df.columns.values
    weighted  = night_df.values.astype(float) * nadir_pdf[:, np.newaxis]
    return pd.DataFrame(
        weighted,
        index=pd.Index(n_centers, name='cos_nadir'),
        columns=pd.Index(e_centers, name='E_MeV'),
    )
