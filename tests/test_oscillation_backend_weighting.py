#!/usr/bin/env python3
"""Compare oscillation weighting across all backends at old best-fit parameters.

Tests that file, prob3, and nufast backends produce identical weighted oscillation
when using the same best-fit dm2, sin13, sin12 values.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, '/pc/choozdsk01/users/manthey/SOLAR')

from lib.defaults import load_analysis_info
from lib.oscillation_backends import (
    get_nadir_pdf_file, get_nadir_pdf_nufast,
    combine_day_night,
    compute_prob3, compute_nufast,
)

print("=" * 80)
print("OSCILLATION BACKEND WEIGHTING COMPARISON")
print("=" * 80)

# Load best-fit parameters (old values, what ROOT files use)
root = '/pc/choozdsk01/users/manthey/SOLAR'
analysis_info = load_analysis_info(root)

dm2 = analysis_info["SOLAR_DM2"]
sin13 = analysis_info["SIN13"]
sin12 = analysis_info["SIN12"]

print(f"\nBest-fit parameters:")
print(f"  dm2={dm2:.3e}, sin13={sin13:.3f}, sin12={sin12:.3f}")

# Setup grids
nadir_edges = np.linspace(-1.0, 1.0, 41)
nadir_centers = 0.5 * (nadir_edges[1:] + nadir_edges[:-1])

energy_edges = np.linspace(0, 30, 121)
energy_centers = 0.5 * (energy_edges[1:] + energy_edges[:-1])

print(f"\nGrids:")
print(f"  Energy: {len(energy_edges)-1} bins [0, 30] MeV")
print(f"  Nadir:  {len(nadir_edges)-1} bins [-1, 1]")

# Load/compute nadir PDFs
try:
    nadir_pdf_file = get_nadir_pdf_file(nadir_centers=nadir_centers)
    print(f"\n✓ File nadir PDF: sum={nadir_pdf_file.sum():.6f}, shape={nadir_pdf_file.shape}")
except Exception as e:
    print(f"\n✗ File nadir PDF error: {e}")
    nadir_pdf_file = None

try:
    nadir_pdf_nufast = get_nadir_pdf_nufast(nadir_centers, latitude_deg=44.35)
    print(f"✓ NuFast nadir PDF: sum={nadir_pdf_nufast.sum():.6f}, shape={nadir_pdf_nufast.shape}")
except Exception as e:
    print(f"✗ NuFast nadir PDF error: {e}")
    nadir_pdf_nufast = None

# Test each backend
results = {}

# 1. FILE BACKEND - load from ROOT pkl
print("\n" + "=" * 80)
print("FILE BACKEND (ROOT oscillation pkl)")
print("=" * 80)

try:
    from lib.oscillation import get_oscillation_map
    osc_map_file = get_oscillation_map(
        dm2=dm2,
        sin13=sin13,
        sin12=sin12,
        backend="file",
        output="df",
        debug=False,
    )
    key = (float("%.3e" % dm2), sin13, float("%.3e" % sin12))
    osc_file_df = osc_map_file[key]

    print(f"Shape: {osc_file_df.shape}")
    print(f"Sum: {osc_file_df.values.sum():.6e}")
    print(f"Mean: {np.nanmean(osc_file_df.values):.6e}")
    print(f"Min/Max: {np.nanmin(osc_file_df.values):.6e} / {np.nanmax(osc_file_df.values):.6e}")

    results['file'] = {
        'df': osc_file_df,
        'sum': osc_file_df.values.sum(),
        'mean': np.nanmean(osc_file_df.values),
    }
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# 2. PROB3 BACKEND - compute from scratch
print("\n" + "=" * 80)
print("PROB3 BACKEND (computed)")
print("=" * 80)

try:
    osc_prob3 = compute_prob3(dm2, sin13, sin12, energy_edges_mev=energy_edges, nadir_edges=nadir_edges)

    # Use file nadir PDF if available, else nufast
    nadir_pdf = nadir_pdf_file if nadir_pdf_file is not None else nadir_pdf_nufast

    if nadir_pdf is None:
        raise ValueError("No nadir PDF available")

    osc_prob3_df = combine_day_night(osc_prob3, nadir_pdf)

    print(f"Shape: {osc_prob3_df.shape}")
    print(f"Sum: {osc_prob3_df.values.sum():.6e}")
    print(f"Mean: {np.nanmean(osc_prob3_df.values):.6e}")
    print(f"Min/Max: {np.nanmin(osc_prob3_df.values):.6e} / {np.nanmax(osc_prob3_df.values):.6e}")

    results['prob3'] = {
        'df': osc_prob3_df,
        'sum': osc_prob3_df.values.sum(),
        'mean': np.nanmean(osc_prob3_df.values),
    }
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# 3. NUFAST BACKEND - compute from scratch
print("\n" + "=" * 80)
print("NUFAST BACKEND (computed)")
print("=" * 80)

try:
    osc_nufast = compute_nufast(dm2, sin13, sin12, energy_edges_mev=energy_edges, nadir_edges=nadir_edges, latitude_deg=44.35)

    # Use file nadir PDF if available, else nufast
    nadir_pdf = nadir_pdf_file if nadir_pdf_file is not None else nadir_pdf_nufast

    if nadir_pdf is None:
        raise ValueError("No nadir PDF available")

    osc_nufast_df = combine_day_night(osc_nufast, nadir_pdf)

    print(f"Shape: {osc_nufast_df.shape}")
    print(f"Sum: {osc_nufast_df.values.sum():.6e}")
    print(f"Mean: {np.nanmean(osc_nufast_df.values):.6e}")
    print(f"Min/Max: {np.nanmin(osc_nufast_df.values):.6e} / {np.nanmax(osc_nufast_df.values):.6e}")

    results['nufast'] = {
        'df': osc_nufast_df,
        'sum': osc_nufast_df.values.sum(),
        'mean': np.nanmean(osc_nufast_df.values),
    }
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# COMPARISON
print("\n" + "=" * 80)
print("BACKEND COMPARISON")
print("=" * 80)

if len(results) >= 2:
    backends = list(results.keys())

    for i, b1 in enumerate(backends):
        for b2 in backends[i+1:]:
            ratio_sum = results[b1]['sum'] / results[b2]['sum'] if results[b2]['sum'] != 0 else float('inf')
            ratio_mean = results[b1]['mean'] / results[b2]['mean'] if results[b2]['mean'] != 0 else float('inf')

            print(f"\n{b1.upper()} vs {b2.upper()}:")
            print(f"  Sum ratio ({b1}/{b2}):  {ratio_sum:.6f}")
            print(f"  Mean ratio ({b1}/{b2}): {ratio_mean:.6f}")

            if abs(ratio_sum - 1.0) < 0.01 and abs(ratio_mean - 1.0) < 0.01:
                print(f"  ✓ MATCH (within 1%)")
            elif abs(ratio_sum - 1.0) < 0.05 and abs(ratio_mean - 1.0) < 0.05:
                print(f"  ⚠ CLOSE (within 5%)")
            else:
                print(f"  ✗ DIVERGE ({abs(ratio_sum-1.0)*100:.1f}% off)")

    # Element-wise comparison if both file and computed available
    if 'file' in results and ('prob3' in results or 'nufast' in results):
        computed = results.get('prob3') or results.get('nufast')
        backend_name = 'prob3' if 'prob3' in results else 'nufast'

        print(f"\n" + "=" * 80)
        print(f"ELEMENT-WISE: FILE vs {backend_name.upper()}")
        print("=" * 80)

        file_vals = results['file']['df'].values
        comp_vals = computed['df'].values

        if file_vals.shape == comp_vals.shape:
            max_diff = np.nanmax(np.abs(file_vals - comp_vals))
            max_rel_diff = np.nanmax(np.abs((file_vals - comp_vals) / (file_vals + 1e-10)))

            print(f"Max absolute difference: {max_diff:.6e}")
            print(f"Max relative difference: {max_rel_diff:.6f}")

            if max_rel_diff < 0.01:
                print(f"✓ Element-wise MATCH (within 1%)")
            elif max_rel_diff < 0.05:
                print(f"⚠ Element-wise CLOSE (within 5%)")
            else:
                print(f"✗ Element-wise DIVERGE ({max_rel_diff*100:.1f}%)")
        else:
            print(f"Shape mismatch: {file_vals.shape} vs {comp_vals.shape}")

else:
    print("Insufficient backends succeeded for comparison")

print("\n" + "=" * 80)
