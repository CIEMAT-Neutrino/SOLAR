#!/usr/bin/env python3
"""Compare weighting schemes across backends at same resolution.

Computes oscillation with prob3 and nufast backends at the same grid resolution,
applies nadir PDF weighting, and verifies they produce equivalent results.

This validates that backend choice doesn't affect the final weighted oscillation
when using the same oscillation parameters and nadir PDF source.
"""

import numpy as np
import pandas as pd
import sys

sys.path.insert(0, '/pc/choozdsk01/users/manthey/SOLAR')

from lib.defaults import load_analysis_info
from lib.oscillation_backends import (
    compute_prob3, compute_nufast,
    get_nadir_pdf_file, get_nadir_pdf_nufast,
    combine_day_night,
)

print("=" * 80)
print("OSCILLATION BACKEND WEIGHTING COMPARISON")
print("=" * 80)

root = '/pc/choozdsk01/users/manthey/SOLAR'
analysis_info = load_analysis_info(root)

dm2 = analysis_info["SOLAR_DM2"]
sin13 = analysis_info["SIN13"]
sin12 = analysis_info["SIN12"]

print(f"\nBest-fit parameters (from config):")
print(f"  dm2={dm2:.3e}, sin13={sin13:.3f}, sin12={sin12:.3f}")

nadir_bins = analysis_info.get("NADIR_BINS", 40)
energy_bins = analysis_info.get("OSC_ENERGY_BINS", 120)
energy_range = analysis_info.get("OSC_ENERGY_RANGE", [0, 30])

print(f"\nAnalysis grid:")
print(f"  Energy: {energy_bins} bins [{energy_range[0]}, {energy_range[1]}] MeV")
print(f"  Nadir:  {nadir_bins} bins [-1, 1]")

# Setup grids
energy_edges = np.linspace(energy_range[0], energy_range[1], energy_bins + 1)
nadir_edges = np.linspace(-1, 1, nadir_bins + 1)
nadir_centers = 0.5 * (nadir_edges[:-1] + nadir_edges[1:])

# Load nadir PDFs (use FILE as reference)
print(f"\n" + "=" * 80)
print("NADIR PDFs")
print("=" * 80)

try:
    nadir_pdf_file = get_nadir_pdf_file(nadir_centers=nadir_centers)
    print(f"\n✓ File nadir PDF: sum={nadir_pdf_file.sum():.6f}")
except Exception as e:
    print(f"\n✗ File nadir PDF error: {e}")
    nadir_pdf_file = None

try:
    nadir_pdf_nufast = get_nadir_pdf_nufast(nadir_centers, latitude_deg=44.35)
    print(f"✓ NuFast nadir PDF: sum={nadir_pdf_nufast.sum():.6f}")
except Exception as e:
    print(f"✗ NuFast nadir PDF error: {e}")
    nadir_pdf_nufast = None

if nadir_pdf_file is not None and nadir_pdf_nufast is not None:
    ratio = nadir_pdf_file.sum() / nadir_pdf_nufast.sum() if nadir_pdf_nufast.sum() != 0 else float('inf')
    print(f"  Ratio (file/nufast): {ratio:.6f}")

# Results storage
results = {}

# ============================================================================
# PROB3 BACKEND
# ============================================================================
print(f"\n" + "=" * 80)
print("PROB3 BACKEND")
print("=" * 80)

try:
    osc_prob3 = compute_prob3(dm2, sin13, sin12, energy_edges_mev=energy_edges, nadir_edges=nadir_edges)

    print(f"Raw probability:")
    print(f"  Shape: {osc_prob3.night.shape}")
    print(f"  Sum: {osc_prob3.night.values.sum():.6e}")
    print(f"  Mean: {np.mean(osc_prob3.night.values):.6f}")
    print(f"  Min/Max: {np.min(osc_prob3.night.values):.6f} / {np.max(osc_prob3.night.values):.6f}")

    # Apply FILE nadir PDF weighting
    if nadir_pdf_file is not None:
        osc_prob3_weighted = combine_day_night(osc_prob3, nadir_pdf_file)
        print(f"\nAfter FILE nadir weighting:")
        print(f"  Sum: {osc_prob3_weighted.values.sum():.6e}")
        print(f"  Mean: {np.mean(osc_prob3_weighted.values):.6f}")
        results['prob3_file_pdf'] = osc_prob3_weighted

    # Also apply NuFast nadir PDF for comparison
    if nadir_pdf_nufast is not None:
        osc_prob3_weighted_nufast = combine_day_night(osc_prob3, nadir_pdf_nufast)
        print(f"\nAfter NUFAST nadir weighting:")
        print(f"  Sum: {osc_prob3_weighted_nufast.values.sum():.6e}")
        print(f"  Mean: {np.mean(osc_prob3_weighted_nufast.values):.6f}")
        results['prob3_nufast_pdf'] = osc_prob3_weighted_nufast

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# NUFAST BACKEND
# ============================================================================
print(f"\n" + "=" * 80)
print("NUFAST BACKEND")
print("=" * 80)

try:
    osc_nufast = compute_nufast(dm2, sin13, sin12, energy_edges_mev=energy_edges, nadir_edges=nadir_edges, latitude_deg=44.35)

    print(f"Raw probability:")
    print(f"  Shape: {osc_nufast.night.shape}")
    print(f"  Sum: {osc_nufast.night.values.sum():.6e}")
    print(f"  Mean: {np.mean(osc_nufast.night.values):.6f}")
    print(f"  Min/Max: {np.min(osc_nufast.night.values):.6f} / {np.max(osc_nufast.night.values):.6f}")

    # Apply FILE nadir PDF weighting
    if nadir_pdf_file is not None:
        osc_nufast_weighted = combine_day_night(osc_nufast, nadir_pdf_file)
        print(f"\nAfter FILE nadir weighting:")
        print(f"  Sum: {osc_nufast_weighted.values.sum():.6e}")
        print(f"  Mean: {np.mean(osc_nufast_weighted.values):.6f}")
        results['nufast_file_pdf'] = osc_nufast_weighted

    # Also apply NuFast nadir PDF for comparison
    if nadir_pdf_nufast is not None:
        osc_nufast_weighted_nufast = combine_day_night(osc_nufast, nadir_pdf_nufast)
        print(f"\nAfter NUFAST nadir weighting:")
        print(f"  Sum: {osc_nufast_weighted_nufast.values.sum():.6e}")
        print(f"  Mean: {np.mean(osc_nufast_weighted_nufast.values):.6f}")
        results['nufast_nufast_pdf'] = osc_nufast_weighted_nufast

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# COMPARISON
# ============================================================================
print(f"\n" + "=" * 80)
print("WEIGHTING SCHEME COMPARISON")
print("=" * 80)

if len(results) >= 2:
    result_keys = list(results.keys())

    # Test 1: Same backend, different PDF
    if 'prob3_file_pdf' in results and 'prob3_nufast_pdf' in results:
        v_file = results['prob3_file_pdf'].values.sum()
        v_nufast = results['prob3_nufast_pdf'].values.sum()
        ratio = v_file / v_nufast if v_nufast != 0 else float('inf')

        print(f"\nPROB3 backend, FILE vs NUFAST PDF:")
        print(f"  Ratio (file_pdf/nufast_pdf): {ratio:.6f}")
        if abs(ratio - 1.0) < 0.05:
            print(f"  ✓ PDFs equivalent (within 5%)")
        else:
            print(f"  ⚠ PDFs differ ({abs(ratio-1.0)*100:.1f}%)")

    # Test 2: Different backend, same PDF
    if 'prob3_file_pdf' in results and 'nufast_file_pdf' in results:
        v_prob3 = results['prob3_file_pdf'].values.sum()
        v_nufast = results['nufast_file_pdf'].values.sum()
        ratio = v_prob3 / v_nufast if v_nufast != 0 else float('inf')

        print(f"\nPROB3 vs NUFAST backend (using FILE PDF):")
        print(f"  Ratio (prob3/nufast): {ratio:.6f}")
        if abs(ratio - 1.0) < 0.05:
            print(f"  ✓ Backends equivalent (within 5%)")
        else:
            print(f"  ✗ Backends diverge ({abs(ratio-1.0)*100:.1f}%)")

    # Test 3: Different backend, different PDF
    if 'prob3_nufast_pdf' in results and 'nufast_nufast_pdf' in results:
        v_prob3 = results['prob3_nufast_pdf'].values.sum()
        v_nufast = results['nufast_nufast_pdf'].values.sum()
        ratio = v_prob3 / v_nufast if v_nufast != 0 else float('inf')

        print(f"\nPROB3 vs NUFAST backend (using NUFAST PDF):")
        print(f"  Ratio (prob3/nufast): {ratio:.6f}")
        if abs(ratio - 1.0) < 0.05:
            print(f"  ✓ Backends equivalent (within 5%)")
        else:
            print(f"  ✗ Backends diverge ({abs(ratio-1.0)*100:.1f}%)")

else:
    print("✗ Insufficient backends succeeded for comparison")

print("\n" + "=" * 80)
