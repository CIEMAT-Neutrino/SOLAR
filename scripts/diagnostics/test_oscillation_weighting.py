#!/usr/bin/env python3
"""Test oscillation probability weighting to detect normalization issues."""

import numpy as np
from lib.oscillation_backends import (
    compute_nufast, compute_prob3,
    get_nadir_pdf_file, get_nadir_pdf_nufast,
    combine_day_night,
)

print("=" * 70)
print("OSCILLATION WEIGHTING TEST")
print("=" * 70)

# Setup grids
nadir_edges = np.linspace(-1.0, 1.0, 41)
nadir_centers = 0.5 * (nadir_edges[1:] + nadir_edges[:-1])

energy_edges = np.linspace(0, 30, 121)
energy_centers = 0.5 * (energy_edges[1:] + energy_edges[:-1])

print(f"\nEnergy grid: {len(energy_centers)} bins [0, 30] MeV")
print(f"Nadir grid:  {len(nadir_centers)} bins [-1, 1]")

# Compute oscillation with nufast backend
print("\n" + "-" * 70)
print("NUFAST BACKEND")
print("-" * 70)

try:
    osc_nufast = compute_nufast(
        dm2=6e-5,
        sin13=0.021,
        sin12=0.303,
        energy_edges_mev=energy_edges,
        nadir_edges=nadir_edges,
        latitude_deg=44.35,
    )

    # Get nadir PDF
    nadir_pdf = get_nadir_pdf_nufast(nadir_centers, latitude_deg=44.35)
    print(f"Nadir PDF sum: {nadir_pdf.sum():.6f} (should be 1.0)")

    # Combine
    combined = combine_day_night(osc_nufast, nadir_pdf)
    print(f"Combined shape: {combined.shape}")
    print(f"Combined sum: {combined.values.sum():.6e}")

    # Expected: oscillation integral should be ~number of nadir bins × number of energy bins
    # Since P_ee is in [0,1] and nadir_pdf is normalized
    expected_order = len(nadir_centers) * len(energy_centers) * 0.5  # ~0.5 average P_ee
    actual = combined.values.sum()
    ratio = actual / expected_order

    print(f"\nExpected order of magnitude: {expected_order:.2e}")
    print(f"Actual sum: {actual:.2e}")
    print(f"Ratio: {ratio:.2f}x")

    if abs(ratio - 1.0) > 10:
        print(f"⚠ WARNING: Oscillation weighting off by {abs(ratio-1)*100:.0f}%")

except Exception as e:
    print(f"✗ Error with nufast: {e}")

# Try with prob3 if available
print("\n" + "-" * 70)
print("PROB3 BACKEND")
print("-" * 70)

try:
    osc_prob3 = compute_prob3(
        dm2=6e-5,
        sin13=0.021,
        sin12=0.303,
        energy_edges_mev=energy_edges,
        nadir_edges=nadir_edges,
    )

    # Get file nadir PDF
    try:
        nadir_pdf = get_nadir_pdf_file(nadir_centers=nadir_centers)
        print(f"File nadir PDF sum: {nadir_pdf.sum():.6f}")
    except:
        nadir_pdf = get_nadir_pdf_nufast(nadir_centers)
        print(f"File nadir PDF: failed, using nufast fallback")

    # Combine
    combined = combine_day_night(osc_prob3, nadir_pdf)
    print(f"Combined shape: {combined.shape}")
    print(f"Combined sum: {combined.values.sum():.6e}")

    expected_order = len(nadir_centers) * len(energy_centers) * 0.5
    actual = combined.values.sum()
    ratio = actual / expected_order

    print(f"\nExpected order: {expected_order:.2e}")
    print(f"Actual sum: {actual:.2e}")
    print(f"Ratio: {ratio:.2f}x")

except Exception as e:
    print(f"⚠ Prob3 not available: {e}")

print("\n" + "=" * 70)
print("If ratios are ~100x off, oscillation weighting is the issue")
print("=" * 70)
