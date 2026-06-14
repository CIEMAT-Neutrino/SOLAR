#!/usr/bin/env python3
"""Test rebin_hist2d logic to verify fix correctness."""

import numpy as np
import pandas as pd
from lib.dataframe import rebin_hist2d

print("=" * 60)
print("TESTING REBIN_HIST2D LOGIC")
print("=" * 60)

# Simulate oscillation input: (nadir=2000, energy=120)
nadir_centers = np.linspace(-0.9975, 0.9975, 2000)
energy_centers = np.linspace(0.5, 29.5, 120)

# Create test 2D array matching signal template shape
# Oscillation probabilities (nadir × energy)
z_test = np.random.rand(2000, 120)
z_test[:] = 0.5  # Constant value for easy verification

print(f"\nInput shape: {z_test.shape} (nadir=2000, energy=120)")
print(f"Input sum: {z_test.sum():.2e}")

# Test rebinning with rebin=3
rebin_factor = 3
print(f"\nRebinning with factor={rebin_factor}...")

try:
    rebin_x, rebin_y, rebin_z, rebin_z_per_x = rebin_hist2d(
        energy_centers,      # x: energy
        nadir_centers,       # y: nadir
        z_test,              # z: (nadir, energy)
        rebin_factor,
    )

    print(f"Output shape: {rebin_z.shape}")
    print(f"Output x bins: {len(rebin_x)}")
    print(f"Output y bins: {len(rebin_y)}")
    print(f"Output sum: {rebin_z.sum():.2e}")

    # Check if sum preserved (should be 2000×120×0.5 = 120,000)
    expected_sum = 2000 * 120 * 0.5
    actual_sum = rebin_z.sum()
    ratio = actual_sum / expected_sum

    print(f"\nConservation check:")
    print(f"  Expected sum: {expected_sum:.2e}")
    print(f"  Actual sum:   {actual_sum:.2e}")
    print(f"  Ratio (actual/expected): {ratio:.6f}")

    if abs(ratio - 1.0) < 0.01:
        print("  ✓ PASS: Sum conserved within 1%")
    else:
        print(f"  ✗ FAIL: Sum not conserved! Off by {abs(ratio-1.0)*100:.1f}%")
        if ratio > 1.0:
            print(f"    → Output is {ratio:.1f}x TOO LARGE")
        else:
            print(f"    → Output is {1/ratio:.1f}x TOO SMALL")

except Exception as e:
    print(f"✗ ERROR in rebin_hist2d: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("If ratio is ~100x off, rebinning bug still exists")
print("=" * 60)
