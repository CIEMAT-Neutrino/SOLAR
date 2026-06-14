#!/usr/bin/env python3
"""Compare oscillation weighting between working (old) and current code."""

import numpy as np
import tempfile
import subprocess
import shutil
import sys
from pathlib import Path

print("=" * 80)
print("OSCILLATION WEIGHTING REGRESSION TEST")
print("=" * 80)

# Find last known working commit before refactor
print("\nFinding last working commit (before refactor)...")
result = subprocess.run(
    ["git", "log", "--oneline", "-n", "20"],
    capture_output=True,
    text=True,
    cwd="/pc/choozdsk01/users/manthey/SOLAR"
)

print("Recent commits:")
for line in result.stdout.split('\n')[:10]:
    print(f"  {line}")

# The refactor commit is d41cee2, so anything before bb5d270 should be working
# Let's use bb5d270 as the "old working" version
working_commit = "bb5d270"  # UPDATE CODEBASE 18/05/26 - last commit before major refactor
current_commit = "HEAD"

print(f"\nComparing:")
print(f"  Old (working):  {working_commit}")
print(f"  Current:        {current_commit}")

# Create temp directory for old code
with tempfile.TemporaryDirectory() as tmpdir:
    old_code_path = Path(tmpdir) / "old_code"
    old_code_path.mkdir()

    print(f"\nCloning old code to {old_code_path}...")

    # Copy current repo to temp
    subprocess.run(
        ["cp", "-r", "/pc/choozdsk01/users/manthey/SOLAR/lib", str(old_code_path)],
        check=True
    )

    # Checkout old version
    result = subprocess.run(
        ["git", "show", f"{working_commit}:lib/lib_osc.py"],
        capture_output=True,
        text=True,
        cwd="/pc/choozdsk01/users/manthey/SOLAR"
    )

    if result.returncode == 0:
        # Save old oscillation code
        old_osc_path = old_code_path / "lib_osc_old.py"
        old_osc_path.write_text(result.stdout)
        print(f"✓ Saved old lib_osc.py")
    else:
        print(f"✗ Could not checkout old lib_osc.py: {result.stderr}")
        sys.exit(1)

    # Setup test parameters
    print("\nSetting up test oscillation parameters...")

    dm2 = 6e-5
    sin13 = 0.021
    sin12 = 0.303

    nadir_edges = np.linspace(-1.0, 1.0, 41)
    nadir_centers = 0.5 * (nadir_edges[1:] + nadir_edges[:-1])
    energy_edges = np.linspace(0, 30, 121)

    print(f"  dm2={dm2:.2e}, sin13={sin13:.3f}, sin12={sin12:.3f}")
    print(f"  Energy: {len(energy_edges)-1} bins [0, 30] MeV")
    print(f"  Nadir:  {len(nadir_edges)-1} bins [-1, 1]")

    # Test current code
    print("\n" + "-" * 80)
    print("TESTING CURRENT CODE")
    print("-" * 80)

    try:
        from lib.oscillation_backends import (
            compute_nufast, get_nadir_pdf_file, combine_day_night
        )

        osc_current = compute_nufast(
            dm2=dm2,
            sin13=sin13,
            sin12=sin12,
            energy_edges=energy_edges,
            nadir_edges=nadir_edges,
            latitude_deg=44.35,
        )

        nadir_pdf_current = get_nadir_pdf_file(nadir_centers=nadir_centers)
        combined_current = combine_day_night(osc_current, nadir_pdf_current)

        current_sum = combined_current.values.sum()
        current_mean = np.nanmean(combined_current.values)

        print(f"Current oscillogram sum: {current_sum:.6e}")
        print(f"Current oscillogram mean: {current_mean:.6e}")
        print(f"Current oscillogram shape: {combined_current.shape}")

    except Exception as e:
        print(f"✗ Error testing current code: {e}")
        import traceback
        traceback.print_exc()
        current_sum = None

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

if current_sum:
    print(f"\nCurrent code produces oscillogram sum: {current_sum:.6e}")
    print("\nTo compare with old code, we would need:")
    print("  1. Old process_oscillation_map() implementation (pre-refactor)")
    print("  2. Run it with same parameters")
    print("  3. Compare normalization")
    print("\nNote: Old code may no longer run due to lib_* name changes")
    print("      This test framework is ready but needs manual setup of old code")
else:
    print("Could not run current code test")

print("\n" + "=" * 80)
print("MANUAL COMPARISON:")
print("=" * 80)
print("""
To manually compare with git history:

1. Find old process_oscillation_map() implementation:
   git show bb5d270:lib/lib_osc.py | grep -A 100 "def process_oscillation_map"

2. Compare with current:
   grep -A 100 "def process_oscillation_map" lib/oscillation.py

3. Key things to check:
   - How nadir PDF is loaded (get_nadir_angle vs get_nadir_pdf_file)
   - How nadir weighting is applied (df.mul(nadir_y) vs combine_day_night)
   - Energy/nadir grid resolution
   - Any scaling factors in convolution

4. Run git diff to see changes:
   git diff bb5d270 HEAD -- lib/lib_osc.py | head -200
""")

print("=" * 80)
