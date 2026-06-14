#!/usr/bin/env python3
"""Compare signal values between old and new codebase."""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def load_template(path_str):
    """Load signal template safely."""
    p = Path(path_str)
    if not p.exists():
        return None
    try:
        df = pd.read_pickle(p)
        return df
    except Exception as e:
        print(f"Error loading {p}: {e}")
        return None

print("=" * 70)
print("SIGNAL TEMPLATE COMPARISON")
print("=" * 70)

new_path = Path("/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/SENSITIVITY/truncated/solarenergy")
old_path = Path("/path/to/old/SENSITIVITY/truncated/solarenergy")  # User must provide

print(f"\nNew codebase path: {new_path}")
print(f"Old codebase path: {old_path}")

# Try to find templates
new_templates = list(new_path.glob("*marley*solar_sin12_df.pkl")) if new_path.exists() else []
old_templates = list(old_path.glob("*marley*solar_sin12_df.pkl")) if old_path.exists() else []

if not new_templates:
    print("✗ No new templates found")
    sys.exit(1)

if not old_templates:
    print("⚠ No old templates found for comparison")
    print("  Please provide path to old SENSITIVITY data")
    sys.exit(0)

new_template = sorted(new_templates, key=lambda x: x.stat().st_mtime)[-1]
old_template = sorted(old_templates, key=lambda x: x.stat().st_mtime)[-1]

print(f"\nNew: {new_template.name}")
print(f"Old: {old_template.name}")

new_df = load_template(str(new_template))
old_df = load_template(str(old_template))

if new_df is None or old_df is None:
    print("✗ Could not load templates for comparison")
    sys.exit(1)

print("\n" + "-" * 70)
print("COMPARISON")
print("-" * 70)

new_sum = new_df.values.sum()
old_sum = old_df.values.sum()
ratio = new_sum / old_sum if old_sum != 0 else float('inf')

print(f"\nNew total signal: {new_sum:.4e}")
print(f"Old total signal: {old_sum:.4e}")
print(f"Ratio (new/old):  {ratio:.2f}x")

if abs(ratio - 1.0) < 0.05:
    print("✓ MATCH: Values are consistent")
elif ratio > 50:
    print(f"✗ NEW IS {ratio:.0f}x LARGER - BUG in new code!")
elif ratio < 0.02:
    print(f"✗ NEW IS {1/ratio:.0f}x SMALLER - BUG in new code!")
else:
    print(f"⚠ Values differ by {abs(ratio-1)*100:.1f}% - investigate")

# Check shape
print(f"\nShape new: {new_df.shape}")
print(f"Shape old: {old_df.shape}")

if new_df.shape != old_df.shape:
    print("⚠ Shapes differ - might affect comparison")

# Per-bin statistics
print(f"\nNew: mean={np.nanmean(new_df.values):.2e}, max={np.nanmax(new_df.values):.2e}")
print(f"Old: mean={np.nanmean(old_df.values):.2e}, max={np.nanmax(old_df.values):.2e}")

print("\n" + "=" * 70)
