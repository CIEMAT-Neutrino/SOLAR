#!/bin/bash
# Diagnostic script: verify rebin fix + signal values

echo "=== 1. VERIFY REBIN_HIST2D FIX ==="
echo "Checking if axis fix is applied:"
grep -A3 "new_z = (" /pc/choozdsk01/users/manthey/SOLAR/lib/dataframe.py | head -6
echo ""

echo "=== 2. CHECK SIGNAL TEMPLATE TIMESTAMPS ==="
DATA_PATH="/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR"
if [ -d "$DATA_PATH/SENSITIVITY" ]; then
    echo "Signal template timestamps (Truncated/SolarEnergy):"
    find "$DATA_PATH/SENSITIVITY/truncated/solarenergy" -name "*solar_sin12_df.pkl" 2>/dev/null | head -3 | while read f; do
        echo "  $(basename $f): $(stat -c %y "$f" 2>/dev/null || echo 'N/A')"
    done
else
    echo "SENSITIVITY data path not accessible: $DATA_PATH"
fi
echo ""

echo "=== 3. EXTRACT SIGNAL VALUES FROM TEMPLATES ==="
python3 << 'PYTHON_EOF'
import pandas as pd
import numpy as np
from pathlib import Path

data_path = Path("/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/SENSITIVITY/truncated/solarenergy")

# Find latest signal template
templates = list(data_path.glob("*marley*solar_sin12_df.pkl"))
if not templates:
    print("ERROR: No signal templates found")
    exit(1)

latest = sorted(templates, key=lambda x: x.stat().st_mtime)[-1]
print(f"Latest template: {latest.name}")

try:
    df = pd.read_pickle(latest)
    total_signal = df.values.sum()
    mean_val = np.nanmean(df.values)

    print(f"Total signal events: {total_signal:.2e}")
    print(f"Mean per bin: {mean_val:.2e}")
    print(f"Matrix shape: {df.shape}")
    print(f"Non-NaN cells: {(~np.isnan(df.values)).sum()}")
    print(f"Max value: {np.nanmax(df.values):.2e}")
    print(f"Min value (>0): {np.nanmin(df.values[df.values>0]):.2e if (df.values>0).any() else 'N/A'}")
except Exception as e:
    print(f"ERROR reading template: {e}")

PYTHON_EOF
echo ""

echo "=== 4. CHECK EXPOSURE/DETECTOR MASS SETTINGS ==="
python3 << 'PYTHON_EOF'
import json
from pathlib import Path

# Check config
config_path = Path("/pc/choozdsk01/users/manthey/SOLAR/config/hd_1x2x6_centralAPA/hd_1x2x6_centralAPA_config.json")
if config_path.exists():
    cfg = json.load(open(config_path))
    print(f"Detector X: {cfg.get('FD_SIZE_X', 'N/A')} cm")
    print(f"Detector Y: {cfg.get('FD_SIZE_Y', 'N/A')} cm")
    print(f"Detector Z: {cfg.get('FD_SIZE_Z', 'N/A')} cm")
    print(f"LAR density (if set): {cfg.get('LAR_DENSITY', 'default 1.396 g/cm³')}")
else:
    print("Config not found")

PYTHON_EOF
echo ""

echo "=== 5. CHECK OSCILLATION PDF NORMALIZATION ==="
python3 << 'PYTHON_EOF'
import numpy as np
from lib.oscillation_backends import get_nadir_pdf_file, get_nadir_pdf_nufast

nadir_centers = np.linspace(-0.975, 0.975, 40)

try:
    pdf_file = get_nadir_pdf_file(nadir_centers=nadir_centers)
    print(f"File nadir PDF sum: {pdf_file.sum():.6f} (should be 1.0)")
    print(f"  Min: {pdf_file.min():.6e}, Max: {pdf_file.max():.6e}")
except Exception as e:
    print(f"File nadir PDF error: {e}")

try:
    pdf_nufast = get_nadir_pdf_nufast(nadir_centers, latitude_deg=44.35)
    print(f"NuFast nadir PDF sum: {pdf_nufast.sum():.6f} (should be 1.0)")
    print(f"  Min: {pdf_nufast.min():.6e}, Max: {pdf_nufast.max():.6e}")
except Exception as e:
    print(f"NuFast nadir PDF error: {e}")

PYTHON_EOF
echo ""

echo "=== 6. COMPARE WITH PREVIOUS RUN (if available) ==="
echo "To compare: check if you have backup of old signal templates"
echo "Example: diff -r /path/to/old/SENSITIVITY truncated/solarenergy/"
