#!/usr/bin/env python3
"""
Script to generate PKL file with OpHit statistics in the same style as other data files.
This replicates the logic from OpHitSignal.ipynb but produces a PKL file instead of a plot.

The output PKL file contains histogram data for OpHitNum, OpHitMeanTime, and OpHitMeanR
with the same filtering and normalization as the notebook.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import pickle

# Add lib to path
sys.path.insert(0, "../../../")

from lib.io import load_multi
from lib.reco import compute_reco_workflow
from lib.filters import compute_filtered_run

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))

# Configuration - matching the notebook logic
configs = {"hd_1x2x6_centralAPA": ["marley_flash"]}
workflow = "OPHIT"
debug = True

# Output directory - following the pattern of output/data/workflow
output_dir = f"{root}/output/data/workflow/ophit/hd_1x2x6_centralAPA/marley_flash/"
os.makedirs(output_dir, exist_ok=True)

print("Loading data...")
run, output = load_multi(
    configs, preset=workflow, debug=debug
)

print("Computing reco workflow...")
run = compute_reco_workflow(
    run, configs, workflow=workflow, debug=debug
)

# Get config info
config = list(configs.keys())[0]
name = configs[config][0]
info = json.load(open(f"{root}/config/{config}/{config}_config.json", "r"))

print(f"\nProcessing {config} - {name}")

# Filter the run - same filters as in the notebook
params = {
    ("Truth", "Geometry"): ("equal", info["GEOMETRY"]),
    ("Truth", "Version"): ("equal", info["VERSION"]),
    ("Truth", "Name"): ("equal", name),
    ("Truth", "OpHitMeanTime"): ("smaller", 100),
    ("Truth", "OpHitTotalPE"): ("bigger", 0),
}

filtered_run, mask, output = compute_filtered_run(
    run,
    configs,
    params=params,
    debug=debug,
)

# Get the filtered Truth data
truth_data = filtered_run["Truth"]
n_events = len(truth_data["Event"])
print(f"Number of events after filtering: {n_events}")

# Variables to compute histograms for (same as notebook)
variables = ["OpHitNum", "OpHitMeanTime", "OpHitMeanR"]

# Store histogram data
histogram_rows = []

for variable in variables:
    # Get the data
    data = truth_data[variable]
    
    # Filter out any NaN or Inf values for clean histogram
    data = data[~np.isnan(data) & ~np.isinf(data)]
    
    # Find xmin and xmax with percentile (same as notebook)
    xmin = np.percentile(data, 0)
    xmax = np.percentile(data, 99.9)
    
    # Create bins (same as notebook)
    if variable == "OpHitNum":
        bins = np.arange(xmin, xmax, 1)
    else:
        bins = np.linspace(xmin, xmax, 75)
    
    # Compute histogram
    hist, bins = np.histogram(data, density=False, bins=bins)
    
    # Normalize by number of events (same as notebook)
    hist = hist / n_events
    
    # Compute statistics (same as notebook)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    mean = float(np.mean(data))
    std = float(np.std(data))
    cumsum = np.cumsum(hist) / np.sum(hist)
    cumsum_99 = float(bin_centers[np.argmax(cumsum > 0.99)])
    
    # Add each bin as a row
    for i in range(len(hist)):
        histogram_rows.append({
            "Config": config,
            "Name": name,
            "Geometry": info["GEOMETRY"],
            "Version": info["VERSION"],
            "Variable": variable,
            "BinCenter": float(bin_centers[i]),
            "BinEdgeLow": float(bins[i]),
            "BinEdgeHigh": float(bins[i+1]),
            "Counts": float(hist[i]),
        })
    
    # Add summary statistics as special rows
    histogram_rows.append({
        "Config": config,
        "Name": name,
        "Geometry": info["GEOMETRY"],
        "Version": info["VERSION"],
        "Variable": f"{variable}_Mean",
        "BinCenter": mean,
        "BinEdgeLow": 0.0,
        "BinEdgeHigh": 0.0,
        "Counts": 0.0,
    })
    
    histogram_rows.append({
        "Config": config,
        "Name": name,
        "Geometry": info["GEOMETRY"],
        "Version": info["VERSION"],
        "Variable": f"{variable}_STD",
        "BinCenter": std,
        "BinEdgeLow": 0.0,
        "BinEdgeHigh": 0.0,
        "Counts": 0.0,
    })
    
    histogram_rows.append({
        "Config": config,
        "Name": name,
        "Geometry": info["GEOMETRY"],
        "Version": info["VERSION"],
        "Variable": f"{variable}_99Percentile",
        "BinCenter": cumsum_99,
        "BinEdgeLow": 0.0,
        "BinEdgeHigh": 0.0,
        "Counts": 0.0,
    })

# Create DataFrame
df = pd.DataFrame(histogram_rows)

# Save to PKL
filename = f"{config}_{name}_TotalOpHit_Stats.pkl"
output_path = os.path.join(output_dir, filename)

with open(output_path, "wb") as f:
    pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"\nSaved to {output_path}")
print(f"DataFrame shape: {df.shape}")
print(f"DataFrame columns: {df.columns.tolist()}")
print(f"\nFirst 10 rows:")
print(df.head(10))
print(f"\nVariable types:")
print(df["Variable"].value_counts())

# Also save a summary to a text file for verification
summary_path = os.path.join(output_dir, f"{config}_{name}_TotalOpHit_Stats_summary.txt")
with open(summary_path, "w") as f:
    f.write(f"Config: {config}\n")
    f.write(f"Name: {name}\n")
    f.write(f"Geometry: {info['GEOMETRY']}\n")
    f.write(f"Version: {info['VERSION']}\n")
    f.write(f"Number of events: {n_events}\n")
    f.write(f"\nStatistics:\n")
    for variable in variables:
        data = truth_data[variable]
        data = data[~np.isnan(data) & ~np.isinf(data)]
        f.write(f"\n{variable}:\n")
        f.write(f"  Mean: {np.mean(data):.4f}\n")
        f.write(f"  STD: {np.std(data):.4f}\n")
        f.write(f"  Min: {np.min(data):.4f}\n")
        f.write(f"  Max: {np.max(data):.4f}\n")
        f.write(f"  99th percentile: {np.percentile(data, 99):.4f}\n")

print(f"\nSummary saved to {summary_path}")
print("Done!")
