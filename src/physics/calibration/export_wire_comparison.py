import os
import sys
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import (
    root,
    load_multi,
    compute_reco_workflow,
    compute_filtered_run,
    save_df,
    get_project_root,
)

root = get_project_root()

data_path = f"{root}/output/data/workflow/wire_comparison"
if not os.path.exists(data_path):
    os.makedirs(data_path)

parser = argparse.ArgumentParser(
    description="Export wire plane comparison pkl for script_compare_hist2d.py"
)
parser.add_argument("--config", type=str, default="hd_1x2x6")
parser.add_argument("--name", type=str, default="marley_signal")
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()
config = args.config
name = args.name
configs = {config: [name]}

run, output = load_multi(configs, preset="CORRECTION", debug=args.debug)
run = compute_reco_workflow(run, configs, workflow="CORRECTION", debug=args.debug)
filtered_run, mask, output = compute_filtered_run(
    run, configs, presets=["CORRECTION"], debug=args.debug
)

data = filtered_run["Reco"]

# Each row holds flat arrays for one variable pair so script_compare_hist2d.py
# can filter by Variable and extract columns via --x / --y CLI args.
# Usage examples:
#   --variables Charge --x Ind0Charge --y Ind1Charge --diagonal
#   --variables NHits  --x Ind0NHits  --y Ind1NHits  --diagonal
#   --variables Charge --x Charge     --y Ind0Charge  (collection vs Ind0)
#   --variables Charge --x Charge     --y Ind1Charge  (collection vs Ind1)
def clean(arr, sentinel=-1e6):
    a = arr.flatten().astype(float)
    a[a == sentinel] = np.nan
    return a

nan_like = lambda arr: np.full(arr.flatten().shape, np.nan)

rows = [
    {
        "Config": config,
        "Name": name,
        "Variable": "Charge",
        "Ind0": clean(data["Ind0Charge"]),
        "Ind0Unit": "ADC x tick",
        "Ind1": clean(data["Ind1Charge"]),
        "Ind1Unit": "ADC x tick",
        "Col": clean(data["Charge"]),
        "ColUnit": "ADC x tick",
    },
    {
        "Config": config,
        "Name": name,
        "Variable": "NHits",
        "Ind0": clean(data["Ind0NHits"], sentinel=0),
        "Ind0Unit": "#Hits",
        "Ind1": clean(data["Ind1NHits"], sentinel=0),
        "Ind1Unit": "#Hits",
        "Col": clean(data["NHits"], sentinel=0),
        "ColUnit": "#Hits",
    },
]

if "Purity" in data:
    _purity = clean(data["Purity"])
    rows.append({
        "Config": config,
        "Name": name,
        "Variable": "Purity",
        "Ind0": nan_like(_purity),
        "Ind0Unit": "",
        "Ind1": nan_like(_purity),
        "Ind1Unit": "",
        "Col": _purity,
        "ColUnit": "",
    })

df = pd.DataFrame(rows)

save_df(
    df,
    data_path,
    config,
    name,
    filename="Wire_Plane_Comparison",
    rm=args.rewrite,
    debug=args.debug,
)
