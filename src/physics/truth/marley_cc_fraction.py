"""
marley_cc_fraction.py — MARLEY CC Energy Channel Fractions
===========================================================
Loads MARLEY truth data, computes per-PDG fractional energy deposition as a
function of neutrino energy, and saves the result as a DataFrame pkl for use
by line_plot.py (kinematic_threshold plot type).

Output
------
  output/data/marley/stacked/{config}/{name}/
    {config}_{name}_Neutrino_CC_Fraction.pkl
      Columns: SignalParticleK (float, MeV), TSignalSumPDG (str),
               TSignalSumK (float, fraction of neutrino energy),
               TMarleyParticle (str), TMarleyColor (str),
               PDG (str), Config (str), Name (str)

Run
---
  python3 src/physics/truth/marley_cc_fraction.py \\
      --config hd_1x2x6 --name marley_official [--rewrite]
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *

save_path = f"{root}/output/images/marley/stacked"
data_path = f"{root}/output/data/marley/stacked"
for _p in [save_path, data_path]:
    os.makedirs(_p, exist_ok=True)

parser = argparse.ArgumentParser(
    description="Compute MARLEY CC energy fraction per PDG channel and save as pkl"
)
parser.add_argument(
    "--config",
    nargs="+",
    type=str,
    help="Detector configuration(s) to process",
    default=["hd_1x2x6"],
)
parser.add_argument(
    "--name",
    nargs="+",
    type=str,
    help="Sample name(s) to process",
    default=["marley_official"],
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
args = parser.parse_args()

configs = {c: args.name for c in args.config}
run, output = load_multi(
    configs,
    load_all=False,
    preset="MARLEY",
    generator_swap=False,
    debug=args.debug,
)
run = compute_reco_workflow(run, configs, workflow="MARLEY", rm_branches=False, debug=args.debug)
truth_df = npy2df(run, "Truth", debug=args.debug)

for config, name in product(args.config, args.name):
    _cfg_info = json.loads(open(f"{root}/config/{config}/{config}_config.json").read())
    _cfg_mask = (
        (truth_df["Name"] == name)
        & (truth_df["Geometry"] == _cfg_info["GEOMETRY"])
        & (truth_df["Version"] == _cfg_info["VERSION"])
    )
    this_df = explode(truth_df[_cfg_mask].copy(), ["TSignalSumK", "TSignalSumPDG"])
    this_df["TSignalSumK"] = this_df["TSignalSumK"].astype(float)

    particle_df = (
        this_df.groupby(
            [pd.cut(this_df["SignalParticleK"], true_energy_edges), "TSignalSumPDG"],
            observed=True,
        )["TSignalSumK"]
        .mean()
        .reset_index()
    )
    particle_df["TSignalSumPDG"] = particle_df["TSignalSumPDG"].astype(str)
    all_pdgs = [str(x) for x in this_df["TSignalSumPDG"].unique()]
    particle_df["TMarleyColor"] = particle_df["TSignalSumPDG"].map(
        get_pdg_color(all_pdgs)
    )

    plot_df = particle_df[particle_df["TMarleyColor"] != "grey"].copy()
    plot_df["SignalParticleK"] = plot_df["SignalParticleK"].apply(lambda x: x.mid)
    plot_df["TMarleyParticle"] = plot_df["TSignalSumPDG"].map(
        get_pdg_name([str(x) for x in truth_df["TSignalSumPDG"][0]])
    )
    plot_df["TMarleyColor"] = plot_df["TSignalSumPDG"].map(
        get_pdg_color(plot_df["TSignalSumPDG"].unique().tolist())
    )
    plot_df = plot_df[plot_df["TMarleyColor"] != "grey"].copy()
    plot_df["TSignalSumK"] = (
        plot_df["TSignalSumK"].astype(float)
        / plot_df["SignalParticleK"].astype(float)
    )
    plot_df["PDG"] = plot_df["TSignalSumPDG"]
    plot_df["Config"] = config
    plot_df["Name"] = name

    save_df(
        plot_df,
        data_path,
        config,
        name,
        filename="Neutrino_CC_Fraction",
        rm=args.rewrite,
        debug=args.debug,
    )
