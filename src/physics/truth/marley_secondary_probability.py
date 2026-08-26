"""
marley_secondary_probability.py — MARLEY CC Secondary Particle Emission Probabilities
======================================================================================
Loads MARLEY truth data, computes per-PDG probability of secondary particle emission
as a function of neutrino energy, and saves the result as a DataFrame pkl.

A secondary particle of type PDG is considered "emitted" if TSignalSumK > 0 for that
PDG in a given event. Probability = fraction of events (per energy bin) where that
condition holds.

Output
------
  output/data/marley/stacked/{config}/{name}/
    {config}_{name}_Neutrino_Secondary_Probability.pkl
      Columns: SignalParticleK (float, MeV), TSignalSumPDG (str),
               Probability (float, 0–1), TMarleyParticle (str),
               TMarleyColor (str), PDG (str), Config (str), Name (str)

Run
---
  python3 src/physics/truth/marley_secondary_probability.py \\
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
    description="Compute per-bin probability of secondary particle emission and save as pkl"
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
_truth_branches = ["SignalParticleK", "TSignalSumK", "TSignalSumPDG", "Name", "Geometry", "Version"]
truth_df = npy2df(run, "Truth", branches=_truth_branches, debug=args.debug)

for config, name in product(args.config, args.name):
    _cfg_info = json.loads(open(f"{root}/config/{config}/{config}_config.json").read())
    _cfg_mask = (
        (truth_df["Name"] == name)
        & (truth_df["Geometry"] == _cfg_info["GEOMETRY"])
        & (truth_df["Version"] == _cfg_info["VERSION"])
    )
    this_df = explode(truth_df[_cfg_mask].copy(), ["TSignalSumK", "TSignalSumPDG"])
    this_df["TSignalSumK"] = this_df["TSignalSumK"].astype(float)
    this_df["TSignalSumPDG"] = this_df["TSignalSumPDG"].astype(str)

    # Binary indicator: secondary of this PDG type was present in the event
    this_df["_emitted"] = (this_df["TSignalSumK"] > 0).astype(float)

    prob_df = (
        this_df.groupby(
            [pd.cut(this_df["SignalParticleK"], true_energy_edges), "TSignalSumPDG"],
            observed=True,
        )["_emitted"]
        .mean()
        .reset_index()
    )
    prob_df.rename(columns={"_emitted": "Probability"}, inplace=True)
    prob_df["TSignalSumPDG"] = prob_df["TSignalSumPDG"].astype(str)

    all_pdgs = [str(x) for x in this_df["TSignalSumPDG"].unique()]
    prob_df["TMarleyColor"] = prob_df["TSignalSumPDG"].map(get_pdg_color(all_pdgs))

    plot_df = prob_df[prob_df["TMarleyColor"] != "grey"].copy()
    plot_df["SignalParticleK"] = plot_df["SignalParticleK"].apply(lambda x: x.mid)
    plot_df["TMarleyParticle"] = plot_df["TSignalSumPDG"].map(
        get_pdg_name([str(x) for x in truth_df["TSignalSumPDG"][0]])
    )
    plot_df["TMarleyColor"] = plot_df["TSignalSumPDG"].map(
        get_pdg_color(plot_df["TSignalSumPDG"].unique().tolist())
    )
    plot_df = plot_df[plot_df["TMarleyColor"] != "grey"].copy()
    plot_df["PDG"] = plot_df["TSignalSumPDG"]
    plot_df["Config"] = config
    plot_df["Name"] = name

    save_df(
        plot_df,
        data_path,
        config,
        name,
        filename="Neutrino_Secondary_Probability",
        rm=args.rewrite,
        debug=args.debug,
    )
