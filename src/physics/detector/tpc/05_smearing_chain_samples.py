"""
05_smearing_chain_samples.py — Per-Step Spectrum Comparison Across Samples
==========================================================================
Reads the per-step spectra written by 04_smearing_chain.py for several samples
(default marley, gamma, neutron, radiological) and draws, for every step of the
primary-cluster energy chain, one figure with the unit-area spectra of all
samples overlaid (linear and log panels). No analysis selection is involved:
run 04_smearing_chain.py for each sample first (same --sample folder).

It also merges the per-sample cutflow_plot.py-format pkls into one combined pkl
per config (no Name column, like cutflow_plot.py's combined output):
  output/data/TPC/smearing/{config}/samples/{sample}/{config}_{energy}_Chain_Cutflow.pkl
"""

import os
import sys

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *

save_path = f"{root}/output/images/TPC/smearing"
data_path = f"{root}/output/data/TPC/smearing"

parser = argparse.ArgumentParser(description="Compare the per-step primary-cluster spectra of several samples")
parser.add_argument("--config", type=str, nargs="+", help="Configurations to plot", default=["hd_1x2x6_centralAPA"])
parser.add_argument("--names", type=str, nargs="+", help="Samples to overlay",
                    default=["marley", "gamma", "neutron", "radiological"])
parser.add_argument("--sample", type=str, default="full",
                    help="04_smearing_chain.py output folder to read (full, true9.5-10.5_unweighted, ...)")
parser.add_argument("--energy", type=str, choices=["ClusterEnergy", "SolarEnergy"], default="SolarEnergy")
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()

STEP_ORDER = ["TrueEnergy", "Charge", "PileUp", "Lifetime", "Gain", "Calibration", "Offset", "Map"]
STEP_TITLES = {
    "TrueEnergy": "True energy",
    "Charge": "Containment & charge response",
    "PileUp": "+ Pile-up",
    "Lifetime": "+ Flash-matched lifetime correction",
    "Gain": "+ NHits charge-per-MeV",
    "Calibration": "+ Electron calibration",
    "Offset": "+ Discriminant offset",
    "Map": f"+ {args.energy} linear calibration",
}
# Sample colours follow the analysis plots (03_analysis.py / backgrounds.json STYLE)
SAMPLE_COLORS = {"marley": "rgb(225,124,5)", "gamma": "black", "neutron": "rgb(15,133,84)",
                 "radiological": "rgb(120,94,240)"}

for config in args.config:
    spectra, truths, weighting, cutflow_frames = {}, {}, set(), []
    for name in args.names:
        base = f"{data_path}/{config}/{name}/{args.sample}/{config}_{name}_Chain"
        if not os.path.exists(f"{base}_Spectra.pkl"):
            rprint(f"[yellow][WARNING][/yellow] {config} {name}: no 04_smearing_chain.py output in {args.sample}; skipping")
            continue
        df = pd.read_pickle(f"{base}_Spectra.pkl")
        df = df.loc[df["Energy"] == args.energy]
        if df.empty:
            rprint(f"[yellow][WARNING][/yellow] {config} {name}: no {args.energy} spectra; skipping")
            continue
        spectra[name] = {row["Step"]: row for _, row in df.iterrows()}
        cutflow_pkl = f"{data_path}/{config}/{name}/{args.sample}/{config}_{name}_{args.energy}_Chain_Cutflow.pkl"
        if os.path.exists(cutflow_pkl):
            cutflow_frames.append(pd.read_pickle(cutflow_pkl))
        summary = json.load(open(f"{base}_Summary.json"))
        truths[name] = summary["truth_variable"]
        weighting.add(summary["weighting"])
    if len(spectra) < 2:
        rprint(f"[yellow][WARNING][/yellow] {config}: fewer than two samples available; nothing to compare")
        continue

    # marley carries the oscillated day+night mean, backgrounds the truth-flux weights: the analysis weights
    if cutflow_frames:
        save_df(pd.concat(cutflow_frames, ignore_index=True).drop(columns=["Name"], errors="ignore"),
                data_path, config, None, subfolder=f"samples/{args.sample}", filename=f"{args.energy}_Chain_Cutflow",
                rm=args.rewrite, debug=args.debug)
    weight_tag = "unweighted" if weighting == {"unweighted"} else "analysis weights" if "unweighted" not in weighting else "mixed weighting"
    steps = [step for step in STEP_ORDER if any(step in rows for rows in spectra.values())]
    for idx, step in enumerate(steps):
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Linear", "Log"), horizontal_spacing=0.1)
        for col in [1, 2]:
            for name, rows in spectra.items():
                if step not in rows:
                    continue
                density = np.asarray(rows[step]["Density"], dtype=float)
                label = f"{name} ({truths[name]})" if step == "TrueEnergy" else name
                fig.add_trace(go.Scatter(x=rows[step]["Bins"], y=np.where(density > 0, density, np.nan), mode="lines",
                                         line_shape="hvh", name=label, legendgroup=name, showlegend=col == 1,
                                         line=dict(color=SAMPLE_COLORS.get(name, "grey"), width=2)),
                              row=1, col=col)
        format_coustom_plotly(fig, title=f"{STEP_TITLES[step]} ({args.energy}, {args.sample}, {weight_tag}) - {config}",
                              matches=(None, None), legend=dict(x=1.02, y=1.0), figsize=(1400, 550),
                              tickformat=(".0f", None), debug=args.debug)
        fig.update_xaxes(title="Energy after this step (MeV)", range=[0, 30])
        fig.update_yaxes(title="Fraction per MeV (unit area)", col=1)
        fig.update_yaxes(dtick=1, exponentformat="power", col=2)
        fit_y_ranges(fig, log_axes=("yaxis2",))
        save_figure(fig, save_path, config, None, subfolder=f"samples/{args.sample}",
                    filename=f"{args.energy}_{idx:02d}_{step}_Samples", rm=args.rewrite, debug=args.debug)
    rprint(f"[green]{config}: {len(steps)} per-step sample comparisons saved to {save_path}/{config}/samples/{args.sample}[/green]")
