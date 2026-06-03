#!/usr/bin/env python3
"""
generate_event_display.py

Combines VisEventEDep, VisEventOpFlash, VisEventOpHits and VisEventTPC notebooks
into a single CLI script. Produces:
  - EDep event display (Plotly)
  - AdjFlash PDS event display (Plotly)
  - OpHit PDS event display (Plotly)
  - TPC cluster event display (Plotly)
  - output DataFrame (pickle) compatible with LOWE_RECONSTRUCTION_PUBLICATION

Usage:
  python generate_event_display.py [options]

DataFrame schema (output_event_display.pkl):
  Config, Name, Variable, X, Y, Z, E, PDG, Event
  Variable values: "EDep", "MainCluster", "AdjCluster"
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib import (
    load_multi,
    compute_reco_workflow,
    compute_filtered_run,
    plot_edep_event,
    plot_adjflash_event,
    plot_pds_event,
    plot_tpc_event,
)
from plotly.subplots import make_subplots

DEFAULTS = {
    "edep":     {"config": "hd_1x2x6_centralAPA",              "name": "marley_edep"},
    "adjflash": {"config": "vd_1x8x14_3view_30deg_optimistic",  "name": "marley_yzprojected"},
    "ophit":    {"config": "vd_1x8x14_3view_30deg",             "name": "marley_ophit"},
    "tpc":      {"config": "vd_1x8x14_3view_30deg_optimistic",  "name": "marley"},
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Combined event display: EDep, AdjFlash, OpHit, TPC"
    )
    for wf, d in DEFAULTS.items():
        flag_config = "--config" if wf == "tpc" else f"--{wf}-config"
        flag_name   = "--name"   if wf == "tpc" else f"--{wf}-name"
        dest_config = "tpc_config" if wf == "tpc" else None
        dest_name   = "tpc_name"   if wf == "tpc" else None
        p.add_argument(flag_config, default=d["config"],
                       **({"dest": dest_config} if dest_config else {}),
                       help=f"{wf} detector config (default: {d['config']})")
        p.add_argument(flag_name, default=d["name"],
                       **({"dest": dest_name} if dest_name else {}),
                       help=f"{wf} dataset name (default: {d['name']})")

    p.add_argument("--plots", nargs="+",
                   choices=["edep", "adjflash", "ophit", "tpc"],
                   default=["edep", "adjflash", "ophit", "tpc"],
                   help="Which displays to generate (default: all)")
    p.add_argument("--event", type=int, default=None,
                   help="Event index to display (default: random)")
    p.add_argument("--save", action="store_true",
                   help="Save 2D projection PNG and DataFrame to disk")
    p.add_argument("--html", action="store_true",
                   help="Also save full interactive HTML (includes 3D view, large files)")
    p.add_argument("--output", type=Path,
                   default=ROOT / "output" / "event_display",
                   help="Output directory for DataFrame and figures")
    p.add_argument("--no-show", action="store_true",
                   help="Do not call fig.show() (useful in batch mode)")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# DataFrame builders
# ---------------------------------------------------------------------------

def _build_edep_df(run, config, name, idx, tree="Truth"):
    x   = [v for v in run[tree]["TSignalXDepList"][idx]   if v != 0]
    y   = [v for v in run[tree]["TSignalYDepList"][idx]   if v != 0]
    z   = [v for v in run[tree]["TSignalZDepList"][idx]   if v != 0]
    e   = [v for v in run[tree]["TSignalEDepList"][idx]   if v != 0]
    pdg = [v for v in run[tree]["TSignalPDGDepList"][idx] if v != 0]
    n = min(len(x), len(y), len(z), len(e), len(pdg))

    event_num = int(run[tree]["Event"][idx])
    nu_energy = float(run[tree]["SignalParticleE"][idx])
    title = f"Neutrino CC Interaction on LAr. $E_\\nu$ = {nu_energy:.2f} MeV"

    # Match Reco cluster by event number
    reco_idx = np.where(np.asarray(run["Reco"]["Event"]) == event_num)[0]
    if len(reco_idx) > 0:
        r = reco_idx[0]
        cluster_x   = float(run["Reco"]["RecoX"][r])
        cluster_y   = float(run["Reco"]["RecoY"][r])
        cluster_z   = float(run["Reco"]["RecoZ"][r])
        cluster_q   = float(run["Reco"]["Charge"][r])
        cluster_pur = float(run["Reco"]["MatchedOpFlashPur"][r])
        cluster_pdg = str(int(run["Reco"]["MainPDG"][r]))
    else:
        cluster_x = cluster_y = cluster_z = cluster_q = cluster_pur = float("nan")
        cluster_pdg = ""

    return pd.DataFrame({
        "Config":          config,
        "Name":            name,
        "Variable":        "EDep",
        "Title":           title,
        "X":               x[:n],
        "Y":               y[:n],
        "Z":               z[:n],
        "E":               e[:n],
        "PDG":             [str(p) for p in pdg[:n]],
        "Event":           event_num,
        "ClusterX":        cluster_x,
        "ClusterY":        cluster_y,
        "ClusterZ":        cluster_z,
        "ClusterCharge":   cluster_q,
        "ClusterPurity":   cluster_pur,
        "ClusterPDG":      cluster_pdg,
    })


def _build_tpc_df(run, config, name, idx, tree="Reco"):
    nu_energy = float(run[tree]["SignalParticleE"][idx])
    title = f"Neutrino CC Interaction on LAr. $E_\\nu$ = {nu_energy:.2f} MeV"
    event_num = int(run[tree]["Event"][idx])
    rows = []
    rows.append({
        "Config":   config,
        "Name":     name,
        "Variable": "MainCluster",
        "Title":    title,
        "X": float(run[tree]["RecoX"][idx]),
        "Y": float(run[tree]["RecoY"][idx]),
        "Z": float(run[tree]["RecoZ"][idx]),
        "E": float(run[tree]["SignalParticleK"][idx]),
        "PDG": str(run[tree]["MainPDG"][idx]),
        "Event": event_num,
    })
    adj_x   = [v for v in run[tree]["AdjClRecoX"][idx]    if v != 0 and v > -1e6]
    adj_y   = [v for v in run[tree]["AdjClRecoY"][idx]    if v != 0 and v > -1e6]
    adj_z   = [v for v in run[tree]["AdjClRecoZ"][idx]    if v != 0 and v > -1e6]
    adj_pdg = [v for v in run[tree]["AdjClMainPDG"][idx]  if v != 0]
    n = min(len(adj_x), len(adj_y), len(adj_z))
    for i in range(n):
        rows.append({
            "Config":   config,
            "Name":     name,
            "Variable": "AdjCluster",
            "Title":    title,
            "X": adj_x[i],
            "Y": adj_y[i],
            "Z": adj_z[i],
            "E": float("nan"),
            "PDG": str(adj_pdg[i]) if i < len(adj_pdg) else "0",
            "Event": event_num,
        })
    return pd.DataFrame(rows)


def _extract_2d_fig(fig):
    """
    Build a 2-panel figure from the ZY (col=1) and XY (col=2) scatter traces,
    dropping all scatter3d/surface traces that live in col=3.
    """
    fig2d = make_subplots(rows=1, cols=2)
    for trace in fig.data:
        if trace.type in ("scatter3d", "surface"):
            continue
        ax = getattr(trace, "xaxis", None) or "x"
        col = 2 if ax == "x2" else 1
        fig2d.add_trace(trace, row=1, col=col)

    try:
        fig2d.update_xaxes(title_text=fig.layout.xaxis.title.text,  row=1, col=1)
        fig2d.update_yaxes(title_text=fig.layout.yaxis.title.text,  row=1, col=1)
        fig2d.update_xaxes(title_text=fig.layout.xaxis2.title.text, row=1, col=2)
        fig2d.update_yaxes(title_text=fig.layout.yaxis2.title.text, row=1, col=2)
    except Exception:
        pass
    fig2d.update_layout(title_text=fig.layout.title.text, height=600)
    return fig2d


def _save_fig(fig, args, filename):
    """
    Default: extract 2D projections → save as PNG.
    --html: also save full interactive figure (3D included) as HTML.
    """
    if args.save:
        fig2d = _extract_2d_fig(fig)
        out_png = args.output / f"{filename}.png"
        fig2d.write_image(str(out_png))
        print(f"Saved → {out_png}")

    if args.html:
        out_html = args.output / f"{filename}.html"
        fig.write_html(str(out_html))
        print(f"Saved → {out_html}")


# ---------------------------------------------------------------------------
# Per-workflow runners
# ---------------------------------------------------------------------------

def run_edep(args, event_idx):
    config = getattr(args, "edep_config")
    name   = getattr(args, "edep_name")
    configs = {config: [name]}

    run, output = load_multi(configs, preset="EDEP", debug=args.debug)
    run = compute_reco_workflow(run, configs, {}, workflow="EDEP", debug=args.debug)

    info = json.load(open(ROOT / "config" / config / f"{config}_config.json"))
    _, _, _ = compute_filtered_run(
        run, configs,
        presets=["EDEP"],
        params={
            ("Truth", "Geometry"): ("equal", info["GEOMETRY"]),
            ("Truth", "Version"):  ("equal", info["VERSION"]),
        },
        debug=args.debug,
    )

    fig, idx = plot_edep_event(run, configs, idx=event_idx, tracked="Truth", zoom=False)
    fig.update_layout(width=1200, height=600)

    if not args.no_show:
        fig.show()
    if args.save:
        _save_fig(fig, args, f"EDep_event_{idx}")

    df = _build_edep_df(run, config, name, idx, tree="Truth")
    return fig, idx, df


def run_adjflash(args, event_idx):
    config = getattr(args, "adjflash_config")
    name   = getattr(args, "adjflash_name")
    configs = {config: [name]}

    run, output = load_multi(configs, preset="ADJFLASH", debug=args.debug)
    run = compute_reco_workflow(run, configs, workflow="ADJFLASH", debug=args.debug)

    info = json.load(open(ROOT / "config" / config / f"{config}_config.json"))
    _, _, _ = compute_filtered_run(
        run, configs,
        presets=["ADJFLASH"],
        params={
            ("Reco", "Geometry"):         ("equal", info["GEOMETRY"]),
            ("Reco", "Version"):          ("equal", info["VERSION"]),
            ("Reco", "MatchedOpFlashPur"): ("bigger", 0),
        },
        debug=args.debug,
    )

    fig, idx = plot_adjflash_event(
        run, configs,
        idx=event_idx,
        tree="Reco",
        tracked="AdjOpFlash",
        adjopflashsignal=None,
        adjopflashsize=100,
        unzoom=1.5,
        debug=args.debug,
    )
    fig.update_layout(width=1200, height=600)

    if not args.no_show:
        fig.show()
    if args.save:
        _save_fig(fig, args, f"PDS_event_{idx}")

    return fig, idx


def run_ophit(args, event_idx):
    config = getattr(args, "ophit_config")
    name   = getattr(args, "ophit_name")
    configs = {config: [name]}

    run, output = load_multi(configs, preset="OPHIT", debug=args.debug)

    info = json.load(open(ROOT / "config" / config / f"{config}_config.json"))
    _, _, _ = compute_filtered_run(
        run, configs,
        presets=["OPHIT"],
        params={
            ("Truth", "Geometry"): ("equal", info["GEOMETRY"]),
            ("Truth", "Version"):  ("equal", info["VERSION"]),
        },
        debug=args.debug,
    )

    fig, idx = plot_pds_event(
        run, configs,
        idx=event_idx,
        tracked="Truth",
        maxophit=50,
        flashid=None,
        debug=args.debug,
    )
    fig.update_layout(width=1200, height=600)

    if not args.no_show:
        fig.show()
    if args.save:
        _save_fig(fig, args, f"OpHit_event_{idx}")

    return fig, idx


def run_tpc(args, event_idx):
    config = getattr(args, "tpc_config")
    name   = getattr(args, "tpc_name")
    configs = {config: [name]}

    run, output = load_multi(configs, preset="VERTEXING", debug=args.debug)
    run = compute_reco_workflow(run, configs, params={}, workflow="VERTEXING", debug=False)

    info = json.load(open(ROOT / "config" / config / f"{config}_config.json"))
    this_run, _, _ = compute_filtered_run(
        run, configs,
        presets=["VERTEXING"],
        params={
            ("Reco", "Geometry"):         ("equal", info["GEOMETRY"]),
            ("Reco", "Version"):          ("equal", info["VERSION"]),
            ("Reco", "MatchedOpFlashPur"): ("bigger", 0),
            ("Reco", "SignalParticleK"):   ("smaller", 20),
        },
        debug=args.debug,
    )

    fig, idx = plot_tpc_event(
        this_run, configs,
        idx=event_idx,
        tracked="Reco",
        adjclnum=1,
        get_adj_color=True,
        unzoom=1.25,
        debug=args.debug,
    )
    fig.update_layout(width=1200, height=600)

    if not args.no_show:
        fig.show()
    if args.save:
        _save_fig(fig, args, f"TPC_event_{idx}")

    df = _build_tpc_df(this_run, config, name, idx, tree="Reco")
    return fig, idx, df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.save:

        args.output.mkdir(parents=True, exist_ok=True)

    plots = set(args.plots)
    dfs = {}

    if "edep" in plots:
        print("[EDep] Loading and plotting...")
        _, edep_idx, edep_df = run_edep(args, args.event)
        dfs["edep"] = (args.edep_config, args.edep_name, edep_idx, edep_df)
        print(f"[EDep] Done — event {edep_idx}, {len(edep_df)} deposition points")

    if "adjflash" in plots:
        print("[AdjFlash] Loading and plotting...")
        _, adjflash_idx = run_adjflash(args, args.event)
        print(f"[AdjFlash] Done — event {adjflash_idx}")

    if "ophit" in plots:
        print("[OpHit] Loading and plotting...")
        _, ophit_idx = run_ophit(args, args.event)
        print(f"[OpHit] Done — event {ophit_idx}")

    if "tpc" in plots:
        print("[TPC] Loading and plotting...")
        _, tpc_idx, tpc_df = run_tpc(args, args.event)
        dfs["tpc"] = (args.tpc_config, args.tpc_name, tpc_idx, tpc_df)
        print(f"[TPC] Done — event {tpc_idx}, {len(tpc_df)} cluster points")

    if args.save:
        for plot, (cfg, nm, evt, df) in dfs.items():
            pkl_path = args.output / f"{cfg}_{nm}_event_{evt}_display.pkl"
            df.to_pickle(pkl_path)
            print(f"Saved DataFrame → {pkl_path}")

    return {plot: df for plot, (_, _, _, df) in dfs.items()}


if __name__ == "__main__":
    main()
