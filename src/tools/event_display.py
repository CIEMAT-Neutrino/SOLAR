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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lib import (
    load_multi,
    compute_reco_workflow,
    compute_filtered_run,
    plot_edep_event,
    plot_adjflash_event,
    plot_pds_event,
    plot_tpc_event,
    get_edep_positions,
    add_data_to_event,
)
from lib.solar import get_pdg_color
from plotly.subplots import make_subplots

DEFAULTS = {
    "edep":     {"config": "hd_1x2x6_centralAPA",               "name": "marley_edep"},
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
                   choices=["edep", "adjflash", "ophit", "tpc", "edep-tpc"],
                   default=["edep", "adjflash", "ophit", "tpc"],
                   help="Which displays to generate (default: all)")
    p.add_argument("--event", type=int, default=None,
                   help="Event index to display (default: random)")
    p.add_argument("--save", action="store_true",
                   help="Save 2D projection PNG and DataFrame to disk")
    p.add_argument("--html", action="store_true",
                   help="Also save full interactive HTML (includes 3D view, large files)")
    p.add_argument("--output", type=Path,
                   default=ROOT / "output" / "images" / "event",
                   help="Output directory for figures (PNG/HTML)")
    p.add_argument("--data-output", type=Path, dest="data_output",
                   default=ROOT / "output" / "data" / "event",
                   help="Output directory for DataFrames (pkl)")
    p.add_argument("--no-show", action="store_true",
                   help="Do not call fig.show() (useful in batch mode)")
    p.add_argument("--cut-flash", action="store_true",
                   help="TPC: require MatchedOpFlashPur > 0")
    p.add_argument("--cut-energy", type=float, default=None, metavar="MAX_MEV",
                   help="TPC: require SignalParticleK < MAX_MEV")
    p.add_argument("--min-adjcl", type=int, default=None, metavar="N", dest="min_adjcl",
                   help="TPC: require at least N adjacent clusters")
    p.add_argument("--layout", choices=["truth", "reco"], default="truth",
                   help="TPC projection: 'truth'=col2 is (X,Y); 'reco'=col2 is (Z,X) (default: truth)")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# DataFrame builders
# ---------------------------------------------------------------------------

def _build_edep_df(run, config, name, idx, tree="Truth"):
    event_num = int(run[tree]["Event"][idx])
    nu_energy = float(run[tree]["SignalParticleE"][idx])
    title     = f"Neutrino CC Interaction on LAr. $E_\\nu$ = {nu_energy:.2f} MeV"

    # EDep deposits — matches "circle" markers in plot_edep_event
    x   = [v for v in run[tree]["TSignalXDepList"][idx]   if v != 0]
    y   = [v for v in run[tree]["TSignalYDepList"][idx]   if v != 0]
    z   = [v for v in run[tree]["TSignalZDepList"][idx]   if v != 0]
    e   = [v for v in run[tree]["TSignalEDepList"][idx]   if v != 0]
    pdg = [str(v) for v in run[tree]["TSignalPDGDepList"][idx] if v != 0]
    n   = min(len(x), len(y), len(z), len(e), len(pdg))

    rows = [{
        "Config":   config,
        "Name":     name,
        "Variable": "truth",
        "Title":    title,
        "Event":    event_num,
        "X":        x[:n],
        "Y":        y[:n],
        "Z":        z[:n],
        "E":        e[:n],
        "PDG":      pdg[:n],
        "Charge":   [float("nan")] * n,
        "Purity":   [float("nan")] * n,
    }]

    # CC daughters — matches "square-open" markers in plot_edep_event
    cc_x   = [v for v in run[tree]["TSignalX"][idx]   if v != 0]
    cc_y   = [v for v in run[tree]["TSignalY"][idx]   if v != 0]
    cc_z   = [v for v in run[tree]["TSignalZ"][idx]   if v != 0]
    cc_pdg = [str(v) for v in run[tree]["TSignalPDG"][idx] if v not in [0, 1000190419]]
    nc     = min(len(cc_x), len(cc_y), len(cc_z), len(cc_pdg))
    if nc > 0:
        rows.append({
            "Config":   config,
            "Name":     name,
            "Variable": "ccint",
            "Title":    title,
            "Event":    event_num,
            "X":        cc_x[:nc],
            "Y":        cc_y[:nc],
            "Z":        cc_z[:nc],
            "E":        [float("nan")] * nc,
            "PDG":      cc_pdg[:nc],
            "Charge":   [float("nan")] * nc,
            "Purity":   [float("nan")] * nc,
        })

    return pd.DataFrame(rows)


def _build_tpc_df(run, config, name, idx, tree="Reco"):
    event_num = int(run[tree]["Event"][idx])
    nu_energy = float(run[tree]["SignalParticleE"][idx])
    title     = f"Neutrino CC Interaction on LAr. $E_\\nu$ = {nu_energy:.2f} MeV"

    # truth: neutrino vertex + CC daughters
    d_x   = [v for v in run[tree]["TSignalX"][idx]   if v != 0]
    d_y   = [v for v in run[tree]["TSignalY"][idx]   if v != 0]
    d_z   = [v for v in run[tree]["TSignalZ"][idx]   if v != 0]
    d_pdg = [str(v) for v in run[tree]["TSignalPDG"][idx] if v not in [0, 1000190419]]
    nd    = min(len(d_x), len(d_y), len(d_z), len(d_pdg))
    truth_x   = [float(run[tree]["SignalParticleX"][idx])] + d_x[:nd]
    truth_y   = [float(run[tree]["SignalParticleY"][idx])] + d_y[:nd]
    truth_z   = [float(run[tree]["SignalParticleZ"][idx])] + d_z[:nd]
    truth_e   = [float(run[tree]["SignalParticleE"][idx])] + [float("nan")] * nd
    truth_pdg = ["12"] + d_pdg[:nd]

    # reco: main cluster + adj clusters
    adj_x   = [v for v in run[tree]["AdjClRecoX"][idx]   if v != 0 and v > -1e6]
    adj_y   = [v for v in run[tree]["AdjClRecoY"][idx]   if v != 0 and v > -1e6]
    adj_z   = [v for v in run[tree]["AdjClRecoZ"][idx]   if v != 0 and v > -1e6]
    adj_pdg = [str(v) for v in run[tree]["AdjClMainPDG"][idx] if v != 0]
    adj_q   = [v for v in run[tree]["AdjClCharge"][idx]  if v != 0]
    na      = min(len(adj_x), len(adj_y), len(adj_z))
    reco_x   = [float(run[tree]["RecoX"][idx])]           + adj_x[:na]
    reco_y   = [float(run[tree]["RecoY"][idx])]           + adj_y[:na]
    reco_z   = [float(run[tree]["RecoZ"][idx])]           + adj_z[:na]
    reco_e   = [float(run[tree]["SignalParticleK"][idx])] + [float("nan")] * na
    reco_pdg = [str(run[tree]["MainPDG"][idx])]           + adj_pdg[:na]
    reco_q   = [float(run[tree]["Charge"][idx])]          + [float(v) for v in adj_q[:na]]

    return pd.DataFrame([
        {
            "Config": config, "Name": name, "Variable": "truth",
            "Title": title, "Event": event_num,
            "X": truth_x, "Y": truth_y, "Z": truth_z,
            "E": truth_e, "PDG": truth_pdg,
            "Charge": [float("nan")] * len(truth_x),
            "Purity": [float("nan")] * len(truth_x),
        },
        {
            "Config": config, "Name": name, "Variable": "reco",
            "Title": title, "Event": event_num,
            "X": reco_x, "Y": reco_y, "Z": reco_z,
            "E": reco_e, "PDG": reco_pdg,
            "Charge": reco_q,
            "Purity": [float(run[tree]["Purity"][idx])] + [float(v) for v in run[tree]["AdjClPur"][idx][:na]],
        },
    ])


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
# Legend helpers
# ---------------------------------------------------------------------------

def _relabel_edep_legend(fig):
    """Rename plot_edep_event's internal legend groups to physics-meaningful labels.

    plot_edep_event assigns:
      legendgroup="1" (title "Raw")    → Geant4 energy deposits
      legendgroup="0" (title "Reco")   → CC daughter track starts (truth, not reco)
    """
    fig.update_traces(legendgrouptitle_text="EDep Truth",   selector=dict(legendgroup="1"))
    fig.update_traces(legendgrouptitle_text="Signal Truth", selector=dict(legendgroup="0"))


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

    if event_idx is not None:
        truth_matches = np.where(np.asarray(run["Truth"]["Event"]) == event_idx)[0]
        if len(truth_matches) == 0:
            raise ValueError(f"Event {event_idx} not found in EDep Truth array")
        event_idx = int(truth_matches[0])

    fig, idx = plot_edep_event(run, configs, idx=event_idx, tracked="Truth", zoom=False)
    fig.update_layout(width=1200, height=600)
    _relabel_edep_legend(fig)

    event_num = int(run["Truth"]["Event"][idx])
    if not args.no_show:
        fig.show()
    if args.save:
        _save_fig(fig, args, f"{config}_{name}_EDep_event_{event_num}")

    df = _build_edep_df(run, config, name, idx, tree="Truth")
    return fig, event_num, df


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
        _save_fig(fig, args, f"{config}_{name}_PDS_event_{idx}")

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
        _save_fig(fig, args, f"{config}_{name}_OpHit_event_{idx}")

    return fig, idx


def run_tpc(args, event_idx):
    config = getattr(args, "tpc_config")
    name   = getattr(args, "tpc_name")
    configs = {config: [name]}

    run, output = load_multi(configs, preset="VERTEXING", debug=args.debug)
    run = compute_reco_workflow(run, configs, params={}, workflow="VERTEXING", debug=False)

    info = json.load(open(ROOT / "config" / config / f"{config}_config.json"))
    run["Reco"]["NAdjCl"] = np.sum(run["Reco"]["AdjClCharge"] > 0, axis=1)

    tpc_params = {
        ("Reco", "Geometry"): ("equal", info["GEOMETRY"]),
        ("Reco", "Version"):  ("equal", info["VERSION"]),
    }
    if args.cut_flash:
        tpc_params[("Reco", "MatchedOpFlashPur")] = ("bigger", 0)
    if args.cut_energy is not None:
        tpc_params[("Reco", "SignalParticleK")] = ("smaller", args.cut_energy)
    if args.min_adjcl is not None:
        tpc_params[("Reco", "NAdjCl")] = ("bigger", args.min_adjcl - 1)

    this_run, _, _ = compute_filtered_run(
        run, configs,
        presets=["VERTEXING"],
        params=tpc_params,
        debug=args.debug,
    )

    reco = this_run["Reco"]
    if "AdjClRecoX" not in reco:
        adj_dt = reco["AdjClTime"] - reco["Time"][:, np.newaxis]
        recox  = reco["RecoX"][:, np.newaxis]
        scale  = info["DETECTOR_SIZE_X"] / 2 / info["EVENT_TICKS"]
        sign   = np.where(reco["TPC"][:, np.newaxis] % 2 == 0, 1.0, -1.0)
        reco["AdjClRecoX"] = adj_dt * sign * scale + recox
    if "AdjClRecoY" not in reco:
        reco["AdjClRecoY"] = reco["AdjClMainY"]
    if "AdjClRecoZ" not in reco:
        reco["AdjClRecoZ"] = reco["AdjClMainZ"]

    if event_idx is not None:
        matches = np.where(np.asarray(this_run["Reco"]["Event"]) == event_idx)[0]
        if len(matches) == 0:
            raise ValueError(f"Event {event_idx} not found in TPC filtered array (filtered out by cuts?)")
        event_idx = int(matches[0])

    fig, idx = plot_tpc_event(
        this_run, configs,
        idx=event_idx,
        tracked="Reco",
        adjclnum=1,
        get_adj_color=True,
        unzoom=1.25,
        projection=args.layout,
        debug=args.debug,
    )
    fig.update_layout(width=1200, height=600)

    if not args.no_show:
        fig.show()
    event_num = int(this_run["Reco"]["Event"][idx])
    if args.save:
        _save_fig(fig, args, f"{config}_{name}_TPC_event_{event_num}")

    df = _build_tpc_df(this_run, config, name, idx, tree="Reco")
    return fig, event_num, df


# ---------------------------------------------------------------------------
# Hybrid EDep + TPC
# ---------------------------------------------------------------------------

def run_edep_tpc(args, event_idx):
    config = getattr(args, "tpc_config")
    name   = getattr(args, "tpc_name")
    configs = {config: [name]}

    # EDEP: truth deposits (same ROOT file as VERTEXING → event numbers match)
    run_e, _ = load_multi(configs, preset="EDEP", debug=args.debug)
    run_e = compute_reco_workflow(run_e, configs, {}, workflow="EDEP", debug=args.debug)

    # VERTEXING: reco clusters (main + adj) with correctly reconstructed RecoX/Y/Z
    run_v, _ = load_multi(configs, preset="VERTEXING", debug=args.debug)

    info = json.load(open(ROOT / "config" / config / f"{config}_config.json"))

    # Same AdjClRecoX fix as run_tpc
    reco_v = run_v["Reco"]
    if "AdjClRecoX" not in reco_v:
        adj_dt = reco_v["AdjClTime"] - reco_v["Time"][:, np.newaxis]
        scale  = info["DETECTOR_SIZE_X"] / 2 / info["EVENT_TICKS"]
        sign   = np.where(reco_v["TPC"][:, np.newaxis] % 2 == 0, 1.0, -1.0)
        reco_v["AdjClRecoX"] = adj_dt * sign * scale + reco_v["RecoX"][:, np.newaxis]
    if "AdjClRecoY" not in reco_v:
        reco_v["AdjClRecoY"] = reco_v["AdjClMainY"]
    if "AdjClRecoZ" not in reco_v:
        reco_v["AdjClRecoZ"] = reco_v["AdjClMainZ"]

    # Resolve --event to EDEP Truth array index
    if event_idx is not None:
        truth_matches = np.where(np.asarray(run_e["Truth"]["Event"]) == event_idx)[0]
        if len(truth_matches) == 0:
            raise ValueError(f"Event {event_idx} not found in EDep Truth array")
        truth_edep_idx = int(truth_matches[0])
    else:
        truth_edep_idx = None

    # Truth EDep deposits → base figure
    fig, truth_idx = plot_edep_event(run_e, configs, idx=truth_edep_idx,
                                     tracked="Truth", zoom=False)
    fig.update_layout(width=1200, height=600)
    _relabel_edep_legend(fig)
    event_num = int(run_e["Truth"]["Event"][truth_idx])

    # Overlay VERTEXING reco clusters (main + adj)
    reco_matches = np.where(np.asarray(run_v["Reco"]["Event"]) == event_num)[0]
    tpc_reco_df = None
    if len(reco_matches) > 0:
        r = int(reco_matches[0])
        tpc_reco_df = _build_tpc_df(run_v, config, name, r, tree="Reco")
        reco_row = tpc_reco_df[tpc_reco_df["Variable"] == "reco"].iloc[0]

        # Main cluster (index 0) + adj clusters (1+) in one overlay call
        reco_xyz  = [reco_row["X"], reco_row["Y"], reco_row["Z"]]
        reco_pdgs = reco_row["PDG"]
        colors = ["red"] + [
            list(get_pdg_color([pdg]).values())[0] for pdg in reco_pdgs[1:]
        ]
        symbols = ["circle"] + ["circle-open"] * (len(reco_pdgs) - 1)
        fig = add_data_to_event(fig, info["GEOMETRY"], reco_xyz, "2",
                                "Reco Cluster", "main",
                                reco_pdgs, symbols, 15, colors, {"lw": 1},
                                projection=args.layout)

    if not args.no_show:
        fig.show()
    if args.save:
        _save_fig(fig, args, f"{config}_{name}_EDep-TPC_event_{event_num}")

    edep_df = _build_edep_df(run_e, config, name, truth_idx, tree="Truth")
    # edep-tpc: truth=edep deposits only, reco=RecoXYZ+AdjCl (no ccint row)
    df = edep_df[edep_df["Variable"] == "truth"].copy()
    if tpc_reco_df is not None:
        df = pd.concat([df, tpc_reco_df[tpc_reco_df["Variable"] == "reco"].copy()],
                       ignore_index=True)
    return fig, event_num, df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.save:
        args.output.mkdir(parents=True, exist_ok=True)
        args.data_output.mkdir(parents=True, exist_ok=True)

    plots = set(args.plots)
    dfs = {}

    # When edep+tpc both run without a fixed event, TPC drives event selection
    # so we only pick events that exist in both Truth and VERTEXING Reco.
    synced_event = args.event
    if "tpc" in plots:
        print("[TPC] Loading and plotting...")
        _, tpc_idx, tpc_df = run_tpc(args, synced_event)
        dfs["tpc"] = (args.tpc_config, args.tpc_name, tpc_idx, tpc_df)
        print(f"[TPC] Done — event {tpc_idx}, {len(tpc_df)} cluster points")
        synced_event = tpc_idx  # propagate resolved event number to other plots

    if "edep" in plots:
        print("[EDep] Loading and plotting...")
        _, edep_idx, edep_df = run_edep(args, synced_event)
        dfs["edep"] = (args.edep_config, args.edep_name, edep_idx, edep_df)
        print(f"[EDep] Done — event {edep_idx}, {len(edep_df)} deposition points")

    if "adjflash" in plots:
        print("[AdjFlash] Loading and plotting...")
        _, adjflash_idx = run_adjflash(args, synced_event)
        print(f"[AdjFlash] Done — event {adjflash_idx}")

    if "ophit" in plots:
        print("[OpHit] Loading and plotting...")
        _, ophit_idx = run_ophit(args, synced_event)
        print(f"[OpHit] Done — event {ophit_idx}")

    if "edep-tpc" in plots:
        print("[EDep-TPC] Loading and plotting...")
        _, hybrid_idx, hybrid_df = run_edep_tpc(args, synced_event)
        dfs["edep-tpc"] = (args.tpc_config, args.tpc_name, hybrid_idx, hybrid_df)
        print(f"[EDep-TPC] Done — event {hybrid_idx}, {len(hybrid_df)} rows")
        synced_event = hybrid_idx

    if args.save:
        for plot, (cfg, nm, evt, df) in dfs.items():
            pkl_path = args.data_output / f"{cfg}_{nm}_{plot}_event_{evt}_display.pkl"
            df.to_pickle(pkl_path)
            print(f"Saved DataFrame → {pkl_path}")

    return {plot: df for plot, (_, _, _, df) in dfs.items()}


if __name__ == "__main__":
    main()
