#!/usr/bin/env python3
"""
edep_electron_analysis.py

Characterizes the primary electron track (PDG=11) from CC νe interactions
by analysing energy deposits (TSignalPDGDepList / TSignalEDepList).

Goal: justify TPC clustering parameter N_adj=3 at 5 mm wire pitch (15 mm gap
limit). Iterative clustering joins deposits above a threshold when consecutive
gaps are <= N_adj * pitch. The threshold scan shows which value maximises the
fraction of events where electron deposits form a fully connected chain.

Key physics: at low threshold, secondary electrons (delta rays, bremsstrahlung
Compton scatters — all PDG=11) are included and create outlier deposits far
from the primary track, breaking chain connectivity. Raising the threshold
removes these secondaries, leaving only the core track. The fraction of
connected events therefore rises with threshold, then drops once real primary
deposits are lost. The plateau region identifies the justified threshold.

Usage:
  python edep_electron_analysis.py [--save] [--html] [--no-show] [--debug]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lib import (
    load_multi,
    compute_reco_workflow,
    format_coustom_plotly,
    save_figure,
    save_df,
)

PITCH_CM     = 0.5
N_ADJ_DEF    = 3
ELECTRON_PDG = 11
DEFAULT_CONFIG = "hd_1x2x6_centralAPA"
DEFAULT_NAME   = "marley_edep"

SAVE_PATH  = str(ROOT / "output" / "images" / "event")
DATA_PATH  = str(ROOT / "output" / "data"   / "event")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="EDep electron track characterisation for TPC clustering justification"
    )
    p.add_argument("--config",  default=DEFAULT_CONFIG)
    p.add_argument("--name",    default=DEFAULT_NAME)
    p.add_argument("--pitch",   type=float, default=5.0, metavar="MM",
                   help="Wire pitch in mm (default: 5.0)")
    p.add_argument("--n-adj",   type=int, default=N_ADJ_DEF, dest="n_adj",
                   help="Adjacency channel parameter to justify (default: 3)")
    p.add_argument("--thresholds", nargs="+", type=float, default=None, metavar="MEV",
                   help="EDep thresholds in MeV (default: 20 linear-spaced 0.1–2.0, step 0.1)")
    p.add_argument("--save",    action="store_true")
    p.add_argument("--html",    action="store_true")
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--debug",   action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Core per-event analysis
# ---------------------------------------------------------------------------

def _max_consecutive_gap(positions: np.ndarray) -> float:
    if len(positions) < 2:
        return 0.0
    return float(np.max(np.diff(np.sort(positions))))


def analyse_events(run, config, name, thresholds, pitch_cm, n_adj, debug=False):
    """
    Loop over all events × all thresholds.
    Returns long-form DataFrame: one row per (event, threshold).
    """
    gap_limit = n_adj * pitch_cm
    truth    = run["Truth"]
    n_events = len(truth["Event"])

    pdg_all  = truth["TSignalPDGDepList"]
    e_all    = truth["TSignalEDepList"]
    y_all    = truth["TSignalYDepList"]
    z_all    = truth["TSignalZDepList"]
    nu_e_all = truth["SignalParticleE"]
    ev_nums  = truth["Event"]

    if debug:
        print(f"[analyse] {n_events} events, {len(thresholds)} thresholds")

    records = []
    for thr in thresholds:
        n_skip = 0
        for idx in range(n_events):
            pdg_raw = np.asarray(pdg_all[idx])
            e_raw   = np.asarray(e_all[idx])
            y_raw   = np.asarray(y_all[idx])
            z_raw   = np.asarray(z_all[idx])

            nonzero  = e_raw != 0
            el_mask  = nonzero & (pdg_raw == ELECTRON_PDG) & (e_raw >= thr)
            n_el     = int(el_mask.sum())

            e_el  = float(e_raw[el_mask].sum()) if n_el > 0 else 0.0
            e_tot = float(e_raw[nonzero].sum())
            e_frac = (e_el / e_tot) if e_tot > 0 else float("nan")

            if n_el < 2:
                n_skip += 1
                records.append({
                    "Event":         int(ev_nums[idx]),
                    "Config":        config,
                    "Name":          name,
                    "threshold_mev": float(thr),
                    "n_el_deps":     n_el,
                    "e_el":          e_el,
                    "e_frac":        e_frac,
                    "max_gap_y_cm":  float("nan"),
                    "max_gap_z_cm":  float("nan"),
                    "max_gap_yz_cm": float("nan"),
                    "connected":     False,
                    "nu_energy_mev": float(nu_e_all[idx]),
                })
                continue

            gap_y  = _max_consecutive_gap(y_raw[el_mask])
            gap_z  = _max_consecutive_gap(z_raw[el_mask])
            gap_yz = max(gap_y, gap_z)

            records.append({
                "Event":         int(ev_nums[idx]),
                "Config":        config,
                "Name":          name,
                "threshold_mev": float(thr),
                "n_el_deps":     n_el,
                "e_el":          e_el,
                "e_frac":        e_frac,
                "max_gap_y_cm":  gap_y,
                "max_gap_z_cm":  gap_z,
                "max_gap_yz_cm": gap_yz,
                "connected":     bool(gap_yz <= gap_limit),
                "nu_energy_mev": float(nu_e_all[idx]),
            })

        if debug:
            print(f"  thr={thr:.4f} MeV — {n_skip}/{n_events} events skipped (<2 el deposits)")

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Figure 1: connected fraction + energy-weighted fraction vs threshold
# ---------------------------------------------------------------------------

def fig_connected_vs_threshold(df, pitch_cm, n_adj_target):
    n_adj_range = [1, 2, 3, 4]
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]
    thresholds = sorted(df["threshold_mev"].unique())

    fig = make_subplots(rows=1, cols=1)

    for i, (n, col) in enumerate(zip(n_adj_range, colors)):
        gap_lim  = n * pitch_cm
        fracs    = []
        e_fracs  = []

        for thr in thresholds:
            sub = df[df["threshold_mev"] == thr].dropna(subset=["max_gap_yz_cm"])
            if len(sub) == 0:
                fracs.append(float("nan"))
                e_fracs.append(float("nan"))
                continue

            connected = sub["max_gap_yz_cm"] <= gap_lim
            fracs.append(float(connected.mean()))

            e_conn  = sub.loc[connected, "e_el"].sum()
            e_total = sub["e_el"].sum()
            e_fracs.append(float(e_conn / e_total) if e_total > 0 else float("nan"))

        width = 3 if n == n_adj_target else 1.5

        # Color-legend group: one solid entry per N_adj
        fig.add_trace(go.Scatter(
            x=thresholds, y=fracs,
            mode="lines+markers",
            name=f"N_adj = {n}  ({n * pitch_cm * 10:.0f} mm)",
            line=dict(color=col, dash="solid", width=width),
            legendgroup="color",
            legendgrouptitle_text="N_adj (channels)" if i == 0 else "",
        ))

        # Energy-weighted trace — same color, dotted; no legend entry (linetype group handles it)
        fig.add_trace(go.Scatter(
            x=thresholds, y=e_fracs,
            mode="lines",
            name=f"N_adj = {n} [E-weighted]",
            line=dict(color=col, dash="dot", width=width * 0.8),
            showlegend=False,
        ))

    # Linetype legend: dummy traces (x/y=None so nothing plotted)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        name="Event fraction",
        line=dict(color="black", dash="solid", width=2),
        legendgroup="linetype",
        legendgrouptitle_text="Line type",
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        name="Energy-weighted",
        line=dict(color="black", dash="dot", width=1.5),
        legendgroup="linetype",
    ))

    fig.add_hline(y=0.9, line_dash="dot", line_color="grey",
                  annotation_text="90 %", annotation_position="top left")

    fig = format_coustom_plotly(
        fig,
        title=(
            f"Connected electron cluster fraction vs EDep threshold  "
            f"[pitch = {pitch_cm * 10:.0f} mm, highlighted N_adj = {n_adj_target}]"
        ),
        matches=("x", None),
        tickformat=(".3g", ".2f"),
    )
    fig.update_xaxes(title_text="EDep threshold (MeV)")
    fig.update_yaxes(title_text="Fraction of events / energy connected",
                     range=[0, 1.05])
    return fig


# ---------------------------------------------------------------------------
# Figure 2: gap distribution at a given threshold
# ---------------------------------------------------------------------------

def fig_gap_distribution(df, thr, gap_limit_cm, pitch_cm, n_adj):
    sub = df[df["threshold_mev"] == thr].dropna(subset=["max_gap_y_cm"])
    max_show = gap_limit_cm * 5

    fig = make_subplots(rows=1, cols=1)

    for key, label, color in [("max_gap_y_cm", "Y", "#636EFA"),
                               ("max_gap_z_cm", "Z", "#EF553B")]:
        vals   = np.sort(sub[key].values)
        cdf    = np.arange(1, len(vals) + 1) / len(vals)
        x_plot = np.concatenate([[0], vals])
        y_plot = np.concatenate([[0], cdf])

        fig.add_trace(go.Scatter(
            x=x_plot, y=y_plot, mode="lines",
            name=f"Max gap {label}",
            line=dict(color=color, width=2),
        ))

    fig.add_vline(x=gap_limit_cm, line_dash="dash", line_color="red",
                  annotation_text=f"{n_adj}×{pitch_cm*10:.0f} mm")
    fig.add_hline(y=0.9, line_dash="dot", line_color="grey",
                  annotation_text="90 %", annotation_position="top left")

    fig = format_coustom_plotly(
        fig,
        title=f"Cumulative max inter-deposit gap at threshold = {thr:.2f} MeV",
        matches=("x", None),
        tickformat=(".3g", ".2f"),
    )
    fig.update_xaxes(title_text="Max consecutive gap (cm)", range=[0, max_show])
    fig.update_yaxes(title_text="Fraction of events", range=[0, 1.05])
    return fig


# ---------------------------------------------------------------------------
# Publication DataFrame: gap CDF
# ---------------------------------------------------------------------------

def _build_gap_cdf_df(df_analysis, config, name, thr, pitch_cm, n_adj):
    """
    Build a publication-compatible DataFrame (LOWE_RECONSTRUCTION_PUBLICATION schema)
    from the empirical CDF of max inter-deposit gap at a given threshold.

    One row per wire direction (Y, Z). Arrays stored as lists in cells.
    Columns: Config, Name, Variable, Threshold, NAdj, PitchMM, GapLimit,
             GapBins, CDF
    """
    sub = df_analysis[df_analysis["threshold_mev"] == thr].dropna(subset=["max_gap_y_cm"])
    rows = []
    for key, label in [("max_gap_y_cm", "Y"), ("max_gap_z_cm", "Z")]:
        vals   = np.sort(sub[key].values)
        cdf    = np.arange(1, len(vals) + 1) / len(vals)
        x_plot = np.concatenate([[0.0], vals]).tolist()
        y_plot = np.concatenate([[0.0], cdf]).tolist()
        rows.append({
            "Config":    config,
            "Name":      name,
            "Variable":       label,
            "Threshold":      float(thr),
            "#AdjChannels":   int(n_adj),
            "Pitch":          float(pitch_cm * 10),
            "GapLimit":  float(n_adj * pitch_cm),
            "GapBins":   x_plot,
            "CDF":       y_plot,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Publication DataFrame: gap percentiles
# ---------------------------------------------------------------------------

def _build_gap_percentiles_df(df_analysis, config, name, pitch_cm, n_adj):
    """
    One row per percentile (50, 90, 95).
    Columns: Config, Name, Variable, #AdjChannels, Pitch, GapLimit,
             Thresholds, GapPercentile
    """
    thresholds = sorted(df_analysis["threshold_mev"].unique())
    rows = []
    for p in [50, 90, 95]:
        vals = []
        for thr in thresholds:
            sub = df_analysis[df_analysis["threshold_mev"] == thr].dropna(subset=["max_gap_yz_cm"])
            vals.append(float(np.nanpercentile(sub["max_gap_yz_cm"], p)) if len(sub) else float("nan"))
        rows.append({
            "Config":         config,
            "Name":           name,
            "Variable":       f"p{p}",
            "#AdjChannels":   int(n_adj),
            "Pitch":          float(pitch_cm * 10),
            "GapLimit":       float(n_adj * pitch_cm),
            "Thresholds":     list(thresholds),
            "GapPercentile":  vals,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure 3: gap percentiles vs threshold
# ---------------------------------------------------------------------------

def fig_gap_percentiles(df, gap_limit_cm):
    thresholds = sorted(df["threshold_mev"].unique())
    pcts = {50: [], 90: [], 95: []}
    for thr in thresholds:
        sub = df[df["threshold_mev"] == thr].dropna(subset=["max_gap_yz_cm"])
        for p, lst in pcts.items():
            lst.append(float(np.nanpercentile(sub["max_gap_yz_cm"], p)) if len(sub) else float("nan"))

    styles = {50: ("solid", 2, "#636EFA"),
              90: ("dash",  1.5, "#EF553B"),
              95: ("dot",   1.5, "#FFA15A")}

    fig = make_subplots(rows=1, cols=1)
    for p, vals in pcts.items():
        dash, width, col = styles[p]
        fig.add_trace(go.Scatter(
            x=thresholds, y=vals, mode="lines",
            name=f"p{p}",
            line=dict(dash=dash, width=width, color=col),
        ))

    fig.add_hline(y=gap_limit_cm, line_dash="dot", line_color="red",
                  annotation_text=f"Gap limit ({gap_limit_cm*10:.0f} mm)",
                  annotation_position="top right")

    fig = format_coustom_plotly(
        fig,
        title="Max electron deposit gap percentiles vs threshold",
        matches=("x", None),
        tickformat=(".3g", ".3g"),
        legend_title="Percentile",
    )
    fig.update_xaxes(title_text="EDep threshold (MeV)")
    fig.update_yaxes(title_text="Max gap (cm)")
    return fig


# ---------------------------------------------------------------------------
# Figure 4: N deposits vs threshold
# ---------------------------------------------------------------------------

def fig_n_deposits(df):
    thresholds = sorted(df["threshold_mev"].unique())
    medians = [float(df[df["threshold_mev"] == t]["n_el_deps"].median()) for t in thresholds]

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(
        x=thresholds, y=medians, mode="lines+markers",
        line=dict(color="#636EFA", width=2),
        name="Median deposits",
    ))

    fig = format_coustom_plotly(
        fig,
        title="Median surviving electron deposits per event vs threshold",
        matches=("x", None),
        tickformat=(".3g", ".3g"),
    )
    fig.update_xaxes(title_text="EDep threshold (MeV)")
    fig.update_yaxes(title_text="Median n electron deposits")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args     = parse_args()
    pitch_cm = args.pitch / 10.0
    gap_limit = args.n_adj * pitch_cm

    configs = {args.config: [args.name]}

    print(f"[EDEP] Loading: config={args.config}, name={args.name}")
    run, _ = load_multi(configs, preset="EDEP", debug=args.debug)
    run     = compute_reco_workflow(run, configs, {}, workflow="EDEP", debug=args.debug)
    print(f"[EDEP] {len(run['Truth']['Event'])} events loaded")

    if args.thresholds is None:
        thresholds = np.linspace(0.1, 2.0, 20)
    else:
        thresholds = np.array(args.thresholds)

    single_thr = len(thresholds) == 1
    print(f"[analyse] Scanning {len(thresholds)} threshold(s): "
          f"{thresholds[0]:.4f} – {thresholds[-1]:.4f} MeV")

    df = analyse_events(run, args.config, args.name, thresholds,
                        pitch_cm, args.n_adj, debug=args.debug)

    stem = "edep_electron"

    if args.save:
        save_df(df, DATA_PATH, config=args.config, name=args.name,
                filename="edep_electron_analysis", rm=True, debug=True)

    def _show_save(fig, filename):
        if not args.no_show:
            fig.show()
        if args.save:
            save_figure(fig, SAVE_PATH, config=args.config, name=args.name,
                        filename=filename, rm=True, filetype="png", debug=True)
        if args.html:
            save_figure(fig, SAVE_PATH, config=args.config, name=args.name,
                        filename=filename, rm=True, filetype="html", debug=True)

    if single_thr:
        thr_val = float(thresholds[0])
        sub = df[df["threshold_mev"] == thr_val]
        n_total = len(sub)
        n_conn  = int(sub["connected"].sum())
        print(f"\nConnected events: {n_conn}/{n_total} = {100*n_conn/n_total:.1f}%")
        f2 = fig_gap_distribution(df, thr_val, gap_limit, pitch_cm, args.n_adj)
        _show_save(f2, f"{stem}_gap_dist_thr{thr_val:.4f}")
    else:
        # Find optimal threshold (max connected fraction for target N_adj)
        thr_vals = sorted(df["threshold_mev"].unique())
        fracs = []
        for t in thr_vals:
            sub = df[df["threshold_mev"] == t].dropna(subset=["max_gap_yz_cm"])
            fracs.append(float((sub["max_gap_yz_cm"] <= gap_limit).mean()) if len(sub) else 0.0)
        opt_thr = float(thr_vals[int(np.argmax(fracs))])
        print(f"\n[analyse] Optimal threshold: {opt_thr:.4f} MeV  "
              f"(connected fraction = {max(fracs)*100:.1f}%)")

        # Use the closest available threshold to 1 MeV as fixed reference for gap CDF
        ref_thr = float(min(thr_vals, key=lambda t: abs(t - 1.0)))
        print(f"[analyse] Gap CDF reference threshold: {ref_thr:.4f} MeV")

        f1 = fig_connected_vs_threshold(df, pitch_cm, args.n_adj)
        f2 = fig_gap_distribution(df, ref_thr, gap_limit, pitch_cm, args.n_adj)
        f3 = fig_gap_percentiles(df, gap_limit)
        f4 = fig_n_deposits(df)

        _show_save(f1, f"{stem}_connected_vs_thr")
        _show_save(f2, f"{stem}_gap_dist_opt")
        _show_save(f3, f"{stem}_gap_percentiles")
        _show_save(f4, f"{stem}_n_deposits")

        if args.save:
            cdf_df = _build_gap_cdf_df(df, args.config, args.name,
                                       ref_thr, pitch_cm, args.n_adj)
            save_df(cdf_df, DATA_PATH, config=args.config, name=args.name,
                    filename="edep_electron_gap_cdf", rm=True, debug=True)

            pct_df = _build_gap_percentiles_df(df, args.config, args.name,
                                               pitch_cm, args.n_adj)
            save_df(pct_df, DATA_PATH, config=args.config, name=args.name,
                    filename="edep_electron_gap_percentiles", rm=True, debug=True)


if __name__ == "__main__":
    main()
