import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *

parser = argparse.ArgumentParser(
    description="1D Δχ² projections from sensitivity contour planes "
                "(sin²θ₁₂ vs Δm²₂₁ and sin²θ₁₃ vs Δm²₂₁)."
)
parser.add_argument("--config",  type=str, default="hd_1x2x6_centralAPA")
parser.add_argument("--name",    type=str, default="marley")
parser.add_argument("--folder",  type=str, default="Nominal",
                    choices=["Reduced", "Truncated", "Nominal"])
parser.add_argument("--energy",  type=str, default="SolarEnergy",
                    choices=["SignalParticleK", "ClusterEnergy", "TotalEnergy",
                             "SelectedEnergy", "SolarEnergy"])
parser.add_argument("--nhits",   type=int, default=None)
parser.add_argument("--ophits",  type=int, default=None)
parser.add_argument("--adjcls",  type=int, default=None)
parser.add_argument("--signal_uncertainty",     type=float, default=0.04)
parser.add_argument("--background_uncertainty", type=float, default=0.02)
parser.add_argument("--nuisance_profile",       type=str,   default=None)
parser.add_argument("--exposure", type=float, default=30.0)
parser.add_argument("--background", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--reference",  type=str, default="SENSITIVITY",
                    choices=["DayNight", "SENSITIVITY", "HEP"])
parser.add_argument(
    "--compare",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Overlay NuFit 6.1 (2025) reference profiles for comparison.",
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug",   action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--plot",    action=argparse.BooleanOptionalAction, default=True)
parser.add_argument(
    "--smooth_window", type=int, default=11,
    help="Savitzky-Golay window length for 1D chi2 profiles (odd int ≥ 5; 0 = no smoothing).",
)
args = parser.parse_args()

# ── NuFit 6.1 (2025) reference data ────────────────────────────────────────────
# Digitized from v61.fig-sun-tension.pdf (right panel: Δχ² vs Δm²₂₁).
# sin²θ₁₃ fixed at 0.0222 in that figure.
# dm2 values in eV²; chi2 values are Δχ² relative to global minimum.
#
# Sources:
#   MB22m       — solar (Super-K SKI–IV, SNO, Borexino, …) NuFit 6.1
#   AAG21       — solar alternative (Appec+Gonzalez-Garcia 2021)
#   KamLAND     — reactor Δm²₂₁ constraint
#   Solar+KamL  — combined solar + KamLAND
#   +JUNO       — projected with JUNO added
#
# Best-fit (1D marginalised) central values and ±1σ uncertainties from NuFit 6.1:
#   Δm²₂₁ (solar):  5.90 × 10⁻⁵ eV²  +0.55/−0.45 × 10⁻⁵
#   sin²θ₁₂ (solar): 0.303            +0.012/−0.011
#   sin²θ₁₃:         0.0222           ± 0.0007  (from all data)

_NUFIT61_DM2_PROFILES = {
    "MB22m": {
        "color": "#e8421a",
        "dash":  "solid",
        "dm2":   [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.3, 5.6, 5.9,
                  6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 10.0],
        "chi2":  [11.5, 9.0, 7.0, 5.2, 3.7, 2.4, 1.4, 0.8, 0.2, 0.0,
                  0.1, 0.6, 1.5, 3.0, 4.7, 6.5, 8.3, 11.0],
    },
    "AAG21": {
        "color": "#1a1a1a",
        "dash":  "dash",
        "dm2":   [2.0, 3.0, 4.0, 5.0, 5.5, 5.9, 6.5, 7.5, 8.5, 10.0],
        "chi2":  [11.0, 5.5, 2.2, 0.5, 0.1, 0.0, 0.6, 2.0, 4.5, 8.0],
    },
    "KamLAND": {
        "color": "#2ca02c",
        "dash":  "solid",
        "dm2":   [5.0, 6.0, 6.5, 7.0, 7.53, 8.0, 8.5, 9.0, 10.0],
        "chi2":  [10.5, 5.0, 2.5, 0.8, 0.0,  1.2, 4.0, 8.0, 12.0],
    },
    "Solar+KamL": {
        "color": "#1f6e1f",
        "dash":  "dot",
        "dm2":   [4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0],
        "chi2":  [8.5, 5.0, 2.5, 0.7, 0.1, 0.0, 0.4, 1.5, 4.0, 7.5],
    },
    "+JUNO": {
        "color": "#6a1e8a",
        "dash":  "dot",
        "dm2":   [6.5, 7.0, 7.3, 7.53, 7.7, 8.0, 8.5],
        "chi2":  [8.0, 2.5, 0.5,  0.0, 0.5, 2.5, 8.0],
    },
}

# 1D marginalised sin²θ₁₂ profiles (Gaussian approximation from NuFit 6.1)
#   Solar only (MB22m): bf=0.303, σ_lo=0.011, σ_hi=0.012
#   Solar+KamL combined: slightly narrower
_NUFIT61_SIN12_PROFILES = {
    "MB22m": {
        "color": "#e8421a", "dash": "solid",
        "bf": 0.303, "sigma_lo": 0.011, "sigma_hi": 0.012,
    },
    "KamLAND": {
        "color": "#2ca02c", "dash": "solid",
        "bf": 0.303, "sigma_lo": 0.020, "sigma_hi": 0.020,
    },
    "Solar+KamL": {
        "color": "#1f6e1f", "dash": "dot",
        "bf": 0.303, "sigma_lo": 0.010, "sigma_hi": 0.011,
    },
}

# 1D marginalised sin²θ₁₃ profiles (Gaussian, from all-data NuFit 6.1)
_NUFIT61_SIN13_PROFILES = {
    "NuFit 6.1 (all)": {
        "color": "#1a1a1a", "dash": "solid",
        "bf": 0.0222, "sigma_lo": 0.0007, "sigma_hi": 0.0007,
    },
}


def _gaussian_chi2(x_arr, bf, sigma_lo, sigma_hi):
    x = np.asarray(x_arr)
    chi2 = np.where(x < bf,
                    ((x - bf) / sigma_lo) ** 2,
                    ((x - bf) / sigma_hi) ** 2)
    return chi2


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    """Savitzky-Golay smoothing. Window auto-reduced for short arrays. 0 = no-op."""
    from scipy.signal import savgol_filter
    n = len(y)
    if window <= 0 or n < 5:
        return y
    w = min(window | 1, n if n % 2 == 1 else n - 1)  # keep odd, ≤ n
    w = max(w, 5)
    if w > n:
        return y
    return savgol_filter(y, window_length=w, polyorder=3)


def _has_coverage(vals: np.ndarray, chi2: np.ndarray, label: str, fig, row: int, col: int) -> bool:
    """Return True if axis has ≥3 unique non-NaN points. Otherwise annotate panel."""
    valid = ~np.isnan(chi2)
    n_unique = len(np.unique(vals[valid]))
    if n_unique >= 3:
        return True
    fig.add_annotation(
        text=f"Insufficient grid coverage<br>({n_unique} unique {label} point(s))<br>Rerun with full oscillation grid",
        xref="x domain", yref="y domain",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=13, color="gray"),
        row=row, col=col,
    )
    return False


# ── path helpers ────────────────────────────────────────────────────────────────

def _load_best_cut_map(info: dict):
    candidates = list(dict.fromkeys(["SENSITIVITY", args.reference.upper()]))
    for analysis in candidates:
        filepath = (
            f"{info['PATH']}/{analysis}/{args.folder.lower()}/{args.config}/{args.name}/"
            f"{args.config}_{args.name}_highest_{analysis}.pkl"
        )
        if os.path.exists(filepath):
            if args.debug:
                rprint(f"[cyan][INFO][/cyan] Best-cut map from {analysis}: {filepath}")
            return pickle.load(open(filepath, "rb"))
    rprint("[yellow][WARNING][/yellow] No best-cut map found; falling back to defaults.")
    return None


def _resolve_cuts(info):
    if args.nhits is not None and args.adjcls is not None and args.ophits is not None:
        return args.nhits, args.adjcls, args.ophits
    cut_map = _load_best_cut_map(info)
    key = (args.config, args.name, args.energy)
    if cut_map is not None and key in cut_map:
        c = cut_map[key]
        return int(c["NHits"]), int(c["AdjCl"]), int(c["OpHits"])
    rprint("[yellow][WARNING][/yellow] Falling back to NHits4 AdjCl10 OpHits4.")
    return 4, 10, 4


def _results_path(info, nhits, adjcl, ophits, profile_name):
    sig_path = f"{info['PATH']}/SENSITIVITY/{args.config}/{args.name}/{args.folder.lower()}/{args.energy}"
    suffix = (
        f"signal_{100*args.signal_uncertainty:.0f}%_and_background_{100*args.background_uncertainty:.0f}%"
        if args.background
        else f"signal_{100*args.signal_uncertainty:.0f}%_only"
    )
    return f"{sig_path}/results/{profile_name}/{suffix}"


# ── main ────────────────────────────────────────────────────────────────────────

save_path = f"{root}/output/images/analysis/sensitivity"
os.makedirs(save_path, exist_ok=True)

info         = json.loads(open(f"{root}/config/{args.config}/{args.config}_config.json").read())
analysis_info = load_analysis_info(str(root))
_nuisance_profiles = analysis_info.get("NUISANCE_PROFILES", {})
_default_profile   = analysis_info.get("DEFAULT_NUISANCE_PROFILE", "full")
profile_name       = args.nuisance_profile or _default_profile or "full"

nhits, adjcl, ophits = _resolve_cuts(info)
data_path = _results_path(info, nhits, adjcl, ophits, profile_name)

rprint(f"[cyan][INFO][/cyan] Loading chi2 dataframes from {data_path}")

prefix = f"{data_path}/{args.name}_{args.energy}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}"

solar_sin12_df = pd.read_pickle(f"{prefix}_solar_sin12_df.pkl").astype(float)
solar_sin13_df = pd.read_pickle(f"{prefix}_solar_sin13_df.pkl").astype(float)
react_sin12_df = pd.read_pickle(f"{prefix}_react_sin12_df.pkl").astype(float)
react_sin13_df = pd.read_pickle(f"{prefix}_react_sin13_df.pkl").astype(float)

# Replace 0 with NaN (unfilled grid cells)
for df in [solar_sin12_df, solar_sin13_df, react_sin12_df, react_sin13_df]:
    df.replace(0.0, np.nan, inplace=True)

global_min = np.nanmin([
    np.nanmin(solar_sin12_df.values),
    np.nanmin(solar_sin13_df.values),
])
react_global_min = np.nanmin([
    np.nanmin(react_sin12_df.values),
    np.nanmin(react_sin13_df.values),
])

if args.debug:
    rprint(f"[cyan]Solar global_min={global_min:.4f}  reactor global_min={react_global_min:.4f}[/cyan]")

# ── 1D projections ──────────────────────────────────────────────────────────────

dm2_vals    = solar_sin12_df.index.astype(float).values
sin12_vals  = solar_sin12_df.columns.astype(float).values
sin13_vals  = solar_sin13_df.columns.astype(float).values

dm2_vals_react    = react_sin12_df.index.astype(float).values
sin12_vals_react  = react_sin12_df.columns.astype(float).values
sin13_vals_react  = react_sin13_df.columns.astype(float).values

_w = args.smooth_window

# Δχ²(dm2) — profile over sin12 (at fixed sin13=best-fit); sort by dm2 first
_sort_s = np.argsort(dm2_vals)
_sort_r = np.argsort(dm2_vals_react)
dm2_vals        = dm2_vals[_sort_s]
dm2_vals_react  = dm2_vals_react[_sort_r]
dchi2_dm2_solar = _smooth(np.nanmin(solar_sin12_df.values, axis=1)[_sort_s] - global_min,       _w)
dchi2_dm2_react = _smooth(np.nanmin(react_sin12_df.values, axis=1)[_sort_r] - react_global_min, _w)

# Δχ²(sin12) — profile over dm2 (at fixed sin13=best-fit); sort by sin12
_sort_s12  = np.argsort(sin12_vals)
_sort_r12  = np.argsort(sin12_vals_react)
sin12_vals       = sin12_vals[_sort_s12]
sin12_vals_react = sin12_vals_react[_sort_r12]
dchi2_sin12_solar = _smooth(np.nanmin(solar_sin12_df.values, axis=0)[_sort_s12] - global_min,       _w)
dchi2_sin12_react = _smooth(np.nanmin(react_sin12_df.values, axis=0)[_sort_r12] - react_global_min, _w)

# Δχ²(sin13) — profile over dm2 (at fixed sin12=best-fit); sort by sin13
_sort_s13  = np.argsort(sin13_vals)
_sort_r13  = np.argsort(sin13_vals_react)
sin13_vals       = sin13_vals[_sort_s13]
sin13_vals_react = sin13_vals_react[_sort_r13]
dchi2_sin13_solar = _smooth(np.nanmin(solar_sin13_df.values, axis=0)[_sort_s13] - global_min,       _w)
dchi2_sin13_react = _smooth(np.nanmin(react_sin13_df.values, axis=0)[_sort_r13] - react_global_min, _w)

sigma_levels = [1.0, 4.0, 9.0]   # Δχ² for 1σ, 2σ, 3σ (1 DOF)
sigma_labels = ["1σ", "2σ", "3σ"]

DUNE_COLOR_SOLAR = "#1f77b4"
DUNE_COLOR_REACT = "#ff7f0e"

compare_tag = "_NuFit61" if args.compare else ""
_cut_tag    = f"NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}"
_base_tag   = f"{args.config}_{args.name}_{args.folder}_{args.energy}_{_cut_tag}"
_subtitle   = f"{args.config} · {args.name} · {args.folder} · {args.energy} · {_cut_tag}"


def _add_sigma_lines(fig, row, col):
    for lvl, lbl in zip(sigma_levels, sigma_labels):
        fig.add_hline(
            y=lvl, row=row, col=col,
            line=dict(color="black", width=1, dash="dot"),
            annotation_text=lbl,
            annotation_position="top right",
            annotation_font_size=11,
        )


def _xref(col): return "x" if col == 1 else f"x{col}"
def _yref(col): return "y" if col == 1 else f"y{col}"

def _add_bf_vline(fig, x, col, y0=0, y1=12):
    """Add vertical reference line restricted to one subplot via explicit axis ref."""
    fig.add_shape(
        type="line",
        x0=x, x1=x, y0=y0, y1=y1,
        xref=_xref(col), yref=_yref(col),
        line=dict(color="gray", width=1, dash="longdash"),
    )




# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Δχ² vs Δm²₂₁
#   col 1: Asimov at solar best-fit (Δm²₂₁ = SOLAR_DM2)
#   col 2: Asimov at reactor best-fit (Δm²₂₁ = REACT_DM2)
# ══════════════════════════════════════════════════════════════════════════════

fig_dm2 = make_subplots(
    rows=1, cols=2,
    subplot_titles=[
        f"Solar reference  (Δm²₂₁ = {analysis_info['SOLAR_DM2']*1e5:.2f} × 10⁻⁵ eV²)",
        f"Reactor reference  (Δm²₂₁ = {analysis_info['REACT_DM2']*1e5:.2f} × 10⁻⁵ eV²)",
    ],
    horizontal_spacing=0.10,
)

for col, (x_dm2, y_chi2, color, label, bf_dm2) in enumerate([
    (dm2_vals,       dchi2_dm2_solar, DUNE_COLOR_SOLAR, "DUNE (solar ref.)",   analysis_info["SOLAR_DM2"]),
    (dm2_vals_react, dchi2_dm2_react, DUNE_COLOR_REACT, "DUNE (reactor ref.)", analysis_info["REACT_DM2"]),
], start=1):
    show_leg = col == 1
    if _has_coverage(x_dm2, y_chi2, "dm2", fig_dm2, 1, col):
        fig_dm2.add_trace(go.Scatter(
            x=x_dm2 * 1e5, y=y_chi2,
            mode="lines", name=label,
            line=dict(color=color, width=2.5),
            legendgroup=f"dune{col}", showlegend=show_leg,
        ), row=1, col=col)
    if args.compare:
        for ref_label, ref in _NUFIT61_DM2_PROFILES.items():
            fig_dm2.add_trace(go.Scatter(
                x=ref["dm2"], y=ref["chi2"],
                mode="lines", name=f"NuFit 6.1 {ref_label}",
                line=dict(color=ref["color"], width=1.5, dash=ref["dash"]),
                legendgroup=f"nf_{ref_label}", showlegend=show_leg,
            ), row=1, col=col)
    _add_sigma_lines(fig_dm2, 1, col)
    _add_bf_vline(fig_dm2, bf_dm2 * 1e5, col)
    fig_dm2.update_xaxes(title_text="Δm²<sub>21</sub> [10⁻⁵ eV²]", row=1, col=col, range=[2, 14])
    fig_dm2.update_yaxes(title_text="Δχ²", row=1, col=col, range=[0, 12])

fig_dm2 = format_coustom_plotly(fig_dm2, title=_subtitle, add_watermark=True)
fig_dm2.update_layout(
    height=520, width=1050,
    legend=dict(
        orientation="v", x=0.97, y=0.97,
        xanchor="right", yanchor="top",
        bgcolor="rgba(255,255,255,0.85)", bordercolor="lightgray", borderwidth=1,
        font=dict(size=11),
    ),
)

if args.plot:
    save_figure(fig_dm2, path=save_path, filename=f"{_base_tag}_dm2_projections{compare_tag}", rm=args.rewrite, filetype="png", debug=args.plot)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Δχ² vs mixing angles
#   col 1: sin²θ₁₂  (profiled over dm2 at sin13=best-fit)
#   col 2: sin²θ₁₃  (profiled over dm2 at sin12=best-fit)
# ══════════════════════════════════════════════════════════════════════════════

fig_ang = make_subplots(
    rows=1, cols=2,
    subplot_titles=[
        "sin²θ<sub>12</sub>  (profiled over Δm²₂₁)",
        "sin²θ<sub>13</sub>  (profiled over Δm²₂₁)",
    ],
    horizontal_spacing=0.10,
)

# sin12 panel
if _has_coverage(sin12_vals, dchi2_sin12_solar, "sin12", fig_ang, 1, 1):
    fig_ang.add_trace(go.Scatter(
        x=sin12_vals, y=dchi2_sin12_solar,
        mode="lines", name="DUNE (solar ref.)",
        line=dict(color=DUNE_COLOR_SOLAR, width=2.5),
        legendgroup="dune_s",
    ), row=1, col=1)
    fig_ang.add_trace(go.Scatter(
        x=sin12_vals_react, y=dchi2_sin12_react,
        mode="lines", name="DUNE (reactor ref.)",
        line=dict(color=DUNE_COLOR_REACT, width=2.5),
        legendgroup="dune_r",
    ), row=1, col=1)

if args.compare:
    x_s12 = np.linspace(0.15, 0.55, 200)
    for ref_label, ref in _NUFIT61_SIN12_PROFILES.items():
        y = _gaussian_chi2(x_s12, ref["bf"], ref["sigma_lo"], ref["sigma_hi"])
        fig_ang.add_trace(go.Scatter(
            x=x_s12, y=y,
            mode="lines", name=f"NuFit 6.1 {ref_label}",
            line=dict(color=ref["color"], width=1.5, dash=ref["dash"]),
            legendgroup=f"nf12_{ref_label}",
        ), row=1, col=1)

_add_sigma_lines(fig_ang, 1, 1)
_add_bf_vline(fig_ang, analysis_info["SIN12"], 1)
fig_ang.update_xaxes(title_text="sin²θ<sub>12</sub>", row=1, col=1, range=[0.15, 0.55])
fig_ang.update_yaxes(title_text="Δχ²", row=1, col=1, range=[0, 12])

# sin13 panel
if _has_coverage(sin13_vals, dchi2_sin13_solar, "sin13", fig_ang, 1, 2):
    fig_ang.add_trace(go.Scatter(
        x=sin13_vals, y=dchi2_sin13_solar,
        mode="lines", name="DUNE (solar ref.)",
        line=dict(color=DUNE_COLOR_SOLAR, width=2.5),
        legendgroup="dune_s", showlegend=False,
    ), row=1, col=2)
    fig_ang.add_trace(go.Scatter(
        x=sin13_vals_react, y=dchi2_sin13_react,
        mode="lines", name="DUNE (reactor ref.)",
        line=dict(color=DUNE_COLOR_REACT, width=2.5),
        legendgroup="dune_r", showlegend=False,
    ), row=1, col=2)

if args.compare:
    x_s13 = np.linspace(0.010, 0.040, 400)
    for ref_label, ref in _NUFIT61_SIN13_PROFILES.items():
        y = _gaussian_chi2(x_s13, ref["bf"], ref["sigma_lo"], ref["sigma_hi"])
        fig_ang.add_trace(go.Scatter(
            x=x_s13, y=y,
            mode="lines", name=f"NuFit 6.1 {ref_label}",
            line=dict(color=ref["color"], width=1.5, dash=ref["dash"]),
            legendgroup=f"nf13_{ref_label}", showlegend=True,
        ), row=1, col=2)

_add_sigma_lines(fig_ang, 1, 2)
_add_bf_vline(fig_ang, analysis_info["SIN13"], 2)
fig_ang = format_coustom_plotly(fig_ang, title=_subtitle, add_watermark=True)
fig_ang.update_xaxes(title_text="sin²θ<sub>13</sub>", row=1, col=2, range=[0.010, 0.040])
fig_ang.update_yaxes(title_text="Δχ²", row=1, col=2, range=[0, 12])

fig_ang.update_layout(
    height=520, width=1050,
    legend=dict(
        orientation="v", x=0.97, y=0.97,
        xanchor="right", yanchor="top",
        bgcolor="rgba(255,255,255,0.85)", bordercolor="lightgray", borderwidth=1,
        font=dict(size=11),
    ),
)

if args.plot:
    save_figure(fig_ang, path=save_path, filename=f"{_base_tag}_angle_projections{compare_tag}", rm=args.rewrite, filetype="png", debug=args.plot)
