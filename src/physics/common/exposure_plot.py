import os
import sys
import pickle
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *

# ── STYLES ─────────────────────────────────────────────────────────────────────

_REFERENCE_COLORS = {
    "Gaussian":         compare[1],
    "Asimov":           "black",
    "ProfileLikelihood": compare[2],
}

_CL_LABELS = {1: 0.6827, 2: 0.9545, 3: 0.9973, 4: 0.99994, 5: 0.9999994}

_COMPARISON_STYLES = {
    ("Asimov",            "Raw"):      dict(color="black",     dash="dot",   width=2),
    ("Asimov",            "Smoothed"): dict(color="black",     dash="solid", width=3),
    ("Gaussian",          "Raw"):      dict(color=compare[1],  dash="dot",   width=2),
    ("Gaussian",          "Smoothed"): dict(color=compare[1],  dash="solid", width=3),
    ("ProfileLikelihood", "Raw"):      dict(color=compare[2],  dash="dot",   width=2),
    ("ProfileLikelihood", "Smoothed"): dict(color=compare[2],  dash="solid", width=3),
}

_REBIN_STYLE_MAP = {
    ("Raw",      "NoRebin"):       dict(color=compare[1], dash="dash",   width=2),
    ("Raw",      "AdaptiveRebin"): dict(color=compare[1], dash="dot",    width=3),
    ("Smoothed", "NoRebin"):       dict(color="black",    dash="dash",   width=2),
    ("Smoothed", "AdaptiveRebin"): dict(color="black",    dash="solid",  width=3),
}

_LEGEND_LAYOUT = dict(font=dict(size=14), bgcolor="rgba(255,255,255,0.7)")

DUNE_COLOR_SOLAR = "#1f77b4"
DUNE_COLOR_REACT = "#ff7f0e"

# NuFit 6.1 reference data (from projections_plot.py:61–120)
_NUFIT61_DM2_SOLAR = {
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
}

_NUFIT61_DM2_REACTOR = {
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

_NUFIT61_SIN13_PROFILES = {
    "NuFit 6.1 (all)": {
        "color": "#1a1a1a", "dash": "solid",
        "bf": 0.0222, "sigma_lo": 0.0007, "sigma_hi": 0.0007,
    },
}

# ── HELPERS ────────────────────────────────────────────────────────────────────

def _safe_array(values):
    return np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def _get_selection_cuts(config: str, name: str, energy: str, args: argparse.Namespace, analysis_key: str):
    """Resolve cuts: explicit args > best-cut pkl > defaults."""
    if args.nhits is not None and args.ophits is not None and args.adjcls is not None:
        return int(args.nhits), int(args.ophits), int(args.adjcls)

    info = json.loads(open(f"{root}/config/{config}/{config}_config.json").read())

    # Sensitivity path: SENSITIVITY/{folder}/{config}/{name}/...
    if analysis_key == "SENSITIVITY":
        sigma_path = (
            f"{info['PATH']}/SENSITIVITY/{args.folder.lower()}/"
            f"{config}/{name}/{config}_{name}_{getattr(args, 'pkl_label', 'highest')}_{analysis_key}.pkl"
        )
    else:
        sigma_path = (
            f"{info['PATH']}/{analysis_key}/{args.folder.lower()}/"
            f"{config}/{name}/{config}_{name}_{getattr(args, 'pkl_label', 'highest')}_{analysis_key}.pkl"
        )

    if not os.path.exists(sigma_path):
        return None

    try:
        sigma_map = pd.read_pickle(sigma_path)
        ref_plot = sigma_map[(config, name, energy)]
    except (KeyError, FileNotFoundError):
        return None

    nhits_value = args.nhits if args.nhits is not None else int(ref_plot["NHits"])
    ophits_value = args.ophits if args.ophits is not None else int(ref_plot["OpHits"])
    adjcl_value = args.adjcls if args.adjcls is not None else int(ref_plot["AdjCl"])
    return int(nhits_value), int(ophits_value), int(adjcl_value)


def _add_sigma_hlines(fig, sigmas=[1, 2, 3], x_annotation=2):
    """Add horizontal dashed lines for sigma levels + CL annotations."""
    for sigma, cl in zip(sigmas, [_CL_LABELS.get(s, 0.0) for s in sigmas]):
        fig.add_hline(y=sigma, line_dash="dash", line_color="black")
        fig.add_annotation(
            x=x_annotation, y=sigma + 0.2,
            text=f"{100*cl:.2f}% CL",
            showarrow=False,
        )


def _add_projection_sigma_lines(fig, row=1, col=1):
    """Add Δχ² = 1, 4, 9 horizontal lines for chi2 projections."""
    for lvl, lbl in zip([1.0, 4.0, 9.0], ["1σ", "2σ", "3σ"]):
        fig.add_hline(
            y=lvl, row=row, col=col,
            line=dict(color="black", width=1, dash="dot"),
            annotation_text=lbl,
            annotation_position="top right",
            annotation_font_size=11,
        )


def _add_bf_vline(fig, x, col=1):
    """Add vertical best-fit reference line."""
    xref = "x" if col == 1 else f"x{col}"
    yref = "y" if col == 1 else f"y{col}"
    fig.add_shape(
        type="line",
        x0=x, x1=x, y0=0, y1=12,
        xref=xref, yref=yref,
        line=dict(color="gray", width=1, dash="longdash"),
    )


def _gaussian_chi2(x_arr, bf, sigma_lo, sigma_hi):
    """Asymmetric Gaussian chi2 profile."""
    x = np.asarray(x_arr)
    chi2 = np.where(x < bf,
                    ((x - bf) / sigma_lo) ** 2,
                    ((x - bf) / sigma_hi) ** 2)
    return chi2


def _smooth_sg(y: np.ndarray, window: int) -> np.ndarray:
    """Savitzky-Golay smoothing. Window auto-reduced for short arrays. 0 = no-op."""
    from scipy.signal import savgol_filter
    n = len(y)
    if window <= 0 or n < 5:
        return y
    w = min(window | 1, n if n % 2 == 1 else n - 1)
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


# ── ARG PARSER ─────────────────────────────────────────────────────────────────

analysis_info = load_analysis_info(str(root))
_hep_default_ref = analysis_info.get("BEST_SIGMA_SIGNIFICANCE_REFERENCE", {}).get("HEP", "Asimov")
if _hep_default_ref not in ["Gaussian", "Asimov", "ProfileLikelihood"]:
    _hep_default_ref = "Asimov"

parser = argparse.ArgumentParser(
    description="Unified exposure and oscillation-parameter plotting for DayNight, HEP, and Sensitivity analyses"
)
parser.add_argument("--analysis", type=str, choices=["DayNight", "HEP", "Sensitivity"], default="DayNight")
parser.add_argument("--config", nargs="+", type=str, default=["hd_1x2x6_centralAPA"])
parser.add_argument("--name", nargs="+", type=str, default=["marley"])
parser.add_argument("--folder", type=str, default="Nominal", choices=["Reduced", "Truncated", "Nominal"])
parser.add_argument("--exposure", type=float, default=30)
parser.add_argument("--energy", nargs="+", type=str)
parser.add_argument("--nhits", type=int, default=None)
parser.add_argument("--ophits", type=int, default=None)
parser.add_argument("--adjcls", type=int, default=None)
parser.add_argument("--threshold", type=float, default=None)
parser.add_argument("--zoom", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--reference", type=str, default=None)
parser.add_argument("--compare", action=argparse.BooleanOptionalAction, default=False)
# HEP-only
parser.add_argument("--mode", type=str, choices=["exposure", "comparison", "rebin", "reference", "all"], default="exposure")
parser.add_argument("--pkl_label", type=str, default="highest")
# Sensitivity-only
parser.add_argument("--signal_uncertainty", type=float, default=None)
parser.add_argument("--background_uncertainty", type=float, default=None)
parser.add_argument("--nuisance_profile", type=str, default=None)
parser.add_argument("--background", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--smooth_window", type=int, default=11)

args = parser.parse_args()

# ── POST-PARSE DEFAULTS ────────────────────────────────────────────────────────

if args.reference is None:
    if args.analysis == "DayNight":
        args.reference = "Gaussian"
    elif args.analysis == "HEP":
        args.reference = _hep_default_ref
    else:
        args.reference = "Asimov"

if args.analysis != "Sensitivity":
    if args.threshold is None:
        _analysis_key = "DAYNIGHT" if args.analysis == "DayNight" else args.analysis.upper()
        args.threshold = get_analysis_threshold(str(root), _analysis_key, stage="SIGNIFICANCE", fallback=0.0)

if args.signal_uncertainty is None or args.background_uncertainty is None:
    _unc = analysis_info.get("ANALYSIS_UNCERTAINTIES", {}).get(
        "DAYNIGHT" if args.analysis == "DayNight" else args.analysis.upper(), {}
    )
    if args.signal_uncertainty is None:
        args.signal_uncertainty = float(
            _unc.get("signal_uncertainty", analysis_info.get("SIGNAL_ERROR", 0.04))
        )
    if args.background_uncertainty is None:
        args.background_uncertainty = float(
            _unc.get("background_uncertainty", analysis_info.get("BACKGROUND_ERROR", 0.02))
        )

# Energy defaults per analysis
if args.energy is None:
    if args.analysis == "DayNight":
        args.energy = ["ClusterEnergy", "TotalEnergy", "SelectedEnergy", "SolarEnergy"]
    elif args.analysis == "HEP":
        args.energy = ["SignalParticleK", "ClusterEnergy", "TotalEnergy", "SelectedEnergy", "SolarEnergy"]
    else:
        args.energy = ["SolarEnergy"]

# ── PATHS ──────────────────────────────────────────────────────────────────────

if args.analysis == "DayNight":
    save_path = f"{root}/images/analysis/day-night"
    data_path = f"{root}/data/analysis/day-night"
elif args.analysis == "HEP":
    save_path = f"{root}/images/analysis/hep"
    data_path = f"{root}/data/analysis/hep"
else:
    save_path = f"{root}/images/analysis/sensitivity"
    data_path = f"{root}/data/analysis/sensitivity"

for this_path in [save_path, data_path]:
    if not os.path.exists(this_path):
        os.makedirs(this_path)

# ── ACCUMULATORS ───────────────────────────────────────────────────────────────

exposure_records = []

# ── MAIN LOOP ──────────────────────────────────────────────────────────────────

for config, name, energy in product(args.config, args.name, args.energy):
    info = json.loads(open(f"{root}/config/{config}/{config}_config.json").read())

    # ══════════════════════════════════════════════════════════════════════════
    # DayNight
    # ══════════════════════════════════════════════════════════════════════════
    if args.analysis == "DayNight":
        sigma_path = f"{info['PATH']}/DAYNIGHT/{args.folder.lower()}/{config}/{name}/{config}_{name}_highest_DayNight.pkl"
        if not os.path.exists(sigma_path):
            rprint(f"[yellow][WARNING][/yellow] Missing best-cut map for {config} {name}")
            continue

        sigma = pickle.load(open(sigma_path, "rb"))
        try:
            ref_plot = sigma[(config, name, energy)]
        except KeyError:
            rprint(f"[yellow][WARNING][/yellow] Not found highest for {config} {name} {energy}")
            continue

        sigmas_df = pd.read_pickle(
            f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/DAYNIGHT/{args.folder.lower()}"
            f"/{config}/{name}/{config}_{name}_{energy}_DayNight_Results.pkl"
        )

        nhits_value = args.nhits if args.nhits is not None else int(ref_plot["NHits"])
        ophits_value = args.ophits if args.ophits is not None else int(ref_plot["OpHits"])
        adjcl_value = args.adjcls if args.adjcls is not None else int(ref_plot["AdjCl"])

        plot_sigmas = sigmas_df.loc[
            (sigmas_df["Config"] == config) * (sigmas_df["Name"] == name)
            * (sigmas_df["NHits"] == nhits_value) * (sigmas_df["OpHits"] == ophits_value)
            * (sigmas_df["AdjCl"] == adjcl_value)
        ].copy()

        if plot_sigmas.empty:
            rprint(f"[yellow][WARNING][/yellow] Missing payload for {config} {name} {energy}")
            continue

        exposure_values = np.asarray(plot_sigmas["Exposure"].values[0], dtype=float)
        raw_gaussian = _safe_array(plot_sigmas["RawGaussian"].values[0])
        smoothed_gaussian = _safe_array(plot_sigmas["Gaussian"].values[0])
        gaussian_upper = _safe_array(plot_sigmas["Gaussian+Error"].values[0])
        gaussian_lower = _safe_array(plot_sigmas["Gaussian-Error"].values[0])

        _has_asimov = "Asimov" in plot_sigmas.columns and "RawAsimov" in plot_sigmas.columns
        if _has_asimov:
            raw_asimov = _safe_array(plot_sigmas["RawAsimov"].values[0])
            smoothed_asimov = _safe_array(plot_sigmas["Asimov"].values[0])
            asimov_upper = _safe_array(plot_sigmas["Asimov+Error"].values[0])
            asimov_lower = _safe_array(plot_sigmas["Asimov-Error"].values[0])

        fig = make_subplots(rows=1, cols=1, subplot_titles=(f"{energy}",))

        add_reference_pair_traces(
            fig, x=exposure_values, y_raw=raw_gaussian, y_smoothed=smoothed_gaussian,
            name="Gaussian", raw_style=dict(color="black", dash="dot", width=2),
            smoothed_style=dict(color="black", dash="solid", width=3),
            row=1, col=1, legend="legend", legendgroup="Gaussian",
            legendgrouptitle="Reference", showlegend_raw=False, showlegend_smoothed=True,
            y_upper=gaussian_upper, y_lower=gaussian_lower,
        )

        if _has_asimov:
            add_reference_pair_traces(
                fig, x=exposure_values, y_raw=raw_asimov, y_smoothed=smoothed_asimov,
                name="Asimov", raw_style=dict(color="rgb(31,119,180)", dash="dot", width=2),
                smoothed_style=dict(color="rgb(31,119,180)", dash="solid", width=3),
                row=1, col=1, legend="legend", legendgroup="Asimov",
                legendgrouptitle="Reference", showlegend_raw=False, showlegend_smoothed=True,
                y_upper=asimov_upper, y_lower=asimov_lower,
            )

        _add_sigma_hlines(fig, sigmas=[1, 2, 3], x_annotation=2)

        fig = format_coustom_plotly(
            fig, tickformat=(".1f", ".0e"), add_units=False,
            title=f"Day-Night Asymmetry - {args.folder} - {config}",
            matches=(None, None),
        )

        fig.update_yaxes(tickformat=".1f", dtick=1, range=[0, 4], title="Significance (σ)", row=1, col=1)
        fig.update_xaxes(range=[-1, args.exposure], zeroline=False, title="Exposure (year)", row=1, col=1)

        figure_name = f"{energy}_DayNight_Exposure"
        if args.nhits is not None or args.ophits is not None or args.adjcls is not None:
            figure_name += f"_NHits{nhits_value:.0f}_OpHits{ophits_value:.0f}_AdjCl{adjcl_value:.0f}"
        if args.threshold is not None:
            figure_name += f"_Threshold_{args.threshold:.0f}"

        save_figure(fig, save_path, config=config, name=name, subfolder=args.folder.lower(),
                    filename=figure_name, rm=args.rewrite, debug=args.plot)

        exposure_records.append({
            "Analysis": "DayNight", "Geometry": info["GEOMETRY"],
            "Config": config, "Name": name, "EnergyLabel": energy,
            "Variable": "Gaussian", "SpectrumType": "Raw", "Mode": "PerBin",
            "NHits": int(nhits_value), "OpHits": int(ophits_value), "AdjCl": int(adjcl_value),
            "Exposure": exposure_values.tolist(), "Significance": raw_gaussian.tolist(),
            "SignificanceError+": None, "SignificanceError-": None,
        })
        exposure_records.append({
            "Analysis": "DayNight", "Geometry": info["GEOMETRY"],
            "Config": config, "Name": name, "EnergyLabel": energy,
            "Variable": "Gaussian", "SpectrumType": "Smoothed", "Mode": "PerBin",
            "NHits": int(nhits_value), "OpHits": int(ophits_value), "AdjCl": int(adjcl_value),
            "Exposure": exposure_values.tolist(), "Significance": smoothed_gaussian.tolist(),
            "SignificanceError+": (gaussian_upper - smoothed_gaussian).tolist(),
            "SignificanceError-": (smoothed_gaussian - gaussian_lower).tolist(),
        })

        if _has_asimov:
            exposure_records.append({
                "Analysis": "DayNight", "Geometry": info["GEOMETRY"],
                "Config": config, "Name": name, "EnergyLabel": energy,
                "Variable": "Asimov", "SpectrumType": "Raw", "Mode": "PerBin",
                "NHits": int(nhits_value), "OpHits": int(ophits_value), "AdjCl": int(adjcl_value),
                "Exposure": exposure_values.tolist(), "Significance": raw_asimov.tolist(),
                "SignificanceError+": None, "SignificanceError-": None,
            })
            exposure_records.append({
                "Analysis": "DayNight", "Geometry": info["GEOMETRY"],
                "Config": config, "Name": name, "EnergyLabel": energy,
                "Variable": "Asimov", "SpectrumType": "Smoothed", "Mode": "PerBin",
                "NHits": int(nhits_value), "OpHits": int(ophits_value), "AdjCl": int(adjcl_value),
                "Exposure": exposure_values.tolist(), "Significance": smoothed_asimov.tolist(),
                "SignificanceError+": (asimov_upper - smoothed_asimov).tolist(),
                "SignificanceError-": (smoothed_asimov - asimov_lower).tolist(),
            })

    # ══════════════════════════════════════════════════════════════════════════
    # HEP — exposure mode
    # ══════════════════════════════════════════════════════════════════════════
    elif args.analysis == "HEP" and args.mode in ["exposure", "all"]:
        cuts = _get_selection_cuts(config, name, energy, args, "HEP")
        if cuts is None:
            rprint(f"[yellow][WARNING][/yellow] Missing best-cut selection for {config} {name} {energy}")
            continue
        nhits_value, ophits_value, adjcl_value = cuts
        detector_mass = get_full_detector_mass(config, info)

        sigmas_df = pd.read_pickle(
            f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/HEP/{args.folder.lower()}"
            f"/{config}/{name}/{config}_{name}_{energy}_HEP_Results.pkl"
        )

        required_base_columns = ["Config", "Name", "NHits", "OpHits", "AdjCl", "Exposure"]
        missing_base_columns = [c for c in required_base_columns if c not in sigmas_df.columns]
        if sigmas_df.empty or missing_base_columns:
            rprint(f"[yellow][WARNING][/yellow] Skipping exposure plot for {config} {name} {energy}")
            continue

        plot_sigmas = sigmas_df[
            (sigmas_df["Config"] == config) * (sigmas_df["Name"] == name)
            * (sigmas_df["NHits"] == int(nhits_value)) * (sigmas_df["OpHits"] == int(ophits_value))
            * (sigmas_df["AdjCl"] == int(adjcl_value))
        ].copy()

        if plot_sigmas.empty:
            rprint(f"[yellow][WARNING][/yellow] Not found for {config} {name} {energy}")
            continue

        fig = make_subplots(rows=1, cols=1, subplot_titles=(f"{energy}",))
        exposure_values = np.asarray(plot_sigmas["Exposure"].values[0], dtype=float)

        significance_peak = 0.0
        significance_plot_styles = {
            "Asimov": {"label": "asimov", "dash": "solid", "raw_dash": "dot"},
            "Gaussian": {"label": "gaussian", "dash": "dash", "raw_dash": "dashdot"},
            "ProfileLikelihood": {"label": "profile-likelihood", "dash": "solid", "raw_dash": "dot"},
        }

        for significance, style in significance_plot_styles.items():
            required_sig_columns = [significance, "Raw" + significance, significance + "+Error", significance + "-Error"]
            missing_sig_columns = [c for c in required_sig_columns if c not in plot_sigmas.columns]
            if missing_sig_columns:
                continue

            smoothed_sig = _safe_array(plot_sigmas[significance].values[0])
            raw_sig = _safe_array(plot_sigmas["Raw" + significance].values[0])
            sig_plus = _safe_array(plot_sigmas[significance + "+Error"].values[0])
            sig_minus = _safe_array(plot_sigmas[significance + "-Error"].values[0])

            # If --reference specified, plot only that metric; otherwise plot all available
            if args.reference and significance != args.reference:
                continue

            y_upper = sig_plus
            y_lower = sig_minus
            significance_peak = max(significance_peak, float(np.max(raw_sig)), float(np.max(smoothed_sig)), float(np.max(sig_plus)))

            add_reference_pair_traces(
                fig, x=exposure_values, y_raw=raw_sig, y_smoothed=smoothed_sig,
                name=style["label"], raw_style=dict(color="black", dash=style["raw_dash"], width=1),
                smoothed_style=dict(color="black", dash=style["dash"], width=2),
                row=1, col=1, legend="legend", legendgroup=significance,
                legendgrouptitle="Significance", showlegend_raw=False, showlegend_smoothed=True,
                y_upper=y_upper, y_lower=y_lower,
            )

        _add_sigma_hlines(fig, sigmas=[1, 2, 3, 4, 5], x_annotation=detector_mass * args.exposure * 0.1)

        fig = format_coustom_plotly(
            fig, tickformat=(".1f", ".0e"), add_units=False,
            title=f"HEP Discovery - {args.folder} - {config}",
            matches=(None, None),
        )

        fig.update_yaxes(tickformat=".1f", dtick=1, range=[0, max(1.0, 1.1 * significance_peak)] if args.zoom else [0, 6],
                         title="Significance (σ)", row=1, col=1)
        fig.update_xaxes(range=[-1, args.exposure], zeroline=False, title="Exposure (years)", row=1, col=1)

        figure_name = f"{energy}_HEP_Exposure_{args.reference}"
        if args.nhits is not None or args.ophits is not None or args.adjcls is not None:
            figure_name += f"_NHits{nhits_value:.0f}_OpHits{ophits_value:.0f}_AdjCl{adjcl_value:.0f}"
        if args.threshold is not None:
            figure_name += f"_Threshold_{args.threshold:.0f}"
        if args.pkl_label != "highest":
            figure_name += f"_{args.pkl_label}"

        save_figure(fig, save_path, config=config, name=name, subfolder=args.folder.lower(),
                    filename=figure_name, rm=args.rewrite, debug=args.plot)

        for spec_type, sig_arr in [("Raw", raw_sig), ("Smoothed", smoothed_sig)]:
            exposure_records.append({
                "Analysis": "HEP", "Geometry": info["GEOMETRY"],
                "Config": config, "Name": name, "EnergyLabel": energy,
                "Variable": args.reference, "SpectrumType": spec_type, "Mode": "NoRebin",
                "NHits": int(nhits_value), "OpHits": int(ophits_value), "AdjCl": int(adjcl_value),
                "Exposure": exposure_values.tolist(), "Significance": sig_arr.tolist(),
                "SignificanceError+": None, "SignificanceError-": None,
            })

    # ══════════════════════════════════════════════════════════════════════════
    # HEP — comparison mode
    # ══════════════════════════════════════════════════════════════════════════
    elif args.analysis == "HEP" and args.mode in ["comparison", "all"]:
        cuts = _get_selection_cuts(config, name, energy, args, "HEP")
        if cuts is None:
            continue
        nhits_value, ophits_value, adjcl_value = cuts

        exposure_file = Path(data_path) / config / name / args.folder.lower() / f"{config}_{name}_HEP_Exposure.pkl"
        if not exposure_file.exists():
            rprint(f"[yellow][WARNING][/yellow] Missing saved HEP exposure data for {config} {name}. Run exposure mode first.")
            continue

        exposure_df = pd.read_pickle(exposure_file)
        required_columns = ["Config", "Name", "EnergyLabel", "Variable", "SpectrumType", "Exposure", "Significance"]
        missing_columns = [c for c in required_columns if c not in exposure_df.columns]
        if missing_columns:
            rprint(f"[yellow][WARNING][/yellow] Missing columns {missing_columns}")
            continue

        exposure_rows = exposure_df.loc[
            (exposure_df["Config"] == config) & (exposure_df["Name"] == name)
            & (exposure_df["EnergyLabel"] == energy)
            & (exposure_df["Variable"].isin(["Asimov", "Gaussian", "ProfileLikelihood"]))
        ].copy()

        for column, value in [("NHits", nhits_value), ("OpHits", ophits_value), ("AdjCl", adjcl_value)]:
            if column in exposure_rows.columns:
                exposure_rows = exposure_rows.loc[exposure_rows[column] == int(value)].copy()

        if exposure_rows.empty:
            rprint(f"[yellow][WARNING][/yellow] Missing comparison inputs for {config} {name} {energy}")
            continue

        fig = make_subplots(rows=1, cols=1)
        significance_max = 0.0

        for variable in ["Asimov", "Gaussian", "ProfileLikelihood"]:
            smoothed_row = exposure_rows.loc[
                (exposure_rows["Variable"] == variable) & (exposure_rows["SpectrumType"] == "Smoothed")
            ]
            raw_row = exposure_rows.loc[
                (exposure_rows["Variable"] == variable) & (exposure_rows["SpectrumType"] == "Raw")
            ]
            if smoothed_row.empty and raw_row.empty:
                continue

            base_row = smoothed_row if not smoothed_row.empty else raw_row
            xvals = np.asarray(base_row["Exposure"].values[0], dtype=float)
            yvals = _safe_array(smoothed_row["Significance"].values[0] if not smoothed_row.empty else np.zeros_like(xvals))
            yraw = _safe_array(raw_row["Significance"].values[0] if not raw_row.empty else np.zeros_like(xvals))

            y_upper = None
            y_lower = None
            if (not smoothed_row.empty and "SignificanceError+" in smoothed_row.columns and
                "SignificanceError-" in smoothed_row.columns and
                smoothed_row["SignificanceError+"].values[0] is not None and
                smoothed_row["SignificanceError-"].values[0] is not None):
                err_plus = _safe_array(smoothed_row["SignificanceError+"].values[0])
                err_minus = _safe_array(smoothed_row["SignificanceError-"].values[0])
                if err_plus.size == yvals.size and err_minus.size == yvals.size:
                    y_upper = yvals + err_plus
                    y_lower = yvals - err_minus

            significance_max = max(significance_max, float(np.max(yvals)), float(np.max(yraw)))
            if y_upper is not None:
                significance_max = max(significance_max, float(np.max(y_upper)))

            style_raw = _COMPARISON_STYLES[(variable, "Raw")]
            style_smoothed = _COMPARISON_STYLES[(variable, "Smoothed")]

            add_reference_pair_traces(
                fig, x=xvals, y_raw=yraw, y_smoothed=yvals, name=variable,
                raw_style=style_raw, smoothed_style=style_smoothed,
                row=1, col=1, legend="legend", legendgroup="reference",
                legendgrouptitle="Reference", showlegend_raw=False, showlegend_smoothed=True,
                y_upper=y_upper, y_lower=y_lower,
            )

        fig = format_coustom_plotly(
            fig, tickformat=(".1f", ".0e"), add_units=False,
            legend_title=energy, title=f"HEP Discovery Comparison - {args.folder} - {config}",
            matches=(None, None),
        )

        fig.update_yaxes(tickformat=".1f", dtick=1,
                         range=[0, max(1.0, 1.1 * significance_max)] if args.zoom else [0, 6],
                         title="Significance (sigma)", row=1, col=1)
        fig.update_xaxes(range=[-1, args.exposure], zeroline=False, title="Exposure (kT·year)", row=1, col=1)

        figure_name = f"{energy}_HEP_Exposure_Comparison"
        if args.threshold is not None:
            figure_name += f"_Threshold_{args.threshold:.0f}"

        save_figure(fig, save_path, config=config, name=name, subfolder=args.folder.lower(),
                    filename=figure_name, rm=args.rewrite, debug=args.plot)

    # ══════════════════════════════════════════════════════════════════════════
    # HEP — rebin mode
    # ══════════════════════════════════════════════════════════════════════════
    elif args.analysis == "HEP" and args.mode in ["rebin", "all"]:
        cuts = _get_selection_cuts(config, name, energy, args, "HEP")
        if cuts is None:
            continue
        nhits_value, ophits_value, adjcl_value = cuts

        sigmas_df = pd.read_pickle(
            f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/HEP/{args.folder.lower()}"
            f"/{config}/{name}/{config}_{name}_{energy}_HEP_Results.pkl"
        )

        required_sigma_columns = ["Config", "Name", "NHits", "OpHits", "AdjCl", "Exposure"]
        missing_sigma_columns = [c for c in required_sigma_columns if c not in sigmas_df.columns]
        if sigmas_df.empty or missing_sigma_columns:
            rprint(f"[yellow][WARNING][/yellow] Invalid HEP results for {config} {name} {energy}")
            continue

        sigma_rows = sigmas_df.loc[
            (sigmas_df["Config"] == config) * (sigmas_df["Name"] == name)
            * (sigmas_df["NHits"] == int(nhits_value)) * (sigmas_df["OpHits"] == int(ophits_value))
            * (sigmas_df["AdjCl"] == int(adjcl_value))
        ].copy()

        if sigma_rows.empty:
            rprint(f"[yellow][WARNING][/yellow] Missing result row for {config} {name} {energy}")
            continue

        sigma_row = sigma_rows.iloc[0]
        exposure_grid = _safe_array(sigma_row["Exposure"])

        for significance_type in ["Asimov", "Gaussian"]:
            curve_keys = {
                ("Raw", "NoRebin"): f"Raw{significance_type}NoRebin",
                ("Raw", "AdaptiveRebin"): f"Raw{significance_type}",
                ("Smoothed", "NoRebin"): f"{significance_type}NoRebin",
                ("Smoothed", "AdaptiveRebin"): f"{significance_type}",
            }
            missing = [k for k in curve_keys.values() if k not in sigma_row.index]
            if missing:
                continue

            fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], vertical_spacing=0,
                               subplot_titles=(f"{significance_type} vs exposure ({energy})", ""))

            significance_peak = 0.0
            for spectrum_label, rebin_mode in [("Raw", "NoRebin"), ("Raw", "AdaptiveRebin"),
                                               ("Smoothed", "NoRebin"), ("Smoothed", "AdaptiveRebin")]:
                curve_key = curve_keys[(spectrum_label, rebin_mode)]
                y_values = _safe_array(sigma_row[curve_key])
                significance_peak = max(significance_peak, float(np.max(y_values)))
                style = _REBIN_STYLE_MAP[(spectrum_label, rebin_mode)]
                label = f"{spectrum_label} {rebin_mode}"

                fig.add_trace(go.Scatter(
                    x=exposure_grid, y=y_values, mode="lines", name=label,
                    line=dict(color=style["color"], dash=style["dash"], width=style["width"]),
                    line_shape="linear",
                ), row=1, col=1)

            fig = format_coustom_plotly(fig, title=f"Rebin Comparison - {args.folder} - {config}",
                                       add_units=False, figsize=(800, 600), matches=("x", None), add_watermark=False)
            fig.update_xaxes(title="", showticklabels=False, row=1, col=1)
            fig.update_xaxes(title="Exposure (kT·year)", row=2, col=1)
            fig.update_yaxes(title="Significance (σ)",
                           range=[0, max(1.0, 1.1 * significance_peak)] if args.zoom else [0, 6],
                           row=1, col=1)
            fig.update_yaxes(title="Grouped bins", row=2, col=1)

            figure_name = f"{energy}_HEP_{significance_type}_AdaptiveRebin_Comparison"
            if args.threshold is not None:
                figure_name += f"_Threshold_{args.threshold:.0f}"

            save_figure(fig, save_path, config=config, name=name, subfolder=args.folder.lower(),
                       filename=figure_name, rm=args.rewrite, debug=args.plot)

    # ══════════════════════════════════════════════════════════════════════════
    # HEP — reference mode
    # ══════════════════════════════════════════════════════════════════════════
    elif args.analysis == "HEP" and args.mode in ["reference", "all"]:
        cuts = _get_selection_cuts(config, name, energy, args, "HEP")
        if cuts is None:
            continue
        nhits_value, ophits_value, adjcl_value = cuts

        significance_file = Path(data_path) / config / name / args.folder.lower() / f"{config}_{name}_HEP_Significance.pkl"
        exposure_file = Path(data_path) / config / name / args.folder.lower() / f"{config}_{name}_HEP_Exposure.pkl"

        if not significance_file.exists() or not exposure_file.exists():
            rprint(f"[yellow][WARNING][/yellow] Missing HEP plot data for {config} {name}. Run plotting macros first.")
            continue

        significance_df = pd.read_pickle(significance_file)
        exposure_df = pd.read_pickle(exposure_file)

        if "EnergyLabel" not in significance_df.columns or "EnergyLabel" not in exposure_df.columns:
            rprint(f"[yellow][WARNING][/yellow] Missing EnergyLabel in saved HEP plot data")
            continue

        significance_rows = significance_df.loc[
            (significance_df["Config"] == config) & (significance_df["Name"] == name)
            & (significance_df["EnergyLabel"] == energy) & (significance_df["Variable"].isin(["Asimov", "Gaussian", "ProfileLikelihood"]))
        ].copy()
        exposure_rows = exposure_df.loc[
            (exposure_df["Config"] == config) & (exposure_df["Name"] == name)
            & (exposure_df["EnergyLabel"] == energy) & (exposure_df["Variable"].isin(["Asimov", "Gaussian", "ProfileLikelihood"]))
        ].copy()

        for column, value in [("NHits", nhits_value), ("OpHits", ophits_value), ("AdjCl", adjcl_value)]:
            if column in significance_rows.columns:
                significance_rows = significance_rows.loc[significance_rows[column] == int(value)].copy()
            if column in exposure_rows.columns:
                exposure_rows = exposure_rows.loc[exposure_rows[column] == int(value)].copy()

        if significance_rows.empty or exposure_rows.empty:
            rprint(f"[yellow][WARNING][/yellow] Missing comparison inputs for {config} {name} {energy}")
            continue

        # Significance comparison figure
        fig_sig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], vertical_spacing=0,
                               subplot_titles=("", ""))

        significance_max = 0.0
        for variable in ["Asimov", "Gaussian", "ProfileLikelihood"]:
            for spectrum_type in ["Raw", "Smoothed"]:
                row = significance_rows.loc[
                    (significance_rows["Variable"] == variable) & (significance_rows["SpectrumType"] == spectrum_type)
                ]
                if row.empty:
                    continue

                xvals = _safe_array(row["Energy"].values[0])
                yvals = _safe_array(row["Significance"].values[0])
                significance_max = max(significance_max, float(np.max(yvals)))

                style = _COMPARISON_STYLES[(variable, spectrum_type)]
                fig_sig.add_trace(go.Scatter(
                    x=xvals, y=yvals, mode="lines", name=variable,
                    line=dict(color=style["color"], dash=style["dash"], width=style["width"]),
                    line_shape="hvh", legend="legend2", legendgroup="reference",
                    legendgrouptitle="Reference", showlegend=(spectrum_type == "Smoothed"),
                ), row=2, col=1)

        fig_sig = format_coustom_plotly(fig_sig, figsize=(800, 600), tickformat=(".1f", ".0e"),
                                       add_units=False, title=f"HEP Significance Comparison - {args.folder} - {config}",
                                       matches=("x", None), add_watermark=False)
        if args.threshold is not None:
            fig_sig.add_vline(x=args.threshold, line_dash="dash", line_color="grey",
                             annotation=dict(text="Threshold", showarrow=False),
                             annotation_position="bottom right")

        fig_sig.update_yaxes(tickformat=".0f", range=[0, max(1.0, 1.1 * significance_max)] if args.zoom else [0, 6],
                            title="Significance (σ)", row=2, col=1)
        fig_sig.update_xaxes(range=[8, 26], showticklabels=False, row=1, col=1)
        fig_sig.update_xaxes(range=[8, 26], title="Reconstructed Neutrino Energy (MeV)", row=2, col=1)

        figure_name = f"{energy}_HEP_Significance_Comparison"
        if args.threshold is not None:
            figure_name += f"_Threshold_{args.threshold:.0f}"

        save_figure(fig_sig, save_path, config=config, name=name, subfolder=args.folder.lower(),
                   filename=figure_name, rm=args.rewrite, debug=args.plot)

        # Exposure comparison figure
        fig_exp = make_subplots(rows=1, cols=1)
        exposure_max = 0.0

        for variable in ["Asimov", "Gaussian"]:
            row = exposure_rows.loc[exposure_rows["Variable"] == variable]
            if row.empty:
                continue

            xvals = _safe_array(row["Exposure"].values[0])
            y_raw = _safe_array(row["RawSignificance"].values[0] if "RawSignificance" in row.columns else row["Significance"].values[0])
            y_smooth = _safe_array(row["Significance"].values[0])
            exposure_max = max(exposure_max, float(np.max(y_smooth)))

            raw_style = _COMPARISON_STYLES[(variable, "Raw")]
            smooth_style = _COMPARISON_STYLES[(variable, "Smoothed")]

            add_reference_pair_traces(fig_exp, x=xvals, y_raw=y_raw, y_smoothed=y_smooth, name=variable,
                                     raw_style=raw_style, smoothed_style=smooth_style,
                                     legend="legend", legendgroup="reference", legendgrouptitle="Reference",
                                     showlegend_raw=False, showlegend_smoothed=True, line_shape="linear")

        fig_exp = format_coustom_plotly(fig_exp, tickformat=(".1f", ".1f"), add_units=False,
                                       title="Selected Sample for Solar Neutrino HEP Exposure Comparison",
                                       matches=(None, None))
        fig_exp.update_yaxes(tickformat=".1f", dtick=1,
                            range=[0, max(1.0, 1.1 * exposure_max)] if args.zoom else [0, 6],
                            title="Significance (σ)")
        fig_exp.update_xaxes(range=[-1, args.exposure], zeroline=False, title="Exposure (kT·year)")

        figure_name = f"{energy}_HEP_Exposure_Comparison"
        if args.threshold is not None:
            figure_name += f"_Threshold_{args.threshold:.0f}"

        save_figure(fig_exp, save_path, config=config, name=name, subfolder=args.folder.lower(),
                   filename=figure_name, rm=args.rewrite, debug=args.plot)

    # ══════════════════════════════════════════════════════════════════════════
    # Sensitivity — 4 single-panel chi2 projections
    # ══════════════════════════════════════════════════════════════════════════
    elif args.analysis == "Sensitivity":
        cuts = _get_selection_cuts(config, name, energy, args, "SENSITIVITY")
        if cuts is None:
            cuts = (4, 10, 4)
        nhits, ophits, adjcl = cuts

        # Path from sensitivity/06_significance.py output
        sig_path = f"{info['PATH']}/SENSITIVITY/{config}/{name}/{args.folder.lower()}/{energy}"
        suffix = (
            f"signal_{100*args.signal_uncertainty:.0f}%_and_background_{100*args.background_uncertainty:.0f}%"
            if args.background else f"signal_{100*args.signal_uncertainty:.0f}%_only"
        )
        profile_name = args.nuisance_profile or analysis_info.get("DEFAULT_NUISANCE_PROFILE", "full")
        prefix = f"{sig_path}/results/{profile_name}/{suffix}/{name}_{energy}_NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}"

        try:
            solar_sin12_df = pd.read_pickle(f"{prefix}_solar_sin12_df.pkl").astype(float)
            solar_sin13_df = pd.read_pickle(f"{prefix}_solar_sin13_df.pkl").astype(float)
            react_sin12_df = pd.read_pickle(f"{prefix}_react_sin12_df.pkl").astype(float)
            react_sin13_df = pd.read_pickle(f"{prefix}_react_sin13_df.pkl").astype(float)
        except FileNotFoundError as e:
            rprint(f"[yellow][WARNING][/yellow] Missing chi2 dataframes for {config} {name} {energy}: {e}")
            continue

        for df in [solar_sin12_df, solar_sin13_df, react_sin12_df, react_sin13_df]:
            df.replace(0.0, np.nan, inplace=True)

        global_min = np.nanmin([np.nanmin(solar_sin12_df.values), np.nanmin(solar_sin13_df.values)])
        react_global_min = np.nanmin([np.nanmin(react_sin12_df.values), np.nanmin(react_sin13_df.values)])

        dm2_vals = solar_sin12_df.index.astype(float).values
        sin12_vals = solar_sin12_df.columns.astype(float).values
        sin13_vals = solar_sin13_df.columns.astype(float).values

        dm2_vals_react = react_sin12_df.index.astype(float).values
        sin12_vals_react = react_sin12_df.columns.astype(float).values
        sin13_vals_react = react_sin13_df.columns.astype(float).values

        _w = args.smooth_window

        _sort_s = np.argsort(dm2_vals)
        _sort_r = np.argsort(dm2_vals_react)
        dm2_vals = dm2_vals[_sort_s]
        dm2_vals_react = dm2_vals_react[_sort_r]
        dchi2_dm2_solar = _smooth_sg(np.nanmin(solar_sin12_df.values, axis=1)[_sort_s] - global_min, _w)
        dchi2_dm2_react = _smooth_sg(np.nanmin(react_sin12_df.values, axis=1)[_sort_r] - react_global_min, _w)

        _sort_s12 = np.argsort(sin12_vals)
        _sort_r12 = np.argsort(sin12_vals_react)
        sin12_vals = sin12_vals[_sort_s12]
        sin12_vals_react = sin12_vals_react[_sort_r12]
        dchi2_sin12_solar = _smooth_sg(np.nanmin(solar_sin12_df.values, axis=0)[_sort_s12] - global_min, _w)
        dchi2_sin12_react = _smooth_sg(np.nanmin(react_sin12_df.values, axis=0)[_sort_r12] - react_global_min, _w)

        _sort_s13 = np.argsort(sin13_vals)
        _sort_r13 = np.argsort(sin13_vals_react)
        sin13_vals = sin13_vals[_sort_s13]
        sin13_vals_react = sin13_vals_react[_sort_r13]
        dchi2_sin13_solar = _smooth_sg(np.nanmin(solar_sin13_df.values, axis=1)[_sort_s13] - global_min, _w)
        dchi2_sin13_react = _smooth_sg(np.nanmin(react_sin13_df.values, axis=1)[_sort_r13] - react_global_min, _w)

        compare_tag = "_NuFit61" if args.compare else ""
        _cut_tag = f"NHits{nhits}_AdjCl{adjcl}_OpHits{ophits}"
        _base_tag = f"{config}_{name}_{args.folder}_{energy}_{_cut_tag}"

        # ── Figure 1: Δχ²(Δm²₂₁) solar ──
        fig_dm2_solar = make_subplots(rows=1, cols=1)
        if _has_coverage(dm2_vals, dchi2_dm2_solar, "dm2", fig_dm2_solar, 1, 1):
            fig_dm2_solar.add_trace(go.Scatter(
                x=dm2_vals * 1e5, y=dchi2_dm2_solar, mode="lines",
                name="DUNE (solar ref.)", line=dict(color=DUNE_COLOR_SOLAR, width=2.5),
                legendgroup="dune", showlegend=True,
            ), row=1, col=1)

        if args.compare:
            for ref_label, ref in _NUFIT61_DM2_SOLAR.items():
                fig_dm2_solar.add_trace(go.Scatter(
                    x=ref["dm2"], y=ref["chi2"], mode="lines",
                    name=f"NuFit 6.1 {ref_label}", line=dict(color=ref["color"], width=1.5, dash=ref["dash"]),
                    legendgroup=f"nf_{ref_label}", showlegend=True,
                ), row=1, col=1)

        _add_projection_sigma_lines(fig_dm2_solar, 1, 1)
        _add_bf_vline(fig_dm2_solar, analysis_info.get("SOLAR_DM2", 5.9e-5) * 1e5, 1)
        fig_dm2_solar.update_xaxes(title_text="Δm²<sub>21</sub> [10⁻⁵ eV²]", row=1, col=1, range=[2, 14])
        fig_dm2_solar.update_yaxes(title_text="Δχ²", row=1, col=1, range=[0, 12])
        fig_dm2_solar = format_coustom_plotly(fig_dm2_solar, title=f"Δm²₂₁ (Solar) — {_base_tag}", add_watermark=True)

        save_figure(fig_dm2_solar, save_path, config=config, name=name, subfolder=args.folder.lower(),
                    filename=f"{_base_tag}_dm2_solar_projection{compare_tag}", rm=args.rewrite, debug=args.plot)

        # ── Figure 2: Δχ²(Δm²₂₁) reactor ──
        fig_dm2_react = make_subplots(rows=1, cols=1)
        if _has_coverage(dm2_vals_react, dchi2_dm2_react, "dm2", fig_dm2_react, 1, 1):
            fig_dm2_react.add_trace(go.Scatter(
                x=dm2_vals_react * 1e5, y=dchi2_dm2_react, mode="lines",
                name="DUNE (reactor ref.)", line=dict(color=DUNE_COLOR_REACT, width=2.5),
                legendgroup="dune", showlegend=True,
            ), row=1, col=1)

        if args.compare:
            for ref_label, ref in _NUFIT61_DM2_REACTOR.items():
                fig_dm2_react.add_trace(go.Scatter(
                    x=ref["dm2"], y=ref["chi2"], mode="lines",
                    name=f"NuFit 6.1 {ref_label}", line=dict(color=ref["color"], width=1.5, dash=ref["dash"]),
                    legendgroup=f"nf_{ref_label}", showlegend=True,
                ), row=1, col=1)

        _add_projection_sigma_lines(fig_dm2_react, 1, 1)
        _add_bf_vline(fig_dm2_react, analysis_info.get("REACT_DM2", 7.53e-5) * 1e5, 1)
        fig_dm2_react.update_xaxes(title_text="Δm²<sub>21</sub> [10⁻⁵ eV²]", row=1, col=1, range=[2, 14])
        fig_dm2_react.update_yaxes(title_text="Δχ²", row=1, col=1, range=[0, 12])
        fig_dm2_react = format_coustom_plotly(fig_dm2_react, title=f"Δm²₂₁ (Reactor) — {_base_tag}", add_watermark=True)

        save_figure(fig_dm2_react, save_path, config=config, name=name, subfolder=args.folder.lower(),
                    filename=f"{_base_tag}_dm2_reactor_projection{compare_tag}", rm=args.rewrite, debug=args.plot)

        # ── Figure 3: Δχ²(sin²θ₁₂) ──
        fig_sin12 = make_subplots(rows=1, cols=1)
        if _has_coverage(sin12_vals, dchi2_sin12_solar, "sin12", fig_sin12, 1, 1):
            fig_sin12.add_trace(go.Scatter(
                x=sin12_vals, y=dchi2_sin12_solar, mode="lines",
                name="DUNE (solar ref.)", line=dict(color=DUNE_COLOR_SOLAR, width=2.5),
                legendgroup="dune_s", showlegend=True,
            ), row=1, col=1)
            fig_sin12.add_trace(go.Scatter(
                x=sin12_vals_react, y=dchi2_sin12_react, mode="lines",
                name="DUNE (reactor ref.)", line=dict(color=DUNE_COLOR_REACT, width=2.5),
                legendgroup="dune_r", showlegend=True,
            ), row=1, col=1)

        if args.compare:
            x_s12 = np.linspace(0.15, 0.55, 200)
            for ref_label, ref in _NUFIT61_SIN12_PROFILES.items():
                if ref_label not in ("KamLAND", "Solar+KamL"):
                    y = _gaussian_chi2(x_s12, ref["bf"], ref["sigma_lo"], ref["sigma_hi"])
                    fig_sin12.add_trace(go.Scatter(
                        x=x_s12, y=y, mode="lines",
                        name=f"NuFit 6.1 {ref_label}", line=dict(color=ref["color"], width=1.5, dash=ref["dash"]),
                        legendgroup=f"nf12_{ref_label}", showlegend=True,
                    ), row=1, col=1)

        _add_projection_sigma_lines(fig_sin12, 1, 1)
        _add_bf_vline(fig_sin12, analysis_info.get("SIN12", 0.303), 1)
        fig_sin12.update_xaxes(title_text="sin²θ<sub>12</sub>", row=1, col=1, range=[0.15, 0.55])
        fig_sin12.update_yaxes(title_text="Δχ²", row=1, col=1, range=[0, 12])
        fig_sin12 = format_coustom_plotly(fig_sin12, title=f"sin²θ₁₂ — {_base_tag}", add_watermark=True)

        save_figure(fig_sin12, save_path, config=config, name=name, subfolder=args.folder.lower(),
                    filename=f"{_base_tag}_sin12_projection{compare_tag}", rm=args.rewrite, debug=args.plot)

        # ── Figure 4: Δχ²(sin²θ₁₃) ──
        fig_sin13 = make_subplots(rows=1, cols=1)
        if _has_coverage(sin13_vals, dchi2_sin13_solar, "sin13", fig_sin13, 1, 1):
            fig_sin13.add_trace(go.Scatter(
                x=sin13_vals, y=dchi2_sin13_solar, mode="lines",
                name="DUNE (solar ref.)", line=dict(color=DUNE_COLOR_SOLAR, width=2.5),
                legendgroup="dune_s", showlegend=True,
            ), row=1, col=1)
            fig_sin13.add_trace(go.Scatter(
                x=sin13_vals_react, y=dchi2_sin13_react, mode="lines",
                name="DUNE (reactor ref.)", line=dict(color=DUNE_COLOR_REACT, width=2.5),
                legendgroup="dune_r", showlegend=True,
            ), row=1, col=1)

        if args.compare:
            x_s13 = np.linspace(0.010, 0.040, 400)
            for ref_label, ref in _NUFIT61_SIN13_PROFILES.items():
                y = _gaussian_chi2(x_s13, ref["bf"], ref["sigma_lo"], ref["sigma_hi"])
                fig_sin13.add_trace(go.Scatter(
                    x=x_s13, y=y, mode="lines",
                    name=f"NuFit 6.1 {ref_label}", line=dict(color=ref["color"], width=1.5, dash=ref["dash"]),
                    legendgroup=f"nf13_{ref_label}", showlegend=True,
                ), row=1, col=1)

        _add_projection_sigma_lines(fig_sin13, 1, 1)
        _add_bf_vline(fig_sin13, analysis_info.get("SIN13", 0.0222), 1)
        fig_sin13.update_xaxes(title_text="sin²θ<sub>13</sub>", row=1, col=1, range=[0.010, 0.040])
        fig_sin13.update_yaxes(title_text="Δχ²", row=1, col=1, range=[0, 12])
        fig_sin13 = format_coustom_plotly(fig_sin13, title=f"sin²θ₁₃ — {_base_tag}", add_watermark=True)

        save_figure(fig_sin13, save_path, config=config, name=name, subfolder=args.folder.lower(),
                    filename=f"{_base_tag}_sin13_projection{compare_tag}", rm=args.rewrite, debug=args.plot)

# ── SINGLE SAVE ────────────────────────────────────────────────────────────────

if exposure_records and args.analysis != "Sensitivity":
    _df = pd.DataFrame(exposure_records)
    _filename = f"{args.analysis}_Exposure"
    save_df(
        _df, data_path,
        config=args.config[0], name=args.name[0],
        subfolder=args.folder.lower(),
        filename=_filename,
        rm=args.rewrite, debug=args.debug,
    )
