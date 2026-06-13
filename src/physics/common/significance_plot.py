import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *

# ── shared helpers ─────────────────────────────────────────────────────────────

def _safe_array(values):
    return np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def _density(values, widths):
    values = _safe_array(values)
    widths = _safe_array(widths)
    return np.divide(values, widths, out=np.zeros_like(values, dtype=float), where=widths > 0)


def _step_fill_coordinates(centers, widths, values):
    centers = _safe_array(centers)
    widths = _safe_array(widths)
    values = _safe_array(values)
    if centers.size == 0 or widths.size == 0 or values.size == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    left_edges = centers - 0.5 * widths
    right_edges = centers + 0.5 * widths
    x_points, y_points = [], []
    for left, right, value in zip(left_edges, right_edges, values):
        x_points.extend([float(left), float(right)])
        y_points.extend([float(value), float(value)])
    return np.asarray(x_points, dtype=float), np.asarray(y_points, dtype=float)


def _group_tail_values(tail_energy, tail_values, grouped_centers, grouped_widths):
    grouped = []
    tail_energy = _safe_array(tail_energy)
    tail_values = _safe_array(tail_values)
    if tail_energy.size == 0:
        return np.zeros(len(grouped_centers), dtype=float)
    for center, width in zip(grouped_centers, grouped_widths):
        half_width = 0.5 * float(width)
        low = float(center) - half_width - 1e-9
        high = float(center) + half_width + 1e-9
        mask = (tail_energy >= low) & (tail_energy <= high)
        if not np.any(mask):
            nearest = int(np.argmin(np.abs(tail_energy - float(center))))
            grouped.append(float(tail_values[nearest]))
        else:
            grouped.append(float(np.sum(tail_values[mask])))
    return np.asarray(grouped, dtype=float)


def _process_component_df(comp_df, component, smoothing_config, scale=1.0, threshold_idx=0):
    counts = _safe_array(comp_df["Counts"].values[0])
    errors = _safe_array(comp_df["Error"].values[0])
    mc_counts = (
        _safe_array(comp_df["MCCounts"].values[0])
        if "MCCounts" in comp_df.columns else np.zeros_like(counts)
    )
    cfg = get_component_smoothing_config(smoothing_config, component)
    smoothed_counts = smooth_threshold_slice(counts, threshold_idx, cfg)
    smoothed_errors = smooth_histogram_errors(errors, cfg, counts=counts, mc_counts=mc_counts)
    return {
        "counts": counts, "errors": errors,
        "smoothed_counts": smoothed_counts, "smoothed_errors": smoothed_errors,
        "raw_y": scale * counts,
        "smoothed_y": scale * smoothed_counts,
        "err_y": scale * smoothed_errors,
    }


def _add_component_traces(fig, render_list, energy_axis, stacked, row=1, col=1):
    for rd in render_list:
        if not stacked:
            fig.add_trace(
                go.Scatter(
                    x=energy_axis, y=rd["raw_y"], name=rd["component_label"], mode="lines",
                    showlegend=False, line=dict(color=rd["color"], width=2, dash="dot"),
                    line_shape="hvh", opacity=rd.get("opacity", 1.0),
                    legend="legend", legendgroup=rd["legend_group"],
                    legendgrouptitle=dict(text=rd["legend_group_title"]),
                ),
                row=row, col=col,
            )
        if stacked:
            fig.add_trace(
                go.Scatter(
                    x=energy_axis, y=rd["smoothed_y"], name=rd["component_label"],
                    mode="lines", stackgroup="one", line_shape="hvh",
                    line=dict(color=rd["color"], width=2),
                    legend="legend", legendgroup=rd["legend_group"],
                    legendgrouptitle=dict(text=rd["legend_group_title"]), showlegend=True,
                ),
                row=row, col=col,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=energy_axis, y=rd["smoothed_y"], name=rd["component_label"],
                    mode="lines", error_y=dict(type="data", array=rd["err_y"]),
                    line_shape="hvh", line=dict(color=rd["color"], width=3),
                    legend="legend", legendgroup=rd["legend_group"],
                    legendgrouptitle=dict(text=rd["legend_group_title"]), showlegend=True,
                ),
                row=row, col=col,
            )


def _update_upper_panel_yaxis(fig, stacked, positive_count_values, title, zoom, log_default_range, **extra):
    if stacked:
        fig.update_yaxes(title=title, row=1, col=1, **extra)
    else:
        y_range = (
            [
                np.floor(np.log10(max(min(positive_count_values), 1e-6))),
                np.ceil(np.log10(max(positive_count_values, 1e10))),
            ]
            if zoom and positive_count_values else log_default_range
        )
        fig.update_yaxes(type="log", range=y_range, title=title, row=1, col=1, **extra)


def _add_rebinned_overlay(fig, grouped_energy, grouped_width, grouped_background_counts, grouped_signal_counts, signal_label="Signal Rebin"):
    for trace_name, trace_values, fill_color, line_color in [
        ("Background Rebin", grouped_background_counts, "rgba(0,0,0,0.18)", "black"),
        (signal_label, grouped_signal_counts, "rgba(204,80,62,0.28)", "rgb(204,80,62)"),
    ]:
        x_points, y_points = _step_fill_coordinates(grouped_energy, grouped_width, trace_values)
        fig.add_trace(
            go.Scatter(
                x=x_points, y=y_points, mode="lines", name=trace_name,
                line=dict(color=line_color, width=1.5), fill="tozeroy", fillcolor=fill_color,
                legend="legend", legendgroup="adaptive_rebin",
                legendgrouptitle=dict(text="Adaptive Rebin"), showlegend=True,
            ),
            row=1, col=1,
        )


_STD_LEGEND = dict(
    legend=dict(font=dict(size=12), x=0.75, y=0.99, xanchor="left", yanchor="top", bgcolor="rgba(255,255,255,0.7)"),
    legend2=dict(font=dict(size=12), x=0.01, y=0.3, xanchor="left", yanchor="top", bgcolor="rgba(255,255,255,0.7)"),
    legend3=dict(font=dict(size=12), x=0.75, y=0.24, xanchor="left", yanchor="top", bgcolor="rgba(255,255,255,0.7)"),
)


def _render_and_save(
    render_list, energy_axis, add_lower_fn,
    subplot_title, fig_title, upper_y_title, lower_y_title,
    x_range, figure_name, positive_count_values, log_default_range,
    save_path, config, save_name, folder, rewrite, debug_plot,
    stacked_range, zoom, threshold,
    secondary_y=False, pre_lower_fn=None,
    upper_y_extra=None, lower_y_extra=None, lower_y_default_range=None,
    legend_layout=None, x_title="Reconstructed Energy (MeV)",
    add_upper_fn=None,
):
    specs = [[{}], [{"secondary_y": True}]] if secondary_y else None
    mkw = dict(rows=2, cols=1, row_heights=[0.7, 0.3], vertical_spacing=0.0,
               subplot_titles=(subplot_title, ""))
    if specs:
        mkw["specs"] = specs
    for _stacked in stacked_range:
        fig = make_subplots(**mkw)
        if add_upper_fn is not None:
            add_upper_fn(fig, _stacked)
        else:
            _add_component_traces(fig, render_list, energy_axis, _stacked)
        if pre_lower_fn:
            pre_lower_fn(fig)
        lower_panel_max = add_lower_fn(fig)
        fig = format_coustom_plotly(
            fig, tickformat=(".1f", ".0e"), add_units=False,
            title=fig_title, matches=("x", None), figsize=(800, 600), add_watermark=False,
        )
        fig.update_layout(**(legend_layout or _STD_LEGEND))
        _update_upper_panel_yaxis(
            fig, _stacked, positive_count_values, upper_y_title, zoom, log_default_range,
            **(upper_y_extra or {}),
        )
        _lower_range = (
            lower_y_default_range if (lower_y_default_range and not zoom)
            else [0, max(1.0, 1.1 * lower_panel_max)]
        )
        fig.update_yaxes(title=lower_y_title, range=_lower_range, row=2, col=1, **(lower_y_extra or {}))
        if secondary_y:
            fig.update_yaxes(title="Purity", range=[0, 1], row=2, col=1, secondary_y=True)
        if x_range:
            fig.update_xaxes(showticklabels=False, range=x_range, row=1, col=1)
            fig.update_xaxes(title=x_title, range=x_range, row=2, col=1)
        else:
            fig.update_xaxes(showticklabels=False, row=1, col=1)
            fig.update_xaxes(title=x_title, row=2, col=1)
        if threshold is not None:
            fig.add_vline(
                x=float(threshold), line_dash="dash", line_color="grey",
                annotation=dict(text="Threshold", showarrow=False),
                annotation_position="bottom right",
            )
        save_figure(
            fig, save_path, config=config, name=save_name, subfolder=folder,
            filename=figure_name + ("_Stacked" if _stacked else ""),
            rm=rewrite, debug=debug_plot,
        )


# ── fiducial helpers ───────────────────────────────────────────────────────────

_FIDUCIAL_COMPONENT_COLORS = {
    "gamma": "black",
    "neutron": "rgb(15,133,84)",
    "radiological": "rgb(120, 94, 240)",
    "8b": "rgb(225,124,5)",
    "hep": "rgb(204,80,62)",
}


def combine_components(df, components):
    component_df = df.loc[df["Component"].isin(components)].copy()
    if component_df.empty:
        return pd.DataFrame(columns=["Energy", "FiducializedX", "FiducializedY", "FiducializedZ", "MCCounts", "Counts", "Error+", "Error-"])
    return (
        component_df.groupby(["Energy", "FiducializedX", "FiducializedY", "FiducializedZ"])
        .agg({
            "MCCounts": "sum",
            "Counts": "sum",
            "Error+": lambda x: float(np.sqrt(np.sum(np.square(np.asarray(x, dtype=float))))),
            "Error-": lambda x: float(np.sqrt(np.sum(np.square(np.asarray(x, dtype=float))))),
        })
        .reset_index()
    )


def _aggregate_significance(significance, mode):
    values = np.nan_to_num(np.asarray(significance, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.sum(values) if str(mode).lower() == "sum" else np.sqrt(np.sum(np.square(values))))


def _build_fiducial_significance(merged_df, analysis_config, smoothing_config, exposure):
    if merged_df.empty:
        return None
    ordered = merged_df.sort_values("Energy").copy()
    energy = ordered["Energy"].to_numpy(dtype=float)
    sig_c = np.nan_to_num(ordered["CountsSignal"].to_numpy(dtype=float), nan=0.0)
    bkg_c = np.nan_to_num(ordered["CountsBackground"].to_numpy(dtype=float), nan=0.0)
    sig_e = np.nan_to_num(ordered["Error+Signal"].to_numpy(dtype=float), nan=0.0)
    bkg_e = np.nan_to_num(ordered["Error+Background"].to_numpy(dtype=float), nan=0.0)
    sig_mc = np.nan_to_num(ordered.get("MCCountsSignal", pd.Series(0.0, index=ordered.index)).to_numpy(dtype=float), nan=0.0)
    bkg_mc = np.nan_to_num(ordered.get("MCCountsBackground", pd.Series(0.0, index=ordered.index)).to_numpy(dtype=float), nan=0.0)

    sm_sig = smooth_histogram_with_config(sig_c, smoothing_config)
    sm_bkg = smooth_histogram_with_config(bkg_c, smoothing_config)
    sm_sig_e = smooth_histogram_errors(sig_e, smoothing_config, counts=sig_c, mc_counts=sig_mc)
    sm_bkg_e = smooth_histogram_errors(bkg_e, smoothing_config, counts=bkg_c, mc_counts=bkg_mc)

    emin, emax = analysis_config.get("energy_min"), analysis_config.get("energy_max")
    mask = np.ones_like(energy, dtype=bool)
    if emin is not None:
        mask &= energy >= float(emin)
    if emax is not None:
        mask &= energy <= float(emax)

    sig_kind = str(analysis_config.get("significance_type", "gaussian")).lower()
    if sig_kind == "asimov":
        raw_sig_eval = evaluate_significance(exposure * sig_c[mask], exposure * bkg_c[mask], background_uncertainty=exposure * bkg_e[mask], type=sig_kind)
        sm_sig_eval = evaluate_significance(exposure * sm_sig[mask], exposure * sm_bkg[mask], background_uncertainty=exposure * sm_bkg_e[mask], type=sig_kind)
    else:
        raw_sig_eval = evaluate_significance(exposure * sig_c[mask], exposure * bkg_c[mask], exposure * sig_e[mask], exposure * bkg_e[mask], type=sig_kind)
        sm_sig_eval = evaluate_significance(exposure * sm_sig[mask], exposure * sm_bkg[mask], exposure * sm_sig_e[mask], exposure * sm_bkg_e[mask], type=sig_kind)

    raw_sig = np.zeros_like(energy)
    sm_sig_out = np.zeros_like(energy)
    raw_sig[mask] = np.nan_to_num(raw_sig_eval, nan=0.0, posinf=0.0, neginf=0.0)
    sm_sig_out[mask] = np.nan_to_num(sm_sig_eval, nan=0.0, posinf=0.0, neginf=0.0)
    combine_mode = analysis_config.get("combine_mode", "quadrature")
    return {
        "Energy": energy,
        "RawSignal": exposure * sig_c, "SmoothedSignal": exposure * sm_sig,
        "RawBackground": exposure * bkg_c, "SmoothedBackground": exposure * sm_bkg,
        "RawSignificance": raw_sig, "SmoothedSignificance": sm_sig_out,
        "RawTotal": _aggregate_significance(raw_sig, combine_mode),
        "SmoothedTotal": _aggregate_significance(sm_sig_out, combine_mode),
    }


def _select_best_fiducial(plot_df, analysis_config, smoothing_config, exposure):
    sig_df = combine_components(plot_df, analysis_config.get("signal_components", []))
    bkg_df = combine_components(plot_df, analysis_config.get("background_components", []))
    merged = pd.merge(sig_df, bkg_df, on=["Energy", "FiducializedX", "FiducializedY", "FiducializedZ"], how="outer", suffixes=("Signal", "Background")).fillna(0.0)
    rows = []
    for fid_vals, group in merged.groupby(["FiducializedX", "FiducializedY", "FiducializedZ"]):
        payload = _build_fiducial_significance(group if isinstance(group, pd.DataFrame) else group.to_frame().T, analysis_config, smoothing_config, exposure)
        if payload is None:
            continue
        rows.append({"FiducializedX": int(fid_vals[0]), "FiducializedY": int(fid_vals[1]), "FiducializedZ": int(fid_vals[2]), "RawSignificance": payload["RawTotal"], "SmoothedSignificance": payload["SmoothedTotal"]})
    return merged, pd.DataFrame(rows)


def _get_fiducial_payload(merged_df, fx, fy, fz, analysis_config, smoothing_config, exposure):
    mask = (merged_df["FiducializedX"] == fx) & (merged_df["FiducializedY"] == fy) & (merged_df["FiducializedZ"] == fz)
    return _build_fiducial_significance(merged_df.loc[mask].copy(), analysis_config, smoothing_config, exposure)


def _build_fiducial_render_list(plot_df, signal_components, smoothing_config, exposure):
    render_list, energy_axis = [], None
    for component in plot_df["Component"].unique():
        cd = plot_df[plot_df["Component"] == component].sort_values("Energy")
        x = cd["Energy"].to_numpy(dtype=float)
        if energy_axis is None:
            energy_axis = x
        y = cd["Counts"].to_numpy(dtype=float)
        ep = cd["Error+"].to_numpy(dtype=float)
        em = cd["Error-"].to_numpy(dtype=float)
        mc = cd["MCCounts"].to_numpy(dtype=float) if "MCCounts" in cd.columns else None
        cfg = get_component_smoothing_config(smoothing_config, component)
        sy = smooth_histogram_with_config(y, cfg)
        sep = smooth_histogram_errors(ep, cfg, counts=y, mc_counts=mc)
        sem = smooth_histogram_errors(em, cfg, counts=y, mc_counts=mc)
        is_sig = str(component).lower() in signal_components
        render_list.append({
            "component_label": component,
            "legend_group": 0 if is_sig else 1,
            "legend_group_title": "Signal" if is_sig else "Background",
            "color": _FIDUCIAL_COMPONENT_COLORS.get(str(component).lower(), "grey"),
            "energy_x": x,
            "raw_y": exposure * y,
            "smoothed_y": exposure * sy,
            "err_y": exposure * sep,
            "err_y_minus": exposure * sem,
            "total_counts": float(exposure * np.sum(y)),
        })
    return render_list, energy_axis if energy_axis is not None else np.array([])


def _add_fiducial_upper(fig, render_list, stacked):
    if stacked:
        for rd in render_list:
            fig.add_trace(
                go.Bar(x=rd["energy_x"], y=rd["raw_y"], name=rd["component_label"],
                       marker_color=rd["color"], legendgroup="component",
                       legendgrouptitle=dict(text="Component", font=dict(size=16))),
                row=1, col=1,
            )
        fig.update_layout(barmode="stack")
    else:
        for rd in render_list:
            fig.add_trace(
                go.Scatter(x=rd["energy_x"], y=rd["raw_y"], mode="lines", line_shape="hvh",
                           line=dict(color=rd["color"], width=2, dash="dot"), opacity=0.45,
                           legendgroup=rd["legend_group"], legendgrouptitle=dict(text=rd["legend_group_title"], font=dict(size=16)),
                           name=rd["component_label"], showlegend=False),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(x=rd["energy_x"], y=rd["smoothed_y"],
                           error_y=dict(type="data", symmetric=False, array=rd["err_y"], arrayminus=rd["err_y_minus"], visible=True),
                           mode="lines+markers", line_shape="hvh",
                           legendgroup=rd["legend_group"], legendgrouptitle=dict(text=rd["legend_group_title"], font=dict(size=16)),
                           name=f"{rd['component_label']} {rd['total_counts']:.1e}",
                           line=dict(color=rd["color"], width=3), marker=dict(size=6, color=rd["color"]), showlegend=True),
                row=1, col=1,
            )
        for label, w, dash in [("Raw", 2, "dot"), ("Smoothed", 3, "solid")]:
            fig.add_trace(
                go.Scatter(x=[None], y=[None], mode="lines", name=label,
                           line=dict(color="gray", width=w, dash=dash),
                           legendgroup="linestyle", legendgrouptitle=dict(text="Histogram", font=dict(size=16)), showlegend=True),
                row=1, col=1,
            )


# ── arg parser ─────────────────────────────────────────────────────────────────

analysis_info = load_analysis_info(str(root))
_hep_default_ref = analysis_info.get("BEST_SIGMA_SIGNIFICANCE_REFERENCE", {}).get("HEP", "Asimov")
if _hep_default_ref not in ["Gaussian", "Asimov", "ProfileLikelihood"]:
    _hep_default_ref = "Asimov"

parser = argparse.ArgumentParser(
    description="Unified significance plot for DayNight, HEP, Sensitivity, and Fiducial analyses"
)
# dispatcher
parser.add_argument(
    "--analysis", type=str,
    choices=["DayNight", "HEP", "Sensitivity", "Fiducial"],
    default="DayNight",
)
# Fiducial sub-analyses (which analyses to compute fiducials for)
parser.add_argument(
    "--fiducial-analyses", nargs="+", type=str,
    choices=["DayNight", "HEP", "Sensitivity"],
    default=["DayNight", "HEP", "Sensitivity"],
    help="Sub-analyses to plot when --analysis Fiducial",
)
# shared
parser.add_argument("--config", nargs="+", type=str, default=["hd_1x2x6_centralAPA"])
parser.add_argument("--name", nargs="+", type=str, default=["marley"])
parser.add_argument("--folder", type=str, default="Reduced")
parser.add_argument("--signal_uncertainty", type=float, default=None)
parser.add_argument("--background_uncertainty", type=float, default=None)
parser.add_argument("--exposure", type=float, default=30)
parser.add_argument(
    "--energy", nargs="+", type=str,
    default=["ClusterEnergy", "TotalEnergy", "SelectedEnergy", "SolarEnergy"],
)
parser.add_argument("--nhits", type=int, default=None)
parser.add_argument("--ophits", type=int, default=None)
parser.add_argument("--adjcls", type=int, default=None)
parser.add_argument(
    "--threshold", type=float, default=None,
    help="Energy threshold; resolved per-analysis from analysis/config.json if not set",
)
parser.add_argument("--zoom", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
# reference (meaning differs per analysis)
parser.add_argument(
    "--reference", type=str, default=None,
    help=(
        "Significance reference: {Gaussian,Asimov} for DayNight; "
        "{Gaussian,Asimov,ProfileLikelihood} for HEP; ignored for Sensitivity (always Asimov inline)"
    ),
)
parser.add_argument("--stacked", action=argparse.BooleanOptionalAction, default=False,
                    help="DayNight/HEP/Sensitivity: also generate stacked-area variant. Fiducial: use Bar stacked (no overlayed)")
parser.add_argument("--show_bin_labels", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--bin_label_digits", type=int, default=1)
parser.add_argument("--bin_label_stride", type=int, default=1)
parser.add_argument("--bin_label_y_offset_fraction", type=float, default=0.1)
# HEP-only
parser.add_argument(
    "--bottom-panel-mode", type=str,
    choices=["rigorous", "intuitive", "both"],
    default="both",
    help="Lower-panel style: rigorous=grouped adaptive-rebin proxy, intuitive=local density, both=two separate figures",
)
parser.add_argument(
    "--pkl-label", type=str, default="highest",
    help="Label of best-cut pkl to read (e.g. 'highest', 'highest_spiked')",
)
parser.add_argument(
    "--local-metric", type=str,
    choices=["AsimovTS", "Fisher", "SNR", "Purity"],
    default="AsimovTS",
    help="Overlay metric shown in HEP intuitive lower panel",
)

args = parser.parse_args()

# ── post-parse defaults ────────────────────────────────────────────────────────

if args.reference is None:
    if args.analysis == "DayNight":
        args.reference = "Gaussian"
    elif args.analysis == "HEP":
        args.reference = _hep_default_ref
    else:
        args.reference = "Asimov"

if args.analysis != "Fiducial":
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

# ── per-analysis paths and smoothing ──────────────────────────────────────────

if args.analysis == "DayNight":
    save_path = f"{root}/images/analysis/day-night"
    data_path = f"{analysis_info['PATH']}/DAYNIGHT"
    smoothing_config = get_smoothing_config(str(root), analysis_name="DAYNIGHT", dimensions="1d", stage="significance")
elif args.analysis == "HEP":
    save_path = f"{root}/images/analysis/hep"
    data_path = f"{analysis_info['PATH']}/HEP"
    smoothing_config = get_smoothing_config(str(root), analysis_name="HEP", dimensions="1d", stage="significance")
elif args.analysis == "Sensitivity":
    save_path = f"{root}/images/analysis/sensitivity"
    data_path = f"{analysis_info['PATH']}/SENSITIVITY"
    smoothing_config = get_smoothing_config(str(root), analysis_name="SENSITIVITY", dimensions="1d", stage="significance")
else:
    save_path = f"{root}/images/solar/fiducial"
    data_path = f"{analysis_info['PATH']}/FIDUCIAL"
    smoothing_config = None

for this_path in [save_path, data_path]:
    if not os.path.exists(this_path):
        os.makedirs(this_path)

smoothing_info = smoothing_metadata(smoothing_config) if smoothing_config is not None else {}

# ── HEP derived vars ───────────────────────────────────────────────────────────

if args.analysis == "HEP":
    _ref = args.reference
    reference_for_bins = "Asimov" if _ref == "ProfileLikelihood" else _ref
    reference_label = "Asimov Proxy" if _ref == "ProfileLikelihood" else f"{reference_for_bins} Proxy"
    local_proxy_label = "Local Discovery"
    saved_variable = "AsimovProxy" if _ref == "ProfileLikelihood" else reference_for_bins

# ── accumulators ───────────────────────────────────────────────────────────────

_counts_list = []
_significance_list = []

# ── main loop ──────────────────────────────────────────────────────────────────

for config, name, energy in product(args.config, args.name, args.energy):
    info = json.loads(open(f"{root}/config/{config}/{config}_config.json").read())
    detector_mass = get_full_detector_mass(config, info)

    # ══════════════════════════════════════════════════════════════════════════
    # DayNight
    # ══════════════════════════════════════════════════════════════════════════
    if args.analysis == "DayNight":
        sigma = pickle.load(
            open(
                f"{info['PATH']}/DAYNIGHT/{args.folder.lower()}/{config}/{args.name[0]}"
                f"/{config}_{args.name[0]}_highest_DayNight.pkl",
                "rb",
            )
        )
        sigmas_df = pd.read_pickle(
            f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/DAYNIGHT/{args.folder.lower()}"
            f"/{config}/{name}/{config}_{name}_{energy}_DayNight_Results.pkl"
        )
        significance_bins_path = (
            f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/DAYNIGHT/{args.folder.lower()}"
            f"/{config}/{name}/{config}_{name}_{energy}_DayNight_SignificanceBins.pkl"
        )
        if not os.path.exists(significance_bins_path):
            rprint(f"[yellow][WARNING][/yellow] Missing per-bin significance payload for {config} {name} {energy}.")
            continue
        significance_bins_df = pd.read_pickle(significance_bins_path)
        if significance_bins_df.empty or "Config" not in significance_bins_df.columns:
            rprint(f"[yellow][WARNING][/yellow] Empty or invalid SignificanceBins pkl for {config} {name} {energy}. Regenerate with the DayNight significance pipeline.")
            continue

        plot_df = pd.read_pickle(
            f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/signal/{args.folder.lower()}"
            f"/DAYNIGHT/{config}/{name}/{config}_{name}_{energy}_Rebin.pkl"
        )
        background_samples = []
        for bkg, filepath in load_available_background_dataframes(str(root), "DAYNIGHT", args.folder, config, energy):
            plot_df = pd.concat([plot_df, pd.read_pickle(filepath)], ignore_index=True)
            background_samples.append(bkg)

        try:
            ref_plot = sigma[(config, name, energy)]
        except KeyError:
            rprint(f"[yellow][WARNING] Not found highest for {config} {name} {energy}[/yellow]")
            continue

        nhits_value = args.nhits if args.nhits is not None else int(ref_plot["NHits"])
        adjcl_value = args.adjcls if args.adjcls is not None else int(ref_plot["AdjCl"])
        ophits_value = args.ophits if args.ophits is not None else int(ref_plot["OpHits"])

        this_plot_df = plot_df.loc[
            (plot_df["NHits"] == nhits_value)
            * (plot_df["OpHits"] == ophits_value)
            * (plot_df["AdjCl"] == adjcl_value)
        ].copy()
        if this_plot_df.empty:
            continue

        plot_sigmas = sigmas_df.loc[
            (sigmas_df["Config"] == config)
            * (sigmas_df["Name"] == name)
            * (sigmas_df["NHits"] == nhits_value)
            * (sigmas_df["OpHits"] == ophits_value)
            * (sigmas_df["AdjCl"] == adjcl_value)
        ].copy()
        if plot_sigmas.empty:
            rprint(
                f"[yellow][WARNING][/yellow] Missing precomputed significance payload for "
                f"{config} {name} {energy} NHits={nhits_value} OpHits={ophits_value} AdjCl={adjcl_value}."
            )
            continue

        energy_axis = np.asarray(this_plot_df["Energy"].values[0], dtype=float)
        threshold_idx = 0
        if args.threshold is not None:
            above = np.where(energy_axis > args.threshold)[0]
            if above.size:
                threshold_idx = int(above[0])
        bin_width = float(np.median(np.diff(energy_axis))) if len(energy_axis) > 1 else 1.0

        selected_bins = significance_bins_df.loc[
            (significance_bins_df["Config"] == config)
            * (significance_bins_df["Name"] == name)
            * (significance_bins_df["EnergyLabel"] == energy)
            * (significance_bins_df["NHits"] == int(nhits_value))
            * (significance_bins_df["OpHits"] == int(ophits_value))
            * (significance_bins_df["AdjCl"] == int(adjcl_value))
        ].copy()
        if selected_bins.empty:
            rprint(
                f"[yellow][WARNING][/yellow] Missing per-bin significance payload for "
                f"{config} {name} {energy} NHits={nhits_value} OpHits={ophits_value} AdjCl={adjcl_value}."
            )
            continue

        selected_bins = selected_bins.sort_values("BinIndex")
        stored_energy_axis = np.asarray(selected_bins["RecoEnergy"].values, dtype=float)
        raw_significance = np.nan_to_num(
            np.asarray(selected_bins["RawGaussian"].values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0,
        )
        smoothed_significance = np.nan_to_num(
            np.asarray(selected_bins["Gaussian"].values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0,
        )

        _has_asimov_bins = "RawAsimov" in selected_bins.columns and "Asimov" in selected_bins.columns
        if _has_asimov_bins:
            raw_asimov_sig = np.nan_to_num(
                np.asarray(selected_bins["RawAsimov"].values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0,
            )
            smoothed_asimov_sig = np.nan_to_num(
                np.asarray(selected_bins["Asimov"].values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0,
            )
        else:
            raw_asimov_sig = None
            smoothed_asimov_sig = None

        signal_day_raw = None
        signal_night_raw = None
        positive_count_values = []
        _dn_render = []

        dn_component_specs = [
            ("Solar", "Solar Day", "Osc", "Day", 0, "Signal", compare[1]),
            ("Solar", "Solar Night", "Osc", "Night", 0, "Signal", compare[0]),
        ]
        for bkg in background_samples:
            style = get_background_style(str(root), bkg)
            dn_component_specs.append(
                (bkg, style.get("label", bkg).title(), "Truth", "Mean", 1, "Background", style.get("color", "grey"))
            )

        for component, component_label, osc, mean, legend_group, legend_group_title, color in dn_component_specs:
            comp_df = this_plot_df.loc[
                (this_plot_df["Component"] == component)
                * (this_plot_df["Oscillation"] == osc)
                * (this_plot_df["Mean"] == mean)
            ].copy()
            if comp_df.empty:
                continue

            counts = np.asarray(comp_df["Counts"].values[0], dtype=float)
            errors = np.asarray(comp_df["Error"].values[0], dtype=float)
            mc_counts = (
                np.asarray(comp_df["MCCounts"].values[0], dtype=float)
                if "MCCounts" in comp_df.columns else None
            )
            component_smoothing_config = get_component_smoothing_config(smoothing_config, component)
            raw_counts_per_energy = np.nan_to_num(
                detector_mass * args.exposure * np.asarray(comp_df["Counts/Energy"].values[0], dtype=float),
                nan=0.0, posinf=0.0, neginf=0.0,
            )
            smoothed_counts = smooth_threshold_slice(counts, threshold_idx, component_smoothing_config)
            smoothed_errors = errors.copy()
            smoothed_errors[threshold_idx:] = smooth_histogram_errors(
                errors[threshold_idx:], component_smoothing_config,
                counts=counts[threshold_idx:],
                mc_counts=mc_counts[threshold_idx:] if mc_counts is not None else None,
            )
            smoothed_counts_per_energy = np.nan_to_num(
                detector_mass * args.exposure * smoothed_counts / bin_width, nan=0.0, posinf=0.0, neginf=0.0,
            )
            smoothed_errors_per_energy = np.nan_to_num(
                detector_mass * args.exposure * smoothed_errors / bin_width, nan=0.0, posinf=0.0, neginf=0.0,
            )
            positive_count_values.extend(raw_counts_per_energy[raw_counts_per_energy > 0])
            positive_count_values.extend(smoothed_counts_per_energy[smoothed_counts_per_energy > 0])

            _dn_render.append({
                "component_label": component_label, "legend_group": legend_group,
                "legend_group_title": legend_group_title, "color": color, "opacity": 0.45,
                "raw_y": raw_counts_per_energy,
                "smoothed_y": smoothed_counts_per_energy,
                "err_y": smoothed_errors_per_energy,
            })

            raw_errors_per_energy = np.nan_to_num(
                detector_mass * args.exposure * errors / bin_width, nan=0.0, posinf=0.0, neginf=0.0,
            )
            for counts_per_energy, errors_per_energy, spectrum_type in [
                (raw_counts_per_energy, raw_errors_per_energy, "Raw"),
                (smoothed_counts_per_energy, smoothed_errors_per_energy, "Smoothed"),
            ]:
                significance_values = None
                if component_label == "Solar Day":
                    sig_source = smoothed_significance if spectrum_type == "Smoothed" else raw_significance
                    energy_len = len(energy_axis)
                    bin_len = len(sig_source)
                    if bin_len >= energy_len:
                        significance_values = sig_source[:energy_len]
                    else:
                        significance_values = np.pad(
                            sig_source, (0, energy_len - bin_len), mode="constant", constant_values=np.nan,
                        )
                _counts_list.append({
                    "Config": config, "Name": name, "EnergyLabel": energy, "Analysis": "DayNight",
                    "Geometry": info["GEOMETRY"],
                    "Component": component_label, "SpectrumType": spectrum_type,
                    "NHits": int(nhits_value), "OpHits": int(ophits_value), "AdjCl": int(adjcl_value),
                    "Exposure": args.exposure, "ExposureUnit": "year",
                    "Energy": energy_axis.tolist(), "EnergyUnit": "MeV",
                    "Counts": np.asarray(counts_per_energy).tolist(), "CountsUnit": r"counts \cdot MeV^{-1}",
                    "CountsError": np.asarray(errors_per_energy).tolist(),
                    "Significance": np.asarray(significance_values).tolist() if significance_values is not None else None,
                    "SignificanceLabel": f"Gaussian {spectrum_type}" if component_label == "Solar Day" else None,
                })

            if osc != "Truth":
                if mean == "Day":
                    signal_day_raw = counts
                else:
                    signal_night_raw = counts

        if signal_day_raw is None or signal_night_raw is None:
            continue

        for label, significance, dash in [
            ("Gaussian", raw_significance, "dot"),
            ("Gaussian", smoothed_significance, "solid"),
        ]:
            _significance_list.append({
                "Config": config, "Name": name, "EnergyLabel": energy, "Analysis": "DayNight",
                "Variable": label, "SpectrumType": "Raw" if dash == "dot" else "Smoothed",
                "BinMode": "PerBin",
                "NHits": int(nhits_value), "OpHits": int(ophits_value), "AdjCl": int(adjcl_value),
                "Exposure": args.exposure, "ExposureUnit": "year",
                "Energy": stored_energy_axis.tolist(), "EnergyUnit": "MeV",
                "Significance": np.asarray(significance).tolist(), "SignificanceUnit": r"\sigma",
                "BinWidth": None, "BinWidthUnit": None,
            })

        if _has_asimov_bins:
            for label, significance, dash in [
                ("Asimov", raw_asimov_sig, "dot"),
                ("Asimov", smoothed_asimov_sig, "solid"),
            ]:
                _significance_list.append({
                    "Config": config, "Name": name, "EnergyLabel": energy, "Analysis": "DayNight",
                    "Variable": label, "SpectrumType": "Raw" if dash == "dot" else "Smoothed",
                    "BinMode": "PerBin",
                    "NHits": int(nhits_value), "OpHits": int(ophits_value), "AdjCl": int(adjcl_value),
                    "Exposure": args.exposure, "ExposureUnit": "year",
                    "Energy": stored_energy_axis.tolist(), "EnergyUnit": "MeV",
                    "Significance": np.asarray(significance).tolist(), "SignificanceUnit": r"\sigma",
                    "BinWidth": None, "BinWidthUnit": None,
                })

        figure_name = f"{energy}_DayNight_Significance"
        if args.nhits is not None or args.ophits is not None or args.adjcls is not None:
            figure_name += f"_NHits{nhits_value:.0f}_OpHits{ophits_value:.0f}_AdjCl{adjcl_value:.0f}"
        if args.exposure is not None:
            figure_name += f"_Exposure_{args.exposure:.0f}"

        def _dn_lower(fig):
            for label, significance, dash, showlegend in [
                ("Gaussian", raw_significance, "dot", False),
                ("Gaussian", smoothed_significance, "solid", True),
            ]:
                add_significance_series_trace(
                    fig, x=stored_energy_axis, y=significance, name_prefix=label,
                    row=2, col=1, color="black", width=3 if dash == "solid" else 2, dash=dash,
                    legend="legend2", legendgroup="Significance", legendgrouptitle="Significance",
                    showlegend=showlegend, append_total=True, total_digits=1,
                )
                if args.show_bin_labels and dash == "solid":
                    add_significance_bin_labels(
                        fig, x=stored_energy_axis, y=significance, row=2, col=1,
                        text_prefix="", digits=args.bin_label_digits, label_stride=args.bin_label_stride,
                        show_zero=False, color="black", font_size=10, textposition="top center",
                        y_offset_fraction=args.bin_label_y_offset_fraction,
                    )
            if _has_asimov_bins:
                for label, significance, dash, showlegend in [
                    ("Asimov", raw_asimov_sig, "dot", False),
                    ("Asimov", smoothed_asimov_sig, "solid", True),
                ]:
                    add_significance_series_trace(
                        fig, x=stored_energy_axis, y=significance, name_prefix=label,
                        row=2, col=1, color="rgb(31,119,180)", width=3 if dash == "solid" else 2, dash=dash,
                        legend="legend2", legendgroup="Significance", legendgrouptitle="Significance",
                        showlegend=showlegend, append_total=True, total_digits=1,
                    )
            add_histogram_style_legend_traces(fig, row=1, col=1, legend="legend")
            _sig_max = max(float(np.max(raw_significance)), float(np.max(smoothed_significance)), 1.0)
            if _has_asimov_bins:
                _sig_max = max(_sig_max, float(np.max(raw_asimov_sig)), float(np.max(smoothed_asimov_sig)))
            return _sig_max

        _render_and_save(
            render_list=_dn_render, energy_axis=energy_axis, add_lower_fn=_dn_lower,
            subplot_title=f"{energy}, min#Hits {nhits_value:.0f}, min#OpHits {ophits_value:.0f}, max#AdjCl {adjcl_value:.0f}",
            fig_title=f"Day-Night Asymmetry - {args.folder} - {config}",
            upper_y_title=f"Counts ({args.exposure:.0f} year·MeV)⁻¹",
            lower_y_title="Significance (σ)",
            x_range=[6.75, 26], figure_name=figure_name,
            positive_count_values=positive_count_values,
            log_default_range=[-2, np.log10(detector_mass * args.exposure * 1e4)],
            save_path=save_path, config=config, save_name=name, folder=args.folder.lower(),
            rewrite=args.rewrite, debug_plot=args.plot,
            stacked_range=[False, True] if args.stacked else [False],
            zoom=args.zoom, threshold=args.threshold,
            upper_y_extra={"tickformat": ".0e", "dtick": 1},
            lower_y_extra={"tickformat": ".0f", "dtick": 1},
            lower_y_default_range=[0, 6],
            legend_layout={"legend2": dict(x=0.79, y=0.14, font=dict(size=12))},
            x_title="Reconstructed Neutrino Energy (MeV)",
        )
        for df, df_name in zip(
            [pd.DataFrame(_counts_list), pd.DataFrame(_significance_list)],
            ["DayNight_Counts", "DayNight_Significance"],
        ):
            save_df(
                df, data_path, config, name,
                subfolder=args.folder.lower(), filename=df_name,
                rm=args.rewrite, debug=True if df_name == "DayNight_Counts" else args.debug,
            )

    # ══════════════════════════════════════════════════════════════════════════
    # HEP
    # ══════════════════════════════════════════════════════════════════════════
    elif args.analysis == "HEP":
        sigma_map_path = (
            f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/HEP/{args.folder.lower()}"
            f"/{config}/{name}/{config}_{name}_{args.pkl_label}_HEP.pkl"
        )
        if not os.path.exists(sigma_map_path):
            rprint(f"[yellow][WARNING][/yellow] Missing best-cut map for {config} {name}: {sigma_map_path}")
            continue

        sigma_map = pd.read_pickle(sigma_map_path)
        try:
            ref_plot = sigma_map[(config, name, energy)]
        except KeyError:
            rprint(f"[yellow][WARNING][/yellow] Missing best-cut entry for {config} {name} {energy}.")
            continue

        nhits_value = args.nhits if args.nhits is not None else int(ref_plot["NHits"])
        ophits_value = args.ophits if args.ophits is not None else int(ref_plot["OpHits"])
        adjcl_value = args.adjcls if args.adjcls is not None else int(ref_plot["AdjCl"])

        significance_bins_path = (
            f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/HEP/{args.folder.lower()}"
            f"/{config}/{name}/{config}_{name}_{energy}_HEP_SignificanceBins.pkl"
        )
        if not os.path.exists(significance_bins_path):
            rprint(f"[yellow][WARNING][/yellow] Missing per-bin significance payload for {config} {name} {energy}.")
            continue

        significance_bins_df = pd.read_pickle(significance_bins_path)
        required_bin_columns = [
            "Config", "Name", "EnergyLabel", "NHits", "OpHits", "AdjCl",
            "BinMode", "BinIndex", "RecoEnergy", "BinWidth",
            f"Raw{reference_for_bins}", f"{reference_for_bins}",
        ]
        missing_bin_columns = [c for c in required_bin_columns if c not in significance_bins_df.columns]
        if significance_bins_df.empty or missing_bin_columns:
            rprint(
                f"[yellow][WARNING][/yellow] Invalid per-bin significance payload for {config} {name} {energy}: "
                f"missing columns {missing_bin_columns}. Skipping."
            )
            continue

        selected_no_rebin = significance_bins_df.loc[
            (significance_bins_df["Config"] == config)
            * (significance_bins_df["Name"] == name)
            * (significance_bins_df["EnergyLabel"] == energy)
            * (significance_bins_df["NHits"] == int(nhits_value))
            * (significance_bins_df["OpHits"] == int(ophits_value))
            * (significance_bins_df["AdjCl"] == int(adjcl_value))
            * (significance_bins_df["BinMode"] == "NoRebin")
        ].copy()
        selected_adaptive = significance_bins_df.loc[
            (significance_bins_df["Config"] == config)
            * (significance_bins_df["Name"] == name)
            * (significance_bins_df["EnergyLabel"] == energy)
            * (significance_bins_df["NHits"] == int(nhits_value))
            * (significance_bins_df["OpHits"] == int(ophits_value))
            * (significance_bins_df["AdjCl"] == int(adjcl_value))
            * (significance_bins_df["BinMode"] == "AdaptiveRebin")
        ].copy()
        if selected_no_rebin.empty:
            rprint(
                f"[yellow][WARNING][/yellow] Missing no-rebin significance bins for "
                f"{config} {name} {energy} NHits={nhits_value} OpHits={ophits_value} AdjCl={adjcl_value}."
            )
            continue

        selected_no_rebin = selected_no_rebin.sort_values("BinIndex")
        selected_adaptive = selected_adaptive.sort_values("BinIndex")

        no_rebin_energy = _safe_array(selected_no_rebin["RecoEnergy"].values)
        no_rebin_width = _safe_array(selected_no_rebin["BinWidth"].values)
        raw_no_rebin_significance = _safe_array(selected_no_rebin[f"Raw{reference_for_bins}"].values)
        smoothed_no_rebin_significance = _safe_array(selected_no_rebin[f"{reference_for_bins}"].values)

        adaptive_energy = _safe_array(selected_adaptive["RecoEnergy"].values)
        adaptive_width = _safe_array(selected_adaptive["BinWidth"].values)
        raw_adaptive_significance = _safe_array(selected_adaptive[f"Raw{reference_for_bins}"].values)
        smoothed_adaptive_significance = _safe_array(selected_adaptive[f"{reference_for_bins}"].values)

        plot_df = pd.read_pickle(
            f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/signal/{args.folder.lower()}"
            f"/HEP/{config}/{name}/{config}_{name}_{energy}_Rebin.pkl"
        )
        for _, filepath in load_available_background_dataframes(str(root), "HEP", args.folder, config, energy):
            plot_df = pd.concat([plot_df, pd.read_pickle(filepath)], ignore_index=True)

        this_plot_df = plot_df.loc[
            (plot_df["NHits"] == int(nhits_value))
            * (plot_df["OpHits"] == int(ophits_value))
            * (plot_df["AdjCl"] == int(adjcl_value))
        ].copy()
        if this_plot_df.empty:
            rprint(f"[yellow][WARNING][/yellow] Missing histogram payload for {config} {name} {energy}.")
            continue

        hep_component_specs = [
            ("hep", "HEP", "Osc", "Mean", "signal", "Signal", "rgb(204,80,62)"),
        ]
        for bkg, oscillation, color in [
            ("gamma", "Truth", get_background_style(str(root), "gamma").get("color", "black")),
            ("neutron", "Truth", get_background_style(str(root), "neutron").get("color", "rgb(15,133,84)")),
            ("radiological", "Truth", get_background_style(str(root), "radiological").get("color", "rgb(120, 94, 240)")),
            ("8B", "Osc", "rgb(225,124,5)"),
        ]:
            style = get_background_style(str(root), bkg)
            hep_component_specs.append(
                (bkg, style.get("label", bkg), oscillation, "Mean", "background", "Background", color)
            )

        scale = detector_mass * args.exposure
        panel_modes = ["rigorous", "intuitive"] if args.bottom_panel_mode == "both" else [args.bottom_panel_mode]

        _hep_energy_axis = hep_rebin_centers
        _hep_n_bins = len(_hep_energy_axis)

        _hep_render = []
        total_raw_counts = np.zeros(_hep_n_bins, dtype=float)
        total_smoothed_counts = np.zeros(_hep_n_bins, dtype=float)
        signal_smoothed_counts = np.zeros(_hep_n_bins, dtype=float)
        background_smoothed_counts = np.zeros(_hep_n_bins, dtype=float)
        radiological_present = False
        positive_count_values = []

        for component, component_label, oscillation, mean_label, legend_group, legend_group_title, color in hep_component_specs:
            comp_df = this_plot_df.loc[
                (this_plot_df["Component"] == component)
                * (this_plot_df["Oscillation"] == oscillation)
                * (this_plot_df["Mean"] == mean_label)
            ].copy()
            if comp_df.empty:
                continue

            if component == "radiological":
                radiological_present = True

            cd = _process_component_df(comp_df, component, smoothing_config, scale=scale)
            total_raw_counts += cd["counts"]
            total_smoothed_counts += cd["smoothed_counts"]
            if component == "hep":
                signal_smoothed_counts += cd["smoothed_counts"]
            else:
                background_smoothed_counts += cd["smoothed_counts"]

            positive_count_values.extend(cd["raw_y"][cd["raw_y"] > 0])
            positive_count_values.extend(cd["smoothed_y"][cd["smoothed_y"] > 0])

            _hep_render.append({
                "component": component, "component_label": component_label,
                "legend_group": legend_group, "legend_group_title": legend_group_title,
                "color": color,
                "counts": cd["counts"], "smoothed_counts": cd["smoothed_counts"],
                "raw_y": cd["raw_y"], "smoothed_y": cd["smoothed_y"], "err_y": cd["err_y"],
            })

        no_rebin_start = 0
        if no_rebin_energy.size > 0:
            no_rebin_start = int(np.argmin(np.abs(_hep_energy_axis - no_rebin_energy[0])))
        signal_tail_counts = signal_smoothed_counts[no_rebin_start:no_rebin_start + len(no_rebin_energy)]
        background_tail_counts = background_smoothed_counts[no_rebin_start:no_rebin_start + len(no_rebin_energy)]

        grouped_background_counts = (
            _group_tail_values(no_rebin_energy, scale * background_tail_counts, adaptive_energy, adaptive_width)
            if adaptive_energy.size > 0 else np.zeros(0, dtype=float)
        )
        grouped_signal_counts = (
            _group_tail_values(no_rebin_energy, scale * signal_tail_counts, adaptive_energy, adaptive_width)
            if adaptive_energy.size > 0 else np.zeros(0, dtype=float)
        )

        no_rebin_signal_counts = scale * signal_tail_counts
        no_rebin_background_counts = scale * background_tail_counts
        no_rebin_b_safe = np.maximum(no_rebin_background_counts, 1e-12)
        with np.errstate(divide="ignore", invalid="ignore"):
            no_rebin_asimov = 2.0 * (
                (no_rebin_signal_counts + no_rebin_b_safe)
                * np.log1p(np.divide(no_rebin_signal_counts, no_rebin_b_safe))
                - no_rebin_signal_counts
            )
        no_rebin_asimov = np.nan_to_num(no_rebin_asimov, nan=0.0, posinf=0.0, neginf=0.0)
        no_rebin_asimov_density = np.divide(
            no_rebin_asimov, no_rebin_width, out=np.zeros_like(no_rebin_asimov), where=no_rebin_width > 0,
        )
        no_rebin_fisher_density = np.divide(
            np.divide(
                no_rebin_signal_counts * no_rebin_signal_counts, no_rebin_b_safe,
                out=np.zeros_like(no_rebin_signal_counts, dtype=float), where=no_rebin_b_safe > 0,
            ),
            no_rebin_width, out=np.zeros_like(no_rebin_signal_counts, dtype=float), where=no_rebin_width > 0,
        )
        no_rebin_snr_density = np.divide(
            np.divide(
                no_rebin_signal_counts, np.sqrt(no_rebin_b_safe),
                out=np.zeros_like(no_rebin_signal_counts, dtype=float), where=no_rebin_b_safe > 0,
            ),
            no_rebin_width, out=np.zeros_like(no_rebin_signal_counts, dtype=float), where=no_rebin_width > 0,
        )
        no_rebin_purity = np.divide(
            no_rebin_signal_counts, no_rebin_signal_counts + no_rebin_background_counts,
            out=np.zeros_like(no_rebin_signal_counts, dtype=float),
            where=(no_rebin_signal_counts + no_rebin_background_counts) > 0,
        )

        raw_local_density = _density(raw_no_rebin_significance, no_rebin_width)
        smooth_local_density = _density(smoothed_no_rebin_significance, no_rebin_width)

        def _hep_pre_lower(fig):
            if not radiological_present:
                fig.add_trace(
                    go.Scatter(
                        x=[None], y=[None], mode="lines",
                        name="radiological (not present after cuts)",
                        line=dict(color=get_background_style(str(root), "radiological").get("color", "rgb(120, 94, 240)"), width=3, dash="dash"),
                        legend="legend", legendgroup="background",
                        legendgrouptitle=dict(text="Background"), showlegend=True,
                    ),
                    row=1, col=1,
                )

        for mode in panel_modes:
            def _hep_lower(fig, mode=mode):
                lmax = 0.0
                if mode == "rigorous" and adaptive_energy.size > 0:
                    _add_rebinned_overlay(
                        fig, adaptive_energy, adaptive_width,
                        grouped_background_counts, grouped_signal_counts, signal_label="HEP Rebin",
                    )
                if mode == "rigorous":
                    if adaptive_energy.size == 0:
                        fig.add_annotation(
                            x=0.5, y=0.15, xref="paper", yref="paper",
                            text="Adaptive-rebin payload not available for this selection.", showarrow=False,
                        )
                    else:
                        fig.add_trace(
                            go.Bar(
                                x=adaptive_energy, y=smoothed_adaptive_significance, width=adaptive_width,
                                name="Smoothed", marker_color="rgba(31,119,180,0.45)",
                                marker_line=dict(color="rgb(31,119,180)", width=1.5),
                                legend="legend2", legendgroup="proxy",
                                legendgrouptitle=dict(text="Local Proxy"), showlegend=True,
                            ),
                            row=2, col=1,
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=adaptive_energy, y=raw_adaptive_significance, mode="lines", name="Raw",
                                line=dict(color="black", width=2, dash="dot", shape="hvh"),
                                error_x=dict(type="data", array=0.5 * adaptive_width, visible=True),
                                legend="legend2", legendgroup="proxy", showlegend=True,
                            ),
                            row=2, col=1,
                        )
                        adaptive_purity = np.divide(
                            grouped_signal_counts, grouped_signal_counts + grouped_background_counts,
                            out=np.zeros_like(grouped_signal_counts, dtype=float),
                            where=(grouped_signal_counts + grouped_background_counts) > 0,
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=adaptive_energy, y=adaptive_purity, mode="lines", name="Signal purity",
                                line=dict(color="rgb(230,159,0)", width=2, dash="dash"),
                                legend="legend3", legendgroup="purity_overlay",
                                legendgrouptitle=dict(text="Signal Purity"), showlegend=True,
                            ),
                            row=2, col=1, secondary_y=True,
                        )
                        lmax = max(float(np.max(smoothed_adaptive_significance)), float(np.max(raw_adaptive_significance)))
                else:
                    intuitive_density = _density(smoothed_no_rebin_significance, no_rebin_width)
                    intuitive_raw_density = _density(raw_no_rebin_significance, no_rebin_width)
                    lmax = max(float(np.max(intuitive_density)), float(np.max(intuitive_raw_density)))
                    fig.add_trace(
                        go.Scatter(
                            x=no_rebin_energy, y=intuitive_raw_density, mode="lines", name="Raw",
                            line=dict(color="black", width=2, dash="dot"), line_shape="hvh",
                            legend="legend2", legendgroup="proxy",
                            legendgrouptitle=dict(text="Local Proxy"), showlegend=True,
                        ),
                        row=2, col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=no_rebin_energy, y=intuitive_density, mode="lines", name="Smoothed",
                            line=dict(color="rgb(31,119,180)", width=3, dash="solid"), line_shape="hvh",
                            legend="legend2", legendgroup="proxy", showlegend=True,
                        ),
                        row=2, col=1,
                    )
                    if args.local_metric == "AsimovTS":
                        plotted_vals, legend_name = no_rebin_asimov_density, "Asimov TS density"
                    elif args.local_metric == "Fisher":
                        plotted_vals, legend_name = no_rebin_fisher_density, "Fisher info density"
                    elif args.local_metric == "SNR":
                        plotted_vals, legend_name = no_rebin_snr_density, "SNR density"
                    else:
                        plotted_vals, legend_name = no_rebin_purity, "Signal purity"
                    lmax = max(lmax, float(np.max(plotted_vals)))
                    fig.add_trace(
                        go.Bar(
                            x=no_rebin_energy, y=plotted_vals, width=no_rebin_width, name=legend_name,
                            marker_color="rgba(31,119,180,0.22)", marker_line=dict(color="rgb(31,119,180)", width=1),
                            legend="legend2", legendgroup="proxy", showlegend=True,
                        ),
                        row=2, col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=no_rebin_energy, y=no_rebin_purity, mode="lines", name="Signal purity",
                            line=dict(color="rgb(230,159,0)", width=2, dash="dash"),
                            legend="legend3", legendgroup="purity_overlay",
                            legendgrouptitle=dict(text="Signal Purity"), showlegend=True,
                        ),
                        row=2, col=1, secondary_y=True,
                    )
                return lmax

            _suffix = "BottomRigorous" if mode == "rigorous" else "BottomIntuitive"
            _hep_figure_name = f"{energy}_HEP_Significance_{args.reference}_Exposure_{args.exposure:.0f}_{_suffix}"
            if args.pkl_label != "highest":
                _hep_figure_name += f"_{args.pkl_label}"

            _render_and_save(
                render_list=_hep_render, energy_axis=_hep_energy_axis, add_lower_fn=_hep_lower,
                subplot_title=f"{energy} NHits={nhits_value} OpHits={ophits_value} AdjCl={adjcl_value}",
                fig_title=f"HEP Significance - {args.folder} - {config}",
                upper_y_title="Counts",
                lower_y_title=reference_label if mode == "rigorous" else local_proxy_label,
                x_range=[8, 30], figure_name=_hep_figure_name,
                positive_count_values=positive_count_values, log_default_range=[-2, 10],
                save_path=save_path, config=config, save_name=None, folder=args.folder.lower(),
                rewrite=args.rewrite, debug_plot=args.plot,
                stacked_range=[False, True] if args.stacked else [False],
                zoom=args.zoom, threshold=args.threshold,
                secondary_y=True, pre_lower_fn=_hep_pre_lower,
            )

        # HEP data output — outside panel_modes loop
        _hep_sig_base = {
            "Config": config, "Name": name, "EnergyLabel": energy, "Analysis": "HEP",
            "Variable": saved_variable,
            "NHits": int(nhits_value), "OpHits": int(ophits_value), "AdjCl": int(adjcl_value),
            "Exposure": args.exposure,
        }
        for spec_type, sig_vals, bw in [
            ("Raw", raw_no_rebin_significance.tolist(), None),
            ("Smoothed", smoothed_no_rebin_significance.tolist(), None),
        ]:
            _significance_list.append({**_hep_sig_base, "SpectrumType": spec_type, "BinMode": "NoRebin", "Energy": no_rebin_energy.tolist(), "EnergyUnit": "MeV", "Significance": sig_vals, "SignificanceUnit": r"\sigma", "BinWidth": bw, "BinWidthUnit": None, "ExposureUnit": "year"})
        if adaptive_energy.size > 0:
            for spec_type, sig_vals in [("Raw", raw_adaptive_significance), ("Smoothed", smoothed_adaptive_significance)]:
                _significance_list.append({**_hep_sig_base, "SpectrumType": spec_type, "BinMode": "AdaptiveRebin", "Energy": adaptive_energy.tolist(), "EnergyUnit": "MeV", "Significance": sig_vals.tolist(), "SignificanceUnit": r"\sigma", "BinWidth": adaptive_width.tolist(), "BinWidthUnit": "MeV", "ExposureUnit": "year"})

            s = scale * signal_tail_counts
            b = scale * background_tail_counts
            b_safe = np.maximum(b, 1e-12)
            with np.errstate(divide="ignore", invalid="ignore"):
                q_asimov = 2.0 * ((s + b_safe) * np.log1p(np.divide(s, b_safe)) - s)
            q_asimov = np.nan_to_num(q_asimov, nan=0.0, posinf=0.0, neginf=0.0)
            fisher = np.divide(s * s, b_safe, out=np.zeros_like(s, dtype=float), where=b_safe > 0)
            snr = np.divide(s, np.sqrt(b_safe), out=np.zeros_like(s, dtype=float), where=b_safe > 0)
            purity_arr = np.divide(s, (s + b_safe), out=np.zeros_like(s, dtype=float), where=(s + b_safe) > 0)

            if args.local_metric == "AsimovTS":
                to_save = np.divide(q_asimov, no_rebin_width, out=np.zeros_like(q_asimov), where=no_rebin_width > 0).tolist()
                binmode_name = "AsimovTS"
            elif args.local_metric == "Fisher":
                to_save = np.divide(fisher, no_rebin_width, out=np.zeros_like(fisher), where=no_rebin_width > 0).tolist()
                binmode_name = "Fisher"
            elif args.local_metric == "SNR":
                to_save = np.divide(snr, no_rebin_width, out=np.zeros_like(snr), where=no_rebin_width > 0).tolist()
                binmode_name = "SNR"
            else:
                to_save = purity_arr.tolist()
                binmode_name = "Purity"

            _hep_sig_unit = {
                "LocalDensity": r"\sigma \cdot MeV^{-1}",
                "AsimovTS": r"MeV^{-1}", "Fisher": r"MeV^{-1}", "SNR": r"MeV^{-1}", "Purity": "1",
            }
            for spec_type, sig_vals, bm in [
                ("Raw", raw_local_density.tolist(), "LocalDensity"),
                ("Smoothed", smooth_local_density.tolist(), "LocalDensity"),
                ("Smoothed", to_save, binmode_name),
            ]:
                _significance_list.append({**_hep_sig_base, "SpectrumType": spec_type, "BinMode": bm, "Energy": no_rebin_energy.tolist(), "EnergyUnit": "MeV", "Significance": sig_vals, "SignificanceUnit": _hep_sig_unit.get(bm, "1"), "BinWidth": no_rebin_width.tolist(), "BinWidthUnit": "MeV", "ExposureUnit": "year"})

        tail_slice = slice(no_rebin_start, no_rebin_start + len(no_rebin_energy))
        for rd in _hep_render:
            _is_sig = rd["component"] == "hep"
            for spec_type, cnt_arr in [("Raw", rd["counts"]), ("Smoothed", rd["smoothed_counts"])]:
                _counts_list.append({
                    "Config": config, "Name": name, "EnergyLabel": energy, "Analysis": "HEP",
                    "Geometry": info["GEOMETRY"],
                    "Component": rd["component"], "SpectrumType": spec_type,
                    "NHits": int(nhits_value), "OpHits": int(ophits_value), "AdjCl": int(adjcl_value),
                    "Exposure": args.exposure, "ExposureUnit": "year",
                    "Energy": no_rebin_energy.tolist(), "EnergyUnit": "MeV",
                    "Counts": (scale * cnt_arr[tail_slice]).tolist(), "CountsUnit": "counts",
                    "CountsError": None,
                    "Significance": (
                        raw_local_density.tolist() if (_is_sig and spec_type == "Raw")
                        else smooth_local_density.tolist() if (_is_sig and spec_type == "Smoothed")
                        else None
                    ),
                    "SignificanceLabel": "Asimov TS density" if _is_sig else None,
                })

        if args.pkl_label == "highest":
            save_df(
                pd.DataFrame(_counts_list), data_path, config=args.config[0], name=args.name[0],
                subfolder=args.folder.lower(), filename="HEP_Counts",
                rm=args.rewrite, debug=True,
            )
            save_df(
                pd.DataFrame(_significance_list), data_path, config=args.config[0], name=args.name[0],
                subfolder=args.folder.lower(), filename="HEP_Significance",
                rm=args.rewrite, debug=args.debug,
            )

    # ══════════════════════════════════════════════════════════════════════════
    # Sensitivity
    # ══════════════════════════════════════════════════════════════════════════
    elif args.analysis == "Sensitivity":
        best_cut_path = (
            f"{info['PATH']}/SENSITIVITY/{args.folder.lower()}"
            f"/{config}/{name}/{config}_{name}_highest_SENSITIVITY.pkl"
        )
        if not os.path.exists(best_cut_path):
            rprint(f"[yellow][WARNING][/yellow] Missing best-cut map for Sensitivity {config} {name}: {best_cut_path}")
            continue
        best_cut_map = pickle.load(open(best_cut_path, "rb"))
        try:
            ref_plot = best_cut_map[(config, name, energy)]
        except KeyError:
            rprint(f"[yellow][WARNING][/yellow] Missing best-cut entry for Sensitivity {config} {name} {energy}.")
            continue

        nhits_value = args.nhits if args.nhits is not None else int(ref_plot["NHits"])
        adjcl_value = args.adjcls if args.adjcls is not None else int(ref_plot["AdjCl"])
        ophits_value = args.ophits if args.ophits is not None else int(ref_plot["OpHits"])

        rebin_path = (
            f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/signal/{args.folder.lower()}"
            f"/SENSITIVITY/{config}/{name}/{config}_{name}_{energy}_Rebin.pkl"
        )
        if not os.path.exists(rebin_path):
            rprint(f"[yellow][WARNING][/yellow] Missing Rebin pkl for Sensitivity {config} {name} {energy}: {rebin_path}")
            continue

        plot_df = pd.read_pickle(rebin_path)
        background_samples = []
        for bkg, filepath in load_available_background_dataframes(str(root), "SENSITIVITY", args.folder, config, energy):
            plot_df = pd.concat([plot_df, pd.read_pickle(filepath)], ignore_index=True)
            background_samples.append(bkg)

        this_plot_df = plot_df.loc[
            (plot_df["NHits"] == nhits_value)
            * (plot_df["OpHits"] == ophits_value)
            * (plot_df["AdjCl"] == adjcl_value)
        ].copy()
        if this_plot_df.empty:
            rprint(f"[yellow][WARNING][/yellow] Missing histogram payload for Sensitivity {config} {name} {energy}.")
            continue

        energy_axis = np.asarray(this_plot_df["Energy"].values[0], dtype=float)
        if energy_axis.size > 1:
            half = 0.5 * (energy_axis[1] - energy_axis[0])
            edges = np.concatenate([
                [energy_axis[0] - half],
                0.5 * (energy_axis[:-1] + energy_axis[1:]),
                [energy_axis[-1] + half],
            ])
            bin_widths = np.diff(edges)
        else:
            bin_widths = np.ones(max(energy_axis.size, 1))

        scale = detector_mass * args.exposure

        sens_component_specs = [
            ("Solar", "Solar (osc)", "Osc", "Mean", "signal", "Signal", compare[1]),
        ]
        for bkg in background_samples:
            style = get_background_style(str(root), bkg)
            sens_component_specs.append(
                (bkg, style.get("label", bkg).title(), "Truth", "Mean", "background", "Background", style.get("color", "grey"))
            )

        signal_smoothed_counts = np.zeros(energy_axis.size, dtype=float)
        background_smoothed_counts = np.zeros(energy_axis.size, dtype=float)
        positive_count_values = []
        _sens_render = []

        for component, component_label, oscillation, mean_label, legend_group, legend_group_title, color in sens_component_specs:
            comp_df = this_plot_df.loc[
                (this_plot_df["Component"] == component)
                * (this_plot_df["Oscillation"] == oscillation)
                * (this_plot_df["Mean"] == mean_label)
            ].copy()
            if comp_df.empty:
                continue

            cd = _process_component_df(comp_df, component, smoothing_config, scale=scale)
            positive_count_values.extend(cd["raw_y"][cd["raw_y"] > 0])
            positive_count_values.extend(cd["smoothed_y"][cd["smoothed_y"] > 0])

            if component == "Solar":
                signal_smoothed_counts += cd["smoothed_counts"]
            else:
                background_smoothed_counts += cd["smoothed_counts"]

            _sens_render.append({
                "component_label": component_label, "legend_group": legend_group,
                "legend_group_title": legend_group_title, "color": color,
                "raw_y": cd["raw_y"], "smoothed_y": cd["smoothed_y"], "err_y": cd["err_y"],
            })

            for spec_type, cnt_arr, err_arr in [
                ("Raw", cd["counts"], cd["errors"]),
                ("Smoothed", cd["smoothed_counts"], cd["smoothed_errors"]),
            ]:
                _counts_list.append({
                    "Config": config, "Name": name, "EnergyLabel": energy, "Analysis": "Sensitivity",
                    "Geometry": info["GEOMETRY"],
                    "Component": component_label, "SpectrumType": spec_type,
                    "NHits": int(nhits_value), "OpHits": int(ophits_value), "AdjCl": int(adjcl_value),
                    "Exposure": args.exposure, "ExposureUnit": "year",
                    "Energy": energy_axis.tolist(), "EnergyUnit": "MeV",
                    "Counts": (scale * cnt_arr).tolist(), "CountsUnit": "counts",
                    "CountsError": (scale * err_arr).tolist(),
                    "Significance": None,
                    "SignificanceLabel": None,
                })

        s = scale * signal_smoothed_counts
        b = scale * background_smoothed_counts
        b_safe = np.maximum(b, 1e-12)
        with np.errstate(divide="ignore", invalid="ignore"):
            asimov_ts = 2.0 * ((s + b_safe) * np.log1p(np.divide(s, b_safe)) - s)
        asimov_ts = np.nan_to_num(asimov_ts, nan=0.0, posinf=0.0, neginf=0.0)
        asimov_density = np.divide(asimov_ts, bin_widths, out=np.zeros_like(asimov_ts), where=bin_widths > 0)
        purity_vals = np.divide(s, s + b, out=np.zeros_like(s, dtype=float), where=(s + b) > 0)
        lower_panel_max = float(np.max(asimov_density)) if asimov_density.size > 0 else 1.0

        figure_name = f"{energy}_Sensitivity_Significance_Exposure_{args.exposure:.0f}"
        if args.nhits is not None or args.ophits is not None or args.adjcls is not None:
            figure_name += f"_NHits{nhits_value}_OpHits{ophits_value}_AdjCl{adjcl_value}"

        def _sens_lower(fig):
            fig.add_trace(
                go.Bar(
                    x=energy_axis, y=asimov_density, width=bin_widths, name="Asimov TS density",
                    marker_color="rgba(31,119,180,0.35)", marker_line=dict(color="rgb(31,119,180)", width=1),
                    legend="legend2", legendgroup="proxy", legendgrouptitle=dict(text="Local Discovery Power"),
                    showlegend=True,
                ),
                row=2, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=energy_axis, y=purity_vals, mode="lines", name="Signal purity",
                    line=dict(color="rgb(230,159,0)", width=2, dash="dash"),
                    legend="legend3", legendgroup="purity_overlay",
                    legendgrouptitle=dict(text="Signal Purity"), showlegend=True,
                ),
                row=2, col=1, secondary_y=True,
            )
            return lower_panel_max

        _render_and_save(
            render_list=_sens_render, energy_axis=energy_axis, add_lower_fn=_sens_lower,
            subplot_title=f"{energy} NHits={nhits_value} OpHits={ophits_value} AdjCl={adjcl_value}",
            fig_title=f"Sensitivity Significance - {args.folder} - {config}",
            upper_y_title="Counts", lower_y_title="Asimov TS density",
            x_range=None, figure_name=figure_name,
            positive_count_values=positive_count_values, log_default_range=[-2, 10],
            save_path=save_path, config=config, save_name=name, folder=args.folder.lower(),
            rewrite=args.rewrite, debug_plot=args.plot,
            stacked_range=[False, True] if args.stacked else [False],
            zoom=args.zoom, threshold=args.threshold,
            secondary_y=True,
            legend_layout=dict(
                legend=dict(font=dict(size=12), x=0.75, y=0.99, xanchor="left", yanchor="top", bgcolor="rgba(255,255,255,0.7)"),
                legend2=dict(font=dict(size=12), x=0.01, y=0.30, xanchor="left", yanchor="top", bgcolor="rgba(255,255,255,0.7)"),
                legend3=dict(font=dict(size=12), x=0.75, y=0.26, xanchor="left", yanchor="top", bgcolor="rgba(255,255,255,0.7)"),
            ),
        )

        _significance_list.append({
            "Config": config, "Name": name, "EnergyLabel": energy, "Analysis": "Sensitivity",
            "Variable": "AsimovTS", "SpectrumType": "Smoothed", "BinMode": "PerBin",
            "NHits": int(nhits_value), "OpHits": int(ophits_value), "AdjCl": int(adjcl_value),
            "Exposure": args.exposure, "ExposureUnit": "year",
            "Energy": energy_axis.tolist(), "EnergyUnit": "MeV",
            "Significance": asimov_density.tolist(), "SignificanceUnit": r"MeV^{-1}",
            "BinWidth": bin_widths.tolist(), "BinWidthUnit": "MeV",
        })

        save_df(
            pd.DataFrame(_counts_list), data_path, config=config, name=name,
            subfolder=args.folder.lower(), filename="Sensitivity_Counts",
            rm=args.rewrite, debug=True,
        )
        save_df(
            pd.DataFrame(_significance_list), data_path, config=config, name=name,
            subfolder=args.folder.lower(), filename="Sensitivity_Significance",
            rm=args.rewrite, debug=args.debug,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Fiducial
    # ══════════════════════════════════════════════════════════════════════════
    elif args.analysis == "Fiducial":
        fid_data_root = f"{root}/data/solar/fiducial/{args.folder.lower()}"
        best_fiducials_path = f"{fid_data_root}/BestFiducials.json"
        best_fiducials = json.loads(open(best_fiducials_path).read()) if os.path.exists(best_fiducials_path) else {}

        signal_pkl = f"{analysis_info['PATH']}/FIDUCIAL/{args.folder.lower()}/{config}/{name}/{config}_{name}_{energy}_Fiducial_Scan.pkl"
        if not os.path.exists(signal_pkl):
            rprint(f"[yellow][WARNING][/yellow] Missing fiducial scan pkl: {signal_pkl}")
            continue
        df_list = [pd.read_pickle(signal_pkl)]
        needed_bkg = set()
        for an in args.fiducial_analyses:
            needed_bkg.update(get_fiducialization_config(str(root), an.upper()).get("background_components", []))
        for bkg_label in sorted(needed_bkg):
            bkg_pkl = f"{analysis_info['PATH']}/FIDUCIAL/{args.folder.lower()}/{config}/{bkg_label}/{config}_{bkg_label}_{energy}_Fiducial_Scan.pkl"
            if os.path.exists(bkg_pkl):
                df_list.append(pd.read_pickle(bkg_pkl))
        raw_df = pd.concat(df_list, ignore_index=True)
        plot_df = explode(raw_df, ["Counts", "Error+", "Error-", "Energy", "MCCounts"], debug=args.debug).copy()
        for col in ["Counts", "Error+", "Error-", "MCCounts"]:
            plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce").fillna(0.0)
        plot_df["Energy"] = pd.to_numeric(plot_df["Energy"], errors="coerce")

        for analysis_name in args.fiducial_analyses:
            analysis_key = analysis_name.upper()
            analysis_config = get_fiducialization_config(str(root), analysis_key)
            sc_fid = get_smoothing_config(str(root), analysis_name=analysis_key, dimensions="1d", stage="fiducial")
            signal_components = {str(c).lower() for c in analysis_config.get("signal_components", [])}

            merged_df, significance_df = _select_best_fiducial(plot_df, analysis_config, sc_fid, args.exposure)
            if significance_df.empty:
                rprint(f"[yellow][WARNING][/yellow] No fiducial significance points for {analysis_name} {energy}.")
                continue

            best_entry = best_fiducials.get(config, {}).get(analysis_key, {}).get(energy, {})
            if best_entry:
                best_x, best_y, best_z = int(best_entry.get("FiducialX", 0)), int(best_entry.get("FiducialY", 0)), int(best_entry.get("FiducialZ", 0))
            else:
                max_row = significance_df.loc[significance_df["SmoothedSignificance"].idxmax()]
                best_x, best_y, best_z = int(max_row["FiducializedX"]), int(max_row["FiducializedY"]), int(max_row["FiducializedZ"])

            if not args.plot:
                continue

            for fiducialx, fiducialy, fiducialz, fiducial_label in [(best_x, best_y, best_z, "Best"), (0, 0, 0, "No")]:
                this_plot = plot_df[
                    (plot_df["Component"].astype(str).str.lower() != "solar")
                    & (plot_df["FiducializedX"] == fiducialx)
                    & (plot_df["FiducializedY"] == fiducialy)
                    & (plot_df["FiducializedZ"] == fiducialz)
                ].copy()
                if this_plot.empty:
                    continue

                render_list, energy_axis = _build_fiducial_render_list(this_plot, signal_components, sc_fid, args.exposure)
                positive_count_values = [v for rd in render_list for arr in [rd["raw_y"], rd["smoothed_y"]] for v in arr[arr > 0]]
                payload = _get_fiducial_payload(merged_df, fiducialx, fiducialy, fiducialz, analysis_config, sc_fid, args.exposure)

                def _fid_lower(fig, _payload=payload, _ac=analysis_config):
                    max_sig = 1.0
                    if _payload is not None:
                        max_sig = max(max_sig, float(np.max(_payload["RawSignificance"])), float(np.max(_payload["SmoothedSignificance"])))
                        for label, sig, dash in [("Raw", _payload["RawSignificance"], "dot"), ("Smooth", _payload["SmoothedSignificance"], "solid")]:
                            total = _payload["RawTotal"] if label == "Raw" else _payload["SmoothedTotal"]
                            fig.add_trace(
                                go.Scatter(
                                    x=_payload["Energy"], y=sig, mode="lines+markers", line_shape="hvh",
                                    legend="legend2", legendgroup="significance",
                                    legendgrouptitle=dict(text="Significance", font=dict(size=16)),
                                    name=f"{label} {total:.1f} (σ)",
                                    line=dict(color="rgb(66,66,66)", dash=dash),
                                    marker=dict(size=6, color="rgb(66,66,66)"),
                                ),
                                row=2, col=1,
                            )
                    emin, emax = _ac.get("energy_min"), _ac.get("energy_max")
                    if emin is not None:
                        fig.add_vline(emin, line=dict(color="grey", dash="dash"))
                    if emax is not None:
                        fig.add_vline(emax, line=dict(color="grey", dash="dash"))
                    return max_sig

                _render_and_save(
                    render_list=render_list, energy_axis=energy_axis, add_lower_fn=_fid_lower,
                    add_upper_fn=lambda fig, st, _rl=render_list: _add_fiducial_upper(fig, _rl, st),
                    subplot_title=f"{analysis_name} {energy} — X={fiducialx}cm Y={fiducialy}cm Z={fiducialz}cm",
                    fig_title=f"{analysis_name} {energy}",
                    upper_y_title=f"Counts ({args.exposure}·kT·year·MeV)⁻¹",
                    lower_y_title="Significance (σ)",
                    x_range=[-2, 32], figure_name=f"{energy}_{analysis_key}_{fiducial_label}Fiducial_Significance",
                    positive_count_values=positive_count_values, log_default_range=[-2, 10],
                    save_path=save_path, config=config, save_name=name, folder=args.folder.lower(),
                    rewrite=args.rewrite, debug_plot=args.plot,
                    stacked_range=[args.stacked],
                    zoom=args.zoom, threshold=None,
                    upper_y_extra={"tickformat": ".0e"},
                    lower_y_extra={"tickformat": ".1f"},
                    lower_y_default_range=[0, 6],
                    legend_layout={"legend2": dict(y=0.1, x=0.74, font=dict(size=12), bgcolor="rgba(255,255,255,0.7)")},
                )
