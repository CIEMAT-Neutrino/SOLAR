import os
import sys
from typing import Dict, List, Optional, Tuple, cast

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *

save_path = f"{root}/images/solar/fiducial"
data_path = f"{root}/data/solar/fiducial"

ANALYSIS_CHOICES = ["DayNight", "HEP", "Sensitivity"]
GROUP_COLUMNS = ["Energy", "FiducializedX", "FiducializedY", "FiducializedZ"]
FIDUCIAL_COLUMNS = ["FiducializedX", "FiducializedY", "FiducializedZ"]


def combine_components(df: pd.DataFrame, components: List[str]) -> pd.DataFrame:
    component_df = cast(pd.DataFrame, df.loc[df["Component"].isin(components)].copy())
    if component_df.empty:
        return pd.DataFrame(columns=GROUP_COLUMNS + ["MCCounts", "Counts", "Error+", "Error-"])
    return (
        component_df.groupby(GROUP_COLUMNS)
        .agg({
            "MCCounts": "sum",
            "Counts": "sum",
            "Error+": lambda x: float(np.sqrt(np.sum(np.square(np.asarray(x, dtype=float))))),
            "Error-": lambda x: float(np.sqrt(np.sum(np.square(np.asarray(x, dtype=float))))),
        })
        .reset_index()
    )


def aggregate_significance(significance: np.ndarray, mode: str) -> float:
    values = np.nan_to_num(np.asarray(significance, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if str(mode).lower() == "sum":
        return float(np.sum(values))
    return float(np.sqrt(np.sum(np.square(values))))


def build_significance_payload(
    merged_df: pd.DataFrame,
    significance_type: str,
    combine_mode: str,
    smoothing_config: dict,
    exposure: float,
    energy_min=None,
    energy_max=None,
) -> Optional[Dict]:
    if merged_df.empty:
        return None
    ordered = merged_df.sort_values("Energy").copy()
    energy = ordered["Energy"].to_numpy(dtype=float)
    signal_counts = np.nan_to_num(ordered["CountsSignal"].to_numpy(dtype=float), nan=0.0)
    background_counts = np.nan_to_num(ordered["CountsBackground"].to_numpy(dtype=float), nan=0.0)
    signal_errors = np.nan_to_num(ordered["Error+Signal"].to_numpy(dtype=float), nan=0.0)
    background_errors = np.nan_to_num(ordered["Error+Background"].to_numpy(dtype=float), nan=0.0)
    signal_mc = np.nan_to_num(ordered["MCCountsSignal"].to_numpy(dtype=float), nan=0.0)
    background_mc = np.nan_to_num(ordered["MCCountsBackground"].to_numpy(dtype=float), nan=0.0)

    smoothed_signal_counts = smooth_histogram_with_config(signal_counts, smoothing_config)
    smoothed_background_counts = smooth_histogram_with_config(background_counts, smoothing_config)
    smoothed_signal_errors = smooth_histogram_errors(
        signal_errors, smoothing_config, counts=signal_counts, mc_counts=signal_mc
    )
    smoothed_background_errors = smooth_histogram_errors(
        background_errors, smoothing_config, counts=background_counts, mc_counts=background_mc
    )

    window_mask = np.ones_like(energy, dtype=bool)
    if energy_min is not None:
        window_mask = window_mask & (energy >= float(energy_min))
    if energy_max is not None:
        window_mask = window_mask & (energy <= float(energy_max))

    signal_counts_eval = signal_counts[window_mask]
    background_counts_eval = background_counts[window_mask]
    signal_errors_eval = signal_errors[window_mask]
    background_errors_eval = background_errors[window_mask]
    smoothed_signal_counts_eval = smoothed_signal_counts[window_mask]
    smoothed_background_counts_eval = smoothed_background_counts[window_mask]
    smoothed_signal_errors_eval = smoothed_signal_errors[window_mask]
    smoothed_background_errors_eval = smoothed_background_errors[window_mask]

    significance_kind = str(significance_type).lower()
    if significance_kind == "asimov":
        raw_significance_eval = evaluate_significance(
            exposure * signal_counts_eval,
            exposure * background_counts_eval,
            background_uncertainty=exposure * background_errors_eval,
            type=significance_kind,
        )
        smoothed_significance_eval = evaluate_significance(
            exposure * smoothed_signal_counts_eval,
            exposure * smoothed_background_counts_eval,
            background_uncertainty=exposure * smoothed_background_errors_eval,
            type=significance_kind,
        )
    else:
        raw_significance_eval = evaluate_significance(
            exposure * signal_counts_eval,
            exposure * background_counts_eval,
            exposure * signal_errors_eval,
            exposure * background_errors_eval,
            type=significance_kind,
        )
        smoothed_significance_eval = evaluate_significance(
            exposure * smoothed_signal_counts_eval,
            exposure * smoothed_background_counts_eval,
            exposure * smoothed_signal_errors_eval,
            exposure * smoothed_background_errors_eval,
            type=significance_kind,
        )
    raw_significance_eval = np.nan_to_num(raw_significance_eval, nan=0.0, posinf=0.0, neginf=0.0)
    smoothed_significance_eval = np.nan_to_num(smoothed_significance_eval, nan=0.0, posinf=0.0, neginf=0.0)

    raw_significance = np.zeros_like(energy, dtype=float)
    smoothed_significance = np.zeros_like(energy, dtype=float)
    raw_significance[window_mask] = raw_significance_eval
    smoothed_significance[window_mask] = smoothed_significance_eval

    return {
        "Energy": energy,
        "RawSignal": exposure * signal_counts,
        "RawBackground": exposure * background_counts,
        "SmoothedSignal": exposure * smoothed_signal_counts,
        "SmoothedBackground": exposure * smoothed_background_counts,
        "RawSignificance": raw_significance,
        "SmoothedSignificance": smoothed_significance,
        "RawTotal": aggregate_significance(raw_significance, combine_mode),
        "SmoothedTotal": aggregate_significance(smoothed_significance, combine_mode),
    }


def select_best_fiducial(plot_df: pd.DataFrame, analysis_config: Dict, smoothing_config: Dict, exposure: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    signal_df = combine_components(
        plot_df,
        analysis_config.get("signal_components", []),
    )
    background_df = combine_components(
        plot_df,
        analysis_config.get("background_components", []),
    )
    merged_df = pd.merge(
        signal_df,
        background_df,
        on=GROUP_COLUMNS,
        how="outer",
        suffixes=("Signal", "Background"),
    )
    for column in [
        "MCCountsSignal",
        "CountsSignal",
        "Error+Signal",
        "Error-Signal",
        "MCCountsBackground",
        "CountsBackground",
        "Error+Background",
        "Error-Background",
    ]:
        if column not in merged_df.columns:
            merged_df[column] = 0.0
    merged_df = merged_df.fillna(0.0)

    significance_rows = []
    for fiducial_values, group in merged_df.groupby(FIDUCIAL_COLUMNS):
        # Pandas type stubs may infer Series here; normalize to dataframe for static checks.
        group_df = group if isinstance(group, pd.DataFrame) else group.to_frame().T
        payload = build_significance_payload(
            group_df,
            analysis_config.get("significance_type", "gaussian"),
            analysis_config.get("combine_mode", "quadrature"),
            smoothing_config,
            exposure,
            analysis_config.get("energy_min"),
            analysis_config.get("energy_max"),
        )
        if payload is None:
            continue
        significance_rows.append({
            "FiducializedX": int(fiducial_values[0]),
            "FiducializedY": int(fiducial_values[1]),
            "FiducializedZ": int(fiducial_values[2]),
            "RawSignificance": payload["RawTotal"],
            "SmoothedSignificance": payload["SmoothedTotal"],
        })

    return merged_df, pd.DataFrame(significance_rows)


def get_payload_for_fiducial(
    merged_df: pd.DataFrame,
    fiducialx: int,
    fiducialy: int,
    fiducialz: int,
    analysis_config: dict,
    smoothing_config: dict,
    exposure: float,
) -> Optional[Dict]:
    mask = (
        (merged_df["FiducializedX"] == fiducialx)
        & (merged_df["FiducializedY"] == fiducialy)
        & (merged_df["FiducializedZ"] == fiducialz)
    )
    fiducial_df = cast(pd.DataFrame, merged_df.loc[mask].copy())
    return build_significance_payload(
        fiducial_df,
        analysis_config.get("significance_type", "gaussian"),
        analysis_config.get("combine_mode", "quadrature"),
        smoothing_config,
        exposure,
        analysis_config.get("energy_min"),
        analysis_config.get("energy_max"),
    )


def get_required_background_samples(root_path: str, analyses: List[str]) -> List[str]:
    samples = set()
    for analysis_name in analyses:
        analysis_key = str(analysis_name).upper()
        analysis_config = get_fiducialization_config(root_path, analysis_key)
        for component in analysis_config.get("background_components", []):
            samples.add(str(component).lower())
    if not samples:
        samples = {str(sample).lower() for sample in get_background_samples(root_path)}
    return sorted(samples)


parser = argparse.ArgumentParser(description="Render fiducialized significance plots")
parser.add_argument("--config", type=str, help="The configuration to load", default="hd_1x2x6_centralAPA")
parser.add_argument("--name", type=str, help="The name of the configuration", default="marley")
parser.add_argument("--folder", type=str, help="The name of the background folder", choices=["Reduced", "Truncated", "Nominal"], default="Nominal")
parser.add_argument("--analysis", nargs="+", type=str, help="The analyses to plot fiducials for", choices=ANALYSIS_CHOICES, default=ANALYSIS_CHOICES)
parser.add_argument("--energy", nargs="+", type=str, help="The energy for the analysis", choices=["SignalParticleK", "ClusterEnergy", "TotalEnergy", "SelectedEnergy", "SolarEnergy"], default=["SolarEnergy"])
parser.add_argument("--exposure", type=float, help="The exposure in kT*year", default=100)
parser.add_argument("--zoom", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--stacked", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()
config = args.config
name = args.name
configs = {config: [name]}

for path in [save_path, data_path]:
    if not os.path.exists(f"{path}/{args.folder.lower()}"):
        os.makedirs(f"{path}/{args.folder.lower()}")

filename = f"{data_path}/{args.folder.lower()}/BestFiducials.json"
best_fiducials = {}
if os.path.exists(filename):
    best_fiducials = json.loads(open(filename, "r").read())

background_samples = get_required_background_samples(str(root), args.analysis)

for config in configs:
    for name, energy_label in product(configs[config], args.energy):
        df_list = []
        signal_df = pd.read_pickle(
            f"{root}/data/solar/fiducial/{args.folder.lower()}/{config}/{name}/{config}_{name}_{energy_label}_Fiducial_Scan.pkl"
        )
        df_list.append(signal_df)
        for bkg_label in background_samples:
            filepath = f"{root}/data/solar/fiducial/{args.folder.lower()}/{config}/{bkg_label}/{config}_{bkg_label}_{energy_label}_Fiducial_Scan.pkl"
            if not os.path.exists(filepath):
                continue
            bkg_df = pd.read_pickle(filepath)
            df_list.append(bkg_df)

        raw_df = pd.concat(df_list, ignore_index=True)
        plot_df = explode(raw_df, ["Counts", "Error+", "Error-", "Energy", "MCCounts"], debug=args.debug).copy()
        plot_df["Counts"] = pd.to_numeric(plot_df["Counts"], errors="coerce").fillna(0.0)
        plot_df["Error+"] = pd.to_numeric(plot_df["Error+"], errors="coerce").fillna(0.0)
        plot_df["Error-"] = pd.to_numeric(plot_df["Error-"], errors="coerce").fillna(0.0)
        plot_df["Energy"] = pd.to_numeric(plot_df["Energy"], errors="coerce")
        plot_df["MCCounts"] = pd.to_numeric(plot_df["MCCounts"], errors="coerce").fillna(0.0)

        for analysis_name in args.analysis:
            analysis_key = analysis_name.upper()
            analysis_config = get_fiducialization_config(str(root), analysis_key)
            smoothing_config = get_smoothing_config(
                str(root), analysis_name=analysis_key, dimensions="1d", stage="fiducial"
            )

            merged_df, significance_df = select_best_fiducial(
                plot_df, analysis_config, smoothing_config, args.exposure
            )
            if significance_df.empty:
                rprint(f"[yellow]No fiducial significance points found for {analysis_name} {energy_label}[/yellow]")
                continue

            best_entry = (
                best_fiducials.get(config, {})
                .get(analysis_key, {})
                .get(energy_label, {})
            )
            if best_entry:
                best_x = int(best_entry.get("FiducialX", 0))
                best_y = int(best_entry.get("FiducialY", 0))
                best_z = int(best_entry.get("FiducialZ", 0))
            else:
                max_row = significance_df.loc[significance_df["SmoothedSignificance"].idxmax()]
                best_x = int(max_row["FiducializedX"])
                best_y = int(max_row["FiducializedY"])
                best_z = int(max_row["FiducializedZ"])

            if not args.plot:
                rprint(
                    f"[cyan]Skipping figure generation for {analysis_name} {energy_label} because --no-plot was requested.[/cyan]"
                )
                continue

            fiducial_configs = [
                (best_x, best_y, best_z, "Best"),
                (0, 0, 0, "No"),
            ]
            max_significance = 1.0
            for fiducialx, fiducialy, fiducialz, fiducial_label in fiducial_configs:
                this_plot = plot_df[
                    (plot_df["Component"].astype(str).str.lower() != "solar")
                    & (plot_df["FiducializedX"] == fiducialx)
                    & (plot_df["FiducializedY"] == fiducialy)
                    & (plot_df["FiducializedZ"] == fiducialz)
                ].copy()
                if this_plot.empty:
                    continue

                positive_count_values = []

                fig = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0,
                    row_heights=[0.7, 0.3],
                )
                component_color = {
                    "gamma": "black",
                    "neutron": "rgb(15,133,84)",
                    "radiological": "rgb(120, 94, 240)",
                    "8B": "rgb(225,124,5)",
                    "hep": "rgb(204,80,62)",
                }
                signal_components = {
                    str(component).lower()
                    for component in analysis_config.get("signal_components", [])
                }

                for component in this_plot["Component"].unique():
                    component_data = this_plot[this_plot["Component"] == component].sort_values("Energy")
                    x = component_data["Energy"].to_numpy(dtype=float)
                    y = component_data["Counts"].to_numpy(dtype=float)
                    y_error_plus = component_data["Error+"].to_numpy(dtype=float)
                    y_error_minus = component_data["Error-"].to_numpy(dtype=float)
                    mc_counts = (
                        component_data["MCCounts"].to_numpy(dtype=float)
                        if "MCCounts" in component_data.columns
                        else None
                    )
                    component_smoothing_config = get_component_smoothing_config(
                        smoothing_config, component
                    )
                    smoothed_y = smooth_histogram_with_config(y, component_smoothing_config)
                    smoothed_error_plus = smooth_histogram_errors(
                        y_error_plus,
                        component_smoothing_config,
                        counts=y,
                        mc_counts=mc_counts if mc_counts is not None else None,
                    )
                    smoothed_error_minus = smooth_histogram_errors(
                        y_error_minus,
                        component_smoothing_config,
                        counts=y,
                        mc_counts=mc_counts if mc_counts is not None else None,
                    )
                    positive_count_values.extend((args.exposure * y)[y > 0])
                    positive_count_values.extend((args.exposure * smoothed_y)[smoothed_y > 0])
                    is_signal_component = str(component).lower() in signal_components
                    legend_group = 0 if is_signal_component else 1
                    legend_group_title = "Signal" if is_signal_component else "Background"
                    if args.stacked:
                        fig.add_trace(
                            go.Bar(
                                x=x,
                                y=args.exposure * y,
                                name=component,
                                marker_color=component_color.get(component, "grey"),
                                legendgroup="component",
                                legendgrouptitle=dict(text="Component", font=dict(size=16)),
                            ),
                            row=1,
                            col=1,
                        )
                    else:
                        fig.add_trace(
                            go.Scatter(
                                x=x,
                                y=args.exposure * y,
                                mode="lines",
                                line_shape="hvh",
                                line=dict(
                                    color=component_color.get(component, "grey"),
                                    width=2,
                                    dash="dot",
                                ),
                                opacity=0.45,
                                legendgroup=legend_group,
                                legendgrouptitle=dict(text=legend_group_title, font=dict(size=16)),
                                name=component,
                                showlegend=False,
                            ),
                            row=1,
                            col=1,
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=x,
                                y=args.exposure * smoothed_y,
                                error_y=dict(
                                    type="data",
                                    symmetric=False,
                                    array=args.exposure * smoothed_error_plus,
                                    arrayminus=args.exposure * smoothed_error_minus,
                                    visible=True,
                                ),
                                mode="lines+markers",
                                line_shape="hvh",
                                legendgroup=legend_group,
                                legendgrouptitle=dict(text=legend_group_title, font=dict(size=16)),
                                name=f"{component} {args.exposure * np.sum(y):.1e}",
                                line=dict(color=component_color.get(component, "grey"), width=3),
                                marker=dict(size=6, color=component_color.get(component, "grey")),
                                showlegend=True,
                            ),
                            row=1,
                            col=1,
                        )

                if not args.stacked:
                    fig.add_trace(
                        go.Scatter(
                            x=[None],
                            y=[None],
                            mode="lines",
                            name="Raw",
                            line=dict(color="gray", width=2, dash="dot"),
                            legendgroup="linestyle",
                            legendgrouptitle=dict(text="Histogram", font=dict(size=16)),
                            showlegend=True,
                        ),
                        row=1,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=[None],
                            y=[None],
                            mode="lines",
                            name="Smoothed",
                            line=dict(color="gray", width=3, dash="solid"),
                            legendgroup="linestyle",
                            legendgrouptitle=dict(text="Histogram", font=dict(size=16)),
                            showlegend=True,
                        ),
                        row=1,
                        col=1,
                    )

                payload = get_payload_for_fiducial(
                    merged_df,
                    fiducialx,
                    fiducialy,
                    fiducialz,
                    analysis_config,
                    smoothing_config,
                    args.exposure,
                )
                if payload is not None:
                    max_significance = max(
                        max_significance,
                        float(np.max(payload["RawSignificance"])),
                        float(np.max(payload["SmoothedSignificance"])),
                    )
                    for label, significance, dash, total in [
                        ("Raw", payload["RawSignificance"], "dot", payload["RawTotal"]),
                        ("Smooth", payload["SmoothedSignificance"], "solid", payload["SmoothedTotal"]),
                    ]:
                        fig.add_trace(
                            go.Scatter(
                                x=payload["Energy"],
                                y=significance,
                                mode="lines+markers",
                                line_shape="hvh",
                                legend="legend2",
                                legendgroup="significance",
                                legendgrouptitle=dict(text="Significance", font=dict(size=16)),
                                name=f"{label} {total:.1f} (σ)",
                                line=dict(color="rgb(66,66,66)", dash=dash),
                                marker=dict(size=6, color="rgb(66,66,66)"),
                            ),
                            row=2,
                            col=1,
                        )

                energy_min = analysis_config.get("energy_min")
                if energy_min is not None:
                    fig.add_vline(energy_min, line=dict(color="grey", dash="dash"))
                if args.stacked:
                    fig.update_layout(barmode="stack")

                fig.update_layout(
                    title=(
                        f"{analysis_name} {energy_label} Significance<br>"
                        f"Fiducial: X={fiducialx}cm, Y={fiducialy}cm, Z={fiducialz}cm"
                    ),
                    showlegend=True,
                )
                fig = format_coustom_plotly(
                    fig,
                    tickformat=(".0f", ".1e"),
                    figsize=(800, 600),
                )
                fig.update_layout(
                    legend2=dict(y=0.1, x=0.74, font=dict(size=12), bgcolor="rgba(255,255,255,0.7)"),
                )

                fig.update_xaxes(range=[8, 26])
                fig.update_xaxes(title_text="Reconstructed Energy (MeV)", row=2, col=1)
                fig.update_yaxes(
                    type="log",
                    range=[
                        np.floor(np.log10(max(min(positive_count_values), 1e-6))),
                        np.ceil(np.log10(max(positive_count_values))),
                    ]
                    if args.zoom and positive_count_values
                    else [-2, 10],
                    tickformat=".0e",
                    title_text=f"Counts ({args.exposure}·kT·year·MeV)⁻¹",
                    row=1,
                    col=1,
                )
                fig.update_yaxes(
                    type="linear",
                    range=[0, max(1.0, 1.1 * max_significance)] if args.zoom else [0, 6],
                    tickformat=".1f",
                    title_text="Significance (σ)",
                    row=2,
                    col=1,
                )

                save_figure(
                    fig,
                    path=f"{save_path}",
                    config=config,
                    name=None,
                    subfolder=args.folder.lower(),
                    filename=f"{energy_label}_{analysis_key}_{fiducial_label}Fiducial_Significance" + ("_Stacked" if args.stacked else ""),
                    rm=args.rewrite,
                    debug=args.plot,
                )
