import os
import sys
from typing import Dict, List, Optional, Set, Tuple

# Add the absolute path to the lib directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *

save_path = f"{root}/images/solar/fiducial"
data_path = f"{root}/data/solar/fiducial"

ANALYSIS_CHOICES = ["DayNight", "HEP", "Sensitivity"]
GROUP_COLUMNS = ["Energy", "FiducializedX", "FiducializedY", "FiducializedZ"]
FIDUCIAL_COLUMNS = ["FiducializedX", "FiducializedY", "FiducializedZ"]


def _deep_merge_dict(base: dict, update: dict) -> dict:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _merge_and_write_json(path: str, payload: dict) -> None:
    existing: dict = {}
    if os.path.exists(path):
        with open(path, "r") as f_read:
            existing = json.load(f_read)
    merged = _deep_merge_dict(existing, payload)
    if os.path.exists(path):
        os.remove(path)
    with open(path, "w") as f_write:
        json.dump(merged, f_write, indent=4)


def combine_components(df: pd.DataFrame, components: List[str]) -> pd.DataFrame:
    component_df = df[df["Component"].isin(components)].copy()
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

    if significance_kind == "gaussian":
        # Correct global formula: S_total / sqrt(B_total + Σσ_s² + Σσ_b²)
        # Per-bin quadrature overestimates significance when bins differ in S/B ratio.
        def _gaussian_global(s, b, se, be):
            s_sum = float(np.sum(s))
            denom_sq = float(np.sum(b) + np.sum(np.square(se)) + np.sum(np.square(be)))
            return float(s_sum / np.sqrt(denom_sq)) if denom_sq > 0 else 0.0

        raw_total = _gaussian_global(
            exposure * signal_counts_eval,
            exposure * background_counts_eval,
            exposure * signal_errors_eval,
            exposure * background_errors_eval,
        )
        smoothed_total = _gaussian_global(
            exposure * smoothed_signal_counts_eval,
            exposure * smoothed_background_counts_eval,
            exposure * smoothed_signal_errors_eval,
            exposure * smoothed_background_errors_eval,
        )
    else:
        # Asimov: q0 is additive over independent bins → sqrt(Σz_i²) is exact.
        raw_total = aggregate_significance(raw_significance, combine_mode)
        smoothed_total = aggregate_significance(smoothed_significance, combine_mode)

    return {
        "Energy": energy,
        "RawSignal": exposure * signal_counts,
        "RawBackground": exposure * background_counts,
        "SmoothedSignal": exposure * smoothed_signal_counts,
        "SmoothedBackground": exposure * smoothed_background_counts,
        "RawSignificance": raw_significance,
        "SmoothedSignificance": smoothed_significance,
        "RawTotal": raw_total,
        "SmoothedTotal": smoothed_total,
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
        payload = build_significance_payload(
            group,
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


def get_essential_background_components(root_path: str, components: List[str]) -> Set[str]:
    background_cfg = get_background_config(root_path)
    essential_map = {
        str(component).lower(): bool(is_essential)
        for component, is_essential in background_cfg.get("ESSENTIAL", {}).items()
    }
    return {
        str(component)
        for component in components
        if essential_map.get(str(component).lower(), False)
    }


def apply_fiducial_mc_threshold(
    plot_df: pd.DataFrame,
    analysis_config: Dict,
    mc_threshold: int,
    root_path: str,
) -> pd.DataFrame:
    background_components = list(analysis_config.get("background_components", []))
    essential_components = get_essential_background_components(root_path, background_components)
    if not essential_components:
        return plot_df

    filtered = plot_df.copy()
    energy_min = analysis_config.get("energy_min")
    energy_max = analysis_config.get("energy_max")
    if energy_min is not None:
        filtered = filtered.loc[filtered["Energy"] >= float(energy_min)]
    if energy_max is not None:
        filtered = filtered.loc[filtered["Energy"] <= float(energy_max)]

    component_mc = (
        filtered.loc[filtered["Component"].isin(essential_components)]
        .groupby(FIDUCIAL_COLUMNS + ["Component"])["MCCounts"]
        .sum()
        .reset_index()
    )
    if component_mc.empty:
        return filtered.iloc[0:0]

    component_mc["Pass"] = component_mc["MCCounts"] >= float(mc_threshold)
    pass_counts = (
        component_mc.loc[component_mc["Pass"]]
        .groupby(FIDUCIAL_COLUMNS)["Component"]
        .nunique()
        .reset_index(name="PassComponents")
    )
    pass_counts = pass_counts.loc[
        pass_counts["PassComponents"] >= len(essential_components),
        FIDUCIAL_COLUMNS,
    ]
    if pass_counts.empty:
        return filtered.iloc[0:0]

    result = plot_df.merge(pass_counts, on=FIDUCIAL_COLUMNS, how="inner")

    # Always include the no-fiducial baseline (0,0,0) regardless of MC threshold.
    # It is the reference point: any tighter fiducial must improve over it.
    # Configs with suppressed backgrounds have fewer MC events by physics, not by
    # poor statistics — excluding (0,0,0) forces the optimizer into spuriously tight cuts.
    no_fid = plot_df.loc[
        (plot_df["FiducializedX"] == 0)
        & (plot_df["FiducializedY"] == 0)
        & (plot_df["FiducializedZ"] == 0)
    ]
    if not no_fid.empty:
        result = pd.concat([result, no_fid]).drop_duplicates(
            subset=GROUP_COLUMNS + ["Component"], keep="first"
        )
    return result


parser = argparse.ArgumentParser(description="Plot the energy distribution of the particles")
parser.add_argument("--config", type=str, help="The configuration to load", default="hd_1x2x6_centralAPA")
parser.add_argument("--name", type=str, help="The name of the configuration", default="marley")
parser.add_argument("--folder", type=str, help="The name of the background folder", choices=["Reduced", "Truncated", "Nominal"], default="Nominal")
parser.add_argument("--analysis", nargs="+", type=str, help="The analyses to optimize fiducials for", choices=ANALYSIS_CHOICES, default=ANALYSIS_CHOICES)
parser.add_argument("--energy", nargs="+", type=str, help="The energy for the analysis", choices=["SignalParticleK", "ClusterEnergy", "TotalEnergy", "SelectedEnergy", "SolarEnergy"], default=["SignalParticleK", "ClusterEnergy", "TotalEnergy", "SelectedEnergy", "SolarEnergy"])
parser.add_argument("--exposure", type=float, help="The exposure in kT·year", default=100)
parser.add_argument("--stacked", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument(
    "--mc_threshold",
    type=float,
    default=get_analysis_threshold(str(root), "FIDUCIALIZATION", stage="MC", fallback=0.0),
    help="Minimum summed MCCounts required for every essential background component in fiducial selection",
)
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
if os.path.exists(filename):
    best_fiducials = json.loads(open(filename, "r").read())
else:
    best_fiducials = {}

for config in configs:
    for name, energy_label in product(configs[config], args.energy):
        df_list = []
        signal_df = pd.read_pickle(
            f"{root}/data/solar/fiducial/{args.folder.lower()}/{config}/{name}/{config}_{name}_{energy_label}_Fiducial_Scan.pkl"
        )
        df_list.append(signal_df)
        for bkg_label in get_background_samples(str(root)):
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

        best_fiducials.setdefault(config, {})

        for analysis_name in args.analysis:
            analysis_key = analysis_name.upper()
            analysis_config = get_fiducialization_config(str(root), analysis_key)
            _analysis_info = load_analysis_info(str(root))
            _sig_ref = str(
                _analysis_info.get("BEST_SIGMA_SIGNIFICANCE_REFERENCE", {}).get(analysis_key, "")
            ).lower()
            if _sig_ref in {"asimov", "gaussian"} and _sig_ref != analysis_config.get("significance_type", "gaussian").lower():
                rprint(
                    f"[yellow][WARNING][/yellow] Overriding {analysis_key} fiducial significance_type "
                    f"({analysis_config.get('significance_type', 'gaussian')!r}) → ({_sig_ref!r}) "
                    f"to match BEST_SIGMA_SIGNIFICANCE_REFERENCE."
                )
                analysis_config = dict(analysis_config)
                analysis_config["significance_type"] = _sig_ref
            gated_plot_df = apply_fiducial_mc_threshold(
                plot_df,
                analysis_config,
                args.mc_threshold,
                str(root),
            )
            smoothing_config = get_smoothing_config(
                str(root), analysis_name=analysis_key, dimensions="1d", stage="fiducial"
            )

            merged_df, significance_df = select_best_fiducial(
                gated_plot_df, analysis_config, smoothing_config, args.exposure
            )
            if significance_df.empty:
                rprint(
                    f"[yellow]No fiducial significance points found for {analysis_name} {energy_label} "
                    f"after essential-component MC threshold {args.mc_threshold}[/yellow]"
                )
                continue

            max_row = significance_df.loc[significance_df["SmoothedSignificance"].idxmax()]
            no_fid_row = significance_df.loc[
                (significance_df["FiducializedX"] == 0)
                & (significance_df["FiducializedY"] == 0)
                & (significance_df["FiducializedZ"] == 0)
            ]
            no_fid_smoothed = (
                float(no_fid_row.iloc[0]["SmoothedSignificance"])
                if not no_fid_row.empty
                else None
            )
            rprint(
                f"{analysis_name} {energy_label}: best smoothed significance {max_row['SmoothedSignificance']:.2f} at X={int(max_row['FiducializedX'])} Y={int(max_row['FiducializedY'])} Z={int(max_row['FiducializedZ'])}"
            )

            best_fiducials[config].setdefault(analysis_key, {})[energy_label] = {
                "FiducialX": int(max_row["FiducializedX"]),
                "FiducialY": int(max_row["FiducializedY"]),
                "FiducialZ": int(max_row["FiducializedZ"]),
                "RawSignificance": float(max_row["RawSignificance"]),
                "SmoothedSignificance": float(max_row["SmoothedSignificance"]),
                "NoFiducialSignificance": no_fid_smoothed,
                "BestFiducialSignificance": float(max_row["SmoothedSignificance"]),
                "SignalComponents": list(analysis_config.get("signal_components", [])),
                "BackgroundComponents": list(analysis_config.get("background_components", [])),
                "MCThreshold": int(args.mc_threshold),
                "EnergyMin": analysis_config.get("energy_min"),
                "EnergyMax": analysis_config.get("energy_max"),
                "SignificanceType": analysis_config.get("significance_type", "gaussian"),
                **smoothing_metadata(smoothing_config),
            }

            # Per-energy-band optimization: find the best fiducial independently for
            # each band. Downstream masking applies the band-specific cut per event.
            band_results = []
            for band in analysis_config.get("energy_bands", []):
                band_config = dict(analysis_config)
                band_config["energy_min"] = band["energy_min"]
                band_config["energy_max"] = band["energy_max"]
                _, band_sig_df = select_best_fiducial(
                    gated_plot_df, band_config, smoothing_config, args.exposure
                )
                if band_sig_df.empty:
                    continue
                band_best = band_sig_df.loc[band_sig_df["SmoothedSignificance"].idxmax()]
                band_label = band.get("label", f"{band['energy_min']}-{band['energy_max']}")
                rprint(
                    f"  [{band_label}] {band['energy_min']}-{band['energy_max']} MeV: "
                    f"significance {band_best['SmoothedSignificance']:.2f} at "
                    f"X={int(band_best['FiducializedX'])} "
                    f"Y={int(band_best['FiducializedY'])} "
                    f"Z={int(band_best['FiducializedZ'])}"
                )
                band_results.append({
                    "label": band_label,
                    "energy_min": band["energy_min"],
                    "energy_max": band["energy_max"],
                    "FiducialX": int(band_best["FiducializedX"]),
                    "FiducialY": int(band_best["FiducializedY"]),
                    "FiducialZ": int(band_best["FiducializedZ"]),
                    "SmoothedSignificance": float(band_best["SmoothedSignificance"]),
                })
            if band_results:
                best_fiducials[config][analysis_key][energy_label]["EnergyBands"] = band_results

_merge_and_write_json(filename, best_fiducials)
