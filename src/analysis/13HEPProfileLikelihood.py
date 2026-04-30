import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *


def get_selection_cuts(
    config: str,
    name: str,
    energy: str,
    folder: str,
    args: argparse.Namespace,
):
    sigma = pd.read_pickle(
        f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/HEP/{folder.lower()}/{config}/{name}/{config}_{name}_highest_HEP.pkl"
    )
    try:
        ref_plot = sigma[(config, name, energy)]
    except KeyError:
        return None

    nhits_value = args.nhits if args.nhits is not None else int(ref_plot["NHits"])
    ophits_value = args.ophits if args.ophits is not None else int(ref_plot["OpHits"])
    adjcl_value = args.adjcls if args.adjcls is not None else int(ref_plot["AdjCl"])
    return nhits_value, ophits_value, adjcl_value


def evaluate_profile_exposure_curve(
    exposure_grid: np.ndarray,
    detector_mass: float,
    signal_rate: np.ndarray,
    background_rate: np.ndarray,
    background_error_rate: np.ndarray,
    signal_uncertainty: float,
    detection_requirement: float,
    adaptive_rebin_config: dict,
    use_adaptive_rebin: bool,
    freeze_rebin_starts: bool,
):
    significance_curve = []
    grouped_bins = []
    frozen_starts = None

    if use_adaptive_rebin and freeze_rebin_starts and len(exposure_grid) > 0:
        reference_factor = float(exposure_grid[-1]) * detector_mass
        reference_signal_events = reference_factor * signal_rate
        reference_background_events = reference_factor * background_rate
        reference_background_error_events = reference_factor * background_error_rate
        reference_detection_signal = reference_signal_events * (
            1.0 - detection_requirement * signal_uncertainty
        )

        (
            _reference_signal,
            _reference_background,
            _reference_background_uncertainty,
            reference_starts,
        ) = apply_adaptive_tail_rebin(
            reference_signal_events,
            reference_background_events,
            reference_background_error_events,
            reference_detection_signal,
            adaptive_rebin_config,
            apply_detection_mask=False,
        )
        frozen_starts = np.asarray(reference_starts, dtype=int)

    for years in exposure_grid:
        factor = years * detector_mass
        signal_events = factor * signal_rate
        background_events = factor * background_rate
        background_error_events = factor * background_error_rate
        detection_signal = signal_events * (
            1.0 - detection_requirement * signal_uncertainty
        )

        if use_adaptive_rebin:
            if frozen_starts is not None and frozen_starts.size > 0:
                signal_input = rebin_with_starts(signal_events, frozen_starts)
                background_input = rebin_with_starts(background_events, frozen_starts)
                background_uncertainty_input = rebin_with_starts(
                    background_error_events, frozen_starts
                )
                rebin_starts = frozen_starts
            else:
                (
                    signal_input,
                    background_input,
                    background_uncertainty_input,
                    rebin_starts,
                ) = apply_adaptive_tail_rebin(
                    signal_events,
                    background_events,
                    background_error_events,
                    detection_signal,
                    adaptive_rebin_config,
                    apply_detection_mask=False,
                )
        else:
            signal_input = detection_signal
            background_input = background_events
            background_uncertainty_input = background_error_events
            rebin_starts = np.arange(signal_events.size, dtype=int)

        significance_curve.append(
            evaluate_profile_likelihood_discovery(
                signal_input,
                background_input,
                background_uncertainty=background_uncertainty_input,
            )
        )
        grouped_bins.append(int(len(rebin_starts)))

    return (
        np.asarray(significance_curve, dtype=float),
        np.asarray(grouped_bins, dtype=float),
    )


parser = argparse.ArgumentParser(
    description=(
        "Compute profile-likelihood discovery exposure curves for HEP using the best cut "
        "combination selected by 0ZBestSigmas"
    )
)
parser.add_argument("--analysis", type=str, default="HEP")
parser.add_argument("--config", nargs="+", type=str, default=["hd_1x2x6_centralAPA"])
parser.add_argument("--name", nargs="+", type=str, default=["marley"])
parser.add_argument(
    "--folder", type=str, default="Nominal", choices=["Reduced", "Truncated", "Nominal"]
)
parser.add_argument("--signal_uncertainty", type=float, default=0.3)
parser.add_argument("--background_uncertainty", type=float, default=0.02)
parser.add_argument("--exposure", type=float, default=30)
parser.add_argument("--exposure_points", type=int, default=100)
parser.add_argument("--detection_center", type=float, default=2.0)
parser.add_argument("--detection_spread", type=float, default=0.1)
parser.add_argument(
    "--adaptive_rebin",
    action=argparse.BooleanOptionalAction,
    default=True,
)
parser.add_argument(
    "--freeze_rebin_starts",
    action=argparse.BooleanOptionalAction,
    default=True,
)
parser.add_argument(
    "--energy",
    nargs="+",
    type=str,
    choices=[
        "SignalParticleK",
        "ClusterEnergy",
        "TotalEnergy",
        "SelectedEnergy",
        "SolarEnergy",
    ],
    default=[
        "SignalParticleK",
        "ClusterEnergy",
        "TotalEnergy",
        "SelectedEnergy",
        "SolarEnergy",
    ],
)
parser.add_argument("--nhits", type=int, default=None)
parser.add_argument("--ophits", type=int, default=None)
parser.add_argument("--adjcls", type=int, default=None)
parser.add_argument(
    "--threshold",
    type=float,
    default=get_analysis_threshold(str(root), "HEP", stage="SIGNIFICANCE", fallback=0.0),
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()

exposure_grid = np.logspace(-1, np.log10(args.exposure), args.exposure_points)
smoothing_config = get_smoothing_config(
    str(root), analysis_name="HEP", dimensions="1d", stage="significance"
)
smoothing_info = smoothing_metadata(smoothing_config)
adaptive_rebin_config = get_adaptive_rebin_config(str(root), analysis_name="HEP")
adaptive_rebin_config["enabled"] = bool(args.adaptive_rebin)
adaptive_rebin_info = adaptive_rebin_metadata(adaptive_rebin_config)

components = ["neutron", "gamma", "8B", "hep"]
oscillations = ["Truth", "Truth", "Osc", "Osc"]
component_uncertainties = [
    args.background_uncertainty,
    args.background_uncertainty,
    args.signal_uncertainty,
    args.signal_uncertainty,
]

detection_requirements = [
    args.detection_center - args.detection_spread,
    args.detection_center,
    args.detection_center + args.detection_spread,
]

for config, name, energy in product(args.config, args.name, args.energy):
    info = json.loads(open(f"{root}/config/{config}/{config}_config.json").read())
    detector_mass = get_full_detector_mass(config, info)

    selection = get_selection_cuts(config, name, energy, args.folder, args)
    if selection is None:
        rprint(
            f"[yellow][WARNING][/yellow] Missing best-cut selection for {config} {name} {energy}."
        )
        continue

    nhits_value, ophits_value, adjcl_value = selection

    plot_df = pd.read_pickle(
        f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/signal/{args.folder.lower()}/HEP/{config}/{name}/{config}_{name}_{energy}_Rebin.pkl"
    )
    for _, filepath in load_available_background_dataframes(
        str(root), "HEP", args.folder, config, energy
    ):
        bkg_df = pd.read_pickle(filepath)
        plot_df = pd.concat([plot_df, bkg_df], ignore_index=True)

    this_df = plot_df.loc[
        (plot_df["NHits"] == int(nhits_value))
        * (plot_df["OpHits"] == int(ophits_value))
        * (plot_df["AdjCl"] == int(adjcl_value))
    ].copy()
    if this_df.empty:
        rprint(
            f"[yellow][WARNING][/yellow] No data for selected cuts in {config} {name} {energy}."
        )
        continue

    energy_axis = np.asarray(this_df["Energy"].values[0], dtype=float)
    threshold_idx = int(np.searchsorted(energy_axis, args.threshold, side="left"))

    raw_background = np.zeros(len(energy_axis) - threshold_idx, dtype=float)
    smoothed_background = np.zeros(len(energy_axis) - threshold_idx, dtype=float)
    background_error = np.zeros(
        (len(components) - 1, len(energy_axis) - threshold_idx), dtype=float
    )
    smoothed_background_error = np.zeros(
        (len(components) - 1, len(energy_axis) - threshold_idx), dtype=float
    )
    hep_signal = None
    smoothed_hep = None

    for idx, (component, osc, uncertainty_error) in enumerate(
        zip(components, oscillations, component_uncertainties)
    ):
        comp_df = this_df.loc[this_df["Component"] == component].copy()
        if comp_df.empty:
            continue

        comp_df = comp_df.fillna(0)
        this_comp_df = comp_df.loc[
            (comp_df["Oscillation"] == osc) * (comp_df["Mean"] == "Mean")
        ]
        if this_comp_df.empty:
            continue

        counts = np.asarray(this_comp_df["Counts"].values[0][threshold_idx:], dtype=float)
        errors = np.asarray(this_comp_df["Error"].values[0][threshold_idx:], dtype=float)
        component_smoothing_config = get_component_smoothing_config(
            smoothing_config, component
        )
        smoothed_counts = smooth_histogram_with_config(counts, component_smoothing_config)

        if component == "hep":
            hep_signal = counts
            smoothed_hep = smoothed_counts
            continue

        raw_background += counts
        smoothed_background += smoothed_counts

        background_statistical = np.divide(
            errors,
            counts,
            out=np.zeros_like(errors, dtype=float),
            where=counts != 0,
        )
        background_systematic = uncertainty_error * np.ones(
            len(background_statistical), dtype=float
        )
        background_error[idx] = np.sqrt(
            background_statistical**2 + background_systematic**2
        )
        background_error[idx] = np.multiply(
            background_error[idx],
            counts,
            out=np.zeros_like(background_error[idx]),
            where=counts != 0,
        )
        background_error[idx] = np.nan_to_num(
            background_error[idx], nan=0.0, posinf=0.0, neginf=0.0
        )

        smoothed_background_error[idx] = smooth_histogram(
            background_error[idx],
            method=component_smoothing_config.get("method", "none"),
            preserve_integral=False,
            **component_smoothing_config.get("params", {}),
        )

    if hep_signal is None or smoothed_hep is None:
        rprint(
            f"[yellow][WARNING][/yellow] Missing HEP signal for {config} {name} {energy}."
        )
        continue

    raw_signal_rate = np.asarray(hep_signal, dtype=float)
    raw_background_rate = np.asarray(raw_background, dtype=float)
    raw_background_error_rate = np.asarray(np.sum(background_error, axis=0), dtype=float)

    smoothed_signal_rate = np.asarray(smoothed_hep, dtype=float)
    smoothed_background_rate = np.asarray(smoothed_background, dtype=float)
    smoothed_background_error_rate = np.asarray(
        np.sum(smoothed_background_error, axis=0), dtype=float
    )

    raw_profile_curves = []
    raw_bins_curves = []
    raw_profile_norebin_curves = []
    raw_bins_norebin_curves = []
    smoothed_profile_curves = []
    smoothed_bins_curves = []
    smoothed_profile_norebin_curves = []
    smoothed_bins_norebin_curves = []

    for detection_requirement in detection_requirements:
        raw_curve, raw_bins_curve = evaluate_profile_exposure_curve(
            exposure_grid,
            detector_mass,
            raw_signal_rate,
            raw_background_rate,
            raw_background_error_rate,
            args.signal_uncertainty,
            detection_requirement,
            adaptive_rebin_config,
            args.adaptive_rebin,
            args.freeze_rebin_starts,
        )
        raw_curve_norebin, raw_bins_curve_norebin = evaluate_profile_exposure_curve(
            exposure_grid,
            detector_mass,
            raw_signal_rate,
            raw_background_rate,
            raw_background_error_rate,
            args.signal_uncertainty,
            detection_requirement,
            adaptive_rebin_config,
            False,
            False,
        )
        smooth_curve, smooth_bins_curve = evaluate_profile_exposure_curve(
            exposure_grid,
            detector_mass,
            smoothed_signal_rate,
            smoothed_background_rate,
            smoothed_background_error_rate,
            args.signal_uncertainty,
            detection_requirement,
            adaptive_rebin_config,
            args.adaptive_rebin,
            args.freeze_rebin_starts,
        )
        smooth_curve_norebin, smooth_bins_curve_norebin = evaluate_profile_exposure_curve(
            exposure_grid,
            detector_mass,
            smoothed_signal_rate,
            smoothed_background_rate,
            smoothed_background_error_rate,
            args.signal_uncertainty,
            detection_requirement,
            adaptive_rebin_config,
            False,
            False,
        )
        raw_profile_curves.append(raw_curve)
        raw_bins_curves.append(raw_bins_curve)
        raw_profile_norebin_curves.append(raw_curve_norebin)
        raw_bins_norebin_curves.append(raw_bins_curve_norebin)
        smoothed_profile_curves.append(smooth_curve)
        smoothed_bins_curves.append(smooth_bins_curve)
        smoothed_profile_norebin_curves.append(smooth_curve_norebin)
        smoothed_bins_norebin_curves.append(smooth_bins_curve_norebin)

    crossing_summary = compute_crossing_summary(
        exposure_grid,
        np.asarray(raw_profile_curves[1], dtype=float),
        np.asarray(smoothed_profile_curves[1], dtype=float),
    )

    profile_df = pd.DataFrame(
        [
            {
                "Config": config,
                "Name": name,
                "Energy": energy,
                "Exposure": exposure_grid.tolist(),
                "NHits": int(nhits_value),
                "OpHits": int(ophits_value),
                "AdjCl": int(adjcl_value),
                "RawProfileLikelihood+Error": raw_profile_curves[0].tolist(),
                "RawProfileLikelihood": raw_profile_curves[1].tolist(),
                "RawProfileLikelihood-Error": raw_profile_curves[2].tolist(),
                "RawProfileLikelihoodNoRebin+Error": raw_profile_norebin_curves[0].tolist(),
                "RawProfileLikelihoodNoRebin": raw_profile_norebin_curves[1].tolist(),
                "RawProfileLikelihoodNoRebin-Error": raw_profile_norebin_curves[2].tolist(),
                "ProfileLikelihood+Error": smoothed_profile_curves[0].tolist(),
                "ProfileLikelihood": smoothed_profile_curves[1].tolist(),
                "ProfileLikelihood-Error": smoothed_profile_curves[2].tolist(),
                "ProfileLikelihoodNoRebin+Error": smoothed_profile_norebin_curves[0].tolist(),
                "ProfileLikelihoodNoRebin": smoothed_profile_norebin_curves[1].tolist(),
                "ProfileLikelihoodNoRebin-Error": smoothed_profile_norebin_curves[2].tolist(),
                "RawAdaptiveBins+Error": raw_bins_curves[0].tolist(),
                "RawAdaptiveBins": raw_bins_curves[1].tolist(),
                "RawAdaptiveBins-Error": raw_bins_curves[2].tolist(),
                "RawAdaptiveBinsNoRebin+Error": raw_bins_norebin_curves[0].tolist(),
                "RawAdaptiveBinsNoRebin": raw_bins_norebin_curves[1].tolist(),
                "RawAdaptiveBinsNoRebin-Error": raw_bins_norebin_curves[2].tolist(),
                "AdaptiveBins+Error": smoothed_bins_curves[0].tolist(),
                "AdaptiveBins": smoothed_bins_curves[1].tolist(),
                "AdaptiveBins-Error": smoothed_bins_curves[2].tolist(),
                "AdaptiveBinsNoRebin+Error": smoothed_bins_norebin_curves[0].tolist(),
                "AdaptiveBinsNoRebin": smoothed_bins_norebin_curves[1].tolist(),
                "AdaptiveBinsNoRebin-Error": smoothed_bins_norebin_curves[2].tolist(),
                "DetectionRequirementCenter": float(args.detection_center),
                "DetectionRequirementSpread": float(args.detection_spread),
                "FreezeAdaptiveRebinStarts": bool(args.freeze_rebin_starts),
                # **smoothing_info,
                # **adaptive_rebin_info,
                # **crossing_summary,
            }
        ]
    )

    save_df(
        profile_df,
        f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/HEP/{args.folder.lower()}",
        config,
        name,
        filename=f"{energy}_HEP_ProfileLikelihood",
        rm=args.rewrite,
        debug=args.debug,
    )

    if args.debug:
        rprint(
            f"[cyan][INFO][/cyan] Saved profile-likelihood exposure curve for {config} {name} {energy}."
        )
