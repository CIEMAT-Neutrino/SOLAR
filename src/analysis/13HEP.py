import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from lib import *
from typing import Optional


# ---------------------------------------------------------------------------
# Module-level helpers for HEP significance computation
# ---------------------------------------------------------------------------

def _rebin_from_starts(
    signal_events: np.ndarray,
    background_events: np.ndarray,
    background_uncertainty_events: np.ndarray,
    detection_events: np.ndarray,
    starts: np.ndarray,
    detection_threshold: float,
) -> tuple:
    """Apply a fixed bin-boundary set (starts) to all arrays and mask bins below detection_threshold."""
    starts_array = np.asarray(starts, dtype=int)
    reb_s = rebin_with_starts(signal_events, starts_array)
    reb_b = rebin_with_starts(background_events, starts_array)
    reb_u = rebin_with_starts(background_uncertainty_events, starts_array)
    reb_d = rebin_with_starts(detection_events, starts_array)
    mask = reb_d >= detection_threshold
    return (
        np.where(mask, reb_s, 0.0),
        np.where(mask, reb_b, 0.0),
        np.where(mask, reb_u, 0.0),
        starts_array,
    )


def _hep_significance_step(
    signal_rate: np.ndarray,
    background_rate: np.ndarray,
    bkg_error_rate: np.ndarray,
    factor: float,
    detection_requirement: float,
    signal_uncertainty: float,
    detection_threshold: float,
    prev_starts: np.ndarray,
    last_asimov: Optional[float],
    adaptive_rebin_config: dict,
) -> dict:
    """Compute HEP significance metrics for one (exposure, detection_requirement) step.

    Applies the adaptive tail rebin with monotonicity enforcement: if the current
    step's optimal binning gives lower Asimov significance than the previous step,
    the previous step's binning is reused to keep the significance curve non-decreasing.

    Returns
    -------
    dict with keys: gaussian_no_rebin, asimov_no_rebin,
                    gaussian_rebinned, asimov_rebinned, starts, n_bins.
    """
    signal_events = factor * signal_rate
    background_events = factor * background_rate
    bkg_error_events = factor * bkg_error_rate
    detection_signal = signal_events * (1.0 - detection_requirement * signal_uncertainty)

    detectable = detection_signal >= detection_threshold
    s_nr = np.where(detectable, signal_events, 0.0)
    b_nr = np.where(detectable, background_events, 0.0)
    u_nr = np.where(detectable, bkg_error_events, 0.0)

    def _global(arr: np.ndarray) -> float:
        return float(np.sqrt(np.sum(np.power(
            np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), 2,
        ))))

    gaussian_no_rebin = _global(evaluate_significance(s_nr, b_nr, background_uncertainty=u_nr, type="gaussian"))
    asimov_no_rebin   = _global(evaluate_significance(s_nr, b_nr, background_uncertainty=u_nr, type="asimov"))

    s_cand, b_cand, u_cand, starts_cand = apply_adaptive_tail_rebin(
        signal_events, background_events, bkg_error_events, detection_signal, adaptive_rebin_config,
    )

    s_in, b_in, u_in, starts_out = s_cand, b_cand, u_cand, starts_cand
    if prev_starts.size > 0 and last_asimov is not None:
        s_prev, b_prev, u_prev, starts_prev = _rebin_from_starts(
            signal_events, background_events, bkg_error_events,
            detection_signal, prev_starts, detection_threshold,
        )
        asimov_cand_g = _global(evaluate_significance(
            s_cand, b_cand, background_uncertainty=u_cand, type="asimov",
        ))
        if asimov_cand_g < last_asimov:
            s_in, b_in, u_in, starts_out = s_prev, b_prev, u_prev, starts_prev

    gaussian_rebinned = _global(evaluate_significance(s_in, b_in, background_uncertainty=u_in, type="gaussian"))
    asimov_rebinned   = _global(evaluate_significance(s_in, b_in, background_uncertainty=u_in, type="asimov"))

    return {
        "gaussian_no_rebin": gaussian_no_rebin,
        "asimov_no_rebin":   asimov_no_rebin,
        "gaussian_rebinned": gaussian_rebinned,
        "asimov_rebinned":   asimov_rebinned,
        "starts": np.asarray(starts_out, dtype=int).copy(),
        "n_bins": int(len(starts_out)),
    }


def _hep_display_spectrum(
    signal_rate: np.ndarray,
    background_rate: np.ndarray,
    bkg_error_rate: np.ndarray,
    display_factor: float,
    display_detection_requirement: float,
    signal_uncertainty: float,
    detection_threshold: float,
    adaptive_rebin_config: dict,
) -> dict:
    """Compute per-bin significance spectra at the full display exposure for plotting.

    No monotonicity enforcement — each spectrum is computed independently of other
    exposure steps. Returns both the original binning and the adaptive-rebinned version.

    Returns
    -------
    dict with keys: asimov_no_rebin, gaussian_no_rebin,
                    asimov_adaptive, gaussian_adaptive, starts_adaptive.
    """
    signal = display_factor * signal_rate
    background = display_factor * background_rate
    uncertainty = display_factor * bkg_error_rate
    detection_signal = signal * (1.0 - display_detection_requirement * signal_uncertainty)

    no_rebin_mask = detection_signal >= detection_threshold
    s_nr = np.where(no_rebin_mask, signal, 0.0)
    b_nr = np.where(no_rebin_mask, background, 0.0)
    u_nr = np.where(no_rebin_mask, uncertainty, 0.0)

    def _clean(arr: np.ndarray) -> np.ndarray:
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    s_ad, b_ad, u_ad, starts_ad = apply_adaptive_tail_rebin(
        signal, background, uncertainty, detection_signal, adaptive_rebin_config,
    )

    return {
        "asimov_no_rebin":   _clean(evaluate_significance(s_nr, b_nr, background_uncertainty=u_nr, type="asimov")),
        "gaussian_no_rebin": _clean(evaluate_significance(s_nr, b_nr, background_uncertainty=u_nr, type="gaussian")),
        "asimov_adaptive":   _clean(evaluate_significance(s_ad, b_ad, background_uncertainty=u_ad, type="asimov")),
        "gaussian_adaptive": _clean(evaluate_significance(s_ad, b_ad, background_uncertainty=u_ad, type="gaussian")),
        "starts_adaptive":   starts_ad,
    }


# ---------------------------------------------------------------------------

data_path = f"{root}/data/solar/"
save_path = f"{root}/images/hep"
if not os.path.exists(save_path):
    os.makedirs(save_path)

parser = argparse.ArgumentParser(
    description="Perform the HEP significance analysis for a given configuration and name"
)
parser.add_argument("--config", nargs="+", type=str, default=["hd_1x2x6_centralAPA"])
parser.add_argument("--name", nargs="+", type=str, default=["marley"])
parser.add_argument("--folder", type=str, default="Nominal")
parser.add_argument("--signal_uncertainty", type=float, default=0.3)
parser.add_argument("--background_uncertainty", type=float, default=0.02)
parser.add_argument(
    "--energy",
    nargs="+",
    type=str,
    default=["ClusterEnergy", "TotalEnergy", "SelectedEnergy", "SolarEnergy"],
)
parser.add_argument("--exposure", type=float, default=30)
parser.add_argument("--nhits", type=int, default=None)
parser.add_argument("--ophits", type=int, default=None)
parser.add_argument("--adjcls", type=int, default=None)
parser.add_argument(
    "--mc_threshold",
    type=int,
    default=get_analysis_threshold(str(root), "HEP", stage="MC", fallback=0.0),
    help="Minimum summed MCCounts required for every essential background component after cuts",
)
parser.add_argument(
    "--threshold",
    type=float,
    default=get_analysis_threshold(str(root), "HEP", stage="SIGNIFICANCE", fallback=0.0),
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()
if args.debug:
    rprint(args)
explicit_debug_flag = "--debug" in sys.argv and "--no-debug" not in sys.argv

components = ["neutron", "gamma", "radiological", "8B", "hep"]
background_components = ["neutron", "gamma", "radiological", "8B"]
background_config = get_background_config(str(root))
essential_map = {
    str(component).lower(): bool(is_essential)
    for component, is_essential in background_config.get("ESSENTIAL", {}).items()
}
essential_background_components = [
    component
    for component in background_components
    if essential_map.get(component.lower(), False)
]
threshold_idx = np.where(hep_rebin_centers >= args.threshold)[0][0]
exposure_grid = np.logspace(-1, np.log10(args.exposure), 100)
smoothing_config = get_smoothing_config(
    str(root), analysis_name="HEP", dimensions="1d", stage="significance"
)
smoothing_info = smoothing_metadata(smoothing_config)
adaptive_rebin_config = get_adaptive_rebin_config(str(root), analysis_name="HEP")
adaptive_rebin_info = adaptive_rebin_metadata(adaptive_rebin_config)
if args.debug:
    rprint(
        f"[cyan][INFO][/cyan] Threshold {args.threshold} found to correspond to index {threshold_idx} of {len(hep_rebin_centers)} with value {hep_rebin_centers[threshold_idx]:.2f} MeV"
    )
    rprint(
        f"[cyan][INFO][/cyan] HEP smoothing method={smoothing_info['SmoothingMethod']} sigma={smoothing_info.get('SmoothingSigma', 0.0):.2f} enabled={smoothing_info['SmoothingEnabled']}"
    )
    rprint(
        f"[cyan][INFO][/cyan] HEP adaptive_rebin enabled={adaptive_rebin_info['AdaptiveRebinEnabled']} "
        f"min_events={adaptive_rebin_info['AdaptiveRebinMinExpectedEvents']:.2f} "
        f"min_prob={adaptive_rebin_info['AdaptiveRebinMinCountProbability']:.3f}"
    )

# detection_threshold depends only on the adaptive-rebin config and is constant
# across all cuts and exposures: compute once here.
_min_expected = float(adaptive_rebin_config.get("min_expected_events", 1.0))
_min_prob     = float(adaptive_rebin_config.get("min_count_probability", 0.6321205588))
_prob_events  = 0.0 if _min_prob <= 0 else -np.log(1.0 - min(_min_prob, 1.0 - 1e-12))
detection_threshold = max(_min_expected, _prob_events)

for config, name, energy in product(args.config, args.name, args.energy):
    info = json.loads(open(f"{root}/config/{config}/{config}_config.json").read())
    detector_mass = get_full_detector_mass(config, info)

    sigmas = []
    significance_bins = []
    essential_mc_samples = {
        component: [] for component in essential_background_components
    }
    total_cut_trials = 0
    cuts_with_any_data = 0
    cuts_with_all_components = 0
    cuts_passing_mc_threshold = 0
    plot_df = pd.read_pickle(
        f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/signal/{args.folder.lower()}/HEP/{config}/{name}/{config}_{name}_{energy}_Rebin.pkl"
    )
    for bkg, filepath in load_available_background_dataframes(str(root), "HEP", args.folder, config, energy):
        bkg_df = pd.read_pickle(filepath)
        plot_df = pd.concat([plot_df, bkg_df], ignore_index=True)

    sigmamax = 0.0
    last_sigma2 = 1e6
    last_sigma3 = 1e6

    for nhit, ophit, adjcl in track(
        product(
            nhits[:10],
            nhits[3:10],
            nhits[::-1][10:],
        ),
        description=f"Looping over analysis cuts for {energy}...",
        total=len(nhits[:10]) * len(nhits[3:10]) * len(nhits[::-1][10:]),
    ):
        total_cut_trials += 1
        this_df = plot_df.loc[
            (plot_df["NHits"] == int(nhit))
            * (plot_df["OpHits"] == int(ophit))
            * (plot_df["AdjCl"] == int(adjcl))
        ]
        if this_df.empty:
            rprint(
                f"[yellow][WARNING] No data for {energy} with {nhit} nhits and {ophit} ophits {adjcl} adjcl[/yellow]"
            )
            continue
        cuts_with_any_data += 1

        raw_background = np.zeros(len(hep_rebin_centers) - threshold_idx, dtype=float)
        smoothed_background = np.zeros(len(hep_rebin_centers) - threshold_idx, dtype=float)
        background_error = np.zeros((len(components) - 1, len(hep_rebin_centers) - threshold_idx), dtype=float)
        smoothed_background_error = np.zeros((len(components) - 1, len(hep_rebin_centers) - threshold_idx), dtype=float)
        hep_signal = None
        smoothed_hep = None
        mc_sums = {}

        for idx, (component, osc, uncertainty_error) in enumerate(
            zip(
                components,
                ["Truth", "Truth", "Truth", "Osc", "Osc"],
                [
                    args.background_uncertainty,
                    args.background_uncertainty,
                    args.background_uncertainty,
                    args.signal_uncertainty,
                    args.signal_uncertainty,
                ],
            )
        ):
            comp_df = this_df.loc[this_df["Component"] == component].copy()
            if comp_df.empty:
                rprint(
                    f"[yellow][WARNING] No data for {component} in {energy} with {nhit} nhits and {ophit} ophits {adjcl} adjcl[/yellow]"
                )
                continue
            comp_df = comp_df.fillna(0)
            this_comp_df = comp_df.loc[
                (comp_df["Oscillation"] == osc) * (comp_df["Mean"] == "Mean")
            ]
            counts = np.asarray(this_comp_df["Counts"].values[0][threshold_idx:], dtype=float)
            errors = np.asarray(this_comp_df["Error"].values[0][threshold_idx:], dtype=float)
            mc_counts = np.asarray(this_comp_df["MCCounts"].values[0][threshold_idx:], dtype=float)
            mc_sums[component] = float(np.sum(mc_counts))
            component_smoothing_config = get_component_smoothing_config(smoothing_config, component)
            smoothed_counts = smooth_histogram_with_config(counts, component_smoothing_config)

            if component == "hep":
                hep_signal = counts
                smoothed_hep = smoothed_counts
            else:
                raw_background += counts
                smoothed_background += smoothed_counts
                background_statistical = np.divide(
                    errors,
                    counts,
                    out=np.zeros_like(errors, dtype=float),
                    where=counts != 0,
                )
                background_systematic = uncertainty_error * np.ones(len(background_statistical), dtype=float)
                background_error[idx] = np.sqrt(background_statistical**2 + background_systematic**2)
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

        has_all_components = True
        for component, osc in [
            ("neutron", "Truth"),
            ("gamma", "Truth"),
            ("radiological", "Truth"),
            ("8B", "Osc"),
            ("hep", "Osc"),
        ]:
            comp_df = this_df.loc[
                (this_df["Component"] == component)
                * (this_df["Oscillation"] == osc)
                * (this_df["Mean"] == "Mean")
            ]
            if comp_df.empty:
                has_all_components = False
                break
        if has_all_components:
            cuts_with_all_components += 1

        has_required_background_mc = all(
            mc_sums.get(component, 0.0) >= args.mc_threshold
            for component in essential_background_components
        )

        for component in essential_background_components:
            essential_mc_samples.setdefault(component, []).append(
                float(mc_sums.get(component, 0.0))
            )

        if not has_required_background_mc or hep_signal is None or smoothed_hep is None:
            continue
        cuts_passing_mc_threshold += 1

        raw_combined_b_error = np.sum(background_error, axis=0)
        smoothed_combined_b_error = np.sum(smoothed_background_error, axis=0)

        found_sigma2 = False
        found_sigma3 = False
        sigma2 = 0.0
        sigma3 = 0.0
        sigma2_curve = []
        sigma3_curve = []
        raw_gaussian_significances = [[], [], []]
        raw_asimov_significances = [[], [], []]
        smoothed_gaussian_significances = [[], [], []]
        smoothed_asimov_significances = [[], [], []]
        raw_gaussian_no_rebin_significances = [[], [], []]
        raw_asimov_no_rebin_significances = [[], [], []]
        smoothed_gaussian_no_rebin_significances = [[], [], []]
        smoothed_asimov_no_rebin_significances = [[], [], []]
        raw_rebinned_bins = [[], [], []]
        smoothed_rebinned_bins = [[], [], []]

        raw_signal_rate = np.asarray(hep_signal, dtype=float)
        raw_background_rate = np.asarray(raw_background, dtype=float)
        raw_bkg_error_rate = np.asarray(raw_combined_b_error, dtype=float)
        smoothed_signal_rate = np.asarray(smoothed_hep, dtype=float)
        smoothed_background_rate = np.asarray(smoothed_background, dtype=float)
        smoothed_bkg_error_rate = np.asarray(smoothed_combined_b_error, dtype=float)

        prev_raw_starts      = [np.zeros(0, dtype=int) for _ in range(3)]
        prev_smoothed_starts = [np.zeros(0, dtype=int) for _ in range(3)]

        for years in exposure_grid:
            factor = years * detector_mass
            for kdx, detection_requirement in enumerate([2.9, 3.0, 3.1]):
                raw = _hep_significance_step(
                    raw_signal_rate, raw_background_rate, raw_bkg_error_rate,
                    factor, detection_requirement, args.signal_uncertainty, detection_threshold,
                    prev_raw_starts[kdx],
                    raw_asimov_significances[kdx][-1] if raw_asimov_significances[kdx] else None,
                    adaptive_rebin_config,
                )
                prev_raw_starts[kdx] = raw["starts"]
                raw_gaussian_no_rebin_significances[kdx].append(raw["gaussian_no_rebin"])
                raw_asimov_no_rebin_significances[kdx].append(raw["asimov_no_rebin"])
                raw_gaussian_significances[kdx].append(raw["gaussian_rebinned"])
                raw_asimov_significances[kdx].append(raw["asimov_rebinned"])
                raw_rebinned_bins[kdx].append(raw["n_bins"])

                smoothed = _hep_significance_step(
                    smoothed_signal_rate, smoothed_background_rate, smoothed_bkg_error_rate,
                    factor, detection_requirement, args.signal_uncertainty, detection_threshold,
                    prev_smoothed_starts[kdx],
                    smoothed_asimov_significances[kdx][-1] if smoothed_asimov_significances[kdx] else None,
                    adaptive_rebin_config,
                )
                prev_smoothed_starts[kdx] = smoothed["starts"]
                smoothed_gaussian_no_rebin_significances[kdx].append(smoothed["gaussian_no_rebin"])
                smoothed_asimov_no_rebin_significances[kdx].append(smoothed["asimov_no_rebin"])
                smoothed_gaussian_significances[kdx].append(smoothed["gaussian_rebinned"])
                smoothed_asimov_significances[kdx].append(smoothed["asimov_rebinned"])
                smoothed_rebinned_bins[kdx].append(smoothed["n_bins"])

            if smoothed_asimov_significances[1][-1] > sigmamax:
                sigmamax = smoothed_asimov_significances[1][-1]

            if smoothed_asimov_significances[1][-1] > 2 and not found_sigma2:
                sigma2 = factor
                found_sigma2 = True
                if sigma2 < last_sigma2 and args.debug:
                    rprint(
                        f"Found smoothed sigma2 with exposure {factor:.0f} for nhits {nhit} ophits {ophit} and adjcls {adjcl}"
                    )
                if sigma2 < last_sigma2:
                    last_sigma2 = sigma2

            if smoothed_asimov_significances[1][-1] > 3 and not found_sigma3:
                sigma3 = factor
                found_sigma3 = True
                if sigma3 < last_sigma3 and args.debug:
                    rprint(
                        f"Found smoothed sigma3 with exposure {factor:.0f} for nhits {nhit} ophits {ophit} and adjcls {adjcl}"
                    )
                if sigma3 < last_sigma3:
                    last_sigma3 = sigma3

            sigma2_curve.append(sigma2)
            sigma3_curve.append(sigma3)

        # Per-bin significance spectra at the configured display exposure for plotting.
        display_factor = args.exposure * detector_mass
        display_detection_requirement = 3.0

        raw_display = _hep_display_spectrum(
            raw_signal_rate, raw_background_rate, raw_bkg_error_rate,
            display_factor, display_detection_requirement, args.signal_uncertainty,
            detection_threshold, adaptive_rebin_config,
        )
        smoothed_display = _hep_display_spectrum(
            smoothed_signal_rate, smoothed_background_rate, smoothed_bkg_error_rate,
            display_factor, display_detection_requirement, args.signal_uncertainty,
            detection_threshold, adaptive_rebin_config,
        )

        raw_asimov_spectrum            = raw_display["asimov_no_rebin"]
        raw_gaussian_spectrum          = raw_display["gaussian_no_rebin"]
        raw_asimov_adaptive_spectrum   = raw_display["asimov_adaptive"]
        raw_gaussian_adaptive_spectrum = raw_display["gaussian_adaptive"]

        smoothed_asimov_spectrum            = smoothed_display["asimov_no_rebin"]
        smoothed_gaussian_spectrum          = smoothed_display["gaussian_no_rebin"]
        smoothed_asimov_adaptive_spectrum   = smoothed_display["asimov_adaptive"]
        smoothed_gaussian_adaptive_spectrum = smoothed_display["gaussian_adaptive"]

        adaptive_energy_axis_display, adaptive_bin_widths_display = grouped_axis_from_starts(
            hep_rebin_centers[threshold_idx:],
            smoothed_display["starts_adaptive"],
        )

        crossing_summary = compute_crossing_summary(
            exposure_grid,
            np.asarray(raw_asimov_significances[1], dtype=float),
            np.asarray(smoothed_asimov_significances[1], dtype=float),
        )
        if smoothing_info["SmoothingReport"] and args.debug and explicit_debug_flag:
            rprint(
                f"[cyan][REPORT][/cyan] {config} {name} {energy} NHits={nhit} OpHits={ophit} AdjCl={adjcl} "
                f"sigma2 raw={crossing_summary['RawSigma2Crossing']:.2f} smooth={crossing_summary['SmoothedSigma2Crossing']:.2f}, "
                f"sigma3 raw={crossing_summary['RawSigma3Crossing']:.2f} smooth={crossing_summary['SmoothedSigma3Crossing']:.2f}"
            )
        if adaptive_rebin_info["AdaptiveRebinReport"] and args.debug and explicit_debug_flag:
            rprint(
                f"[cyan][REPORT][/cyan] {config} {name} {energy} NHits={nhit} OpHits={ophit} AdjCl={adjcl} "
                f"adaptive bins raw={np.mean(raw_rebinned_bins[1]):.2f} smooth={np.mean(smoothed_rebinned_bins[1]):.2f}"
            )

        sigmas.append(
            {
                "Config": config,
                "Name": name,
                "Energy": energy,
                "Sigma2": sigma2_curve,
                "Sigma3": sigma3_curve,
                "Exposure": exposure_grid.tolist(),
                "NHits": nhit,
                "OpHits": ophit,
                "AdjCl": adjcl,
                "Gaussian+Error": smoothed_gaussian_significances[0],
                "Gaussian": smoothed_gaussian_significances[1],
                "Gaussian-Error": smoothed_gaussian_significances[2],
                "Asimov+Error": smoothed_asimov_significances[0],
                "Asimov": smoothed_asimov_significances[1],
                "Asimov-Error": smoothed_asimov_significances[2],
                "RawGaussian+Error": raw_gaussian_significances[0],
                "RawGaussian": raw_gaussian_significances[1],
                "RawGaussian-Error": raw_gaussian_significances[2],
                "RawAsimov+Error": raw_asimov_significances[0],
                "RawAsimov": raw_asimov_significances[1],
                "RawAsimov-Error": raw_asimov_significances[2],
                "RawGaussianNoRebin+Error": raw_gaussian_no_rebin_significances[0],
                "RawGaussianNoRebin": raw_gaussian_no_rebin_significances[1],
                "RawGaussianNoRebin-Error": raw_gaussian_no_rebin_significances[2],
                "RawAsimovNoRebin+Error": raw_asimov_no_rebin_significances[0],
                "RawAsimovNoRebin": raw_asimov_no_rebin_significances[1],
                "RawAsimovNoRebin-Error": raw_asimov_no_rebin_significances[2],
                "GaussianNoRebin+Error": smoothed_gaussian_no_rebin_significances[0],
                "GaussianNoRebin": smoothed_gaussian_no_rebin_significances[1],
                "GaussianNoRebin-Error": smoothed_gaussian_no_rebin_significances[2],
                "AsimovNoRebin+Error": smoothed_asimov_no_rebin_significances[0],
                "AsimovNoRebin": smoothed_asimov_no_rebin_significances[1],
                "AsimovNoRebin-Error": smoothed_asimov_no_rebin_significances[2],
                "RawAdaptiveBins+Error": raw_rebinned_bins[0],
                "RawAdaptiveBins": raw_rebinned_bins[1],
                "RawAdaptiveBins-Error": raw_rebinned_bins[2],
                "AdaptiveBins+Error": smoothed_rebinned_bins[0],
                "AdaptiveBins": smoothed_rebinned_bins[1],
                "AdaptiveBins-Error": smoothed_rebinned_bins[2],
                **smoothing_info,
                **adaptive_rebin_info,
                **crossing_summary,
            }
        )

        no_rebin_energy = np.asarray(hep_rebin_centers[threshold_idx:], dtype=float)
        no_rebin_width = float(np.median(np.diff(no_rebin_energy))) if len(no_rebin_energy) > 1 else 1.0
        for bin_idx, (energy_value, raw_asimov_value, asimov_value, raw_gauss_value, gauss_value) in enumerate(
            zip(
                no_rebin_energy,
                raw_asimov_spectrum,
                smoothed_asimov_spectrum,
                raw_gaussian_spectrum,
                smoothed_gaussian_spectrum,
            )
        ):
            significance_bins.append(
                {
                    "Config": config,
                    "Name": name,
                    "EnergyLabel": energy,
                    "NHits": int(nhit),
                    "OpHits": int(ophit),
                    "AdjCl": int(adjcl),
                    "Threshold": float(args.threshold),
                    "ExposureYears": float(args.exposure),
                    "BinMode": "NoRebin",
                    "BinIndex": int(bin_idx),
                    "RecoEnergy": float(energy_value),
                    "BinWidth": float(no_rebin_width),
                    "RawAsimov": float(raw_asimov_value),
                    "Asimov": float(asimov_value),
                    "RawGaussian": float(raw_gauss_value),
                    "Gaussian": float(gauss_value),
                    **smoothing_info,
                    **adaptive_rebin_info,
                }
            )

        adaptive_energy_axis = np.asarray(adaptive_energy_axis_display, dtype=float)
        adaptive_widths = np.asarray(adaptive_bin_widths_display, dtype=float)
        for bin_idx, (energy_value, width_value, raw_asimov_value, asimov_value, raw_gauss_value, gauss_value) in enumerate(
            zip(
                adaptive_energy_axis,
                adaptive_widths,
                raw_asimov_adaptive_spectrum,
                smoothed_asimov_adaptive_spectrum,
                raw_gaussian_adaptive_spectrum,
                smoothed_gaussian_adaptive_spectrum,
            )
        ):
            significance_bins.append(
                {
                    "Config": config,
                    "Name": name,
                    "EnergyLabel": energy,
                    "NHits": int(nhit),
                    "OpHits": int(ophit),
                    "AdjCl": int(adjcl),
                    "Threshold": float(args.threshold),
                    "ExposureYears": float(args.exposure),
                    "BinMode": "AdaptiveRebin",
                    "BinIndex": int(bin_idx),
                    "RecoEnergy": float(energy_value),
                    "BinWidth": float(width_value),
                    "RawAsimov": float(raw_asimov_value),
                    "Asimov": float(asimov_value),
                    "RawGaussian": float(raw_gauss_value),
                    "Gaussian": float(gauss_value),
                    **smoothing_info,
                    **adaptive_rebin_info,
                }
            )

    if args.debug:
        rprint(f"Maximum significance for {energy}: {sigmamax:.2f} sigma")
    sigmas_df = pd.DataFrame(sigmas)
    if sigmas_df.empty:
        rprint(
            f"[yellow][WARNING][/yellow] No HEP result rows produced for {config} {name} {energy}. "
            f"Cuts tried={total_cut_trials}, with_data={cuts_with_any_data}, "
            f"with_required_components={cuts_with_all_components}, "
            f"passing_mc_threshold={cuts_passing_mc_threshold}, mc_threshold={args.mc_threshold}."
        )
        rprint(
            "[yellow][WARNING][/yellow] Current gate requires every essential background component "
            f"({', '.join(essential_background_components) if essential_background_components else 'none configured'}) "
            "to pass --mc_threshold after cuts. Consider lowering it (for example 0 or 10) "
            "if this dataset has low MC statistics after cuts."
        )
        for component in essential_background_components:
            values = np.asarray(essential_mc_samples.get(component, []), dtype=float)
            if values.size == 0:
                continue
            rprint(
                f"[yellow][WARNING][/yellow] Essential component '{component}' MCCounts summary after threshold: "
                f"min={np.min(values):.3f}, median={np.median(values):.3f}, max={np.max(values):.3f}, "
                f"pass@{args.mc_threshold:.3f}={(values >= args.mc_threshold).sum()}/{values.size}."
            )
        raise SystemExit(
            "13HEP.py produced no valid HEP rows after cut and MC-threshold filtering. "
            "Aborting to avoid writing empty payloads for downstream plotting."
        )
    save_df(
        sigmas_df,
        f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/HEP/{args.folder.lower()}",
        config,
        name,
        filename=f"{energy}_HEP_Results",
        rm=args.rewrite,
        debug=args.debug,
    )
    significance_bins_df = pd.DataFrame(significance_bins)
    save_df(
        significance_bins_df,
        f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/HEP/{args.folder.lower()}",
        config,
        name,
        filename=f"{energy}_HEP_SignificanceBins",
        rm=args.rewrite,
        debug=args.debug,
    )
