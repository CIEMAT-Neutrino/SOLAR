import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from lib import *


parser = argparse.ArgumentParser(
    description="Perform the day-night analysis for a given configuration and name",
    allow_abbrev=False,
)
parser.add_argument("--config", nargs="+", type=str, default=["hd_1x2x6_centralAPA"])
parser.add_argument("--name", nargs="+", type=str, default=["marley"])
parser.add_argument("--folder", type=str, default="Nominal")
parser.add_argument("--signal_uncertainty", type=float, default=0.00)
parser.add_argument("--background_uncertainty", type=float, default=0.02)
parser.add_argument(
    "--earth_density_band",
    type=float,
    default=0.13,
    help=(
        "Fractional spread in the expected day-night asymmetry from Earth density profile "
        "variations (MSW matter effect). Combined in quadrature with --oscillation_band to "
        "produce the total asymmetry uncertainty band. Default ±13%% (published PREM range)."
    ),
)
parser.add_argument(
    "--oscillation_band",
    type=float,
    default=0.05,
    help=(
        "Fractional uncertainty on the predicted asymmetry from oscillation parameter "
        "uncertainties (theta12, dm221). Combined in quadrature with --earth_density_band. "
        "Default 5%% covering published PDG uncertainties on theta12 and dm221."
    ),
)
parser.add_argument(
    "--day_fraction",
    type=float,
    default=0.493,
    help=(
        "Fraction of total exposure attributed to daytime. Default 0.493 (SURF latitude ~44.3°N, averaged over a full year)."
    ),
)
parser.add_argument(
    "--day_fraction_band",
    type=float,
    default=0.01,
    help=(
        "Absolute uncertainty on --day_fraction from imperfect knowledge of the solar "
        "zenith angle cut and run schedule. Propagated as an additional background "
        "uncertainty: sigma_B_dayfrac = day_fraction_band * N_background_total. Default 1%%."
    ),
)
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
    type=float,
    default=get_analysis_threshold(str(root), "DAYNIGHT", stage="MC", fallback=0.0),
    help="Minimum summed MCCounts required for every essential background component after cuts",
)
parser.add_argument(
    "--threshold",
    type=float,
    default=get_analysis_threshold(str(root), "DAYNIGHT", stage="SIGNIFICANCE", fallback=0.0),
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument(
    "--all_metrics",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Run all configured significance metrics (Gaussian, Asimov, raw variants, "
        "background error bands, significance bins). By default only Asimov runs. "
        "Pass --all_metrics to enable everything configured in WORKFLOW."
    ),
)
parser.add_argument(
    "--test_statistic",
    type=str,
    choices=["asimov", "gaussian", "all"],
    default="asimov",
    help=(
        "Test statistic type: 'asimov' (default), 'gaussian', or 'all' for both. "
        "Overrides WORKFLOW config METRICS for gaussian/asimov selection."
    ),
)

args = parser.parse_args()
if args.debug:
    rprint(args)
explicit_debug_flag = "--debug" in sys.argv and "--no-debug" not in sys.argv

# The signal is the oscillation-induced asymmetry (N_night - N_day).
# Two independent sources of uncertainty on the predicted asymmetry amplitude are
# combined in quadrature to form a single total band:
#   earth_density_band : MSW matter effect from Earth density profile variations (PREM)
#   oscillation_band   : residual uncertainty from theta12 and dm221 (PDG values)
# The three asymmetry_scales bracket the full predicted range:
#   upper  (1 + total_band): stronger matter effect / larger oscillation parameters
#   nominal (1.0)           : best-fit prediction
#   lower  (1 - total_band): weaker matter effect / smaller oscillation parameters
total_asymmetry_band = float(np.sqrt(args.earth_density_band**2 + args.oscillation_band**2))
asymmetry_scales = [1.0 + total_asymmetry_band, 1.0, 1.0 - total_asymmetry_band]

background_config = get_background_config(str(root))
essential_map = {
    str(component).lower(): bool(is_essential)
    for component, is_essential in background_config.get("ESSENTIAL", {}).items()
}
daynight_background_components = get_background_samples(str(root), "DAYNIGHT")
essential_background_components = [
    component
    for component in daynight_background_components
    if essential_map.get(component.lower(), False)
]

threshold_idx = np.where(daynight_rebin_centers >= args.threshold)[0][0]
exposure_grid = np.logspace(-1, np.log10(args.exposure), 100)
smoothing_config = get_smoothing_config(
    str(root), analysis_name="DAYNIGHT", dimensions="1d", stage="significance"
)
smoothing_info = smoothing_metadata(smoothing_config)
_workflow = get_workflow_flags(str(root), "DAYNIGHT")
_metrics  = get_metrics_config(str(root), "DAYNIGHT", all_metrics=args.all_metrics)

# Override gaussian/asimov based on --test_statistic flag
if args.test_statistic == "asimov":
    _metrics["gaussian"] = False
    _metrics["asimov"] = True
elif args.test_statistic == "gaussian":
    _metrics["gaussian"] = True
    _metrics["asimov"] = False
elif args.test_statistic == "all":
    _metrics["gaussian"] = True
    _metrics["asimov"] = True

_compute_gaussian         = _metrics.get("gaussian", False)
_compute_asimov           = _metrics.get("asimov", True)
_compute_raw              = _metrics.get("raw_variants", False)
_compute_error_bands      = _metrics.get("error_bands", False) and _workflow["background_error"]
_compute_significance_bins = _metrics.get("significance_bins", False) and _workflow["significance_bins"]

_dn_active_metrics = [
    *(["gaussian"]         if _compute_gaussian else []),
    *(["asimov"]           if _compute_asimov else []),
    *(["raw_variants"]     if _compute_raw else []),
    *(["error_bands"]      if _compute_error_bands else []),
    *(["significance_bins"] if _compute_significance_bins else []),
]
rprint(
    f"[cyan][INFO][/cyan] DayNight metrics active: {', '.join(_dn_active_metrics)}"
    + ("" if args.all_metrics else "  [dim](pass --all_metrics for full set)[/dim]")
)

if args.debug:
    rprint(
        f"[cyan][INFO][/cyan] Threshold {args.threshold} found to correspond to index {threshold_idx}"
    )
    rprint(
        f"[cyan][INFO][/cyan] DayNight smoothing method={smoothing_info['SmoothingMethod']} sigma={smoothing_info.get('SmoothingSigma', 0.0):.2f} enabled={smoothing_info['SmoothingEnabled']}"
    )

for config, name, energy in product(args.config, args.name, args.energy):
    info = json.loads(open(f"{root}/config/{config}/{config}_config.json").read())
    detector_mass = get_full_detector_mass(config, info)

    sigmas = []
    significance_bins = []
    plot_df = pd.read_pickle(
        f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/signal/{args.folder.lower()}/DAYNIGHT/{config}/{name}/{config}_{name}_{energy}_Rebin.pkl"
    )
    configured_backgrounds = get_background_samples(str(root), "DAYNIGHT")
    loaded_backgrounds = []
    for bkg, filepath in load_available_background_dataframes(str(root), "DAYNIGHT", args.folder, config, energy):
        bkg_df = pd.read_pickle(filepath)
        plot_df = pd.concat([plot_df, bkg_df], ignore_index=True)
        loaded_backgrounds.append(bkg)
    components = [
        bkg for bkg in configured_backgrounds if bkg in loaded_backgrounds
    ] + ["Solar"]

    sigmamax = 0.0
    last_sigma2 = 1e6
    last_sigma3 = 1e6

    _dn_tag = "+".join(_dn_active_metrics[:2]) + ("..." if len(_dn_active_metrics) > 2 else "")
    for nhit, ophit, adjcl in track(
        product(
            nhits,
            nhits[3:],
            nhits[::-1],
        ),
        description=f"DayNight [{_dn_tag}] {config} {energy}...",
        total=len(nhits) * len(nhits[3:]) * len(nhits[::-1]),
    ):
        nhit_value = int(nhit)
        ophit_value = int(ophit)
        adjcl_value = int(adjcl)
        this_df = plot_df.loc[
            (plot_df["NHits"] == nhit_value)
            * (plot_df["OpHits"] == ophit_value)
            * (plot_df["AdjCl"] == adjcl_value)
        ]
        if this_df.empty:
            if args.debug:
                rprint(
                    f"[yellow][WARNING] No data for {energy} with {nhit_value} nhits and {ophit_value} ophits {adjcl_value} adjcl[/yellow]"
                )
            continue

        raw_background = np.zeros(len(daynight_rebin_centers) - threshold_idx, dtype=float)
        smoothed_background = np.zeros(len(daynight_rebin_centers) - threshold_idx, dtype=float)
        raw_signal_day = None
        raw_signal_night = None
        smoothed_signal_day = None
        smoothed_signal_night = None
        mc_sums = {}

        for component in components:
            comp_df = this_df.loc[this_df["Component"] == component].copy()
            if comp_df.empty:
                if args.debug:
                    rprint(
                        f"[yellow][WARNING] No data for {component} in {energy} with {nhit_value} nhits, {ophit_value} ophits, {adjcl_value} adjcl[/yellow]"
                    )
                continue
            comp_df = comp_df.fillna(0)
            component_smoothing_config = get_component_smoothing_config(smoothing_config, component)
            if component == "Solar":
                if "Oscillation" in comp_df.columns:
                    comp_df = comp_df.loc[comp_df["Oscillation"] == "Osc"]
                day_df = comp_df.loc[comp_df["Mean"] == "Day"]
                night_df = comp_df.loc[comp_df["Mean"] == "Night"]
                if day_df.empty or night_df.empty:
                    rprint(
                        f"[yellow][WARNING] Missing Solar Day/Night rows in {energy} with {nhit_value} nhits, {ophit_value} ophits, {adjcl_value} adjcl[/yellow]"
                    )
                    continue
                raw_signal_day = np.asarray(day_df["Counts"].values[0][threshold_idx:], dtype=float)
                raw_signal_night = np.asarray(night_df["Counts"].values[0][threshold_idx:], dtype=float)
                smoothed_signal_day = smooth_histogram_with_config(raw_signal_day, component_smoothing_config)
                smoothed_signal_night = smooth_histogram_with_config(raw_signal_night, component_smoothing_config)
            else:
                if "Oscillation" in comp_df.columns:
                    comp_df = comp_df.loc[comp_df["Oscillation"] == "Truth"]
                if "Mean" in comp_df.columns:
                    comp_df = comp_df.loc[comp_df["Mean"] == "Mean"]
                if comp_df.empty:
                    rprint(
                        f"[yellow][WARNING] Missing Truth/Mean rows for {component} in {energy} with {nhit_value} nhits, {ophit_value} ophits, {adjcl_value} adjcl[/yellow]"
                    )
                    continue
                raw_counts = np.sum(
                    np.asarray(
                        [
                            np.asarray(row["Counts"][threshold_idx:], dtype=float)
                            for _, row in comp_df.iterrows()
                        ]
                    ),
                    axis=0,
                )
                raw_background += raw_counts
                smoothed_background += smooth_histogram_with_config(raw_counts, component_smoothing_config)
                if "MCCounts" in comp_df.columns:
                    mc_counts = np.sum(
                        np.asarray(
                            [np.asarray(row["MCCounts"][threshold_idx:], dtype=float) for _, row in comp_df.iterrows()],
                        ),
                        axis=0,
                    )
                    mc_sums[component] = float(np.sum(mc_counts))

        if raw_signal_day is None or raw_signal_night is None or smoothed_signal_day is None or smoothed_signal_night is None:
            continue

        has_required_background_mc = all(
            mc_sums.get(component, 0.0) >= args.mc_threshold
            for component in essential_background_components
        )
        if not has_required_background_mc:
            continue

        found_sigma2 = False
        found_sigma3 = False
        sigma2 = 0.0
        sigma3 = 0.0
        sigma2_curve = []
        sigma3_curve = []

        found_asimov_sigma2 = False
        found_asimov_sigma3 = False
        asimov_sigma2 = 0.0
        asimov_sigma3 = 0.0
        asimov_sigma2_curve = []
        asimov_sigma3_curve = []

        raw_gaussian_significances = [[], [], []]
        raw_gaussian_error_significances = [[], [], []]
        smoothed_gaussian_significances = [[], [], []]
        smoothed_gaussian_error_significances = [[], [], []]
        raw_asimov_significances = [[], [], []]
        smoothed_asimov_significances = [[], [], []]

        # Backgrounds are assumed time-uniform: equal exposure in daytime and nighttime.
        # This holds for cosmogenic, geological, and detector-intrinsic backgrounds
        # that have no coupling to the solar zenith angle.
        # args.day_fraction is the fraction of exposure in daytime (default 0.5).
        day_fraction = args.day_fraction

        night_fraction = 1.0 - day_fraction

        for years in exposure_grid:
            factor = years * detector_mass

            # Smoothed effective background — always needed (main Gaussian metric).
            smoothed_night_counts = factor * (night_fraction * smoothed_background + smoothed_signal_night)
            smoothed_day_counts   = factor * (day_fraction   * smoothed_background + smoothed_signal_day)
            smoothed_background_effective = (
                smoothed_night_counts / night_fraction ** 2
                + smoothed_day_counts  / day_fraction  ** 2
            )

            # Raw effective background — only needed for raw variants, error bands, or raw Asimov.
            _need_raw_eff = _compute_raw or _compute_error_bands
            if _need_raw_eff:
                raw_night_counts = factor * (night_fraction * raw_background + raw_signal_night)
                raw_day_counts   = factor * (day_fraction  * raw_background + raw_signal_day)
                raw_background_effective = (
                    raw_night_counts / night_fraction ** 2
                    + raw_day_counts  / day_fraction  ** 2
                )

            # Background uncertainties (error bands) — only when explicitly enabled.
            # Three independent sources per period: Poisson statistical, normalisation
            # systematic, and day-fraction uncertainty, combined in quadrature.
            if _compute_error_bands:
                raw_night_bkg = factor * night_fraction * raw_background
                raw_day_bkg   = factor * day_fraction   * raw_background
                raw_sigma_night = np.sqrt(
                    raw_night_bkg
                    + (args.background_uncertainty * raw_night_bkg) ** 2
                    + (args.day_fraction_band * factor * raw_background) ** 2
                )
                raw_sigma_day = np.sqrt(
                    raw_day_bkg
                    + (args.background_uncertainty * raw_day_bkg) ** 2
                    + (args.day_fraction_band * factor * raw_background) ** 2
                )
                raw_bkg_uncertainty_eff = np.where(
                    raw_background_effective > 0,
                    np.sqrt(
                        (raw_sigma_night / night_fraction) ** 2
                        + (raw_sigma_day  / day_fraction)  ** 2
                    ),
                    0.0,
                )

                smoothed_night_bkg = factor * night_fraction * smoothed_background
                smoothed_day_bkg   = factor * day_fraction   * smoothed_background
                smoothed_sigma_night = np.sqrt(
                    smoothed_night_bkg
                    + (args.background_uncertainty * smoothed_night_bkg) ** 2
                    + (args.day_fraction_band * factor * smoothed_background) ** 2
                )
                smoothed_sigma_day = np.sqrt(
                    smoothed_day_bkg
                    + (args.background_uncertainty * smoothed_day_bkg) ** 2
                    + (args.day_fraction_band * factor * smoothed_background) ** 2
                )
                smoothed_bkg_uncertainty_eff = np.where(
                    smoothed_background_effective > 0,
                    np.sqrt(
                        (smoothed_sigma_night / night_fraction) ** 2
                        + (smoothed_sigma_day  / day_fraction)  ** 2
                    ),
                    0.0,
                )

            for kdx, asymmetry_scale in enumerate(asymmetry_scales):
                # ── Optional: raw Gaussian + raw error bands ──────────────────
                if _compute_raw or _compute_error_bands:
                    raw_signal = factor * asymmetry_scale * (raw_signal_night - raw_signal_day)
                    raw_signal = np.where(raw_background_effective == 0, 0, raw_signal)

                if _compute_error_bands:
                    raw_gaussian_error = evaluate_significance(
                        raw_signal, raw_background_effective,
                        background_uncertainty=raw_bkg_uncertainty_eff, type="gaussian",
                    )
                    raw_gaussian_error = np.nan_to_num(raw_gaussian_error, nan=0.0)
                    raw_gaussian_error_significances[kdx].append(
                        float(np.sqrt(np.sum(np.power(raw_gaussian_error, 2))))
                    )
                else:
                    raw_gaussian_error_significances[kdx].append(0.0)

                if _compute_raw:
                    raw_gaussian = evaluate_significance(
                        raw_signal, raw_background_effective, type="gaussian",
                    )
                    raw_gaussian = np.nan_to_num(raw_gaussian, nan=0.0)
                    raw_gaussian_significances[kdx].append(
                        float(np.sqrt(np.sum(np.power(raw_gaussian, 2))))
                    )
                else:
                    raw_gaussian_significances[kdx].append(0.0)

                # ── Main: smoothed Gaussian (always) ──────────────────────────
                smoothed_signal = factor * asymmetry_scale * (
                    smoothed_signal_night - smoothed_signal_day
                )
                smoothed_signal = np.where(smoothed_background_effective == 0, 0, smoothed_signal)

                if _compute_error_bands:
                    smoothed_gaussian_error = evaluate_significance(
                        smoothed_signal, smoothed_background_effective,
                        background_uncertainty=smoothed_bkg_uncertainty_eff, type="gaussian",
                    )
                    smoothed_gaussian_error = np.nan_to_num(smoothed_gaussian_error, nan=0.0)
                    smoothed_gaussian_error_significances[kdx].append(
                        float(np.sqrt(np.sum(np.power(smoothed_gaussian_error, 2))))
                    )
                else:
                    smoothed_gaussian_error_significances[kdx].append(0.0)

                smoothed_gaussian = evaluate_significance(
                    smoothed_signal, smoothed_background_effective, type="gaussian",
                )
                smoothed_gaussian = np.nan_to_num(smoothed_gaussian, nan=0.0)
                smoothed_gaussian_significances[kdx].append(
                    float(np.sqrt(np.sum(np.power(smoothed_gaussian, 2))))
                )

                # ── Optional: Asimov two-sample Poisson LLR ───────────────────
                # H0 (no asymmetry): h0_n = g*(n_n+n_d), h0_d = f*(n_n+n_d)
                if _compute_asimov:
                    raw_signal_night_k = raw_signal_day + asymmetry_scale * (raw_signal_night - raw_signal_day)
                    raw_n_night_k = factor * (night_fraction * raw_background + raw_signal_night_k)
                    raw_n_day_k   = factor * (day_fraction   * raw_background + raw_signal_day)
                    raw_total_k   = raw_n_night_k + raw_n_day_k
                    _raw_mask = (raw_n_night_k > 0) & (raw_n_day_k > 0) & (raw_total_k > 0)
                    raw_llr = np.zeros_like(raw_n_night_k)
                    raw_llr[_raw_mask] = 2.0 * (
                        raw_n_night_k[_raw_mask] * np.log(raw_n_night_k[_raw_mask] / (night_fraction * raw_total_k[_raw_mask]))
                        + raw_n_day_k[_raw_mask] * np.log(raw_n_day_k[_raw_mask]   / (day_fraction   * raw_total_k[_raw_mask]))
                    )
                    raw_llr = np.nan_to_num(raw_llr, nan=0.0, posinf=0.0, neginf=0.0)
                    raw_asimov_significances[kdx].append(float(np.sqrt(max(float(np.sum(raw_llr)), 0.0))))

                    smoothed_signal_night_k = smoothed_signal_day + asymmetry_scale * (smoothed_signal_night - smoothed_signal_day)
                    smoothed_n_night_k = factor * (night_fraction * smoothed_background + smoothed_signal_night_k)
                    smoothed_n_day_k   = factor * (day_fraction   * smoothed_background + smoothed_signal_day)
                    smoothed_total_k   = smoothed_n_night_k + smoothed_n_day_k
                    _sm_mask = (smoothed_n_night_k > 0) & (smoothed_n_day_k > 0) & (smoothed_total_k > 0)
                    smoothed_llr = np.zeros_like(smoothed_n_night_k)
                    smoothed_llr[_sm_mask] = 2.0 * (
                        smoothed_n_night_k[_sm_mask] * np.log(smoothed_n_night_k[_sm_mask] / (night_fraction * smoothed_total_k[_sm_mask]))
                        + smoothed_n_day_k[_sm_mask] * np.log(smoothed_n_day_k[_sm_mask]   / (day_fraction   * smoothed_total_k[_sm_mask]))
                    )
                    smoothed_llr = np.nan_to_num(smoothed_llr, nan=0.0, posinf=0.0, neginf=0.0)
                    smoothed_asimov_significances[kdx].append(float(np.sqrt(max(float(np.sum(smoothed_llr)), 0.0))))
                else:
                    raw_asimov_significances[kdx].append(0.0)
                    smoothed_asimov_significances[kdx].append(0.0)

            if smoothed_gaussian_significances[1][-1] > sigmamax:
                sigmamax = smoothed_gaussian_significances[1][-1]

            if smoothed_gaussian_significances[1][-1] > 2 and not found_sigma2:
                sigma2 = factor
                found_sigma2 = True
                if sigma2 < last_sigma2 and args.debug:
                    rprint(
                        f"Found smoothed sigma2 with exposure {factor:.0f} for nhits {nhit_value} ophits {ophit_value} and adjcls {adjcl_value}"
                    )
                if sigma2 < last_sigma2:
                    last_sigma2 = sigma2

            if smoothed_gaussian_significances[1][-1] > 3 and not found_sigma3:
                sigma3 = factor
                found_sigma3 = True
                if sigma3 < last_sigma3 and args.debug:
                    rprint(
                        f"Found smoothed sigma3 with exposure {factor:.0f} for nhits {nhit_value} ophits {ophit_value} and adjcls {adjcl_value}"
                    )
                if sigma3 < last_sigma3:
                    last_sigma3 = sigma3

            sigma2_curve.append(sigma2)
            sigma3_curve.append(sigma3)

            if _compute_asimov:
                if smoothed_asimov_significances[1][-1] > 2 and not found_asimov_sigma2:
                    asimov_sigma2 = factor
                    found_asimov_sigma2 = True
                if smoothed_asimov_significances[1][-1] > 3 and not found_asimov_sigma3:
                    asimov_sigma3 = factor
                    found_asimov_sigma3 = True
            asimov_sigma2_curve.append(asimov_sigma2)
            asimov_sigma3_curve.append(asimov_sigma3)

        crossing_summary = compute_crossing_summary(
            exposure_grid,
            np.array(raw_gaussian_significances[1], dtype=float),
            np.array(smoothed_gaussian_significances[1], dtype=float),
        )
        asimov_crossing_raw = compute_crossing_summary(
            exposure_grid,
            np.array(raw_asimov_significances[1], dtype=float),
            np.array(smoothed_asimov_significances[1], dtype=float),
        )
        asimov_crossing_summary = {f"Asimov{k}": v for k, v in asimov_crossing_raw.items()}

        # Per-bin display spectra — only computed when significance_bins is active.
        _n_bins_display = len(daynight_rebin_centers) - threshold_idx
        if _compute_significance_bins:
            factor_display = args.exposure * detector_mass
            raw_night_display   = factor_display * (night_fraction * raw_background + raw_signal_night)
            raw_day_display     = factor_display * (day_fraction   * raw_background + raw_signal_day)
            raw_background_effective_display = (
                raw_night_display / night_fraction ** 2
                + raw_day_display  / day_fraction  ** 2
            )
            raw_signal_display = factor_display * (raw_signal_night - raw_signal_day)
            raw_signal_display = np.where(raw_background_effective_display == 0, 0, raw_signal_display)
            raw_gaussian_spectrum = np.nan_to_num(
                evaluate_significance(raw_signal_display, raw_background_effective_display, type="gaussian"),
                nan=0.0, posinf=0.0, neginf=0.0,
            )

            smoothed_night_display = factor_display * (night_fraction * smoothed_background + smoothed_signal_night)
            smoothed_day_display   = factor_display * (day_fraction   * smoothed_background + smoothed_signal_day)
            smoothed_background_effective_display = (
                smoothed_night_display / night_fraction ** 2
                + smoothed_day_display  / day_fraction  ** 2
            )
            smoothed_signal_display = factor_display * (smoothed_signal_night - smoothed_signal_day)
            smoothed_signal_display = np.where(smoothed_background_effective_display == 0, 0, smoothed_signal_display)
            smoothed_gaussian_spectrum = np.nan_to_num(
                evaluate_significance(smoothed_signal_display, smoothed_background_effective_display, type="gaussian"),
                nan=0.0, posinf=0.0, neginf=0.0,
            )

            # Asimov per-bin display spectrum (nominal asymmetry_scale=1)
            _raw_dm = (raw_night_display > 0) & (raw_day_display > 0) & ((raw_night_display + raw_day_display) > 0)
            raw_asimov_spectrum = np.zeros_like(raw_night_display)
            raw_asimov_spectrum[_raw_dm] = np.sqrt(np.maximum(
                2.0 * (
                    raw_night_display[_raw_dm] * np.log(raw_night_display[_raw_dm] / (night_fraction * (raw_night_display[_raw_dm] + raw_day_display[_raw_dm])))
                    + raw_day_display[_raw_dm] * np.log(raw_day_display[_raw_dm]   / (day_fraction   * (raw_night_display[_raw_dm] + raw_day_display[_raw_dm])))
                ),
                0.0,
            ))
            raw_asimov_spectrum = np.nan_to_num(raw_asimov_spectrum, nan=0.0, posinf=0.0, neginf=0.0)

            _sm_dm = (smoothed_night_display > 0) & (smoothed_day_display > 0) & ((smoothed_night_display + smoothed_day_display) > 0)
            smoothed_asimov_spectrum = np.zeros_like(smoothed_night_display)
            smoothed_asimov_spectrum[_sm_dm] = np.sqrt(np.maximum(
                2.0 * (
                    smoothed_night_display[_sm_dm] * np.log(smoothed_night_display[_sm_dm] / (night_fraction * (smoothed_night_display[_sm_dm] + smoothed_day_display[_sm_dm])))
                    + smoothed_day_display[_sm_dm] * np.log(smoothed_day_display[_sm_dm]   / (day_fraction   * (smoothed_night_display[_sm_dm] + smoothed_day_display[_sm_dm])))
                ),
                0.0,
            ))
            smoothed_asimov_spectrum = np.nan_to_num(smoothed_asimov_spectrum, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            raw_gaussian_spectrum     = np.zeros(_n_bins_display, dtype=float)
            smoothed_gaussian_spectrum = np.zeros(_n_bins_display, dtype=float)
            raw_asimov_spectrum        = np.zeros(_n_bins_display, dtype=float)
            smoothed_asimov_spectrum   = np.zeros(_n_bins_display, dtype=float)

        if smoothing_info["SmoothingReport"] and args.debug and explicit_debug_flag:
            rprint(
                f"[cyan][REPORT][/cyan] {config} {name} {energy} NHits={nhit_value} OpHits={ophit_value} AdjCl={adjcl_value} "
                f"sigma2 raw={crossing_summary['RawSigma2Crossing']:.2f} smooth={crossing_summary['SmoothedSigma2Crossing']:.2f}, "
                f"sigma3 raw={crossing_summary['RawSigma3Crossing']:.2f} smooth={crossing_summary['SmoothedSigma3Crossing']:.2f}"
            )

        sigmas.append(
            {
                "Config": config,
                "Name": name,
                "Energy": energy,
                "Sigma2": sigma2_curve,
                "Sigma3": sigma3_curve,
                "AsimovSigma2": asimov_sigma2_curve,
                "AsimovSigma3": asimov_sigma3_curve,
                "Exposure": exposure_grid.tolist(),
                "NHits": nhit_value,
                "OpHits": ophit_value,
                "AdjCl": adjcl_value,
                "ErrorGaussian+Error": smoothed_gaussian_error_significances[0],
                "ErrorGaussian": smoothed_gaussian_error_significances[1],
                "ErrorGaussian-Error": smoothed_gaussian_error_significances[2],
                "Gaussian+Error": smoothed_gaussian_significances[0],
                "Gaussian": smoothed_gaussian_significances[1],
                "Gaussian-Error": smoothed_gaussian_significances[2],
                "RawErrorGaussian+Error": raw_gaussian_error_significances[0],
                "RawErrorGaussian": raw_gaussian_error_significances[1],
                "RawErrorGaussian-Error": raw_gaussian_error_significances[2],
                "RawGaussian+Error": raw_gaussian_significances[0],
                "RawGaussian": raw_gaussian_significances[1],
                "RawGaussian-Error": raw_gaussian_significances[2],
                "Asimov+Error": smoothed_asimov_significances[0],
                "Asimov": smoothed_asimov_significances[1],
                "Asimov-Error": smoothed_asimov_significances[2],
                "RawAsimov+Error": raw_asimov_significances[0],
                "RawAsimov": raw_asimov_significances[1],
                "RawAsimov-Error": raw_asimov_significances[2],
                "EarthDensityBand": args.earth_density_band,
                "OscillationBand": args.oscillation_band,
                "TotalAsymmetryBand": total_asymmetry_band,
                "DayFraction": args.day_fraction,
                "DayFractionBand": args.day_fraction_band,
                "BackgroundUncertainty": args.background_uncertainty,
                **smoothing_info,
                **crossing_summary,
                **asimov_crossing_summary,
            }
        )

        if _compute_significance_bins:
            significance_energy = np.asarray(daynight_rebin_centers[threshold_idx:], dtype=float)
            for bin_idx, (energy_value, raw_gauss, smooth_gauss, raw_asimov, smooth_asimov) in enumerate(
                zip(significance_energy, raw_gaussian_spectrum, smoothed_gaussian_spectrum, raw_asimov_spectrum, smoothed_asimov_spectrum)
            ):
                significance_bins.append(
                    {
                        "Config": config,
                        "Name": name,
                        "EnergyLabel": energy,
                        "NHits": nhit_value,
                        "OpHits": ophit_value,
                        "AdjCl": adjcl_value,
                        "Threshold": float(args.threshold),
                        "ExposureYears": float(args.exposure),
                        "BinIndex": int(bin_idx),
                        "RecoEnergy": float(energy_value),
                        "RawGaussian": float(raw_gauss),
                        "Gaussian": float(smooth_gauss),
                        "RawAsimov": float(raw_asimov),
                        "Asimov": float(smooth_asimov),
                        **smoothing_info,
                    }
                )

    if args.debug:
        rprint(f"Maximum significance for {energy}: {sigmamax:.2f} sigma")
    sigmas_df = pd.DataFrame(sigmas)
    sigmas_df = merge_with_existing_df(
        sigmas_df,
        f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/DAYNIGHT/{args.folder.lower()}",
        config=config,
        name=name,
        filename=f"{energy}_DayNight_Results",
        key_cols=["Config", "Name", "Energy", "NHits", "OpHits", "AdjCl"],
        debug=args.debug,
    )
    save_df(
        sigmas_df,
        f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/DAYNIGHT/{args.folder.lower()}",
        config=config,
        name=name,
        filename=f"{energy}_DayNight_Results",
        rm=args.rewrite,
        debug=args.debug,
    )
    if _compute_significance_bins:
        significance_bins_df = pd.DataFrame(significance_bins)
        save_df(
            significance_bins_df,
            f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/DAYNIGHT/{args.folder.lower()}",
            config=config,
            name=name,
            filename=f"{energy}_DayNight_SignificanceBins",
            rm=args.rewrite,
            debug=args.debug,
        )
