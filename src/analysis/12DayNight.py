import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

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
        "variations (MSW matter effect). The analysis produces three significance curves: "
        "nominal (scale=1.0), upper (1+band), and lower (1-band) Earth model predictions. "
        "The default ±13%% brackets published PREM-based oscillation probability ranges."
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
    "--threshold",
    type=float,
    default=get_analysis_threshold(str(root), "DAYNIGHT", stage="SIGNIFICANCE", fallback=0.0),
)
parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()
if args.debug:
    rprint(args)
explicit_debug_flag = "--debug" in sys.argv and "--no-debug" not in sys.argv

# The signal in the day-night analysis is the oscillation-induced asymmetry
# (N_night - N_day), not the raw neutrino count.  Earth's density profile enters
# via the MSW matter effect, which modifies the effective oscillation length inside
# the Earth and shifts the expected nighttime excess.  The three asymmetry_scales
# bracket the range of predicted asymmetries from published Earth density models:
#   upper  (1 + band): denser profile → stronger matter effect → larger asymmetry
#   nominal (1.0)     : best-fit PREM profile
#   lower  (1 - band): lower-density profile → weaker matter effect → smaller asymmetry
#
# Extension opportunities:
#   - Load per-energy scale factors from file for an energy-dependent matter effect.
#   - Specify N density models directly to produce N significance curves rather than
#     assuming a symmetric ±band approximation.
#   - Use separate --earth_density_band_up / _down arguments when the uncertainty
#     is asymmetric around the nominal prediction.
asymmetry_scales = [1.0 + args.earth_density_band, 1.0, 1.0 - args.earth_density_band]

threshold_idx = np.where(daynight_rebin_centers > args.threshold)[0][0]
exposure_grid = np.logspace(-1, np.log10(args.exposure), 100)
smoothing_config = get_smoothing_config(
    str(root), analysis_name="DAYNIGHT", dimensions="1d", stage="significance"
)
smoothing_info = smoothing_metadata(smoothing_config)
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

    for nhit, ophit, adjcl in track(
        product(
            nhits,
            nhits[3:],
            nhits[::-1],
        ),
        description=f"Looping over analysis cuts for {energy}...",
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

        for component in components:
            comp_df = this_df.loc[this_df["Component"] == component].copy()
            if comp_df.empty:
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

        if raw_signal_day is None or raw_signal_night is None or smoothed_signal_day is None or smoothed_signal_night is None:
            continue

        found_sigma2 = False
        found_sigma3 = False
        sigma2 = 0.0
        sigma3 = 0.0
        sigma2_curve = []
        sigma3_curve = []

        raw_gaussian_significances = [[], [], []]
        raw_gaussian_error_significances = [[], [], []]
        smoothed_gaussian_significances = [[], [], []]
        smoothed_gaussian_error_significances = [[], [], []]

        # Backgrounds are assumed time-uniform: equal exposure in daytime and nighttime.
        # This holds for cosmogenic, geological, and detector-intrinsic backgrounds
        # that have no coupling to the solar zenith angle.
        DAY_FRACTION = 0.5

        for years in exposure_grid:
            factor = years * detector_mass
            for kdx, asymmetry_scale in enumerate(asymmetry_scales):
                # total background seen in the daytime half of the run:
                #   - DAY_FRACTION of the isotropic background
                #   - plus the daytime solar signal (which is background for the asymmetry measurement)
                raw_background_total = factor * (DAY_FRACTION * raw_background + raw_signal_day)
                # The asymmetry_scale brackets Earth density uncertainty via the MSW matter
                # effect: it multiplies only the day-night difference, leaving the total
                # normalization unchanged.
                raw_signal = factor * asymmetry_scale * (raw_signal_night - raw_signal_day)
                raw_signal = np.where(raw_background_total == 0, 0, raw_signal)

                # "ErrorGaussian" includes the Poisson statistical uncertainty on the
                # background count (absolute: sqrt(N_bkg), not relative 1/sqrt(N_bkg)).
                raw_gaussian_error = evaluate_significance(
                    raw_signal,
                    raw_background_total,
                    background_uncertainty=np.where(
                        factor * DAY_FRACTION * raw_background > 0,
                        np.sqrt(factor * DAY_FRACTION * raw_background),
                        0.0,
                    ),
                    type="gaussian",
                )
                raw_gaussian_error = np.nan_to_num(raw_gaussian_error, nan=0.0)
                raw_gaussian_error_significances[kdx].append(
                    float(np.sqrt(np.sum(np.power(raw_gaussian_error, 2))))
                )

                raw_gaussian = evaluate_significance(
                    raw_signal,
                    raw_background_total,
                    type="gaussian",
                )
                raw_gaussian = np.nan_to_num(raw_gaussian, nan=0.0)
                raw_gaussian_significances[kdx].append(
                    float(np.sqrt(np.sum(np.power(raw_gaussian, 2))))
                )

                smoothed_background_total = factor * (
                    DAY_FRACTION * smoothed_background + smoothed_signal_day
                )
                smoothed_signal = factor * asymmetry_scale * (
                    smoothed_signal_night - smoothed_signal_day
                )
                smoothed_signal = np.where(smoothed_background_total == 0, 0, smoothed_signal)

                smoothed_gaussian_error = evaluate_significance(
                    smoothed_signal,
                    smoothed_background_total,
                    background_uncertainty=np.where(
                        factor * DAY_FRACTION * smoothed_background > 0,
                        np.sqrt(factor * DAY_FRACTION * smoothed_background),
                        0.0,
                    ),
                    type="gaussian",
                )
                smoothed_gaussian_error = np.nan_to_num(smoothed_gaussian_error, nan=0.0)
                smoothed_gaussian_error_significances[kdx].append(
                    float(np.sqrt(np.sum(np.power(smoothed_gaussian_error, 2))))
                )

                smoothed_gaussian = evaluate_significance(
                    smoothed_signal,
                    smoothed_background_total,
                    type="gaussian",
                )
                smoothed_gaussian = np.nan_to_num(smoothed_gaussian, nan=0.0)
                smoothed_gaussian_significances[kdx].append(
                    float(np.sqrt(np.sum(np.power(smoothed_gaussian, 2))))
                )

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

        crossing_summary = compute_crossing_summary(
            exposure_grid,
            np.array(raw_gaussian_significances[1], dtype=float),
            np.array(smoothed_gaussian_significances[1], dtype=float),
        )

        # Persist per-bin significance at the configured exposure so plotting macros
        # can render spectra without recomputing significance.
        factor_display = args.exposure * detector_mass
        raw_background_total_display = factor_display * (DAY_FRACTION * raw_background + raw_signal_day)
        raw_signal_display = factor_display * (raw_signal_night - raw_signal_day)
        raw_signal_display = np.where(raw_background_total_display == 0, 0, raw_signal_display)
        raw_gaussian_spectrum = evaluate_significance(
            raw_signal_display,
            raw_background_total_display,
            type="gaussian",
        )
        raw_gaussian_spectrum = np.nan_to_num(
            raw_gaussian_spectrum, nan=0.0, posinf=0.0, neginf=0.0
        )

        smoothed_background_total_display = factor_display * (
            DAY_FRACTION * smoothed_background + smoothed_signal_day
        )
        smoothed_signal_display = factor_display * (
            smoothed_signal_night - smoothed_signal_day
        )
        smoothed_signal_display = np.where(
            smoothed_background_total_display == 0, 0, smoothed_signal_display
        )
        smoothed_gaussian_spectrum = evaluate_significance(
            smoothed_signal_display,
            smoothed_background_total_display,
            type="gaussian",
        )
        smoothed_gaussian_spectrum = np.nan_to_num(
            smoothed_gaussian_spectrum, nan=0.0, posinf=0.0, neginf=0.0
        )

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
                "EarthDensityBand": args.earth_density_band,
                **smoothing_info,
                **crossing_summary,
            }
        )

        significance_energy = np.asarray(daynight_rebin_centers[threshold_idx:], dtype=float)
        for bin_idx, (energy_value, raw_value, smooth_value) in enumerate(
            zip(significance_energy, raw_gaussian_spectrum, smoothed_gaussian_spectrum)
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
                    "RawGaussian": float(raw_value),
                    "Gaussian": float(smooth_value),
                    **smoothing_info,
                }
            )

    if args.debug:
        rprint(f"Maximum significance for {energy}: {sigmamax:.2f} sigma")
    sigmas_df = pd.DataFrame(sigmas)
    save_df(
        sigmas_df,
        f"/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/DAYNIGHT/{args.folder.lower()}",
        config=config,
        name=name,
        filename=f"{energy}_DayNight_Results",
        rm=args.rewrite,
        debug=args.debug,
    )
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
