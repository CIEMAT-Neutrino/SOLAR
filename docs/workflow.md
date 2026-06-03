# Workflow Guide

The repository is organized as a staged pipeline. Each stage writes reusable products into `data/` and diagnostic plots into `images/`, so later stages can be rerun without always restarting from the ROOT inputs.

## Detector And Reconstruction Stages

- `src/physics/calibration/`: correction, calibration, discrimination, reconstruction, smearing.
- `src/physics/detector/preselection/`: production-level summaries, efficiencies, and clustering studies.
- `src/physics/detector/pds/`: optical-flash studies, adjacent-flash diagnostics, and flash-matching efficiency.
- `src/physics/detector/tpc/`: adjacent-cluster studies and electron/energy-resolution scans.
- `src/physics/detector/vertex/`: vertex smearing, fiducial scans, and reconstruction performance.

The usual top-level entry point for the full detector chain is:

```bash
python3 src/pipelines/run_all.py --config hd_1x2x6_centralAPA --name marley
```

If calibration products already exist, start from the downstream detector-analysis chain with:

```bash
python3 src/pipelines/run_analysis.py --config hd_1x2x6_centralAPA --name marley
```

## Truth And Weighting Stages

The `src/physics/truth/` scripts prepare the signal and background ingredients used by the later solar analyses, including oscillation-grid processing, nadir weighting, and background surface or external PDFs. Entry point: `src/pipelines/run_truth.py`.

## Solar Analysis Stages

The current analysis layer lives in `src/physics/` (per-domain subdirectories) and is orchestrated by `src/pipelines/run_sensitivity.py`.

Important scripts include:

- `src/physics/signal/01_fiducialize.py`: builds fiducial scan products.
- `src/physics/signal/02_best_fiducial.py`: selects optimized fiducials and writes best-fiducial summaries.
- `src/physics/signal/fiducialization_plot.py`: renders best/no-fiducial significance plots from fiducial scan products.
- `src/physics/sensitivity/05_best_sigmas.py`: records the best significance curves for downstream plots.
- `src/pipelines/run_sensitivity.py`: orchestrates the full DayNight, HEP, and Sensitivity workflow.
- `src/physics/daynight/01_daynight.py`, `exposure_plot.py`, `significance_plot.py`: Day-Night spectrum, exposure, and significance products.
- `src/physics/hep/01_hep.py`, `exposure_plot.py`, `significance_plot.py`, `significance_comparison.py`: HEP spectrum, exposure, and significance products.
- `src/physics/sensitivity/01_background_template.py`, `02_signal_template.py`, `06_significance.py`, `contour_plot.py`: signal/background templates, oscillation fits, and contour plots.

## Shared Configuration

Analysis defaults are centralized in the `analysis/` directory (split JSON files merged at runtime by `lib/defaults.load_analysis_info`):

- `analysis/config.json`: workflow flags, analysis thresholds, adaptive rebinning, background component policy.
- `analysis/smoothing.json`: Gaussian smoothing parameters per analysis, energy, and stage.
- `analysis/fiducialization.json`: fiducialization settings for DayNight, HEP, and Sensitivity.
- `analysis/backgrounds.json`: background component lists and truth-pipeline defaults.
- `analysis/physics.json`: oscillation parameters and detector geometry defaults.

Those settings are consumed by helpers in `lib/smoothing.py`, `lib/fiducial.py`, and `lib/defaults.py`.

## HEP Profile-Likelihood Updates

Full mathematical derivations of all significance methods, the profile-likelihood formulation, adaptive rebinning, Barlow-Beeston masking, PL smoothing pipeline, and spike detection are in [`docs/hep_likelihood_derivation.tex`](hep_likelihood_derivation.tex).

Three improvements to the HEP profile-likelihood significance computation:

**Monotonicity enforcement and continuous smoothing** (`src/physics/hep/01_hep.py`): the post-scan running maximum (`np.maximum.accumulate`) has been replaced by a two-step pipeline. First, [`scipy.ndimage.gaussian_filter1d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html) convolves the raw PL significance array with a Gaussian kernel (σ = 6 exposure-grid index units by default, tunable via `_PL_SMOOTH_SIGMA`). This is the standard HEP technique for smoothing discrete numerical curves, equivalent to [ROOT's `TH1::Smooth`](https://root.cern.ch/doc/master/classTH1.html#a16), and removes solver oscillations at low signal-to-background ratios. Second, [`sklearn.isotonic.IsotonicRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html) (PAVA) enforces strict monotonicity by finding the non-decreasing sequence with minimum L2 distance from the smoothed values. At low signal-to-background ratios this pipeline converts a stepped curve with few unique significance values to a fully continuous one.

**±1σ expected discovery bands** (`src/physics/hep/01_hep.py`): the PL error band uses **signal normalization variation** — signal events are scaled by `(1 ± σ_s)` where `σ_s = signal_uncertainty` (reconstruction efficiency systematic, passed via `--signal_uncertainty`, typically 0.1). The background is never shifted, so the β̂_null nuisance parameter is unaffected and both bands collapse symmetrically when signal is negligible. Bands converge to the nominal line when signal is negligible (±σ_s × 0 = 0 change) and remain narrow and symmetric otherwise. The PL computation uses the original fine binning throughout — no adaptive rebin — since PL is optimal at the finest available resolution and the likelihood ratio naturally suppresses bins with negligible signal.

**Barlow-Beeston MC mask and smoothing clip** (`src/physics/hep/01_hep.py`): two numerical robustness measures protect the PL LLR from divergence. First, a static per-bin mask (`pl_bin_mask`) zeros signal and background in bins whose raw background MC count is below `min_mc_per_bin` (configurable). This is the [Barlow-Beeston lite approach](https://www.sciencedirect.com/science/article/pii/009350659390005W) used by [ROOT HistFactory](https://root.cern.ch/doc/master/classRooStats_1_1HistFactory_1_1Measurement.html): bins with no MC support produce unreliable LLR terms that grow super-linearly with exposure, so they are excluded rather than floored. Second, histogram rates after Gaussian smoothing are clipped to `≥ 0` before the PL computation. Gaussian smoothing at distribution edges can produce small negative rate values; after the `min_expected` floor these appear as effectively signal-only bins whose LLR grows as `signal × log(signal/floor)` — diverging at large exposure. Clipping removes this artifact without affecting the unsmoothed path.

**Spike-robust best-cut selection** (`src/physics/sensitivity/05_best_sigmas.py`): PL curves that pass spike detection are excluded from the `max(significance)` cut selection. A curve is flagged if any consecutive step in `RawPreIsotonicProfileLikelihood` or `PreIsotonicProfileLikelihood` exceeds `--max_pl_jump` σ (default 1.0). Spiked cuts are saved separately to `{config}_{name}_highest_spiked_HEP.pkl` for diagnostic review. If all cuts for a given (config, name, energy) are spiked, the filter is bypassed with a warning to preserve output. `--max_pl_jump 0` disables filtering entirely (backward-compatible).

**Fastest-sigma cut selection** (`src/physics/sensitivity/05_best_sigmas.py`): when the significance reference is `ProfileLikelihood`, the `fastest_sigma2` and `fastest_sigma3` cut selection now uses `PLSigma2` / `PLSigma3` (the exposure at which the PL significance crosses 2σ / 3σ) instead of the Asimov-based `Sigma2` / `Sigma3`. Backward-compatible fallback to Asimov columns applies for output files that pre-date the PL crossing columns.

## Orchestrator Flags

`src/pipelines/run_sensitivity.py` exposes boolean flags to skip stages without re-running the full pipeline:

| Flag | Default | Effect when disabled |
| ---- | ------- | -------------------- |
| `--computation` / `--no-computation` | True | Skip all computation; run only plot-producing macros |
| `--significance` / `--no-significance` | True | Skip `01_daynight.py`, `01_hep.py`, `06_significance.py` |
| `--fiducialization` / `--no-fiducialization` | True | Skip `signal/01_fiducialize.py` and `signal/02_best_fiducial.py` |
| `--rebin` / `--no-rebin` | True | Skip `signal/03_analysis.py` adaptive rebinning |

Flag precedence: `--no-computation` → `--no-significance` → `--no-fiducialization` → `--no-rebin`. Presentation generation always runs regardless of computation flags. `sensitivity/05_best_sigmas.py` runs when `--computation` is enabled, independently of `--significance`.

## Component Policy In Analysis Orchestration

The high-level orchestrator `src/pipelines/run_sensitivity.py` applies a component-selection policy before launching per-sample analysis jobs.

The policy is configured in `analysis/config.json` (or `analysis/backgrounds.json`) under `BACKGROUND_SAMPLES`:

- `ANALYSES`: per-analysis background component lists (`DAYNIGHT`, `HEP`, `SENSITIVITY`).
- `ESSENTIAL`: map of components that must be present (`true`) vs optional (`false`).

Runtime behavior:

- Non-essential components not listed in the selected analysis component list are not processed.
- Essential components that are missing on disk produce warnings.
- Optional components that are missing are skipped with warnings.

This avoids failures when optional backgrounds (for example `radiological`) are unavailable for a given detector configuration while still protecting required components.
