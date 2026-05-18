# Workflow Guide

The repository is organized as a staged pipeline. Each stage writes reusable products into `data/` and diagnostic plots into `images/`, so later stages can be rerun without always restarting from the ROOT inputs.

## Detector And Reconstruction Stages

- `src/workflow/`: correction, calibration, discrimination, reconstruction, smearing, and the aggregated drivers `00AllWorkflow.py` and `10Workflow.py`.
- `src/preselection/`: production-level summaries, efficiencies, and clustering studies.
- `src/PDS/`: optical-flash studies, adjacent-flash diagnostics, and flash-matching efficiency.
- `src/TPC/`: adjacent-cluster studies and electron/energy-resolution scans.
- `src/vertex/`: vertex smearing, fiducial scans, and reconstruction performance.

The usual top-level entry point is:

```bash
python3 src/00All.py --config hd_1x2x6_centralAPA --name marley
```

If the base workflow products already exist, you can start from the downstream detector-analysis chain with:

```bash
python3 src/01Analysis.py --config hd_1x2x6_centralAPA --name marley
```

## Truth And Weighting Stages

The `src/truth/` scripts prepare the signal and background ingredients used by the later solar analyses, including oscillation-grid processing, azimuth weighting, and background surface or external PDFs.

## Solar Analysis Stages

The current analysis layer lives in `src/analysis/`.

Important scripts include:

- `0XFiducializeSignal.py`: builds fiducial scan products.
- `0YBestFiducial.py`: selects optimized fiducials using analysis-specific significance settings and writes best-fiducial summaries.
- `10FiducializationPlot.py`: renders best/no-fiducial significance plots from fiducial scan products.
- `0ZBestSigmas.py`: records the best significance curves for downstream plots.
- `10SensitivityAnalysis.py`: orchestrates the full DayNight, HEP, and Sensitivity workflow.
- `12DayNight*.py`: Day-Night spectrum, exposure, and significance products.
- `13HEP*.py`: HEP spectrum, exposure, and significance products.
- `14Sensitivity*.py`: signal/background templates, oscillation fits, and contour plots.

## Shared Configuration

Analysis defaults are centralized in `import/analysis.json`. Recent additions there include:

- default Gaussian smoothing for 1D and 2D histograms
- analysis-specific best-significance references
- fiducialization settings for DayNight, HEP, and Sensitivity
- analysis-stage thresholds via `ANALYSIS_THRESHOLDS` (`DAYNIGHT`, `HEP`, `SENSITIVITY`)
- background component policy controls (`essential` vs `non-essential`)

Those settings are consumed by helpers in `lib/lib_smooth.py`, `lib/lib_fiducial.py`, and `lib/lib_default.py`.

## HEP Profile-Likelihood Updates

Full mathematical derivations of all significance methods, the profile-likelihood formulation, adaptive rebinning, Barlow-Beeston masking, PL smoothing pipeline, and spike detection are in [`docs/hep_likelihood_derivation.tex`](hep_likelihood_derivation.tex).

Three improvements to the HEP profile-likelihood significance computation:

**Monotonicity enforcement and continuous smoothing** (`src/analysis/13HEP.py`): the post-scan running maximum (`np.maximum.accumulate`) has been replaced by a two-step pipeline. First, [`scipy.ndimage.gaussian_filter1d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html) convolves the raw PL significance array with a Gaussian kernel (σ = 6 exposure-grid index units by default, tunable via `_PL_SMOOTH_SIGMA`). This is the standard HEP technique for smoothing discrete numerical curves, equivalent to [ROOT's `TH1::Smooth`](https://root.cern.ch/doc/master/classTH1.html#a16), and removes solver oscillations at low signal-to-background ratios. Second, [`sklearn.isotonic.IsotonicRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html) (PAVA) enforces strict monotonicity by finding the non-decreasing sequence with minimum L2 distance from the smoothed values. At low signal-to-background ratios this pipeline converts a stepped curve with few unique significance values to a fully continuous one.

**±1σ expected discovery bands** (`src/analysis/13HEP.py`): the PL error band uses **signal normalization variation** — signal events are scaled by `(1 ± σ_s)` where `σ_s = signal_uncertainty` (reconstruction efficiency systematic, passed via `--signal_uncertainty`, typically 0.1). The background is never shifted, so the β̂_null nuisance parameter is unaffected and both bands collapse symmetrically when signal is negligible. Bands converge to the nominal line when signal is negligible (±σ_s × 0 = 0 change) and remain narrow and symmetric otherwise. The PL computation uses the original fine binning throughout — no adaptive rebin — since PL is optimal at the finest available resolution and the likelihood ratio naturally suppresses bins with negligible signal.

**Barlow-Beeston MC mask and smoothing clip** (`src/analysis/13HEP.py`): two numerical robustness measures protect the PL LLR from divergence. First, a static per-bin mask (`pl_bin_mask`) zeros signal and background in bins whose raw background MC count is below `min_mc_per_bin` (configurable). This is the [Barlow-Beeston lite approach](https://www.sciencedirect.com/science/article/pii/009350659390005W) used by [ROOT HistFactory](https://root.cern.ch/doc/master/classRooStats_1_1HistFactory_1_1Measurement.html): bins with no MC support produce unreliable LLR terms that grow super-linearly with exposure, so they are excluded rather than floored. Second, histogram rates after Gaussian smoothing are clipped to `≥ 0` before the PL computation. Gaussian smoothing at distribution edges can produce small negative rate values; after the `min_expected` floor these appear as effectively signal-only bins whose LLR grows as `signal × log(signal/floor)` — diverging at large exposure. Clipping removes this artifact without affecting the unsmoothed path.

**Spike-robust best-cut selection** (`src/analysis/0ZBestSigmas.py`): PL curves that pass spike detection are excluded from the `max(significance)` cut selection. A curve is flagged if any consecutive step in `RawPreIsotonicProfileLikelihood` or `PreIsotonicProfileLikelihood` exceeds `--max_pl_jump` σ (default 1.0). Spiked cuts are saved separately to `{config}_{name}_highest_spiked_HEP.pkl` for diagnostic review. If all cuts for a given (config, name, energy) are spiked, the filter is bypassed with a warning to preserve output. `--max_pl_jump 0` disables filtering entirely (backward-compatible).

**Fastest-sigma cut selection** (`src/analysis/0ZBestSigmas.py`): when the significance reference is `ProfileLikelihood`, the `fastest_sigma2` and `fastest_sigma3` cut selection now uses `PLSigma2` / `PLSigma3` (the exposure at which the PL significance crosses 2σ / 3σ) instead of the Asimov-based `Sigma2` / `Sigma3`. Backward-compatible fallback to Asimov columns applies for output files that pre-date the PL crossing columns.

## Orchestrator Flags

`src/analysis/10SensitivityAnalysis.py` exposes boolean flags to skip stages without re-running the full pipeline:

| Flag | Default | Effect when disabled |
|------|---------|----------------------|
| `--computation` / `--no-computation` | True | Skip all computation; run only plot-producing macros |
| `--significance` / `--no-significance` | True | Skip `12DayNight.py`, `13HEP.py`, `14Sensitivity*.py` |
| `--fiducialization` / `--no-fiducialization` | True | Skip `0XFiducializeSignal.py` and `0YBestFiducial.py` |
| `--rebin` / `--no-rebin` | True | Skip `11AnalysisSignal.py` adaptive rebinning |

Flag precedence: `--no-computation` → `--no-significance` → `--no-fiducialization` → `--no-rebin`. Presentation generation always runs regardless of computation flags. `0ZBestSigmas.py` runs when `--computation` is enabled, independently of `--significance`.

## Component Policy In Analysis Orchestration

The high-level orchestrator `src/analysis/10SensitivityAnalysis.py` applies a component-selection policy before launching per-sample analysis jobs.

The policy is configured in `import/analysis.json` under `BACKGROUND_SAMPLES`:

- `ANALYSES`: per-analysis background component lists (`DAYNIGHT`, `HEP`, `SENSITIVITY`).
- `ESSENTIAL`: map of components that must be present (`true`) vs optional (`false`).

Runtime behavior:

- Non-essential components not listed in the selected analysis component list are not processed.
- Essential components that are missing on disk produce warnings.
- Optional components that are missing are skipped with warnings.

This avoids failures when optional backgrounds (for example `radiological`) are unavailable for a given detector configuration while still protecting required components.
