# Workflow Guide

The repository is organized as a staged pipeline. Each stage writes reusable products into `output/data/` and diagnostic plots into `output/images/`, so later stages can be rerun without always restarting from the ROOT inputs.

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

Key scripts:

- `src/physics/truth/marley_cc_fraction.py`: computes per-PDG CC energy-channel fractions as a function of neutrino energy and writes `{config}_{name}_Neutrino_CC_Fraction.pkl` to `output/data/marley/stacked/`. Replaces the deprecated `TruthMarleyStacked.ipynb` notebook. Consumed by `src/physics/common/line_plot.py` for the kinematic-threshold overlay plot.

## Solar Analysis Stages

The current analysis layer lives in `src/physics/` (per-domain subdirectories) and is orchestrated by `src/pipelines/run_sensitivity.py`.

Important scripts include:

- `src/physics/signal/01_fiducialize.py`: builds fiducial scan products. Applies `MatchedOpFlashPlane == QUALITY_CUTS.OPFLASH_PLANE` quality cut (from `config/analysis/config.json`).
- `src/physics/signal/02_best_fiducial.py`: selects optimized fiducials and writes `output/data/solar/fiducial/{folder}/BestFiducials.json`.
- `src/physics/signal/03_analysis.py`: produces rebinned signal/background arrays (written to PNFS via `SIGNAL_REBIN`) and optional pkl checkpoints. Pass `--save_weighted` to also write per-cut DataFrames to `output/data/solar/weighted/` (off by default).
- `src/physics/signal/fiducialization_plot.py`: renders best/no-fiducial significance plots from fiducial scan products.
- `src/physics/sensitivity/03_template_compute.py`: lightweight orchestrator that calls `01_background_template.py` and `02_signal_template.py` in one command without writing files itself.
- `src/physics/sensitivity/05_best_sigmas.py`: records the best significance curves for downstream plots.
- `src/pipelines/run_sensitivity.py`: orchestrates the full DayNight, HEP, and Sensitivity workflow.
- `src/physics/daynight/01_daynight.py`, `exposure_plot.py`, `significance_plot.py`: Day-Night spectrum, exposure, and significance products. Exposure diagnostic written to `output/data/daynight/`.
- `src/physics/hep/01_hep.py`, `exposure_plot.py`, `significance_plot.py`, `significance_comparison.py`, `rebin_comparison.py`: HEP spectrum, exposure, and significance products. `rebin_comparison.py` (run under `--all_metrics`) compares Pre-PAVA vs Post-PAVA PL curves at the best cut. Adaptive-rebin comparison written to `output/data/hep/`.
- `src/physics/sensitivity/01_background_template.py`, `02_signal_template.py`, `06_significance.py`, `contour_plot.py`: signal/background templates, oscillation fits, and contour plots.

### Quality Cuts

`01_fiducialize.py`, `03_analysis.py`, and `02_signal_template.py` all apply the same `MatchedOpFlashPlane` cut, ensuring signal and background event populations are identical across all analysis stages. The cut value is centrally defined in `config/analysis/config.json` under `QUALITY_CUTS.OPFLASH_PLANE` (default 0) and propagated at runtime via `load_analysis_info()`.

### Smoothing Optimisation

`src/tools/optimize_smoothing.py` scans KDE bandwidth strategies and writes a `{folder}_{energy}_{analysis}_sigma.json` file to `output/data/smoothing/{config}/{name}/`. `run_sensitivity.py` reads these files at launch and exports the recommended sigma values as `SOLAR_SMOOTHING_SIGMA_*` environment variables for child processes.

## Shared Configuration

Analysis defaults are centralized in the `config/analysis/` directory (split JSON files merged at runtime by `lib/defaults.load_analysis_info`):

- `config/analysis/config.json`: workflow flags, analysis thresholds, adaptive rebinning, background component policy, and quality cuts (`QUALITY_CUTS.OPFLASH_PLANE`).
- `config/analysis/smoothing.json`: Gaussian smoothing parameters per analysis, energy, and stage.
- `config/analysis/fiducialization.json`: fiducialization settings for DayNight, HEP, and Sensitivity.
- `config/analysis/backgrounds.json`: background component lists and truth-pipeline defaults.
- `config/analysis/physics.json`: oscillation parameters and detector geometry defaults.
- `config/analysis/pkl_paths.json`: central registry of every pkl and json file the pipeline produces or consumes. Three categories: `INTERMEDIATE` (in PNFS, read by another pipeline stage), `REPRODUCIBILITY` (local checkpoint arrays under `output/data/results/` and `output/data/solar/fiducial/`), and `OUTPUT_ONLY` (local write-once diagnostics). Use this file to trace data provenance, verify path consistency, and identify stale outputs after code changes.

Those settings are consumed by helpers in `lib/smoothing.py`, `lib/fiducial.py`, and `lib/defaults.py`.

## Output Data Index

`output/data/index.json` is a git-tracked nested tree of every file under `output/data/`. It is the canonical discovery mechanism for external repositories that need to locate pipeline artefacts without access to the full `output/data/` tree.

Regenerate after any change to `output/data/`:

```bash
python3 src/tools/generate_data_index.py
```

All other files under `output/data/` are excluded from git (`.gitignore: output/data/**`); `index.json` is tracked via the `!output/data/index.json` exception.

## HEP Profile-Likelihood Updates

Full mathematical derivations of all significance methods, the profile-likelihood formulation, adaptive rebinning, Barlow-Beeston masking, PL smoothing pipeline, and spike detection are in [`docs/hep_likelihood_derivation.tex`](hep_likelihood_derivation.tex).

Three improvements to the HEP profile-likelihood significance computation:

**Monotonicity enforcement via PAVA** (`src/physics/hep/01_hep.py`): when `pl_isotonic: true` is set in the workflow config, the raw per-cut PL significance array is post-processed with [`sklearn.isotonic.IsotonicRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html) (PAVA) to enforce strict monotonicity. PAVA finds the non-decreasing sequence with minimum L2 distance from the raw values, ensuring accumulated exposure cannot reduce significance. The pre-PAVA values are saved as `PreIsotonicProfileLikelihood` in the output pkl and are used by the spike detector in `05_best_sigmas.py` — so spikes visible before PAVA flattens them are still caught. Note: Gaussian kernel smoothing is **not** applied to PL curves; smoothing is only used for visual display of background histograms and must not touch the per-bin rates that enter the likelihood ratio (see *Barlow-Beeston* note below).

**±1σ expected discovery bands** (`src/physics/hep/01_hep.py`): the PL error band uses **signal normalization variation** — signal events are scaled by `(1 ± σ_s)` where `σ_s = signal_uncertainty` (reconstruction efficiency systematic, passed via `--signal_uncertainty`, typically 0.1). The background is never shifted, so the β̂_null nuisance parameter is unaffected and both bands collapse symmetrically when signal is negligible. Bands converge to the nominal line when signal is negligible (±σ_s × 0 = 0 change) and remain narrow and symmetric otherwise. The PL computation uses the original fine binning throughout — no adaptive rebin — since PL is optimal at the finest available resolution and the likelihood ratio naturally suppresses bins with negligible signal.

**Barlow-Beeston MC mask** (`src/physics/hep/01_hep.py`): a static per-bin mask (`pl_bin_mask = background_mc_counts >= min_mc_per_bin`, default `min_mc_per_bin = 1`) zeros signal and background for bins with insufficient MC support before entering the profile likelihood. This is the [Barlow-Beeston lite approach](https://www.sciencedirect.com/science/article/pii/009350659390005W) used by [ROOT HistFactory](https://root.cern.ch/doc/master/classRooStats_1_1HistFactory_1_1Measurement.html): bins with no MC events produce LLR terms that grow as `signal × log(signal / ε)` (super-linear in exposure), so they are excluded rather than floored. Crucially, **only raw histogram background rates are used for the PL computation** — the Gaussian-smoothed background is strictly for visual display. Gaussian smoothing redistributes background away from high-energy signal-region bins via kernel tails, producing near-zero (but nonzero) smoothed rates in bins where raw MC count ≥ 1. These near-zero denominators produce the same log-amplification artifact as empty bins (`signal × log(signal / 1e-6) ≈ signal × 13.8`), inflating PL significance in proportion to signal strength — a systematic bias that affects denser detector configs more severely. Using raw rates with the Barlow-Beeston mask guarantees `background_rate > 0` for all unmasked bins by construction.

**Spike-robust best-cut selection** (`src/physics/sensitivity/05_best_sigmas.py`): PL curves that contain abrupt jumps are excluded from the `max(significance)` cut selection. A curve is flagged if any consecutive step in `PreIsotonicProfileLikelihood` (the pre-PAVA values, available when `pl_isotonic: true`) exceeds `--max_pl_jump` σ (default 1.0); if `PreIsotonicProfileLikelihood` is absent, the post-PAVA `ProfileLikelihood` column is used as fallback. Spiked cuts are saved separately to `{config}_{name}_highest_spiked_HEP.pkl` for diagnostic review. If all cuts for a given (config, name, energy) are spiked, the filter is bypassed with a warning to preserve output. `--max_pl_jump 0` disables filtering entirely (backward-compatible).

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

The policy is configured in `config/analysis/config.json` (or `config/analysis/backgrounds.json`) under `BACKGROUND_SAMPLES`:

- `ANALYSES`: per-analysis background component lists (`DAYNIGHT`, `HEP`, `SENSITIVITY`).
- `ESSENTIAL`: map of components that must be present (`true`) vs optional (`false`).

Runtime behavior:

- Non-essential components not listed in the selected analysis component list are not processed.
- Essential components that are missing on disk produce warnings.
- Optional components that are missing are skipped with warnings.

This avoids failures when optional backgrounds (for example `radiological`) are unavailable for a given detector configuration while still protecting required components.
