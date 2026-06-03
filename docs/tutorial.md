# Tutorial

This tutorial follows the current structure of the repository: generate or collect `SolarNuAna_module` outputs, run the staged reconstruction pipeline, then build the high-level solar analyses.

## 1. Prepare Inputs

SOLAR expects ROOT outputs produced with DUNE and the `SolarNuAna_module`. Detector and path defaults are stored in the configuration JSON files under `config/`.

Common analysis-wide defaults, including smoothing and fiducialization rules, live in `import/analysis.json`.

## 2. Run The Full Reconstruction Chain

To execute calibration, preselection, PDS, TPC, and vertex stages for one configuration and sample:

```bash
python3 src/pipelines/run_all.py --config hd_1x2x6_centralAPA --name marley
```

This orchestrates:

- `src/pipelines/run_calibration.py` — energy correction, calibration, discrimination, reconstruction, smearing
- `src/pipelines/run_preselection.py` — production, efficiency, clustering
- `src/pipelines/run_pds.py` — opflash, adj opflash, matched opflash, efficiency
- `src/pipelines/run_tpc.py` — electron energy, adj clusters, energy resolution
- `src/pipelines/run_vertex.py` — smearing, vertex distributions, fiducial, reconstruction

If the detector products already exist and you only want to rerun the downstream analysis studies, you can start from:

```bash
python3 src/pipelines/run_analysis.py --config hd_1x2x6_centralAPA --name marley
```

## 3. Run The High-Level Solar Analyses

The main analysis entry point is `src/pipelines/run_sensitivity.py`. It can coordinate fiducial scans and the DayNight, HEP, and Sensitivity analyses in one pass.

Example:

```bash
python3 src/pipelines/run_sensitivity.py \
  --config hd_1x2x6_centralAPA \
  --names marley gamma neutron \
  --analysis DayNight HEP Sensitivity \
  --folder Reduced \
  --exposure 30
```

What it does today:

- Builds fiducialized signal scans with `src/physics/signal/01_fiducialize.py`.
- Selects best fiducials with `src/physics/signal/02_best_fiducial.py`.
- Produces per-analysis signal summaries with `src/physics/signal/03_analysis.py`.
- Runs Day-Night products: `src/physics/daynight/01_daynight.py`, exposure plots, significance plots, and best-sigma selection.
- Runs HEP products: `src/physics/hep/01_hep.py`, exposure plots, significance plots, and best-sigma selection.
- Runs oscillation sensitivity templates and contour plots: `src/physics/sensitivity/01_background_template.py`, `src/physics/sensitivity/02_signal_template.py`, `src/physics/sensitivity/06_significance.py`, and `src/physics/sensitivity/contour_plot.py`.

Component policy note:

- The orchestration applies the background component policy from `import/analysis.json` (`BACKGROUND_SAMPLES.ANALYSES` and `BACKGROUND_SAMPLES.ESSENTIAL`).
- Non-essential components not selected for the requested analyses are skipped.
- Missing essential components produce warnings.
- Missing optional components are skipped with warnings.

This is useful when optional samples (for example `radiological`) are available only for a subset of detector configurations.

## 4. Inspect Outputs

The repository writes intermediate and final artefacts into project-local folders, especially:

- `data/` for processed arrays, scan products, and pickled analysis results.
- `images/` for plots and figure outputs.
- `output/presentations/` for presentation-ready material when used.

## 5. Validate New Smoothing Behaviour

Recent analysis updates introduced config-driven Gaussian smoothing in `lib/smoothing.py`. The current regression coverage is:

```bash
python3 -m pytest tests/test_smoothing.py
```

These tests verify integral preservation and configuration defaults for both 1D and 2D smoothing.
