# Tutorial

This tutorial follows the current structure of the repository: generate or collect `SolarNuAna_module` outputs, run the staged reconstruction pipeline, then build the high-level solar analyses.

## 1. Prepare Inputs

SOLAR expects ROOT outputs produced with DUNE and the `SolarNuAna_module`. Detector and path defaults are stored in the configuration JSON files under `config/`.

Common analysis-wide defaults, including smoothing and fiducialization rules, live in `import/analysis.json`.

## 2. Run The Full Reconstruction Chain

To execute the main workflow plus preselection, PDS, TPC, and vertex stages for one configuration and sample:

```bash
python3 src/00All.py --config hd_1x2x6_centralAPA --name marley
```

This orchestrates:

- `src/workflow/00AllWorkflow.py`
- `src/preselection/10AllPreselection.py`
- `src/PDS/20AllPDS.py`
- `src/TPC/30AllTPC.py`
- `src/vertex/40AllVertex.py`

If the workflow products already exist and you only want to rerun the downstream detector studies, you can start from:

```bash
python3 src/01Analysis.py --config hd_1x2x6_centralAPA --name marley
```

## 3. Run The High-Level Solar Analyses

The main analysis entry point is `src/analysis/10SensitivityAnalysis.py`. It can coordinate fiducial scans and the DayNight, HEP, and Sensitivity analyses in one pass.

Example:

```bash
python3 src/analysis/10SensitivityAnalysis.py \
  --config hd_1x2x6_centralAPA \
  --names marley gamma neutron \
  --analysis DayNight HEP Sensitivity \
  --folder Reduced \
  --exposure 30
```

What it does today:

- Builds fiducialized signal scans with `0XFiducializeSignal.py`.
- Selects best fiducials with `0YBestFiducial.py`.
- Produces per-analysis signal summaries with `11AnalysisSignal.py`.
- Runs Day-Night products: `12DayNight.py`, exposure plots, significance plots, and best-sigma selection.
- Runs HEP products: `13HEP.py`, exposure plots, significance plots, and best-sigma selection.
- Runs oscillation sensitivity templates and contour plots: `14SensitivityBackgroundTemplate.py`, `14SensitivitySignalTemplate.py`, `14Sensitivity.py`, and `14SensitivityContourPlot.py`.

## 4. Inspect Outputs

The repository writes intermediate and final artefacts into project-local folders, especially:

- `data/` for processed arrays, scan products, and pickled analysis results.
- `images/` for plots and figure outputs.
- `presentations/` for presentation-ready material when used.

## 5. Validate New Smoothing Behaviour

Recent analysis updates introduced config-driven Gaussian smoothing in `lib/lib_smooth.py`. The current regression coverage is:

```bash
python3 -m pytest tests/test_smoothing.py
```

These tests verify integral preservation and configuration defaults for both 1D and 2D smoothing.
