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
- `0YBestFiducial.py`: selects optimized fiducials using analysis-specific significance settings.
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

Those settings are consumed by helpers in `lib/lib_smooth.py` and `lib/lib_fiducial.py`.
