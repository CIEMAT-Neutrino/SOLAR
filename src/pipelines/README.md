# Pipelines

Top-level run macros. Each script is a self-contained entrypoint for one major analysis block.
Individual step scripts live in [`../physics/`](../physics/).

---

## Quick Reference

| Script | Block | Key inputs |
|--------|-------|------------|
| `run_truth.py` | Truth-level preprocessing | oscillation files, background PDFs, signal KDE |
| `run_calibration.py` | Detector calibration + reconstruction | per config/name |
| `run_calibration_all.py` | Calibration over all configs | loops run_calibration |
| `run_sensitivity.py` | DayNight + HEP + Sensitivity | requires truth + calibration done |
| `run_pds.py` | PDS (OpFlash) diagnostics | |
| `run_tpc.py` | TPC energy resolution diagnostics | |
| `run_vertex.py` | Vertex reconstruction diagnostics | |
| `run_preselection.py` | Preselection efficiency | |
| `run_all.py` | Full chain | |
| `run_analysis.py` | Analysis-only (no detector steps) | |

---

## `run_truth.py` — Truth Preprocessing

Orchestrates oscillation file processing, background PDFs, and signal nadir KDE.
Steps live in [`../physics/truth/`](../physics/truth/).

```bash
python3 src/pipelines/run_truth.py \
  --config hd_1x2x6_centralAPA \
  --names marley gamma neutron \
  --folder Nominal \
  --oscillation_backend nufast \
  --no-rewrite
```

| Flag | Default | Description |
| ------ | ------- | ----------- |
| `--config` | from `analysis/backgrounds.json` | Detector configuration(s) |
| `--names` | `marley gamma neutron` | Sample names to process |
| `--folder` | `Nominal` | Fiducial folder(s) for signal KDE |
| `--oscillation_backend` | `file` | `file` / `prob3` / `nufast` |
| `--oscillations` | off | Run oscillation file processing step |
| `--no-spectra` | on | Skip background spectra step |
| `--no-external-pdf` | on | Skip external background PDF step |
| `--no-surface-pdf` | on | Skip surface PDF (legacy MC) step |
| `--no-signal-kde` | on | Skip signal nadir KDE step |
| `--rewrite` / `--no-rewrite` | rewrite | Overwrite existing outputs |
| `--debug` | off | Verbose output |

---

## `run_calibration.py` — Calibration & Reconstruction

Runs correction → calibration → discrimination → reconstruction → smearing chain.
Steps live in [`../physics/calibration/`](../physics/calibration/).

```bash
python3 src/pipelines/run_calibration.py \
  --config hd_1x2x6_centralAPA \
  --name marley
```

---

## `run_sensitivity.py` — DayNight + HEP + Sensitivity

Master orchestrator. Runs signal fiducialization, background templates, oscillation grid,
cut optimisation, chi2 grid fit, and all plots.
Steps live in [`../physics/`](../physics/) under `signal/`, `sensitivity/`, `daynight/`, `hep/`, `common/`.

```bash
# Full run — all three analyses
python3 src/pipelines/run_sensitivity.py \
  --config hd_1x2x6_centralAPA \
  --names marley gamma neutron radiological \
  --analysis DayNight HEP Sensitivity \
  --folder Truncated \
  --energy SolarEnergy \
  --exposure 30 \
  --oscillation_backend nufast \
  --no-rewrite \
  --plot

# Sensitivity only, skip already-computed steps
python3 src/pipelines/run_sensitivity.py \
  --analysis Sensitivity \
  --oscillation_backend nufast \
  --no-rewrite
```

| Flag | Default | Description |
| ------ | ------- | ----------- |
| `--config` | `hd_1x2x6_centralAPA` | Detector configuration(s) |
| `--names` | `marley gamma neutron radiological` | Sample names |
| `--analysis` | all three | `DayNight` / `HEP` / `Sensitivity` |
| `--folder` | `Truncated` | Fiducial folder(s) |
| `--energy` | `SolarEnergy` | Energy estimator |
| `--exposure` | `30` | Exposure in kt·yr |
| `--oscillation_backend` | `file` | `file` / `prob3` / `nufast` |
| `--no-fiducialization` | | Skip signal fiducialization step |
| `--no-rebin` | | Skip signal rebin step |
| `--no-significance` | | Skip significance computation |
| `--rewrite` / `--no-rewrite` | rewrite | Overwrite existing outputs |
| `--plot` / `--no-plot` | plot | Generate output figures |
| `--debug` | off | Verbose output |

---

## `run_pds.py` / `run_tpc.py` / `run_vertex.py` / `run_preselection.py`

Detector subsystem diagnostics. Steps live in [`../physics/detector/`](../physics/detector/).

```bash
python3 src/pipelines/run_pds.py --config hd_1x2x6_centralAPA --name marley
python3 src/pipelines/run_tpc.py --config hd_1x2x6_centralAPA --name marley
python3 src/pipelines/run_vertex.py --config hd_1x2x6_centralAPA --name marley
python3 src/pipelines/run_preselection.py --config hd_1x2x6_centralAPA --name marley
```

---

## Execution Order

For a clean environment run these blocks in order:

```text
1. run_truth.py          ← oscillation files + background PDFs
2. run_calibration.py    ← detector response chain
3. run_sensitivity.py    ← full physics analysis
```
