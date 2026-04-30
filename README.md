# SOLAR

SolarNuAna output analysis and reconstruction toolkit for DUNE solar-neutrino studies.

The repository turns `SolarNuAna_module` outputs into analysis products, detector-performance studies, and oscillation sensitivity plots. It now includes a full staged pipeline covering truth-level weighting, reconstruction workflow studies, preselection, PDS/TPC/vertex performance, and high-level solar analyses such as Day-Night, HEP, fiducial optimization, and oscillation sensitivity.

Full documentation lives in [`docs/`](docs/) and is published at [dune-solar.readthedocs.io](https://dune-solar.readthedocs.io/en/latest/).

## What Is In The Repo?

- `src/workflow/`: detector-response and reconstruction workflow drivers.
- `src/preselection/`: production, efficiency, and clustering studies.
- `src/PDS/`: optical-flash and matching studies.
- `src/TPC/`: cluster and energy-resolution studies.
- `src/vertex/`: smearing, fiducial, and reconstruction performance scans.
- `src/truth/`: truth-level weighting and background PDF preparation.
- `src/analysis/`: solar-physics analyses, fiducial optimization, Day-Night, HEP, and sensitivity products.
- `lib/`: shared IO, plotting, smoothing, fiducial, oscillation, and workflow helpers.
- `tests/`: regression coverage for newer smoothing utilities.
- `config/`: detector-configuration-specific JSON files.
- `import/`: shared analysis defaults, including smoothing and fiducialization settings.

## Typical Workflow

1. Start from ROOT outputs produced with DUNE + `SolarNuAna_module`.

   ```bash
   python3 src/XXProcessing.py
   ```

2. Run the detector calibration chain:

   ```bash
   python3 src/WORKFLOW/00AllWorkflow.py --config hd_1x2x6_centralAPA --name marley_official
   ```

3. Run preselection, PDS, TPC, and vertex studies:

   ```bash
   python3 src/01Analysis.py \
     --config hd_1x2x6_centralAPA \
     --names marley gamma neutron \
     --analysis Preselection PDS TPC Vertex \
     --folder Nominal
   ```

4. Run the high-level solar analyses:

   ```bash
   python3 src/analysis/10SensitivityAnalysis.py \
     --config hd_1x2x6_centralAPA \
     --names marley gamma neutron \
     --analysis DayNight HEP Sensitivity \
     --energy SolarEnergy \
     --folder Nominal
   ```

5. Inspect generated artefacts under `data/` and `images/`.

For analysis-only reruns on existing reconstructed products, `src/01Analysis.py` skips the initial workflow stage and starts from preselection/PDS/TPC/vertex.

## Recent Developments Reflected In This Repo

- Config-driven histogram smoothing via `lib/lib_smooth.py`.
- Fiducial optimization helpers in `lib/lib_fiducial.py`, with split analysis/plot macros in `src/analysis/0YBestFiducial.py` and `src/analysis/10FiducializationPlot.py`.
- Updated Day-Night, HEP, and oscillation-sensitivity orchestration in `src/analysis/10SensitivityAnalysis.py`.
- Component-policy filtering for analysis inputs (essential vs non-essential backgrounds):
  - Configured in `import/analysis.json` under `BACKGROUND_SAMPLES.ESSENTIAL` and `BACKGROUND_SAMPLES.ANALYSES`.
  - Non-essential components not listed for the selected analyses are skipped automatically.
  - Missing essential components emit warnings, while missing optional ones are skipped.
- Shared analysis defaults in [`import/`](import/), including:
  - General analysis binnings and plotting settings.
  - Gaussian smoothing defaults for 1D and 2D products.
  - Best-significance reference selection.
  - Analysis-specific fiducialization rules for DayNight, HEP, and Sensitivity.
  - Centralized significance thresholds in `ANALYSIS_THRESHOLDS` (for `DAYNIGHT`, `HEP`, `SENSITIVITY`), used by the DayNight/HEP/Sensitivity analysis macros as their default threshold input.
  - Fiducial-stage energy windows from `FIDUCIALIZATION.ANALYSES.*.energy_min` and `energy_max`.
- Regression tests for smoothing behaviour in [`tests/test_smoothing.py`](tests/test_smoothing.py).

## Component Selection Policy (Essential vs Non-Essential)

The analysis orchestrator (`src/analysis/10SensitivityAnalysis.py`) now applies a policy layer before launching per-sample jobs.

- `BACKGROUND_SAMPLES.ANALYSES` in `import/analysis.json` defines which background components are used by each analysis (`DAYNIGHT`, `HEP`, `SENSITIVITY`).
- `BACKGROUND_SAMPLES.ESSENTIAL` defines whether a component is required (`true`) or optional (`false`).

Behavior:

- Non-essential + not included in selected analysis component list: not processed.
- Essential + missing input files: warning is printed.
- Optional + missing input files: skipped with warning.

Example config snippet:

```json
"BACKGROUND_SAMPLES": {
  "ESSENTIAL": {
    "gamma": true,
    "neutron": true,
    "alpha": false,
    "radiological": false
  },
  "ANALYSES": {
    "DAYNIGHT": ["gamma", "neutron"],
    "HEP": ["gamma", "neutron"],
    "SENSITIVITY": ["gamma", "neutron"]
  }
}
```

This allows adding optional components (for example `radiological`) to command-line `--names` defaults without forcing failures in detector configurations where those productions are not available.

## Installation

SOLAR is a Python analysis repository intended to run alongside DUNE-produced files. A lightweight local setup is:

### Environment Setup

  ```bash
  git clone https://github.com/CIEMAT-Neutrino/SOLAR.git
  cd SOLAR
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r docs/requirements.txt
  ```

  For the Sphinx documentation environment, use:

  ```bash
  conda env create -f docs/environment.yaml
  conda activate test
  ```

### Container Setup

A containerized setup is also available through [DUNE's `dune-solar` image](https://hub.docker.com/r/dune/dune-solar). This image includes all dependencies and a pre-cloned version of this repository. It can be installed with apptainer by running:

```bash
apptainer pull dune-solar.sif docker://dune/dune-solar:latest
```

If you work against shared CIEMAT storage, `source scripts/setup.sh` can mount the expected `data/` and `sensitivity/` directories through `sshfs`.

## Documentation Build

```bash
cd docs
make html
```

The generated site is written to `docs/_build/html/`.

## Testing

```bash
python3 -m pytest tests/test_smoothing.py
```

## Authors

- [Sergio Manthey Corchado](https://github.com/mantheys)

## License

[MIT](https://choosealicense.com/licenses/mit/)
