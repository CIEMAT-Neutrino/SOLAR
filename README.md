# SOLAR

SolarNuAna output analysis and reconstruction toolkit for DUNE solar-neutrino studies.

The repository turns `SolarNuAna_module` outputs into reusable analysis products, detector-performance studies, and oscillation sensitivity plots. It now includes a full staged pipeline covering truth-level weighting, reconstruction workflow studies, preselection, PDS/TPC/vertex performance, and high-level solar analyses such as Day-Night, HEP, fiducial optimization, and oscillation sensitivity.

Full documentation lives in [`docs/`](docs/) and is published at [dune-solar.readthedocs.io](https://dune-solar.readthedocs.io/en/latest/).

## What Is In The Repo

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
- `import/analysis.json`: shared analysis defaults, including smoothing and fiducialization settings.

## Typical Workflow

1. Start from ROOT outputs produced with DUNE + `SolarNuAna_module`.
2. Run the detector workflow and reconstruction chain:

   ```bash
   python3 src/00All.py --config hd_1x2x6_centralAPA --name marley
   ```

3. Run the high-level solar analyses:

   ```bash
   python3 src/analysis/10SensitivityAnalysis.py \
     --config hd_1x2x6_centralAPA \
     --names marley gamma neutron \
     --analysis DayNight HEP Sensitivity \
     --folder Reduced
   ```

4. Inspect generated artefacts under `data/` and `images/`.

For analysis-only reruns on existing reconstructed products, `src/01Analysis.py` skips the initial workflow stage and starts from preselection/PDS/TPC/vertex.

## Recent Developments Reflected In This Repo

- Config-driven histogram smoothing via `lib/lib_smooth.py`.
- Fiducial optimization helpers in `lib/lib_fiducial.py` and `src/analysis/0YBestFiducial.py`.
- Updated Day-Night, HEP, and oscillation-sensitivity orchestration in `src/analysis/10SensitivityAnalysis.py`.
- Shared analysis defaults in [`import/analysis.json`](import/analysis.json), including:
  - Gaussian smoothing defaults for 1D and 2D products.
  - Best-significance reference selection.
  - Analysis-specific fiducialization rules for DayNight, HEP, and Sensitivity.
- Regression tests for smoothing behaviour in [`tests/test_smoothing.py`](tests/test_smoothing.py).

## Installation

SOLAR is a Python analysis repository intended to run alongside DUNE-produced files. A lightweight local setup is:

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
