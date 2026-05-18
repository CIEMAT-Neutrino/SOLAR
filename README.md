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
- HEP profile-likelihood improvements in `src/analysis/13HEP.py` and `lib/lib_sigma.py`:
  - **Global background normalization model** (replacing the legacy per-bin model): `evaluate_profile_likelihood_discovery` now profiles a single global background scale factor β ~ Gaussian(1, σ_rel) across all bins jointly, rather than one independent β per bin. The per-bin model was over-parameterized: with N bins each carrying a 2% Gaussian constraint, the null hypothesis had N independent dials to absorb signal bin-by-bin, causing an artificial flat region in significance vs exposure that resolved in a sharp kink once the collective absorbing capacity was exhausted. A global β correctly represents a rate systematic that is correlated across all bins; its crossover from the absorbing regime to the saturated regime occurs at total-background scale (`f ~ 1/(B_total · σ_rel²)`, typically sub-year), eliminating the artefact. The analytical solution is the same closed-form quadratic root, now applied to summed counts `(N_total, B_total)` with a single Gaussian pull term.
  - PL exposure curves are post-processed with **Gaussian kernel smoothing** ([`scipy.ndimage.gaussian_filter1d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html), σ = 3 exposure-grid index units, tunable via `_PL_SMOOTH_SIGMA`) followed by **isotonic regression** ([`sklearn.isotonic.IsotonicRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html), PAVA). The Gaussian step removes numerical oscillations from the profile-likelihood solver at low signal-to-background ratios; PAVA then enforces strict monotonicity by finding the non-decreasing sequence minimising L2 distance from the smoothed values.
  - PL error bands use **signal normalization variation**: signal is scaled by `(1 ± σ_s)` where `σ_s = signal_uncertainty` (the reconstruction efficiency systematic, typically 10%). Both bands collapse symmetrically for configurations where signal is negligible (±10% of zero is zero). This avoids the asymmetric-collapse artifact produced by background-shifting approaches, where β̂_null pull contributions drive the upper band non-zero independent of signal strength.
  - PL significance computed on the original fine binning with no adaptive rebin. Merging bins reduces information for PL (unlike Gaussian/Asimov where empty bins cause numerical issues); the likelihood ratio naturally suppresses bins with negligible signal.
- Bug fix in `src/analysis/0ZBestSigmas.py`: fastest-sigma cut selection for HEP now uses `PLSigma2`/`PLSigma3` (PL-based exposure crossings) instead of Asimov-based `Sigma2`/`Sigma3` when the significance reference is `ProfileLikelihood`, with automatic fallback to Asimov columns for older output files.
- Day-Night significance methodology follows the energy-spectral counting approach of Super-Kamiokande [[Abe et al. (Super-K), PRD 94, 052010 (2016)](https://doi.org/10.1103/PhysRevD.94.052010); [Renshaw et al. (Super-K), PRL 112, 091805 (2014)](https://doi.org/10.1103/PhysRevLett.112.091805)]:
  - **Equivalence**: DUNE's $Z_{global}^2 = \sum_i Z_i^2 = \sum_i (\Delta S_i)^2 / B_i$ is equivalent to Super-K's energy-spectral $\chi^2$ in the statistical-only limit ($\sigma_i = \sqrt{B_i}$).
  - **Shared background model**: both analyses embed the daytime solar signal in the null-hypothesis background; DUNE uses $B_i = B_i^{raw}/2 + S_i^{day}$.
  - **Key differences**: Super-K includes systematic nuisance penalty terms (energy scale, cross-section, flux normalisation); DUNE's baseline is statistical-only, with a second curve that adds background uncertainty in quadrature. Super-K measures the asymmetry $A_{DN} = 2(\Phi_N - \Phi_D)/(\Phi_N + \Phi_D)$ from existing data; DUNE projects discovery significance as a function of future exposure. Super-K additionally sub-bins events by solar zenith angle for sensitivity gains; DUNE currently uses energy binning only.

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
