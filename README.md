# SOLAR

SolarNuAna output analysis and reconstruction toolkit for DUNE solar-neutrino studies.

Turns `SolarNuAna_module` ROOT outputs into analysis products: detector-performance studies, truth-level weighting, oscillation templates, and oscillation-sensitivity plots covering Day-Night asymmetry, HEP flux measurement, and full Δm²/sin²θ sensitivity contours.

Full documentation lives in [`docs/`](docs/) and is published at [dune-solar.readthedocs.io](https://dune-solar.readthedocs.io/en/latest/).

---

## Repository Layout

```text
src/
  pipelines/          Top-level run macros (one per analysis block)
  physics/
    calibration/      Correction → calibration → discrimination → reconstruction → smearing
    detector/
      pds/            PDS optical-flash matching and efficiency
      tpc/            TPC electron energy and cluster resolution
      vertex/         Vertex smearing, fiducial, and reconstruction
      preselection/   Production, efficiency, and clustering studies
    truth/            Oscillation templates, background spectra, signal KDE
    signal/           Fiducialization, rebinning, and analysis signal products
    common/           Shared significance-plot macros
    daynight/         Day-Night significance and exposure curves
    hep/              HEP significance, exposure, and comparison plots
    sensitivity/      Background/signal templates, cut optimisation, contour plots
  tools/
    presentations/    Auto-generated Reveal.js slide decks (daynight/hep/sensitivity)
    optimize_smoothing.py
    event_display.py
    compare_backends.py
    processing.py

lib/                  Shared helpers (IO, smoothing, oscillation, fiducial, log, …)
config/               Per-detector-configuration JSON files
analysis/             Active analysis settings (backgrounds, physics, smoothing, …)
tests/                Regression tests
docs/                 Sphinx documentation source
external/             Prob3plusplus and NuFast-Earth oscillation backends
```

---

## Pipelines

All entry-points live in [`src/pipelines/`](src/pipelines/). Each script is self-contained and accepts `--verbose quiet|normal|verbose` (default `normal`).

| Script | Block |
| ------ | ----- |
| `run_truth.py` | Oscillation templates + background PDFs + signal KDE |
| `run_calibration.py` | Detector calibration chain (single config/name) |
| `run_calibration_all.py` | Calibration over multiple configs |
| `run_sensitivity.py` | DayNight + HEP + Sensitivity analyses |
| `run_pds.py` | PDS (OpFlash) diagnostics |
| `run_tpc.py` | TPC energy resolution diagnostics |
| `run_vertex.py` | Vertex reconstruction diagnostics |
| `run_preselection.py` | Preselection efficiency diagnostics |
| `run_analysis.py` | All detector diagnostics (no calibration) |
| `run_all.py` | Full detector chain |

### Execution Order

For a clean environment run blocks in order:

```text
1. run_truth.py          ← oscillation files + background PDFs
2. run_calibration.py    ← detector response chain
3. run_sensitivity.py    ← physics analyses
```

---

## Typical Workflow

### 1 — Truth preprocessing

```bash
python3 src/pipelines/run_truth.py \
  --config hd_1x2x6_centralAPA \
  --names marley gamma neutron \
  --folder Nominal \
  --oscillation_backend nufast \
  --no-rewrite
```

Key flags:

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--config` | from `analysis/backgrounds.json` | Detector configuration(s) |
| `--names` | `marley gamma neutron` | Signal and background sample names |
| `--folder` | `Nominal` | Fiducial folder(s) for signal KDE |
| `--oscillation_backend` | `file` | `file` / `prob3` / `nufast` |
| `--oscillations` | off | Enable oscillation template step |
| `--no-spectra` | on | Skip background spectra step |
| `--verbose` | `normal` | `quiet` / `normal` / `verbose` |

### 2 — Detector calibration

```bash
python3 src/pipelines/run_calibration.py \
  --config hd_1x2x6_centralAPA \
  --name marley
```

### 3 — Physics analyses

```bash
# Full three-analysis run
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

Key flags:

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--config` | `hd_1x2x6_centralAPA` | Detector configuration(s) |
| `--names` | `marley gamma neutron radiological` | Sample names |
| `--analysis` | all three | `DayNight` / `HEP` / `Sensitivity` |
| `--folder` | `Truncated` | Fiducial folder(s) |
| `--energy` | `SolarEnergy` | Energy estimator |
| `--exposure` | `30` | Exposure in kt·yr |
| `--oscillation_backend` | `file` | `file` / `prob3` / `nufast` |
| `--no-computation` | | Plot-only mode (skip all computation) |
| `--no-significance` | | Skip DayNight/HEP/Sensitivity computation |
| `--no-fiducialization` | | Skip fiducialization step |
| `--no-rebin` | | Skip signal rebin step |
| `--rewrite` / `--no-rewrite` | rewrite | Overwrite existing outputs |
| `--plot` / `--no-plot` | plot | Generate output figures |
| `--verbose` | `normal` | `quiet` / `normal` / `verbose` |

---

## Verbosity

All pipelines share a common verbosity model backed by `lib/log.py`:

| Level | What is shown |
| ----- | ------------- |
| `quiet` | Errors and warnings only |
| `normal` | Progress markers and key results (default) |
| `verbose` | All output, including per-file debug messages |

The selected level is propagated to every subprocess via the `SOLAR_VERBOSE` environment variable so child scripts filter their own `WorkflowLogBuffer` output consistently.

```bash
# Run sensitivity with full debug output
python3 src/pipelines/run_sensitivity.py --verbose verbose ...

# Run truth pipeline silently (CI/batch use)
python3 src/pipelines/run_truth.py --verbose quiet ...
```

---

## Component Selection Policy (Essential vs Non-Essential)

The sensitivity pipeline applies a policy layer before launching per-sample jobs.

- `BACKGROUND_SAMPLES.ANALYSES` in `analysis/backgrounds.json` defines which background components each analysis (`DAYNIGHT`, `HEP`, `SENSITIVITY`) uses.
- `BACKGROUND_SAMPLES.ESSENTIAL` marks components as required (`true`) or optional (`false`).

Behavior:

- Non-essential + not listed for selected analyses → not processed.
- Essential + missing input files → hard stop.
- Optional + missing input files → skipped with warning.

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
    "DAYNIGHT":     ["gamma", "neutron"],
    "HEP":          ["gamma", "neutron"],
    "SENSITIVITY":  ["gamma", "neutron"]
  }
}
```

This allows passing optional components (e.g. `radiological`) via `--names` without forcing failures in configurations where those productions do not exist.

---

## Oscillation Backends

Three backends are supported for oscillation probability computation:

| Backend | Description |
| ------- | ----------- |
| `file` | Load pre-computed ROOT oscillogram files and rebin to analysis grid |
| `prob3` | Compute on-the-fly with [Prob3plusplus](external/Prob3plusplus/) |
| `nufast` | Compute on-the-fly with [NuFast-Earth](external/NuFast-Earth/) (faster) |

Select via `--oscillation_backend` in any pipeline that accepts it.

---

## Physics Notes

### HEP Profile-Likelihood

`src/physics/hep/01_hep.py` and `lib/sigma.py` implement a **global background normalization model**: a single scale factor β ~ Gaussian(1, σ_rel) is profiled across all bins jointly. The legacy per-bin model was over-parameterized (N independent dials to absorb signal bin-by-bin), producing an artificial flat significance region. A global β correctly represents a correlated rate systematic; the saturation threshold is `f ~ 1/(B_total · σ_rel²)`.

Exposure curves are post-processed with Gaussian kernel smoothing (`scipy.ndimage.gaussian_filter1d`, σ = 3 grid units) followed by isotonic regression (PAVA) for strict monotonicity.

### Day-Night Methodology

Follows the energy-spectral counting approach of Super-Kamiogande [[Abe et al. PRD 94, 052010 (2016)](https://doi.org/10.1103/PhysRevD.94.052010); [Renshaw et al. PRL 112, 091805 (2014)](https://doi.org/10.1103/PhysRevLett.112.091805)]:

- DUNE's $Z_{global}^2 = \sum_i Z_i^2 = \sum_i (\Delta S_i)^2 / B_i$ is equivalent to Super-K's energy-spectral $\chi^2$ in the statistical-only limit.
- Background model embeds the daytime solar signal in the null hypothesis: $B_i = B_i^{raw}/2 + S_i^{day}$.
- Baseline is statistical-only; a second curve adds background uncertainty in quadrature.

---

## Installation

### Local (venv)

```bash
git clone https://github.com/CIEMAT-Neutrino/SOLAR.git
cd SOLAR
python3 -m venv .venv
source .venv/bin/activate
pip install -r docs/requirements.txt
```

### Container (Apptainer)

```bash
apptainer pull dune-solar.sif docker://dune/dune-solar:latest
```

If working against shared CIEMAT storage, `source scripts/setup.sh` mounts the expected data directories via `sshfs`.

### Oscillation backends (optional)

```bash
# Prob3plusplus
cd external/Prob3plusplus && make && cd python && make

# NuFast-Earth
cd external/NuFast-Earth && make && cd python && make
```

---

## Documentation Build

```bash
cd docs && make html
```

Output: `docs/_build/html/`.

---

## Testing

```bash
python3 -m pytest tests/test_smoothing.py
```

---

## Authors

- [Sergio Manthey Corchado](https://github.com/mantheys)

## License

[MIT](https://choosealicense.com/licenses/mit/)
