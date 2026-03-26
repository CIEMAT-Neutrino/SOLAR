# Installation

SOLAR is a Python analysis repository built around outputs from the DUNE `SolarNuAna_module`. The codebase is currently centered on Python 3.6-era environments, ROOT-compatible analysis tooling, and detector-configuration JSON files under `config/`.

## Clone The Repository

```bash
git clone https://github.com/CIEMAT-Neutrino/SOLAR.git
cd SOLAR
```

## Python Environment

For a lightweight local environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r docs/requirements.txt
```

For the full documentation environment used by Sphinx:

```bash
conda env create -f docs/environment.yaml
conda activate test
```

The conda environment includes Sphinx plus the plotting and analysis dependencies used throughout the repository.

## Data Access

Many workflows expect detector outputs and derived products to live under repository-local `data/` and `sensitivity/` paths.

If you are working from the CIEMAT environment with shared storage available, the helper script can mount those directories via `sshfs`:

```bash
source scripts/setup.sh
```

That script is intended for shared remote storage workflows. If you already have local copies of the inputs, you can skip it and point your configuration files to the correct paths.

## Build The Documentation

```bash
cd docs
make html
```

The generated site will be available in `docs/_build/html/`.
