# Development Notes

## Key Repository Paths

- `config/`: detector-configuration JSON files.
- `import/analysis.json`: shared physics-analysis defaults.
- `lib/`: reusable analysis helpers.
- `tests/`: automated regression coverage.
- `scripts/`: helper scripts for setup, docs, and repository utilities.

## Current Validation

The repository now includes a focused regression test for smoothing behaviour:

```bash
python3 -m pytest tests/test_smoothing.py
```

This verifies that the newer Gaussian smoothing utilities preserve histogram integrals and expose the expected config defaults.

## Building Docs

```bash
cd docs
make html
```

## Practical Notes

- Many scripts assume existing DUNE-produced ROOT outputs and configuration-dependent filesystem paths.
- `scripts/setup.sh` is aimed at shared CIEMAT storage workflows using `sshfs` mounts.
- Generated products typically land in `data/`, `images/`, and sometimes `presentations/`.
