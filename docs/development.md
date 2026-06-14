# Development Notes

## Key Repository Paths

- `config/`: detector-configuration JSON files plus analysis defaults (`config/analysis/`, `config/import/`).
- `lib/`: reusable analysis helpers.
- `tests/`: automated regression coverage.
- `src/tools/`: helper scripts for setup, docs, presentations, and repository utilities.

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
- `src/tools/setup.sh` is aimed at shared CIEMAT storage workflows using `sshfs` mounts.
- Generated products land in `output/data/`, `output/images/`, and `output/presentations/`.
