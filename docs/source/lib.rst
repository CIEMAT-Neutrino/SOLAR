lib package
===========

The `lib` package contains the shared helpers used across the staged workflow and solar analyses.

Core modules
------------

- `lib.io`: input/output helpers, data loading, and path handling.
- `lib.dataframe`: dataframe and tabular-analysis helpers.
- `lib.reco`: reconstruction workflow helpers.
- `lib.workflow`: workflow assembly and event-matching helpers.
- `lib.oscillation`: oscillation and likelihood utilities.
- `lib.root`: ROOT-facing fitting helpers.
- `lib.plotting`: plotting helpers used across the project.
- `lib.smoothing`: config-driven 1D and 2D histogram smoothing.
- `lib.fiducial`: fiducial-selection configuration and lookup helpers.
- `lib.sigma`: significance evaluation helpers.
- `lib.weights`: truth-level weighting and PDF evaluation.

Notes
-----

The Read the Docs build intentionally keeps this page descriptive instead of importing the full runtime package tree. That avoids requiring ROOT-era analysis dependencies just to render the documentation site.
