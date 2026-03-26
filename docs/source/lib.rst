lib package
===========

The `lib` package contains the shared utilities used across the staged workflow and solar analyses.

Core modules
------------

- `lib.lib_io`: input/output helpers, data loading, and path handling.
- `lib.lib_df`: dataframe and tabular-analysis helpers.
- `lib.lib_reco`: reconstruction workflow helpers.
- `lib.lib_wkf`: workflow assembly and event-matching helpers.
- `lib.lib_osc`: oscillation and likelihood utilities.
- `lib.lib_root`: ROOT-facing fitting helpers.
- `lib.lib_plt`: plotting helpers used across the project.
- `lib.lib_smooth`: config-driven 1D and 2D histogram smoothing.
- `lib.lib_fiducial`: fiducial-selection configuration and lookup helpers.
- `lib.lib_sigma`: significance evaluation helpers.
- `lib.lib_weights`: truth-level weighting and PDF evaluation.

Package contents
----------------

.. automodule:: lib
   :members:
   :undoc-members:
   :show-inheritance:

Recent additions
----------------

Recent analysis work introduced smoothing and fiducial helper modules that are now used by the DayNight, HEP, and Sensitivity workflows through `import/analysis.json`.
