# output/docs — Analysis Working Notes

Internal reference docs for DUNE solar neutrino analysis. Not end-user docs (see `docs/` for Sphinx tree).

Math renders with MathJax/KaTeX (`$...$` inline, `$$...$$` display). Open in GitHub, Obsidian, or any MathJax-aware viewer.

## Contents

| File | Purpose | Updated | Currency |
|---|---|---|---|
| [solar_analyses.md](solar_analyses.md) | Mathematical derivations: DayNight, HEP, Sensitivity | 2026-08-25 | Current |
| [analysis_error_bands.md](analysis_error_bands.md) | Error band convention + `SignificanceError±` justification | 2026-09-01 | Current |
| [numerical_results.md](numerical_results.md) | Numerical pipeline outputs — cut thresholds, significance values, background model | 2026-06-15 | Partially stale (VD HEP wrong cuts — see flags section) |
| [thesis_plots_runbook.md](thesis_plots_runbook.md) | Chapter 9 plot commands, sync commands, availability matrix | 2026-08-26 | Active working doc |
| [artifact_guide.md](artifact_guide.md) | Significance artifact regeneration guide, HTML structure, update workflow | 2026-08-27 | Active working doc |

## Missing / Blocked

| Item | Blocked on |
|---|---|
| VD HEP significance values | Re-run `src/physics/hep/exposure_plot.py` with current best-sigma cuts |
| `fiduc_truth` study results | Pipeline not yet complete |
| `bkg_gamma` study results | Pipeline not yet complete |
| HEP energy_maink / energy_spk | `--ignore_energy_window` flag not yet implemented |
| oscpoint_reactor DayNight | Re-run — Results pkls predate corrected Rebin pkls |
