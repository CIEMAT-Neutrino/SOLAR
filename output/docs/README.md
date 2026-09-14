# output/docs — Analysis Working Notes

Internal reference docs for DUNE solar neutrino analysis. Not end-user docs (see `docs/` for Sphinx tree).

Math renders with MathJax/KaTeX (`$...$` inline, `$$...$$` display). Open in GitHub, Obsidian, or any MathJax-aware viewer.

## Contents

| File | Purpose | Updated | Currency |
|---|---|---|---|
| [solar_analyses.md](solar_analyses.md) | Mathematical derivations: DayNight, HEP, Sensitivity + statistical methodology updates. **§5 is the full formal description of the Sensitivity analysis** — templates, pull statistic, nuisances, grid, contours, validation, studies, background-uncertainty result, data products, configuration | **2026-09-14** | **Current** — §5 rewritten for the pull method; §8.1/§8.5/§9 revised |
| [analysis_error_bands.md](analysis_error_bands.md) | Error band convention + `SignificanceError±` justification | 2026-09-01 | Current |
| [numerical_results.md](numerical_results.md) | Numerical pipeline outputs — cut thresholds, significance values, background model | 2026-06-15 | Partially stale (VD HEP wrong cuts — see flags section) |
| [thesis_plots_runbook.md](thesis_plots_runbook.md) | Chapter 9 plot commands, sync commands, per-study availability + validity matrix | 2026-09-07 | Current — statuses verified against PNFS |
| [artifact_guide.md](artifact_guide.md) | Significance artifact regeneration guide, HTML structure, update workflow, per-study status vocabulary | 2026-09-07 | Current — baselines re-verified |

## Missing / Blocked

*Verified 2026-09-07; `unc_bkg` row updated 2026-09-14. `thesis_plots_runbook.md` carries the full per-study matrix.*

| Item | Blocked on |
|---|---|
| `membrane_veto_off`, `nuisance_sin13`, `nuisance_escale`, `fiduc_truth` Sensitivity leg | **Code fix** — each is bit-identical to nominal where its knob must move; a re-run alone reproduces it |
| ~~`unc_bkg0/4/6` Sensitivity leg~~ | **Not blocked — resolved 2026-09-14.** Identical results are the correct physics answer, not a bug: the background prior is non-binding (see [solar_analyses.md §5.14](solar_analyses.md#sens-bkg-prior), summarised in its §8.5). Report as an established insensitivity. |
| Sensitivity `bkg_gamma` (all 4 configs), `energy_maink` (all 4), `energy_spk` (lAPA/vdN/vdS), `charge_Q100` (lAPA/vdN/vdS), `charge_Q500` (all) | Pipeline stage not yet run |
| `unc_bkg4`/`unc_bkg6` HEP | `RERUN-B` (08-28); vdN/vdS values pathological (vdN maxPL[80] 2.88→7.02) |
| `charge_Q100`, `charge_Q500`, `energy_maink` | `RERUN-B` — pre-date the one-knob policy |
| Orphan purge (337 files, 2.29 GB) | Awaiting approval; manifest at `output/logs/orphan_purge_manifest_20260907_023811.txt` |
| VD HEP values in `numerical_results.md` | Superseded — that file predates the 09-06/09-07 productions |
