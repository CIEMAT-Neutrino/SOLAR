# Thesis Chapter 9 — Plot Commands

*Generated 2026-08-26. Availability and validity re-verified against PNFS 2026-09-07.*

> ## Status: most variants are usable; a short list is not
>
> An earlier revision of this file declared nearly everything invalid. That was too
> pessimistic and its central piece of evidence was wrong. Corrected below.
>
> **The pipeline bug had a narrow window.** `--skip_best_cuts` wrongly gated `01_daynight.py`
> and `01_hep.py` as well as `04_best_cuts.py`, so an affected variant never computed its own
> significance grid. `git log -S` pins the window exactly: introduced in `9278bd6`
> (2026-09-02 18:11), removed in `3043c09` (2026-09-06 12:42). **Only artifacts written inside
> that window are suspect.** Anything older ran on code where both scripts always executed and
> is physically genuine for its own knob. `seed_study_artifacts_from_nominal()` also skipped
> when the destination already existed and used `copy2` (preserving mtime); no variant/default
> mtime collisions survive, so its damage has been overwritten everywhere.
>
> **The "unc_bkg is broken" evidence was a stale-default artifact.** The previous revision read
> "unc_bkg DayNight Asimov must equal the default but does not — LAPA 0.915 vs 0.908, VDN 0.505
> vs 0.500, VDS 0.972 vs 1.075". Those right-hand numbers were *old default* values. Verified
> 2026-09-07 against current defaults, Asimov is **byte-identical** (same MD5) between default
> and every `unc_bkg` variant on all four configs, and ErrorGaussian moves monotonically with
> σ_bkg as required:
>
> | Config | Asimov[80] (default == all unc_bkg) | EG σ=0% | EG σ=2% (default) | EG σ=4% | EG σ=6% |
> |---|---|---|---|---|---|
> | CAPA | 2.41220 | 2.23438 | 2.23305 | 2.22910 | 2.22267 |
> | LAPA | 0.91489 | 0.46805 | 0.44712 | 0.40854 | 0.37182 |
> | VDN  | 0.50468 | 0.36527 | 0.36058 | 0.34937 | 0.33548 |
> | VDS  | 0.97206 | 0.76389 | 0.74784 | 0.70565 | 0.66892 |
>
> **The vdS default was the real problem, and it is fixed.** The 2026-09-06 default production
> ran CAPA → LAPA → VDN to completion and was interrupted partway through VDS, leaving VDS
> DayNight/HEP dated 09-02 and Sensitivity dated 08-19 — older than their own Rebin inputs.
> Every VDS delta was being measured against that wrong baseline. Re-run 2026-09-07:
> DN Asimov[80] 1.07549 → **0.97206** (now matches its variants), HEP maxPL[80] **3.12824**
> (matches `oscpoint_solar`, as construction requires), Sensitivity Score **0.586294**
> (√Score 0.7657).
>
> **One-knob policy.** Every variant now holds cuts and smoothing sigmas at nominal
> (`skip_best_cuts=True`, `skip_best_sigmas=True`) and changes exactly one thing. Variants
> predating that policy produced valid physics but under a re-tuned analysis, so their delta
> confounds the knob with the re-fit — that is what `RERUN-B` marks below.

`sync_solar_data.sh` default remote: `gae_out:/pc/choozdsk01/users/manthey/SOLAR`
Study-variant pkls auto-route to `input/data/studies/` by the sync script.
Plot scripts fall back `input/data/` → `input/data/studies/` transparently.
`--study`/`--config`/`--name`/`--folder` are repeatable (one value per flag).

> **The plot scripts below do not live in this repo.** `script_iterable_scan.py`,
> `script_compare_contour.py` and `script_compare_pareto.py`, and the `input/data/` tree they
> read, exist only in the downstream analysis checkout. Run `sync_solar_data.sh` first, then
> these commands there. What this repo produces is the `*_Counts` / `*_Exposure` /
> `*_Significance` pkls those commands consume.

Datafile stem pattern (`--configs` + `--name` supplied separately):
- `--datafile {Analysis}_{Type}` → `input/data/{config}_{name}_{Analysis}_{Type}.pkl`
- `--datafile {Analysis}_{Type}_{study}` → `input/data/studies/{config}_{name}_{...}.pkl`

## Status vocabulary

| Status | Meaning |
|---|---|
| `READY` | outside the bug window, cuts at nominal, moves as its knob predicts — safe to cite |
| `RERUN-A` | written inside 09-02 18:11 → 09-06 12:42; grid may never have been computed |
| `RERUN-B` | valid physics but pre-dates the one-knob policy; cuts re-optimised, not comparable |
| `BUG` | bit-exact degeneracy against nominal — needs a **code fix**, a re-run alone reproduces it |
| `MISSING` | no pkl on disk |
| `ORPHAN` | label no longer in `STUDY_VARIANTS`; cannot be regenerated — purge, do not plot |

Labels come from `lib/study.py`; `all_study_labels(analysis=...)` is the authoritative list and
filters by `analysis_override`. Do not maintain a copy of that list by hand.

## Config Shortnames

| Shortname | Full config key |
|---|---|
| `CAPA` | `hd_1x2x6_centralAPA` |
| `LAPA` | `hd_1x2x6_lateralAPA` |
| `VDN` | `vd_1x8x14_3view_30deg_nominal` |
| `VDS` | `vd_1x8x14_3view_30deg_shielded` |

## Availability Matrix (truncated / marley — PNFS, verified 2026-09-07)

Presence is what is on disk. **Validity decides whether you may plot it.**

| Study variant | DayNight | HEP | Sensitivity | Validity |
|---|---|---|---|---|
| default | all 4 | all 4 | all 4 | **READY** — VDS refreshed 09-07 |
| oscpoint_solar | all 4 | all 4 | *(n/a)* | **READY** — identical to default **by construction** (`skip_rebin`, nominal Δm²₂₁). A null check, not a defect |
| oscpoint_reactor | all 4 | all 4 | — *(invariant, by design)* | **READY** (09-06) |
| unc_bkg0 | all 4 | all 4 | all 4 | **READY** (09-06) |
| unc_bkg4 | all 4 | CAPA | all 4 | DN **READY**; HEP `RERUN-B` (08-28) |
| unc_bkg6 | all 4 | CAPA | all 4 | DN **READY**; HEP `RERUN-B` + VDN/VDS values pathological |
| unc_sig20/40 | — | all 4 | — | **READY** (09-06, post-fix) |
| unc_sig0/2/6 | — | — | all 4 | **READY** (08-28, pre-window) |
| charge_Q50 | all 4 | all 4 | CAPA/LAPA/VDN | **READY** (09-06); Sens VDS `RERUN-B` (08-26) |
| energy_spk | all 4 | all 4 | CAPA | **READY** on CAPA/LAPA/VDN (09-06); VDS `RERUN-B` (09-01); Sens `MISSING` L/VN/VS |
| bkg_gamma | all 4 | all 4 | — | DN/HEP **READY** CAPA/LAPA/VDN (09-06), VDS `RERUN-B` (08-31); **Sens `MISSING` on all 4, CAPA included** |
| fiduc_truth | all 4 | all 4 | all 4 | DN/HEP **READY** CAPA/LAPA/VDN; VDS `RERUN-A` (09-04, in-window); **Sens `BUG`** |
| nuisance_nominal | — | — | all 4 | **READY** (08-31) |
| nuisance_sin13 | — | — | all 4 | **`BUG`** — bit-identical to `nuisance_nominal` on all 4 |
| nuisance_escale | — | — | all 4 | **`BUG`** — bit-identical to `default` (full) on all 4 |
| membrane_veto_off | VD only | VD only | VD only | **`BUG`** — bit-identical to nominal, VDN+VDS, all 3 analyses. `MISSING` on CAPA/LAPA |
| charge_Q100 | all 4 | all 4 | CAPA* | `RERUN-B` (DN 08-23, HEP 08-29); Sens `MISSING` L/VN/VS, *CAPA pkl stores wrong energy |
| charge_Q500 | all 4 | all 4 | all 4 | `RERUN-B` (DN 08-25/26, HEP 08-29) |
| energy_maink | all 4 | all 4 | — | `RERUN-B` (09-01); DN CAPA grid degenerate (191 rows vs 1030); Sens `MISSING` all 4 |
| fiduc (Nominal/Reduced/Truncated) | all 4 | all 4 | all 4 | Nominal **READY**; **Reduced `RERUN-A`** on LAPA/VDN/VDS (HEP 09-04 in-window) |
| bkgmodel (Nominal/Reduced) | all 4 | all 4 | all 4 | same trees as `fiduc` — same verdicts |
| unc_bkg10 · unc_bkg20 · unc_sig8 · charge_Q200 · metric_raw · metric_smoothed | — | — | — | **`ORPHAN`** — dropped from `STUDY_VARIANTS`, `Study` column absent, pending purge |

### The four `BUG` entries

Each is a *bit-exact* match to nominal where the knob must move the answer. `membrane_veto_off`
was written 09-06 13:58/15:20, **after** the fix, so `01_daynight.py` genuinely ran and returned
the nominal answer — this is not the `skip_best_cuts` bug. Suspicion falls on the labeled-Rebin
path: `study.py` sets `template_sfx` when `not membrane_veto`, so DayNight should be reading
`SolarEnergy_Rebin_membrane_veto_off`; getting nominal numbers means either those labeled Rebin
pkls were never regenerated or the suffix is not reaching `01_daynight.py`. `fiduc_truth`'s
Sensitivity leg is bit-identical to default on all 4 while its DayNight and HEP legs *do* move
(CAPA DN 2.31781 vs 2.41220), so the truth-fiducial selection reaches DN/HEP but not the Score.
**Re-running any of these without a code fix reproduces the same number.**

### Orphans

`unc_bkg10`, `unc_bkg20`, `unc_sig8`, `charge_Q200`, `metric_raw`, `metric_smoothed` are not in
`STUDY_VARIANTS`, so they cannot be regenerated and will never be policy-compliant. All of them
also lack a usable `Study` column, so `--select Study` silently drops them. A purge manifest is
at `output/logs/orphan_purge_manifest_20260907_023811.txt` (337 files, 2.29 GB, including
wrong-energy `SignalParticleK_*_unc_*` leftovers and pre-refactor Feb/May formats). Plot
commands below use **Q500** in place of Q200.

### Sensitivity contours — `Sensitivity_Contours.pkl`

`06_significance.py` writes its Δχ² grids to a **centralised** tree keyed by energy, nuisance
profile and uncertainty suffix, with **no study label anywhere in the path**:

```
SENSITIVITY/{cfg}/{name}/{folder}/{Energy}{template_suffix}/results/{profile}/signal_{X}%_and_background_{Y}%/
```

So the contours never appeared in the per-study output dirs that the plot scripts and
`sync_solar_data.sh` read, and the older commands here pointed at `Sensitivity_Significance`,
which carries **no `Label` / `Dm2` / `Values` columns** — those commands could not have worked.

`tools/export_sensitivity_contours.py` resolves each study's source directory from
`lib/study.py` STUDY_VARIANTS (so the mapping cannot drift) and publishes one tidy pkl per
study, named to the sync stem convention:

```
output/data/analysis/sensitivity/{cfg}/{name}/{folder}/{study}/{cfg}_{name}_Sensitivity_Contours.pkl
```

Columns: `Config Name Analysis EnergyLabel Study Label Variable Dm2 Values Significance
SignificanceUnit NuisanceProfile SignalUncertainty BackgroundUncertainty NHits AdjCl OpHits
SourcePath` — i.e. exactly `--select Label Variable Study -y Dm2 -x Values -z Significance`,
with `Label` ∈ {solar, react} and `Variable` ∈ {sin12, sin13}. Each row holds one full grid
(Dm2 × Values), and `SourcePath` records which PNFS file it came from.

Re-run it after any Sensitivity rerun, then sync `--datafile Sensitivity_Contours`:

```bash
python3 tools/export_sensitivity_contours.py                       # all configs, all studies
python3 tools/export_sensitivity_contours.py --config hd_1x2x6_centralAPA --dry_run
```

**Coverage as of 2026-09-07** — 54 written: cAPA 15 studies, lAPA/vdN/vdS 13 each.
Genuinely absent (templates exist, `06_significance.py` never ran, so there is nothing to
export): `energy_maink` and `bkg_gamma` on all 4 configs, `membrane_veto_off` on all 4,
`energy_spk` on vdN/vdS, `charge_Q100` on lAPA/vdN/vdS, and `charge_Q0` (not yet run).

The export deliberately falls back **only** unlabeled → labeled when locating a source dir.
Some historical runs wrote the energy dir labeled and some unlabeled for the same study
(cAPA `SignalParticleK` vs lAPA `SignalParticleK_energy_spk`). Falling back the other way
would read the *default* directory and publish default contours under a study label, so it
is refused.

### Known structural gaps (not staleness)

- `Sensitivity_Exposure.pkl` can never exist — `exposure_plot.py` guards `args.analysis != "Sensitivity"`. Use `Sensitivity_Significance` instead.
- `DayNight_Significance` / `HEP_Significance` are written to PNFS only, never mirrored locally, so C+E+S completeness is a PNFS-only target.
- `Oscillogram.pkl` carries no `Study` column by design; per-study copies are indistinguishable duplicates of the default.
- `charge_Q50` / `charge_Q100` Exposure pkls carry NaN `Study` rows merged in by `upsert_df_rows`; regenerate with `--rewrite`, not upsert.

### Fitting Methodology Notes

**Background Normalization Fitting (--fit_background):**
- **Default (--no-fit_background):** Background normalization is **fixed** at its nominal value; only signal amplitude (`A_pred`) is fitted. This produces physically meaningful sensitivity where contours **properly loosen** with increased background uncertainty.
- **Legacy mode (--fit_background):** Both signal amplitude (`A_pred`) and background normalization (`A_bkg`) are fitted as free parameters. This can cause the background to **absorb signal mismatches**, producing physically incorrect results where contours **shrink** (tighten) instead of **loosen** with increased background uncertainty.
- **Study variants with corrected fitting:** `unc_bkg4_nobkgfit`, `unc_bkg6_nobkgfit`, `unc_sig6_nobkgfit` demonstrate the proper behavior.
- **Legacy validation studies:** The `fit_background` study group (`fit_background_legacy`, `fit_background_unc_sig6`, `fit_background_unc_bkg6`) runs with the legacy behavior for comparison and validation.
- **Default change:** As of 2026-09-09, `run_studies.py` defaults to `--no-fit_background`. Use `--fit_background` to run the legacy mode.
- **Validation:** Running `06_significance.py` with `--fit_background=True` and `σ_bkg > 5%` will emit a warning recommending `--no-fit_background`.

**Contour Plotting (Δχ² vs Absolute χ²):**
- As of 2026-09-09, contours are drawn using **Δχ² = χ² - χ²_min** (proper confidence levels) instead of absolute χ² values.
- This ensures contours represent true confidence intervals (Δχ² = 1, 4, 9 for 1, 2, 3σ).
- The `Chi2Min` field is now included in contour DataFrames for diagnostics.

---

## Sync

`sync_solar_data.sh` prompts `[y/N]` — run interactively.

```bash
# DayNight (all 4 configs, all available studies)
sync_solar_data.sh \
  --analysis daynight \
  --config hd_1x2x6_centralAPA \
  --config hd_1x2x6_lateralAPA \
  --config vd_1x8x14_3view_30deg_nominal \
  --config vd_1x8x14_3view_30deg_shielded \
  --name marley \
  --folder truncated \
  --study default \
  --study unc_bkg0 --study unc_bkg4 --study unc_bkg6 \
  --study oscpoint_solar --study oscpoint_reactor \
  --study energy_maink --study energy_spk \
  --study charge_Q50 --study charge_Q100 --study charge_Q500 \
  --study fiduc_truth --study bkg_gamma \
  --study fiduc --study bkgmodel

# HEP (all 4 configs, all available studies)
sync_solar_data.sh \
  --analysis hep \
  --config hd_1x2x6_centralAPA \
  --config hd_1x2x6_lateralAPA \
  --config vd_1x8x14_3view_30deg_nominal \
  --config vd_1x8x14_3view_30deg_shielded \
  --name marley \
  --folder truncated \
  --study default \
  --study unc_bkg0 --study unc_bkg4 --study unc_bkg6 \
  --study unc_sig20 --study unc_sig40 \
  --study oscpoint_solar --study oscpoint_reactor \
  --study charge_Q50 --study charge_Q100 --study charge_Q500 \
  --study fiduc_truth --study bkg_gamma

# Sensitivity (all 4 configs, all available studies)
sync_solar_data.sh \
  --analysis sensitivity \
  --config hd_1x2x6_centralAPA \
  --config hd_1x2x6_lateralAPA \
  --config vd_1x8x14_3view_30deg_nominal \
  --config vd_1x8x14_3view_30deg_shielded \
  --name marley \
  --folder truncated \
  --study default \
  --study unc_bkg0 --study unc_bkg4 --study unc_bkg6 \
  --study unc_sig0 --study unc_sig2 --study unc_sig6 \
  --study oscpoint_solar --study oscpoint_reactor \
  --study energy_spk \
  --study charge_Q50 --study charge_Q100 --study charge_Q500 \
  --study fiduc_truth --study bkg_gamma

# NOTE: sync AFTER the reruns land, not before — otherwise you pull the invalid grids
# described at the top of this file into input/data/studies/ and plot them.
# Still genuinely absent on PNFS (nothing to sync until the pipeline runs):
#   (metric_raw / metric_smoothed are ORPHANS — Raw vs Smoothed comes from the
#    default pkl's SpectrumType rows, no separate study run exists or is needed)
#   Sensitivity energy_maink     — all 4 configs
#   Sensitivity energy_spk       — LAPA, VDN, VDS
#   Sensitivity charge_Q100      — LAPA, VDN, VDS
#   Sensitivity bkg_gamma        — LAPA, VDN, VDS
# Sensitivity oscpoint_reactor is absent BY DESIGN (Score invariant to dm2) — do not chase it.
```

## Common Variables

```bash
CAPA=hd_1x2x6_centralAPA
LAPA=hd_1x2x6_lateralAPA
VDN=vd_1x8x14_3view_30deg_nominal
VDS=vd_1x8x14_3view_30deg_shielded
ALL="$CAPA $LAPA $VDN $VDS"
HD="$CAPA $LAPA"
VD="$VDN $VDS"
NAME=marley
```

---

## §9.1.1 — Choice of Histogram Processing and Metric

*Figures: `fig:study_metric_dn`, `fig:study_metric_hep`, `fig:study_metric_sens`*
**READY — all 4 configs.** Raw vs Smoothed comes from the `SpectrumType` rows of the
**default** pkl (`--all_metrics` writes both); the old `metric_raw`/`metric_smoothed`
study labels are ORPHANS and must not be used.

```bash
# fig:study_metric_dn
python3 scripts/script_iterable_scan.py \
  --datafile DayNight_Exposure \
  --configs $ALL --name $NAME \
  -i Study -y Significance -x Exposure \
  --select SpectrumType -s Raw Smoothed --select Study -s default \
  --labelx 'Exposure (kt$\cdot$yr)' --labely 'Significance ($\sigma$)' \
  --rangex 0 20 --rangey 0 6 --horizontal 3 \
  --output fig_study_metric_dn

# fig:study_metric_hep
python3 scripts/script_iterable_scan.py \
  --datafile HEP_Exposure \
  --configs $ALL --name $NAME \
  -i Study -y Significance -x Exposure \
  --select SpectrumType -s Raw Smoothed --select Study -s default \
  --labelx 'Exposure (kt$\cdot$yr)' --labely 'Significance ($\sigma$)' \
  --rangex 0 20 --rangey 0 12 --horizontal 3 \
  --output fig_study_metric_hep

# fig:study_metric_sens  (contour: CAPA only — 4-config overlay too cluttered)
python3 scripts/script_compare_contour.py \
  --datafile Sensitivity_Contours \
  --configs $CAPA --name $NAME \
  --select Label Variable Study -s solar sin12 default \
  -y Dm2 -x Values -z Significance \
  --labelx '$\sin^2\theta_{12}$' --labely '$\Delta m^2_{21}$ (eV$^2$)' \
  --background_smoothing_sigma 3 --contour_linestyles dotted dashed solid \
  --point 0.303 6.0e-5 --point_label 'Solar' \
  --point 0.303 7.54e-5 --point_label 'Reactor' \
  --output fig_study_metric_sens
```

## §9.1.2 — Impact of Assumed Uncertainties

*Figures: `fig:study_uncertainties_dn`, `fig:study_uncertainties_hep`, `fig:study_uncertainties_sens`*
*Plot: significance at 20 kt·yr vs $\sigma^\mathrm{c}_\mathrm{rel}$ (point scan at fixed exposure)*
**DayNight READY — all 4 configs** (Asimov invariance and EG monotonicity verified 09-07).
**HEP: only `unc_bkg0` is READY**; `unc_bkg4`/`unc_bkg6` are `RERUN-B` (08-28) and the
VDN/VDS values are pathological (VDN maxPL[80] jumps 2.88→7.02). Sensitivity `unc_sig0/2/6`
READY. Scan is σ_bkg ∈ {0, 2 (default), 4, 6}% — `unc_bkg10/20` are ORPHANS, now removed.

```bash
# fig:study_uncertainties_dn
python3 scripts/script_iterable_scan.py \
  --datafile DayNight_Exposure \
  --configs $ALL --name $NAME \
  -i Study -y Significance -x Exposure \
  --select Study -s default unc_bkg0 unc_bkg4 unc_bkg6 \
  --fixed_x Exposure 20.0 \
  --labelx '$\sigma^c_\mathrm{rel}$ (%)' \
  --labely 'Significance at 20 kt$\cdot$yr ($\sigma$)' \
  --horizontal 3 --output fig_study_uncertainties_dn

# fig:study_uncertainties_hep
python3 scripts/script_iterable_scan.py \
  --datafile HEP_Exposure \
  --configs $ALL --name $NAME \
  -i Study -y Significance -x Exposure \
  --select Study -s default unc_bkg0 unc_bkg4 unc_bkg6 \
  --fixed_x Exposure 20.0 \
  --labelx '$\sigma^c_\mathrm{rel}$ (%)' \
  --labely 'Significance at 20 kt$\cdot$yr ($\sigma$)' \
  --horizontal 3 --output fig_study_uncertainties_hep

# fig:study_uncertainties_sens  (contour: CAPA only)
python3 scripts/script_iterable_scan.py \
  --datafile Sensitivity_Contours \
  --configs $CAPA --name $NAME \
  -i Study -y Score -x Exposure \
  --select Study -s default unc_bkg0 unc_bkg4 unc_bkg6 \
  --fixed_x Exposure 20.0 \
  --labelx '$\sigma^c_\mathrm{rel}$ (%)' \
  --labely 'Sensitivity Score at 20 kt$\cdot$yr' \
  --output fig_study_uncertainties_sens
```

## §9.1.3 — Impact of Oscillation Parameter Choice

*Figures: `fig:study_oscillation_dn`, `fig:study_oscillation_hep`, `fig:study_oscillation_sens`*
**READY — all 4 configs** (09-06). `oscpoint_solar` is identical to the default **by
construction** (`skip_rebin`, nominal Δm²₂₁ = 6e-5) — plot it as the null check, not as a
separate physics point. Sensitivity `oscpoint_reactor` is absent **by design**: the Score is
invariant to Δm²₂₁ because discrimination always uses both templates. Do not chase it.

```bash
# fig:study_oscillation_dn
python3 scripts/script_iterable_scan.py \
  --datafile DayNight_Exposure \
  --configs $ALL --name $NAME \
  -i Study -y Significance -x Exposure \
  --select Study -s default oscpoint_solar oscpoint_reactor \
  --labelx 'Exposure (kt$\cdot$yr)' --labely 'Significance ($\sigma$)' \
  --rangex 0 20 --rangey 0 6 --horizontal 3 \
  --output fig_study_oscillation_dn

# fig:study_oscillation_hep
python3 scripts/script_iterable_scan.py \
  --datafile HEP_Exposure \
  --configs $ALL --name $NAME \
  -i Study -y Significance -x Exposure \
  --select Study -s default oscpoint_solar oscpoint_reactor \
  --labelx 'Exposure (kt$\cdot$yr)' --labely 'Significance ($\sigma$)' \
  --rangex 0 20 --rangey 0 12 --horizontal 3 \
  --output fig_study_oscillation_hep

# fig:study_oscillation_sens  (contour: CAPA only)
python3 scripts/script_compare_contour.py \
  --datafile Sensitivity_Contours \
  --configs $CAPA --name $NAME \
  --select Label Variable Study -s solar sin12 default \
  -y Dm2 -x Values -z Significance \
  --labelx '$\sin^2\theta_{12}$' --labely '$\Delta m^2_{21}$ (eV$^2$)' \
  --background_smoothing_sigma 3 --contour_linestyles dotted dashed solid \
  --point 0.303 6.0e-5 --point_label 'Solar' \
  --point 0.303 7.54e-5 --point_label 'Reactor' \
  --overlay_datafile Sensitivity_Contours_oscpoint_solar \
  --overlay_datafile Sensitivity_Contours_oscpoint_reactor \
  --output fig_study_oscillation_sens
```

## §9.2.1 — Impact of Energy Resolution

*Figures: `fig:study_energy_resolution_dn`, `fig:study_energy_resolution_hep`, `fig:study_energy_resolution_sens`*
**cAPA `energy_maink`: re-run and verified 2026-09-07 — values reproduce exactly.**
Re-run under the one-knob policy gave DN Asimov[80] **0.32809** (Δ −2.084 vs default 2.41220)
and HEP maxPL[80] **11.92088** (Δ +4.296), identical to the pre-rerun numbers. So the
`RERUN-B` concern does not apply here either — but read the next paragraph before using it.

> **The MainK cut grid is structurally sparse, and a re-run does not change that.** Its
> DayNight grid holds **191 rows against the default's 1030** (HEP: 343 vs 1466), and the
> re-run reproduced 191 exactly. This was previously suspected to be a truncated/degenerate
> grid that a re-run would repair; it is not. MainK simply yields far fewer viable cut
> combinations. The consequence for the thesis figure: `energy_maink`'s "max across cuts" is
> a maximum over a **5× smaller grid** than the default's, so its Δ is not a like-for-like
> comparison and the −2.08σ should not be read as a pure energy-estimator effect. Its two
> `fastest_sigma2/3` JSONs are empty (2 bytes) for the same reason — no cut reaches 2σ.

*Status: DN + HEP present all 4 configs · Sensitivity: `energy_spk` cAPA only, `energy_maink`
absent on all 4 (templates exist under `MainK_energy_maink/` but `06_significance.py` never
ran, so there are no contours to export).*
*Validity: `energy_spk` cAPA/lAPA/vdN **READY**, vdS `RERUN-B`; `energy_maink` cAPA verified,
lAPA/vdN/vdS still `RERUN-B`.*

```bash
# fig:study_energy_resolution_dn  [present all 4 — RERUN-B]
python3 scripts/script_iterable_scan.py \
  --datafile DayNight_Exposure \
  --configs $ALL --name $NAME \
  -i Study -y Significance -x Exposure \
  --select Study -s default energy_maink energy_spk \
  --labelx 'Exposure (kt$\cdot$yr)' --labely 'Significance ($\sigma$)' \
  --rangex 0 20 --rangey 0 6 --horizontal 3 \
  --output fig_study_energy_resolution_dn

# fig:study_energy_resolution_hep  [present all 4 — RERUN-B]
python3 scripts/script_iterable_scan.py \
  --datafile HEP_Exposure \
  --configs $ALL --name $NAME \
  -i Study -y Significance -x Exposure \
  --select Study -s default energy_maink energy_spk \
  --labelx 'Exposure (kt$\cdot$yr)' --labely 'Significance ($\sigma$)' \
  --rangex 0 20 --rangey 0 12 --horizontal 3 \
  --output fig_study_energy_resolution_hep

# fig:study_energy_resolution_sens  [BLOCKED — Sensitivity energy_maink absent all configs,
#   energy_spk CAPA only. Needs 04_best_cuts.py for the energy group.]
# python3 scripts/script_compare_contour.py \
#   --datafile Sensitivity_Contours \
#   --configs $CAPA --name $NAME \
#   --select Label Variable Study -s solar sin12 default \
#   -y Dm2 -x Values -z Significance \
#   --labelx '$\sin^2\theta_{12}$' --labely '$\Delta m^2_{21}$ (eV$^2$)' \
#   --background_smoothing_sigma 3 --contour_linestyles dotted dashed solid \
#   --overlay_datafile Sensitivity_Contours_energy_maink \
#   --overlay_datafile Sensitivity_Contours_energy_spk \
#   --output fig_study_energy_resolution_sens
```

## §9.2.2–9.2.3 — Energy Reconstruction / Photon Detection (truth fiducialisation)

*Figures: `fig:study_fiducialisation_dn`, `fig:study_fiducialisation_hep`, `fig:study_fiducialisation_sens`*
*Status: DN + HEP + Sensitivity present all 4 configs (completed 2026-09-04)*
*Validity: `RERUN-B` — ran with re-optimised cuts. Rerun `--study fiduc_truth` before plotting.*

> **Verify VDN first.** VDN `fiduc_truth` reports DN 0.500 / EG 0.357 / HEP 2.886 — identical
> to VDN default on all three metrics. VDS HEP (3.129) also matches its default. Confirm these
> are real before they reach a figure. The gamma/neutron *"zero MCCounts — refusing to
> overwrite"* messages in `fiduc_truth.log` are the guard working as designed (backgrounds stay
> in nominal coordinates), not the cause.

```bash
# fig:study_fiducialisation_dn  [present all 4 — RERUN-B]
python3 scripts/script_iterable_scan.py \
  --datafile DayNight_Exposure \
  --configs $ALL --name $NAME \
  -i Study -y Significance -x Exposure \
  --select Study -s default fiduc_truth \
  --labelx 'Exposure (kt$\cdot$yr)' --labely 'Significance ($\sigma$)' \
  --rangex 0 20 --rangey 0 6 --horizontal 3 \
  --output fig_study_fiducialisation_dn

# fig:study_fiducialisation_hep  [present all 4 — RERUN-B]
python3 scripts/script_iterable_scan.py \
  --datafile HEP_Exposure \
  --configs $ALL --name $NAME \
  -i Study -y Significance -x Exposure \
  --select Study -s default fiduc_truth \
  --labelx 'Exposure (kt$\cdot$yr)' --labely 'Significance ($\sigma$)' \
  --rangex 0 20 --rangey 0 12 --horizontal 3 \
  --output fig_study_fiducialisation_hep

# fig:study_fiducialisation_sens  (contour: CAPA only)
python3 scripts/script_compare_contour.py \
  --datafile Sensitivity_Contours \
  --configs $CAPA --name $NAME \
  --select Label Variable Study -s solar sin12 default \
  -y Dm2 -x Values -z Significance \
  --labelx '$\sin^2\theta_{12}$' --labely '$\Delta m^2_{21}$ (eV$^2$)' \
  --background_smoothing_sigma 3 --contour_linestyles dotted dashed solid \
  --overlay_datafile Sensitivity_Contours_fiduc_truth \
  --output fig_study_fiducialisation_sens
```

## §9.2.4 — Impact of Charge Threshold

*Figures: `fig:study_charge_threshold_dn`, `fig:study_charge_threshold_hep`, `fig:study_charge_threshold_sens`*
*Plot: significance at 20 kt·yr vs $N^\mathrm{min}_\mathrm{hits}$ threshold*
**cAPA: READY and settled 2026-09-07.** All three variants re-run under the one-knob policy
(cuts and sigmas held at nominal) and all three **reproduced their pre-rerun values exactly** —
so the `RERUN-B` concern (re-tuned cuts confounding the knob) did not materialise here, and the
non-monotonic Q behaviour is real physics rather than a fitting artifact.

| Variant | DN Asimov | Δ DN | HEP PL | Δ HEP | Results written |
|---|---|---|---|---|---|
| default (no charge cut) | 2.41220 | — | 7.62441 | — | 09-06 18:32 |
| charge_Q50 | 1.58381 | −0.828 | 6.66128 | −0.963 | 09-06 13:59 |
| charge_Q100 | 1.36458 | −1.048 | 6.23669 | −1.388 | 09-07 11:28 |
| charge_Q500 | 1.63178 | −0.780 | 6.46938 | −1.155 | 09-07 12:38 |

Plot pkls regenerated clean 09-07 15:44–15:52 (Counts 10 rows, Exposure DN 10 / HEP 2, single
`Study` value, zero NaN). Provenance: `run_studies.py --study charge --variant charge_Q100
charge_Q500 --config hd_1x2x6_centralAPA --folder Truncated --rewrite`, then
`significance_plot.py` / `exposure_plot.py` per variant with `--charge_threshold`.

**lAPA / vdN / vdS still pending** — same rerun needed. Sensitivity leg: `Q50` cAPA/lAPA/vdN
only, `Q100` cAPA-only, `Q500` absent everywhere.

> **Two code bugs were fixed to make this work** (2026-09-07). `significance_plot.py` had no
> `--charge_threshold` flag, so `study_context()` never marked charge runs as template variants
> and looked for *unlabeled* Rebin pkls; and its three `load_available_background_dataframes()`
> call sites never passed `study_label`, though the loader has always accepted it. Together
> these made every charge variant fail with `RuntimeError: Essential background 'gamma' missing`
> and produce **no Counts pkl at all** — which is why the inventory found charge variants to be
> Exposure-only. `run_sensitivity.py` also now threads `charge_threshold_only_args_for()` into
> all four `significance_plot.py` invocations. This affected every template variant
> (`charge_*`, and any study where `template_suffix` is set).

```bash
# fig:study_charge_threshold_dn  [OK — all 4 configs]
python3 scripts/script_iterable_scan.py \
  --datafile DayNight_Exposure \
  --configs $ALL --name $NAME \
  -i Study -y Significance -x Exposure \
  --select Study -s default charge_Q50 charge_Q100 charge_Q500 \
  --fixed_x Exposure 20.0 \
  --labelx '$N^\mathrm{min}_\mathrm{hits}$' \
  --labely 'Significance at 20 kt$\cdot$yr ($\sigma$)' \
  --horizontal 3 --output fig_study_charge_threshold_dn

# fig:study_charge_threshold_hep  [OK — all 4 configs]
python3 scripts/script_iterable_scan.py \
  --datafile HEP_Exposure \
  --configs $ALL --name $NAME \
  -i Study -y Significance -x Exposure \
  --select Study -s default charge_Q50 charge_Q100 charge_Q500 \
  --fixed_x Exposure 20.0 \
  --labelx '$N^\mathrm{min}_\mathrm{hits}$' \
  --labely 'Significance at 20 kt$\cdot$yr ($\sigma$)' \
  --horizontal 3 --output fig_study_charge_threshold_hep

# fig:study_charge_threshold_sens  (contour: CAPA only)
python3 scripts/script_iterable_scan.py \
  --datafile Sensitivity_Contours \
  --configs $CAPA --name $NAME \
  -i Study -y Score -x Exposure \
  --select Study -s default charge_Q50 charge_Q100 charge_Q500 \
  --fixed_x Exposure 20.0 \
  --labelx '$N^\mathrm{min}_\mathrm{hits}$' \
  --labely 'Sensitivity Score at 20 kt$\cdot$yr' \
  --output fig_study_charge_threshold_sens
```

## §9.2.5 — Impact of Background Model (gamma suppression)

*Figures: `fig:study_improved_bkg_dn`, `fig:study_improved_bkg_hep`, `fig:study_improved_bkg_sens`*
*Status: DN + HEP present all 4 configs · Sensitivity CAPA only*
*Validity: `RERUN-B` — ran with re-optimised cuts. Rerun `--study bkg_gamma` before plotting.*
*Uses `ClusterEnergy`, not `SolarEnergy` — datafile stems differ.*

```bash
# fig:study_improved_bkg_dn  [present all 4 — RERUN-B]
python3 scripts/script_iterable_scan.py \
  --datafile DayNight_Exposure \
  --configs $ALL --name $NAME \
  -i Study -y Significance -x Exposure \
  --select Study -s default bkg_gamma \
  --labelx 'Exposure (kt$\cdot$yr)' --labely 'Significance ($\sigma$)' \
  --rangex 0 20 --rangey 0 6 --horizontal 3 \
  --output fig_study_improved_bkg_dn

# fig:study_improved_bkg_hep  [present all 4 — RERUN-B]
python3 scripts/script_iterable_scan.py \
  --datafile HEP_Exposure \
  --configs $ALL --name $NAME \
  -i Study -y Significance -x Exposure \
  --select Study -s default bkg_gamma \
  --labelx 'Exposure (kt$\cdot$yr)' --labely 'Significance ($\sigma$)' \
  --rangex 0 20 --rangey 0 12 --horizontal 3 \
  --output fig_study_improved_bkg_hep

# fig:study_improved_bkg_sens  (contour: CAPA only — LAPA/VDN/VDS Sensitivity absent)
# The Sensitivity stage errored on 2026-09-04: "No background templates found in
# .../SENSITIVITY/{cfg}/background/truncated/ClusterEnergy". Build those templates first.
python3 scripts/script_compare_contour.py \
  --datafile Sensitivity_Contours \
  --configs $CAPA --name $NAME \
  --select Label Variable Study -s solar sin12 default \
  -y Dm2 -x Values -z Significance \
  --labelx '$\sin^2\theta_{12}$' --labely '$\Delta m^2_{21}$ (eV$^2$)' \
  --background_smoothing_sigma 3 --contour_linestyles dotted dashed solid \
  --overlay_datafile Sensitivity_Contours_bkg_gamma \
  --output fig_study_improved_bkg_sens
```

## §9.3.2 — Summary Bar Chart

*Figure: `fig:sensitivity_studies_summary`*
*Baseline (HD Central 20 kt·yr): DN=3.85σ, HEP=10.63σ, Sens score=1.31*
*Summary bar chart: CAPA only (primary detector for comparison)*
*[BLOCKED — this figure aggregates every variant, so it inherits every `RERUN-A`/`RERUN-B`
mark in the matrix. Build it last, after all reruns land. Baselines below predate the
2026-09-02 default rerun and need re-deriving too.]*

```bash
# python3 scripts/script_compare_pareto.py \
#   --datafile DayNight_Exposure HEP_Exposure Sensitivity_Significance \
#   --configs $CAPA --name $NAME \
#   --select Study -s default \
# #              unc_bkg0 unc_bkg4 unc_bkg6 \
#              oscpoint_solar oscpoint_reactor \
#              energy_spk energy_maink \
#              charge_Q50 charge_Q100 charge_Q500 \
#              fiduc_truth bkg_gamma \
#   --fixed_x Exposure 20.0 \
#   --baseline_dn 3.85 --baseline_hep 10.63 --baseline_sens 1.31 \
#   --output fig_sensitivity_studies_summary
```
