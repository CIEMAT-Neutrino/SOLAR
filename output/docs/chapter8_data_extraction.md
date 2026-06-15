# Chapter 8 Data Extraction — DUNE Solar Neutrino Sensitivity

Generated: 2026-06-15. All values extracted directly from repository output files and configs.
Fiducial: `truncated`. Signal: `marley`. Energy variable: `SolarEnergy` throughout.

---

## TASK 1: Analysis cut thresholds

Cuts are **analysis-specific** — DayNight, HEP, Sensitivity each have independent optimal cuts per config.
No `ref_exposure` field is encoded in any cut JSON; the optimization score functions are exposure-independent.

**Source:** `config/<CONFIG>/best-sigma-json/<ANALYSIS>/truncated/marley/<CONFIG>_marley_highest_<ANALYSIS>.json`

Cut semantics (from `src/physics/signal/03_analysis.py:331–335`):
`NHits ≥ N_hits_min`  AND  `AdjClNum ≤ N_adjcl_max − 1`  AND  `MatchedOpFlashNHits ≥ N_ophits_min`

| Analysis    | Config                          | N_hits_min | N_ophits_min | N_adjcl_max | ref_exposure       |
|-------------|----------------------------------|-----------|--------------|-------------|--------------------|
| DayNight    | hd_1x2x6_centralAPA              | 3         | 12           | 3           | N/A — not in config|
| DayNight    | hd_1x2x6_lateralAPA              | 4         | 4            | 4           | N/A                |
| DayNight    | vd_1x8x14_3view_30deg_nominal    | 6         | 19           | 6           | N/A                |
| DayNight    | vd_1x8x14_3view_30deg_shielded   | 6         | 13           | 9           | N/A                |
| HEP         | hd_1x2x6_centralAPA              | 4         | 13           | 5           | N/A                |
| HEP         | hd_1x2x6_lateralAPA              | 6         | 4            | 5           | N/A                |
| HEP         | vd_1x8x14_3view_30deg_nominal    | 10        | 10           | 7           | N/A                |
| HEP         | vd_1x8x14_3view_30deg_shielded   | 8         | 20           | 6           | N/A                |
| Sensitivity | hd_1x2x6_centralAPA              | 1         | 8            | 4           | N/A                |
| Sensitivity | hd_1x2x6_lateralAPA              | 3         | 10           | 2           | N/A                |
| Sensitivity | vd_1x8x14_3view_30deg_nominal    | 8         | 10           | 8           | N/A                |
| Sensitivity | vd_1x8x14_3view_30deg_shielded   | 8         | 10           | 10          | N/A                |

---

## TASK 2: Precise numerical sensitivity results

**Source:** `output/data/analysis/day-night/<CONFIG>/marley/truncated/<CONFIG>_marley_DayNight_Exposure.pkl`
and `output/data/analysis/hep/<CONFIG>/marley/truncated/<CONFIG>_marley_HEP_Exposure.pkl`

Each pkl contains a DataFrame with (exposure, significance) arrays interpolated over ~100 exposure points.
Z at 10/20 yr and exposure crossing computed by linear interpolation.

### DAY-NIGHT ASYMMETRY

**Asimov / Smoothed** (primary):

| Config                          | Z at 10 yr | Z at 20 yr | Exposure for 3σ (yr) | Exposure for 5σ (yr) |
|---------------------------------|-----------|-----------|---------------------|---------------------|
| hd_1x2x6_centralAPA             | 2.720     | 3.846     | 12.17               | >scan range (>25 yr)|
| hd_1x2x6_lateralAPA             | 1.080     | 1.527     | >scan range         | >scan range         |
| vd_1x8x14_3view_30deg_nominal   | 0.526     | 0.744     | >scan range         | >scan range         |
| vd_1x8x14_3view_30deg_shielded  | 1.137     | 1.609     | >scan range         | >scan range         |
| Phase I Combined                | —         | —         | —                   | —                   |

**Gaussian / Smoothed** (cross-check):

| Config                          | Z at 10 yr | Z at 20 yr | Exposure for 3σ (yr) |
|---------------------------------|-----------|-----------|---------------------|
| hd_1x2x6_centralAPA             | 2.692     | 3.807     | 12.42               |
| hd_1x2x6_lateralAPA             | 0.984     | 1.392     | >scan range         |
| vd_1x8x14_3view_30deg_nominal   | 0.499     | 0.706     | >scan range         |
| vd_1x8x14_3view_30deg_shielded  | 1.110     | 1.570     | >scan range         |

Phase I Combined: **NOT FOUND** in output tree.

---

### HEP DISCOVERY (profile-likelihood significance)

> **WARNING:** VD exposure pkl files (`vd_1x8x14_3view_30deg_nominal/shielded`) were generated
> with stale cuts (NHits=1, OpHits=4–5, AdjCl=13–19) — these do NOT match current best-sigma JSON
> (NHits=10/8). Re-run `src/physics/hep/exposure_plot.py` before citing VD HEP numbers.

**ProfileLikelihood / Raw** (row 0):

| Config                          | Z at 10 yr | Z at 20 yr | Exposure for 3σ (yr) | Exposure for 5σ (yr) |
|---------------------------------|-----------|-----------|---------------------|---------------------|
| hd_1x2x6_centralAPA             | 7.497     | 10.603    | 1.60                | 4.45                |
| hd_1x2x6_lateralAPA             | 3.484     | 4.933     | 7.42                | 20.55               |
| vd_1x8x14_3view_30deg_nominal   | 1.528 ⚠   | 2.192 ⚠   | >scan range         | >scan range         |
| vd_1x8x14_3view_30deg_shielded  | 1.589 ⚠   | 2.277 ⚠   | >scan range         | >scan range         |
| Phase I Combined                | —         | —         | —                   | —                   |

**ProfileLikelihood / Smoothed** (row 1):

| Config                          | Z at 10 yr | Z at 20 yr | Exposure for 3σ (yr) | Exposure for 5σ (yr) |
|---------------------------------|-----------|-----------|---------------------|---------------------|
| hd_1x2x6_centralAPA             | 5.256     | 7.433     | 3.26                | 9.05                |
| hd_1x2x6_lateralAPA             | 1.382     | 1.956     | >scan range         | >scan range         |
| vd_1x8x14_3view_30deg_nominal   | 0.649 ⚠   | 0.930 ⚠   | 25.43 ⚠             | 25.62 ⚠             |
| vd_1x8x14_3view_30deg_shielded  | 0.855 ⚠   | 1.230 ⚠   | 20.20 ⚠             | 20.37 ⚠             |
| Phase I Combined                | —         | —         | —                   | —                   |

---

### OSCILLATION PARAMETER SENSITIVITY

Two parallel sets — same optimal cuts, different normalization:

- **Set A** `config/<CONFIG>/sensitivity-json/truncated/<CONFIG>_highest_Sensitivity.json`
  (truth-backend chi², higher absolute values)
- **Set B** `config/<CONFIG>/best-sigma-json/sensitivity/truncated/marley/<CONFIG>_marley_highest_Sensitivity.json`
  (marley-weighted chi²)

**Set A** (truth backend, `sensitivity-json/truncated/`):

| Config                          | χ²(solar fit @ reactor) | χ²(reactor fit @ solar) | Score = avg | Significance ≈ √Score |
|---------------------------------|------------------------|------------------------|-------------|------------------------|
| hd_1x2x6_centralAPA             | 142.833                | 142.517                | 142.675     | 11.95 σ                |
| hd_1x2x6_lateralAPA             | 0.464                  | 0.363                  | 0.413       | 0.64 σ                 |
| vd_1x8x14_3view_30deg_nominal   | 0.121                  | 0.115                  | 0.118       | 0.34 σ                 |
| vd_1x8x14_3view_30deg_shielded  | 0.551                  | 0.541                  | 0.546       | 0.74 σ                 |
| Phase I Combined                | —                      | —                      | —           | —                      |

**Set B** (marley-weighted, `best-sigma-json/sensitivity/truncated/marley/`):

| Config                          | χ²(solar fit @ reactor) | χ²(reactor fit @ solar) | Score = avg |
|---------------------------------|------------------------|------------------------|-------------|
| hd_1x2x6_centralAPA             | 2.929                  | 2.354                  | 2.641       |
| hd_1x2x6_lateralAPA             | 0.199                  | 0.135                  | 0.167       |
| vd_1x8x14_3view_30deg_nominal   | 0.049                  | 0.047                  | 0.048       |
| vd_1x8x14_3view_30deg_shielded  | 0.245                  | 0.241                  | 0.243       |

Phase I Combined: **NOT FOUND** in output tree.

---

## TASK 3: Oscillation parameter best-fit values

**Source:** `config/analysis/physics.json` (lines 8–12)

`SIN12 = 0.304` applies to both reference points; Δm²₂₁ is the distinguishing parameter.
sin²θ₁₃ is marginalized over `[0.017, 0.025]` in `NUISANCE_PROFILE = "full"` mode.

```
Solar best-fit:    sin²θ₁₂ = 0.304,   Δm²₂₁ = 6.00×10⁻⁵ eV²,  sin²θ₁₃ = 0.0220
Reactor best-fit:  sin²θ₁₂ = 0.304,   Δm²₂₁ = 7.54×10⁻⁵ eV²,  sin²θ₁₃ = 0.0220
```

[file: config/analysis/physics.json:8–12]

---

## TASK 4: Background model confirmation

**Source:** `config/analysis/config.json` (lines 22–53), `config/analysis/backgrounds.json` (lines 26–30)

```
BACKGROUND_ERROR (σ_rel):  0.02  (2%)  — all three analyses (DayNight, HEP, Sensitivity)

SIGNAL_ERROR (σ_pred) per analysis:
  DayNight:    0.00  (0%)
  HEP:         0.30  (30%)
  Sensitivity: 0.04  (4%)

Background components in all analyses: gamma, neutron, radiological

vd_1x8x14_3view_30deg_shielded shielding (cavernwall_gamma only):
  reduction_factor:  0.2846  →  71.5% gamma reduction
  threshold_mev:     3.3 MeV  (reduction applied only above this energy)

Neutron reduction factor:  NOT DEFINED in backgrounds.json — no separate shielding entry for neutron.
```

---

## TASK 5: Cut optimization scan range

**Source:** `lib/defaults.py:107–111`, `src/physics/signal/03_analysis.py:307–309`,
`src/physics/sensitivity/04_best_cuts.py:228,247`

```
N_hits   scan values:  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]  (20 values)
N_ophits scan values:  [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]            (17 values, nhits[3:])
N_adjcl  scan values:  [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]   (20 values, nhits reversed)

Grid total:  20 × 17 × 20 = 6 800 cut combinations per (config × energy × analysis)

Score function (DayNight):     Maximize Σ_bin Z_Gaussian²  over nadir bins
                               Stored as "Sigma2" in best-sigma JSON
                               DEFAULT_METRIC = "gaussian"  [config/analysis/config.json:77]

Score function (HEP):          Maximize profile-likelihood significance Z_PL
                               Stored as "Values" = Z_PL
                               DEFAULT_METRIC = "profile_likelihood"  [config/analysis/config.json:66]

Score function (Sensitivity):  Maximize Score = 0.5 × (χ²_solar@reactor + χ²_reactor@solar)
                               where χ²_X@Y = Sensitivity_Fitter chi² with X-truth fit evaluated at Y parameter point
                               Best cut = argmax(Score) over all grid candidates
                               [src/physics/sensitivity/04_best_cuts.py:228, 247]
```

---

## Flags / action items

1. **HEP VD configs stale:** `vd_1x8x14_3view_30deg_nominal` and `vd_1x8x14_3view_30deg_shielded`
   HEP exposure pkls were built with NHits=1 cuts. Re-run `src/physics/hep/exposure_plot.py`
   with current best-sigma cuts (NHits=10/8) before citing those significance values.

2. **Sensitivity chi² sets:** Two parallel sets (truth vs marley-weighted) exist.
   Clarify which is canonical for the thesis before citing Set A vs Set B values.

3. **Phase I Combined:** No combined-significance output found in `output/`.
   Either has not been computed or lives outside the checked output tree.

4. **DayNight 5σ:** No config reaches 5σ within the scan range shown in the pkls (~25 yr).
   Only `hd_1x2x6_centralAPA` reaches 3σ at ~12.2 yr.
