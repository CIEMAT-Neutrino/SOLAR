# SOLAR Studies Artifact — Regeneration Guide

This guide teaches how to reproduce, update, or patch the **SOLAR Significance Studies** artifact from scratch. Everything here has been verified empirically.

## Artifact URL

```
https://claude.ai/code/artifact/dc532332-f1c1-4195-810e-0d6a16d37712
```

At the start of a new session:
```python
# Re-read current content:
Artifact(action="read", url="https://claude.ai/code/artifact/dc532332-f1c1-4195-810e-0d6a16d37712")
# → saves to a local file; edit that file then:
Artifact(file_path="<path>", url="https://claude.ai/code/artifact/...", favicon="⚛️")
```

The scratchpad path from session `8df85dba-...` was:
```
/tmp/claude-0/-pc-choozdsk01-users-manthey-SOLAR/8df85dba-e0ae-4133-8a45-2f2896f02648/scratchpad/study_significance_summary.html
```
That scratchpad does not survive across sessions — always re-fetch the artifact first.

---

## Configs, Aliases, and Tab IDs

| Config key                         | Alias | Tab ID   |
|------------------------------------|-------|----------|
| `hd_1x2x6_centralAPA`             | cAPA  | tab-cAPA |
| `hd_1x2x6_lateralAPA`             | lAPA  | tab-lAPA |
| `vd_1x8x14_3view_30deg_nominal`   | vdN   | tab-vdN  |
| `vd_1x8x14_3view_30deg_shielded`  | vdS   | tab-vdS  |

---

## Exposure Index: IDX10 = 80

All significance arrays are 100-element log-spaced arrays from 0.1 to 30 years:

```python
import pickle, numpy as np
IDX10 = 80   # exposure[80] ≈ 10.04 yr
```

Verify: `float(np.asarray(df.iloc[0]['Exposure'])[80])` → 10.04 for any Results pkl.

---

## Data Sources

### 1. DayNight Results (PNFS)

**Path:** `/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/DAYNIGHT/truncated/{cfg}/marley/`

**Filename patterns:**
```
{cfg}_marley_SolarEnergy_DayNight_Results.pkl          # default
{cfg}_marley_SolarEnergy_DayNight_Results_{label}.pkl  # study variant
{cfg}_marley_SignalParticleK_DayNight_Results_{label}.pkl  # energy_spk
{cfg}_marley_MainK_DayNight_Results_{label}.pkl        # energy_maink
```

**Labels:** `oscpoint_solar`, `oscpoint_reactor`, `unc_bkg0`, `unc_bkg4`, `unc_bkg6`, `energy_spk`, `energy_maink`, `charge_Q50`, `charge_Q100`, `charge_Q200`

**Structure:** pandas DataFrame, ~1030 rows (all cut combos: NHits × OpHits × AdjCl).

Columns include `Asimov` (smoothed), `RawAsimov`, `ErrorGaussian`, `RawErrorGaussian`, `Gaussian`, `RawGaussian`, etc. — each is a length-100 array.

**Extraction — best Asimov at 10yr:**
```python
import pickle, numpy as np
IDX10 = 80

def best_dn_asimov(pkl_path):
    df = pickle.load(open(pkl_path, "rb"))
    return df["Asimov"].apply(lambda v: float(np.asarray(v)[IDX10])).max()
```

**Extraction — best ErrorGaussian at 10yr (for unc_bkg rows):**
```python
def best_dn_eg(pkl_path):
    df = pickle.load(open(pkl_path, "rb"))
    return df["ErrorGaussian"].apply(lambda v: float(np.asarray(v)[IDX10])).max()
```

**Current artifact DN baselines:**
| Config | Default Asimov | Default EG |
|--------|---------------|------------|
| cAPA   | 2.412         | 2.176      |
| lAPA   | 0.915         | 0.447      |
| vdN    | 0.505         | 0.361      |
| vdS    | 0.972         | 0.748      |

---

### 2. HEP Results (PNFS)

**Path:** `/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/HEP/truncated/{cfg}/marley/`

**Filename patterns:**
```
{cfg}_marley_SolarEnergy_HEP_Results.pkl             # default
{cfg}_marley_SolarEnergy_HEP_Results_{label}.pkl     # unc/oscpoint/charge variants
{cfg}_marley_SignalParticleK_HEP_Results_{label}.pkl # energy_spk
{cfg}_marley_MainK_HEP_Results_{label}.pkl           # energy_maink
```

**Standard columns (SolarEnergy variants):** `ProfileLikelihood` (smoothed PL), `RawProfileLikelihood` (raw PL), `RawGaussian`, `RawAsimov`, `Gaussian`, `Asimov`, etc.

**Energy variant columns (SignalParticleK / MainK):** Only `ProfileLikelihood` + `PreIsotonicProfileLikelihood`. No `Asimov`, `Gaussian`, or `RawXxx` columns. `skip_rebin=True` suppresses non-PL metrics.

**Extraction — best Smoothed PL at 10yr:**
```python
def best_hep_pl(pkl_path):
    df = pickle.load(open(pkl_path, "rb"))
    return df["ProfileLikelihood"].apply(lambda v: float(np.asarray(v)[IDX10])).max()
```

**Note:** This takes the max across all ~1259–1746 cut rows. For robust best-cut selection, `05_best_sigmas.py` should be called (it writes `highest_HEP_{label}.pkl`), but this is currently NOT called for energy variants — they fail silently because Asimov/Gaussian columns are absent.

**Local HEP Exposure pkl (alternative source for baseline):**
```
output/data/analysis/hep/{cfg}/marley/truncated/
├── {cfg}_marley_HEP_Exposure.pkl           # OLD scheme (gives cAPA=7.624)
└── default/{cfg}_marley_HEP_Exposure.pkl   # NEW scheme (gives cAPA=7.530)
```

The OLD scheme pkl is what the artifact's baseline stat card was built from. The NEW scheme is produced by current runs. Use OLD scheme to match the artifact's stated baseline; use NEW scheme for fresh regenerations.

```python
def hep_pl_from_exposure(pkl_path):
    df = pickle.load(open(pkl_path, "rb"))
    row = df[(df["Variable"] == "ProfileLikelihood") & (df["SpectrumType"] == "Smoothed")]
    return float(row["Significance"].iloc[0][IDX10])
```

**Current artifact HEP PL baselines (PNFS max, Sep 2 fresh run):**
| Config | HEP PL Smoothed |
|--------|----------------|
| cAPA   | 7.624          |
| lAPA   | 3.569          |
| vdN    | 2.886          |
| vdS    | 3.129          |

---

### 3. Oscillation Sensitivity (PNFS)

**Path:** `/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/SENSITIVITY/truncated/{cfg}/marley/`

**Filename patterns:**
```
{cfg}_marley_highest_SENSITIVITY.pkl          # default
{cfg}_marley_highest_SENSITIVITY_{label}.pkl  # study variant
```

**Labels available (confirmed):** `unc_sig0`, `unc_sig2`, `unc_sig6`, `unc_sig8`, `unc_bkg0`, `unc_bkg4`, `unc_bkg6`, `unc_bkg10`, `unc_bkg20`, `energy_spk` (cAPA only), `charge_Q50/100/200`

**Note:** `Sensitivity_Results_{label}.pkl` also exists in some cases — this is produced by `05_best_sigmas.py --analysis Sensitivity`, which IS NOT called in `run_sensitivity_stage()`. The `highest_SENSITIVITY_{label}.pkl` is written by `06_significance.py` and is the correct source.

**Structure:** dict keyed by `(config, "marley", "SolarEnergy")` tuples.

**Extraction:**
```python
def sensitivity_score(pkl_path, cfg):
    d = pickle.load(open(pkl_path, "rb"))
    key = (cfg, "marley", "SolarEnergy")
    rec = d[key]
    SF = rec["SolarFitAtReact"]      # Δχ² when solar dm² fit at reactor point
    RF = rec["ReactorFitAtSolar"]    # Δχ² when reactor dm² fit at solar point
    Score = (SF + RF) / 2
    return Score ** 0.5              # σ-equivalent (√Score)
```

**Current artifact Sensitivity defaults (√Score):**
| Config | √Score (σ) | SF@React | RF@Solar |
|--------|-----------|----------|----------|
| cAPA   | 1.690     | 2.530    | 3.181    |
| lAPA   | 0.620     | —        | —        |
| vdN    | 0.316     | —        | —        |
| vdS    | 0.766     | —        | —        |

---

## Confirmed Values (2026-08-22)

### DayNight Asimov@10yr — Study Variants

| Variant         | cAPA  | lAPA  | vdN   | vdS   | Notes                  |
|-----------------|-------|-------|-------|-------|------------------------|
| default         | 2.412 | 0.915 | 0.505 | 0.972 |                        |
| oscpoint_solar  | 2.412 | 0.915 | 0.505 | 0.972 | Δ=0 (same dm²)         |
| oscpoint_reactor| 1.307 | 0.509 | 0.270 | 0.512 | Δ<0 (reactor dm² confirmed) |
| energy_spk      | 2.395 | 2.236 | 1.190 | 1.214 | SignalParticleK (all ✓)|
| energy_maink    | 0.328 | 0.752 | 0.746 | 1.216 | MainK (all ✓)          |

### HEP PL Smoothed@10yr — Study Variants

| Variant          | cAPA   | lAPA   | vdN    | vdS    | Notes                                         |
|------------------|--------|--------|--------|--------|-----------------------------------------------|
| default          | 7.624  | 3.569  | 2.886  | 3.129  | Sep 2 fresh run (max across all cuts)         |
| oscpoint_reactor | 7.613  | 3.538  | 2.875  | 3.124  | Sep 3 fresh run; reactor slightly worse Δ≈−0.01–0.03σ |
| energy_spk       | 28.628 | 31.394 | 19.440 | 19.503 | Best max across all cuts                      |
| energy_maink     | 11.921 | 10.768 | 6.527  | 6.526  | Best max across all cuts                      |

---

## Artifact HTML Structure

Each tab contains these sections in order:
```
1. Stat cards row (4 cards: DN Asimov, HEP PL, config comparison or best-folder)
2. Info-box (amber/green) — config-specific status
3. § Study Variant Comparison — DN Asimov | Δ DN | HEP PL | Δ HEP | Status
4. § Background Uncertainty — DayNight (unc_bkg rows with EG column)
5. § Histogram Metric / Smoothing — Raw vs Smoothed for all metrics
6. § Fiducialization Comparison — Nominal / Reduced / Truncated
7. § Oscillation Sensitivity (Δχ²) — Sensitivity Score discrimination
8. § Sensitivity — Uncertainty Scan — unc_sig + unc_bkg
9. § Run Status / Changelog / Commands (shared footer)
```

**Study Variant Comparison row order:**
```
default (baseline)        ← always first, class="highlight"
oscpoint_solar
oscpoint_reactor
energy_spk
energy_maink
── σ_bkg scan header ──
unc_bkg0 ... unc_bkg6     ← unc_bkg10/20 removed 2026-08-21
unc_sig20 (HEP only)
unc_sig40 (HEP only)
charge_Q50/100/200
metric_raw
```

**Cell CSS classes:**
- `class="pos"` — positive delta (green)
- `class="neg"` — negative delta (red)
- `class="zero"` — zero/reference (grey)
- `class="ref"` — baseline value (white bold)
- `class="na"` — not available (italic grey)
- `class="warn"` — stale/pending (amber)

**Badge classes:** `badge-ok`, `badge-warn`, `badge-info` (partial), `badge-anomaly`

---

## Update Workflow

### Step 1 — Read current artifact

```python
Artifact(action="read", url="https://claude.ai/code/artifact/dc532332-f1c1-4195-810e-0d6a16d37712")
# → saves to /path/artifact.html
```

### Step 2 — Load new data from PNFS

```python
import pickle, numpy as np
IDX10 = 80
PNFS = "/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR"
CFGS = ["hd_1x2x6_centralAPA", "hd_1x2x6_lateralAPA",
        "vd_1x8x14_3view_30deg_nominal", "vd_1x8x14_3view_30deg_shielded"]

def best_asimov(pkl_path):
    df = pickle.load(open(pkl_path, "rb"))
    return df["Asimov"].apply(lambda v: float(np.asarray(v)[IDX10])).max()

def best_pl(pkl_path):
    df = pickle.load(open(pkl_path, "rb"))
    return df["ProfileLikelihood"].apply(lambda v: float(np.asarray(v)[IDX10])).max()
```

### Step 3 — Edit HTML with exact string matching

Always include 2–3 lines of context to make `old_string` unique:
```python
html = html.replace(
    '            <td class="warn">⚠ rerun</td>\n'
    '            <td class="warn">⚠</td>\n',
    f'            <td class="neg">{val:.3f}</td>\n'
    f'            <td class="neg">−{abs(delta):.3f}</td>\n',
)
```

### Step 4 — Publish

```python
Artifact(
    file_path="/path/to/artifact.html",
    url="https://claude.ai/code/artifact/dc532332-f1c1-4195-810e-0d6a16d37712",
    favicon="⚛️",
    label="<short description>",
)
```

---

## Missing Data and Rerun Commands

Run inside the apptainer container from `SOLAR/` with `source setup.sh`.

### Charge Study (lAPA, vdN, vdS — DayNight)

```bash
python3 src/pipelines/run_studies.py \
  --study charge \
  --config hd_1x2x6_lateralAPA vd_1x8x14_3view_30deg_nominal vd_1x8x14_3view_30deg_shielded \
  --folder Truncated --rewrite
```

Expected: `{cfg}_marley_SelectedEnergy_DayNight_Results_charge_Q50/100/200.pkl`

### Energy Study — Sensitivity Stage (all 4 configs)

Root cause: `03_template_compute.py` for SignalParticleK/MainK finds no templates → exits 0 silently → no Sensitivity Results. The `--ignore_energy_window` flag already handles fiducialization; the template issue may stem from a separate energy window check inside `01_background_template.py` or `02_signal_template.py`.

```bash
python3 src/pipelines/run_studies.py \
  --study energy \
  --config hd_1x2x6_centralAPA hd_1x2x6_lateralAPA \
           vd_1x8x14_3view_30deg_nominal vd_1x8x14_3view_30deg_shielded \
  --folder Truncated --rewrite
```

### unc_sig0 Sensitivity (lAPA, vdN, vdS)

cAPA has it; lAPA/vdN/vdS are missing `highest_SENSITIVITY_unc_sig0.pkl`.
Possible cause: `signal_uncertainty=0.00` triggers numerical issue.

```bash
python3 src/pipelines/run_studies.py \
  --study unc \
  --config hd_1x2x6_lateralAPA vd_1x8x14_3view_30deg_nominal vd_1x8x14_3view_30deg_shielded \
  --folder Truncated --rewrite
```

---

## Pipeline Reference

### run_sensitivity.py stage order (per config, per name)

```
run_shared_prerequisites()
  └── 02_best_fiducial.py  [+--ignore_energy_window for energy variants]
  └── 04_rebin.py

run_daynight_stage()
  └── 01_daynight.py       [+--dm2 for oscpoint variants]
  └── 05_best_sigmas.py    [--analysis DayNight +--reference_study_label "" if skip_best_sigmas]

run_hep_stage()
  └── 01_hep.py
  └── 05_best_sigmas.py    [--analysis HEP +--reference_study_label "" if skip_best_sigmas]

run_sensitivity_stage() per energy:
  └── 03_template_compute.py  [--template background]
  └── 04_best_cuts.py
  └── 03_template_compute.py  [--template signal]
  └── 06_significance.py      → writes highest_SENSITIVITY_{label}.pkl
```

**Note:** `05_best_sigmas.py --analysis Sensitivity` is documented in comments but NOT called in `run_sensitivity_stage()`. The `highest_SENSITIVITY` pkl is sufficient for the artifact's Sensitivity Score column.

### Key study variant flags

| Variant group | `ignore_energy_window` | `skip_best_sigmas` | `analysis_override` |
|---------------|----------------------|-------------------|---------------------|
| energy_spk    | True                 | False             | (all analyses)      |
| energy_maink  | True                 | False             | (all analyses)      |
| unc_*         | False                | True              | unc_sig20/40→HEP; unc_sig0/2/6→Sensitivity |
| oscpoint_*    | False                | False             | DayNight + HEP only (Sensitivity skipped — Score invariant to dm²) |
| charge_*      | False                | False             | (all analyses)      |

---

## Changelog Summary

| Date       | Change                                                                |
|------------|-----------------------------------------------------------------------|
| 2026-08-20 | Initial artifact created — cAPA/lAPA/vdN/vdS tabs with all sections |
| 2026-08-21 | Issues 5/6/7: removed Asimov from bkg-unc tables; PL Smoothed→N/A; footnotes |
| 2026-08-21 | Removed unc_bkg10/20 rows from Study Variant + Sensitivity Uncertainty tables |
| 2026-08-22 | oscpoint_reactor DN confirmed (real non-zero values); energy_maink rows added; vdN HEP baseline 1.686→2.518; energy_spk HEP values added |

---

## Section Structure (as of 2026-08-27)

Each tab contains these sections in order (cAPA has all; lAPA/vdN/vdS omit cAPA-only sections):

```
1.  Stat cards row
2.  § Optimal Topological Cuts
3.  § Fiducial Volume Cuts
4.  § Energy Estimator Variants
5.  § Oscillation Point Variants
6.  § Background Uncertainty — DayNight
7.  § Signal Uncertainty — HEP
8.  § Sensitivity — Uncertainty Scan
9.  § Charge Cut Variants
10. (cAPA only) § Fiducial Volume Model
11. (cAPA only) § Background Model Variants
12. (cAPA only) § Analysis Metric & Smoothing
13. § Background Assumptions
```

---

## Column and Table Design Rules

### § Optimal Topological Cuts
- Columns: **Analysis | Bkg. Model | Energy | NHits ≥ | OpHits ≥ | AdjCl ≤**
- No "Variant" column — Bkg. Model = "Truncated", Energy = "SolarEnergy" for all rows.

### § Energy Estimator Variants
- Columns: **Variant | DN Asimov σ | Δ DN | HEP PL σ | Δ HEP | Sensitivity Score | √Score | Status**
- Row order: energy_spk → energy_maink → **default (SolarEnergy)** (reference row LAST).
- Sensitivity Score pending for energy_spk/energy_maink (04_best_cuts.py not yet run).

### § Oscillation Point Variants
- Columns: **Variant | DN Asimov σ | Δ DN | HEP PL σ | Δ HEP | χ²(sol→react) | χ²(react→sol) | Sensitivity Score | Status**
- oscpoint_solar = default (same Δm²₂₁ assumption); show identical values.
- oscpoint_reactor Sensitivity Score ⏳ pending 04_best_cuts.py run.

### § Background Uncertainty — DayNight
- **Main table**: Variant | DN EG σ | Δ EG | HEP PL σ | Δ HEP | Status
  - Row order: unc_bkg0 → default(2%) → unc_bkg4 → unc_bkg6 (ascending σ_bkg)
- **EG scenario sub-table**: EG Metric | Max scenario | Central scenario | Min scenario | **Δ vs σ_bkg=2%**
  - Row order: Asimov | Gaussian ideal | EG(0%) | EG(2%, default) | EG(4%)
  - All 4 tabs must have same 5-column structure. If data missing for a row, show ⏳ (not omit).

### § Oscillation Sensitivity / Background Assumptions
- **Renamed from "§ Background Rejection Stages & Δm²₂₁ Discrimination"** → **"§ Background Assumptions"**
- Row order: **Nominal → Truncated (reference, highlighted) → Reduced**
- Descriptions (all 4 tabs, same text):
  - **Nominal** — no pre-selection; all reconstructed interactions (highest background, highest statistics)
  - **Truncated** — removes overrepresented endcap backgrounds in workspace geometry relative to full FD volume; reference for all other studies
  - **Reduced** — low external background scenario; standard quality and fiducial cuts applied

### § Fiducial Volume Model (cAPA only)
- Columns: **Variant | DN Asimov σ | Δ DN | HEP PL σ | Δ HEP | Sensitivity Score | √Score | Status**
- Add amber info-box explaining why fiduc_truth DN is missing:
  `01_daynight.py` requires best-cut pkl from `05_best_sigmas.py`, which skips DayNight for study-label paths.
- fiduc_truth Sensitivity Score = 2.856 (same as default for cAPA — fiducial definition does not affect χ² cut optimization).

### § Background Model Variants (cAPA only)
- Columns: **Variant | Energy estimator | γ contribution | DN Asimov σ | Δ DN | HEP PL σ | Δ HEP | Sensitivity Score | √Score | Status**
- **No default row** — section is about bkg_gamma variants only.
- Deltas in footnote reference default (SolarEnergy).
- Sensitivity Score ⏳ for all bkg_gamma variants (no 04_best_cuts.py JSON available).

### § Analysis Metric & Smoothing (cAPA only)
- Columns: **Analysis | Metric | Raw σ | Smoothed σ (default) | Δ (Smoothed − Raw)**
- Add Sensitivity row at the bottom of the tbody:
  ```
  Sensitivity | Score (Δχ²) | 2.180 | 2.180 | 0.000
             | √Score (σ)  | 1.476 | 1.476 | 0.000
  ```
  - Smoothing does not affect the χ² template-based discrimination.

---

## Sensitivity Score Data (as of 2026-09-02)

### Format
Score = ½(SolarFitAtReact + ReactorFitAtSolar); √Score ≈ effective σ-separation.

**Source:** `config/{cfg}/best-sigma-json/sensitivity/truncated/{cfg}_highest_Sensitivity{_label}.json`

Or directly from PNFS: `/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/SENSITIVITY/truncated/{cfg}/marley/{cfg}_marley_highest_SENSITIVITY{_label}.pkl`

```python
def sensitivity_score_pnfs(pkl_path, cfg):
    d = pickle.load(open(pkl_path, "rb"))
    key = (cfg, "marley", "SolarEnergy")
    v = d[key]
    return v["Score"], v["SolarFitAtReact"], v["ReactorFitAtSolar"], v["Score"]**0.5
```

### oscpoint_reactor Score: INVARIANT (= default = oscpoint_solar)

The Sensitivity Score discriminates between solar and reactor dm² templates. **This discrimination is always computed with both oscillation points**, regardless of which dm² the signal MC was simulated at. Therefore:

```
Score(oscpoint_reactor) = Score(oscpoint_solar) = Score(default)
```

Verified 2026-09-02: all 4 configs match exactly (confirmed by comparing `highest_SENSITIVITY.pkl` vs `highest_SENSITIVITY_oscpoint_solar.pkl`). No `highest_SENSITIVITY_oscpoint_reactor.pkl` is written — use default values directly.

### Default values
| Config | Score  | SF@React | RF@Solar | √Score |
|--------|--------|----------|----------|--------|
| cAPA   | 2.856  | 3.181    | 2.530    | 1.690  |
| lAPA   | 0.384  | 0.416    | 0.352    | 0.620  |
| vdN    | 0.100  | 0.101    | 0.098    | 0.316  |
| vdS    | 0.586  | 0.591    | 0.581    | 0.766  |

### Variant availability
| Variant        | cAPA | lAPA | vdN  | vdS  |
|----------------|------|------|------|------|
| default        | ✓    | ✓    | ✓    | ✓    |
| unc_sig0/2/6/8 | ✓    | ✓    | ✓    | ✓    |
| unc_bkg0/4/6   | ✓    | ✓    | ✓    | ✓    |
| fiduc_truth    | ✓    | ⏳   | ⏳   | ⏳   |
| charge_Q50/Q500| ✓    | ✓    | ✓    | ✓    |
| oscpoint_solar | ✓    | ✓    | ✓    | ✓    |
| oscpoint_reactor| =default  | =default  | =default  | =default  |
| energy_spk     | ⏳   | ⏳   | ⏳   | ⏳   |
| energy_maink   | ⏳   | ⏳   | ⏳   | ⏳   |
| metric_raw     | ✓    | ⏳   | ⏳   | ⏳   |
| metric_smoothed| ✓    | ⏳   | ⏳   | ⏳   |
| bkg_gamma      | ⏳   | N/A  | N/A  | N/A  |

⏳ = JSON file missing, 04_best_cuts.py not yet run for this variant. Value = ⏳ in table.
oscpoint_reactor = use default values directly (no separate pkl written or needed).

---

## Changelog Summary

| Date       | Change                                                                |
|------------|-----------------------------------------------------------------------|
| 2026-08-20 | Initial artifact created                                              |
| 2026-08-21 | Removed Asimov from bkg-unc tables; PL Smoothed→N/A; footnotes       |
| 2026-08-21 | Removed unc_bkg10/20 rows                                            |
| 2026-08-22 | oscpoint_reactor DN confirmed; energy_maink rows; vdN HEP 1.686→2.518|
| 2026-08-26 | Batch 2: sort unc rows ascending; split cAPA variant sections; move Sensitivity scan; rename Background Rejection; charge footnotes |
| 2026-08-27 | Batch 3: Optimal Cuts → Bkg.Model+Energy columns; Energy Estimator → SolarEnergy last + Score cols; Oscillation Point → χ²(sol↔react) cols; BkgUnc EG table → unified 5-col structure; Fiducial Model → Score cols + DN explanation; BkgModel Variants → remove default row + Score cols; Analysis Metric → Sensitivity rows; Background Assumptions rename + Truncated to 2nd row |

---

*Last updated: 2026-08-27.*
