---
marp: true
description: Inputs, workflow outputs, and per-config DayNight results
paginate: true
theme: dune
---

<!-- AUTO-GENERATED: scripts/generate_daynight_presentation.py -->

<!-- _class: titlepage -->

# DayNight Significance Workflow

---

## Introduction

This presentation summarizes the workflow and outputs of the DayNight significance analysis for the SOLAR project.
- This deck is auto-generated from workflow outputs.
- This deck is scoped to the **Truncated** folder for the **SolarEnergy** reconstruction algorithm.

Config aliases:
- hd_1x2x6_centralAPA: HD Central
- hd_1x2x6_lateralAPA: HD Lateral
- vd_1x8x14_3view_30deg_nominal: VD Top
- vd_1x8x14_3view_30deg_shielded: VD Bottom Shielded

---

### Workflow

- config: list of detector configs
- folder: **Truncated**
- analysis: DayNight
- exposure: default **30 years**
- threshold in daynight/01_daynight.py: default 8.0 MeV
- optional cuts override: nhits, ophits, adjcls
- MC threshold (`--mc_threshold`): minimum MC counts required in each essential background (gamma, neutron) per cut; prevents selecting cuts that eliminate backgrounds statistically
- best-curve reference in sensitivity/05_best_sigmas.py: **Asimov** (two-sample Poisson LLR; fastest-discovery crossing uses `AsimovSigma2`/`AsimovSigma3`)
- day-fraction (`--day_fraction`): fraction of exposure in daytime; default **0.493** (SURF latitude ~44.3°N, full-year average — not 0.5)
- day-fraction uncertainty (`--day_fraction_band`): absolute uncertainty on day fraction from run schedule; default 0.01 (1%)
- earth density band (`--earth_density_band`): fractional asymmetry spread from PREM variations (MSW matter effect); default 0.13 (±13%)
- oscillation band (`--oscillation_band`): residual uncertainty on θ₁₂, Δm²₂₁ (PDG); combined in quadrature with earth-density band; default 0.05 (±5%)

---

### Workflow Outputs

- Fiducial optimization: [config/analysis/fiducial/truncated/BestFiducials.json](../../config/analysis/fiducial/truncated/BestFiducials.json)
- Best cut summaries (JSON): [data/analysis/best-sigma-json/daynight/truncated](../../data/analysis/best-sigma-json/daynight/truncated)
- Backward-compatible local fallback: [data/analysis/daynight-json/truncated](../../data/analysis/daynight-json/truncated)
- Significance scans (PNFS outputs): [/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/DAYNIGHT/truncated](/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/DAYNIGHT/truncated)
- Figures: [output/images/analysis/day-night/truncated](../../output/images/analysis/day-night/truncated)

---

### Day-Night Discovery Statistic

Two statistics are computed in parallel by [src/physics/daynight/01_daynight.py](../../src/physics/daynight/01_daynight.py):

**Gaussian (legacy):** per-bin $Z_i = \Delta S_i / \sqrt{B_i^{eff}}$ combined as $Z = \sqrt{\sum_i Z_i^2}$, where $B_i^{eff}$ accounts for unequal day/night fractions:
$$
B_i^{eff} = 
rac{n_i^{night}}{g^2} + 
rac{n_i^{day}}{f^2}, \quad n_i^{night} = g(B_i + S_i^{night}),\; n_i^{day} = f(B_i + S_i^{day})
$$

**Asimov LLR (default):** two-sample Poisson log-likelihood ratio — see next slide.

---

### Day-Night Discovery Statistic — Asimov LLR

Under $H_0$ (common day/night rate), the MLE is the pooled rate, giving expected counts $h_i^{night} = g(n_i^{night} + n_i^{day})$, $h_i^{day} = f(n_i^{night} + n_i^{day})$.  The test statistic sums linearly over bins:
$$
q_0 = 2\sum_i \left[ n_i^{night} \ln\frac{n_i^{night}}{h_i^{night}} + n_i^{day} \ln\frac{n_i^{day}}{h_i^{day}} \right], \quad Z = \sqrt{q_0}
$$

Asymmetry uncertainty is bracketed by scaling the night signal: $S_i^{night,k} = S_i^{day} + k(S_i^{night} - S_i^{day})$ with $k \in \{1 \pm \sigma_{tot}\}$, $\sigma_{tot} = \sqrt{\sigma_{earth}^2 + \sigma_{osc}^2}$.

---

### Day-Night Discovery Statistic Details

- Both Gaussian and Asimov curves are stored per cut; **Asimov is the default** for best-cut selection in [src/physics/sensitivity/05_best_sigmas.py](../../src/physics/sensitivity/05_best_sigmas.py) and exposure plots.
- σ2/σ3 crossing exposures tracked independently: `Sigma2`/`Sigma3` (Gaussian crossings), `AsimovSigma2`/`AsimovSigma3` (Asimov crossings); fastest-discovery selection uses Asimov columns.
- MC threshold gate: cuts where any essential background (gamma, neutron) has fewer than `--mc_threshold` MC events are skipped; prevents selecting cuts that deplete backgrounds statistically.
- Smoothing (Gaussian kernel) is applied to signal and background histograms before significance evaluation. **Smoothing is safe for the Gaussian statistic**: the per-bin formula `Z_i = ΔS_i / sqrt(B_i^eff)` does not involve a log-ratio, so near-zero smoothed background gives `Z_i ≈ 0` (not log-amplification). This is in contrast to the HEP profile likelihood, where smoothed background rates must not be used.
- Raw (unsmoothed) variants are stored alongside smoothed curves for diagnostic comparison (`RawGaussian`, `RawAsimov` columns).

---

### Asymmetry Uncertainty Bands

Two independent sources of uncertainty on the predicted day-night asymmetry amplitude are combined in quadrature:

| Source | Parameter | Default | Physical origin |
|---|---|---:|---|
| Earth density | `--earth_density_band` | 0.13 (±13%) | PREM profile variations; MSW matter effect spread |
| Oscillation params | `--oscillation_band` | 0.05 (±5%) | θ₁₂, Δm²₂₁ (PDG uncertainties) |

$$\sigma_\mathrm{tot} = \sqrt{\sigma_\mathrm{earth}^2 + \sigma_\mathrm{osc}^2} \approx 0.139$$

Three asymmetry scales bracket the full predicted range:
$$k \in \{1+\sigma_\mathrm{tot},\; 1.0,\; 1-\sigma_\mathrm{tot}\} \;\Rightarrow\; S_i^{night,k} = S_i^{day} + k(S_i^{night} - S_i^{day})$$

Upper band = stronger matter effect; lower = weaker. Both bands collapse to the nominal curve when the asymmetry signal is zero.

---

### Background Uncertainty Model

When `background_error: true` is set in the workflow config, three uncertainty sources per time period are combined in quadrature before entering `evaluate_significance`:

$$
\sigma_{night} = \sqrt{n_{bkg}^{night} + (\sigma_B \cdot n_{bkg}^{night})^2 + (f_{day}^\Delta \cdot f \cdot B_{total})^2}
$$

where $\sigma_B$ = `--background_uncertainty` (default 2%), $f_{day}^\Delta$ = `--day_fraction_band` (default 1%), $f$ = exposure factor.

1. **Poisson statistical**: $\sqrt{n_{bkg}^{period}}$
2. **Normalization systematic**: $\sigma_B \times n_{bkg}^{period}$
3. **Day-fraction uncertainty**: $f_{day}^\Delta \times \mathrm{factor} \times B_{total}$ — propagates run-schedule uncertainty on the day/night exposure split

Effective combined uncertainty for the significance formula:
$$\sigma_{eff} = \sqrt{\left(\frac{\sigma_{night}}{g}\right)^2 + \left(\frac{\sigma_{day}}{f}\right)^2}$$

Stored in `ErrorGaussian` columns. Nominal `Gaussian` columns omit the normalization/day-fraction terms.

---

### PKL Output Columns

Per cut (NHits, OpHits, AdjCl) × energy × config/name, `{energy}_DayNight_Results.pkl` stores:

| Column group | Description |
|---|---|
| `Gaussian` / `Gaussian±Error` | Smoothed Gaussian significance ± asymmetry band |
| `RawGaussian` / `RawGaussian±Error` | Unsmoothed Gaussian significance ± asymmetry band |
| `ErrorGaussian` / `ErrorGaussian±Error` | Gaussian with background uncertainty (Poisson + norm + day-frac) |
| `RawErrorGaussian` / `RawErrorGaussian±Error` | Raw variant of above |
| `Asimov` / `Asimov±Error` | Smoothed two-sample Poisson LLR significance ± asymmetry band |
| `RawAsimov` / `RawAsimov±Error` | Unsmoothed Asimov significance ± asymmetry band |
| `Sigma2`, `Sigma3` | Exposure (kt·yr) at Gaussian 2σ/3σ crossing |
| `AsimovSigma2`, `AsimovSigma3` | Exposure (kt·yr) at Asimov 2σ/3σ crossing |
| `EarthDensityBand`, `OscillationBand`, `TotalAsymmetryBand` | Uncertainty inputs |
| `DayFraction`, `DayFractionBand`, `BackgroundUncertainty` | Exposure model params |

---

### Context: Super-Kamiokande Day-Night Analysis

Super-K measures the solar day-night effect with an energy-spectral chi-squared [[Abe et al., PRD 94, 052010 (2016)](https://doi.org/10.1103/PhysRevD.94.052010); [Renshaw et al., PRL 112, 091805 (2014)](https://doi.org/10.1103/PhysRevLett.112.091805)]:
$$
\chi^2_{SK} = \sum_{k\in\{D,N\}} \sum_j \frac{(N_{kj}-\mu_{kj})^2}{\sigma_{kj}^2} + \text{(systematic penalties)}
$$
In the statistical-only limit, DUNE's $Z_{global}^2$ is equivalent:
$$
Z_{global}^2 = \sum_i \frac{(\Delta S_i)^2}{B_i} \equiv \chi^2_{DN}\bigg|_{\sigma_i=\sqrt{B_i}}
$$

---

### Similarities and Differences vs. Super-K

**Shared structure:**
- Energy-binned counting; day signal enters null hypothesis as background
- MSW Earth matter effect is the physical driver of the night excess

**DUNE vs. Super-K differences:**
- No systematic nuisance penalty terms in DUNE baseline; second curve folds in background uncertainty
- DUNE projects future discovery exposure; Super-K measures $A_{DN} = 2(\Phi_N - \Phi_D)/(\Phi_N + \Phi_D)$ from existing data
- Energy binning only; Super-K also sub-bins by solar zenith angle for additional sensitivity

---

### Histogram Smoothing Math I

- Linear smoothing model used per histogram bin:
$$
\tilde{h}_i = \sum_j K_{ij} h_j
$$

- Integral-preserving normalization applied after smoothing:
$$
\tilde{h}_i \leftarrow \tilde{h}_i \cdot \frac{\sum_j h_j}{\sum_j \tilde{h}_j}
$$

---

### Histogram Smoothing Math II

- Threshold-slice smoothing used for DayNight threshold region:
$$
h^{\mathrm{out}}_i =
\begin{cases}
h_i, & i < i_{\mathrm{thr}} \\
\tilde{h}_i, & i \ge i_{\mathrm{thr}}
\end{cases}
$$

- Variance propagation through the same linear operator:
$$
v^{\mathrm{out}} = (K \odot K)\,v, \qquad \sigma^{\mathrm{out}}_i = \sqrt{v^{\mathrm{out}}_i}
$$

---

## Fiducialization

---

### HD Central

<div class="two-col">
  <div>
<p><strong>No Fiducial</strong></p>
<img src="../../output/images/solar/fiducial/hd_1x2x6_centralAPA/marley/truncated/hd_1x2x6_centralAPA_marley_SolarEnergy_DAYNIGHT_NoFiducial_Significance.png">
  </div>
  <div>
<p><strong>Best Fiducial</strong></p>
<img src="../../output/images/solar/fiducial/hd_1x2x6_centralAPA/marley/truncated/hd_1x2x6_centralAPA_marley_SolarEnergy_DAYNIGHT_BestFiducial_Significance.png">
  </div>
</div>

---

### HD Lateral

<div class="two-col">
  <div>
<p><strong>No Fiducial</strong></p>
<img src="../../output/images/solar/fiducial/hd_1x2x6_lateralAPA/marley/truncated/hd_1x2x6_lateralAPA_marley_SolarEnergy_DAYNIGHT_NoFiducial_Significance.png">
  </div>
  <div>
<p><strong>Best Fiducial</strong></p>
<img src="../../output/images/solar/fiducial/hd_1x2x6_lateralAPA/marley/truncated/hd_1x2x6_lateralAPA_marley_SolarEnergy_DAYNIGHT_BestFiducial_Significance.png">
  </div>
</div>

---

### VD Top

<div class="two-col">
  <div>
<p><strong>No Fiducial</strong></p>
<img src="../../output/images/solar/fiducial/vd_1x8x14_3view_30deg_nominal/marley/truncated/vd_1x8x14_3view_30deg_nominal_marley_SolarEnergy_DAYNIGHT_NoFiducial_Significance.png">
  </div>
  <div>
<p><strong>Best Fiducial</strong></p>
<img src="../../output/images/solar/fiducial/vd_1x8x14_3view_30deg_nominal/marley/truncated/vd_1x8x14_3view_30deg_nominal_marley_SolarEnergy_DAYNIGHT_BestFiducial_Significance.png">
  </div>
</div>

---

### VD Bottom Shielded

<div class="two-col">
  <div>
<p><strong>No Fiducial</strong></p>
<img src="../../output/images/solar/fiducial/vd_1x8x14_3view_30deg_shielded/marley/truncated/vd_1x8x14_3view_30deg_shielded_marley_SolarEnergy_DAYNIGHT_NoFiducial_Significance.png">
  </div>
  <div>
<p><strong>Best Fiducial</strong></p>
<img src="../../output/images/solar/fiducial/vd_1x8x14_3view_30deg_shielded/marley/truncated/vd_1x8x14_3view_30deg_shielded_marley_SolarEnergy_DAYNIGHT_BestFiducial_Significance.png">
  </div>
</div>

---

### Fiducial Optimization Summary

| Config | Fiducial X | Fiducial Y | Fiducial Z | Before Fiducialization | After Fiducialization | Fiducial Mass (kt) |
|---|---:|---:|---:|---:|---:|---:|
| HD Central | 20 | 100 | 320 | 0.418 | 3.721 | 2.89 |
| HD Lateral | 80 | 140 | 200 | 0.091 | 3.293 | 2.88 |
| VD Top | 60 | 140 | 0 | 0.000 | 0.034 | 5.65 |
| VD Bottom Shielded | 20 | 120 | 80 | 0.022 | 0.036 | 5.77 |

---

## DayNight Results

---

### HD Central

<div class="two-col">
  <div>
<p><strong>Significance</strong></p>
<img src="../../output/images/analysis/day-night/hd_1x2x6_centralAPA/marley/truncated/hd_1x2x6_centralAPA_marley_SolarEnergy_DayNight_Significance_Exposure_30.png">
  </div>
  <div>
<p><strong>Exposure</strong></p>
<img src="../../output/images/analysis/day-night/hd_1x2x6_centralAPA/marley/truncated/hd_1x2x6_centralAPA_marley_SolarEnergy_DayNight_Exposure_Threshold_0.png">
  </div>
</div>

---

### HD Lateral

<div class="two-col">
  <div>
<p><strong>Significance</strong></p>
<img src="../../output/images/analysis/day-night/hd_1x2x6_lateralAPA/marley/truncated/hd_1x2x6_lateralAPA_marley_SolarEnergy_DayNight_Significance_Exposure_30.png">
  </div>
  <div>
<p><strong>Exposure</strong></p>
<img src="../../output/images/analysis/day-night/hd_1x2x6_lateralAPA/marley/truncated/hd_1x2x6_lateralAPA_marley_SolarEnergy_DayNight_Exposure_Threshold_0.png">
  </div>
</div>

---

### VD Top

<div class="two-col">
  <div>
<p><strong>Significance</strong></p>
<img src="../../output/images/analysis/day-night/vd_1x8x14_3view_30deg_nominal/marley/truncated/vd_1x8x14_3view_30deg_nominal_marley_SolarEnergy_DayNight_Significance_Exposure_30.png">
  </div>
  <div>
<p><strong>Exposure</strong></p>
<img src="../../output/images/analysis/day-night/vd_1x8x14_3view_30deg_nominal/marley/truncated/vd_1x8x14_3view_30deg_nominal_marley_SolarEnergy_DayNight_Exposure_Threshold_0.png">
  </div>
</div>

---

### VD Bottom Shielded

<div class="two-col">
  <div>
<p><strong>Significance</strong></p>
<img src="../../output/images/analysis/day-night/vd_1x8x14_3view_30deg_shielded/marley/truncated/vd_1x8x14_3view_30deg_shielded_marley_SolarEnergy_DayNight_Significance_Exposure_30.png">
  </div>
  <div>
<p><strong>Exposure</strong></p>
<img src="../../output/images/analysis/day-night/vd_1x8x14_3view_30deg_shielded/marley/truncated/vd_1x8x14_3view_30deg_shielded_marley_SolarEnergy_DayNight_Exposure_Threshold_0.png">
  </div>
</div>

---

### Best DayNight Cuts by Config

| Config | NHits | OpHits | AdjCl | Significance |
|---|---:|---:|---:|---:|
| HD Central | 3 | 12 | 3 | 4.438 |
| HD Lateral | 4 | 4 | 4 | 1.616 |
| VD Top | 6 | 19 | 6 | 0.874 |
| VD Bottom Shielded | 6 | 13 | 9 | 1.820 |

---

## Oscillograms

---

### HD Central

<div class="two-col">
  <div>
<p><strong>P(ν<sub>e</sub>→ν<sub>e</sub>) heatmap</strong></p>
<img src="../../output/images/analysis/day-night/oscillogram/hd_1x2x6_centralAPA/marley/truncated/hd_1x2x6_centralAPA_marley_Oscillogram_SolarEnergy.png">
  </div>
  <div>
<p><strong>Nadir projection</strong></p>
<img src="../../output/images/analysis/day-night/oscillogram/hd_1x2x6_centralAPA/marley/truncated/hd_1x2x6_centralAPA_marley_Oscillogram_NadirProjection_SolarEnergy.png">
  </div>
</div>

---

### HD Central (cont.)

<div class="two-col">
  <div>
<p><strong>Nadir-weighted oscillogram</strong></p>
<img src="../../output/images/analysis/day-night/oscillogram/hd_1x2x6_centralAPA/marley/truncated/hd_1x2x6_centralAPA_marley_Oscillogram_NadirWeighted_SolarEnergy.png">
  </div>
  <div>
<p><strong>1D fiducial signal spectrum</strong></p>
<img src="../../output/images/analysis/day-night/oscillogram/hd_1x2x6_centralAPA/marley/truncated/hd_1x2x6_centralAPA_marley_Signal1D_SolarEnergy_FidOnly.png">
  </div>
</div>

---

### HD Lateral

<div class="two-col">
  <div>
<p><strong>P(ν<sub>e</sub>→ν<sub>e</sub>) heatmap</strong></p>
<img src="../../output/images/analysis/day-night/oscillogram/hd_1x2x6_lateralAPA/marley/truncated/hd_1x2x6_lateralAPA_marley_Oscillogram_SolarEnergy.png">
  </div>
  <div>
<p><strong>Nadir projection</strong></p>
<img src="../../output/images/analysis/day-night/oscillogram/hd_1x2x6_lateralAPA/marley/truncated/hd_1x2x6_lateralAPA_marley_Oscillogram_NadirProjection_SolarEnergy.png">
  </div>
</div>

---

### HD Lateral (cont.)

<div class="two-col">
  <div>
<p><strong>Nadir-weighted oscillogram</strong></p>
<img src="../../output/images/analysis/day-night/oscillogram/hd_1x2x6_lateralAPA/marley/truncated/hd_1x2x6_lateralAPA_marley_Oscillogram_NadirWeighted_SolarEnergy.png">
  </div>
  <div>
<p><strong>1D fiducial signal spectrum</strong></p>
<img src="../../output/images/analysis/day-night/oscillogram/hd_1x2x6_lateralAPA/marley/truncated/hd_1x2x6_lateralAPA_marley_Signal1D_SolarEnergy_FidOnly.png">
  </div>
</div>

---

### VD Top

<div class="two-col">
  <div>
<p><strong>P(ν<sub>e</sub>→ν<sub>e</sub>) heatmap</strong></p>
<img src="../../output/images/analysis/day-night/oscillogram/vd_1x8x14_3view_30deg_nominal/marley/truncated/vd_1x8x14_3view_30deg_nominal_marley_Oscillogram_SolarEnergy.png">
  </div>
  <div>
<p><strong>Nadir projection</strong></p>
<img src="../../output/images/analysis/day-night/oscillogram/vd_1x8x14_3view_30deg_nominal/marley/truncated/vd_1x8x14_3view_30deg_nominal_marley_Oscillogram_NadirProjection_SolarEnergy.png">
  </div>
</div>

---

### VD Top (cont.)

<div class="two-col">
  <div>
<p><strong>Nadir-weighted oscillogram</strong></p>
<img src="../../output/images/analysis/day-night/oscillogram/vd_1x8x14_3view_30deg_nominal/marley/truncated/vd_1x8x14_3view_30deg_nominal_marley_Oscillogram_NadirWeighted_SolarEnergy.png">
  </div>
  <div>
<p><strong>1D fiducial signal spectrum</strong></p>
<img src="../../output/images/analysis/day-night/oscillogram/vd_1x8x14_3view_30deg_nominal/marley/truncated/vd_1x8x14_3view_30deg_nominal_marley_Signal1D_SolarEnergy_FidOnly.png">
  </div>
</div>

---

### VD Bottom Shielded

<div class="two-col">
  <div>
<p><strong>P(ν<sub>e</sub>→ν<sub>e</sub>) heatmap</strong></p>
<img src="../../output/images/analysis/day-night/oscillogram/vd_1x8x14_3view_30deg_shielded/marley/truncated/vd_1x8x14_3view_30deg_shielded_marley_Oscillogram_SolarEnergy.png">
  </div>
  <div>
<p><strong>Nadir projection</strong></p>
<img src="../../output/images/analysis/day-night/oscillogram/vd_1x8x14_3view_30deg_shielded/marley/truncated/vd_1x8x14_3view_30deg_shielded_marley_Oscillogram_NadirProjection_SolarEnergy.png">
  </div>
</div>

---

### VD Bottom Shielded (cont.)

<div class="two-col">
  <div>
<p><strong>Nadir-weighted oscillogram</strong></p>
<img src="../../output/images/analysis/day-night/oscillogram/vd_1x8x14_3view_30deg_shielded/marley/truncated/vd_1x8x14_3view_30deg_shielded_marley_Oscillogram_NadirWeighted_SolarEnergy.png">
  </div>
  <div>
<p><strong>1D fiducial signal spectrum</strong></p>
<img src="../../output/images/analysis/day-night/oscillogram/vd_1x8x14_3view_30deg_shielded/marley/truncated/vd_1x8x14_3view_30deg_shielded_marley_Signal1D_SolarEnergy_FidOnly.png">
  </div>
</div>

---



---

## Coverage and Notes

- Config coverage in best-cut JSON outputs:
  - nominal: 5
  - reduced: 4
  - truncated: 4
- Table values are read from workflow-generated JSON at generation time.
- Re-run script to refresh this folder after each workflow run:
- /usr/bin/python3 scripts/generate_daynight_presentation.py --folder truncated
