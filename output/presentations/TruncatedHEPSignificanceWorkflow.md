---
marp: true
math: katex
description: Inputs, workflow outputs, and per-config HEP results
paginate: true
theme: dune
---

<!-- AUTO-GENERATED: scripts/generate_hep_presentation.py -->

<!-- _class: titlepage -->

# HEP Significance Workflow

---

## Introduction

This presentation summarizes the workflow and outputs of the HEP significance analysis for the SOLAR project.
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
- analysis: HEP
- exposure: default **30 years**
- threshold in hep/01_hep.py: from [config/analysis/config.json](../../config/analysis/config.json) HEP -> THRESHOLDS -> (no threshold config found)
- optional cuts override: nhits, ophits, adjcls
- significance reference in plots: ProfileLikelihood
- best-cut selection in sensitivity/05_best_sigmas.py: **ProfileLikelihood** (PAVA-monotone, 3σ crossing)

---

### Workflow Skip Flags

Used for [src/pipelines/run_sensitivity.py](../../src/pipelines/run_sensitivity.py):
- `--no-computation`: skip all analysis, run plot macros only
- `--no-significance`: skip 01_hep.py/01_daynight.py/06_significance.py only
- `--no-fiducialization`: skip signal/01_fiducialize.py only
- `--no-rebin`: skip signal/03_analysis.py rebinning step only

---

### Workflow Outputs

- Fiducial optimization: [config/analysis/fiducial/truncated/BestFiducials.json](../../config/analysis/fiducial/truncated/BestFiducials.json)
- Best cut summaries (JSON): [data/analysis/hep-json/truncated](../../data/analysis/hep-json/truncated)
- Significance scans (PNFS outputs): [/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/HEP/truncated](/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/HEP/truncated)
- Figures: [output/images/analysis/hep/truncated](../../output/images/analysis/hep/truncated)

---

### Histogram and Significance Flow I: Building, Smoothing, and Evaluation

- Step 1: Build HEP rates and threshold region in [src/physics/hep/01_hep.py](../../src/physics/hep/01_hep.py) per component.
- Step 2: Apply component-aware smoothing via [lib/smoothing.py](../../lib/smoothing.py) using HEP smoothing config.
- Step 3: Evaluate Gaussian, Asimov, and ProfileLikelihood significance curves in [src/physics/hep/01_hep.py](../../src/physics/hep/01_hep.py) for **all** analysis cuts. ProfileLikelihood uses a single global background normalization nuisance profiled jointly across all bins (see *Background Normalization Model* slide). Background bins with fewer than `min_mc_per_bin` raw MC events are masked using the [Barlow-Beeston lite criterion](https://www.sciencedirect.com/science/article/pii/009350659390005W) (as implemented in [ROOT HistFactory](https://root.cern.ch/doc/master/classRooStats_1_1HistFactory_1_1Measurement.html)) to suppress LLR divergence from empty bins. **Only raw histogram background rates** enter the PL — Gaussian-smoothed rates are for visual display only. Using smoothed rates produces near-zero per-bin denominators in signal-region bins (kernel tail leakage), inflating PL as `signal × log(signal/ε)` — a bias proportional to signal strength that is strongest for dense detector configs.

---

### Histogram and Significance Flow II: Post-Processing and Plotting

- Step 4: Select the best cut by ProfileLikelihood in [src/physics/sensitivity/05_best_sigmas.py](../../src/physics/sensitivity/05_best_sigmas.py). Cuts whose PL curve contains a single-step jump exceeding `max_pl_jump` σ in `PreIsotonicProfileLikelihood` (pre-PAVA values, available when `pl_isotonic: true`) are flagged as spiked, excluded from the main `highest` selection, and saved separately as `highest_spiked` for inspection. Falls back to post-PAVA `ProfileLikelihood` if `PreIsotonicProfileLikelihood` is absent.
- Step 5: Render exposure/significance and comparison plots in [src/physics/hep/exposure_plot.py](../../src/physics/hep/exposure_plot.py), [src/physics/hep/significance_plot.py](../../src/physics/hep/significance_plot.py), [src/physics/hep/significance_comparison.py](../../src/physics/hep/significance_comparison.py), and [src/physics/hep/exposure_comparison.py](../../src/physics/hep/exposure_comparison.py). With `--all_metrics`: [src/physics/hep/rebin_comparison.py](../../src/physics/hep/rebin_comparison.py) renders Pre-PAVA vs Post-PAVA PL curves at the best cut for direct comparison.

---

### Background Normalization Model

The background systematic is a **single global scale factor** β — not independent per-bin nuisances.
Significance is computed from $q_0 = -2\ln\lambda(\mu=0)$ following [Cowan et al. 2010 (arXiv:1007.1727)](https://arxiv.org/abs/1007.1727) — the standard ATLAS/CMS formulation for discovery tests.

β satisfies the closed-form quadratic (total counts $N = \sum n_i$, $B = \sum b_i$, $\sigma^2 = \sigma_\mathrm{rel}^2$):
$$
\hat{\beta}^2 + (B\sigma^2 - 1)\hat{\beta} - N\sigma^2 = 0
$$

**Why not per-bin nuisances?** With $k$ independent per-bin β's each with a 2% Gaussian constraint, the null hypothesis has $k$ dials to absorb signal bin-by-bin. This creates an artificial flat region in significance vs exposure that ends in a sharp kink when the absorbing capacity is exhausted (at $f \sim 1/(b_r \cdot \sigma^2)$ per bin — typically around 20 years for shielded configs with tight OpHits cuts). A global β has its absorbing regime end at $f \sim 1/(B_\mathrm{total} \cdot \sigma^2)$ (typically sub-year), giving a smooth, physically correct PL curve throughout the analysis window.

---

### ProfileLikelihood Monotonicity (PAVA)

When `pl_isotonic: true` is set in the workflow config, the raw per-cut PL significance array is post-processed with [`sklearn.isotonic.IsotonicRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html) (PAVA) to enforce strict monotonicity. PAVA finds the non-decreasing sequence with minimum L2 distance from the raw values, ensuring accumulated exposure cannot reduce significance.

The pre-PAVA values are saved as `PreIsotonicProfileLikelihood` in the output pkl. This column is what the spike detector in `05_best_sigmas.py` reads — abrupt numerical jumps that PAVA would otherwise flatten are still caught before they contaminate the best-cut selection.

Note: **Gaussian kernel smoothing is not applied to PL curves.** Smoothing is reserved for visual display of background histograms. Applying it to the per-bin background rates that enter the likelihood ratio produces near-zero denominators in signal-region bins via kernel tail leakage, inflating PL significance as `signal × log(signal / ε)` — a signal-proportional bias that is most severe for dense detector configs with many signal-region bins.

---

### ProfileLikelihood Error Bands

The ±1σ bands on PL exposure curves use **signal normalization variation**: signal events are scaled by $(1 \pm \sigma_s)$ where $\sigma_s$ is the signal reconstruction efficiency systematic (`--signal_uncertainty`, typically 10%):
$$
s_i^{\pm} = s_i \cdot (1 \pm \sigma_s)
$$

The background is **never shifted**, so the profiled nuisance $\hat{\beta}$ is unaffected. Both bands collapse symmetrically when signal is negligible ($\pm\sigma_s \cdot 0 = 0$). This avoids the asymmetric-collapse artifact of background-shifting approaches, where $\hat{\beta}_{null}$ pull contributions drive the upper band non-zero independent of signal strength.

- **Upper band** ($\delta = +1$): more signal → higher $q_0$ → easier discovery.
- **Lower band** ($\delta = -1$): less signal → lower $q_0$. Both bands collapse symmetrically for configurations where signal is negligible.

---

### Smoothing by Stage (HEP)

| Stage | Enabled | Method | Component Mode | Smoothed Components | Sigma |
|---|---|---|---|---|---|
| *(no HEP smoothing stage config found)* | - | - | - | - | - |

---

## Fiducialization

---

### HD Central

<div class="two-col">
  <div>
<p><strong>No Fiducial</strong></p>
<img src="../../output/images/solar/fiducial/hd_1x2x6_centralAPA/marley/truncated/hd_1x2x6_centralAPA_marley_SolarEnergy_HEP_NoFiducial_Significance.png">
  </div>
  <div>
<p><strong>Best Fiducial</strong></p>
<img src="../../output/images/solar/fiducial/hd_1x2x6_centralAPA/marley/truncated/hd_1x2x6_centralAPA_marley_SolarEnergy_HEP_BestFiducial_Significance.png">
  </div>
</div>

---

### HD Lateral

<div class="two-col">
  <div>
<p><strong>No Fiducial</strong></p>
<img src="../../output/images/solar/fiducial/hd_1x2x6_lateralAPA/marley/truncated/hd_1x2x6_lateralAPA_marley_SolarEnergy_HEP_NoFiducial_Significance.png">
  </div>
  <div>
<p><strong>Best Fiducial</strong></p>
<img src="../../output/images/solar/fiducial/hd_1x2x6_lateralAPA/marley/truncated/hd_1x2x6_lateralAPA_marley_SolarEnergy_HEP_BestFiducial_Significance.png">
  </div>
</div>

---

### VD Top

<div class="two-col">
  <div>
<p><strong>No Fiducial</strong></p>
<img src="../../output/images/solar/fiducial/vd_1x8x14_3view_30deg_nominal/marley/truncated/vd_1x8x14_3view_30deg_nominal_marley_SolarEnergy_HEP_NoFiducial_Significance.png">
  </div>
  <div>
<p><strong>Best Fiducial</strong></p>
<img src="../../output/images/solar/fiducial/vd_1x8x14_3view_30deg_nominal/marley/truncated/vd_1x8x14_3view_30deg_nominal_marley_SolarEnergy_HEP_BestFiducial_Significance.png">
  </div>
</div>

---

### VD Bottom Shielded

<div class="two-col">
  <div>
<p><strong>No Fiducial</strong></p>
<img src="../../output/images/solar/fiducial/vd_1x8x14_3view_30deg_shielded/marley/truncated/vd_1x8x14_3view_30deg_shielded_marley_SolarEnergy_HEP_NoFiducial_Significance.png">
  </div>
  <div>
<p><strong>Best Fiducial</strong></p>
<img src="../../output/images/solar/fiducial/vd_1x8x14_3view_30deg_shielded/marley/truncated/vd_1x8x14_3view_30deg_shielded_marley_SolarEnergy_HEP_BestFiducial_Significance.png">
  </div>
</div>

---

### Fiducial Optimization Summary

| Config | Fiducial X | Fiducial Y | Fiducial Z | Before Fiducialization | After Fiducialization | Fiducial Mass (kt) |
|---|---:|---:|---:|---:|---:|---:|
| HD Central | 0 | 80 | 0 | 0.303 | 1.226 | 5.85 |
| HD Lateral | 60 | 80 | 20 | 0.095 | 0.817 | 4.74 |
| VD Top | 0 | 0 | 40 | 0.551 | 0.575 | 7.54 |
| VD Bottom Shielded | 0 | 0 | 20 | 0.782 | 0.782 | 7.69 |

---

## HEP Results

---

### HD Central

<div class="two-col">
  <div>
<p><strong>Significance</strong></p>
<p>Significance plot not available.</p>
  </div>
  <div>
<p><strong>Exposure</strong></p>
<img src="../../output/images/analysis/hep/hd_1x2x6_centralAPA/marley/truncated/hd_1x2x6_centralAPA_marley_SolarEnergy_HEP_Exposure_ProfileLikelihood_Threshold_0.png">
  </div>
</div>

<div class="comparison-note">
  <strong>Lower subplot guide:</strong> Lower panel note: not available (significance plot missing).
</div>

---

### HD Lateral

<div class="two-col">
  <div>
<p><strong>Significance</strong></p>
<p>Significance plot not available.</p>
  </div>
  <div>
<p><strong>Exposure</strong></p>
<img src="../../output/images/analysis/hep/hd_1x2x6_lateralAPA/marley/truncated/hd_1x2x6_lateralAPA_marley_SolarEnergy_HEP_Exposure_ProfileLikelihood_Threshold_0.png">
  </div>
</div>

<div class="comparison-note">
  <strong>Lower subplot guide:</strong> Lower panel note: not available (significance plot missing).
</div>

---

### VD Top

<div class="two-col">
  <div>
<p><strong>Significance</strong></p>
<p>Significance plot not available.</p>
  </div>
  <div>
<p><strong>Exposure</strong></p>
<img src="../../output/images/analysis/hep/vd_1x8x14_3view_30deg_nominal/marley/truncated/vd_1x8x14_3view_30deg_nominal_marley_SolarEnergy_HEP_Exposure_ProfileLikelihood_Threshold_0.png">
  </div>
</div>

<div class="comparison-note">
  <strong>Lower subplot guide:</strong> Lower panel note: not available (significance plot missing).
</div>

---

### VD Bottom Shielded

<div class="two-col">
  <div>
<p><strong>Significance</strong></p>
<p>Significance plot not available.</p>
  </div>
  <div>
<p><strong>Exposure</strong></p>
<img src="../../output/images/analysis/hep/vd_1x8x14_3view_30deg_shielded/marley/truncated/vd_1x8x14_3view_30deg_shielded_marley_SolarEnergy_HEP_Exposure_ProfileLikelihood_Threshold_0.png">
  </div>
</div>

<div class="comparison-note">
  <strong>Lower subplot guide:</strong> Lower panel note: not available (significance plot missing).
</div>

---

## Reference Comparison

---

### HD Central

<div class="two-col">
  <div>
<p><strong>Significance Reference Comparison</strong></p>
<img src="../../output/images/analysis/hep/hd_1x2x6_centralAPA/marley/truncated/hd_1x2x6_centralAPA_marley_SolarEnergy_HEP_Significance_Comparison_Exposure_30_Threshold_0.png">
  </div>
  <div>
<p><strong>Exposure Reference Comparison</strong></p>
<img src="../../output/images/analysis/hep/hd_1x2x6_centralAPA/marley/truncated/hd_1x2x6_centralAPA_marley_SolarEnergy_HEP_Exposure_Comparison_Threshold_0.png">
  </div>
</div>

---

### HD Lateral

<div class="two-col">
  <div>
<p><strong>Significance Reference Comparison</strong></p>
<img src="../../output/images/analysis/hep/hd_1x2x6_lateralAPA/marley/truncated/hd_1x2x6_lateralAPA_marley_SolarEnergy_HEP_Significance_Comparison_Exposure_30_Threshold_0.png">
  </div>
  <div>
<p><strong>Exposure Reference Comparison</strong></p>
<img src="../../output/images/analysis/hep/hd_1x2x6_lateralAPA/marley/truncated/hd_1x2x6_lateralAPA_marley_SolarEnergy_HEP_Exposure_Comparison_Threshold_0.png">
  </div>
</div>

---

### VD Top

No matching reference-comparison pair found for VD Top

---

### VD Bottom Shielded

No matching reference-comparison pair found for VD Bottom Shielded

---

## Adaptive Rebin Comparison

---

### HD Central

<div class="three-col">
  <div>
  <p><strong>Gaussian</strong></p>
  <img src="../../output/images/analysis/hep/hd_1x2x6_centralAPA/marley/truncated/hd_1x2x6_centralAPA_marley_SolarEnergy_HEP_Gaussian_AdaptiveRebin_Comparison_Threshold_0.png">
</div>
  <div>
  <p><strong>Asimov</strong></p>
  <img src="../../output/images/analysis/hep/hd_1x2x6_centralAPA/marley/truncated/hd_1x2x6_centralAPA_marley_SolarEnergy_HEP_Asimov_AdaptiveRebin_Comparison_Threshold_0.png">
</div>
</div>

<div class="comparison-note">
  <strong>How to compare:</strong> Left = Asimov, middle = Gaussian, right = ProfileLikelihood.
  Compare the exposure turn-on point near threshold 0 MeV, relative ordering between methods, and curve smoothness/step behavior after adaptive rebinning.
</div>

---

### HD Lateral

<div class="three-col">
  <div>
  <p><strong>Gaussian</strong></p>
  <img src="../../output/images/analysis/hep/hd_1x2x6_lateralAPA/marley/truncated/hd_1x2x6_lateralAPA_marley_SolarEnergy_HEP_Gaussian_AdaptiveRebin_Comparison_Threshold_0.png">
</div>
  <div>
  <p><strong>Asimov</strong></p>
  <img src="../../output/images/analysis/hep/hd_1x2x6_lateralAPA/marley/truncated/hd_1x2x6_lateralAPA_marley_SolarEnergy_HEP_Asimov_AdaptiveRebin_Comparison_Threshold_0.png">
</div>
</div>

<div class="comparison-note">
  <strong>How to compare:</strong> Left = Asimov, middle = Gaussian, right = ProfileLikelihood.
  Compare the exposure turn-on point near threshold 0 MeV, relative ordering between methods, and curve smoothness/step behavior after adaptive rebinning.
</div>

---

### VD Top

<div class="three-col">
  <div>
  <p><strong>Gaussian</strong></p>
  <img src="../../output/images/analysis/hep/vd_1x8x14_3view_30deg_nominal/marley/truncated/vd_1x8x14_3view_30deg_nominal_marley_SolarEnergy_HEP_Gaussian_AdaptiveRebin_Comparison_Threshold_0.png">
</div>
  <div>
  <p><strong>Asimov</strong></p>
  <img src="../../output/images/analysis/hep/vd_1x8x14_3view_30deg_nominal/marley/truncated/vd_1x8x14_3view_30deg_nominal_marley_SolarEnergy_HEP_Asimov_AdaptiveRebin_Comparison_Threshold_0.png">
</div>
</div>

<div class="comparison-note">
  <strong>How to compare:</strong> Left = Asimov, middle = Gaussian, right = ProfileLikelihood.
  Compare the exposure turn-on point near threshold 0 MeV, relative ordering between methods, and curve smoothness/step behavior after adaptive rebinning.
</div>

---

### VD Bottom Shielded

<div class="three-col">
  <div>
  <p><strong>Gaussian</strong></p>
  <img src="../../output/images/analysis/hep/vd_1x8x14_3view_30deg_shielded/marley/truncated/vd_1x8x14_3view_30deg_shielded_marley_SolarEnergy_HEP_Gaussian_AdaptiveRebin_Comparison_Threshold_0.png">
</div>
  <div>
  <p><strong>Asimov</strong></p>
  <img src="../../output/images/analysis/hep/vd_1x8x14_3view_30deg_shielded/marley/truncated/vd_1x8x14_3view_30deg_shielded_marley_SolarEnergy_HEP_Asimov_AdaptiveRebin_Comparison_Threshold_0.png">
</div>
</div>

<div class="comparison-note">
  <strong>How to compare:</strong> Left = Asimov, middle = Gaussian, right = ProfileLikelihood.
  Compare the exposure turn-on point near threshold 0 MeV, relative ordering between methods, and curve smoothness/step behavior after adaptive rebinning.
</div>

---

### Best HEP Cuts by Config

| Config | NHits | OpHits | AdjCl | Significance |
|---|---:|---:|---:|---:|
| HD Central | 4 | 13 | 5 | 12.182 |
| HD Lateral | 3 | 4 | 9 | 6.263 |
| VD Top | 10 | 11 | 7 | 4.682 |
| VD Bottom Shielded | 8 | 20 | 6 | 5.065 |

---

## Spike Debug (excluded from selection)

---

### HD Central — best excluded (spiked)

No spiked plots found.

---

### HD Lateral — best excluded (spiked)

No spiked plots found.

---

### VD Top — best excluded (spiked)

<div class="comparison-note">
  <strong>Debug:</strong> Highest-significance cut <em>excluded</em> from main selection due to a spike in the pre-isotonic PL curve. Compare against the main result to assess the impact of the filter.
</div>

<div class="two-col">
  <div>
<p><strong>Significance</strong></p>
<p>Significance plot not available.</p>
  </div>
  <div>
<p><strong>Exposure</strong></p>
<img src="../../output/images/analysis/hep/vd_1x8x14_3view_30deg_nominal/marley/truncated/vd_1x8x14_3view_30deg_nominal_marley_SolarEnergy_HEP_Exposure_ProfileLikelihood_Threshold_0_highest_spiked.png">
  </div>
</div>

---

### VD Bottom Shielded — best excluded (spiked)

<div class="comparison-note">
  <strong>Debug:</strong> Highest-significance cut <em>excluded</em> from main selection due to a spike in the pre-isotonic PL curve. Compare against the main result to assess the impact of the filter.
</div>

<div class="two-col">
  <div>
<p><strong>Significance</strong></p>
<p>Significance plot not available.</p>
  </div>
  <div>
<p><strong>Exposure</strong></p>
<img src="../../output/images/analysis/hep/vd_1x8x14_3view_30deg_shielded/marley/truncated/vd_1x8x14_3view_30deg_shielded_marley_SolarEnergy_HEP_Exposure_ProfileLikelihood_Threshold_0_highest_spiked.png">
  </div>
</div>

## Oscillograms

---

### HD Central

<div class="two-col">
  <div>
<p><strong>P(ν<sub>e</sub>→ν<sub>e</sub>) heatmap</strong></p>
<img src="../../output/images/analysis/hep/oscillogram/hd_1x2x6_centralAPA/marley/truncated/hd_1x2x6_centralAPA_marley_Oscillogram_SolarEnergy.png">
  </div>
  <div>
<p><strong>Nadir projection</strong></p>
<img src="../../output/images/analysis/hep/oscillogram/hd_1x2x6_centralAPA/marley/truncated/hd_1x2x6_centralAPA_marley_Oscillogram_NadirProjection_SolarEnergy.png">
  </div>
</div>

---

### HD Central (cont.)

<div class="two-col">
  <div>
<p><strong>Nadir-weighted oscillogram</strong></p>
<img src="../../output/images/analysis/hep/oscillogram/hd_1x2x6_centralAPA/marley/truncated/hd_1x2x6_centralAPA_marley_Oscillogram_NadirWeighted_SolarEnergy.png">
  </div>
  <div>
<p><strong>1D fiducial signal spectrum</strong></p>
<img src="../../output/images/analysis/hep/oscillogram/hd_1x2x6_centralAPA/marley/truncated/hd_1x2x6_centralAPA_marley_Signal1D_SolarEnergy_FidOnly.png">
  </div>
</div>

---

### HD Lateral

<div class="two-col">
  <div>
<p><strong>P(ν<sub>e</sub>→ν<sub>e</sub>) heatmap</strong></p>
<img src="../../output/images/analysis/hep/oscillogram/hd_1x2x6_lateralAPA/marley/truncated/hd_1x2x6_lateralAPA_marley_Oscillogram_SolarEnergy.png">
  </div>
  <div>
<p><strong>Nadir projection</strong></p>
<img src="../../output/images/analysis/hep/oscillogram/hd_1x2x6_lateralAPA/marley/truncated/hd_1x2x6_lateralAPA_marley_Oscillogram_NadirProjection_SolarEnergy.png">
  </div>
</div>

---

### HD Lateral (cont.)

<div class="two-col">
  <div>
<p><strong>Nadir-weighted oscillogram</strong></p>
<img src="../../output/images/analysis/hep/oscillogram/hd_1x2x6_lateralAPA/marley/truncated/hd_1x2x6_lateralAPA_marley_Oscillogram_NadirWeighted_SolarEnergy.png">
  </div>
  <div>
<p><strong>1D fiducial signal spectrum</strong></p>
<img src="../../output/images/analysis/hep/oscillogram/hd_1x2x6_lateralAPA/marley/truncated/hd_1x2x6_lateralAPA_marley_Signal1D_SolarEnergy_FidOnly.png">
  </div>
</div>

---

### VD Top

<div class="two-col">
  <div>
<p><strong>P(ν<sub>e</sub>→ν<sub>e</sub>) heatmap</strong></p>
<img src="../../output/images/analysis/hep/oscillogram/vd_1x8x14_3view_30deg_nominal/marley/truncated/vd_1x8x14_3view_30deg_nominal_marley_Oscillogram_SolarEnergy.png">
  </div>
  <div>
<p><strong>Nadir projection</strong></p>
<img src="../../output/images/analysis/hep/oscillogram/vd_1x8x14_3view_30deg_nominal/marley/truncated/vd_1x8x14_3view_30deg_nominal_marley_Oscillogram_NadirProjection_SolarEnergy.png">
  </div>
</div>

---

### VD Top (cont.)

<div class="two-col">
  <div>
<p><strong>Nadir-weighted oscillogram</strong></p>
<img src="../../output/images/analysis/hep/oscillogram/vd_1x8x14_3view_30deg_nominal/marley/truncated/vd_1x8x14_3view_30deg_nominal_marley_Oscillogram_NadirWeighted_SolarEnergy.png">
  </div>
  <div>
<p><strong>1D fiducial signal spectrum</strong></p>
<img src="../../output/images/analysis/hep/oscillogram/vd_1x8x14_3view_30deg_nominal/marley/truncated/vd_1x8x14_3view_30deg_nominal_marley_Signal1D_SolarEnergy_FidOnly.png">
  </div>
</div>

---

### VD Bottom Shielded

<div class="two-col">
  <div>
<p><strong>P(ν<sub>e</sub>→ν<sub>e</sub>) heatmap</strong></p>
<img src="../../output/images/analysis/hep/oscillogram/vd_1x8x14_3view_30deg_shielded/marley/truncated/vd_1x8x14_3view_30deg_shielded_marley_Oscillogram_SolarEnergy.png">
  </div>
  <div>
<p><strong>Nadir projection</strong></p>
<img src="../../output/images/analysis/hep/oscillogram/vd_1x8x14_3view_30deg_shielded/marley/truncated/vd_1x8x14_3view_30deg_shielded_marley_Oscillogram_NadirProjection_SolarEnergy.png">
  </div>
</div>

---

### VD Bottom Shielded (cont.)

<div class="two-col">
  <div>
<p><strong>Nadir-weighted oscillogram</strong></p>
<img src="../../output/images/analysis/hep/oscillogram/vd_1x8x14_3view_30deg_shielded/marley/truncated/vd_1x8x14_3view_30deg_shielded_marley_Oscillogram_NadirWeighted_SolarEnergy.png">
  </div>
  <div>
<p><strong>1D fiducial signal spectrum</strong></p>
<img src="../../output/images/analysis/hep/oscillogram/vd_1x8x14_3view_30deg_shielded/marley/truncated/vd_1x8x14_3view_30deg_shielded_marley_Signal1D_SolarEnergy_FidOnly.png">
  </div>
</div>

---



---

## Coverage and Notes

- Config coverage in best-cut JSON outputs:
  - nominal: 4
  - reduced: 4
  - truncated: 4
- Table values are read from workflow-generated JSON at generation time.
- Re-run script to refresh this folder after each workflow run:
- /usr/bin/python3 scripts/generate_hep_presentation.py --folder truncated
- Full mathematical derivations (signal model, PL formulation, adaptive rebinning, BB mask, spike detection): [docs/hep\_likelihood\_derivation.tex](../../docs/hep_likelihood_derivation.tex)

---

### Adaptive Rebinning: Strategy

- Rebinning is applied in [src/physics/hep/01_hep.py](../../src/physics/hep/01_hep.py) through [lib/smoothing.py](../../lib/smoothing.py) using `apply_adaptive_tail_rebin`.
- It is controlled by [config/analysis/config.json](../../config/analysis/config.json) under `ADAPTIVE_REBIN -> ANALYSES -> HEP`.
- At each exposure, bins are merged from the high-energy tail until the expected detectable signal per rebinned group reaches the configured threshold.
- This stabilizes low-statistics significance estimates while preserving discovery sensitivity.

Rebin threshold criterion used for each grouped bin:
$$
S_{\mathrm{group}} = \sum_{i \in \mathrm{group}} S_i^{\mathrm{det}} \ge T
$$
$$
T = \max\!\left(\texttt{min\_expected\_events},\,-\ln(1-\texttt{min\_count\_probability})\right)
$$

---

### Adaptive Rebinning: Discovery

Discovery significance is then evaluated on rebinned inputs:
$$
Z = Z\!\left(S_{\mathrm{group}},\,B_{\mathrm{group}},\,\sigma_{B,\mathrm{group}}\right)
$$

ProfileLikelihood implementation in [src/physics/hep/01_hep.py](../../src/physics/hep/01_hep.py):
- PL is computed for **every** analysis cut combination.
- Original fine binning used throughout — no adaptive rebin. PL is optimal at the finest resolution; the likelihood ratio naturally suppresses bins with negligible signal without merging.
- A **single global background normalization nuisance** (β ~ Gaussian(1, σ_rel)) is profiled jointly across all bins. See the *Background Normalization Model* slide.
