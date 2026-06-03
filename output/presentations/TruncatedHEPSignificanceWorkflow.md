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
- threshold in hep/01_hep.py: from [analysis/config.json](../../analysis/config.json) HEP -> THRESHOLDS -> (no threshold config found)
- optional cuts override: nhits, ophits, adjcls
- significance reference in plots: ProfileLikelihood
- best-cut selection in sensitivity/05_best_sigmas.py: **ProfileLikelihood** (smoothed, 3σ crossing)

---

### Workflow Skip Flags

Used for [src/pipelines/run_sensitivity.py](../../src/pipelines/run_sensitivity.py):
- `--no-computation`: skip all analysis, run plot macros only
- `--no-significance`: skip 01_hep.py/01_daynight.py/06_significance.py only
- `--no-fiducialization`: skip signal/01_fiducialize.py only
- `--no-rebin`: skip signal/03_analysis.py rebinning step only

---

### Workflow Outputs

- Fiducial optimization: [data/solar/fiducial/truncated/BestFiducials.json](../../data/solar/fiducial/truncated/BestFiducials.json)
- Best cut summaries (JSON): [data/analysis/hep-json/truncated](../../data/analysis/hep-json/truncated)
- Significance scans (PNFS outputs): [/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/HEP/truncated](/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/HEP/truncated)
- Figures: [images/analysis/hep/truncated](../../images/analysis/hep/truncated)

---

### Histogram and Significance Flow I: Building, Smoothing, and Evaluation

- Step 1: Build HEP rates and threshold region in [src/physics/hep/01_hep.py](../../src/physics/hep/01_hep.py) per component.
- Step 2: Apply component-aware smoothing via [lib/smoothing.py](../../lib/smoothing.py) using HEP smoothing config.
- Step 3: Evaluate Gaussian, Asimov, and ProfileLikelihood significance curves in [src/physics/hep/01_hep.py](../../src/physics/hep/01_hep.py) for **all** analysis cuts. ProfileLikelihood uses a single global background normalization nuisance profiled jointly across all bins (see *Background Normalization Model* slide). Background bins with fewer than `min_mc_per_bin` raw MC events are masked using the [Barlow-Beeston lite criterion](https://www.sciencedirect.com/science/article/pii/009350659390005W) (as implemented in [ROOT HistFactory](https://root.cern.ch/doc/master/classRooStats_1_1HistFactory_1_1Measurement.html)) to suppress LLR divergence from empty bins. Smoothed histogram rates are clipped to ≥ 0 before the PL step to prevent negative-rate blowup at high exposures.

---

### Histogram and Significance Flow II: Post-Processing and Plotting

- Step 4: Select the best cut by ProfileLikelihood in [src/physics/sensitivity/05_best_sigmas.py](../../src/physics/sensitivity/05_best_sigmas.py). Cuts whose PL curve contains a single-step jump exceeding `max_pl_jump` σ in either the raw or smoothed pre-isotonic column are flagged as spiked, excluded from the main `highest` selection, and saved separately as `highest_spiked` for inspection.
- Step 5: Render exposure/significance and comparison plots in [src/physics/hep/exposure_plot.py](../../src/physics/hep/exposure_plot.py), [src/physics/hep/significance_plot.py](../../src/physics/hep/significance_plot.py), [src/physics/hep/significance_comparison.py](../../src/physics/hep/significance_comparison.py), and [src/physics/hep/exposure_comparison.py](../../src/physics/hep/exposure_comparison.py).

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

### ProfileLikelihood Smoothing

PL curves are post-processed with **Gaussian kernel smoothing followed by isotonic regression** to produce a continuous, monotone exposure curve:
  1. [`scipy.ndimage.gaussian_filter1d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html) convolves the raw PL significance array with a Gaussian kernel (σ = 6 exposure-grid index units, tunable via `_PL_SMOOTH_SIGMA` in [`src/physics/hep/01_hep.py`](../../src/physics/hep/01_hep.py)). This mirrors the approach used by [ROOT `TH1::Smooth`](https://root.cern.ch/doc/master/classTH1.html#a16) for smoothing discrete numerical histograms.
  2. [`sklearn.isotonic.IsotonicRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html) (PAVA) is then applied to enforce strict monotonicity. It finds the non-decreasing sequence that minimises the L2 distance from the smoothed values, ensuring more data cannot reduce sensitivity.

These steps remove residual numerical oscillations from the profile-likelihood solver at low signal-to-background ratios.

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
| Fiducial | Yes | gaussian | only | gamma, neutron, radiological, 8B | 0.62 |
| Significance | Yes | gaussian | only | gamma, neutron, radiological, 8B | 0.62 |

---

## Fiducialization

---

### HD Central

<div class="two-col">
  <div>
<p><strong>No Fiducial</strong></p>
<img src="../../images/solar/fiducial/hd_1x2x6_centralAPA/marley/truncated/hd_1x2x6_centralAPA_marley_SolarEnergy_HEP_NoFiducial_Significance.png">
  </div>
  <div>
<p><strong>Best Fiducial</strong></p>
<img src="../../images/solar/fiducial/hd_1x2x6_centralAPA/marley/truncated/hd_1x2x6_centralAPA_marley_SolarEnergy_HEP_BestFiducial_Significance.png">
  </div>
</div>

---

### HD Lateral

<div class="two-col">
  <div>
<p><strong>No Fiducial</strong></p>
<img src="../../images/solar/fiducial/hd_1x2x6_lateralAPA/marley/truncated/hd_1x2x6_lateralAPA_marley_SolarEnergy_HEP_NoFiducial_Significance.png">
  </div>
  <div>
<p><strong>Best Fiducial</strong></p>
<img src="../../images/solar/fiducial/hd_1x2x6_lateralAPA/marley/truncated/hd_1x2x6_lateralAPA_marley_SolarEnergy_HEP_BestFiducial_Significance.png">
  </div>
</div>

---

### VD Top

<div class="two-col">
  <div>
<p><strong>No Fiducial</strong></p>
<img src="../../images/solar/fiducial/vd_1x8x14_3view_30deg_nominal/marley/truncated/vd_1x8x14_3view_30deg_nominal_marley_SolarEnergy_HEP_NoFiducial_Significance.png">
  </div>
  <div>
<p><strong>Best Fiducial</strong></p>
<img src="../../images/solar/fiducial/vd_1x8x14_3view_30deg_nominal/marley/truncated/vd_1x8x14_3view_30deg_nominal_marley_SolarEnergy_HEP_BestFiducial_Significance.png">
  </div>
</div>

---

### VD Bottom Shielded

<div class="two-col">
  <div>
<p><strong>No Fiducial</strong></p>
<img src="../../images/solar/fiducial/vd_1x8x14_3view_30deg_shielded/marley/truncated/vd_1x8x14_3view_30deg_shielded_marley_SolarEnergy_HEP_NoFiducial_Significance.png">
  </div>
  <div>
<p><strong>Best Fiducial</strong></p>
<img src="../../images/solar/fiducial/vd_1x8x14_3view_30deg_shielded/marley/truncated/vd_1x8x14_3view_30deg_shielded_marley_SolarEnergy_HEP_BestFiducial_Significance.png">
  </div>
</div>

---

### Fiducial Optimization Summary

| Config | Fiducial X | Fiducial Y | Fiducial Z | Before Fiducialization | After Fiducialization | Fiducial Mass (kt) |
|---|---:|---:|---:|---:|---:|---:|
| HD Central | 0 | 80 | 0 | 0.303 | 1.226 | 5.85 |
| HD Lateral | 60 | 80 | 20 | 0.095 | 0.817 | 4.74 |
| VD Top | 0 | 0 | 40 | 0.551 | 0.575 | 7.54 |
| VD Bottom Shielded | 0 | 0 | 20 | 0.781 | 0.782 | 7.69 |

---

## HEP Results

---

### HD Central

<div class="two-col">
  <div>
<p><strong>Significance</strong></p>
<img src="../../images/analysis/hep/hd_1x2x6_centralAPA/marley/truncated/hd_1x2x6_centralAPA_marley_SolarEnergy_HEP_Significance_ProfileLikelihood_Exposure_30_BottomIntuitive.png">
  </div>
  <div>
<p><strong>Exposure</strong></p>
<img src="../../images/analysis/hep/hd_1x2x6_centralAPA/marley/truncated/hd_1x2x6_centralAPA_marley_SolarEnergy_HEP_Exposure_ProfileLikelihood_Threshold_0.png">
  </div>
</div>

<div class="comparison-note">
  <strong>Lower subplot guide:</strong> local discovery density, estimated as z_local / DeltaE (sigma per MeV), where z_local comes from the per-bin discovery test statistic. Compare where discovery is concentrated.
</div>

---

### HD Lateral

<div class="two-col">
  <div>
<p><strong>Significance</strong></p>
<img src="../../images/analysis/hep/hd_1x2x6_lateralAPA/marley/truncated/hd_1x2x6_lateralAPA_marley_SolarEnergy_HEP_Significance_ProfileLikelihood_Exposure_30_BottomIntuitive.png">
  </div>
  <div>
<p><strong>Exposure</strong></p>
<img src="../../images/analysis/hep/hd_1x2x6_lateralAPA/marley/truncated/hd_1x2x6_lateralAPA_marley_SolarEnergy_HEP_Exposure_ProfileLikelihood_Threshold_0.png">
  </div>
</div>

<div class="comparison-note">
  <strong>Lower subplot guide:</strong> local discovery density, estimated as z_local / DeltaE (sigma per MeV), where z_local comes from the per-bin discovery test statistic. Compare where discovery is concentrated.
</div>

---

### VD Top

<div class="two-col">
  <div>
<p><strong>Significance</strong></p>
<img src="../../images/analysis/hep/vd_1x8x14_3view_30deg_nominal/marley/truncated/vd_1x8x14_3view_30deg_nominal_marley_SolarEnergy_HEP_Significance_ProfileLikelihood_Exposure_30_BottomIntuitive.png">
  </div>
  <div>
<p><strong>Exposure</strong></p>
<img src="../../images/analysis/hep/vd_1x8x14_3view_30deg_nominal/marley/truncated/vd_1x8x14_3view_30deg_nominal_marley_SolarEnergy_HEP_Exposure_ProfileLikelihood_Threshold_0.png">
  </div>
</div>

<div class="comparison-note">
  <strong>Lower subplot guide:</strong> local discovery density, estimated as z_local / DeltaE (sigma per MeV), where z_local comes from the per-bin discovery test statistic. Compare where discovery is concentrated.
</div>

---

### VD Bottom Shielded

<div class="two-col">
  <div>
<p><strong>Significance</strong></p>
<img src="../../images/analysis/hep/vd_1x8x14_3view_30deg_shielded/marley/truncated/vd_1x8x14_3view_30deg_shielded_marley_SolarEnergy_HEP_Significance_ProfileLikelihood_Exposure_30_BottomIntuitive.png">
  </div>
  <div>
<p><strong>Exposure</strong></p>
<img src="../../images/analysis/hep/vd_1x8x14_3view_30deg_shielded/marley/truncated/vd_1x8x14_3view_30deg_shielded_marley_SolarEnergy_HEP_Exposure_ProfileLikelihood_Threshold_0.png">
  </div>
</div>

<div class="comparison-note">
  <strong>Lower subplot guide:</strong> local discovery density, estimated as z_local / DeltaE (sigma per MeV), where z_local comes from the per-bin discovery test statistic. Compare where discovery is concentrated.
</div>

---

## Reference Comparison

---

### HD Central

<div class="two-col">
  <div>
<p><strong>Significance Reference Comparison</strong></p>
<img src="../../images/analysis/hep/hd_1x2x6_centralAPA/marley/truncated/hd_1x2x6_centralAPA_marley_SolarEnergy_HEP_Significance_ProfileLikelihood_Exposure_30_BottomRigorous.png">
  </div>
  <div>
<p><strong>Exposure Reference Comparison</strong></p>
<img src="../../images/analysis/hep/hd_1x2x6_centralAPA/marley/truncated/hd_1x2x6_centralAPA_marley_SolarEnergy_HEP_Exposure_Comparison_Threshold_0.png">
  </div>
</div>

---

### HD Lateral

<div class="two-col">
  <div>
<p><strong>Significance Reference Comparison</strong></p>
<img src="../../images/analysis/hep/hd_1x2x6_lateralAPA/marley/truncated/hd_1x2x6_lateralAPA_marley_SolarEnergy_HEP_Significance_ProfileLikelihood_Exposure_30_BottomRigorous.png">
  </div>
  <div>
<p><strong>Exposure Reference Comparison</strong></p>
<img src="../../images/analysis/hep/hd_1x2x6_lateralAPA/marley/truncated/hd_1x2x6_lateralAPA_marley_SolarEnergy_HEP_Exposure_Comparison_Threshold_0.png">
  </div>
</div>

---

### VD Top

<div class="two-col">
  <div>
<p><strong>Significance Reference Comparison</strong></p>
<img src="../../images/analysis/hep/vd_1x8x14_3view_30deg_nominal/marley/truncated/vd_1x8x14_3view_30deg_nominal_marley_SolarEnergy_HEP_Significance_ProfileLikelihood_Exposure_30_BottomRigorous.png">
  </div>
  <div>
<p><strong>Exposure Reference Comparison</strong></p>
<img src="../../images/analysis/hep/vd_1x8x14_3view_30deg_nominal/marley/truncated/vd_1x8x14_3view_30deg_nominal_marley_SolarEnergy_HEP_Exposure_Comparison_Threshold_0.png">
  </div>
</div>

---

### VD Bottom Shielded

<div class="two-col">
  <div>
<p><strong>Significance Reference Comparison</strong></p>
<img src="../../images/analysis/hep/vd_1x8x14_3view_30deg_shielded/marley/truncated/vd_1x8x14_3view_30deg_shielded_marley_SolarEnergy_HEP_Significance_ProfileLikelihood_Exposure_30_BottomRigorous_highest_spiked.png">
  </div>
  <div>
<p><strong>Exposure Reference Comparison</strong></p>
<img src="../../images/analysis/hep/vd_1x8x14_3view_30deg_shielded/marley/truncated/vd_1x8x14_3view_30deg_shielded_marley_SolarEnergy_HEP_Exposure_Comparison_Threshold_0.png">
  </div>
</div>

---

## Adaptive Rebin Comparison

---

### HD Central

<div class="three-col">
  <div>
  <p><strong>Gaussian</strong></p>
  <img src="../../images/analysis/hep/hd_1x2x6_centralAPA/marley/truncated/hd_1x2x6_centralAPA_marley_SolarEnergy_HEP_Gaussian_AdaptiveRebin_Comparison_Threshold_0.png">
</div>
  <div>
  <p><strong>Asimov</strong></p>
  <img src="../../images/analysis/hep/hd_1x2x6_centralAPA/marley/truncated/hd_1x2x6_centralAPA_marley_SolarEnergy_HEP_Asimov_AdaptiveRebin_Comparison_Threshold_0.png">
</div>
  <div>
  <p><strong>ProfileLikelihood</strong></p>
  <img src="../../images/analysis/hep/hd_1x2x6_centralAPA/marley/truncated/hd_1x2x6_centralAPA_marley_SolarEnergy_HEP_ProfileLikelihood_AdaptiveRebin_Comparison_Threshold_0.png">
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
  <img src="../../images/analysis/hep/hd_1x2x6_lateralAPA/marley/truncated/hd_1x2x6_lateralAPA_marley_SolarEnergy_HEP_Gaussian_AdaptiveRebin_Comparison_Threshold_0.png">
</div>
  <div>
  <p><strong>Asimov</strong></p>
  <img src="../../images/analysis/hep/hd_1x2x6_lateralAPA/marley/truncated/hd_1x2x6_lateralAPA_marley_SolarEnergy_HEP_Asimov_AdaptiveRebin_Comparison_Threshold_0.png">
</div>
  <div>
  <p><strong>ProfileLikelihood</strong></p>
  <img src="../../images/analysis/hep/hd_1x2x6_lateralAPA/marley/truncated/hd_1x2x6_lateralAPA_marley_SolarEnergy_HEP_ProfileLikelihood_AdaptiveRebin_Comparison_Threshold_0.png">
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
  <img src="../../images/analysis/hep/vd_1x8x14_3view_30deg_nominal/marley/truncated/vd_1x8x14_3view_30deg_nominal_marley_SolarEnergy_HEP_Gaussian_AdaptiveRebin_Comparison_Threshold_0.png">
</div>
  <div>
  <p><strong>Asimov</strong></p>
  <img src="../../images/analysis/hep/vd_1x8x14_3view_30deg_nominal/marley/truncated/vd_1x8x14_3view_30deg_nominal_marley_SolarEnergy_HEP_Asimov_AdaptiveRebin_Comparison_Threshold_0.png">
</div>
  <div>
  <p><strong>ProfileLikelihood</strong></p>
  <img src="../../images/analysis/hep/vd_1x8x14_3view_30deg_nominal/marley/truncated/vd_1x8x14_3view_30deg_nominal_marley_SolarEnergy_HEP_ProfileLikelihood_AdaptiveRebin_Comparison_Threshold_0.png">
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
  <img src="../../images/analysis/hep/vd_1x8x14_3view_30deg_shielded/marley/truncated/vd_1x8x14_3view_30deg_shielded_marley_SolarEnergy_HEP_Gaussian_AdaptiveRebin_Comparison_Threshold_0.png">
</div>
  <div>
  <p><strong>Asimov</strong></p>
  <img src="../../images/analysis/hep/vd_1x8x14_3view_30deg_shielded/marley/truncated/vd_1x8x14_3view_30deg_shielded_marley_SolarEnergy_HEP_Asimov_AdaptiveRebin_Comparison_Threshold_0.png">
</div>
  <div>
  <p><strong>ProfileLikelihood</strong></p>
  <img src="../../images/analysis/hep/vd_1x8x14_3view_30deg_shielded/marley/truncated/vd_1x8x14_3view_30deg_shielded_marley_SolarEnergy_HEP_ProfileLikelihood_AdaptiveRebin_Comparison_Threshold_0.png">
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
| HD Central | 4 | 13 | 5 | 9.099 |
| HD Lateral | 6 | 4 | 5 | 2.395 |
| VD Top | 1 | 4 | 20 | 0.000 |
| VD Bottom Shielded | 1 | 4 | 19 | 0.000 |

---

## Spike Debug (excluded from selection)

---

### HD Central — best excluded (spiked)

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
<img src="../../images/analysis/hep/hd_1x2x6_centralAPA/marley/truncated/hd_1x2x6_centralAPA_marley_SolarEnergy_HEP_Exposure_ProfileLikelihood_Threshold_0_highest_spiked.png">
  </div>
</div>

---

### HD Lateral — best excluded (spiked)

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
<img src="../../images/analysis/hep/hd_1x2x6_lateralAPA/marley/truncated/hd_1x2x6_lateralAPA_marley_SolarEnergy_HEP_Exposure_ProfileLikelihood_Threshold_0_highest_spiked.png">
  </div>
</div>

---

### VD Top — best excluded (spiked)

<div class="comparison-note">
  <strong>Debug:</strong> Highest-significance cut <em>excluded</em> from main selection due to a spike in the pre-isotonic PL curve. Compare against the main result to assess the impact of the filter.
</div>

<div class="two-col">
  <div>
<p><strong>Significance</strong></p>
<img src="../../images/analysis/hep/vd_1x8x14_3view_30deg_nominal/marley/truncated/vd_1x8x14_3view_30deg_nominal_marley_SolarEnergy_HEP_Significance_ProfileLikelihood_Exposure_30_BottomIntuitive_highest_spiked.png">
  </div>
  <div>
<p><strong>Exposure</strong></p>
<img src="../../images/analysis/hep/vd_1x8x14_3view_30deg_nominal/marley/truncated/vd_1x8x14_3view_30deg_nominal_marley_SolarEnergy_HEP_Exposure_ProfileLikelihood_Threshold_0_highest_spiked.png">
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
<img src="../../images/analysis/hep/vd_1x8x14_3view_30deg_shielded/marley/truncated/vd_1x8x14_3view_30deg_shielded_marley_SolarEnergy_HEP_Significance_ProfileLikelihood_Exposure_30_BottomIntuitive_highest_spiked.png">
  </div>
  <div>
<p><strong>Exposure</strong></p>
<img src="../../images/analysis/hep/vd_1x8x14_3view_30deg_shielded/marley/truncated/vd_1x8x14_3view_30deg_shielded_marley_SolarEnergy_HEP_Exposure_ProfileLikelihood_Threshold_0_highest_spiked.png">
  </div>
</div>



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
- It is controlled by [analysis/config.json](../../analysis/config.json) under `ADAPTIVE_REBIN -> ANALYSES -> HEP`.
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
