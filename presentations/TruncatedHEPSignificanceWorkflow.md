---
marp: true
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
- threshold in 13HEP.py: default 10.0 MeV
- optional cuts override: nhits, ophits, adjcls
- significance reference in plots: ProfileLikelihood
- best-curve reference in 0ZBestSigmas.py: Smoothed or Raw

---

### Workflow Outputs

- Fiducial optimization: [data/solar/fiducial/truncated/BestFiducials.json](../data/solar/fiducial/truncated/BestFiducials.json)
- Best cut summaries (JSON): [data/analysis/daynight-json/truncated](../data/analysis/daynight-json/truncated)
- Significance scans (PNFS outputs): [/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/HEP/truncated](/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/HEP/truncated)
- Figures: [images/analysis/hep/truncated](../images/analysis/hep/truncated)

---

### Histogram and Significance Flow

- Step 1: Build HEP rates and threshold region in [src/analysis/13HEP.py](../src/analysis/13HEP.py) per component.
- Step 2: Apply component-aware smoothing via [lib/lib_smooth.py](../lib/lib_smooth.py) using HEP smoothing config.
- Step 3: Evaluate Gaussian and Asimov significance curves in [src/analysis/13HEP.py](../src/analysis/13HEP.py).
- Step 4: Pick best cuts in [src/analysis/0ZBestSigmas.py](../src/analysis/0ZBestSigmas.py), then evaluate ProfileLikelihood exposure in [src/analysis/13HEPProfileLikelihood.py](../src/analysis/13HEPProfileLikelihood.py).
- Step 5: Render exposure/significance and comparison plots in [src/analysis/13HEPExposurePlot.py](../src/analysis/13HEPExposurePlot.py), [src/analysis/13HEPSignificancePlot.py](../src/analysis/13HEPSignificancePlot.py), [src/analysis/13HEPSignificanceComparisonPlot.py](../src/analysis/13HEPSignificanceComparisonPlot.py), and [src/analysis/13HEPExposureComparisonPlot.py](../src/analysis/13HEPExposureComparisonPlot.py).

---

### Adaptative Rebinning: Strategy

- Rebinning is applied in [src/analysis/13HEP.py](../src/analysis/13HEP.py) through [lib/lib_smooth.py](../lib/lib_smooth.py) using `apply_adaptive_tail_rebin`.
- It is controlled by [import/analysis.json](../import/analysis.json) under `ADAPTIVE_REBIN -> ANALYSES -> HEP`.
- At each exposure, bins are merged from the high-energy tail until the expected detectable signal per rebinned group reaches the configured threshold.
- This stabilizes low-statistics significance estimates while preserving discovery sensitivity in sparse tails.

Rebin threshold criterion used for each grouped bin:
$$
S_{\mathrm{group}} = \sum_{i \in \mathrm{group}} S_i^{\mathrm{det}} \ge T
$$
$$
T = \max\!\left(\texttt{min\_expected\_events},\,-\ln(1-\texttt{min\_count\_probability})\right)
$$

---

### Adaptative Rebinning: Discovery

Discovery significance is then evaluated on rebinned inputs:
$$
Z = Z\!\left(S_{\mathrm{group}},\,B_{\mathrm{group}},\,\sigma_{B,\mathrm{group}}\right)
$$

Latest ProfileLikelihood updates used in this deck:
- Adaptive rebinning is enabled by default in [src/analysis/13HEPProfileLikelihood.py](../src/analysis/13HEPProfileLikelihood.py).
- Detection-mask zeroing is disabled for the ProfileLikelihood path to avoid artificial early-exposure suppression.
- Adaptive rebin starts are frozen (computed once at reference exposure and reused across the scan) to reduce discontinuities and keep curves monotonic.

---

### Smoothing by Stage (HEP)

| Stage | Enabled | Method | Component Mode | Smoothed Components | Sigma |
|---|---|---|---|---|---|
| Fiducial | Yes | gaussian | only | gamma, neutron | 0.50 |
| Significance | Yes | gaussian | only | gamma, neutron | 0.50 |

---

## Fiducialization

---

### HD Central

<div class="two-col">
  <div>
<p><strong>No Fiducial</strong></p>
<img src="../images/solar/fiducial/hd_1x2x6_centralAPA/truncated/hd_1x2x6_centralAPA_SolarEnergy_HEP_NoFiducial_Significance.png">
  </div>
  <div>
<p><strong>Best Fiducial</strong></p>
<img src="../images/solar/fiducial/hd_1x2x6_centralAPA/truncated/hd_1x2x6_centralAPA_SolarEnergy_HEP_BestFiducial_Significance.png">
  </div>
</div>

---

### HD Lateral

<div class="two-col">
  <div>
<p><strong>No Fiducial</strong></p>
<img src="../images/solar/fiducial/hd_1x2x6_lateralAPA/truncated/hd_1x2x6_lateralAPA_SolarEnergy_HEP_NoFiducial_Significance.png">
  </div>
  <div>
<p><strong>Best Fiducial</strong></p>
<img src="../images/solar/fiducial/hd_1x2x6_lateralAPA/truncated/hd_1x2x6_lateralAPA_SolarEnergy_HEP_BestFiducial_Significance.png">
  </div>
</div>

---

### VD Top

<div class="two-col">
  <div>
<p><strong>No Fiducial</strong></p>
<img src="../images/solar/fiducial/vd_1x8x14_3view_30deg_nominal/truncated/vd_1x8x14_3view_30deg_nominal_SolarEnergy_HEP_NoFiducial_Significance.png">
  </div>
  <div>
<p><strong>Best Fiducial</strong></p>
<img src="../images/solar/fiducial/vd_1x8x14_3view_30deg_nominal/truncated/vd_1x8x14_3view_30deg_nominal_SolarEnergy_HEP_BestFiducial_Significance.png">
  </div>
</div>

---

### VD Bottom Shielded

<div class="two-col">
  <div>
<p><strong>No Fiducial</strong></p>
<img src="../images/solar/fiducial/vd_1x8x14_3view_30deg_shielded/truncated/vd_1x8x14_3view_30deg_shielded_SolarEnergy_HEP_NoFiducial_Significance.png">
  </div>
  <div>
<p><strong>Best Fiducial</strong></p>
<img src="../images/solar/fiducial/vd_1x8x14_3view_30deg_shielded/truncated/vd_1x8x14_3view_30deg_shielded_SolarEnergy_HEP_BestFiducial_Significance.png">
  </div>
</div>

---

### Fiducial Optimization Summary

| Config | Fiducial X | Fiducial Y | Fiducial Z | Before Fiducialization | After Fiducialization |
|---|---:|---:|---:|---:|---:|
| HD Central | 40 | 280 | 280 | 0.091 | 1.789 |
| HD Lateral | 40 | 220 | 20 | 0.078 | 1.132 |
| VD Top | 20 | 200 | 100 | 0.072 | 0.110 |
| VD Bottom Shielded | 0 | 0 | 0 | 0.393 | 0.393 |

---

## HEP Results

---

### HD Central

<div class="two-col">
  <div>
<p><strong>Significance</strong></p>
<img src="../images/analysis/hep/hd_1x2x6_centralAPA/truncated/hd_1x2x6_centralAPA_SolarEnergy_HEP_Significance_ProfileLikelihood_Exposure_30_BottomIntuitive.png">
  </div>
  <div>
<p><strong>Exposure</strong></p>
<img src="../images/analysis/hep/hd_1x2x6_centralAPA/truncated/hd_1x2x6_centralAPA_SolarEnergy_HEP_Exposure_ProfileLikelihood_Threshold_10.png">
  </div>
</div>

<div class="comparison-note">
  <strong>Lower subplot guide:</strong> local discovery density, estimated as z_local / DeltaE (sigma per MeV), where z_local comes from the per-bin discovery test statistic. Compare where discovery is concentrated 
</div>

---

### HD Lateral

<div class="two-col">
  <div>
<p><strong>Significance</strong></p>
<img src="../images/analysis/hep/hd_1x2x6_lateralAPA/truncated/hd_1x2x6_lateralAPA_SolarEnergy_HEP_Significance_ProfileLikelihood_Exposure_30_BottomIntuitive.png">
  </div>
  <div>
<p><strong>Exposure</strong></p>
<img src="../images/analysis/hep/hd_1x2x6_lateralAPA/truncated/hd_1x2x6_lateralAPA_SolarEnergy_HEP_Exposure_ProfileLikelihood_Threshold_10.png">
  </div>
</div>

<div class="comparison-note">
  <strong>Lower subplot guide:</strong> local discovery density, estimated as z_local / DeltaE (sigma per MeV), where z_local comes from the per-bin discovery test statistic. Compare where discovery is concentrated 
</div>

---

### VD Top

<div class="two-col">
  <div>
<p><strong>Significance</strong></p>
<img src="../images/analysis/hep/vd_1x8x14_3view_30deg_nominal/truncated/vd_1x8x14_3view_30deg_nominal_SolarEnergy_HEP_Significance_ProfileLikelihood_Exposure_30_BottomIntuitive.png">
  </div>
  <div>
<p><strong>Exposure</strong></p>
<img src="../images/analysis/hep/vd_1x8x14_3view_30deg_nominal/truncated/vd_1x8x14_3view_30deg_nominal_SolarEnergy_HEP_Exposure_ProfileLikelihood_Threshold_10.png">
  </div>
</div>

<div class="comparison-note">
  <strong>Lower subplot guide:</strong> local discovery density, estimated as z_local / DeltaE (sigma per MeV), where z_local comes from the per-bin discovery test statistic. Compare where discovery is concentrated 
</div>

---

### VD Bottom Shielded

<div class="two-col">
  <div>
<p><strong>Significance</strong></p>
<img src="../images/analysis/hep/vd_1x8x14_3view_30deg_shielded/truncated/vd_1x8x14_3view_30deg_shielded_SolarEnergy_HEP_Significance_ProfileLikelihood_Exposure_30_BottomIntuitive.png">
  </div>
  <div>
<p><strong>Exposure</strong></p>
<img src="../images/analysis/hep/vd_1x8x14_3view_30deg_shielded/truncated/vd_1x8x14_3view_30deg_shielded_SolarEnergy_HEP_Exposure_ProfileLikelihood_Threshold_10.png">
  </div>
</div>

<div class="comparison-note">
  <strong>Lower subplot guide:</strong> local discovery density, estimated as z_local / DeltaE (sigma per MeV), where z_local comes from the per-bin discovery test statistic. Compare where discovery is concentrated 
</div>

---

## Reference Comparison

---

### HD Central

<div class="two-col">
  <div>
<p><strong>Significance Reference Comparison</strong></p>
<img src="../images/analysis/hep/hd_1x2x6_centralAPA/truncated/hd_1x2x6_centralAPA_SolarEnergy_HEP_Significance_ProfileLikelihood_Exposure_30_BottomRigorous.png">
  </div>
  <div>
<p><strong>Exposure Reference Comparison</strong></p>
<img src="../images/analysis/hep/hd_1x2x6_centralAPA/truncated/hd_1x2x6_centralAPA_SolarEnergy_HEP_Exposure_Comparison_Threshold_10.png">
  </div>
</div>

---

### HD Lateral

<div class="two-col">
  <div>
<p><strong>Significance Reference Comparison</strong></p>
<img src="../images/analysis/hep/hd_1x2x6_lateralAPA/truncated/hd_1x2x6_lateralAPA_SolarEnergy_HEP_Significance_ProfileLikelihood_Exposure_30_BottomRigorous.png">
  </div>
  <div>
<p><strong>Exposure Reference Comparison</strong></p>
<img src="../images/analysis/hep/hd_1x2x6_lateralAPA/truncated/hd_1x2x6_lateralAPA_SolarEnergy_HEP_Exposure_Comparison_Threshold_10.png">
  </div>
</div>

---

### VD Top

<div class="two-col">
  <div>
<p><strong>Significance Reference Comparison</strong></p>
<img src="../images/analysis/hep/vd_1x8x14_3view_30deg_nominal/truncated/vd_1x8x14_3view_30deg_nominal_SolarEnergy_HEP_Significance_ProfileLikelihood_Exposure_30_BottomRigorous.png">
  </div>
  <div>
<p><strong>Exposure Reference Comparison</strong></p>
<img src="../images/analysis/hep/vd_1x8x14_3view_30deg_nominal/truncated/vd_1x8x14_3view_30deg_nominal_SolarEnergy_HEP_Exposure_Comparison_Threshold_10.png">
  </div>
</div>

---

### VD Bottom Shielded

<div class="two-col">
  <div>
<p><strong>Significance Reference Comparison</strong></p>
<img src="../images/analysis/hep/vd_1x8x14_3view_30deg_shielded/truncated/vd_1x8x14_3view_30deg_shielded_SolarEnergy_HEP_Significance_ProfileLikelihood_Exposure_30_BottomRigorous.png">
  </div>
  <div>
<p><strong>Exposure Reference Comparison</strong></p>
<img src="../images/analysis/hep/vd_1x8x14_3view_30deg_shielded/truncated/vd_1x8x14_3view_30deg_shielded_SolarEnergy_HEP_Exposure_Comparison_Threshold_10.png">
  </div>
</div>

---

## Adaptive Rebin Comparison

---

### HD Central

<div class="three-col">
  <div>
  <p><strong>Adaptive Rebin Comparison (Asimov)</strong></p>
  <img src="../images/analysis/hep/hd_1x2x6_centralAPA/marley/truncated/hd_1x2x6_centralAPA_marley_SolarEnergy_HEP_Asimov_AdaptiveRebin_Comparison_Threshold_10.png">
</div>
  <div>
  <p><strong>Adaptive Rebin Comparison (Gaussian)</strong></p>
  <img src="../images/analysis/hep/hd_1x2x6_centralAPA/marley/truncated/hd_1x2x6_centralAPA_marley_SolarEnergy_HEP_Gaussian_AdaptiveRebin_Comparison_Threshold_10.png">
</div>
  <div>
  <p><strong>Adaptive Rebin Comparison (ProfileLikelihood)</strong></p>
  <img src="../images/analysis/hep/hd_1x2x6_centralAPA/marley/truncated/hd_1x2x6_centralAPA_marley_SolarEnergy_HEP_ProfileLikelihood_AdaptiveRebin_Comparison_Threshold_10.png">
</div>
</div>

<div class="comparison-note">
  <strong>How to compare:</strong> Left = Asimov, middle = Gaussian, right = ProfileLikelihood.
  Compare the exposure turn-on point near threshold 10 MeV, relative ordering between methods, and curve smoothness/step behavior after adaptive rebinning.
</div>

---

### HD Lateral

<div class="three-col">
  <div>
  <p><strong>Adaptive Rebin Comparison (Asimov)</strong></p>
  <img src="../images/analysis/hep/hd_1x2x6_lateralAPA/marley/truncated/hd_1x2x6_lateralAPA_marley_SolarEnergy_HEP_Asimov_AdaptiveRebin_Comparison_Threshold_10.png">
</div>
  <div>
  <p><strong>Adaptive Rebin Comparison (Gaussian)</strong></p>
  <img src="../images/analysis/hep/hd_1x2x6_lateralAPA/marley/truncated/hd_1x2x6_lateralAPA_marley_SolarEnergy_HEP_Gaussian_AdaptiveRebin_Comparison_Threshold_10.png">
</div>
  <div>
  <p><strong>Adaptive Rebin Comparison (ProfileLikelihood)</strong></p>
  <img src="../images/analysis/hep/hd_1x2x6_lateralAPA/marley/truncated/hd_1x2x6_lateralAPA_marley_SolarEnergy_HEP_ProfileLikelihood_AdaptiveRebin_Comparison_Threshold_10.png">
</div>
</div>

<div class="comparison-note">
  <strong>How to compare:</strong> Left = Asimov, middle = Gaussian, right = ProfileLikelihood.
  Compare the exposure turn-on point near threshold 10 MeV, relative ordering between methods, and curve smoothness/step behavior after adaptive rebinning.
</div>

---

### VD Top

<div class="three-col">
  <div>
  <p><strong>Adaptive Rebin Comparison (Asimov)</strong></p>
  <img src="../images/analysis/hep/vd_1x8x14_3view_30deg_nominal/marley/truncated/vd_1x8x14_3view_30deg_nominal_marley_SolarEnergy_HEP_Asimov_AdaptiveRebin_Comparison_Threshold_10.png">
</div>
  <div>
  <p><strong>Adaptive Rebin Comparison (Gaussian)</strong></p>
  <img src="../images/analysis/hep/vd_1x8x14_3view_30deg_nominal/marley/truncated/vd_1x8x14_3view_30deg_nominal_marley_SolarEnergy_HEP_Gaussian_AdaptiveRebin_Comparison_Threshold_10.png">
</div>
  <div>
  <p><strong>Adaptive Rebin Comparison (ProfileLikelihood)</strong></p>
  <img src="../images/analysis/hep/vd_1x8x14_3view_30deg_nominal/marley/truncated/vd_1x8x14_3view_30deg_nominal_marley_SolarEnergy_HEP_ProfileLikelihood_AdaptiveRebin_Comparison_Threshold_10.png">
</div>
</div>

<div class="comparison-note">
  <strong>How to compare:</strong> Left = Asimov, middle = Gaussian, right = ProfileLikelihood.
  Compare the exposure turn-on point near threshold 10 MeV, relative ordering between methods, and curve smoothness/step behavior after adaptive rebinning.
</div>

---

### VD Bottom Shielded

<div class="three-col">
  <div>
  <p><strong>Adaptive Rebin Comparison (Asimov)</strong></p>
  <img src="../images/analysis/hep/vd_1x8x14_3view_30deg_shielded/marley/truncated/vd_1x8x14_3view_30deg_shielded_marley_SolarEnergy_HEP_Asimov_AdaptiveRebin_Comparison_Threshold_10.png">
</div>
  <div>
  <p><strong>Adaptive Rebin Comparison (Gaussian)</strong></p>
  <img src="../images/analysis/hep/vd_1x8x14_3view_30deg_shielded/marley/truncated/vd_1x8x14_3view_30deg_shielded_marley_SolarEnergy_HEP_Gaussian_AdaptiveRebin_Comparison_Threshold_10.png">
</div>
  <div>
  <p><strong>Adaptive Rebin Comparison (ProfileLikelihood)</strong></p>
  <img src="../images/analysis/hep/vd_1x8x14_3view_30deg_shielded/marley/truncated/vd_1x8x14_3view_30deg_shielded_marley_SolarEnergy_HEP_ProfileLikelihood_AdaptiveRebin_Comparison_Threshold_10.png">
</div>
</div>

<div class="comparison-note">
  <strong>How to compare:</strong> Left = Asimov, middle = Gaussian, right = ProfileLikelihood.
  Compare the exposure turn-on point near threshold 10 MeV, relative ordering between methods, and curve smoothness/step behavior after adaptive rebinning.
</div>

---

### Best HEP Cuts by Config

| Config | NHits | OpHits | AdjCl | Significance |
|---|---:|---:|---:|---:|
| HD Central | 1 | 8 | 8 | 9.843 |
| HD Lateral | 7 | 4 | 4 | 1.810 |
| VD Top | 9 | 8 | 7 | 0.171 |
| VD Bottom Shielded | 8 | 9 | 10 | 0.361 |


---

## Coverage and Notes

- Config coverage in best-cut JSON outputs:
  - nominal: 4
  - reduced: 4
  - truncated: 4
- Table values are read from workflow-generated JSON at generation time.
- Re-run script to refresh this folder after each workflow run:
- /usr/bin/python3 scripts/generate_hep_presentation.py --folder truncated
