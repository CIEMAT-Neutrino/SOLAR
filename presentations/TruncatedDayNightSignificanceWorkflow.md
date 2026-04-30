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
- threshold in 12DayNight.py: default 8.0 MeV
- optional cuts override: nhits, ophits, adjcls
- best-curve reference in 0ZBestSigmas.py: Smoothed or Raw

---

### Workflow Outputs

- Fiducial optimization: [data/solar/fiducial/truncated/BestFiducials.json](../data/solar/fiducial/truncated/BestFiducials.json)
- Best cut summaries (JSON): [data/analysis/best-sigma-json/daynight/truncated](../data/analysis/best-sigma-json/daynight/truncated)
- Backward-compatible local fallback: [data/analysis/daynight-json/truncated](../data/analysis/daynight-json/truncated)
- Significance scans (PNFS outputs): [/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/DAYNIGHT/truncated](/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/DAYNIGHT/truncated)
- Figures: [images/analysis/day-night/truncated](../images/analysis/day-night/truncated)

---

### Day-Night Discovery Statistic

- [src/analysis/12DayNight.py](../src/analysis/12DayNight.py) forms the baseline spectrum as background plus the day signal, using $B_i = B_i^{raw}/2 + S_i^{day}$ and the corresponding smoothed components above threshold.
- The discovery spectrum is the day-night difference, $\Delta S_i = S_i^{night} - S_i^{day}$, evaluated in the thresholded energy region.
- Per-bin discovery proxies are computed with `evaluate_significance(..., type="gaussian")` in [lib/lib_sigma.py](../lib/lib_sigma.py), then combined into the global curve as:
$$
$Z_{global} = \sqrt{\sum_i Z_i^2}$.
$$

---

### Day-Night Discovery Statistic Details

- [src/analysis/12DayNight.py](../src/analysis/12DayNight.py) stores both the plain Gaussian curve and an alternate version with background-statistical uncertainty included; this workflow does not perform a likelihood or chi-square fit.
- Smoothing is only used to build alternate component spectra: each component is smoothed separately, and the threshold slice keeps unsmoothed bins below threshold while replacing bins above threshold with their smoothed values.

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
<img src="../images/solar/fiducial/hd_1x2x6_centralAPA/truncated/hd_1x2x6_centralAPA_SolarEnergy_DAYNIGHT_NoFiducial_Significance.png">
  </div>
  <div>
<p><strong>Best Fiducial</strong></p>
<img src="../images/solar/fiducial/hd_1x2x6_centralAPA/truncated/hd_1x2x6_centralAPA_SolarEnergy_DAYNIGHT_BestFiducial_Significance.png">
  </div>
</div>

---

### HD Lateral

<div class="two-col">
  <div>
<p><strong>No Fiducial</strong></p>
<img src="../images/solar/fiducial/hd_1x2x6_lateralAPA/truncated/hd_1x2x6_lateralAPA_SolarEnergy_DAYNIGHT_NoFiducial_Significance.png">
  </div>
  <div>
<p><strong>Best Fiducial</strong></p>
<img src="../images/solar/fiducial/hd_1x2x6_lateralAPA/truncated/hd_1x2x6_lateralAPA_SolarEnergy_DAYNIGHT_BestFiducial_Significance.png">
  </div>
</div>

---

### VD Top

<div class="two-col">
  <div>
<p><strong>No Fiducial</strong></p>
<img src="../images/solar/fiducial/vd_1x8x14_3view_30deg_nominal/truncated/vd_1x8x14_3view_30deg_nominal_SolarEnergy_DAYNIGHT_NoFiducial_Significance.png">
  </div>
  <div>
<p><strong>Best Fiducial</strong></p>
<img src="../images/solar/fiducial/vd_1x8x14_3view_30deg_nominal/truncated/vd_1x8x14_3view_30deg_nominal_SolarEnergy_DAYNIGHT_BestFiducial_Significance.png">
  </div>
</div>

---

### VD Bottom Shielded

<div class="two-col">
  <div>
<p><strong>No Fiducial</strong></p>
<img src="../images/solar/fiducial/vd_1x8x14_3view_30deg_shielded/truncated/vd_1x8x14_3view_30deg_shielded_SolarEnergy_DAYNIGHT_NoFiducial_Significance.png">
  </div>
  <div>
<p><strong>Best Fiducial</strong></p>
<img src="../images/solar/fiducial/vd_1x8x14_3view_30deg_shielded/truncated/vd_1x8x14_3view_30deg_shielded_SolarEnergy_DAYNIGHT_BestFiducial_Significance.png">
  </div>
</div>

---

### Fiducial Optimization Summary

| Config | Fiducial X | Fiducial Y | Fiducial Z | Before Fiducialization | After Fiducialization |
|---|---:|---:|---:|---:|---:|
| HD Central | 20 | 280 | 280 | 0.131 | 8.268 |
| HD Lateral | 80 | 200 | 340 | 0.040 | 2.293 |
| VD Top | 0 | 0 | 100 | 0.004 | 0.024 |
| VD Bottom Shielded | 0 | 0 | 60 | 0.020 | 0.036 |

---

## DayNight Results

---

### HD Central

<div class="two-col">
  <div>
<p><strong>Significance</strong></p>
<img src="../images/analysis/day-night/hd_1x2x6_centralAPA/truncated/hd_1x2x6_centralAPA_SolarEnergy_DayNight_Significance_Exposure_30_Threshold_8.png">
  </div>
  <div>
<p><strong>Exposure</strong></p>
<img src="../images/analysis/day-night/hd_1x2x6_centralAPA/truncated/hd_1x2x6_centralAPA_SolarEnergy_DayNight_Exposure_Threshold_8.png">
  </div>
</div>

---

### HD Lateral

<div class="two-col">
  <div>
<p><strong>Significance</strong></p>
<img src="../images/analysis/day-night/hd_1x2x6_lateralAPA/truncated/hd_1x2x6_lateralAPA_SolarEnergy_DayNight_Significance_Exposure_30_Threshold_8.png">
  </div>
  <div>
<p><strong>Exposure</strong></p>
<img src="../images/analysis/day-night/hd_1x2x6_lateralAPA/truncated/hd_1x2x6_lateralAPA_SolarEnergy_DayNight_Exposure_Threshold_8.png">
  </div>
</div>

---

### VD Top

<div class="two-col">
  <div>
<p><strong>Significance</strong></p>
<img src="../images/analysis/day-night/vd_1x8x14_3view_30deg_nominal/truncated/vd_1x8x14_3view_30deg_nominal_SolarEnergy_DayNight_Significance_Exposure_30_Threshold_8.png">
  </div>
  <div>
<p><strong>Exposure</strong></p>
<img src="../images/analysis/day-night/vd_1x8x14_3view_30deg_nominal/truncated/vd_1x8x14_3view_30deg_nominal_SolarEnergy_DayNight_Exposure_Threshold_8.png">
  </div>
</div>

---

### VD Bottom Shielded

<div class="two-col">
  <div>
<p><strong>Significance</strong></p>
<img src="../images/analysis/day-night/vd_1x8x14_3view_30deg_shielded/truncated/vd_1x8x14_3view_30deg_shielded_SolarEnergy_DayNight_Significance_Exposure_30_Threshold_8.png">
  </div>
  <div>
<p><strong>Exposure</strong></p>
<img src="../images/analysis/day-night/vd_1x8x14_3view_30deg_shielded/truncated/vd_1x8x14_3view_30deg_shielded_SolarEnergy_DayNight_Exposure_Threshold_8.png">
  </div>
</div>

---

### Best DayNight Cuts by Config

| Config | NHits | OpHits | AdjCl | Significance |
|---|---:|---:|---:|---:|
| HD Central | 3 | 10 | 3 | 5.903 |
| HD Lateral | 4 | 4 | 4 | 1.394 |
| VD Top | 8 | 4 | 4 | 1.114 |
| VD Bottom Shielded | 6 | 4 | 7 | 3.269 |


---

## Coverage and Notes

- Config coverage in best-cut JSON outputs:
  - nominal: 5
  - reduced: 4
  - truncated: 4
- Table values are read from workflow-generated JSON at generation time.
- Re-run script to refresh this folder after each workflow run:
- /usr/bin/python3 scripts/generate_daynight_presentation.py --folder truncated
