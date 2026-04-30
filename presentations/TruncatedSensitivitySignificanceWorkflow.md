---
marp: true
description: Inputs, workflow outputs, and per-config Sensitivity results
paginate: true
theme: dune
---

<!-- AUTO-GENERATED: scripts/generate_sensitivity_presentation.py -->

<!-- _class: titlepage -->

# Sensitivity Workflow

---

## Introduction

This presentation summarizes the workflow and outputs of the Sensitivity analysis for the SOLAR project.
- This deck is auto-generated from workflow outputs.
- This deck is scoped to the **Truncated** folder for the **SolarEnergy** reconstruction algorithm.

Config aliases:
- hd_1x2x6_centralAPA: HD Central
- hd_1x2x6_lateralAPA: HD Lateral
- vd_1x8x14_3view_30deg_nominal: VD Top
- vd_1x8x14_3view_30deg_shielded: VD Bottom Shielded

---

### Workflow

- Orchestrator: [src/analysis/10SensitivityAnalysis.py](../src/analysis/10SensitivityAnalysis.py)
- Step 1 (Background template): [src/analysis/14SensitivityBackgroundTemplate.py](../src/analysis/14SensitivityBackgroundTemplate.py)
- Step 2 (Signal template): [src/analysis/14SensitivitySignalTemplate.py](../src/analysis/14SensitivitySignalTemplate.py)
- Step 3 (Grid fit scan and best-cut storage): [src/analysis/14Sensitivity.py](../src/analysis/14Sensitivity.py)
- Step 4 (Contour rendering): [src/analysis/14SensitivityContourPlot.py](../src/analysis/14SensitivityContourPlot.py)

---

### Workflow Outputs

- Main contour grids (sin12/sin13): [images/analysis/sensitivity](../images/analysis/sensitivity)
- Signal/background templates (figures): [images/analysis/sensitivity/templates](../images/analysis/sensitivity/templates)
- Grid-scan data products (PKL): [data/analysis/sensitivity](../data/analysis/sensitivity)
- Remote workflow outputs (PNFS): [/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/SENSITIVITY](/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/SENSITIVITY)

---

### Sensitivity Fit Summary

- [src/analysis/14Sensitivity.py](../src/analysis/14Sensitivity.py) builds fake observed maps as signal template at each oscillation point plus the corresponding background template, then fits that map against the solar and reactor reference templates with free signal and background normalizations.
- The fit in [lib/lib_root.py](../lib/lib_root.py) minimizes a Poisson deviance-like objective, not a generic least-squares chi-square.
- For observed count $o_i$ and expected count $e_i$, the per-bin contribution is $2(e_i - o_i + o_i \log(o_i / e_i))$ for $o_i > 0$ and $2e_i$ for $o_i = 0$, plus quadratic penalty terms on the fitted signal and background normalization shifts.
- The saved grid values are the minimized fit objective returned by `Sensitivity_Fitter.Fit`; contour labels may display $\sqrt{\chi^2}$ as a visualization proxy, but the workflow fundamentally stores the minimized deviance / chi-square-like statistic itself.

---

## Main Result: Contour Grids (sin12)

---

### HD Central

<div class="two-col">
  <div>
<p><strong>Solar Contour (sin12)</strong></p>
<img src="../images/analysis/sensitivity/hd_1x2x6_centralAPA/truncated/hd_1x2x6_centralAPA_solar_sin12_df_Truncated_SolarEnergy_NHits3_AdjCl2_OpHits10_Signal4_Bkg2.png">
  </div>
  <div>
<p><strong>Reactor Contour (sin12)</strong></p>
<img src="../images/analysis/sensitivity/hd_1x2x6_centralAPA/truncated/hd_1x2x6_centralAPA_react_sin12_df_Truncated_SolarEnergy_NHits3_AdjCl2_OpHits10_Signal4_Bkg2.png">
  </div>
</div>

---

### HD Lateral

<div class="two-col">
  <div>
<p><strong>Solar Contour (sin12)</strong></p>
<img src="../images/analysis/sensitivity/hd_1x2x6_lateralAPA/truncated/hd_1x2x6_lateralAPA_solar_sin12_df_Truncated_SolarEnergy_NHits3_AdjCl2_OpHits10_Signal4_Bkg2.png">
  </div>
  <div>
<p><strong>Reactor Contour (sin12)</strong></p>
<img src="../images/analysis/sensitivity/hd_1x2x6_lateralAPA/truncated/hd_1x2x6_lateralAPA_react_sin12_df_Truncated_SolarEnergy_NHits3_AdjCl2_OpHits10_Signal4_Bkg2.png">
  </div>
</div>

---

### VD Top

<div class="two-col">
  <div>
<p><strong>Solar Contour (sin12)</strong></p>
<img src="../images/analysis/sensitivity/vd_1x8x14_3view_30deg_nominal/truncated/vd_1x8x14_3view_30deg_nominal_solar_sin12_df_Truncated_SolarEnergy_NHits8_AdjCl8_OpHits10_Signal4_Bkg2.png">
  </div>
  <div>
<p><strong>Reactor Contour (sin12)</strong></p>
<img src="../images/analysis/sensitivity/vd_1x8x14_3view_30deg_nominal/truncated/vd_1x8x14_3view_30deg_nominal_react_sin12_df_Truncated_SolarEnergy_NHits8_AdjCl8_OpHits10_Signal4_Bkg2.png">
  </div>
</div>

---

### VD Bottom Shielded

<div class="two-col">
  <div>
<p><strong>Solar Contour (sin12)</strong></p>
<img src="../images/analysis/sensitivity/vd_1x8x14_3view_30deg_shielded/truncated/vd_1x8x14_3view_30deg_shielded_solar_sin12_df_Truncated_SolarEnergy_NHits8_AdjCl10_OpHits10_Signal4_Bkg2.png">
  </div>
  <div>
<p><strong>Reactor Contour (sin12)</strong></p>
<img src="../images/analysis/sensitivity/vd_1x8x14_3view_30deg_shielded/truncated/vd_1x8x14_3view_30deg_shielded_react_sin12_df_Truncated_SolarEnergy_NHits8_AdjCl10_OpHits10_Signal4_Bkg2.png">
  </div>
</div>

---

## Contour Grids (sin13)

---

### HD Central

<div class="two-col">
  <div>
<p><strong>Solar Contour (sin13)</strong></p>
<img src="../images/analysis/sensitivity/hd_1x2x6_centralAPA/truncated/hd_1x2x6_centralAPA_solar_sin13_df_Truncated_SolarEnergy_NHits3_AdjCl2_OpHits10_Signal4_Bkg2.png">
  </div>
  <div>
<p><strong>Reactor Contour (sin13)</strong></p>
<img src="../images/analysis/sensitivity/hd_1x2x6_centralAPA/truncated/hd_1x2x6_centralAPA_react_sin13_df_Truncated_SolarEnergy_NHits3_AdjCl2_OpHits10_Signal4_Bkg2.png">
  </div>
</div>

---

### HD Lateral

<div class="two-col">
  <div>
<p><strong>Solar Contour (sin13)</strong></p>
<img src="../images/analysis/sensitivity/hd_1x2x6_lateralAPA/truncated/hd_1x2x6_lateralAPA_solar_sin13_df_Truncated_SolarEnergy_NHits3_AdjCl2_OpHits10_Signal4_Bkg2.png">
  </div>
  <div>
<p><strong>Reactor Contour (sin13)</strong></p>
<img src="../images/analysis/sensitivity/hd_1x2x6_lateralAPA/truncated/hd_1x2x6_lateralAPA_react_sin13_df_Truncated_SolarEnergy_NHits3_AdjCl2_OpHits10_Signal4_Bkg2.png">
  </div>
</div>

---

### VD Top

<div class="two-col">
  <div>
<p><strong>Solar Contour (sin13)</strong></p>
<img src="../images/analysis/sensitivity/vd_1x8x14_3view_30deg_nominal/truncated/vd_1x8x14_3view_30deg_nominal_solar_sin13_df_Truncated_SolarEnergy_NHits8_AdjCl8_OpHits10_Signal4_Bkg2.png">
  </div>
  <div>
<p><strong>Reactor Contour (sin13)</strong></p>
<img src="../images/analysis/sensitivity/vd_1x8x14_3view_30deg_nominal/truncated/vd_1x8x14_3view_30deg_nominal_react_sin13_df_Truncated_SolarEnergy_NHits8_AdjCl8_OpHits10_Signal4_Bkg2.png">
  </div>
</div>

---

### VD Bottom Shielded

<div class="two-col">
  <div>
<p><strong>Solar Contour (sin13)</strong></p>
<img src="../images/analysis/sensitivity/vd_1x8x14_3view_30deg_shielded/truncated/vd_1x8x14_3view_30deg_shielded_solar_sin13_df_Truncated_SolarEnergy_NHits8_AdjCl10_OpHits10_Signal4_Bkg2.png">
  </div>
  <div>
<p><strong>Reactor Contour (sin13)</strong></p>
<img src="../images/analysis/sensitivity/vd_1x8x14_3view_30deg_shielded/truncated/vd_1x8x14_3view_30deg_shielded_react_sin13_df_Truncated_SolarEnergy_NHits8_AdjCl10_OpHits10_Signal4_Bkg2.png">
  </div>
</div>

---

## Template Building

---

### HD Central Background Template

<div class="center">
  <img src="../images/analysis/sensitivity/templates/truncated/hd_1x2x6_centralAPA/marley/hd_1x2x6_centralAPA_marley_Sensitivity_Templates_SolarEnergy_NHits3_AdjCl2_OpHits10.png">
</div>

---

### HD Central Signal Template

<div class="center">
  <img src="../images/analysis/sensitivity/templates/truncated/hd_1x2x6_centralAPA/hd_1x2x6_centralAPA_Selected_Signal_SolarEnergy_NHits3_AdjCl2_OpHits10.png">
</div>

---

### HD Lateral Background Template

<div class="center">
  <img src="../images/analysis/sensitivity/templates/truncated/hd_1x2x6_lateralAPA/marley/hd_1x2x6_lateralAPA_marley_Sensitivity_Templates_SolarEnergy_NHits3_AdjCl2_OpHits10.png">
</div>

---

### HD Lateral Signal Template

<div class="center">
  <img src="../images/analysis/sensitivity/templates/truncated/hd_1x2x6_lateralAPA/hd_1x2x6_lateralAPA_Selected_Signal_SolarEnergy_NHits3_AdjCl2_OpHits10.png">
</div>

---

### VD Top Background Template

<div class="center">
  <img src="../images/analysis/sensitivity/templates/truncated/vd_1x8x14_3view_30deg_nominal/marley/vd_1x8x14_3view_30deg_nominal_marley_Sensitivity_Templates_SolarEnergy_NHits8_AdjCl8_OpHits10.png">
</div>

---

### VD Top Signal Template

<div class="center">
  <img src="../images/analysis/sensitivity/templates/truncated/vd_1x8x14_3view_30deg_nominal/vd_1x8x14_3view_30deg_nominal_Selected_Signal_SolarEnergy_NHits8_AdjCl8_OpHits10.png">
</div>

---

### VD Bottom Shielded Background Template

<div class="center">
  <img src="../images/analysis/sensitivity/templates/truncated/vd_1x8x14_3view_30deg_shielded/marley/vd_1x8x14_3view_30deg_shielded_marley_Sensitivity_Templates_SolarEnergy_NHits8_AdjCl10_OpHits10.png">
</div>

---

### VD Bottom Shielded Signal Template

<div class="center">
  <img src="../images/analysis/sensitivity/templates/truncated/vd_1x8x14_3view_30deg_shielded/vd_1x8x14_3view_30deg_shielded_Selected_Signal_SolarEnergy_NHits8_AdjCl10_OpHits10.png">
</div>

---

### Selected Sensitivity Cuts by Config

| Config | NHits | OpHits | AdjCl | Signal Unc. (%) | Bkg Unc. (%) |
|---|---:|---:|---:|---:|---:|
| HD Central | 3 | 10 | 2 | 4 | 2 |
| HD Lateral | 3 | 10 | 2 | 4 | 2 |
| VD Top | 8 | 10 | 8 | 4 | 2 |
| VD Bottom Shielded | 8 | 10 | 10 | 4 | 2 |


---

## Coverage and Notes

- Configs with selected sin12 solar contour plots:
- truncated: 4
- Cut table values are parsed from selected result filenames when available.
- Re-run script to refresh this folder after each workflow run:
- /usr/bin/python3 scripts/generate_sensitivity_presentation.py --folder truncated
