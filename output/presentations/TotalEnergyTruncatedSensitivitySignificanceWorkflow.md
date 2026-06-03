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
- This deck is scoped to the **Truncated** folder for the **TotalEnergy** reconstruction algorithm.

Config aliases:
- hd_1x2x6_centralAPA: HD Central
- hd_1x2x6_lateralAPA: HD Lateral
- vd_1x8x14_3view_30deg_nominal: VD Top
- vd_1x8x14_3view_30deg_shielded: VD Bottom Shielded

---

### Workflow

- Orchestrator: [src/analysis/10SensitivityAnalysis.py](../../src/analysis/10SensitivityAnalysis.py)
- Step 1 (Background template): [src/analysis/14SensitivityBackgroundTemplate.py](../../src/analysis/14SensitivityBackgroundTemplate.py)
- Step 2 (Signal template): [src/analysis/14SensitivitySignalTemplate.py](../../src/analysis/14SensitivitySignalTemplate.py)
- Step 3 (Grid fit scan and best-cut storage): [src/analysis/14Sensitivity.py](../../src/analysis/14Sensitivity.py)
- Step 4 (Contour rendering): [src/analysis/14SensitivityContourPlot.py](../../src/analysis/14SensitivityContourPlot.py)

---

### Workflow Outputs

- Main contour grids (sin12/sin13): [images/analysis/sensitivity](../../images/analysis/sensitivity)
- Signal/background templates (figures): [images/analysis/sensitivity/templates](../../images/analysis/sensitivity/templates)
- Grid-scan data products (PKL): [data/analysis/sensitivity](../../data/analysis/sensitivity)
- Remote workflow outputs (PNFS): [/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/SENSITIVITY](/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/SENSITIVITY)

---

### 2D Template Construction

Signal and background are represented as 2D histograms with axes **(reconstructed neutrino energy × nadir cos(η))**.

For each oscillation point $(\Delta m^2,\, \sin^2\theta_{13},\, \sin^2\theta_{12})$, the signal template is built by convolving the detector energy-response matrix $H$ with the oscillation-probability matrix $P$:
$$
T^{\mathrm{sig}}_{ij}(\vec{\theta}) = T \cdot M_{\mathrm{det}} \cdot \left[ P(\vec{\theta})\, H \right]_{ij}
$$
where $i$ indexes nadir bins and $j$ indexes energy bins ([src/analysis/14SensitivitySignalTemplate.py](../../src/analysis/14SensitivitySignalTemplate.py)).

The background template $T^{\mathrm{bkg}}_{ij}$ is independent of oscillation parameters ([src/analysis/14SensitivityBackgroundTemplate.py](../../src/analysis/14SensitivityBackgroundTemplate.py)).

---

### Asimov Grid Construction

For each oscillation point $\vec{\theta}_k$ in the scan grid, the **Asimov (fake) observed dataset** is:
$$
o_{ij}(\vec{\theta}_k) = T^{\mathrm{sig}}_{ij}(\vec{\theta}_k) + T^{\mathrm{bkg}}_{ij}
$$
Same Asimov construction as the HEP profile-likelihood (*Background Normalization Model* slide): no statistical fluctuations, expected sensitivity in the median experiment.

Two **reference templates** are fixed at the solar and reactor best-fit oscillation points:
$$
p^{\mathrm{solar}}_{ij} = T^{\mathrm{sig}}_{ij}(\vec{\theta}_{\mathrm{solar}}), \qquad p^{\mathrm{react}}_{ij} = T^{\mathrm{sig}}_{ij}(\vec{\theta}_{\mathrm{react}})
$$

---

### Objective Function (Poisson Deviance)

The fit minimizes a **Baker-Cousins Poisson deviance** ([Baker & Cousins 1984](https://doi.org/10.1016/0029-554X(84)90016-4)) with two free normalization nuisances $A_{\mathrm{pred}},\, A_{\mathrm{bkg}}$:
$$
\chi^2(A_{\mathrm{pred}}, A_{\mathrm{bkg}}) = 2\sum_{i,j} \Delta\ell_{ij} + \left(\frac{A_{\mathrm{pred}}}{\sigma_{\mathrm{pred}}}\right)^{\!2} + \left(\frac{A_{\mathrm{bkg}}}{\sigma_{\mathrm{bkg}}}\right)^{\!2}
$$
Expected model: $e_{ij} = (1+A_{\mathrm{bkg}})\,T^{\mathrm{bkg}}_{ij} + (1+A_{\mathrm{pred}})\,p_{ij}$.
Per-bin deviance: $\Delta\ell_{ij} = e_{ij} - o_{ij} + o_{ij}\ln(o_{ij}/e_{ij})$ for $o_{ij}>0$, else $\Delta\ell_{ij} = e_{ij}$.

Implemented in [lib/lib_root.py: Sensitivity_Fitter](../../lib/lib_root.py). Minimized with [iminuit (Minuit)](https://iminuit.readthedocs.io/en/stable/).

---

### Nuisance Parameters: Comparison with HEP

Both analyses use **Gaussian-constrained normalization nuisances** added to the Poisson deviance:

| Feature | HEP Profile-Likelihood | Sensitivity |
|---|---|---|
| Histogram | 1D (energy) | 2D (energy × nadir) |
| Goal | Discovery significance | $\chi^2$ map over $(\Delta m^2, \sin^2\theta)$ |
| Nuisances | 1 global $\beta$ (background) | $A_{\mathrm{pred}} + A_{\mathrm{bkg}}$ |
| Solution | Closed-form quadratic ([Cowan 2010](https://arxiv.org/abs/1007.1727)) | scipy L-BFGS-B (joint 2D) |
| Deviance | $2\sum_i [n_i \ln(n_i/\hat{\beta}b_i) - (n_i - \hat{\beta}b_i)]$ | $2\sum_{ij} [e_{ij} - o_{ij} + o_{ij}\ln(o_{ij}/e_{ij})]$ |
| Penalty | $[(\hat{\beta}-1)/\sigma_{\mathrm{rel}}]^2$ | $(A_{\mathrm{pred}}/\sigma_{\mathrm{pred}})^2 + (A_{\mathrm{bkg}}/\sigma_{\mathrm{bkg}})^2$ |
| MC mask | Barlow-Beeston (static) | BB mask: bins with bkg template = 0 excluded |

---

### Oscillation Grid Scan and Best-Cut Score

For each analysis cut $(N_{\mathrm{hits}}, N_{\mathrm{ophits}}, N_{\mathrm{adjcl}})$ and each oscillation point $\vec{\theta}_k$:
1. Build Asimov dataset $o_{ij}(\vec{\theta}_k)$.
2. Fit against **solar** reference template → $\chi^2_{\mathrm{solar}}(\vec{\theta}_k)$.
3. Fit against **reactor** reference template → $\chi^2_{\mathrm{react}}(\vec{\theta}_k)$.

The **cut quality score** is the average $\chi^2$ when fitting with the *wrong* hypothesis:
$$
\mathrm{Score} = \tfrac{1}{2}\left[\chi^2_{\mathrm{solar}}(\vec{\theta}_{\mathrm{react}}) + \chi^2_{\mathrm{react}}(\vec{\theta}_{\mathrm{solar}})\right]
$$
Higher score = better discrimination between solar and reactor hypotheses. The best cut maximizes this.

---

### Implemented Improvements

Improvements 2–5 implemented in [lib/lib_root.py](../../lib/lib_root.py) and [src/analysis/14Sensitivity.py](../../src/analysis/14Sensitivity.py):

1. **Replace heuristic score with profile-LR** *(proposed, not yet implemented)*: use $\Delta\chi^2 = \chi^2_{\mathrm{null}} - \chi^2_{\mathrm{best}}$ and report $Z = \sqrt{\Delta\chi^2}$ (Wilks theorem) instead of average cross-hypothesis $\chi^2$.
2. ✅ **Barlow-Beeston mask** (`bb_mask = bkg > 0`): bins where the background template is zero are excluded from the fit, preventing spurious large deviance contributions from zero-MC-support bins.
3. ✅ **Removed `abs()`** from `ROOTOperator` and `NumpyOperator`: the Baker-Cousins deviance is always $\ge 0$ at the minimum; `abs()` distorts gradients and can impair Minuit convergence.
4. ✅ **Tightened parameter limits**: $\pm 100\sigma \to \pm 10\sigma$, reducing search space and avoiding minimization in flat tails.
5. ✅ **Replaced Minuit with scipy L-BFGS-B** (joint 2D): `scipy.optimize.minimize(..., method="L-BFGS-B")` minimizes over $(A_{\mathrm{pred}}, A_{\mathrm{bkg}})$ jointly with ±10$\sigma$ bounds. Uses gradient info → fewer function evaluations than Minuit for smooth convex objectives. `_profile_a_bkg` retained as `profile_bkg=True` option for comparison. No analytic closed form exists for either nuisance (unlike HEP's $\hat{\beta}$) due to per-bin coupling.

---

### Sensitivity Fit Summary

- [src/analysis/14Sensitivity.py](../../src/analysis/14Sensitivity.py) builds Asimov maps from signal + background templates, then fits each oscillation-grid point against solar and reactor reference templates with free normalizations.
- The fit minimizes the **Baker-Cousins Poisson deviance** ([Baker & Cousins 1984](https://doi.org/10.1016/0029-554X(84)90016-4)) — identical in form to the per-bin LLR in the HEP profile-likelihood, extended to 2D (energy × nadir).
- Penalty terms $(A_{\mathrm{pred}}/\sigma_{\mathrm{pred}})^2 + (A_{\mathrm{bkg}}/\sigma_{\mathrm{bkg}})^2$ play the same role as HEP's $[(\hat{\beta}-1)/\sigma_{\mathrm{rel}}]^2$; HEP solves analytically, Sensitivity solves numerically.
- Improvements 2–5 implemented in [lib/lib_root.py](../../lib/lib_root.py): BB mask, no `abs()`, ±10σ limits, scipy L-BFGS-B joint 2D minimization.
- Full mathematical derivations: [docs/hep\_likelihood\_derivation.tex](../../docs/hep_likelihood_derivation.tex).

---

## Fiducialization

---

### Fiducial plots

No fiducial optimization plots were found for this folder.

---

### Fiducial Optimization Summary

| Config | Fiducial X | Fiducial Y | Fiducial Z | Before Fiducialization | After Fiducialization | Fiducial Mass (kt) |
|---|---:|---:|---:|---:|---:|---:|
| HD Central | 100 | 140 | 140 | 0.001 | 0.225 | 2.99 |
| HD Lateral | 0 | 120 | 80 | 0.001 | 0.002 | 4.79 |
| VD Top | 20 | 140 | 180 | 0.007 | 0.013 | 4.99 |
| VD Bottom Shielded | 120 | 300 | 20 | 0.003 | 0.013 | 3.49 |

---

## Main Result: Contour Grids (sin12)

---

### HD Central

<div class="two-col">
  <div>
<p><strong>Solar Contour (sin12)</strong></p>
<img src="../../images/analysis/sensitivity/hd_1x2x6_centralAPA/marley/truncated/energy_scale/hd_1x2x6_centralAPA_marley_solar_sin12_df_Truncated_TotalEnergy_NHits1_AdjCl4_OpHits8_Signal4_Bkg2.png">
  </div>
  <div>
<p><strong>Reactor Contour (sin12)</strong></p>
<img src="../../images/analysis/sensitivity/hd_1x2x6_centralAPA/marley/truncated/energy_scale/hd_1x2x6_centralAPA_marley_react_sin12_df_Truncated_TotalEnergy_NHits1_AdjCl4_OpHits8_Signal4_Bkg2.png">
  </div>
</div>

---

### HD Lateral

<div class="two-col">
  <div>
<p><strong>Solar Contour (sin12)</strong></p>
<img src="../../images/analysis/sensitivity/hd_1x2x6_lateralAPA/marley/truncated/energy_scale/hd_1x2x6_lateralAPA_marley_solar_sin12_df_Truncated_TotalEnergy_NHits3_AdjCl2_OpHits10_Signal4_Bkg2.png">
  </div>
  <div>
<p><strong>Reactor Contour (sin12)</strong></p>
<img src="../../images/analysis/sensitivity/hd_1x2x6_lateralAPA/marley/truncated/energy_scale/hd_1x2x6_lateralAPA_marley_react_sin12_df_Truncated_TotalEnergy_NHits3_AdjCl2_OpHits10_Signal4_Bkg2.png">
  </div>
</div>

---

### VD Top

<div class="two-col">
  <div>
<p><strong>Solar Contour (sin12)</strong></p>
<img src="../../images/analysis/sensitivity/vd_1x8x14_3view_30deg_nominal/marley/truncated/energy_scale/vd_1x8x14_3view_30deg_nominal_marley_solar_sin12_df_Truncated_TotalEnergy_NHits8_AdjCl8_OpHits10_Signal4_Bkg2.png">
  </div>
  <div>
<p><strong>Reactor Contour (sin12)</strong></p>
<img src="../../images/analysis/sensitivity/vd_1x8x14_3view_30deg_nominal/marley/truncated/energy_scale/vd_1x8x14_3view_30deg_nominal_marley_react_sin12_df_Truncated_TotalEnergy_NHits8_AdjCl8_OpHits10_Signal4_Bkg2.png">
  </div>
</div>

---

### VD Bottom Shielded

<div class="two-col">
  <div>
<p><strong>Solar Contour (sin12)</strong></p>
<img src="../../images/analysis/sensitivity/vd_1x8x14_3view_30deg_shielded/marley/truncated/energy_scale/vd_1x8x14_3view_30deg_shielded_marley_solar_sin12_df_Truncated_TotalEnergy_NHits8_AdjCl10_OpHits10_Signal4_Bkg2.png">
  </div>
  <div>
<p><strong>Reactor Contour (sin12)</strong></p>
<img src="../../images/analysis/sensitivity/vd_1x8x14_3view_30deg_shielded/marley/truncated/energy_scale/vd_1x8x14_3view_30deg_shielded_marley_react_sin12_df_Truncated_TotalEnergy_NHits8_AdjCl10_OpHits10_Signal4_Bkg2.png">
  </div>
</div>

---

## Contour Grids (sin13)

---

### HD Central

<div class="two-col">
  <div>
<p><strong>Solar Contour (sin13)</strong></p>
<img src="../../images/analysis/sensitivity/hd_1x2x6_centralAPA/marley/truncated/energy_scale/hd_1x2x6_centralAPA_marley_solar_sin13_df_Truncated_TotalEnergy_NHits1_AdjCl4_OpHits8_Signal4_Bkg2.png">
  </div>
  <div>
<p><strong>Reactor Contour (sin13)</strong></p>
<img src="../../images/analysis/sensitivity/hd_1x2x6_centralAPA/marley/truncated/energy_scale/hd_1x2x6_centralAPA_marley_react_sin13_df_Truncated_TotalEnergy_NHits1_AdjCl4_OpHits8_Signal4_Bkg2.png">
  </div>
</div>

---

### HD Lateral

<div class="two-col">
  <div>
<p><strong>Solar Contour (sin13)</strong></p>
<img src="../../images/analysis/sensitivity/hd_1x2x6_lateralAPA/marley/truncated/energy_scale/hd_1x2x6_lateralAPA_marley_solar_sin13_df_Truncated_TotalEnergy_NHits3_AdjCl2_OpHits10_Signal4_Bkg2.png">
  </div>
  <div>
<p><strong>Reactor Contour (sin13)</strong></p>
<img src="../../images/analysis/sensitivity/hd_1x2x6_lateralAPA/marley/truncated/energy_scale/hd_1x2x6_lateralAPA_marley_react_sin13_df_Truncated_TotalEnergy_NHits3_AdjCl2_OpHits10_Signal4_Bkg2.png">
  </div>
</div>

---

### VD Top

<div class="two-col">
  <div>
<p><strong>Solar Contour (sin13)</strong></p>
<img src="../../images/analysis/sensitivity/vd_1x8x14_3view_30deg_nominal/marley/truncated/energy_scale/vd_1x8x14_3view_30deg_nominal_marley_solar_sin13_df_Truncated_TotalEnergy_NHits8_AdjCl8_OpHits10_Signal4_Bkg2.png">
  </div>
  <div>
<p><strong>Reactor Contour (sin13)</strong></p>
<img src="../../images/analysis/sensitivity/vd_1x8x14_3view_30deg_nominal/marley/truncated/energy_scale/vd_1x8x14_3view_30deg_nominal_marley_react_sin13_df_Truncated_TotalEnergy_NHits8_AdjCl8_OpHits10_Signal4_Bkg2.png">
  </div>
</div>

---

### VD Bottom Shielded

<div class="two-col">
  <div>
<p><strong>Solar Contour (sin13)</strong></p>
<img src="../../images/analysis/sensitivity/vd_1x8x14_3view_30deg_shielded/marley/truncated/energy_scale/vd_1x8x14_3view_30deg_shielded_marley_solar_sin13_df_Truncated_TotalEnergy_NHits8_AdjCl10_OpHits10_Signal4_Bkg2.png">
  </div>
  <div>
<p><strong>Reactor Contour (sin13)</strong></p>
<img src="../../images/analysis/sensitivity/vd_1x8x14_3view_30deg_shielded/marley/truncated/energy_scale/vd_1x8x14_3view_30deg_shielded_marley_react_sin13_df_Truncated_TotalEnergy_NHits8_AdjCl10_OpHits10_Signal4_Bkg2.png">
  </div>
</div>

---

## Significance Spectra

---

### HD Central

<div class="center">
  <img src="../../images/analysis/sensitivity/hd_1x2x6_centralAPA/marley/truncated/hd_1x2x6_centralAPA_marley_TotalEnergy_Sensitivity_Significance_Exposure_30.png">
</div>

---

### HD Lateral

<div class="center">
  <img src="../../images/analysis/sensitivity/hd_1x2x6_lateralAPA/marley/truncated/hd_1x2x6_lateralAPA_marley_TotalEnergy_Sensitivity_Significance_Exposure_30.png">
</div>

---

### VD Top

<div class="center">
  <img src="../../images/analysis/sensitivity/vd_1x8x14_3view_30deg_nominal/marley/truncated/vd_1x8x14_3view_30deg_nominal_marley_TotalEnergy_Sensitivity_Significance_Exposure_30.png">
</div>

---

### VD Bottom Shielded

<div class="center">
  <img src="../../images/analysis/sensitivity/vd_1x8x14_3view_30deg_shielded/marley/truncated/vd_1x8x14_3view_30deg_shielded_marley_TotalEnergy_Sensitivity_Significance_Exposure_30.png">
</div>

---

## Template Building

---

### HD Central Templates

<div class="center">
  <img src="../../images/analysis/sensitivity/templates/truncated/hd_1x2x6_centralAPA/marley/hd_1x2x6_centralAPA_marley_Sensitivity_Templates_TotalEnergy_NHits1_AdjCl4_OpHits8.png">
</div>

---

### HD Lateral Templates

<div class="center">
  <img src="../../images/analysis/sensitivity/templates/truncated/hd_1x2x6_lateralAPA/marley/hd_1x2x6_lateralAPA_marley_Sensitivity_Templates_TotalEnergy_NHits3_AdjCl2_OpHits10.png">
</div>

---

### VD Top Templates

<div class="center">
  <img src="../../images/analysis/sensitivity/templates/truncated/vd_1x8x14_3view_30deg_nominal/marley/vd_1x8x14_3view_30deg_nominal_marley_Sensitivity_Templates_TotalEnergy_NHits8_AdjCl8_OpHits10.png">
</div>

---

### VD Bottom Shielded Templates

<div class="center">
  <img src="../../images/analysis/sensitivity/templates/truncated/vd_1x8x14_3view_30deg_shielded/marley/vd_1x8x14_3view_30deg_shielded_marley_Sensitivity_Templates_TotalEnergy_NHits8_AdjCl10_OpHits10.png">
</div>

---

### Selected Sensitivity Cuts by Config

| Config | NHits | OpHits | AdjCl | Signal Unc. (%) | Bkg Unc. (%) | 1D Asimov Z (σ) |
|---|---:|---:|---:|---:|---:|---:|
| HD Central | 1 | 8 | 4 | 4 | 2 | 22.30 |
| HD Lateral | 3 | 10 | 2 | 4 | 2 | 18.83 |
| VD Top | 8 | 10 | 8 | 4 | 2 | 15.18 |
| VD Bottom Shielded | 8 | 10 | 10 | 4 | 2 | 70.29 |


---

## Nuisance Parameter Comparison

---

### Profile Settings

| Profile | Settings |
|---|---|
| **energy_scale** | MARGINALIZE_SIN13=False, ENERGY_SCALE_UNCERTAINTY=True |
| **full** | MARGINALIZE_SIN13=True, ENERGY_SCALE_UNCERTAINTY=True |

---

### energy_scale vs full

---

### HD Central: Solar sin²θ₁₂

<div class="two-col">
  <div>
<p><strong>energy_scale</strong></p>
<img src="../../images/analysis/sensitivity/hd_1x2x6_centralAPA/marley/truncated/energy_scale/hd_1x2x6_centralAPA_marley_solar_sin12_df_Truncated_TotalEnergy_NHits1_AdjCl4_OpHits8_Signal4_Bkg2.png">
  </div>
  <div>
<p><strong>full</strong></p>
<img src="../../images/analysis/sensitivity/hd_1x2x6_centralAPA/marley/truncated/full/hd_1x2x6_centralAPA_marley_solar_sin12_df_Truncated_TotalEnergy_NHits1_AdjCl4_OpHits8_Signal4_Bkg2.png">
  </div>
</div>

---

### HD Central: Reactor sin²θ₁₂

<div class="two-col">
  <div>
<p><strong>energy_scale</strong></p>
<img src="../../images/analysis/sensitivity/hd_1x2x6_centralAPA/marley/truncated/energy_scale/hd_1x2x6_centralAPA_marley_react_sin12_df_Truncated_TotalEnergy_NHits1_AdjCl4_OpHits8_Signal4_Bkg2.png">
  </div>
  <div>
<p><strong>full</strong></p>
<img src="../../images/analysis/sensitivity/hd_1x2x6_centralAPA/marley/truncated/full/hd_1x2x6_centralAPA_marley_react_sin12_df_Truncated_TotalEnergy_NHits1_AdjCl4_OpHits8_Signal4_Bkg2.png">
  </div>
</div>

---

### HD Lateral: Solar sin²θ₁₂

<div class="two-col">
  <div>
<p><strong>energy_scale</strong></p>
<img src="../../images/analysis/sensitivity/hd_1x2x6_lateralAPA/marley/truncated/energy_scale/hd_1x2x6_lateralAPA_marley_solar_sin12_df_Truncated_TotalEnergy_NHits3_AdjCl2_OpHits10_Signal4_Bkg2.png">
  </div>
  <div>
<p><strong>full</strong></p>
<img src="../../images/analysis/sensitivity/hd_1x2x6_lateralAPA/marley/truncated/full/hd_1x2x6_lateralAPA_marley_solar_sin12_df_Truncated_TotalEnergy_NHits3_AdjCl2_OpHits10_Signal4_Bkg2.png">
  </div>
</div>

---

### HD Lateral: Reactor sin²θ₁₂

<div class="two-col">
  <div>
<p><strong>energy_scale</strong></p>
<img src="../../images/analysis/sensitivity/hd_1x2x6_lateralAPA/marley/truncated/energy_scale/hd_1x2x6_lateralAPA_marley_react_sin12_df_Truncated_TotalEnergy_NHits3_AdjCl2_OpHits10_Signal4_Bkg2.png">
  </div>
  <div>
<p><strong>full</strong></p>
<img src="../../images/analysis/sensitivity/hd_1x2x6_lateralAPA/marley/truncated/full/hd_1x2x6_lateralAPA_marley_react_sin12_df_Truncated_TotalEnergy_NHits3_AdjCl2_OpHits10_Signal4_Bkg2.png">
  </div>
</div>

---

### VD Top: Solar sin²θ₁₂

<div class="two-col">
  <div>
<p><strong>energy_scale</strong></p>
<img src="../../images/analysis/sensitivity/vd_1x8x14_3view_30deg_nominal/marley/truncated/energy_scale/vd_1x8x14_3view_30deg_nominal_marley_solar_sin12_df_Truncated_TotalEnergy_NHits8_AdjCl8_OpHits10_Signal4_Bkg2.png">
  </div>
  <div>
<p><strong>full</strong></p>
<img src="../../images/analysis/sensitivity/vd_1x8x14_3view_30deg_nominal/marley/truncated/full/vd_1x8x14_3view_30deg_nominal_marley_solar_sin12_df_Truncated_TotalEnergy_NHits8_AdjCl8_OpHits10_Signal4_Bkg2.png">
  </div>
</div>

---

### VD Top: Reactor sin²θ₁₂

<div class="two-col">
  <div>
<p><strong>energy_scale</strong></p>
<img src="../../images/analysis/sensitivity/vd_1x8x14_3view_30deg_nominal/marley/truncated/energy_scale/vd_1x8x14_3view_30deg_nominal_marley_react_sin12_df_Truncated_TotalEnergy_NHits8_AdjCl8_OpHits10_Signal4_Bkg2.png">
  </div>
  <div>
<p><strong>full</strong></p>
<img src="../../images/analysis/sensitivity/vd_1x8x14_3view_30deg_nominal/marley/truncated/full/vd_1x8x14_3view_30deg_nominal_marley_react_sin12_df_Truncated_TotalEnergy_NHits8_AdjCl8_OpHits10_Signal4_Bkg2.png">
  </div>
</div>

---

### VD Bottom Shielded: Solar sin²θ₁₂

<div class="two-col">
  <div>
<p><strong>energy_scale</strong></p>
<img src="../../images/analysis/sensitivity/vd_1x8x14_3view_30deg_shielded/marley/truncated/energy_scale/vd_1x8x14_3view_30deg_shielded_marley_solar_sin12_df_Truncated_TotalEnergy_NHits8_AdjCl10_OpHits10_Signal4_Bkg2.png">
  </div>
  <div>
<p><strong>full</strong></p>
<img src="../../images/analysis/sensitivity/vd_1x8x14_3view_30deg_shielded/marley/truncated/full/vd_1x8x14_3view_30deg_shielded_marley_solar_sin12_df_Truncated_TotalEnergy_NHits8_AdjCl10_OpHits10_Signal4_Bkg2.png">
  </div>
</div>

---

### VD Bottom Shielded: Reactor sin²θ₁₂

<div class="two-col">
  <div>
<p><strong>energy_scale</strong></p>
<img src="../../images/analysis/sensitivity/vd_1x8x14_3view_30deg_shielded/marley/truncated/energy_scale/vd_1x8x14_3view_30deg_shielded_marley_react_sin12_df_Truncated_TotalEnergy_NHits8_AdjCl10_OpHits10_Signal4_Bkg2.png">
  </div>
  <div>
<p><strong>full</strong></p>
<img src="../../images/analysis/sensitivity/vd_1x8x14_3view_30deg_shielded/marley/truncated/full/vd_1x8x14_3view_30deg_shielded_marley_react_sin12_df_Truncated_TotalEnergy_NHits8_AdjCl10_OpHits10_Signal4_Bkg2.png">
  </div>
</div>

---

## Coverage and Notes

- Configs with selected sin12 solar contour plots:
- truncated: 4
- Cut table values are parsed from selected result filenames when available.
- Re-run script to refresh this folder after each workflow run:
- /usr/bin/python3 scripts/generate_sensitivity_presentation.py --folder truncated
- Full mathematical derivations: [docs/hep\_likelihood\_derivation.tex](../../docs/hep_likelihood_derivation.tex)
