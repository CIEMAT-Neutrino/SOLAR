---
marp: true
description: Inputs, workflow outputs, and per-config Sensitivity results
paginate: true
theme: dune
---

<!-- AUTO-GENERATED: src/tools/presentations/sensitivity.py -->

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

- Orchestrator: [src/pipelines/run\_sensitivity.py](../../src/pipelines/run_sensitivity.py)
- Step 1 (Background template): [src/physics/sensitivity/01\_background\_template.py](../../src/physics/sensitivity/01_background_template.py)
- Step 2 (Signal template): [src/physics/sensitivity/02\_signal\_template.py](../../src/physics/sensitivity/02_signal_template.py)
- Step 3 (Grid fit scan and best-cut storage): [src/physics/sensitivity/04\_best\_cuts.py](../../src/physics/sensitivity/04_best_cuts.py)
- Step 4 (Contour rendering): [src/physics/sensitivity/contour\_plot.py](../../src/physics/sensitivity/contour_plot.py)

---

### Workflow Outputs

- Main contour grids (sin12/sin13): [output/images/analysis/sensitivity](../../output/images/analysis/sensitivity)
- Signal/background templates (figures): [output/images/analysis/sensitivity/templates](../../output/images/analysis/sensitivity/templates)
- Grid-scan data products (PKL): [data/analysis/sensitivity](../../data/analysis/sensitivity)
- Remote workflow outputs (PNFS): [/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/SENSITIVITY](/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/SENSITIVITY)

---

### 2D Template Construction

Signal and background are represented as 2D histograms with axes **(reconstructed neutrino energy × nadir cos(η))**.

For each oscillation point $(\Delta m^2,\, \sin^2\theta_{13},\, \sin^2\theta_{12})$, the signal template is built by convolving the detector energy-response matrix $H$ with the oscillation-probability matrix $P$:
$$
T^{\mathrm{sig}}_{ij}(\vec{\theta}) = T \cdot M_{\mathrm{det}} \cdot \left[ P(\vec{\theta})\, H \right]_{ij}
$$
where $i$ indexes nadir bins and $j$ indexes energy bins ([src/physics/sensitivity/02\_signal\_template.py](../../src/physics/sensitivity/02_signal_template.py)).

The background template $T^{\mathrm{bkg}}_{ij}$ is independent of oscillation parameters ([src/physics/sensitivity/01\_background\_template.py](../../src/physics/sensitivity/01_background_template.py)).

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

Implemented in [`lib/root.py: Sensitivity_Fitter`](../../lib/root.py). Default minimizer: **scipy L-BFGS-B** (joint 2D); ROOT TH2F input uses [iminuit (Minuit)](https://iminuit.readthedocs.io/en/stable/).

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
| Penalty | $[(\hat{\beta}-1)/\sigma_{\mathrm{rel}}]^2$ | Conditional (see next slide) |
| MC mask | Barlow-Beeston (static) | BB mask: bins with bkg template = 0 excluded |
| Nuisance disable | — | set $\sigma \le 0$ to drop that penalty term |

---

### Nuisance Parameter Model

The penalty term is applied **conditionally** based on each nuisance being active ($\sigma > 0$):

$$
\mathcal{P}(A_{\mathrm{pred}}, A_{\mathrm{bkg}}) = \begin{cases}
\left(\dfrac{A_{\mathrm{pred}}}{\sigma_{\mathrm{pred}}}\right)^{\!2} + \left(\dfrac{A_{\mathrm{bkg}}}{\sigma_{\mathrm{bkg}}}\right)^{\!2} & \sigma_{\mathrm{pred}} > 0 \text{ and } \sigma_{\mathrm{bkg}} > 0 \\[6pt]
\left(\dfrac{A_{\mathrm{bkg}}}{\sigma_{\mathrm{bkg}}}\right)^{\!2} & \sigma_{\mathrm{pred}} \le 0 \text{ and } \sigma_{\mathrm{bkg}} > 0 \\[6pt]
\left(\dfrac{A_{\mathrm{pred}}}{\sigma_{\mathrm{pred}}}\right)^{\!2} & \sigma_{\mathrm{pred}} > 0 \text{ and } \sigma_{\mathrm{bkg}} \le 0 \\[6pt]
0 & \text{otherwise}
\end{cases}
$$

Setting $\sigma \le 0$ **disables** that nuisance entirely — the corresponding $A$ is still a free parameter in the minimization but receives no Gaussian pull. Default: $\sigma_{\mathrm{pred}} = 4\%$, $\sigma_{\mathrm{bkg}} = 2\%$.

---

### Minimization Backends

**Three minimization backends** (selected automatically by input type):

| Backend | Input | Method |
|---|---|---|
| scipy L-BFGS-B | `np.ndarray` | Joint 2D over $(A_{\mathrm{pred}}, A_{\mathrm{bkg}})$ — **default** |
| Minuit 1D + profiled bkg | `np.ndarray` + `profile_bkg=True` | 1D Minuit over $A_{\mathrm{pred}}$; `minimize_scalar` at each step for $A_{\mathrm{bkg}}$ |
| Minuit 2D | `ROOT.TH2F` | Joint 2D Minuit |

Implemented in [`lib/root.py: Sensitivity_Fitter.NumpyOperator`](../../lib/root.py).

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

Improvements 2–5 implemented in [lib/root.py](../../lib/root.py) and [src/physics/sensitivity/04\_best\_cuts.py](../../src/physics/sensitivity/04_best_cuts.py):

1. **Replace heuristic score with profile-LR** *(proposed, not yet implemented)*: use $\Delta\chi^2 = \chi^2_{\mathrm{null}} - \chi^2_{\mathrm{best}}$ and report $Z = \sqrt{\Delta\chi^2}$ (Wilks theorem) instead of average cross-hypothesis $\chi^2$.
2. ✅ **Barlow-Beeston mask** (`bb_mask = bkg > 0`): bins where the background template is zero are excluded from the fit, preventing spurious large deviance contributions from zero-MC-support bins.
3. ✅ **Removed `abs()`** from `ROOTOperator` and `NumpyOperator`: the Baker-Cousins deviance is always $\ge 0$ at the minimum; `abs()` distorts gradients and can impair Minuit convergence.
4. ✅ **Tightened parameter limits**: $\pm 100\sigma \to \pm 10\sigma$, reducing search space and avoiding minimization in flat tails.
5. ✅ **Replaced Minuit with scipy L-BFGS-B** (joint 2D): `scipy.optimize.minimize(..., method="L-BFGS-B")` minimizes over $(A_{\mathrm{pred}}, A_{\mathrm{bkg}})$ jointly with ±10$\sigma$ bounds. Uses gradient info → fewer function evaluations than Minuit for smooth convex objectives. `_profile_a_bkg` retained as `profile_bkg=True` option for comparison. No analytic closed form exists for either nuisance (unlike HEP's $\hat{\beta}$) due to per-bin coupling.

---

### Sensitivity Fit Summary

- [src/physics/sensitivity/04\_best\_cuts.py](../../src/physics/sensitivity/04_best_cuts.py) builds Asimov maps from signal + background templates, then fits each oscillation-grid point against solar and reactor reference templates with free normalizations.
- The fit minimizes the **Baker-Cousins Poisson deviance** ([Baker & Cousins 1984](https://doi.org/10.1016/0029-554X(84)90016-4)) — identical in form to the per-bin LLR in the HEP profile-likelihood, extended to 2D (energy × nadir).
- Penalty terms are conditional on $\sigma > 0$; both active by default ($\sigma_{\mathrm{pred}}=4\%$, $\sigma_{\mathrm{bkg}}=2\%$). Set $\sigma \le 0$ to disable.
- Improvements 2–5 implemented in [lib/root.py](../../lib/root.py): BB mask, no `abs()`, ±10σ limits, scipy L-BFGS-B joint 2D minimization.
- Full mathematical derivations: [docs/hep\_likelihood\_derivation.tex](../../docs/hep_likelihood_derivation.tex).

---

## Fiducialization

---

### HD Central

<div class="two-col">
  <div>
<p><strong>No Fiducial</strong></p>
<img src="../../output/images/solar/fiducial/hd_1x2x6_centralAPA/marley/truncated/hd_1x2x6_centralAPA_marley_SolarEnergy_SENSITIVITY_NoFiducial_Significance.png">
  </div>
  <div>
<p><strong>Best Fiducial</strong></p>
<img src="../../output/images/solar/fiducial/hd_1x2x6_centralAPA/marley/truncated/hd_1x2x6_centralAPA_marley_SolarEnergy_SENSITIVITY_BestFiducial_Significance.png">
  </div>
</div>

---

### HD Lateral

<div class="two-col">
  <div>
<p><strong>No Fiducial</strong></p>
<img src="../../output/images/solar/fiducial/hd_1x2x6_lateralAPA/marley/truncated/hd_1x2x6_lateralAPA_marley_SolarEnergy_SENSITIVITY_NoFiducial_Significance.png">
  </div>
  <div>
<p><strong>Best Fiducial</strong></p>
<img src="../../output/images/solar/fiducial/hd_1x2x6_lateralAPA/marley/truncated/hd_1x2x6_lateralAPA_marley_SolarEnergy_SENSITIVITY_BestFiducial_Significance.png">
  </div>
</div>

---

### VD Top

<div class="two-col">
  <div>
<p><strong>No Fiducial</strong></p>
<img src="../../output/images/solar/fiducial/vd_1x8x14_3view_30deg_nominal/marley/truncated/vd_1x8x14_3view_30deg_nominal_marley_SolarEnergy_SENSITIVITY_NoFiducial_Significance.png">
  </div>
  <div>
<p><strong>Best Fiducial</strong></p>
<img src="../../output/images/solar/fiducial/vd_1x8x14_3view_30deg_nominal/marley/truncated/vd_1x8x14_3view_30deg_nominal_marley_SolarEnergy_SENSITIVITY_BestFiducial_Significance.png">
  </div>
</div>

---

### VD Bottom Shielded

<div class="two-col">
  <div>
<p><strong>No Fiducial</strong></p>
<img src="../../output/images/solar/fiducial/vd_1x8x14_3view_30deg_shielded/marley/truncated/vd_1x8x14_3view_30deg_shielded_marley_SolarEnergy_SENSITIVITY_NoFiducial_Significance.png">
  </div>
  <div>
<p><strong>Best Fiducial</strong></p>
<img src="../../output/images/solar/fiducial/vd_1x8x14_3view_30deg_shielded/marley/truncated/vd_1x8x14_3view_30deg_shielded_marley_SolarEnergy_SENSITIVITY_BestFiducial_Significance.png">
  </div>
</div>

---

### Fiducial Optimization Summary

| Config | Fiducial X | Fiducial Y | Fiducial Z | Before Fiducialization | After Fiducialization | Fiducial Mass (kt) |
|---|---:|---:|---:|---:|---:|---:|
| HD Central | 0 | 80 | 100 | 0.344 | 2.971 | 5.02 |
| HD Lateral | 60 | 260 | 200 | 0.097 | 2.746 | 2.28 |
| VD Top | 0 | 0 | 20 | 0.632 | 0.642 | 7.69 |
| VD Bottom Shielded | 0 | 0 | 20 | 0.703 | 0.704 | 7.69 |

---

## Main Result: Contour Grids (sin12)

---

### HD Central

<div class="two-col">
  <div>
<p><strong>Solar Contour (sin12)</strong></p>
<img src="../../output/images/analysis/sensitivity/hd_1x2x6_centralAPA/marley/truncated/full/hd_1x2x6_centralAPA_marley_solar_sin12_df_Truncated_SolarEnergy_NHits1_AdjCl4_OpHits8_Signal4_Bkg2.png">
  </div>
  <div>
<p><strong>Reactor Contour (sin12)</strong></p>
<img src="../../output/images/analysis/sensitivity/hd_1x2x6_centralAPA/marley/truncated/full/hd_1x2x6_centralAPA_marley_react_sin12_df_Truncated_SolarEnergy_NHits1_AdjCl4_OpHits8_Signal4_Bkg2.png">
  </div>
</div>

---

### HD Lateral

<div class="two-col">
  <div>
<p><strong>Solar Contour (sin12)</strong></p>
<img src="../../output/images/analysis/sensitivity/hd_1x2x6_lateralAPA/marley/truncated/full/hd_1x2x6_lateralAPA_marley_solar_sin12_df_Truncated_SolarEnergy_NHits3_AdjCl2_OpHits10_Signal4_Bkg2.png">
  </div>
  <div>
<p><strong>Reactor Contour (sin12)</strong></p>
<img src="../../output/images/analysis/sensitivity/hd_1x2x6_lateralAPA/marley/truncated/full/hd_1x2x6_lateralAPA_marley_react_sin12_df_Truncated_SolarEnergy_NHits3_AdjCl2_OpHits10_Signal4_Bkg2.png">
  </div>
</div>

---

### VD Top

<div class="two-col">
  <div>
<p><strong>Solar Contour (sin12)</strong></p>
<img src="../../output/images/analysis/sensitivity/vd_1x8x14_3view_30deg_nominal/marley/truncated/full/vd_1x8x14_3view_30deg_nominal_marley_solar_sin12_df_Truncated_SolarEnergy_NHits8_AdjCl8_OpHits10_Signal4_Bkg2.png">
  </div>
  <div>
<p><strong>Reactor Contour (sin12)</strong></p>
<img src="../../output/images/analysis/sensitivity/vd_1x8x14_3view_30deg_nominal/marley/truncated/full/vd_1x8x14_3view_30deg_nominal_marley_react_sin12_df_Truncated_SolarEnergy_NHits8_AdjCl8_OpHits10_Signal4_Bkg2.png">
  </div>
</div>

---

### VD Bottom Shielded

<div class="two-col">
  <div>
<p><strong>Solar Contour (sin12)</strong></p>
<img src="../../output/images/analysis/sensitivity/vd_1x8x14_3view_30deg_shielded/marley/truncated/full/vd_1x8x14_3view_30deg_shielded_marley_solar_sin12_df_Truncated_SolarEnergy_NHits8_AdjCl10_OpHits10_Signal4_Bkg2.png">
  </div>
  <div>
<p><strong>Reactor Contour (sin12)</strong></p>
<img src="../../output/images/analysis/sensitivity/vd_1x8x14_3view_30deg_shielded/marley/truncated/full/vd_1x8x14_3view_30deg_shielded_marley_react_sin12_df_Truncated_SolarEnergy_NHits8_AdjCl10_OpHits10_Signal4_Bkg2.png">
  </div>
</div>

---

## Contour Grids (sin13)

---

### HD Central

<div class="two-col">
  <div>
<p><strong>Solar Contour (sin13)</strong></p>
<img src="../../output/images/analysis/sensitivity/hd_1x2x6_centralAPA/marley/truncated/full/hd_1x2x6_centralAPA_marley_solar_sin13_df_Truncated_SolarEnergy_NHits1_AdjCl4_OpHits8_Signal4_Bkg2.png">
  </div>
  <div>
<p><strong>Reactor Contour (sin13)</strong></p>
<img src="../../output/images/analysis/sensitivity/hd_1x2x6_centralAPA/marley/truncated/full/hd_1x2x6_centralAPA_marley_react_sin13_df_Truncated_SolarEnergy_NHits1_AdjCl4_OpHits8_Signal4_Bkg2.png">
  </div>
</div>

---

### HD Lateral

<div class="two-col">
  <div>
<p><strong>Solar Contour (sin13)</strong></p>
<img src="../../output/images/analysis/sensitivity/hd_1x2x6_lateralAPA/marley/truncated/full/hd_1x2x6_lateralAPA_marley_solar_sin13_df_Truncated_SolarEnergy_NHits3_AdjCl2_OpHits10_Signal4_Bkg2.png">
  </div>
  <div>
<p><strong>Reactor Contour (sin13)</strong></p>
<img src="../../output/images/analysis/sensitivity/hd_1x2x6_lateralAPA/marley/truncated/full/hd_1x2x6_lateralAPA_marley_react_sin13_df_Truncated_SolarEnergy_NHits3_AdjCl2_OpHits10_Signal4_Bkg2.png">
  </div>
</div>

---

### VD Top

<div class="two-col">
  <div>
<p><strong>Solar Contour (sin13)</strong></p>
<img src="../../output/images/analysis/sensitivity/vd_1x8x14_3view_30deg_nominal/marley/truncated/full/vd_1x8x14_3view_30deg_nominal_marley_solar_sin13_df_Truncated_SolarEnergy_NHits8_AdjCl8_OpHits10_Signal4_Bkg2.png">
  </div>
  <div>
<p><strong>Reactor Contour (sin13)</strong></p>
<img src="../../output/images/analysis/sensitivity/vd_1x8x14_3view_30deg_nominal/marley/truncated/full/vd_1x8x14_3view_30deg_nominal_marley_react_sin13_df_Truncated_SolarEnergy_NHits8_AdjCl8_OpHits10_Signal4_Bkg2.png">
  </div>
</div>

---

### VD Bottom Shielded

<div class="two-col">
  <div>
<p><strong>Solar Contour (sin13)</strong></p>
<img src="../../output/images/analysis/sensitivity/vd_1x8x14_3view_30deg_shielded/marley/truncated/full/vd_1x8x14_3view_30deg_shielded_marley_solar_sin13_df_Truncated_SolarEnergy_NHits8_AdjCl10_OpHits10_Signal4_Bkg2.png">
  </div>
  <div>
<p><strong>Reactor Contour (sin13)</strong></p>
<img src="../../output/images/analysis/sensitivity/vd_1x8x14_3view_30deg_shielded/marley/truncated/full/vd_1x8x14_3view_30deg_shielded_marley_react_sin13_df_Truncated_SolarEnergy_NHits8_AdjCl10_OpHits10_Signal4_Bkg2.png">
  </div>
</div>

---

## Significance Spectra

---

### HD Central

<div class="center">
  <img src="../../output/images/analysis/sensitivity/hd_1x2x6_centralAPA/marley/truncated/hd_1x2x6_centralAPA_marley_SolarEnergy_Sensitivity_Significance_Exposure_30.png">
</div>

---

### HD Lateral

<div class="center">
  <img src="../../output/images/analysis/sensitivity/hd_1x2x6_lateralAPA/marley/truncated/hd_1x2x6_lateralAPA_marley_SolarEnergy_Sensitivity_Significance_Exposure_30.png">
</div>

---

### VD Top

<div class="center">
  <img src="../../output/images/analysis/sensitivity/vd_1x8x14_3view_30deg_nominal/marley/truncated/vd_1x8x14_3view_30deg_nominal_marley_SolarEnergy_Sensitivity_Significance_Exposure_30.png">
</div>

---

### VD Bottom Shielded

<div class="center">
  <img src="../../output/images/analysis/sensitivity/vd_1x8x14_3view_30deg_shielded/marley/truncated/vd_1x8x14_3view_30deg_shielded_marley_SolarEnergy_Sensitivity_Significance_Exposure_30.png">
</div>

---

## Template Building

---

### HD Central Templates

<div class="center">
  <img src="../../output/images/analysis/sensitivity/templates/truncated/hd_1x2x6_centralAPA/marley/hd_1x2x6_centralAPA_marley_Sensitivity_Templates_SolarEnergy_NHits1_AdjCl4_OpHits8.png">
</div>

---

### HD Lateral Templates

<div class="center">
  <img src="../../output/images/analysis/sensitivity/templates/truncated/hd_1x2x6_lateralAPA/marley/hd_1x2x6_lateralAPA_marley_Sensitivity_Templates_SolarEnergy_NHits3_AdjCl2_OpHits10.png">
</div>

---

### VD Top Templates

<div class="center">
  <img src="../../output/images/analysis/sensitivity/templates/truncated/vd_1x8x14_3view_30deg_nominal/marley/vd_1x8x14_3view_30deg_nominal_marley_Sensitivity_Templates_SolarEnergy_NHits8_AdjCl8_OpHits10.png">
</div>

---

### VD Bottom Shielded Templates

<div class="center">
  <img src="../../output/images/analysis/sensitivity/templates/truncated/vd_1x8x14_3view_30deg_shielded/marley/vd_1x8x14_3view_30deg_shielded_marley_Sensitivity_Templates_SolarEnergy_NHits8_AdjCl10_OpHits10.png">
</div>

---

## 1D Parameter Projections

---

### HD Central: Mixing Angle Projections

<div class="two-col">
  <div>
<p><strong>sin²θ₁₂</strong></p>
<img src="../../output/images/analysis/sensitivity/hd_1x2x6_centralAPA/marley/truncated/hd_1x2x6_centralAPA_marley_hd_1x2x6_centralAPA_marley_Truncated_SolarEnergy_NHits1_AdjCl4_OpHits8_sin12_projection_NuFit61.png">
  </div>
  <div>
<p><strong>sin²θ₁₃</strong></p>
<img src="../../output/images/analysis/sensitivity/hd_1x2x6_centralAPA/marley/truncated/hd_1x2x6_centralAPA_marley_hd_1x2x6_centralAPA_marley_Truncated_SolarEnergy_NHits1_AdjCl4_OpHits8_sin13_projection_NuFit61.png">
  </div>
</div>

---

### HD Central: Mass Splitting Projections

<div class="two-col">
  <div>
<p><strong>Δm²<sub>sol</sub></strong></p>
<img src="../../output/images/analysis/sensitivity/hd_1x2x6_centralAPA/marley/truncated/hd_1x2x6_centralAPA_marley_hd_1x2x6_centralAPA_marley_Truncated_SolarEnergy_NHits1_AdjCl4_OpHits8_dm2_solar_projection_NuFit61.png">
  </div>
  <div>
<p><strong>Δm²<sub>react</sub></strong></p>
<img src="../../output/images/analysis/sensitivity/hd_1x2x6_centralAPA/marley/truncated/hd_1x2x6_centralAPA_marley_hd_1x2x6_centralAPA_marley_Truncated_SolarEnergy_NHits1_AdjCl4_OpHits8_dm2_reactor_projection_NuFit61.png">
  </div>
</div>

---

### HD Lateral: Mixing Angle Projections

<div class="two-col">
  <div>
<p><strong>sin²θ₁₂</strong></p>
<img src="../../output/images/analysis/sensitivity/hd_1x2x6_lateralAPA/marley/truncated/hd_1x2x6_lateralAPA_marley_hd_1x2x6_lateralAPA_marley_Truncated_SolarEnergy_NHits3_AdjCl2_OpHits10_sin12_projection_NuFit61.png">
  </div>
  <div>
<p><strong>sin²θ₁₃</strong></p>
<img src="../../output/images/analysis/sensitivity/hd_1x2x6_lateralAPA/marley/truncated/hd_1x2x6_lateralAPA_marley_hd_1x2x6_lateralAPA_marley_Truncated_SolarEnergy_NHits3_AdjCl2_OpHits10_sin13_projection_NuFit61.png">
  </div>
</div>

---

### HD Lateral: Mass Splitting Projections

<div class="two-col">
  <div>
<p><strong>Δm²<sub>sol</sub></strong></p>
<img src="../../output/images/analysis/sensitivity/hd_1x2x6_lateralAPA/marley/truncated/hd_1x2x6_lateralAPA_marley_hd_1x2x6_lateralAPA_marley_Truncated_SolarEnergy_NHits3_AdjCl2_OpHits10_dm2_solar_projection_NuFit61.png">
  </div>
  <div>
<p><strong>Δm²<sub>react</sub></strong></p>
<img src="../../output/images/analysis/sensitivity/hd_1x2x6_lateralAPA/marley/truncated/hd_1x2x6_lateralAPA_marley_hd_1x2x6_lateralAPA_marley_Truncated_SolarEnergy_NHits3_AdjCl2_OpHits10_dm2_reactor_projection_NuFit61.png">
  </div>
</div>

---

### VD Top: Mixing Angle Projections

<div class="two-col">
  <div>
<p><strong>sin²θ₁₂</strong></p>
<img src="../../output/images/analysis/sensitivity/vd_1x8x14_3view_30deg_nominal/marley/truncated/vd_1x8x14_3view_30deg_nominal_marley_vd_1x8x14_3view_30deg_nominal_marley_Truncated_SolarEnergy_NHits8_AdjCl8_OpHits10_sin12_projection_NuFit61.png">
  </div>
  <div>
<p><strong>sin²θ₁₃</strong></p>
<img src="../../output/images/analysis/sensitivity/vd_1x8x14_3view_30deg_nominal/marley/truncated/vd_1x8x14_3view_30deg_nominal_marley_vd_1x8x14_3view_30deg_nominal_marley_Truncated_SolarEnergy_NHits8_AdjCl8_OpHits10_sin13_projection_NuFit61.png">
  </div>
</div>

---

### VD Top: Mass Splitting Projections

<div class="two-col">
  <div>
<p><strong>Δm²<sub>sol</sub></strong></p>
<img src="../../output/images/analysis/sensitivity/vd_1x8x14_3view_30deg_nominal/marley/truncated/vd_1x8x14_3view_30deg_nominal_marley_vd_1x8x14_3view_30deg_nominal_marley_Truncated_SolarEnergy_NHits8_AdjCl8_OpHits10_dm2_solar_projection_NuFit61.png">
  </div>
  <div>
<p><strong>Δm²<sub>react</sub></strong></p>
<img src="../../output/images/analysis/sensitivity/vd_1x8x14_3view_30deg_nominal/marley/truncated/vd_1x8x14_3view_30deg_nominal_marley_vd_1x8x14_3view_30deg_nominal_marley_Truncated_SolarEnergy_NHits8_AdjCl8_OpHits10_dm2_reactor_projection_NuFit61.png">
  </div>
</div>

---

### VD Bottom Shielded: Mixing Angle Projections

<div class="two-col">
  <div>
<p><strong>sin²θ₁₂</strong></p>
<img src="../../output/images/analysis/sensitivity/vd_1x8x14_3view_30deg_shielded/marley/truncated/vd_1x8x14_3view_30deg_shielded_marley_vd_1x8x14_3view_30deg_shielded_marley_Truncated_SolarEnergy_NHits8_AdjCl10_OpHits10_sin12_projection_NuFit61.png">
  </div>
  <div>
<p><strong>sin²θ₁₃</strong></p>
<img src="../../output/images/analysis/sensitivity/vd_1x8x14_3view_30deg_shielded/marley/truncated/vd_1x8x14_3view_30deg_shielded_marley_vd_1x8x14_3view_30deg_shielded_marley_Truncated_SolarEnergy_NHits8_AdjCl10_OpHits10_sin13_projection_NuFit61.png">
  </div>
</div>

---

### VD Bottom Shielded: Mass Splitting Projections

<div class="two-col">
  <div>
<p><strong>Δm²<sub>sol</sub></strong></p>
<img src="../../output/images/analysis/sensitivity/vd_1x8x14_3view_30deg_shielded/marley/truncated/vd_1x8x14_3view_30deg_shielded_marley_vd_1x8x14_3view_30deg_shielded_marley_Truncated_SolarEnergy_NHits8_AdjCl10_OpHits10_dm2_solar_projection_NuFit61.png">
  </div>
  <div>
<p><strong>Δm²<sub>react</sub></strong></p>
<img src="../../output/images/analysis/sensitivity/vd_1x8x14_3view_30deg_shielded/marley/truncated/vd_1x8x14_3view_30deg_shielded_marley_vd_1x8x14_3view_30deg_shielded_marley_Truncated_SolarEnergy_NHits8_AdjCl10_OpHits10_dm2_reactor_projection_NuFit61.png">
  </div>
</div>

---

### Selected Sensitivity Cuts by Config

| Config | NHits | OpHits | AdjCl | Signal Unc. (%) | Bkg Unc. (%) | 1D Asimov Z (σ) |
|---|---:|---:|---:|---:|---:|---:|
| HD Central | 1 | 8 | 4 | 4 | 2 | 63.00 |
| HD Lateral | 3 | 10 | 2 | 4 | 2 | 28.75 |
| VD Top | 8 | 10 | 8 | 4 | 2 | 14.91 |
| VD Bottom Shielded | 8 | 10 | 10 | 4 | 2 | 74.48 |

## Oscillograms

---

### HD Central

<div class="two-col">
  <div>
<p><strong>P(ν<sub>e</sub>→ν<sub>e</sub>) heatmap</strong></p>
<img src="../../output/images/analysis/sensitivity/oscillogram/hd_1x2x6_centralAPA/marley/truncated/hd_1x2x6_centralAPA_marley_Oscillogram_SolarEnergy.png">
  </div>
  <div>
<p><strong>Nadir projection</strong></p>
<img src="../../output/images/analysis/sensitivity/oscillogram/hd_1x2x6_centralAPA/marley/truncated/hd_1x2x6_centralAPA_marley_Oscillogram_NadirProjection_SolarEnergy.png">
  </div>
</div>

---

### HD Central (cont.)

<div class="two-col">
  <div>
<p><strong>Nadir-weighted oscillogram</strong></p>
<img src="../../output/images/analysis/sensitivity/oscillogram/hd_1x2x6_centralAPA/marley/truncated/hd_1x2x6_centralAPA_marley_Oscillogram_NadirWeighted_SolarEnergy.png">
  </div>
  <div>
<p><strong>1D fiducial signal spectrum</strong></p>
<img src="../../output/images/analysis/sensitivity/oscillogram/hd_1x2x6_centralAPA/marley/truncated/hd_1x2x6_centralAPA_marley_Signal1D_SolarEnergy_FidOnly.png">
  </div>
</div>

---

### HD Lateral

<div class="two-col">
  <div>
<p><strong>P(ν<sub>e</sub>→ν<sub>e</sub>) heatmap</strong></p>
<img src="../../output/images/analysis/sensitivity/oscillogram/hd_1x2x6_lateralAPA/marley/truncated/hd_1x2x6_lateralAPA_marley_Oscillogram_SolarEnergy.png">
  </div>
  <div>
<p><strong>Nadir projection</strong></p>
<img src="../../output/images/analysis/sensitivity/oscillogram/hd_1x2x6_lateralAPA/marley/truncated/hd_1x2x6_lateralAPA_marley_Oscillogram_NadirProjection_SolarEnergy.png">
  </div>
</div>

---

### HD Lateral (cont.)

<div class="two-col">
  <div>
<p><strong>Nadir-weighted oscillogram</strong></p>
<img src="../../output/images/analysis/sensitivity/oscillogram/hd_1x2x6_lateralAPA/marley/truncated/hd_1x2x6_lateralAPA_marley_Oscillogram_NadirWeighted_SolarEnergy.png">
  </div>
  <div>
<p><strong>1D fiducial signal spectrum</strong></p>
<img src="../../output/images/analysis/sensitivity/oscillogram/hd_1x2x6_lateralAPA/marley/truncated/hd_1x2x6_lateralAPA_marley_Signal1D_SolarEnergy_FidOnly.png">
  </div>
</div>

---

### VD Top

<div class="two-col">
  <div>
<p><strong>P(ν<sub>e</sub>→ν<sub>e</sub>) heatmap</strong></p>
<img src="../../output/images/analysis/sensitivity/oscillogram/vd_1x8x14_3view_30deg_nominal/marley/truncated/vd_1x8x14_3view_30deg_nominal_marley_Oscillogram_SolarEnergy.png">
  </div>
  <div>
<p><strong>Nadir projection</strong></p>
<img src="../../output/images/analysis/sensitivity/oscillogram/vd_1x8x14_3view_30deg_nominal/marley/truncated/vd_1x8x14_3view_30deg_nominal_marley_Oscillogram_NadirProjection_SolarEnergy.png">
  </div>
</div>

---

### VD Top (cont.)

<div class="two-col">
  <div>
<p><strong>Nadir-weighted oscillogram</strong></p>
<img src="../../output/images/analysis/sensitivity/oscillogram/vd_1x8x14_3view_30deg_nominal/marley/truncated/vd_1x8x14_3view_30deg_nominal_marley_Oscillogram_NadirWeighted_SolarEnergy.png">
  </div>
  <div>
<p><strong>1D fiducial signal spectrum</strong></p>
<img src="../../output/images/analysis/sensitivity/oscillogram/vd_1x8x14_3view_30deg_nominal/marley/truncated/vd_1x8x14_3view_30deg_nominal_marley_Signal1D_SolarEnergy_FidOnly.png">
  </div>
</div>

---

### VD Bottom Shielded

<div class="two-col">
  <div>
<p><strong>P(ν<sub>e</sub>→ν<sub>e</sub>) heatmap</strong></p>
<img src="../../output/images/analysis/sensitivity/oscillogram/vd_1x8x14_3view_30deg_shielded/marley/truncated/vd_1x8x14_3view_30deg_shielded_marley_Oscillogram_SolarEnergy.png">
  </div>
  <div>
<p><strong>Nadir projection</strong></p>
<img src="../../output/images/analysis/sensitivity/oscillogram/vd_1x8x14_3view_30deg_shielded/marley/truncated/vd_1x8x14_3view_30deg_shielded_marley_Oscillogram_NadirProjection_SolarEnergy.png">
  </div>
</div>

---

### VD Bottom Shielded (cont.)

<div class="two-col">
  <div>
<p><strong>Nadir-weighted oscillogram</strong></p>
<img src="../../output/images/analysis/sensitivity/oscillogram/vd_1x8x14_3view_30deg_shielded/marley/truncated/vd_1x8x14_3view_30deg_shielded_marley_Oscillogram_NadirWeighted_SolarEnergy.png">
  </div>
  <div>
<p><strong>1D fiducial signal spectrum</strong></p>
<img src="../../output/images/analysis/sensitivity/oscillogram/vd_1x8x14_3view_30deg_shielded/marley/truncated/vd_1x8x14_3view_30deg_shielded_marley_Signal1D_SolarEnergy_FidOnly.png">
  </div>
</div>

---



---





## Coverage and Notes

- Configs with selected sin12 solar contour plots:
- truncated: 4
- Cut table values are parsed from selected result filenames when available.
- Re-run script to refresh this folder after each workflow run:
- /usr/bin/python3 scripts/generate_sensitivity_presentation.py --folder truncated
- Full mathematical derivations: [docs/hep\_likelihood\_derivation.tex](../../docs/hep_likelihood_derivation.tex)
