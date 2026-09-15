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
- This deck is scoped to the **Truncated** folder for the **TotalEnergy** reconstruction algorithm.

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
- best-curve reference in sensitivity/05_best_sigmas.py: **Asimov** (two-sample Poisson LLR)
- day-fraction (`--day_fraction`): fraction of exposure in daytime; default 0.5
- oscillation band (`--oscillation_band`): residual uncertainty on θ₁₂, Δm²₂₁; combined in quadrature with earth-density band

---

### Workflow Outputs

- Fiducial optimization: [config/analysis/fiducial/truncated/BestFiducials.json](../../config/analysis/fiducial/truncated/BestFiducials.json)
- Best cut summaries (JSON): [config/*/best-sigma-json/daynight/{folder}/{config}_highest_DayNight.json](../../config)
- Local fallback: [config/*/daynight-json/{folder}/{config}_highest_DayNight.json](../../config)
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
- σ2/σ3 crossing exposures: `Sigma2`/`Sigma3` and `AsimovSigma2`/`AsimovSigma3` are both Asimov-based (profiled nuisance); Gaussian significance is stored as a diagnostic only.
- MC threshold gate: cuts where any essential background (gamma, neutron) has fewer than `--mc_threshold` MC events are skipped; prevents selecting cuts that deplete backgrounds statistically.
- Smoothing is applied per component above threshold; the threshold slice keeps unsmoothed bins below threshold and replaces bins above with smoothed values.

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

### Fiducial plots

No fiducial optimization plots were found for this folder.

---

### Fiducial Optimization Summary

| Config | Fiducial X | Fiducial Y | Fiducial Z | Before Fiducialization | After Fiducialization | Fiducial Mass (kt) |
|---|---:|---:|---:|---:|---:|---:|
| HD Central | 100 | 280 | 120 | 0.049 | 0.215 | 2.16 |
| HD Lateral | 0 | 60 | 0 | 0.045 | 0.055 | 6.08 |
| VD Top | 0 | 60 | 40 | 0.095 | 0.097 | 6.87 |
| VD Bottom Shielded | 0 | 40 | 0 | 0.092 | 0.107 | 7.37 |

---

## DayNight Results

---

### HD Central

No matching exposure/significance pair found for HD Central

---

### HD Lateral

No matching exposure/significance pair found for HD Lateral

---

### VD Top

No matching exposure/significance pair found for VD Top

---

### VD Bottom Shielded

No matching exposure/significance pair found for VD Bottom Shielded

---

### Best DayNight Cuts by Config

| Config | NHits | OpHits | AdjCl | Significance |
|---|---:|---:|---:|---:|
| HD Central | 1 | 10 | 2 | 4.223 |
| HD Lateral | 4 | 4 | 3 | 0.733 |
| VD Top | 7 | 8 | 3 | 0.791 |
| VD Bottom Shielded | 6 | 9 | 8 | 2.105 |

---

## Oscillograms

---

### HD Central

No oscillogram found.

---

### HD Lateral

No oscillogram found.

---

### VD Top

No oscillogram found.

---

### VD Bottom Shielded

No oscillogram found.

---



---

## Coverage and Notes

- Config coverage in best-cut JSON outputs:
  - nominal: 4
  - reduced: 4
  - truncated: 4
- Table values are read from workflow-generated JSON at generation time.
- Re-run script to refresh this folder after each workflow run:
- /usr/bin/python3 scripts/generate_daynight_presentation.py --folder truncated
