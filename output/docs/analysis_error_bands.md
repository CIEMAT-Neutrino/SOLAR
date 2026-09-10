# Analysis Error Bands — Convention and Justification

*Generated: 2026-09-01. Updated 2026-09-10. Covers DayNight asymmetry uncertainty bands as stored in
`DayNight_Results.pkl` and propagated to `DayNight_Exposure.pkl`. For Sensitivity analysis error bands and fitting methodology, see [Recent Statistical Methodology Updates](#stat-updates) in `solar_analyses.md`.*

---

## Day-Night Asymmetry Error Bands

### Physical Model

The day-night effect is parameterised through an asymmetry amplitude $\theta$, defined so that the signal rate scales as $\theta \times$ (nominal rate). The nominal prediction corresponds to $\theta = 1$. Theoretical uncertainty in $\theta$ arises from two independent sources combined in quadrature:

$$\sigma_\text{tot} = \sqrt{\sigma_\oplus^2 + \sigma_\text{osc}^2}$$

where $\sigma_\oplus$ is the fractional uncertainty from PREM-based Earth density profile variations (`--earth_density_band`, default 0.13) and $\sigma_\text{osc}$ is the residual uncertainty from PDG $\theta_{12}$ and $\Delta m^2_{21}$ ranges (`--oscillation_band`, default 0.05). Three scenarios are evaluated at:

$$\theta_s \in \{1 + \sigma_\text{tot},\; 1.0,\; 1 - \sigma_\text{tot}\} \quad \text{(indices 0, 1, 2)}$$

**Source:** `src/physics/daynight/01_daynight.py`, lines 137–146 (asymmetry scale definitions).

### Significance Columns in DayNight_Results.pkl

| Column | Meaning |
|---|---|
| `Gaussian` | Smoothed significance at $\theta = 1$ (nominal) |
| `Gaussian+Error` | Smoothed significance at $\theta = 1 + \sigma_\text{tot}$ (max asymmetry) |
| `Gaussian-Error` | Smoothed significance at $\theta = 1 - \sigma_\text{tot}$ (min asymmetry) |

Asimov analogues (`Asimov`, `Asimov+Error`, `Asimov-Error`) follow the same convention. `ErrorGaussian` and its `±Error` variants are computed with background uncertainty included. All six values are **absolute significance values** (in $\sigma$), not offsets.

### Why the Bands Are Asymmetric

The raw log-likelihood ratio $q_0$ is not linear in the asymmetry amplitude $\theta_s$: the signal enters non-linearly through Poisson likelihoods and per-bin normalisation. Therefore the gain from going from nominal to $+\sigma_\text{tot}$ need not equal the loss from nominal to $-\sigma_\text{tot}$, and `Gaussian+Error` and `Gaussian-Error` need not be symmetric about `Gaussian`. This asymmetry is a genuine feature of the Poisson statistics, not an artifact.

### Why Gaussian+Error Can Fall Below Gaussian

Because the bands are evaluated independently **without penalty terms**, it is possible for `Gaussian+Error` (the upper asymmetry scenario) to produce a lower significance than `Gaussian` (nominal). This occurs when the detector is already in a regime of good sensitivity where the marginal gain from the additional asymmetry signal is small. This is physically meaningful: the $+1\sigma$ asymmetry scenario provides no additional discriminating power beyond what is already achieved at the nominal prediction.

---

## SignificanceError± Convention in _Exposure.pkl

**Source:** `src/physics/common/exposure_plot.py`.

The exposure summary pkl stores **relative offsets**, not absolute significance values:

$$\text{SignificanceError+} = \max\!\left(\text{Gaussian+Error} - \text{Gaussian},\; 0\right)$$
$$\text{SignificanceError-} = \max\!\left(\text{Gaussian} - \text{Gaussian-Error},\; 0\right)$$

Both quantities are clamped to non-negative with `np.maximum(..., 0)`.

**Justification for the clamp:** when `Gaussian+Error < Gaussian` (the upper asymmetry scenario produces less significance than nominal), the raw offset `Gaussian+Error − Gaussian` is negative. A negative upper error bar has no meaningful interpretation in a plot: it would invert the band, placing the "upper" scenario below the central value. Setting it to zero correctly represents the situation as "the max-asymmetry scenario provides no additional reach," rendering a one-sided band (no upper extension) rather than a misleading crossed band.

**Information preservation:** the clamp does not discard information. The underlying absolute values `Gaussian+Error` and `Gaussian-Error` are retained in `DayNight_Results.pkl` and can be retrieved directly for analyses that require the full scenario range.
