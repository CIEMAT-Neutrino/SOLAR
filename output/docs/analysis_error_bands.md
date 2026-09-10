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

**Source:** `src/physics/daynight/01_daynight.py`, lines 137–138.

### Significance Columns in DayNight_Results.pkl

| Column | Meaning |
|---|---|
| `Gaussian` | Smoothed significance at $\theta = 1$ (nominal) |
| `Gaussian+Error` | Smoothed significance at $\theta = 1 + \sigma_\text{tot}$ (max asymmetry) |
| `Gaussian-Error` | Smoothed significance at $\theta = 1 - \sigma_\text{tot}$ (min asymmetry) |

Asimov analogues (`Asimov`, `Asimov+Error`, `Asimov-Error`) follow the same convention. `ErrorGaussian` and its `±Error` variants are computed with background uncertainty included. All six values are **absolute significance values** (in $\sigma$), not offsets.

### Why the Central Value Carries No Penalty

For each off-nominal scenario the Asimov LLR includes a Gaussian constraint term:

$$\text{penalty}(\theta_s) = \left(\frac{\theta_s - 1}{\sigma_\text{tot}}\right)^2$$

At the nominal scenario $\theta_s = 1$ the deviation from the prior mean is zero, so $\text{penalty} = 0$. The constraint represents a Gaussian prior centred on $\theta = 1$; evaluating at the prior maximum incurs no pull. At $\theta = 1 \pm \sigma_\text{tot}$ the deviation is $\pm 1\sigma$, giving $\text{penalty} = 1$, which deflates $q_0$ by 1 unit ($\approx 0.5\,\sigma$ in significance). This reflects the cost of claiming a result that requires the asymmetry amplitude to sit at the edge of its physical range.

**Source:** `src/physics/daynight/01_daynight.py`, lines 487–490.

### Why the Bands Are Asymmetric

Significance is a concave function of $q_0 = -2\ln\lambda$. Both off-nominal scenarios pay the same penalty of 1, but they start from different raw $q_0$ values:

$$q_0(+\sigma) = q_0^\text{raw}(1 + \sigma_\text{tot}) - 1 \quad (\text{larger raw signal, subtract constraint})$$
$$q_0(-\sigma) = q_0^\text{raw}(1 - \sigma_\text{tot}) - 1 \quad (\text{smaller raw signal, subtract constraint})$$

The raw LLR is not linear in the asymmetry amplitude: the signal enters non-linearly through Poisson likelihoods and per-bin normalisation. The gain from going from nominal to $+\sigma$ is therefore not the mirror image of the loss from nominal to $-\sigma$, and `Gaussian+Error` and `Gaussian-Error` need not be symmetric about `Gaussian`.

### Why Gaussian+Error Can Fall Below Gaussian

If the signal gain from increasing $\theta$ by one band unit is smaller than the penalty of 1, then $q_0(+\sigma) < q_0(\text{nominal})$, so `Gaussian+Error < Gaussian`. This occurs when the detector is already in a regime of good sensitivity (marginal value of extra signal is low) or when the intrinsic day-night amplitude is small for a given configuration. This outcome is physically meaningful: the $+1\sigma$ asymmetry scenario provides no additional discriminating power once the nuisance cost is accounted for.

---

## SignificanceError± Convention in _Exposure.pkl

**Source:** `src/physics/common/exposure_plot.py`.

The exposure summary pkl stores **relative offsets**, not absolute significance values:

$$\text{SignificanceError+} = \max\!\left(\text{Gaussian+Error} - \text{Gaussian},\; 0\right)$$
$$\text{SignificanceError-} = \max\!\left(\text{Gaussian} - \text{Gaussian-Error},\; 0\right)$$

Both quantities are clamped to non-negative with `np.maximum(..., 0)`.

**Justification for the clamp:** when `Gaussian+Error < Gaussian` (constraint penalty exceeds signal gain), the raw offset `Gaussian+Error − Gaussian` is negative. A negative upper error bar has no meaningful interpretation in a plot: it would invert the band, placing the "upper" scenario below the central value. Setting it to zero correctly represents the situation as "the max-asymmetry scenario provides no additional reach," rendering a one-sided band (no upper extension) rather than a misleading crossed band.

**Information preservation:** the clamp does not discard information. The underlying absolute values `Gaussian+Error` and `Gaussian-Error` are retained in `DayNight_Results.pkl` and can be retrieved directly for analyses that require the full scenario range.
