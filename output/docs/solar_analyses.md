# SOLAR/DUNE Significance Analyses
## Mathematical Derivations: Day-Night, HEP, and Sensitivity

*SOLAR/DUNE Analysis — May 2026*

---

$$
\newcommand{\Ecal}{\mathcal{E}}
\newcommand{\Mdet}{M_{\mathrm{det}}}
\newcommand{\rc}{r_{i}^{c}}
\newcommand{\muc}{\mu_{i}^{c}}
\newcommand{\si}{s_{i}}
\newcommand{\bi}{b_{i}}
\newcommand{\nni}{n_{i}}
\newcommand{\ZG}{Z^{\mathrm{G}}_{i}}
\newcommand{\ZGerr}{Z^{\mathrm{G,err}}_{i}}
\newcommand{\ZA}{Z_A}
\newcommand{\ZAsig}{Z_A^{\sigma}}
\newcommand{\fd}{f}
\newcommand{\fn}{g}
\newcommand{\Beff}{B^{\mathrm{eff}}_{i}}
\newcommand{\seff}{\sigma^{\mathrm{eff}}_{i}}
\newcommand{\epstot}{\varepsilon_{\mathrm{tot}}}
\newcommand{\bhat}{\hat{\beta}}
\newcommand{\srel}{\sigma_{\mathrm{rel}}}
\newcommand{\srelsq}{\sigma_{\mathrm{rel}}^{2}}
\newcommand{\Btot}{B_{\mathrm{tot}}}
\newcommand{\Ntot}{N_{\mathrm{tot}}}
\newcommand{\minexp}{\epsilon_{\min}}
\newcommand{\Tsig}{T^{\mathrm{sig}}_{ij}}
\newcommand{\Tbkg}{T^{\mathrm{bkg}}_{ij}}
\newcommand{\Apred}{A_{\mathrm{pred}}}
\newcommand{\Abkg}{A_{\mathrm{bkg}}}
\newcommand{\spred}{\sigma_{\mathrm{pred}}}
\newcommand{\sbkg}{\sigma_{\mathrm{bkg}}}
\newcommand{\eij}{e_{ij}}
\newcommand{\oij}{o_{ij}}
$$

---

## Contents

1. [Overview and Scientific Goals](#overview)
2. [Common Ingredients](#common)
3. [Day-Night Asymmetry Analysis](#daynight)
4. [HEP Discovery Analysis](#hep)
5. [Sensitivity Analysis](#sensitivity)
6. [Comparison Across Analyses](#comparison)
7. [Summary of Significance Outputs](#summary)
8. [Recent Statistical Methodology Updates](#stat-updates)
9. [Workflow Flags and Configuration](#flags)
10. [References](#references)

---

## 1. Overview and Scientific Goals {#overview}

This document presents the mathematical derivations underlying the three significance analyses in the SOLAR/DUNE software framework, ordered from simplest to most complex:

1. **Day-Night Asymmetry** (`12DayNight.py`): searches for a time-modulated excess of solar neutrinos during nighttime relative to daytime, driven by the MSW matter effect inside the Earth. Significance is computed from both a Gaussian approximation on the rate-difference spectrum and a two-sample Poisson log-likelihood ratio (Asimov).

2. **HEP Discovery** (`13HEP.py`): searches for the hep solar neutrino flux ($^3\mathrm{He}+p \to {}^4\mathrm{He}+e^++\nu_e$) as an absolute excess above background. Three significance estimators are computed: Gaussian, Asimov, and a profile-likelihood (PL) significance with a single global background nuisance, analytically profiled.

3. **Sensitivity** (`14Sensitivity.py`): maps the 2D sensitivity contour in the oscillation-parameter plane $(\Delta m^2,\,\sin^2\theta_{12})$. A Baker-Cousins Poisson deviance with two normalization nuisances is minimized numerically over 2D templates in energy and azimuth.

The three analyses share common ingredients (histogram smoothing, thresholds, adaptive rebinning for HEP) but differ in the statistical complexity of their model. See [Comparison Across Analyses](#comparison) for a summary table.

**Common notation.** Throughout, $T$ is the exposure in kt·yr, $\Mdet$ the active detector mass in kt, and the exposure factor is $\Ecal = T\,\Mdet$. Index $i$ runs over energy bins; index $j$ over azimuth bins (Sensitivity only). Component label $c$ identifies one physical process (signal, neutron, gamma, radiological, ${}^8\mathrm{B}$).

---

## 2. Common Ingredients {#common}

### 2.1 Rate Histograms and Exposure Scaling

All three analyses work with per-unit-exposure, per-unit-mass rate histograms $\rc$ (units: $(\mathrm{yr\cdot kt\cdot MeV})^{-1}$, integrated over the bin width $\Delta E$). Expected event counts at exposure $\Ecal$ are:

$$\muc = \Ecal\,\rc = T\,\Mdet\,\rc$$

All analyses apply a reconstructed-energy threshold $E_{\mathrm{th}}$ (configured in `analysis/config.json`); only bins with $E_i \ge E_{\mathrm{th}}$ enter the significance computation.

### 2.2 Histogram Smoothing

Each rate histogram $\rc$ is convolved with a one-dimensional Gaussian kernel before entering the significance computation:

$$\tilde{r}_{i}^{c} = \sum_{j} G_\sigma(i-j)\, r_j^{c}, \qquad G_\sigma(k) = \frac{1}{\sqrt{2\pi}\,\sigma} \exp\!\left(-\frac{k^2}{2\sigma^2}\right)$$

implemented via `scipy.ndimage.gaussian_filter1d` with `mode='nearest'`. Whether smoothing is applied to a given component, and the width $\sigma$, are configured in `analysis/smoothing.json` under `SMOOTHING.ANALYSES.{ANALYSIS}.STAGES`. Separate stages are provided for the fiducial scan and the significance computation.

**Non-negativity clipping.** Gaussian convolution at distribution tails can produce small negative values. Negative rates are unphysical and cause profile-likelihood divergences at high exposure (see [Numerical Stability](#numerical-stability)). All smoothed rates are therefore clipped to zero before any significance computation:

$$\tilde{r}_{i}^{c} \leftarrow \max\!\left(0,\,\tilde{r}_{i}^{c}\right)$$

### 2.3 Per-Component Background Error Model

For each background component $c$ the absolute uncertainty on the raw rate in bin $i$ combines MC-statistical and systematic contributions in quadrature:

$$\sigma_{i}^{c} = r_i^{c}\, \sqrt{\left(\frac{\delta_i^c}{r_i^c}\right)^{\!2} + \left(\srel^c\right)^2}$$

where $\delta_i^c / r_i^c$ is the relative histogram statistical error and $\srel^c$ is the relative systematic uncertainty (`--background_uncertainty`, typically 2%). The total combined background uncertainty in bin $i$ is:

$$\sigma_i^{\mathrm{bkg}} = \sqrt{\sum_c \bigl(\sigma_i^c\bigr)^2}$$

also convolved with the component-specific smoothing kernel. For the hep signal component the corresponding quantity uses $\srel^{\mathrm{sig}} = 30\%$ (HEP) or $4\%$ (Sensitivity). In the HEP analysis ${}^8$B is treated as a background component and uses $\srel^{{}^8\mathrm{B}} = \srel^{\mathrm{bkg}} = 2\%$ (see [HEP Signal Model](#hep-signal)).

---

## 3. Day-Night Asymmetry Analysis {#daynight}

### 3.1 Physical Observable: MSW-Enhanced Day-Night Effect

Solar $\nu_e$ produced in the solar core oscillate in vacuum on their way to Earth. During nighttime passages the neutrinos traverse the Earth's interior and experience additional matter-induced (MSW) oscillations that partially restore the $\nu_e$ flavour component. This creates a night-time excess characterised by the asymmetry:

$$\Delta_{\mathrm{DN}} \equiv \frac{N_{\mathrm{night}} - N_{\mathrm{day}}} {{\tfrac{1}{2}(N_{\mathrm{night}} + N_{\mathrm{day}})}}$$

The size of $\Delta_{\mathrm{DN}}$ depends on the Earth electron-density profile through the MSW potential $V = \sqrt{2}\,G_F\, n_e(r)$ and on the oscillation parameters $(\Delta m^2,\,\sin^2\theta_{12})$.

### 3.2 Two-Period Signal and Background Model

Let $\fd \in (0,1)$ be the fraction of total exposure attributed to daytime and $\fn = 1 - \fd$ the nighttime fraction ($\fd = 0.5$ by default; the SURF latitude of $44.35^\circ$N gives $\fd \approx 0.493$ averaged over a full year). The oscillation-weighted solar rates split into day and night components:

$$\begin{align}
r_i^{\mathrm{day}} &= \text{solar rate in daytime sky exposure},\\
r_i^{\mathrm{night}} &= \text{solar rate in nighttime (Earth-crossing) exposure}.
\end{align}$$

Isotropic backgrounds (cosmogenic, geological, detector-intrinsic) are assumed time-uniform and therefore split proportionally between periods.

Event counts observed in day and night periods are:

$$\begin{align}
n_i^{\mathrm{night}} &= \Ecal\bigl(\fn\, r_i^{\mathrm{bkg}} + r_i^{\mathrm{night}}\bigr),\\
n_i^{\mathrm{day}}   &= \Ecal\bigl(\fd\, r_i^{\mathrm{bkg}} + r_i^{\mathrm{day}}\bigr).
\end{align}$$

The **asymmetry signal** at scale factor $\theta_s$ (see [Asymmetry Uncertainty Band](#asymmetry-band)) is:

$$\si(\theta_s) = \Ecal\cdot\theta_s\cdot\bigl(r_i^{\mathrm{night}} - r_i^{\mathrm{day}}\bigr)$$

### 3.3 Effective Two-Sample Background

For a two-sample rate-difference test with unequal period lengths $\fd$ and $\fn$, the optimal significance statistic requires an *inverse-fraction-weighted* effective background:

$$\Beff = \frac{n_i^{\mathrm{night}}}{\fn^2} + \frac{n_i^{\mathrm{day}}}{\fd^2}$$

In the equal-period limit $\fd=\fn=\tfrac{1}{2}$ and with $\si\ll\Beff$, this reduces to $\Beff \approx 4\Ecal r_i^{\mathrm{bkg}}$ and the Gaussian significance becomes the familiar $s_i / \sqrt{b_i}$ form. Bins where $\Beff = 0$ are zeroed.

### 3.4 Gaussian Significance

Two Gaussian estimators are computed per bin.

**Simple Gaussian** (no additional background uncertainty):

$$\ZG = \frac{\si}{\sqrt{\Beff}}$$

**Error Gaussian** (with combined background uncertainty $\seff$, see [Background Uncertainty](#dn-bkg-uncertainty)):

$$\ZGerr = \frac{\si}{\sqrt{\Beff + \bigl(\seff\bigr)^2}}$$

The global significance is the quadrature sum over all bins above threshold:

$$Z = \sqrt{\sum_{i \ge i_{\mathrm{th}}} \!\left(Z_i\right)^2}$$

Both raw and Gaussian-smoothed rate versions are used, producing four output curves per cut per asymmetry-scale value (`RawGaussian`, `RawErrorGaussian`, `Gaussian`, `ErrorGaussian`) plus upper/lower band variants (see [Asymmetry Uncertainty Band](#asymmetry-band)).

### 3.5 Background Uncertainty for the Rate-Difference Test {#dn-bkg-uncertainty}

Three independent noise sources are combined in quadrature per period and per bin, then merged into an effective uncertainty:

1. **Poisson statistical**: $\sqrt{N_{\mathrm{bkg},\,p}}$, where $N_{\mathrm{bkg},\,p} = \Ecal\,f_p\,r_i^{\mathrm{bkg}}$ is the background count in period $p \in \{\mathrm{day, night}\}$.
2. **Normalization systematic**: $\srel^{\mathrm{bkg}}\,N_{\mathrm{bkg},\,p}$ (`--background_uncertainty`).
3. **Day-fraction systematic**: $\delta_f\,\Ecal\,r_i^{\mathrm{bkg}}$ (`--day_fraction_band`, $\delta_f = 0.01$ by default), reflecting uncertainty in the solar zenith angle cut and the run schedule.

Per-period uncertainties:

$$\sigma_{p,i} = \sqrt{N_{\mathrm{bkg},\,p,i} + \bigl(\srel^{\mathrm{bkg}}\,N_{\mathrm{bkg},\,p,i}\bigr)^2 + \bigl(\delta_f\,\Ecal\,r_i^{\mathrm{bkg}}\bigr)^2}$$

Combining across periods via inverse-fraction weighting:

$$\seff = \sqrt{\left(\frac{\sigma_{\mathrm{night},i}}{\fn}\right)^{\!2} +\left(\frac{\sigma_{\mathrm{day},i}}{\fd}\right)^{\!2}}$$

set to zero wherever $\Beff = 0$.

### 3.6 Two-Sample Poisson Asimov Significance

The Day-Night analysis also computes a two-sample Poisson log-likelihood ratio (Asimov) as an alternative to the Gaussian estimators. Under the signal hypothesis (asymmetry scale $\theta_s$), the Asimov (expected) observed counts are $n_i^{\mathrm{night}}$ and $n_i^{\mathrm{day}}$ from above. The null hypothesis $H_0$ assumes no asymmetry: both periods share the same per-bin total rate, allocated proportionally to period length:

$$h_{0,i}^{\mathrm{night}} = \fn\,(n_i^{\mathrm{night}} + n_i^{\mathrm{day}}), \qquad h_{0,i}^{\mathrm{day}} = \fd\,(n_i^{\mathrm{night}} + n_i^{\mathrm{day}})$$

The per-bin log-likelihood ratio contribution is:

$$q_i(\theta_s) = 2\!\left[ n_i^{\mathrm{night}}\ln\frac{n_i^{\mathrm{night}}}{h_{0,i}^{\mathrm{night}}} + n_i^{\mathrm{day}}\ln\frac{n_i^{\mathrm{day}}}{h_{0,i}^{\mathrm{day}}} \right]$$

defined as zero whenever any count or null hypothesis count is non-positive.

**Gaussian constraint on the asymmetry amplitude.** The asymmetry amplitude $\theta_s$ is treated as a constrained nuisance: $\theta_s \sim \mathcal{N}(1, \epstot^2)$, where $\epstot$ is the total band (see [Asymmetry Uncertainty Band](#asymmetry-band)). For each band scenario $\theta_s \in \{1+\epstot,\;1,\;1-\epstot\}$ the penalised test statistic is:

$$q^{\mathrm{pen}}(\theta_s) = \sum_{i \ge i_{\mathrm{th}}} q_i(\theta_s) - \left(\frac{\theta_s - 1}{\epstot}\right)^{\!2}$$

The Asimov significance for each band scenario is:

$$\ZA(\theta_s) = \sqrt{\max\!\left(0,\,q^{\mathrm{pen}}(\theta_s)\right)}$$

At the nominal point $\theta_s=1$ the penalty vanishes. At $\theta_s = 1\pm\epstot$ the penalty equals $1$, deflating the raw sum by exactly $1\,\sigma^2$ relative to the nominal curve. This correctly reflects the prior information on the asymmetry scale: the off-nominal band significances are not free upper/lower bounds but penalised estimates constrained by the prior uncertainty.

**Asimov as the primary cut-optimization metric.** The nominal-band smoothed Asimov significance $\ZA(\theta_s{=}1)$ is the primary metric for selecting the optimal topological cuts: the cut that minimises the exposure needed to reach $\ZA \ge 2\sigma$ is retained. The Gaussian significance is computed in parallel as a diagnostic output but does not drive cut selection. The output columns `Sigma2`/`Sigma3` and `AsimovSigma2`/`AsimovSigma3` are identical (both record the Asimov-based exposure thresholds); `AsimovSigma2` is an alias retained for backward compatibility.

### 3.7 Asymmetry Uncertainty Band {#asymmetry-band}

Two independent sources of uncertainty on the predicted asymmetry amplitude are combined in quadrature to define a total band $\epstot$:

1. **Earth density band** $\varepsilon_{\oplus}$ (`--earth_density_band`, default $0.13$): fractional uncertainty from PREM-based oscillation probability calculations.
2. **Oscillation parameter band** $\varepsilon_{\mathrm{osc}}$ (`--oscillation_band`, default $0.05$): residual uncertainty from PDG $\theta_{12}$ and $\Delta m^2_{21}$ ranges.

$$\epstot = \sqrt{\varepsilon_{\oplus}^2 + \varepsilon_{\mathrm{osc}}^2} \approx 0.138 \;\text{(defaults)}$$

Three scale factors bracket the full predicted range:

$$\theta_s \in \{1+\epstot,\;1,\;1-\epstot\}$$

**Gaussian estimators.** The scale $\theta_s$ enters the signal multiplicatively, leaving the background unchanged. Three curves (upper, nominal, lower) are produced for each Gaussian estimator.

**Asimov estimator.** For the Asimov LLR the asymmetry amplitude is treated as a constrained nuisance (see [Two-Sample Poisson Asimov](#three-six)): $\theta_s$ enters both the signal and the Gaussian penalty term $(\theta_s-1)^2/\epstot^2$. The three Asimov curves are therefore deflated from their unconstrained values by the penalty, correctly encoding the prior information on the asymmetry scale.

### 3.8 Results

| Figure | Description |
|---|---|
| `figures/dn_central_exposure` | centralAPA: exposure needed for $3\sigma$ and $5\sigma$ discovery vs. day-night asymmetry magnitude |
| `figures/dn_central_significance` | centralAPA: significance vs. exposure at 30 kt·yr for nominal, upper, lower asymmetry bands |
| `figures/dn_lateral_exposure` | lateralAPA: exposure threshold for $3\sigma$ and $5\sigma$ Day-Night discovery |
| `figures/dn_lateral_significance` | lateralAPA: significance vs. exposure at 30 kt·yr |

---

## 4. HEP Discovery Analysis {#hep}

### 4.1 Signal and Background Model {#hep-signal}

The HEP analysis searches for an absolute rate excess of hep solar neutrinos above a known background in a one-dimensional energy spectrum above $E_{\mathrm{th}}$. Signal (hep) and total background event counts in bin $i$ are:

$$\begin{align}
\si &= \Ecal\, r_i^{\mathrm{hep}},\\
\bi &= \Ecal\sum_{c\neq\mathrm{hep}} r_i^c = \Ecal\bigl(r_i^{\gamma} + r_i^{n} + r_i^{\mathrm{rad}} + r_i^{{}^8\mathrm{B}}\bigr).
\end{align}$$

The Asimov dataset sets observed counts to $\nni = \si + \bi$.

**Component uncertainty assignment.** The analysis distinguishes three uncertainty classes:

- **Pure backgrounds** ($\gamma$, $n$, radiological): $\srel^c = \srel^{\mathrm{bkg}}$ (`--background_uncertainty`, default 2%).
- **${}^8$B solar (oscillated background)**: also assigned $\srel^{{}^8\mathrm{B}} = \srel^{\mathrm{bkg}}$ (2%). Although ${}^8$B is an oscillated solar signal, at the energies probed by HEP (14–30 MeV) its rate is well constrained and it enters as a known background. Treating it with background uncertainty suppresses the absorbing-plateau problem that would arise from inflating its normalization freedom to the 30% level.
- **hep signal**: $\srel^{\mathrm{hep}} = \srel^{\mathrm{sig}}$ (`--signal_uncertainty`, default 30%). The `--signal_uncertainty` flag and the `unc_sig` study scan affect *only* the hep component.

### 4.2 Background Error Aggregation

Each background component $c \in \{\gamma, n, \mathrm{rad}, {}^8\mathrm{B}\}$ contributes an absolute uncertainty $\sigma_i^c$ (from [Per-Component Background Error Model](#background-error)) computed with $\srel^c = \srel^{\mathrm{bkg}}$. The combined error entering the significance computation is:

$$\sigma_i^{\mathrm{bkg}} = \sqrt{\sum_{c \neq \mathrm{hep}} \bigl(\sigma_i^c\bigr)^2}$$

### 4.3 Adaptive Tail Rebinning

#### Detection Threshold

A bin (or merged group) is declared *detectable* if the expected (conservatively shifted) signal exceeds:

$$T_{\mathrm{det}} = \max\!\left(T_{\mathrm{min}},\;-\ln(1-p_{\mathrm{min}})\right)$$

with defaults $T_{\mathrm{min}}=1.0$ and $p_{\mathrm{min}}=1-e^{-1}$, giving $T_{\mathrm{det}}=1.0$. The Poisson interpretation: $P(\ge 1;\lambda)\ge p \Leftrightarrow \lambda\ge -\ln(1-p)$.

The conservative (downward-fluctuated) detectable signal is:

$$s_i^{\mathrm{det}} = \si\,(1 - d\cdot\srel^{\mathrm{sig}}), \quad d \in \{2.9,\;3.0,\;3.1\}$$

where the three values of $d$ correspond to the upper (+), nominal, and lower ($-$) error bands respectively.

#### Greedy Algorithm and Monotonicity Enforcement

Bins are merged from the high-energy tail inward until each group exceeds $T_{\mathrm{det}}$. Groups below $T_{\mathrm{det}}$ are zeroed out. To guarantee a non-decreasing significance vs. exposure curve, if the current step's optimal binning yields lower Asimov significance than the previous step the previous binning is retained:

$$\mathcal{B}(t) = \begin{cases} \mathcal{B}^*(t) & Z_A(\mathcal{B}^*(t),t) \ge Z_A(\mathcal{B}(t{-}1),t{-}1),\\ \mathcal{B}(t-1) & \text{otherwise.} \end{cases}$$

### 4.4 Gaussian and Asimov Significance

With $\si$ and $\bi$ from the signal and background model above, and combined background uncertainty $\sigma_i^{\mathrm{bkg}}$, the per-bin Gaussian estimator is:

$$\ZG = \frac{\si}{\sqrt{\bi + \bigl(\sigma_i^{\mathrm{bkg}}\bigr)^2}}$$

and the global significance is the quadrature sum over all bins above threshold.

The Asimov significance [Cowan 2010], with and without background uncertainty, is:

$$\begin{align}
Z_A &= \sqrt{2\left[(\si+\bi)\ln\!\left(1+\frac{\si}{\bi}\right) - \si\right]},\\
Z_A^{\sigma} &= \sqrt{2\left[ (\si+\bi)\ln\!\frac{(\si+\bi)(\bi+(\sigma_i^{\mathrm{bkg}})^2)} {\bi^2+(\si+\bi)(\sigma_i^{\mathrm{bkg}})^2} -\frac{\bi^2}{(\sigma_i^{\mathrm{bkg}})^2} \ln\!\left(1+\frac{(\sigma_i^{\mathrm{bkg}})^2 \si}{\bi(\bi+(\sigma_i^{\mathrm{bkg}})^2)}\right) \right]}.
\end{align}$$

These are computed both with and without adaptive tail rebinning. The profile-likelihood (see below) is the primary metric.

### 4.5 Profile-Likelihood Significance

#### Test Statistic

The profile-likelihood significance follows Cowan et al. [Cowan 2010]. The test statistic for the null hypothesis $\mu=0$ (no HEP signal) is:

$$q_0 = -2\ln\!\frac{\mathcal{L}(0,\,\bhat)}{\mathcal{L}(\hat{s}+b,\,1)}$$

evaluated at the Asimov point $\nni = \si + \bi$, with $\bhat$ the background scale factor profiled under the null. The median discovery significance is $Z = \sqrt{q_0}$.

#### Global Background Normalization Nuisance

A single global background scale factor $\beta$ — fully correlated across all bins — is constrained by a Gaussian penalty:

$$\mathcal{L}(0,\beta) = \prod_{i=1}^{K} \frac{(\beta \bi)^{\nni} e^{-\beta \bi}}{\nni!} \cdot\exp\!\left(-\frac{(\beta-1)^2}{2\srelsq}\right)$$

where $\srel$ is the relative background uncertainty (`--background_uncertainty`, typically 2%). A global (fully correlated) $\beta$ avoids the $K$-parameter absorbing plateau that per-bin nuisances produce; the plateau ends at $T \sim 1/(B_{\mathrm{tot}}\srelsq)$ (typically sub-year), giving a physically correct PL curve.

#### Profiling $\bhat$

The stationarity condition $\partial\ln\mathcal{L}/\partial\beta\big|_{\bhat}=0$ at the Asimov point $\nni=\si+\bi$ gives:

$$\Ntot = \bhat\Btot + \frac{\bhat-1}{\srelsq}, \quad \Ntot = \sum_i \nni,\;\Btot = \sum_i \bi$$

Because $\beta$ is *global*, the per-bin denominators $\beta\bi$ factorize and the stationarity condition is a scalar equation. Rearranging:

$$\bhat^2 + \left(\Btot\srelsq-1\right)\bhat - \Ntot\srelsq = 0$$

with analytic positive root:

$$\bhat = \frac{-(\Btot\srelsq-1) + \sqrt{(\Btot\srelsq-1)^2 + 4\Ntot\srelsq}}{2}$$

When $\Btot=0$ or $\srel=0$, $\bhat=1$ (no constraint active).

#### Log-Likelihood Ratio

Substituting $\nni = \si+\bi$ into the test statistic:

$$q_0 = 2\sum_{i=1}^{K} \left[\nni\ln\!\frac{\nni}{\bhat \bi} - (\nni-\bhat \bi)\right] + \left(\frac{\bhat-1}{\srel}\right)^{\!2}$$

Each per-bin term is the Baker-Cousins Poisson deviance [Baker & Cousins 1984] between the Asimov observation $\nni$ and the null expectation $\bhat\bi$. This expression — a sum of Poisson deviances with a global nuisance penalty — is the structural prototype for the Sensitivity objective (see [Sensitivity](#sensitivity)), extended to 2D with two nuisances.

#### Numerical Stability {#numerical-stability}

When $S^2/B \ll 1$, naive subtraction of two $\mathcal{O}(N\log N)$ log-likelihoods loses precision. The stable per-bin form is:

$$\Delta\ell_i = \nni\ln\!\frac{\nni}{\mu_i^{\mathrm{null}}} - \bigl(\nni - \mu_i^{\mathrm{null}}\bigr), \qquad \mu_i^{\mathrm{null}} = \bhat \bi$$

with a floor $\minexp=10^{-12}$ replacing zero denominators.

**Negative-rate blowup.** If a smoothed rate is negative, $\bhat\bi<0$, the floor gives $\Delta\ell_i \approx \nni\ln(\nni/\minexp) \propto T$, causing a spurious superlinear spike. Clipping (non-negativity clamp from [Smoothing](#histogram-smoothing)) eliminates this path entirely.

### 4.6 Barlow-Beeston MC Mask

Bins with summed background MC count below $N_{\mathrm{MC}}^{\min}$ (default 1.0) lack a reliable background model and are excluded by a static binary mask [Barlow & Beeston 1993]:

$$\mathcal{M}_i = \mathbf{1}\!\left[N_i^{\mathrm{MC}} \ge N_{\mathrm{MC}}^{\min}\right], \qquad \si \leftarrow \mathcal{M}_i \si,\quad \bi \leftarrow \mathcal{M}_i \bi$$

The mask is static (MC counts do not change with exposure), so it introduces no discrete transitions in the significance curve. The same mask is applied to both raw and smoothed histograms.

### 4.7 Profile-Likelihood Signal Bands and Post-Processing

Three signal normalizations are evaluated to produce $\pm1\sigma_s$ bands, using the same scale factor $d \in \{+1, 0, -1\}$ applied to the signal:

$$s_i^{(d)} = \si\,(1 + d\cdot\srel^{\mathrm{sig}})$$

Background is never shifted; $\bhat$ and its quadratic solution are unaffected.

The raw PL significance curve $Z(T)$ is post-processed by two steps when `pl_isotonic` is enabled (controlled by `analysis/config.json` under `WORKFLOW.HEP.pl_isotonic`):

1. **Gaussian kernel smoothing** with $\sigma_{\mathrm{PL}}=6$ exposure-grid index units (`scipy.ndimage.gaussian_filter1d`), which suppresses numerical oscillations from the PL solver at low signal-to-background ratio.
2. **Isotonic regression (PAVA)** [Robertson 1988]:

$$\min_{\{z_k\}}\sum_k(z_k-\tilde{Z}(T_k))^2 \quad\text{s.t.}\quad z_k\le z_{k+1}$$

via `sklearn.isotonic.IsotonicRegression`, which enforces a non-decreasing significance curve while minimising the $\ell^2$ deviation from the computed values.

Pipeline: $Z \to \max(0,Z) \xrightarrow{\mathrm{Gauss}} \tilde{Z} \xrightarrow{\mathrm{PAVA}} Z'$.

After isotonic regression, upper/lower bands are enforced to remain above/below the nominal curve respectively: $Z'^{(+1)}\leftarrow\max(Z'^{(+1)}, Z'^{(0)})$ and $Z'^{(-1)}\leftarrow\min(Z'^{(-1)}, Z'^{(0)})$. Pre-PAVA curves are saved separately for diagnostic purposes.

### 4.8 Spike Detection and Best-Cut Selection

A PL curve is declared *spiked* if any consecutive step in the pre-isotonic columns exceeds $\Delta_{\max}$ (`--max_pl_jump`, default $1\,\sigma$):

$$\text{spiked} \;\Leftrightarrow\; \max_k\!\left[Z_{\mathrm{pre}}(T_{k+1})-Z_{\mathrm{pre}}(T_k)\right]>\Delta_{\max}$$

The best cut maximises the reference significance over non-spiked cuts:

$$(N^*,N_{\mathrm{op}}^*,N_{\mathrm{adj}}^*) = \arg\max_{\text{not spiked}} Z_{\mathrm{ref}}(T_{\max})$$

### 4.9 Results

| Figure | Description |
|---|---|
| `figures/hep_central_exposure` | centralAPA: Gaussian and Asimov exposure needed to reach $3\sigma$ and $5\sigma$ hep discovery |
| `figures/hep_central_pl_exposure` | centralAPA: profile-likelihood significance vs. exposure |
| `figures/hep_central_pl_adaptive` | centralAPA: PL significance with and without adaptive rebinning |
| `figures/hep_central_asimov_adaptive` | centralAPA: Asimov significance with and without adaptive rebinning |
| `figures/hep_lateral_exposure` | lateralAPA: Gaussian and Asimov exposure threshold for hep discovery |
| `figures/hep_lateral_pl_exposure` | lateralAPA: profile-likelihood significance vs. exposure |
| `figures/hep_lateral_pl_adaptive` | lateralAPA: PL significance with and without adaptive rebinning |
| `figures/hep_lateral_asimov_adaptive` | lateralAPA: Asimov significance with and without adaptive rebinning |

---

## 5. Sensitivity Analysis {#sensitivity}

### 5.1 Overview

The Sensitivity analysis maps how well the detector can distinguish between the solar ($\Delta m^2_\odot$) and reactor ($\Delta m^2_{\mathrm{react}}$) best-fit oscillation hypotheses. Rather than computing a single discovery significance it evaluates a $\chi^2$ surface over the oscillation-parameter grid $(\Delta m^2,\,\sin^2\theta_{13},\,\sin^2\theta_{12})$.

Key differences from HEP:

- Histograms are **two-dimensional** (energy $\times$ azimuth $\cos\eta$).
- **Two** normalization nuisances are floated: $\Apred$ (signal) and $\Abkg$ (background).
- The nuisance stationarity conditions have **no closed-form solution** — unlike HEP's quadratic for $\bhat$ — requiring full 2D numerical minimization.
- The goal is a $\chi^2$ *map*, not a single significance.

### 5.2 Two-Dimensional Template Construction

Signal and background are represented as 2D arrays with axes energy ($j=1,\ldots,J$) and azimuth ($i=1,\ldots,I$).

**Signal template.** For oscillation parameters $\vec{\theta} = (\Delta m^2,\sin^2\theta_{13},\sin^2\theta_{12})$, the signal template at exposure $\Ecal$ is:

$$\Tsig(\vec{\theta}) = \Ecal\cdot\Mdet\cdot\bigl[P(\vec{\theta})\,H\bigr]_{ij}$$

where $P(\vec{\theta})$ is the oscillation-probability matrix (azimuth $\times$ neutrino flavour) and $H$ is the detector energy-response matrix (neutrino energy $\times$ reconstructed energy), implemented in `14SensitivitySignalTemplate.py`.

**Background template.** The background template $\Tbkg$ is independent of oscillation parameters and is constructed from detector simulations in `14SensitivityBackgroundTemplate.py`.

### 5.3 Asimov Dataset Construction

For each scan point $\vec{\theta}_k$ the Asimov observed dataset is:

$$\oij(\vec{\theta}_k) = \Tsig(\vec{\theta}_k) + \Tbkg$$

Two *reference* templates fixed at the solar and reactor best-fit points:

$$p^{\mathrm{solar}}_{ij} = \Tsig(\vec{\theta}_{\odot}), \qquad p^{\mathrm{react}}_{ij} = \Tsig(\vec{\theta}_{\mathrm{react}})$$

### 5.4 Objective Function: Baker-Cousins Poisson Deviance

The fit minimizes a Baker-Cousins Poisson deviance [Baker & Cousins 1984] with two free normalization nuisances:

$$\chi^2(\Apred,\Abkg) = 2\sum_{i,j}\Delta\ell_{ij} + \left(\frac{\Apred}{\spred}\right)^{\!2} + \left(\frac{\Abkg}{\sbkg}\right)^{\!2}$$

where the expected model is:

$$\eij = (1+\Abkg)\,\Tbkg + (1+\Apred)\,p_{ij}$$

and the per-bin Poisson deviance is:

$$\Delta\ell_{ij} = \begin{cases} \eij - \oij + \oij\ln(\oij/\eij) & \oij>0,\; \eij>0,\\ \eij & \oij=0,\; \eij>0,\\ 0 & \eij=0 \;\text{(or masked).} \end{cases}$$

Each term $\Delta\ell_{ij}\ge0$ by the Gibbs inequality; the sum is zero if and only if $\eij=\oij$ for all bins.

**Structural connection to HEP.** Under $H_0$ (no signal, $p_{ij}=0$), the $\chi^2$ reduces to the HEP log-likelihood ratio with one nuisance per normalisation parameter. The Sensitivity objective is therefore the direct 2D two-nuisance generalization of the HEP test statistic.

### 5.5 Nuisance Parameters: Constraints and Non-Analyticity

The penalty terms are Gaussian constraints identical in form to the HEP penalty $[(\bhat-1)/\srel]^2$, but with separate widths: $\spred = 4\%$ (signal flux uncertainty) and $\sbkg = 2\%$ (background uncertainty). These are configured via `analysis/config.json` under `ANALYSIS_UNCERTAINTIES.SENSITIVITY`.

**Why no analytic solution exists.** In HEP, $e_i = \beta\bi$ under $H_0$: the single global $\beta$ factors out and the stationarity condition collapses to a scalar quadratic. In Sensitivity, $\eij=(1+\Abkg)\Tbkg+(1+\Apred)p_{ij}$ mixes $\Tbkg$ and $p_{ij}$ in every bin. The two stationarity conditions are:

$$\begin{align}
\sum_{ij} p_{ij}\!\left(1-\frac{\oij}{\eij}\right) &= -\frac{\Apred}{\spred^2},\\
\sum_{ij} \Tbkg\!\left(1-\frac{\oij}{\eij}\right) &= -\frac{\Abkg}{\sbkg^2},
\end{align}$$

which are jointly non-linear in $(\Apred,\Abkg)$ because $\eij$ in the denominators depends on both unknowns. Full 2D minimization is required.

### 5.6 Barlow-Beeston Mask for 2D Templates

Bins where the background template is zero are excluded:

$$\mathcal{M}_{ij} = \mathbf{1}\!\left[\Tbkg > 0\right]$$

passed as `bb_mask` to `Sensitivity_Fitter`. This prevents spurious large deviance from signal-only bins lacking a reliable background model, following the same spirit as the HEP Barlow-Beeston mask.

### 5.7 Minimization: scipy L-BFGS-B

The objective is minimized over $(\Apred,\Abkg)$ using the L-BFGS-B algorithm [Byrd et al. 1995; Zhu et al. 1997]:

$$(\hat{A}_{\mathrm{pred}},\hat{A}_{\mathrm{bkg}}) = \arg\min_{\Apred,\Abkg}\chi^2(\Apred,\Abkg)$$

with box constraints $|\Apred| \le 10\,\spred$, $|\Abkg| \le 10\,\sbkg$. L-BFGS-B uses gradient information and is well-suited to smooth, convex objectives with box constraints. Implemented in `lib/lib_root.py:Sensitivity_Fitter.Fit`.

### 5.8 Oscillation Grid Scan and Cut Quality Score

For each analysis cut $(N_{\mathrm{hits}},N_{\mathrm{ophits}},N_{\mathrm{adjcl}})$ and each grid point $\vec{\theta}_k$:

1. Construct the Asimov dataset $\oij(\vec{\theta}_k)$.
2. Minimize $\chi^2$ against $p^{\mathrm{solar}}$: obtain $\chi^2_\odot(\vec{\theta}_k)$.
3. Minimize $\chi^2$ against $p^{\mathrm{react}}$: obtain $\chi^2_{\mathrm{react}}(\vec{\theta}_k)$.

The **cut quality score** is the average wrong-hypothesis $\chi^2$ (larger $\Rightarrow$ better discrimination):

$$\mathrm{Score}(\mathrm{cut}) = \tfrac{1}{2}\!\left[ \chi^2_\odot(\vec{\theta}_{\mathrm{react}}) + \chi^2_{\mathrm{react}}(\vec{\theta}_\odot) \right]$$

The resulting $\chi^2$ surface over the oscillation grid gives the sensitivity contours.

### 5.9 Template Construction Pipeline

The 2D templates $T^{\mathrm{sig}}$ and $T^{\mathrm{bkg}}$ are assembled from three physical inputs via a two-stage convolution.

| Figure | Description |
|---|---|
| `figures/sens_conv_nadir` | Nadir time-fraction $p(\cos\eta)$ at DUNE latitude (44.35°N), derived from solar ephemeris |
| `figures/sens_conv_oscillogram` | Solar-neutrino oscillogram $P(\nu_e\!\to\!\nu_e;\,E_\nu,\cos\eta)$ at best-fit parameters ($\Delta m^2_{21}=6\times10^{-5}$ eV², $\sin^2\theta_{12}=0.303$) |
| `figures/sens_conv_background1d` | 1D background spectrum $b(E_{\mathrm{reco}})$ after optimal cut selection |
| `figures/sens_conv_signal1d` | 1D marginal signal spectrum $s(E_{\mathrm{reco}})=\sum_\eta T^{\mathrm{sig}}$ at solar best-fit and 30 kt·yr |
| `figures/sens_conv_background_template` | 2D background template $T^{\mathrm{bkg}}(\cos\eta,\,E_{\mathrm{reco}})$ |
| `figures/sens_conv_signal_template` | 2D signal template $T^{\mathrm{sig}}(\cos\eta,\,E_{\mathrm{reco}})$; nadir-dependent wiggles encode the MSW day-night modulation |

The background template $T^{\mathrm{bkg}}$ is formed by weighting $b(E_{\mathrm{reco}})$ by $p(\cos\eta)$ via `_project_1d_to_2d`; $T^{\mathrm{sig}}$ is the result of the full convolution $T^{\mathrm{sig}} = P_{\nu_e\to\nu_e}\cdot h^T$, where $h$ is the $(E_{\mathrm{reco}}\times E_{\mathrm{true}})$ smearing matrix.

### 5.10 Results

| Figure | Description |
|---|---|
| `figures/sens_central_solar_sin12` | centralAPA: $\Delta\chi^2$ contours in $(\sin^2\theta_{12},\,\Delta m^2_{21})$ under the solar oscillation hypothesis |
| `figures/sens_central_react_sin12` | centralAPA: $\Delta\chi^2$ contours under the reactor hypothesis |
| `figures/sens_central_significance` | centralAPA: hypothesis-separation significance vs. exposure at 30 kt·yr |
| `figures/sens_lateral_solar_sin12` | lateralAPA: $\Delta\chi^2$ contours under the solar hypothesis |
| `figures/sens_lateral_react_sin12` | lateralAPA: $\Delta\chi^2$ contours under the reactor hypothesis |
| `figures/sens_lateral_significance` | lateralAPA: hypothesis-separation significance vs. exposure |

---

## 6. Comparison Across Analyses {#comparison}

| Feature | Day-Night | HEP | Sensitivity |
|---|---|---|---|
| Observable | 1D rate-difference spectrum | 1D energy spectrum | 2D (energy × azimuth) |
| Signal | $\si{=}\Ecal\theta_s(r_i^{\rm night}{-}r_i^{\rm day})$ | $\si{=}\Ecal r_i^{\rm hep}$ | $\Tsig(\vec\theta)$ |
| Goal | Detect day-night asymmetry | Discover hep flux | Map $(\Delta m^2,\sin^2\theta)$ contour |
| Significance type | Gaussian (diagnostic), two-sample Asimov (primary) | Gaussian, Asimov, PL | $\chi^2$ surface |
| Nuisances | None (error propagated) | $1\times\beta$ (background) | $\Apred + \Abkg$ |
| Nuisance solution | — | Closed-form quadratic | L-BFGS-B (2D) |
| Penalty form | — | $[(\bhat{-}1)/\srel]^2$ | $(\Apred/\spred)^2{+}(\Abkg/\sbkg)^2$ |
| Per-bin deviance | Two-sample LLR | Baker-Cousins per bin | Baker-Cousins 2D |
| Effective background | $\Beff$ (inv-fraction-weighted) | $\bi = \Ecal\sum_{c\neq\mathrm{hep}} r_i^c$ | $\Tbkg$ (2D template) |
| Barlow-Beeston mask | No | $N_i^{\rm MC}\ge N_{\rm MC}^{\min}$ | $\Tbkg>0$ |
| Asymmetry/signal band | $\theta_s\in\{1\pm\epstot,1\}$ | $d\in\{2.9,3.0,3.1\}$ | — |
| PL post-processing | — | Gauss($\sigma{=}6$) + PAVA (optional) | — |
| Spike filter | — | $\Delta_{\max}$ on pre-PAVA | — |
| Output | $Z$ vs. exposure | $Z$ vs. exposure | $\chi^2$ map over grid |

**Narrative progression.** The three analyses exhibit a clear progression in statistical complexity centred on the treatment of nuisance parameters.

The Day-Night analysis makes no nuisance-parameter approximations: signal and background are evaluated exactly in two separate exposure intervals, and the asymmetry uncertainty is propagated analytically as a scale factor on the signal. The three-source background uncertainty in the Error Gaussian is the most detailed error propagation of the three analyses, but no LLR profiling is performed.

The HEP analysis introduces the full profile-likelihood framework. The critical observation is that a single global $\beta$ correlated across all bins leads to a scalar stationarity equation, whose positive root is a closed-form quadratic. This makes the PL computation exact and fast. PAVA post-processing and adaptive rebinning address practical numerical issues (oscillations at low $S/B$; low-statistics tail bins).

The Sensitivity analysis takes the same Baker-Cousins Poisson deviance as HEP's per-bin LLR, but extends it to 2D and introduces a second nuisance $\Apred$ that couples the model both to signal and background simultaneously. This coupling breaks the factorization that enabled the HEP quadratic, requiring a full numerical 2D minimization. The result is a $\chi^2$ map rather than a significance curve.

---

## 7. Summary of Significance Outputs {#summary}

| Analysis | Label | Histogram | Rebinning | Estimator | Monotone? |
|---|---|---|---|---|---|
| **Day-Night** | `RawGaussian` | Raw | None | Gaussian | No |
| | `RawErrorGaussian` | Raw | None | Gaussian + $\seff$ | No |
| | `Gaussian` | Smooth | None | Gaussian | No |
| | `ErrorGaussian` | Smooth | None | Gaussian + $\seff$ | No |
| | `RawAsimov` | Raw | None | Two-sample LLR (penalised) | No |
| | `Asimov` | Smooth | None | Two-sample LLR (penalised) | No |
| | `Sigma2` / `AsimovSigma2` | Smooth | None | Asimov exposure threshold (alias pair) | — |
| **HEP** | `RawGaussianNoRebin` | Raw | None | Gaussian | No |
| | `RawAsimovNoRebin` | Raw | None | Asimov | No |
| | `RawGaussian` | Raw | Adaptive | Gaussian | Yes |
| | `RawAsimov` | Raw | Adaptive | Asimov | Yes |
| | `GaussianNoRebin` | Smooth | None | Gaussian | No |
| | `AsimovNoRebin` | Smooth | None | Asimov | No |
| | `Gaussian` | Smooth | Adaptive | Gaussian | Yes |
| | `Asimov` | Smooth | Adaptive | Asimov | Yes |
| | `RawProfileLikelihood` | Raw | None | PL | Yes (post) |
| | `ProfileLikelihood` | Smooth | None | PL | Yes (post) |
| | `PreIsotonicProfileLikelihood` | Smooth | None | PL (pre-PAVA) | No |
| **Sensitivity** | `chi2_solar` / `chi2_react` | 2D smooth | None | Baker-Cousins | — |

---

## 8. Recent Statistical Methodology Updates {#stat-updates}

### 8.1 Sensitivity Analysis: Fitting Methodology Correction

**Problem:** The original Sensitivity analysis implementation (legacy mode) fitted **both** signal amplitude ($A_{\mathrm{pred}}$) and background normalization ($A_{\mathrm{bkg}}$) as free parameters in the $\chi^2$ minimization. This caused physically incorrect behavior: when background uncertainty increased, the background normalization could absorb signal mismatches, causing sensitivity contours to **shrink** (tighten) instead of **loosen** — the opposite of expected physical behavior.

**Solution:** Introduced the `--fit_background` control flag in `Sensitivity_Fitter` and `06_significance.py`:

- **Corrected mode (new default):** `--no-fit_background` fixes $A_{\mathrm{bkg}} = 0$ (background normalization at nominal). Only $A_{\mathrm{pred}}$ is fitted. This produces physically meaningful sensitivity where contours properly loosen with increased background uncertainty.
- **Legacy mode:** `--fit_background` enables the old behavior (both parameters free) for validation against historical results.

**Implementation details:**
- When `fit_background=False`, the background term in the likelihood becomes $e_{ij} = \Tbkg + (1+A_{\mathrm{pred}})\cdot p_{ij}$, and only the signal pull term $(A_{\mathrm{pred}}/\spred)^2$ appears in the $\chi^2$ penalty.
- When `fit_background=True` (legacy), $e_{ij} = (1+A_{\mathrm{bkg}})\Tbkg + (1+A_{\mathrm{pred}})p_{ij}$, with both pull terms active.
- The stationarity conditions for the corrected mode reduce to a single parameter optimization (exact for 1D, L-BFGS-B for 2D), avoiding the coupled non-linearity that required full 2D minimization in legacy mode.

**Validation:** Study variants `unc_bkg4_nobkgfit`, `unc_bkg6_nobkgfit`, `unc_sig6_nobkgfit` demonstrate the corrected behavior. The `fit_background` study group provides legacy validation.

### 8.2 Sensitivity Analysis: Delta Chi-Square Contour Plotting

**Problem:** Contours were originally drawn at fixed **absolute** $\chi^2$ thresholds (e.g., 0, 1, 4, 9). When overall $\chi^2$ values increased (e.g., due to higher background uncertainty), points would be excluded from the contour, causing it to shrink artificially.

**Solution:** Contours now use **Delta chi-square** $\Delta\chi^2 = \chi^2 - \chi^2_{\min}$:

- This ensures contours represent proper confidence levels: $\Delta\chi^2 = 1, 4, 9$ for 1, 2, 3$\sigma$ confidence.
- The `Chi2Min` field is now stored in contour DataFrames for diagnostics.
- Implementation: `contour_plot.py` computes $\chi^2_{\min}$ as the minimum finite value in each grid, then plots $\sqrt{\Delta\chi^2}$.

**Mathematical justification:** For a $\chi^2$ distribution with $k$ degrees of freedom, the confidence level is determined by the difference from the minimum, not the absolute value. Using absolute thresholds conflates the overall goodness-of-fit with the shape of the likelihood surface.

### 8.3 Template Normalization: Per-Year Storage (v2)

**Problem:** Original templates (v1) were pre-scaled by their creation exposure. Rescaling them for different exposures would compound the scaling, producing incorrect results.

**Solution:** Templates are now stored **per-year** (v2), with normalization metadata in `TEMPLATE_NORMALIZATION.json`:

- Templates represent rates: units of $(\mathrm{yr\cdot kt\cdot MeV})^{-1}$ integrated over bin width.
- Exposure scaling is applied at load time via `scale_to_exposure(arr, exposure_yr)`, which also zeros bins below 1 expected event.
- Validation: `require_per_year_templates()` refuses v1 templates with a descriptive error message.
- Benefit: A single template set serves all exposure values without regeneration.

### 8.4 Day-Night Analysis: Asymmetry Uncertainty Bands

The Day-Night analysis evaluates three asymmetry amplitude scenarios to bracket the theoretical uncertainty:

- **Total band:** $\epstot = \sqrt{\varepsilon_{\oplus}^2 + \varepsilon_{\mathrm{osc}}^2}$, where $\varepsilon_{\oplus}$ is the Earth density band (default 0.13, from PREM-based oscillation probability calculations) and $\varepsilon_{\mathrm{osc}}$ is the oscillation parameter band (default 0.05, from PDG ranges on $\theta_{12}$ and $\Delta m^2_{21}$).
- **Scale factors:** $\theta_s \in \{1+\epstot,\,1,\,1-\epstot\}$ (indices 0, 1, 2), corresponding to upper, nominal, and lower asymmetry predictions.

Each scale factor is evaluated independently without penalty terms. The Asimov log-likelihood ratio is computed separately for each scenario:

$$q_0(\theta_s) = \sum_{i \ge i_{\mathrm{th}}} 2\left[ n_i^{\mathrm{night}}(\theta_s)\ln\frac{n_i^{\mathrm{night}}(\theta_s)}{h_{0,i}^{\mathrm{night}}} + n_i^{\mathrm{day}}(\theta_s)\ln\frac{n_i^{\mathrm{day}}(\theta_s)}{h_{0,i}^{\mathrm{day}}} \right]$$

where the observed counts under asymmetry scale $\theta_s$ are:
$$n_i^{\mathrm{night}}(\theta_s) = \Ecal\,g\,(r_i^{\mathrm{bkg}} + r_i^{\mathrm{night}} + \theta_s\,(r_i^{\mathrm{night}} - r_i^{\mathrm{day}}))$$  
$$n_i^{\mathrm{day}}(\theta_s) = \Ecal\,f\,(r_i^{\mathrm{bkg}} + r_i^{\mathrm{day}})\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad$$

**Note on asymmetric bands:** The LLR $q_0$ is non-linear in $\theta_s$ (Poisson likelihoods, per-bin normalisation). Therefore `Gaussian+Error` (at $\theta_s=1+\epstot$) and `Gaussian-Error` (at $\theta_s=1-\epstot$) need not be symmetric about `Gaussian` (at $\theta_s=1$). This asymmetry is a genuine feature of the Poisson statistics, not an artifact.

**Note on band ordering:** Because the bands are unpenalised, it is possible for the upper band ($\+\epstot$) to produce a lower significance than the nominal when the detector is in a regime where the marginal gain from increased asymmetry is small. This indicates the physical prediction is saturated — additional asymmetry provides no extra discriminating power.


---



## 9. Workflow Flags and Configuration {#flags}

**Orchestrator flags in `run_sensitivity.py`:**

| Flag | Default | Effect when disabled |
|---|---|---|
| `--computation` | True | Skip all computation; run only plot macros |
| `--significance` | True | Skip `12DayNight.py`, `13HEP.py`, `14Sensitivity*.py` |
| `--fiducialization` | True | Skip `0XFiducializeSignal.py`, `0YBestFiducial.py` |
| `--rebin` | True | Skip `11AnalysisSignal.py` adaptive rebinning |
| `--optimization` | False | Run smoothing sigma optimizer `optimize_smoothing.py` |

**Per-analysis workflow feature flags** (controlled by `analysis/config.json` under `WORKFLOW.{ANALYSIS}`):

| Key | Default | Effect when true |
|---|---|---|
| `DAYNIGHT.background_error` | `true` | Compute Error Gaussian curves |
| `DAYNIGHT.significance_bins` | `true` | Save per-bin significance spectra at the display exposure |
| `HEP.pl_signal_bands` | `true` | Evaluate three signal normalizations for $\pm1\sigma_s$ PL bands |
| `HEP.pl_isotonic` | `false` | Apply Gaussian+PAVA post-processing to PL curves |
| `HEP.significance_bins` | `false` | Save per-bin significance spectra at the display exposure |

---

## 10. References {#references}

[Cowan 2010] G. Cowan, K. Cranmer, E. Gross, O. Vitells, *Asymptotic formulae for likelihood-based tests of new physics*, Eur. Phys. J. C **71** (2011) 1554. <https://arxiv.org/abs/1007.1727>

[Baker & Cousins 1984] S. Baker, R.D. Cousins, *Clarification of the use of chi-square and likelihood functions in fits to histograms*, Nucl. Instrum. Meth. **221** (1984) 437–442. <https://doi.org/10.1016/0167-5087(84)90016-4>

[Barlow & Beeston 1993] R. Barlow, C. Beeston, *Fitting using finite Monte Carlo samples*, Comput. Phys. Commun. **77** (1993) 219–228. <https://doi.org/10.1016/0010-4655(93)90005-W>

[Robertson 1988] T. Robertson, F.T. Wright, R.L. Dykstra, *Order Restricted Statistical Inference*, Wiley, 1988.

[Byrd et al. 1995] R.H. Byrd, P. Lu, J. Nocedal, C. Zhu, *A limited memory algorithm for bound constrained optimization*, SIAM J. Sci. Comput. **16** (1995) 1190–1208.

[Zhu et al. 1997] C. Zhu, R.H. Byrd, P. Lu, J. Nocedal, *Algorithm 778: L-BFGS-B: Fortran subroutines for large-scale bound-constrained optimization*, ACM Trans. Math. Softw. **23** (1997) 550–560.
