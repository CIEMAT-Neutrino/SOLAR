# SOLAR/DUNE Significance Analyses
## Mathematical Derivations: Day-Night, HEP, and Sensitivity

*SOLAR/DUNE Analysis — May 2026. Sensitivity sections (§5, §8.1, §8.5, §9) revised 2026-09-14:
§5 is now the complete description of the Sensitivity analysis, from template construction through
the pull statistic, grid scan, validation and systematic studies to data products and
configuration.*

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
\newcommand{\pij}{p_{ij}}
\newcommand{\muij}{\mu_{ij}}
\newcommand{\muzero}{\mu^{0}_{ij}}
\newcommand{\ses}{\sigma_{E}}
\newcommand{\ssin}{\sigma_{13}}
$$

---

## Contents

1. [Overview and Scientific Goals](#overview)
2. [Common Ingredients](#common)
3. [Day-Night Asymmetry Analysis](#daynight)
4. [HEP Discovery Analysis](#hep)
5. [Sensitivity Analysis](#sensitivity) — *full description; subsections:*
   [5.1 Goal](#sens-goal) ·
   [5.2 Observable](#sens-observable) ·
   [5.3 Templates](#sens-templates) ·
   [5.4 Asimov data](#sens-asimov) ·
   [5.5 Deviance](#sens-deviance) ·
   [5.6 Response model](#sens-response) ·
   [5.7 Profiled $\chi^2$](#sens-profile) ·
   [5.8 Bin masking](#sens-mask) ·
   [5.9 Nuisance profiles](#sens-profiles) ·
   [5.10 Grid scan & cuts](#sens-scan) ·
   [5.11 Contours](#sens-contours) ·
   [5.12 Validation](#sens-validation) ·
   [5.13 Studies](#sens-studies) ·
   [5.14 Background uncertainty](#sens-bkg-prior) ·
   [5.15 Legacy fitter](#sens-legacy) ·
   [5.16](#sens-conv-figures)–[5.17 Figures](#sens-results) ·
   [5.18 Data products & configuration](#sens-products) ·
   [5.19 Reproduction](#sens-reproduction)
6. [Comparison Across Analyses](#comparison)
7. [Summary of Significance Outputs](#summary)
8. [Recent Statistical Methodology Updates](#stat-updates)
9. [Workflow Flags and Configuration](#flags)
10. [References](#references)

---

## 1. Overview and Scientific Goals {#overview}

This document presents the mathematical derivations underlying the three significance analyses in the SOLAR/DUNE software framework, ordered from simplest to most complex:

1. **Day-Night Asymmetry** (`daynight/01_daynight.py`): searches for a time-modulated excess of solar neutrinos during nighttime relative to daytime, driven by the MSW matter effect inside the Earth. Significance is computed from both a Gaussian approximation on the rate-difference spectrum and a two-sample Poisson log-likelihood ratio (Asimov).

2. **HEP Discovery** (`hep/01_hep.py`): searches for the hep solar neutrino flux ($^3\mathrm{He}+p \to {}^4\mathrm{He}+e^++\nu_e$) as an absolute excess above background. Three significance estimators are computed: Gaussian, Asimov, and a profile-likelihood (PL) significance with a single global background nuisance, analytically profiled.

3. **Sensitivity** (`sensitivity/06_significance.py`): maps the sensitivity contours in the oscillation-parameter planes spanned by $(\Delta m^2_{21},\,\sin^2\theta_{12},\,\sin^2\theta_{13})$. A Baker-Cousins Poisson deviance over 2D templates in reconstructed energy and solar nadir angle is profiled over up to four nuisance parameters — signal and background normalisation, reconstructed energy scale and $\sin^2\theta_{13}$ — through a linear response model with Gaussian priors.

The three analyses share common ingredients (histogram smoothing, thresholds, adaptive rebinning for HEP) but differ in the statistical complexity of their model. See [Comparison Across Analyses](#comparison) for a summary table.

**Common notation.** Throughout, $T$ is the exposure in kt·yr, $\Mdet$ the active detector mass in kt, and the exposure factor is $\Ecal = T\,\Mdet$. Component label $c$ identifies one physical process (signal, neutron, gamma, radiological, ${}^8\mathrm{B}$).

Bin indices follow the array layout of each analysis. For the 1D analyses (Day-Night, HEP) index $i$ runs over energy bins. For Sensitivity the templates are 2D arrays of shape (nadir, energy), so $i$ runs over the 40 nadir bins and $j$ over the 30 energy bins, and a bin is written $ij$.

---

## 2. Common Ingredients {#common}

### 2.1 Rate Histograms and Exposure Scaling

All three analyses work with per-unit-exposure, per-unit-mass rate histograms $\rc$ (units: $(\mathrm{yr\cdot kt\cdot MeV})^{-1}$, integrated over the bin width $\Delta E$). Expected event counts at exposure $\Ecal$ are:

$$\muc = \Ecal\,\rc = T\,\Mdet\,\rc$$

All analyses apply a reconstructed-energy threshold $E_{\mathrm{th}}$ (configured in `analysis/config.json`); only bins with $E_i \ge E_{\mathrm{th}}$ enter the significance computation.

### 2.2 Histogram Smoothing {#histogram-smoothing}

Each rate histogram $\rc$ is convolved with a one-dimensional Gaussian kernel before entering the significance computation:

$$\tilde{r}_{i}^{c} = \sum_{j} G_\sigma(i-j)\, r_j^{c}, \qquad G_\sigma(k) = \frac{1}{\sqrt{2\pi}\,\sigma} \exp\!\left(-\frac{k^2}{2\sigma^2}\right)$$

implemented via `scipy.ndimage.gaussian_filter1d` with `mode='nearest'`. Whether smoothing is applied to a given component, and the width $\sigma$, are configured in `analysis/smoothing.json` under `SMOOTHING.ANALYSES.{ANALYSIS}.STAGES`. Separate stages are provided for the fiducial scan and the significance computation.

**Non-negativity clipping.** Gaussian convolution at distribution tails can produce small negative values. Negative rates are unphysical and cause profile-likelihood divergences at high exposure (see [Numerical Stability](#numerical-stability)). All smoothed rates are therefore clipped to zero before any significance computation:

$$\tilde{r}_{i}^{c} \leftarrow \max\!\left(0,\,\tilde{r}_{i}^{c}\right)$$

### 2.3 Per-Component Background Error Model {#background-error}

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

### 3.6 Two-Sample Poisson Asimov Significance {#three-six}

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

*Complete description of the Sensitivity analysis: physical goal, template construction, test
statistic, nuisance treatment, grid scan, contour construction, validation, systematic studies,
data products and configuration.*

### 5.1 Scientific Goal and Structure {#sens-goal}

The Sensitivity analysis quantifies the precision with which a DUNE far-detector module can
determine the solar oscillation parameters $(\Delta m^2_{21},\,\sin^2\theta_{12},\,\sin^2\theta_{13})$
from the $^8$B solar-neutrino event sample, and in particular whether it can discriminate between
the two hypotheses that currently disagree at the $\sim2\sigma$ level:

- the **solar** global-fit value, $\Delta m^2_\odot = 6.0\times10^{-5}\ \mathrm{eV}^2$;
- the **reactor** (KamLAND) value, $\Delta m^2_{\mathrm{react}} = 7.54\times10^{-5}\ \mathrm{eV}^2$.

It is an Asimov sensitivity study: no pseudo-experiments are thrown. For each point $\vec\theta_k$
of a three-parameter oscillation grid a noiseless expected dataset is constructed and fitted
against two fixed reference hypotheses, producing a pair of $\chi^2$ surfaces from which confidence
contours are drawn.

Key differences from HEP (§4):

- Histograms are **two-dimensional** (reconstructed energy $\times$ solar nadir angle $\cos\eta$).
- Systematics enter as **profiled nuisance parameters with Gaussian priors** — signal
  normalisation, background normalisation, reconstructed energy scale and $\sin^2\theta_{13}$ —
  rather than as a propagated error band.
- The nuisances are profiled through a **linear response model**, which admits a closed-form
  Gaussian solution and a fast safeguarded Newton iteration on the exact Poisson objective.
- The goal is a $\chi^2$ *map* over parameter space, not a single significance versus exposure.

### 5.2 Observable and Binning {#sens-observable}

The observable is the two-dimensional distribution of selected cluster events in
$(\cos\eta,\;E_{\mathrm{reco}})$, where $\eta$ is the solar **nadir angle** at the detector and
$E_{\mathrm{reco}}$ is the reconstructed neutrino energy (`SolarEnergy` by default; §5.13 lists the
variants).

| Axis | Range | Bins | Configuration key |
|---|---|---|---|
| $\cos\eta$ | $[-1,\,1]$ | 40 | `NADIR_BINS` |
| $E_{\mathrm{reco}}$ | $[0,\,30]$ MeV | 30 (1 MeV) | `sensitivity_rebin` (`lib/__init__.py`) |

Templates are therefore $40\times30$ arrays, indexed below by $i$ (nadir) and $j$ (energy).

The nadir axis carries the physics that separates this analysis from a pure spectral fit: the
neutrino path length through the Earth, and hence the MSW regeneration probability, depends on
$\cos\eta$ alone. The solar and reactor $\Delta m^2_{21}$ hypotheses differ both in the low-energy
shape of the survival probability and in the amplitude and nadir structure of Earth regeneration,
so the two-dimensional distribution discriminates between them more strongly than either
projection.

### 5.3 Template Construction {#sens-templates}

#### Signal templates

For oscillation parameters $\vec{\theta} = (\Delta m^2,\sin^2\theta_{13},\sin^2\theta_{12})$ the
per-year signal template is the product of an oscillation-probability matrix and the detector
response,

$$\Tsig(\vec\theta) \;=\; \Mdet \sum_{k} P_{\nu_e\to\nu_e}\!\left(\vec\theta;\,E^{\mathrm{true}}_k,\,\cos\eta_i\right)H_{kj},$$

where $H_{kj}$ is the $(E^{\mathrm{true}}\times E^{\mathrm{reco}})$ smearing matrix built from the
fiducialised MARLEY sample and $\Mdet$ is the fiducial mass in kt. Construction is performed by
`src/physics/sensitivity/03_template_compute.py` invoked with `--template signal`; this is the
entry point the pipeline uses, and the standalone `01_background_template.py` /
`02_signal_template.py` scripts remain for manual regeneration.

The survival probability is obtained from **NuFast** (`OSCILLATION_BACKEND = "nufast"`), evaluated
on a fine grid of `OSC_ENERGY_BINS = 120` points over `OSC_ENERGY_RANGE = [0, 30]` MeV.

#### Nadir oversampling

$P_{\nu_e\to\nu_e}$ is **not** sampled at the centre of each of the 40 nadir bins. Each bin is
subdivided into `OSC_NADIR_OVERSAMPLE = 4` sub-bins, the probability is evaluated in each, and the
result is averaged with the nadir-exposure PDF as weight; the fine energy grid is likewise averaged
within each 1 MeV reconstruction bin.

This is not a refinement of convenience. Point-sampling the bin centres aliases the
Earth-regeneration oscillation against the nadir binning and paints whole-row stripes into the
$\chi^2$ grids: at $\Delta m^2_{21} = 3.83\times10^{-5}\ \mathrm{eV}^2$ in the HD geometry,
neighbouring grid rows differed by $\Delta\chi^2 = 38$ at an absolute level of 23, biasing
$\Delta\chi^2$ near the $3\sigma$ contour by up to $\sim4$. The oversampling is recorded in the
template sampling marker and checked by `lib/template_guards.py` before a scan is allowed to reuse
existing templates.

#### Flyweight evaluation

Evaluating all 14 702 grid points as stored templates is possible (`--no-flyweight`) but costly in
storage. The default `--flyweight` mode stores a single **base template**
$\mathrm{BASE}_{kj} = \Mdet H_{kj}$ per selection cut and forms $\Tsig(\vec\theta)$ on the fly as
the matrix product of the freshly computed oscillation map with the base template
(`_flyweight_convolve`, `06_significance.py:696`). Flyweight mode requires the `nufast` backend;
any other choice of `--oscillation_backend` is overridden, with a warning.

#### Background template

$\Tbkg$ is independent of the oscillation parameters. It is built from the radiological and
cosmogenic simulation samples by `03_template_compute.py --template background`, and the nadir
dependence is imposed by weighting the one-dimensional background spectrum $b(E_{\mathrm{reco}})$
with the nadir-exposure PDF $p(\cos\eta)$ via `_project_1d_to_2d`.

The background is overwhelmingly larger than the signal. For the reference configuration
(`hd_1x2x6_centralAPA`, Truncated fiducialisation, cut `NHits1/AdjCl4/OpHits8`, 30 yr) the totals
are

$$B = 4.18\times10^{12}\ \text{events}, \qquad S = 3.01\times10^{5}\ \text{events}, \qquad S/B = 7.2\times10^{-8},$$

with the background concentrated below $\sim8$ MeV and the signal extending to $\sim20$ MeV. The
discriminating power therefore resides almost entirely in the high-energy tail. This hierarchy has
direct consequences for the treatment of background systematics, developed in §5.14.

#### Per-year normalisation and exposure scaling

Templates are stored **per year** (normalisation schema v2, `TEMPLATE_NORMALIZATION.json`, units
`detector_mass_kT × rate`, one file per template directory; see §8.3). Absolute counts are formed
at load time:

$$X_{ij}(\Ecal) \;=\; \Ecal\,X_{ij}^{\mathrm{yr}}, \qquad X_{ij}(\Ecal) \to 0 \ \ \text{if } X_{ij}(\Ecal) < 1 ,$$

implemented in `scale_to_exposure` (`06_significance.py:375`). The truncation of bins carrying less
than one expected event is exposure dependent, which is why it is applied here and not baked into
the stored templates. `require_per_year_templates()` refuses v1 (pre-scaled) templates with a
descriptive error rather than silently double-scaling them.

Two exposures are evaluated in the same pass: the primary $\Ecal = 30$ yr
(`ANALYSIS_EXPOSURES.SENSITIVITY.PRIMARY`) and the secondary $\Ecal = 10$ yr (`SECONDARY`), whose
outputs carry the filename tag `_10Y`.

### 5.4 Asimov Data and the Hypothesis Pair {#sens-asimov}

For every grid point $\vec\theta_k$ the Asimov dataset is the noiseless expectation

$$\oij(\vec\theta_k) \;=\; \Tsig(\vec\theta_k) + \Tbkg .$$

Two reference predictions are held fixed throughout the scan,

$$p^{\mathrm{solar}}_{ij} = \Tsig(\vec\theta_\odot), \qquad p^{\mathrm{react}}_{ij} = \Tsig(\vec\theta_{\mathrm{react}}),$$

evaluated at $\vec\theta_\odot = (6.0\times10^{-5},\,0.022,\,0.304)$ and
$\vec\theta_{\mathrm{react}} = (7.54\times10^{-5},\,0.022,\,0.304)$. Each Asimov dataset is fitted
against **both** references, yielding $\chi^2_\odot(\vec\theta_k)$ and
$\chi^2_{\mathrm{react}}(\vec\theta_k)$.

The same background template appears in the data and in both models; no background mismatch is
injected anywhere in the procedure. This is a deliberate choice — the study measures parameter
resolution, not robustness against background mismodelling — and it is one premise of the result in
§5.14.

### 5.5 Objective Function: Baker-Cousins Poisson Deviance {#sens-deviance}

The fit minimises a Baker-Cousins Poisson deviance [Baker & Cousins 1984] between the Asimov data
and a model expectation $\muij$:

$$D(\oij\,\|\,\muij) = 2\sum_{ij\in\mathcal{M}}\Delta\ell_{ij}, \qquad
\Delta\ell_{ij} = \begin{cases} \muij - \oij + \oij\ln(\oij/\muij) & \oij>0,\; \muij>0,\\ \muij & \oij=0,\; \muij>0,\\ 0 & \text{masked.} \end{cases}$$

Each term $\Delta\ell_{ij}\ge0$ by the Gibbs inequality; the sum vanishes if and only if
$\muij=\oij$ for all bins.

**Numerical form.** The analysis operates at $\oij\sim\muij\sim10^{9}$ per bin, where the closed
form cancels catastrophically. With $x = (\oij-\muij)/\muij$, `_poisson_deviance_terms`
(`lib/fitting.py:1449`) evaluates $2\muij\left[(1+x)\ln(1+x)-x\right]$, switching below
$|x|<10^{-3}$ to the series

$$(1+x)\ln(1+x)-x \;=\; \tfrac{x^2}{2}-\tfrac{x^3}{6}+\tfrac{x^4}{12}-\tfrac{x^5}{20}+\mathcal{O}(x^6).$$

Without this the deviance is dominated by round-off over most of the grid.

**Structural connection to HEP.** Under $H_0$ (no signal, $\pij=0$) the statistic reduces to the
HEP log-likelihood ratio with one nuisance per normalisation parameter. The Sensitivity objective
is the 2D, multi-nuisance generalisation of the HEP test statistic.

### 5.6 Nuisance Parameters as a Linear Response Model {#sens-response}

Systematics enter through a linear response model,

$$\muij(\alpha) = \muzero + \sum_k \alpha_k J^{(k)}_{ij}, \qquad \muzero = \pij + \Tbkg,$$

with $J^{(k)} = \partial\mu/\partial\alpha_k$ the response template of nuisance $k$ and $\alpha_k$
its value in physical units. The response templates are built once per scan by
`sensitivity_pull_jacobian` (`lib/fitting.py:1476`):

| $k$ | Name | $J^{(k)}_{ij}$ | Prior width | Exact? |
|---|---|---|---|---|
| 1 | `signal_norm` | $\pij$ | $\spred = 4\%$ | exact |
| 2 | `background_norm` | $\Tbkg$ | $\sbkg = 2\%$ | exact |
| 3 | `energy_scale` | $\bigl[\pij(+\ses)-\pij(-\ses)\bigr]/2\ses$ | $\ses = 2\%$ | linearised |
| 4 | `sin13` | $\partial \pij/\partial\sin^2\theta_{13}$ | $\ssin = 5.6\times10^{-4}$ | linearised |

A nuisance whose prior width is zero or absent is dropped from the problem entirely rather than
fixed at zero with a column of zeros; this keeps the linear algebra well conditioned.

Two properties of this table deserve emphasis.

**The normalisation responses are exact, not approximations.** For $k=1$,
$\muzero + \alpha_1\pij \equiv (1+\alpha_1)\pij + \Tbkg$, and likewise for $k=2$. The linear model
is a first-order approximation only for the energy-scale and $\sin^2\theta_{13}$ nuisances, whose
response templates are central differences — the energy scale from a linear interpolation of the
template onto the shifted energy axis (`_sensitivity_apply_energy_scale`), the mixing angle from a
secant between the nearest configured $\sin^2\theta_{13}$ grid neighbours of the reference value,
computed without the sub-one-event truncation so that the derivative stays smooth.

**The energy-scale response is built from the signal template only.** $J^{(3)}$ involves $\pij$ and
never $\Tbkg$. The background consequently possesses exactly one degree of freedom in the entire
fit — a global normalisation — and no shape freedom of any kind. §5.14 shows that this is not, in
this analysis, a limitation with observable consequences.

### 5.7 The Profiled $\chi^2$ and its Solution {#sens-profile}

The statistic reported per grid point and hypothesis is the profile

$$\boxed{\;\chi^2 \;=\; \min_{\alpha}\left[\,D\!\left(\oij \,\middle\|\, \muzero + \textstyle\sum_k \alpha_k J^{(k)}_{ij}\right) \;+\; \sum_k \left(\frac{\alpha_k}{\sigma_k}\right)^{2} \right]\;}$$

implemented by `sensitivity_pull_profile` (`lib/fitting.py:1516`). This is the **pull method**,
selected by `--fit_method pull`, the default since the revision recorded in §8.1.

Alongside the profiled value the routine returns two diagnostics used downstream:

- $\chi^2_0 \equiv D(\oij\|\muzero)$, the deviance with every nuisance held at nominal. Since the
  profile minimises over a set containing $\alpha=0$, the bound $\chi^2\le\chi^2_0$ holds at every
  point; the `profile_bound` validation gate (§5.12) tests exactly this.
- $\chi^2_{\mathrm{gauss}}$, the closed-form Gaussian (Pearson) profile below.

**Algorithm.** The minimisation is performed in **standardised** coordinates
$\beta_k=\alpha_k/\sigma_k$, so that every prior becomes a unit Gaussian and the penalty is simply
$\beta^{\!\top}\beta$. Writing $A_{k,ij}=\sigma_k J^{(k)}_{ij}$ restricted to the active bins and
$r = \oij - \muzero$:

1. **Closed-form Gaussian start.** Approximating the Poisson variance by
   $V=\mathrm{diag}(\muzero)$ and the deviance by the Pearson form $r^{\!\top}V^{-1}r$, the profile
   has the Woodbury solution
   $$\beta^{\mathrm{G}} = \left(\mathbb{1}+AV^{-1}A^{\!\top}\right)^{-1}AV^{-1}r, \qquad
     \chi^2_{\mathrm{gauss}} = r^{\!\top}V^{-1}r - \left(AV^{-1}r\right)^{\!\top}\beta^{\mathrm{G}} .$$
   This is exact for the Gaussian objective and seeds the Poisson one.
2. **Safeguard.** If $\beta^{\mathrm{G}}$ leads to a non-positive expectation in any bin, or to an
   objective worse than $\chi^2_0$, the iteration restarts from $\beta=0$.
3. **Newton iteration with backtracking.** With $\rho=\oij/\muij$,
   $$g = 2A(\mathbb{1}-\rho) + 2\beta, \qquad
     \mathcal{H} = 2\,A\,\mathrm{diag}\!\left(\rho/\muij\right)A^{\!\top} + 2\,\mathbb{1},$$
   and the step $\delta=-\mathcal{H}^{-1}g$ is taken with an Armijo backtracking line search
   (factor $\tfrac12$, sufficient-decrease constant $10^{-4}$). Convergence is declared when the
   Newton decrement $-g^{\!\top}\delta$ falls below $2\times10^{-9}$, with at most 50 iterations.

All small linear systems are solved by `_solve_equilibrated` (`lib/fitting.py:1465`), which applies
Jacobi (diagonal) scaling before `numpy.linalg.solve` and falls back to a least-squares solve on
`LinAlgError`. The scaling is necessary because the diagonal entries of $\mathcal{H}$ span roughly
ten decades: the background-normalisation direction is constrained by $\sim10^{12}$ events while
the $\sin^2\theta_{13}$ direction is constrained by the high-energy tail alone.

The problem is small — at most four nuisances — and convex in the region of interest, so the
procedure converges in a handful of iterations per grid point. The scan is distributed over worker
processes by `sensitivity_chi2_worker_pull` (`lib/fitting.py:1622`); the response templates depend
only on the fixed reference predictions and are therefore computed **once** per scan in the parent
process and passed to the workers, rather than being rebuilt at each of the 14 702 grid points.

**Why this supersedes a general-purpose optimiser.** The energy-scale shift becomes a profiled
parameter instead of an outer grid scan; the Gaussian start removes the dependence on an
optimiser's convergence behaviour at $10^9$ counts per bin; and the $\sin^2\theta_{13}$ constraint
can be applied inside the fit rather than as a post-hoc minimisation over the parameter grid
(§5.9).

### 5.8 Bin Masking {#sens-mask}

A bin enters the sum only if

$$\mathcal{M}_{ij} = \mathbf{1}\!\left[\muzero>0\right]\wedge\mathbf{1}\!\left[\Tbkg > 0\right],$$

the second condition being applied whenever the background template is not identically zero. This
excludes bins in which a signal is predicted but no reliable background model exists, where the
deviance would otherwise be unbounded. For the reference configuration 722 of the 1200 bins
survive. The condition is the two-dimensional analogue of the HEP Barlow-Beeston mask
[Barlow & Beeston 1993].

### 5.9 Nuisance Profiles {#sens-profiles}

Which of $k=3,4$ are active is set by a named **nuisance profile**, a key of `NUISANCE_PROFILES` in
`config/analysis/config.json`, whose entries are merged into the analysis configuration at run
time:

| Profile | `MARGINALIZE_SIN13` | `ENERGY_SCALE_UNCERTAINTY` | Active nuisances |
|---|---|---|---|
| `full` *(default)* | true | true | 1, 2, 3, 4 |
| `marginalize_sin13` | true | false | 1, 2, 4 |
| `energy_scale` | false | true | 1, 2, 3 |
| `nominal` | false | false | 1, 2 |

The signal- and background-normalisation nuisances are **not** governed by the profile: they are
present in every profile whenever their prior widths are non-zero. The `nuisance` study group runs
the four profiles against one another to decompose the sensitivity loss attributable to each
systematic. The default is `full`, the conservative choice; `run_sensitivity.py` warns at start-up
that this marginalisation reduces the apparent sensitivity relative to a fixed-parameter fit.

**Treatment of $\sin^2\theta_{13}$.** It is both a scan axis and a constrained parameter, and the
two roles must not be applied simultaneously. The code distinguishes three modes
(`06_significance.py:1238`):

- **`fit`** — `--fit_method pull` with `MARGINALIZE_SIN13` true. $\sin^2\theta_{13}$ is nuisance
  $k=4$ inside the profile, with prior $\ssin$. No further profiling is applied.
- **`grid`** — `MARGINALIZE_SIN13` true under the legacy fitter. The constraint is imposed *after*
  the scan by minimising
  $\chi^2 + [(\sin^2\theta_{13}-\overline{\sin^2\theta_{13}})/\ssin]^2$ over the
  $\sin^2\theta_{13}$ values present at each $(\Delta m^2,\sin^2\theta_{12})$ cell.
- **`none`** — `MARGINALIZE_SIN13` false. $\sin^2\theta_{13}$ is held at its reference value.

Applying the grid minimisation on top of a fit that already profiles $\sin^2\theta_{13}$ would
double-count the constraint; the code guards against this explicitly. The `grid` mode additionally
requires every $(\Delta m^2,\sin^2\theta_{12})$ cell to contain more than one $\sin^2\theta_{13}$
value — otherwise the minimisation is a no-op on the plane through the reference point and a
genuine minimisation elsewhere, painting a cross-shaped artifact through the grid. The
`sin13_profile` validation gate (§5.12) enforces this.

**Prior widths.**

| Symbol | Meaning | Value | Source |
|---|---|---|---|
| $\spred$ | $^8$B flux normalisation | 0.04 | `ANALYSIS_UNCERTAINTIES.SENSITIVITY.signal_uncertainty` |
| $\sbkg$ | background normalisation | 0.02 | `ANALYSIS_UNCERTAINTIES.SENSITIVITY.background_uncertainty` |
| $\ses$ | reconstructed energy scale | 0.02 | **code fallback** — `ENERGY_SCALE_SIGMA` absent from configuration |
| $\ssin$ | $\sin^2\theta_{13}$ | $5.6\times10^{-4}$ | **code fallback** — `SIN13_SIGMA` absent from configuration |

$\ses$ and $\ssin$ are currently taken from the literal defaults in `06_significance.py:1089` and
`:1123`, because the corresponding keys are not present in any configuration file. The values are
defensible — $\ssin = 5.6\times10^{-4}$ is the PDG uncertainty on $\sin^2\theta_{13}$ — but they
are not traceable through the configuration, and a thesis quoting them should either add them to
`config/analysis/physics.json` or state that they are code defaults.

### 5.10 Grid Scan and Selection Optimisation {#sens-scan}

#### The oscillation grid

The grid is not a dense three-dimensional cube. No Cartesian product is taken; instead
`make_oscillation_grid` (`lib/oscillation.py:98`) forms the union of four two-dimensional **planes**
and five denser one-dimensional **lines**, all passing through the reference points, as specified by
`OSCILLATION_GRID` in `config/analysis/physics.json`:

| | Scanned axes | Held fixed at | Steps |
|---|---|---|---|
| Plane 1 | $(\Delta m^2_{21},\ \sin^2\theta_{12})$ | $\sin^2\theta_{13}$ | $45\times45$ |
| Plane 2 | $(\Delta m^2_{21},\ \sin^2\theta_{13})$ | $\sin^2\theta_{12}$ | $45\times45$ |
| Plane 3 | $(\sin^2\theta_{13},\ \sin^2\theta_{12})$ | $\Delta m^2_\odot$ | $45\times45$ |
| Plane 4 | $(\sin^2\theta_{13},\ \sin^2\theta_{12})$ | $\Delta m^2_{\mathrm{react}}$ | $45\times45$ |
| Line 1 | $\Delta m^2_{21}$ | $\sin^2\theta_{13},\ \sin^2\theta_{12}$ | 50 |
| Lines 2-3 | $\sin^2\theta_{12}$ | $\sin^2\theta_{13}$, and $\Delta m^2_\odot$ / $\Delta m^2_{\mathrm{react}}$ | 50 each |
| Lines 4-5 | $\sin^2\theta_{13}$ | $\sin^2\theta_{12}$, and $\Delta m^2_\odot$ / $\Delta m^2_{\mathrm{react}}$ | 50 each |

Axis ranges are $\Delta m^2_{21}\in[3\times10^{-5},\,1\times10^{-4}]\ \mathrm{eV}^2$,
$\sin^2\theta_{13}\in[0.01,\,0.04]$ and $\sin^2\theta_{12}\in[0.15,\,0.45]$, all linear. The four
reference values are pinned onto their axes unconditionally, so the reference points are guaranteed
to be grid points — a prerequisite for the `asimov` validation gate. After de-duplication the union
contains **14 702 distinct points**, each fitted twice. `--draft` substitutes a coarse grid for fast
validation runs.

Two planes are devoted to $(\sin^2\theta_{13},\sin^2\theta_{12})$ — one at each reference
$\Delta m^2_{21}$ — and the lines duplicate the plane intersections at finer spacing, because the
one-dimensional projections enter the parameter-resolution quotes while the planes supply the
contours.

#### Cut quality score

For each analysis cut $(N_{\mathrm{hits}},N_{\mathrm{ophits}},N_{\mathrm{adjcl}})$ and each grid
point $\vec\theta_k$:

1. Construct the Asimov dataset $\oij(\vec\theta_k)$.
2. Profile $\chi^2$ against $p^{\mathrm{solar}}$: obtain $\chi^2_\odot(\vec\theta_k)$.
3. Profile $\chi^2$ against $p^{\mathrm{react}}$: obtain $\chi^2_{\mathrm{react}}(\vec\theta_k)$.

`04_best_cuts.py` scores each candidate cut by the symmetric **wrong-hypothesis** $\chi^2$ — the
average of the two cross-fits, larger meaning the hypotheses are harder to confuse:

$$\mathrm{Score}(\mathrm{cut}) = \tfrac{1}{2}\!\left[ \chi^2_\odot(\vec\theta_{\mathrm{react}}) + \chi^2_{\mathrm{react}}(\vec\theta_\odot) \right]$$

The winning cut is written to `highest_SENSITIVITY.pkl` and used by the full scan. The scorer uses
the same statistic as the scan itself (`sensitivity_pull_chi2` under `--fit_method pull`), so
optimisation and final result are internally consistent.

The ordering of pipeline stages matters and is enforced by `run_sensitivity.py`: background
templates for all candidate cuts (phase 1) → cut optimisation (phase 2) → signal template grid for
the *selected* cut (phase 3) → $\chi^2$ scan (phase 4). Running the scan against templates
generated for a different cut is the single most common way to obtain silently wrong results, and
`validate_template_cut_consistency` aborts the run if the stored templates do not match the
resolved cut.

### 5.11 Contour Construction {#sens-contours}

Contours are drawn from the **difference** to the grid minimum, never from absolute $\chi^2$:

$$\Delta\chi^2(\vec\theta_k) = \chi^2(\vec\theta_k) - \min_{k'}\chi^2(\vec\theta_{k'}), \qquad
Z(\vec\theta_k) = \sqrt{\max\left(\Delta\chi^2,\,0\right)} .$$

`contour_plot.py` plots $Z$ with contour levels at $Z = 1,2,3$ and stores $\chi^2_{\min}$ alongside
the grid for diagnostics. The distinction is essential, and is discussed further in §8.2: absolute
thresholds conflate the overall goodness of fit with the *shape* of the likelihood surface, so any
systematic that raises the whole surface would exclude points from the contour and make it shrink,
which is precisely backwards.

### 5.12 Validation Gates {#sens-validation}

Every scan writes a `*_Validation.json` report next to its grids, produced by
`sensitivity_validation_gates` (`lib/fitting.py:1712`). The gates are method-independent; a gate
that cannot be evaluated reports `null` and does not fail the run.

| Gate | Condition |
|---|---|
| `finite` | every $\chi^2$ is finite and none equals the legacy failure sentinel |
| `asimov` | the reference point fitted against its own hypothesis gives $\chi^2 \le 0.01$ |
| `profile_bound` | $\chi^2 \le \chi^2_0$ at every point, and every fit converged |
| `smoothness` | no isolated spikes along either grid axis (`_grid_spikes`) |
| `sin13_profile` | grid-mode $\sin^2\theta_{13}$ profiling only where every cell has $>1$ value |

The `asimov` gate is the sharpest single check available in an Asimov study: fitting a dataset
against the hypothesis that generated it must return $\chi^2 = 0$ up to numerical tolerance, and any
error in template alignment, exposure scaling or bin masking breaks it immediately.

Under `--fit_method pull` a failing gate is **fatal** by default: the run writes a
`*_Sensitivity_INVALID.json` marker into the results directory and terminates with `os._exit(1)`,
so a broken grid cannot silently propagate into a plot. The hard exit is deliberate — `lib`
registers an `atexit` handler that would otherwise convert a `SystemExit` into exit status 0 and
let the pipeline continue. Under the legacy method the gates are warn-only. The behaviour is
overridable with `--strict_validation` / `--no-strict_validation`.

### 5.13 Systematic Studies {#sens-studies}

Study variants are registered in `lib/study.py` and dispatched by `run_studies.py`, which isolates
each variant's outputs with a `--study_label` so the main analysis products are never overwritten.

| Group | Thesis § | Knob varied |
|---|---|---|
| `metric` | 9.1.1 | raw vs. smoothed histogram metric |
| `unc` | 9.1.2 | $\spred \in \{0,2,6\}\%$; $\sbkg \in \{0,4,6\}\%$ |
| `oscpoint` | 9.1.3 | reference $\Delta m^2_{21}$ (solar vs. reactor) |
| `energy` | 9.2.1 | energy estimator (`SolarEnergy`, `SignalParticleK`, `MainK`) |
| `fiduc_truth` | 9.2.2 | truth vs. reconstructed position for the fiducial mask |
| `fiduc` | 9.2.3 | fiducialisation folder (Nominal / Reduced / Truncated) |
| `charge` | 9.2.4 | charge threshold scan, replacing the NHits/AdjCl axes |
| `bkg_gamma` | 9.2.5 | background gamma model |
| `bkgmodel` | 9.2.6 | background model normalisation |
| `membrane_veto` | — | membrane/endcap optical veto on/off (VD) |
| `nuisance` | — | nuisance-profile decomposition (§5.9) |
| `legacy_fit` | — | legacy minimiser on the default templates (§5.15) |

Only variants that change the event selection or the oscillation weighting receive their own
template directory (`template_suffix` in `lib/study.py`); the remainder — including all of the
`unc` group — reuse the default templates and differ only in the fit configuration, which is why
they are cheap to run and why their outputs are separated by results directory rather than by
filename.

### 5.14 Sensitivity to the Background Normalisation Uncertainty {#sens-bkg-prior}

#### Observation

The `unc_bkg*` variants scan $\sbkg \in \{0,\,2,\,4,\,6\}\%$. Under `--fit_method pull` the
resulting $\chi^2$ grids are **numerically identical** for every non-zero value. For the reference
configuration (`hd_1x2x6_centralAPA`, Truncated, `SolarEnergy`, `NHits1/AdjCl4/OpHits8`, 30 yr):

| $\sbkg$ | $\sum\chi^2_\odot$ (sin12 plane) | $\sum\chi^2_{\mathrm{react}}$ (sin12 plane) | $\max\lvert\Delta\rvert$ vs. $\sbkg=0$ |
|---|---|---|---|
| 0 % | 98 523.15018 | 96 149.12246 | — |
| 2 % | 98 522.75789 | 96 148.76414 | $6.5\times10^{-4}$ |
| 4 % | 98 522.75789 | 96 148.76414 | $6.5\times10^{-4}$ |
| 6 % | 98 522.75789 | 96 148.76414 | $6.5\times10^{-4}$ |

The $2/4/6\%$ grids agree to ten significant figures. The small offset of the $\sbkg = 0$ grid is
not physical: removing the nuisance changes the dimension of the linear systems in §5.7 and hence
the rounding, and $6\times10^{-4}$ is the resulting floating-point difference on a $\chi^2$ of
order $10^2$.

This is an expected property of the analysis, not a defect in the study machinery. The arguments do
reach the fit — the results directory is named `signal_4%_and_background_6%` from the value
actually used.

#### Why the prior cannot bind

Consider a single nuisance with response template $J$ and prior width $\sigma$, and let
$r = \oij - \muzero$ be the residual. In the Gaussian limit the profile reduces the $\chi^2$ by

$$\delta\chi^2 \;=\; \frac{\left(\sum_{ij} J_{ij}\,r_{ij}/\muzero\right)^{2}}{\sum_{ij} J_{ij}^{2}/\muzero \;+\; \sigma^{-2}} .$$

Define the precision with which the **data themselves** determine that parameter,

$$\sigma_{\mathrm{data}} \;\equiv\; \left(\sum_{ij} J_{ij}^{2}\big/\muzero\right)^{-1/2}.$$

The prior enters only through the $\sigma^{-2}$ term in the denominator, and is therefore
irrelevant whenever $\sigma \gg \sigma_{\mathrm{data}}$. In that regime the nuisance is effectively
unconstrained by its prior, the profiled value is fixed entirely by the data, and $\delta\chi^2$
becomes independent of $\sigma$.

For the background normalisation $J = \Tbkg$ and $\muzero \simeq \Tbkg$, because $S/B \sim 10^{-7}$,
so the sum collapses to the total background count:

$$\sigma_{\mathrm{data}} \;\simeq\; \Big(\textstyle\sum_{ij} \Tbkg\Big)^{-1/2} \;=\; B^{-1/2}.$$

The Asimov data determine the background normalisation to the Poisson precision of the total
background count. Numerically, $B = 4.18\times10^{12}$ gives $B^{-1/2} = 4.9\times10^{-7}$, which
reproduces the measured value of $4.892\times10^{-7}$ exactly. A $2\%$ prior is looser than the data
constraint by a factor of $4\times10^{4}$, the profiled background shift comes out at
$\alpha_2 \approx -1.0\times10^{-9}$, and the penalty $(\alpha_2/\sbkg)^2 \sim 10^{-15}$ lies below
the numerical noise floor of the deviance.

Direct evaluation of the reference fit confirms the argument: with $\spred = 4\%$ and the background
nuisance active, $\chi^2_{\mathrm{react}}$ at the solar Asimov point is $1.36640410$ for every
$\sbkg$ from $0.1\%$ to $20\%$, against $1.36640833$ with the nuisance removed altogether. The
entire effect of granting the background a free normalisation is $4\times10^{-6}$ in $\chi^2$: the
solar-reactor difference lives in the shape of the high-energy tail, very nearly orthogonal to a
global background rescaling.

#### The argument extends to background shape

The conclusion is not a consequence of the background having only a normalisation degree of freedom
(§5.6). For any smooth, fully correlated background deformation $J_{ij} = \Tbkg f(E_j)$,

$$\sigma_{\mathrm{data}} = \Big(\textstyle\sum_{ij} \Tbkg f^{2}\Big)^{-1/2} \;\sim\; B^{-1/2}\big/\sqrt{\langle f^{2}\rangle},$$

which remains of order $10^{-6}$ for any $f$ of order unity. Two candidate background shape
nuisances were constructed and evaluated explicitly:

| Candidate nuisance | $\sigma_{\mathrm{data}}$ | Shift in $\chi^2$ vs. current default | Dependence on its prior width over $10^{-3}$–$0.5$ |
|---|---|---|---|
| power-law tilt, $J = \Tbkg\ln(E/10\ \mathrm{MeV})$ | $6.3\times10^{-7}$ | $-4.8\times10^{-5}$ | none |
| energy scale applied to $\Tbkg$ | $7.2\times10^{-8}$ | $-9.9\times10^{-5}$ | none |

Against contour levels at $\Delta\chi^2 = 1,4,9$ a shift of $10^{-4}$ displaces the $1\sigma$
contour by $\sim5\times10^{-5}\sigma$. Adding a background shape nuisance would therefore change
neither the contours nor the outcome of a prior scan.

#### Why breaking the correlation is not the remedy

The only construction that makes a background systematic bite is to give each bin its own
independent term, which is equivalent to inflating the per-bin variance from $\muij$ to
$\muij + (\sigma \Tbkg)^2$. The relevant dimensionless quantity is $\sigma^2 \Tbkg$. With a mean of
$B/722 = 5.8\times10^{9}$ events per active bin and $\sigma = 2\%$,

$$\frac{\sigma^2 (\Tbkg)^{2}}{\muij} \;\simeq\; \sigma^{2}\,\Tbkg \;=\; 2.3\times10^{6},$$

so each bin's effective variance is inflated by more than six orders of magnitude and $\Delta\chi^2$
collapses towards zero. The analysis possesses no intermediate regime: a percent-level background
uncertainty is either fully correlated, in which case the data calibrate it away and it is inert,
or uncorrelated between bins, in which case it destroys the sensitivity entirely. Attempts to force
a background uncertainty into the fit have repeatedly produced artifacts for this reason, and the
`--fit_background` warning at `06_significance.py:304` is a symptom of the same structure.

#### Where the prior would begin to matter

$\sigma_{\mathrm{data}}$ is dominated by the low-energy bins, where the background is largest and
the signal absent. Raising the analysis threshold removes them, and the prior becomes progressively
more relevant:

| Threshold | $B$ (30 yr) | $S/B$ | $\sigma_{\mathrm{data}}$ | $\chi^2(\sbkg{=}0)$ | $\chi^2(2\%)$ | $\chi^2(6\%)$ |
|---|---|---|---|---|---|---|
| 0 MeV *(default)* | $4.18\times10^{12}$ | $7.2\times10^{-8}$ | $4.9\times10^{-7}$ | 1.366408 | 1.366404 | 1.366404 |
| 6 MeV | $2.47\times10^{11}$ | $1.1\times10^{-6}$ | $2.0\times10^{-6}$ | 1.366408 | 1.366360 | 1.366360 |
| 8 MeV | $1.50\times10^{10}$ | $1.5\times10^{-5}$ | $8.2\times10^{-6}$ | 1.366391 | 1.366080 | 1.366080 |
| 10 MeV | $1.09\times10^{8}$ | $1.1\times10^{-3}$ | $9.6\times10^{-5}$ | 1.366009 | 1.362028 | 1.362027 |
| 12 MeV | $2.98\times10^{5}$ | $1.3\times10^{-1}$ | $1.9\times10^{-3}$ | 1.331534 | 1.290772 | 1.289475 |

Only above $\sim12$ MeV, where the data constraint has degraded to $2\times10^{-3}$, does the
$2\%\to6\%$ variation produce any change at all, and it is then $1.3\times10^{-3}$ in $\chi^2$. The
analysis threshold `ANALYSIS_THRESHOLDS.SENSITIVITY.SIGNIFICANCE` is currently 0.

#### Statement for the thesis

The `unc_bkg*` variants should be presented as establishing an insensitivity rather than as a
failed scan. A defensible formulation is:

> Because the selected sample is background dominated by seven orders of magnitude, the Asimov
> dataset constrains the background normalisation *in situ* to the Poisson precision of the total
> background count, $B^{-1/2}\approx5\times10^{-7}$. Any external prior on the background
> normalisation at the percent level is looser than this by four orders of magnitude and is
> therefore non-binding: the oscillation-parameter contours are unchanged for $\sbkg$ anywhere
> between 0 and 20 %. The same holds for any smooth, fully correlated deformation of the background
> spectrum. The sensitivity reported here is consequently limited by the signal flux normalisation,
> the reconstructed energy scale and the statistical power of the high-energy tail, and not by
> knowledge of the background rate.

The corresponding entry in the systematic-uncertainty table should read "< 0.001 in $\Delta\chi^2$;
negligible" rather than being left blank or reported as an unavailable result.

*Verified 2026-09-14; reproduction recipe in §5.19.*

### 5.15 Legacy Fitter (superseded) {#sens-legacy}

The original implementation (`--fit_method legacy`, class `Sensitivity_Fitter`) minimised the same
deviance over the two normalisations $(\Apred,\Abkg)$ with `scipy.optimize` L-BFGS-B
[Byrd et al. 1995; Zhu et al. 1997] under box constraints $|A|\le10\sigma$, wrapping the whole fit
in an outer scan over discrete energy-scale shifts. It remains available for validation and writes
into a parallel `results/<profile>_legacy/` tree so it can never overwrite the default outputs; the
`legacy_fit` study group exists for this comparison. Under `--fit_method pull` the legacy flags
`--fit_background` and `--use_legacy_background_penalty` are ignored, with a warning
(`06_significance.py:330`).

### 5.16 Template Construction Pipeline {#sens-conv-figures}

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

### 5.17 Results {#sens-results}

| Figure | Description |
|---|---|
| `figures/sens_central_solar_sin12` | centralAPA: $\Delta\chi^2$ contours in $(\sin^2\theta_{12},\,\Delta m^2_{21})$ under the solar oscillation hypothesis |
| `figures/sens_central_react_sin12` | centralAPA: $\Delta\chi^2$ contours under the reactor hypothesis |
| `figures/sens_central_significance` | centralAPA: hypothesis-separation significance vs. exposure at 30 kt·yr |
| `figures/sens_lateral_solar_sin12` | lateralAPA: $\Delta\chi^2$ contours under the solar hypothesis |
| `figures/sens_lateral_react_sin12` | lateralAPA: $\Delta\chi^2$ contours under the reactor hypothesis |
| `figures/sens_lateral_significance` | lateralAPA: hypothesis-separation significance vs. exposure |

### 5.18 Data Products and Directory Layout {#sens-products}

Scan outputs are written beneath the signal template directory:

```
{PATH}/SENSITIVITY/{config}/{signal}/{folder}/{energy}{template_suffix}/
  results/{profile}[_legacy]/signal_{S}%_and_background_{B}%/       (--background, the default)
  results/{profile}[_legacy]/signal_{S}%_only/                      (--no-background)
      {signal}_{energy}_NHits{n}_AdjCl{a}_OpHits{o}_solar_df[_10Y].pkl
      {signal}_{energy}_NHits{n}_AdjCl{a}_OpHits{o}_react_df[_10Y].pkl
      {signal}_{energy}_NHits{n}_AdjCl{a}_OpHits{o}_solar_sin12_df[_10Y].pkl
      {signal}_{energy}_NHits{n}_AdjCl{a}_OpHits{o}_solar_sin13_df[_10Y].pkl
      {signal}_{energy}_NHits{n}_AdjCl{a}_OpHits{o}_react_sin12_df[_10Y].pkl
      {signal}_{energy}_NHits{n}_AdjCl{a}_OpHits{o}_react_sin13_df[_10Y].pkl
      {signal}_{energy}_NHits{n}_AdjCl{a}_OpHits{o}_Validation[_10Y].json
      {signal}_{energy}_Sensitivity_INVALID[_10Y].json        (only on gate failure)
```

- `*_df.pkl` — tidy long-format tables, one row per grid point, columns `dm2, sin13, sin12, chi2`.
- `*_sin12_df.pkl`, `*_sin13_df.pkl` — the corresponding planes as $\Delta m^2$-indexed matrices,
  ready for contour plotting.
- `_10Y` — the secondary exposure, evaluated in the same pass.
- The `{profile}` and `signal_{S}%_and_background_{B}%` levels are what separate the `nuisance` and
  `unc` study variants; the study label itself does **not** appear in these filenames.

**Configuration reference.**

| Key | File | Value | Used for |
|---|---|---|---|
| `SOLAR_DM2` / `REACT_DM2` | `physics.json` | $6.0/7.54\times10^{-5}$ eV² | reference hypotheses |
| `SIN13` / `SIN12` | `physics.json` | 0.022 / 0.304 | reference hypotheses |
| `NADIR_BINS` | `physics.json` | 40 | template nadir axis |
| `OSC_NADIR_OVERSAMPLE` | `physics.json` | 4 | anti-aliasing (§5.3) |
| `OSC_ENERGY_BINS` / `_RANGE` | `physics.json` | 120 / [0,30] MeV | oscillation sampling |
| `OSCILLATION_GRID` | `physics.json` | 4 planes + 5 lines | scan grid (§5.10) |
| `OSCILLATION_BACKEND` | `physics.json` | `nufast` | probability engine |
| `ANALYSIS_EXPOSURES.SENSITIVITY` | `config.json` | 30 yr / 10 yr | primary / secondary |
| `ANALYSIS_THRESHOLDS.SENSITIVITY.SIGNIFICANCE` | `config.json` | 0 | low-energy cut (§5.14) |
| `ANALYSIS_UNCERTAINTIES.SENSITIVITY` | `config.json` | 0.04 / 0.02 | $\spred$ / $\sbkg$ |
| `NUISANCE_PROFILES` | `config.json` | 4 profiles | §5.9 |
| `DEFAULT_NUISANCE_PROFILE` | `config.json` | `full` | §5.9 |
| `ENERGY_SCALE_SIGMA` | — | **absent**, code default 0.02 | $\ses$ (§5.9) |
| `SIN13_SIGMA` | — | **absent**, code default $5.6\times10^{-4}$ | $\ssin$ (§5.9) |

### 5.19 Reproduction Recipe {#sens-reproduction}

The numerical statements of §5.14 are reproduced by the following, which loads the stored templates
directly and calls the same fitting routines used by the scan. Run inside the analysis container
(`apptainer exec -B /pnfs:/pnfs -B /pc:/pc containers/solar_v1.0.sif python3 …`).

```python
import sys; sys.path.insert(0, "<repo root>")
import numpy as np, pandas as pd
from lib.fitting import sensitivity_pull_jacobian, sensitivity_pull_profile

S = "<PATH>/SENSITIVITY/hd_1x2x6_centralAPA/marley/truncated/SolarEnergy"
B = "<PATH>/SENSITIVITY/hd_1x2x6_centralAPA/background/truncated/SolarEnergy"
cut, E = "NHits1_AdjCl4_OpHits8", 30.0
L = lambda p: np.nan_to_num(np.asarray(pd.read_pickle(p), dtype=float), nan=0.0)

bkg = E * L(f"{B}/hd_1x2x6_centralAPA_background_{cut}.pkl")
p1  = E * L(f"{S}/hd_1x2x6_centralAPA_marley_{cut}_dm2_6.000e-05_sin13_2.200e-02_sin12_3.040e-01.pkl")
p2  = E * L(f"{S}/hd_1x2x6_centralAPA_marley_{cut}_dm2_7.540e-05_sin13_2.200e-02_sin12_3.040e-01.pkl")
obs, mask = p1 + bkg, bkg > 0                       # Asimov at the solar point

# data-driven precision on the background normalisation, and its B^{-1/2} closed form
sigma_data = 1.0 / np.sqrt(np.sum(bkg[mask] ** 2 / (p2 + bkg)[mask]))
print(f"sigma_data = {sigma_data:.3e}   B^-1/2 = {bkg.sum() ** -0.5:.3e}")

for sb in [0.0, 0.001, 0.02, 0.06, 0.20]:           # chi2 is identical for every sb > 0
    jac, sig, _ = sensitivity_pull_jacobian(p2, bkg, sigma_pred=0.04, sigma_bkg=sb)
    chi2 = sensitivity_pull_profile(obs, p2 + bkg, jac, sig, mask=mask)["chi2"]
    print(f"sigma_bkg = {sb:<6} chi2_react = {chi2:.8f}")
```

Replace `p2` by `p1` for the solar-hypothesis fit. To activate the energy-scale nuisance, pass
`e_centers=Ec` and `sigma_e_scale=0.02`, where `Ec` is `lib.sensitivity_rebin_centers`. To
reproduce the background-tilt row of §5.14, append the column and its prior width by hand:

```python
from lib import sensitivity_rebin_centers as Ec
tilt = bkg * np.log(np.asarray(Ec, dtype=float) / 10.0)[None, :]
jac, sig, _ = sensitivity_pull_jacobian(p2, bkg, e_centers=Ec, sigma_pred=0.04,
                                        sigma_bkg=0.02, sigma_e_scale=0.02)
jac = np.concatenate([jac, tilt[None, :, :]])
sig = np.concatenate([sig, [0.02]])                 # the result is independent of this value
print(sensitivity_pull_profile(obs, p2 + bkg, jac, sig, mask=mask)["chi2"])
```

---

## 6. Comparison Across Analyses {#comparison}

| Feature | Day-Night | HEP | Sensitivity |
|---|---|---|---|
| Observable | 1D rate-difference spectrum | 1D energy spectrum | 2D (nadir angle × energy) |
| Signal | $\si{=}\Ecal\theta_s(r_i^{\rm night}{-}r_i^{\rm day})$ | $\si{=}\Ecal r_i^{\rm hep}$ | $\Tsig(\vec\theta)$ |
| Goal | Detect day-night asymmetry | Discover hep flux | Map $(\Delta m^2,\sin^2\theta)$ contour |
| Significance type | Gaussian (diagnostic), two-sample Asimov (primary) | Gaussian, Asimov, PL | $\chi^2$ surface |
| Nuisances | None (error propagated) | $1\times\beta$ (background) | signal + background norm.; energy scale and $\sin^2\theta_{13}$ by profile |
| Nuisance solution | — | Closed-form quadratic | Pull profile: Gaussian (Woodbury) start + safeguarded Newton |
| Penalty form | — | $[(\bhat{-}1)/\srel]^2$ | $\sum_k(\alpha_k/\sigma_k)^2$ |
| Per-bin deviance | Two-sample LLR | Baker-Cousins per bin | Baker-Cousins 2D |
| Effective background | $\Beff$ (inv-fraction-weighted) | $\bi = \Ecal\sum_{c\neq\mathrm{hep}} r_i^c$ | $\Tbkg$ (2D template) |
| Barlow-Beeston mask | No | $N_i^{\rm MC}\ge N_{\rm MC}^{\min}$ | $\muzero>0 \wedge \Tbkg>0$ |
| Asymmetry/signal band | $\theta_s\in\{1\pm\epstot,1\}$ | $d\in\{2.9,3.0,3.1\}$ | — |
| PL post-processing | — | Gauss($\sigma{=}6$) + PAVA (optional) | — |
| Spike filter | — | $\Delta_{\max}$ on pre-PAVA | — |
| Output | $Z$ vs. exposure | $Z$ vs. exposure | $\chi^2$ map over grid |

**Narrative progression.** The three analyses exhibit a clear progression in statistical complexity centred on the treatment of nuisance parameters.

The Day-Night analysis makes no nuisance-parameter approximations: signal and background are evaluated exactly in two separate exposure intervals, and the asymmetry uncertainty is propagated analytically as a scale factor on the signal. The three-source background uncertainty in the Error Gaussian is the most detailed error propagation of the three analyses, but no LLR profiling is performed.

The HEP analysis introduces the full profile-likelihood framework. The critical observation is that a single global $\beta$ correlated across all bins leads to a scalar stationarity equation, whose positive root is a closed-form quadratic. This makes the PL computation exact and fast. PAVA post-processing and adaptive rebinning address practical numerical issues (oscillations at low $S/B$; low-statistics tail bins).

The Sensitivity analysis takes the same Baker-Cousins Poisson deviance as HEP's per-bin LLR, extends it to 2D, and admits up to four nuisances at once. Coupling the model simultaneously to signal and background breaks the factorization that enabled the HEP quadratic, so no scalar closed form exists. The resolution is not a general-purpose optimiser but a *linear response model*: writing $\muij = \muzero + \sum_k\alpha_k J^{(k)}_{ij}$ makes the Gaussian limit exactly solvable by Woodbury, and that solution seeds a safeguarded Newton iteration on the exact Poisson objective. The result is a $\chi^2$ map rather than a significance curve.

A structural feature of this analysis, absent from the other two, is that the Asimov data themselves constrain some nuisances far more tightly than their priors. The background normalisation is determined *in situ* to $B^{-1/2}\approx5\times10^{-7}$, so its 2 % prior never binds and the contours are independent of it ([§5.14](#sens-bkg-prior)). Systematic uncertainties in this analysis are therefore not interchangeable: only those whose prior is tighter than the corresponding $\sigma_{\mathrm{data}}$ affect the result.

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
| **Sensitivity** | `chi2_solar` / `chi2_react` | 2D smooth | None | Baker-Cousins + pull profile | — |

---

## 8. Recent Statistical Methodology Updates {#stat-updates}

### 8.1 Sensitivity Analysis: Fitting Methodology

The Sensitivity fit has passed through three stages. The current default is the **pull method**
(`--fit_method pull`); the two earlier stages are recorded here because published intermediate
results exist for both.

**Stage 1 — original (superseded).** Both the signal amplitude $\Apred$ and the background
normalisation $\Abkg$ were free in an L-BFGS-B minimisation. This produced physically incorrect
behaviour: increasing the background uncertainty let the background normalisation absorb signal
mismatches, so the sensitivity contours **shrank** instead of loosening.

**Stage 2 — `--fit_background` control (superseded).** A flag was introduced to fix $\Abkg = 0$,
leaving only $\Apred$ free, on the reasoning that a background free to float was the source of the
artifact. The legacy behaviour remained reachable with `--fit_background`.

**Stage 3 — pull method (current default).** The fit was restructured around the linear response
model of [§5.6](#sens-response). All active nuisances — signal normalisation, background normalisation, and,
according to the profile, energy scale and $\sin^2\theta_{13}$ — are profiled together against
their Gaussian priors, with a closed-form Gaussian start and a safeguarded Newton iteration ([§5.7](#sens-profile)).
Under `--fit_method pull` the background normalisation is **always** profiled with its prior, and
`--fit_background` / `--use_legacy_background_penalty` are ignored with a warning.

**Status of the stage-1 diagnosis.** The shrinking-contour artifact was real, but attributing it to
the background being free was incomplete. [§5.14](#sens-bkg-prior) shows that the Asimov data constrain the background
normalisation to $B^{-1/2}\approx5\times10^{-7}$, so a floating background normalisation with a
percent-level prior costs only $4\times10^{-6}$ in $\chi^2$ and cannot move a contour in either
direction. What the stage-1 fitter actually did was let a bounded optimiser wander in a direction
the data pin almost exactly, at $10^{9}$ counts per bin and without equilibration — a numerical
failure, not a statistical one. The pull method removes it by construction, which is why
`--fit_background` no longer has any effect under the default method.

**Validation.** The `legacy_fit` study group runs the stage-1/2 minimiser on the default templates
and writes to `results/<profile>_legacy/`, so the two methods can be compared without either
overwriting the other. Validation gates ([§5.12](#sens-validation), `profile_bound`; `sensitivity_validation_gates`) are
strict by default under the pull method and warn-only under legacy.

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



### 8.5 Sensitivity Analysis: The Background Uncertainty Is Non-Binding

**Observation.** Under `--fit_method pull` the `unc_bkg*` study variants
($\sbkg\in\{0,2,4,6\}\%$) produce $\chi^2$ grids that agree to ten significant figures for every
non-zero $\sbkg$. This is a property of the analysis, not a defect in the study machinery: the
values do reach the fit, and the results directory is named from the value actually used.

**Explanation.** A Gaussian prior of width $\sigma$ on a nuisance with response template $J$ affects
the profile only through the $\sigma^{-2}$ term in

$$\delta\chi^2 = \frac{\left(\sum_{ij}J_{ij}r_{ij}/\muzero\right)^2}{\sum_{ij}J^2_{ij}/\muzero + \sigma^{-2}},$$

so it is irrelevant once $\sigma$ exceeds the precision
$\sigma_{\mathrm{data}} = (\sum J^2/\muzero)^{-1/2}$ to which the data themselves determine the
parameter. For the background normalisation this evaluates to $B^{-1/2} = 4.9\times10^{-7}$ at
$B = 4.18\times10^{12}$ events — four orders of magnitude tighter than a 2 % prior.

**Scope.** The conclusion extends to any smooth, fully correlated background deformation: a
power-law spectral tilt and a background energy scale were both constructed and evaluated, and both
have $\sigma_{\mathrm{data}}\sim10^{-7}$ and shift $\chi^2$ by $\lesssim10^{-4}$ irrespective of
their prior width. Adding a background shape nuisance would therefore not change the default result
and is not worth implementing. Conversely, breaking the bin-to-bin correlation inflates each bin's
variance by $\sigma^2 b_{ij}\approx2\times10^{6}$ and collapses the sensitivity entirely; there is
no intermediate regime.

**Consequences for the thesis.** The `unc_bkg*` variants establish an insensitivity and should be
reported as such rather than as an unavailable result. The systematic-uncertainty table entry for
the background normalisation should read "< 0.001 in $\Delta\chi^2$; negligible". The full
derivation, the measured tables, the threshold dependence and suggested wording are in
[§5.14](#sens-bkg-prior).

*Verified 2026-09-14 on `hd_1x2x6_centralAPA`, Truncated, `SolarEnergy`, cut `NHits1/AdjCl4/OpHits8`,
30 yr, profile `full`.*


---

## 9. Workflow Flags and Configuration {#flags}

**Stage flags in `run_sensitivity.py`** (all `--x` / `--no-x` pairs):

| Flag | Default | Effect when disabled |
|---|---|---|
| `--computation` | on | Skip all computation; run only plot macros |
| `--significance` | on | Skip `01_daynight.py`, `01_hep.py`, `06_significance.py` |
| `--fiducialization` | on | Skip the fiducialisation scan |
| `--rebin` | on | Skip `03_analysis.py` adaptive rebinning |
| `--plot` | on | Skip all figure output |
| `--skip_best_cuts` | off | *When set:* reuse the stored best-cut map instead of re-optimising |
| `--skip-templates` | off | *When set:* reuse existing templates (guarded — regenerates if stale) |
| `--optimization` | off | *When set:* run the smoothing-sigma optimiser |

**Sensitivity-specific flags:**

| Flag | Default | Meaning |
|---|---|---|
| `--fit_method {pull,legacy}` | `pull` | Statistic of [§5.7](#sens-profile) vs. the superseded fitter of [§5.15](#sens-legacy) |
| `--nuisance_profiles` | `full` | One or more profiles from [§5.9](#sens-profiles); each gets its own `results/<profile>/` tree |
| `--flyweight` | on | Convolve signal templates on the fly from a stored base template |
| `--strict_validation` | pull: on, legacy: off | Fail the stage when a validation gate fails |
| `--draft` | off | Coarse oscillation grid for fast validation |
| `--exposure` | 30 yr | Primary exposure; templates are per-year and scaled at load |
| `--secondary_exposure` | 10 yr | Second exposure evaluated in the same pass (`_10Y` outputs) |
| `--signal_uncertainty` | 0.04 | $\spred$ |
| `--background_uncertainty` | 0.02 | $\sbkg$ — see §8.5: the result is independent of this value |
| `--fit_background` | off | Legacy only; **ignored** under `--fit_method pull`, with a warning |
| `--study_label` | — | Isolate outputs of a study variant (§5, `run_studies.py`) |

**Per-analysis workflow feature flags** (`config/analysis/config.json`, `WORKFLOW.{ANALYSIS}`):

| Key | Default | Effect when true |
|---|---|---|
| `DAYNIGHT.background_error` | `true` | Compute Error Gaussian curves |
| `DAYNIGHT.significance_bins` | `true` | Save per-bin significance spectra at the display exposure |
| `HEP.pl_signal_bands` | `true` | Evaluate three signal normalizations for $\pm1\sigma_s$ PL bands |
| `HEP.pl_isotonic` | `false` | Apply Gaussian+PAVA post-processing to PL curves |
| `HEP.significance_bins` | `false` | Save per-bin significance spectra at the display exposure |

Sensitivity configuration keys are tabulated in [§5.18](#sens-products), including the two prior
widths ($\ses$, $\ssin$) that are currently code defaults rather than configuration entries.

---

## 10. References {#references}

[Cowan 2010] G. Cowan, K. Cranmer, E. Gross, O. Vitells, *Asymptotic formulae for likelihood-based tests of new physics*, Eur. Phys. J. C **71** (2011) 1554. <https://arxiv.org/abs/1007.1727>

[Baker & Cousins 1984] S. Baker, R.D. Cousins, *Clarification of the use of chi-square and likelihood functions in fits to histograms*, Nucl. Instrum. Meth. **221** (1984) 437–442. <https://doi.org/10.1016/0167-5087(84)90016-4>

[Barlow & Beeston 1993] R. Barlow, C. Beeston, *Fitting using finite Monte Carlo samples*, Comput. Phys. Commun. **77** (1993) 219–228. <https://doi.org/10.1016/0010-4655(93)90005-W>

[Robertson 1988] T. Robertson, F.T. Wright, R.L. Dykstra, *Order Restricted Statistical Inference*, Wiley, 1988.

[Byrd et al. 1995] R.H. Byrd, P. Lu, J. Nocedal, C. Zhu, *A limited memory algorithm for bound constrained optimization*, SIAM J. Sci. Comput. **16** (1995) 1190–1208.

[Zhu et al. 1997] C. Zhu, R.H. Byrd, P. Lu, J. Nocedal, *Algorithm 778: L-BFGS-B: Fortran subroutines for large-scale bound-constrained optimization*, ACM Trans. Math. Softw. **23** (1997) 550–560.
