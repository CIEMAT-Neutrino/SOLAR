# Fiducialisation and the truth-position checks

Support document for the thesis sections on the fiducial volume and on the `fiduc_truth` study
(Chapter 9.2; section numbers are inconsistent across the repo, see §6).
It collects (i) the fiducialisation procedure as implemented, (ii) the definition of the truth-position variant,
and (iii) the event-level checks made on 2026-09-20 to explain why using the true position barely
changes the results. Every number below is taken from the generated tables
[truth_position_tables_daynight.md](truth_position_tables_daynight.md),
[truth_position_tables_sensitivity.md](truth_position_tables_sensitivity.md),
[truth_position_faces_daynight.md](truth_position_faces_daynight.md) and
[truth_position_faces_sensitivity.md](truth_position_faces_sensitivity.md).

Status: written against commit `c800b4f` plus the uncommitted working tree of 2026-09-20. The study code
(`src/physics/signal/truth_position_study.py`, `src/tools/*truth_position*`) was not yet under version control
when this file was written. Weights are `SignalParticleWeight` only (no oscillation, no MC-support gate, no
smoothing), so the checks explain position information; they are not replacement significances.

---

## 1. Fiducialisation procedure

### 1.1 Geometry and the mask

Detector bounds in the analysis frame (cm; `config/{config}/{config}_config.json`, no gaps):

| config | X | Y | Z |
|---|---|---|---|
| `hd_1x2x6_centralAPA` | −360 … 360 | −600 … 600 | 0 … 1400 |
| `hd_1x2x6_lateralAPA` | 0 … 360 | −600 … 600 | 0 … 1400 |
| `vd_1x8x14_3view_30deg_nominal` / `_shielded` | −330 … 330 | −675 … 675 | 0 … 2100 |

The fiducial volume is defined by three margins $(F_X, F_Y, F_Z)$ measured inwards from the boundaries
(`lib/fiducial.py: build_fiducial_spatial_mask`), with $L_X, L_Y$ the full X and Y sizes:

- **HD central APA:** $|x| < L_X/2 - F_X$ (margin from both drift ends; the plane at $x=0$ is not a wall).
- **HD lateral APA:** $|x| > F_X$, i.e. the margin is measured from the $x=0$ plane; the other X face is the
  far wall at $x=360$ and is not cut.
- **VD:** $x < L_X/2 - F_X$, i.e. only the upper (top) X boundary is fiducialised, because the workspace simulates one drift.
- **Y:** $|y| < L_Y/2 - F_Y$ for all configs.
- **Z:** $F_Z < z < L_Z - F_Z$, applied only where the folder rejects the Z endcaps geometrically
  (`Nominal`; see §1.5). For `Truncated` and `Reduced` the endcap backgrounds are removed by the
  surface filter `0 <= SignalParticleSurface < 3` instead.

**Walls used by the checks.** The distance-to-wall and entry-face checks of §3 use these walls: HD central $x=\pm360$ cm
(the plane $x=0$ is interior); HD lateral $x=0$ only, because the background piles up there (see §3.1) and the pipeline itself cuts
from it, while the far $x=360$ is not counted; VD $x=\pm330$ cm; the two Y and two Z faces for all configs.

The fiducial mass entering the exposure is described in
[SOLARReference.md, "Fiducialization"](../presentations/SOLARReference.md) (formula with
$\rho_{\rm LAr}=1.396$ g/cm³ and the drift factor of each config); it is not re-derived here.

### 1.2 Events entering the scan

Before any fiducial cut, an event must pass the quality mask that `01_fiducialize.py`, `03_analysis.py`
and `02_signal_template.py` share: the matched flash lies on the accepted optical plane
(`QUALITY_CUTS.OPFLASH_PLANE = 0`: cathode for VD, APA for HD; the membrane veto), `MatchedOpFlashPE > 0`,
and, for surface-filtered backgrounds, the surface cut above. The $(0,0,0)$ point of the scan applies this
mask only, so that every tighter volume is compared against the same baseline.

### 1.3 The volume scan (`src/physics/signal/01_fiducialize.py`)

For each sample (signal `marley`, and `gamma`, `neutron`, `radiological`), energy variable and folder the script
scans a grid of margins,

$$F_X \in [0, L_X/4),\quad F_Y \in [0, L_Y/4),\quad F_Z \in [0, L_Z/4)\quad\text{in steps of 20 cm},$$

and for each triple histograms the weighted events in reconstructed-energy bins per component. It stores
`Counts`, the raw MC counts `MCCounts` (half of a non-empty bin's content is moved into an empty bin on its left, to bridge isolated zeros in the tail), and relative errors
$1/\sqrt{N_{\rm MC,i}}$ per bin (backgrounds add a 2% systematic). Output:
`{PATH}/FIDUCIAL/{folder}/{config}/{sample}/{config}_{sample}_{energy}_Fiducial_Scan.pkl`
(`..._Fiducial_Scan_fiduc_truth.pkl` for the truth variant).

### 1.4 Choosing the volume (`src/physics/signal/02_best_fiducial.py`)

The metric is configured per analysis in [config/analysis/fiducialization.json](../../config/analysis/fiducialization.json):

| analysis | energy window | bands | signal components | background components |
|---|---|---|---|---|
| DayNight | 6–18 MeV | 6–12, 12–18 | 8B, hep | gamma, neutron, radiological |
| HEP | 14–30 MeV | 14–20, 20–30 | hep | gamma, neutron, radiological, 8B |
| Sensitivity | 10–30 MeV | 10–20, 20–30 | hep, 8B | gamma, neutron, radiological |

For every grid point the signal and background spectra in the window are combined, optionally smoothed with the
`stage="fiducial"` smoothing configuration, and scored with the configured significance
(`asimov` for all three analyses, unless `BEST_SIGMA_SIGNIFICANCE_REFERENCE` overrides it) evaluated at `--exposure` (default 100 yr). Asimov bin significances add in quadrature
($q_0$ is additive over independent bins), which is what `combine_mode: quadrature` implements. The volume
with the largest **smoothed** significance is written to `config/analysis/fiducial/{folder}/BestFiducials.json`
together with the no-fiducial significance, the significance type, the MC threshold and the smoothing
metadata. The optimisation is repeated per energy band; downstream the band-specific volume is applied per event
(`lib/fiducial.py: build_energy_band_spatial_mask`).

**MC-support gate.** A grid point is admissible only if every *essential* background component has at least
`--mc_threshold` summed MC counts in the window (`FIDUCIALIZATION.MC = 100` in `config/analysis/config.json`).
The no-fiducial point is always admissible, because suppressed backgrounds have few MC events by physics, and
excluding it drives the optimiser to spuriously tight corners.

**Truth variant fallback.** If the purity cut leaves no admissible volume in the truth scan (VD HEP), the
reference volume from `BestFiducials.json` is written into `BestFiducials_fiduc_truth.json` with the flag
`"Fallback": "reference"`. (An earlier version reused the previous run's entry, then a lowered threshold; both
were replaced on 2026-09-19.)

### 1.5 Folders

| folder | Z endcap rejection | reduction | role |
|---|---|---|---|
| `Nominal` | fiducial Z cut | none | full geometry |
| `Truncated` | `SignalParticleSurface < 3` | none | default analysis folder |
| `Reduced` | `SignalParticleSurface < 3` | gamma /6.667 and neutron /50 above 3.5 MeV | shielded-like background |

(`config/analysis/folder_configs.json`.) The `bkgmodel` study runs Nominal and Reduced at the Truncated volumes
and cuts (`--reference_folder Truncated`), so that only the background model differs.

### 1.6 Downstream use

`03_analysis.py`, the templates and the DayNight, HEP and Sensitivity stages read the volume of their own
analysis from `BestFiducials.json` (falling back to the energy-level entry). `run_sensitivity.py` chains
scan, selection, analysis and the three significance stages.

---

## 2. The truth-position variant (`--truth_fiducial`)

Definition and history are in [solar_analyses.md §5.13](solar_analyses.md); this section records what the
checks below rely on.

1. **Position keys** (`BACKGROUND_SAMPLES.truth_position_keys` in `config/analysis/backgrounds.json`,
   the same for every config): marley `SignalParticleX/Y/Z`, gamma `EndX/Y/Z`, neutron and radiological
   `MainX/Y/Z`. Neutron `SignalParticle*` is the emission point outside the cryostat, so it cannot be fiducialised.
2. **Containment:** events whose truth position lies outside the active box plus 10 cm are dropped
   (`truth_containment_margin_cm`), because reco positions never leave the volume.
3. **Position-consistency cut** (backgrounds only): keep events with $|X_{\rm reco}-X_{\rm true}|\le100$ cm and
   $|Y,Z_{\rm reco}-Y,Z_{\rm true}|\le50$ cm. It re-imposes the rejection the reco cut obtains by accident from
   wrong flash matches. The signal is not gated, since gating it cost 20–50% of the signal.
4. **Volume and cuts:** the truth scan gets its own volume (`BestFiducials_fiduc_truth.json`) and free
   topological cuts. `fiduc_truth_refvol` keeps the reference (reco) volume, isolating what position knowledge
   is worth at an unchanged volume.

### 2.1 Do the truth keys describe the reconstructed cluster?

Fraction of weight whose reco Y,Z (wire-based cluster position) lie within 30 cm of each candidate key, after the
DayNight cut (Table F5):

| sample | configured key | HD central | HD lateral | VD nominal | VD shielded |
|---|---|---|---|---|---|
| marley | SignalParticle | 100% | 100% | 99% | 100% |
| gamma | End | 96% | 90% | 94% | 89% |
| neutron | Main | 36% | 66% | 41% | 70% |

For gamma the other keys give 0–38%, so `End` is the right key. For neutron no key describes the cluster in
a majority of the HD central and VD nominal events; on HD lateral `End` matches better (89%) than `Main` (66%).
Radiological has 2–4 MC events after the cut and is not assessed.

---

## 3. Checks

Common selection (all tables): reconstructed energy 10–20 MeV, quality mask of §1.2, and, where stated, the
default best cut of the analysis (DayNight: NHits/OpHits/AdjCl = 3/11/2 on HD central, 4/4/4 lateral, 6/13/6 on VD
nominal; the Sensitivity cuts differ). Fiducial = the default (reco) volume of that analysis, unless stated.

### 3.1 Where the background sits (Tables 1, F1)

The weighted background-to-signal ratio in a shell of distance to the nearest active-box face, relative to
the 100–200 cm shell (truth position, no topological cut):

| | 0–20 cm | 20–100 cm |
|---|---|---|
| gamma, HD central | ×60 | ×18 |
| gamma, HD lateral | ×133 | ×25 |
| gamma, VD nominal | ×40 | ×15 |
| gamma, VD shielded | ×53 | ×19 |
| neutron, HD central / lateral | ×0.7 / ×3.5 | ×0.9 / ×1.1 |
| neutron, VD (nominal / shielded) | ×19 / ×19 | ×2.2 / ×2.4 |

Gamma piles up at the walls, signal and (in HD) neutron do not.

Reading the cumulative curves: the largest possible distance to the nearest wall is set by the smallest extent, i.e. 360 cm for HD central
(half of its 720 cm X extent) and for HD lateral (its X extent from the $x=0$ wall to $x=360$ is 360 cm), and 330 cm for VD. On HD
lateral the gamma pile-up is at $x=0$: 21% of the gamma weight lies within 20 cm of that plane against 4% of the signal (×5), and
the far $x=360$ wall carries less than the signal (2.5% against 4.5% beyond $x=340$). On HD central the plane $x=0$ is not a wall:
gamma is 7.8% within 20 cm of it against 4.5% for the signal (×1.7), and its wall pile-up is at the Y faces (96% of the gamma has a Y face as
its nearest wall). (An earlier version of the figure also counted $x=360$ as a lateral wall; the curves then ended at 180 cm by geometry.)

The nearest face of the *surviving* background
(after the DayNight cut and the reco fiducial, weight share):

- HD lateral: gamma 91% X low (the $x=0$ plane), neutron 58% X low, 26% Y and 16% Z; the Y face, 52% of the gamma before the volume cut,
  falls to 3.7%. Radiological (4 MC events): 25% Y and 75% Z.
- VD: gamma 98% (nominal) and 100% (shielded) through the top face; neutron 97% (nominal), 74% top and 25% X low (shielded).
- HD central: the survivors are Y-face dominated (gamma 99%, neutron 94%).

### 3.2 Reco against truth position (Table 2, residual figures)

Weighted fractions of the 10–20 MeV window with |reco − truth| < 30 cm, and the median:

| | X | Y | Z |
|---|---|---|---|
| HD central signal | 67% (median −0.03 cm) | 99.7% | 99.9% |
| HD central gamma | 76% | 94% | 94% |
| VD signal (nominal) | 44% (37% beyond 150 cm) | 99% | 99.8% |
| VD gamma (nominal) | 4.5% (median −336 cm) | 88% | 90% |
| VD neutron (nominal) | 9% (median −379 cm) | 67% | 71% |

Y and Z are wire-based and tight; the drift coordinate, which comes from the flash-matched drift time
(`lib/workflow.py`, `RecoX = |DriftTime|·(L_X/2)/EVENT_TICKS`), is the weak one. The gross-X failure rate
($|\Delta X|>100$ cm after the cut) is 5% (HD central), 14% (lateral) and 15–17% (VD) for signal, but
53–79% for gamma on HD central and VD, against 16% for gamma on HD lateral (Table F4).

### 3.3 Why the truth cut selects the same neutrons as the reco cut (Table 2)

HD central, DayNight cut, neutrons: 24 MC events have reco and truth Main within 30 cm on all axes (the reco cut
passes 99.3% of them and the truth cut 98.2%); 12 do not. Their pass fraction in the default volume:

| position treatment | pass fraction, the 12 disagreeing neutrons |
|---|---|
| reco position | 1.1% |
| truth position, fiducial only | 99.5% |
| truth position + containment + consistency (the `fiduc_truth` pipeline) | 0.7% |

The reco fiducial cut rejects them because their `RecoX` is wrong or their reconstructed cluster is a different
object from the label particle; a plain truth cut would keep them. The consistency cut of §2 restores the
same rejection, so the truth and reco pipelines end up selecting the same events.

### 3.4 Drift-coordinate residuals at the entry face (Table F2)

Gamma and neutron within 100 cm of the entry X wall(s) (HD central both walls, HD lateral $x=0$ only, VD the top face), after the cut:

| | median \|ΔX\| gamma / neutron | \|ΔX\| > 100 cm gamma / neutron | RecoX at the volume edge |
|---|---|---|---|
| HD lateral (x = 0 only) | 4.1 / 5.5 cm | 2% / 2% (neutron: 328 MC events) | 0 |
| VD nominal | 386 / 335 cm | 78% / 100% (48 MC events) | ≈ 0 |
| VD shielded | 357 / 203 cm | 83% / 81% | ≈ 0 |

For VD gamma the `End` key is validated in Y,Z (§2.1) while its X agrees within 30 cm in only 4–7% of the
events, so this is a genuine reco failure, not a label mismatch. For VD neutron even Y,Z agree with `Main`
in only 41–70% of the events, so the neutron residuals mix label mismatch and flash mismatch.

### 3.5 Mixed scans and signal recovered (Tables F3, `signal_efficiency_x`)

Grid $F_X, F_Y \in [0,340]$ cm in 20 cm steps, $F_Z = 0$. Figure of merit $S/\sqrt{B}$ with $B$ = gamma + neutron
at the analysis cut, at least 5 surviving gamma+neutron MC events (radiological excluded: 2–4 MC events with
weight ~1e5 each). This is a proxy, not the significance.

X-only best value (Y = 0), DayNight cut, and its signal efficiency:

| config | reco X, reco Y/Z | truth X, reco Y/Z | reco X, truth Y/Z |
|---|---|---|---|
| VD nominal | 3.28 (X=220, 93.9%) | 8.57 (X=100, 96.4%) | 3.28 |
| VD shielded | 14.0 (X=240, 92.3%) | 26.9 (X=100, 96.8%) | 13.6 |
| HD lateral | 6.44 (X=20, 94.9%) | 6.83 (X=20, 95.5%) | 6.98 |

- Truth X reproduces the full-truth result, and truth Y/Z with reco X reproduces the reco result: **the gain is
  the drift coordinate alone**. A control without the containment cut leaves VD unchanged (8.57), so it is
  not containment. On HD lateral the control gives 6.44 = reco, so the small lateral gain is the containment cut.
- Sensitivity cut, VD: nominal 3.62 → 7.71 (signal 80.0% → 93.6%, gamma+neutron weight 1938 → 587), shielded
  14.2 → 35.7 (77.3% → 92.1%, 117 → 26).
- Signal efficiency at the same X changes by −2.6 to +1.4 percentage points between reco and truth X (HD lateral +1.0 to +1.4; VD −0.8 to −2.6 at X = 100 cm, +0.0 to +0.3 at the default volume).

### 3.6 Flash-match purity (Table F4)

`MatchedOpFlashPur` is backtracked MC information, so it cannot be applied to data. After the cut:

| | Pur < 0.5 marley / gamma / neutron | \|ΔX\| > 100 cm among Pur ≥ 0.5 |
|---|---|---|
| HD central | 12% / 57% / 0.3% | 0% (gamma), 64% (neutron) |
| HD lateral | 31% / 32% / 34% | 0% (marley, gamma), 4.6% (neutron) |
| VD nominal | 84% / 98% / 91% | 0% |
| VD shielded | 85% / 97% / 62% | 0% |

Almost every large X error has low purity (P(low | mismatch) = 100% for VD, 97–100% for HD lateral marley and gamma, 86% for HD lateral
neutron), but purity is far less selective than position: in HD lateral only about 46% of low-purity events are mispositioned, and in VD
84% of the signal is low purity. HD central neutrons are mispositioned (64%) *despite* a pure flash, which
is the label mismatch of §3.3.

### 3.7 Radiological weight and MC statistics (Tables 4, `background_split`)

Radiological weight share of the total background after the reco fiducial (raw weights, no MC-support gate):

| | DayNight | HEP | Sensitivity |
|---|---|---|---|
| VD nominal | 99.4% | 99.7% | 99.0% |
| VD shielded | 99.96% | 99.97% | 99.9% |
| HD lateral | 98.0% | 94.1% | 98.3% |
| HD central | 99.9% | 98.7% | 99.99% |

The radiological MC support is 1–65 events with weights of about 1e5 each (2–3 events for DayNight and
Sensitivity on VD, 65 for HEP on VD nominal). On HD central the truth pipeline rejects every radiological event
(1–2 of them), because their truth `MainX/Y/Z` lies outside the active box for 94–100%: a consequence of the
truth cuts, not the absence of radiological background. The surviving gamma and neutron MC after the cut are
few (HD central: 286 and 36; VD nominal: 443 and 128), and after the reco fiducial one neutron event carries 91% of the neutron weight
on HD central.

### 3.8 Significance summary (Table 3)

| analysis | config | default | fiduc_truth | fiduc_truth_refvol |
|---|---|---|---|---|
| DayNight (σ) | HD central | 5.468 | 5.470 | 5.454 |
| | HD lateral | 1.537 | 2.113 | 1.517 |
| | VD nominal | 0.857 | 0.860 | 0.857 |
| | VD shielded | 1.910 | 1.944 | 1.896 |
| HEP (σ) | HD central | 12.181 | 12.178 | 12.178 |
| | HD lateral | 5.672 | 5.833 | 5.451 |
| | VD nominal | 2.755 | 2.754 | 2.754 |
| | VD shielded | 3.663 | 3.660 | 3.660 |
| Sensitivity (Score, Δχ²) | HD central | 5.728 | 7.539 | 7.616 |
| | HD lateral | 0.405 | 0.547 | 0.528 |
| | VD nominal | 0.080 | 0.865 | 0.751 |
| | VD shielded | 0.396 | 0.906 | 1.025 |

**Units.** DayNight and HEP are significances in σ. The Sensitivity Score is
$\mathrm{Score}=\tfrac12[\chi^2_\odot(\vec\theta_{\rm react})+\chi^2_{\rm react}(\vec\theta_\odot)]$
(`04_best_cuts.py`, `06_significance.py`; [solar_analyses.md §5.10](solar_analyses.md)), a wrong-hypothesis
$\Delta\chi^2$; its σ-equivalent is $\sqrt{\rm Score}$ (HD central 2.39 → 2.75σ, VD nominal 0.28 → 0.93σ).

**`Sigma2`.** The `Sigma2` field of the DayNight JSONs is a grid quantity: the first point of
`logspace(-1, log10(30 yr), 100) × mass` (HD central: 7.2369 kt, grid index 65 = 30.6156) at which the significance
exceeds 2σ. Default and truth runs therefore share it while `Values` differ.

---

## 4. Conclusions for the text

> **Superseded 2026-09-23 for DayNight and HEP.** These conclusions were drawn from DayNight/HEP numbers that fitted the
> truth signal against the default background (§9.7). With that fixed, truth position raises VD DayNight ×2.3–3.8 and HEP
> ×1.5–1.9; the per-detector argument is in §9.8. Points 1, 2 and 5 below still hold; point 3's "it does not move DayNight
> or HEP" and its radiological explanation do not.

1. Y and Z are wire-based and already accurate; the drift coordinate is where reco and truth differ, and the
   volume cut removes the Y-face gamma background in both.
2. **HD lateral:** the surviving background enters through the $x=0$ plane and reco localises it (median
   \|ΔX\| 4–6 cm at the $x=0$ plane). The DayNight and HEP gain of `fiduc_truth` comes from the smaller truth volume: at the
   reference volume the truth position gives 1.517 (DayNight) and 5.451 (HEP) against 1.537 and 5.672. The limit
   is geometrical at these cuts; a further gain would have to come from light-based tagging.
3. **VD:** gamma and neutron enter through the top face and reco X is wrong by more than 100 cm in 76–100% of them.
   Truth X would raise the gamma+neutron figure of merit by a factor of about 2–2.6 (DayNight and Sensitivity cuts) at equal or higher signal efficiency. It does not move DayNight
   or HEP because radiological is more than 99% of the raw background weight there (subject to the MC-support gate,
   not tested).
4. **Sensitivity** moves on all four configs. Where one radiological MC event carries essentially the whole
   background weight and the truth pipeline rejects it, the Score gain is not a measure of position information.
   The test that would settle it is a `fiduc_truth_refvol` run with radiological removed from
   `truth_match_purity.apply_to`; it has not been run.
5. The neutron truth keys describe the reconstructed cluster in only 36–70% of events, so a truth-position
   study on neutrons is not a clean measure of position resolution.

## 5. Caveats and open points

- **Z margin (partially resolved 2026-09-22, no rerun).** `build_fiducial_spatial_mask` used to hard-code `folder == "Nominal"`;
  it now reads `folder_applies_z_fiducial(root, folder)` (the `z_endcap_rejection` flag), which returns the same True/False for
  every folder configured today (Nominal only), so this is behaviour-identical and needed no rerun. The scan
  (`01_fiducialize.py`) still applies $F_Z$ unconditionally for every folder. For `Truncated` and `Reduced` the stored $F_Z$ is therefore optimised with a cut that the analysis
  later drops. Measured with [check_fiducial_z_margin.py](../../src/tools/check_fiducial_z_margin.py) on the stored Truncated volumes:
  - Scan significance (the metric `02_best_fiducial.py` maximises, 100 yr) at the stored $(F_X,F_Y,F_Z)$ against the same $(F_X,F_Y)$ with $F_Z=0$:
    DayNight 3.86 → 1.08 (HD central), 4.69 → 0.64 (HD lateral); Sensitivity 5.34 → 3.05 (HD central), 3.75 → 3.44 (HD lateral);
    HEP and VD change by 0–0.1σ except VD DayNight (0.06 → 0.04 nominal, 0.07 → 0.00 shielded). The best volume that has $F_Z=0$ differs from the
    stored one for DayNight (HD central (20,180,0) at 1.37σ against the stored (20,100,320); HD lateral (40,200,0) at 1.60σ).
  - What the Z margin removes (10–20 MeV, weight retained with Z / with $F_Z=0$), HD central DayNight $(20,100,320)$: signal 42% / 76%,
    gamma 0.9% / 1.5%, neutron 37% / 63%, radiological 9.7% / 61% (3 / 19 MC events). The gain in the scan therefore comes largely
    from the radiological component, which has no MC-support requirement in the fiducial gate (`apply_fiducial_mc_threshold` only
    counts the essential components), so the Z optimum on HD is probably an artefact of a few radiological MC events.
    VD is barely affected (retention differs by at most about 9 percentage points between the two cases).
  - Consequence: the downstream analysis does not use $F_Z$ for Truncated and Reduced, so the analysed volume is not the one that was
    optimised; the effect on the final numbers is bounded by that scan difference (largest for DayNight on HD) and has not been measured
    end to end.
  - Options: (a) scan $F_Z$ only where the folder flag `z_endcap_rejection` is `"fiducial"` (Nominal), so the scan uses the same mask as the
    analysis (matches the folder's documented intent) and re-optimise $F_X,F_Y$ (DayNight changes, see above); (b) apply the stored $F_Z$ downstream for
    every folder, which changes the analysed volume and would need the radiological support gate first; (c) leave the scan and gate the
    radiological component in the fiducial selection. Either (a) or (b) requires re-running the scan, best-fiducial selection and all
    downstream stages for Truncated and Reduced, and changes thesis numbers; none has been applied.
- Radiological has 1–65 MC events, so its fractions are not statistical statements; in particular the 0% share on HD
  central under the truth pipeline means all events were rejected. **Superseded 2026-09-22: see §9 — radiological's
  configured truth-position key was wrong (`Main`, not `End`), which is the root cause of most of the "radiological is
  unassessable / dominates everything" statements below and in §3.7. With the fix, radiological is measurable and, at
  the truth-optimal fiducial volume, is largely rejected rather than dominant — see §9.3.**
- The purity checks use MC truth (`MatchedOpFlashPur`); absent or NaN purity is counted as 0.
- VD background truth X can lie outside the active box while `RecoX` is confined to it.
- Only DayNight and Sensitivity cuts were used for the event-level checks; HEP enters through the export tables.

## 6. Section numbering

Resolved 2026-09-22 by the user: chapter 9 order is `fiduc_truth` (9.1), `energy` (9.2), `bkg_gamma` (9.3), `charge` (9.4),
`bkgmodel` (9.5), `membrane_veto` (9.6). `metric`, `unc` and `oscpoint` are not in Chapter 9 at all: they are grouped together in
Chapter 8 (main results); no per-study subsection number was given for them. `run_studies.py`, `lib/study.py` and
`solar_analyses.md` use this mapping; `thesis_plots_runbook.md`'s `## §9.x.y` headings (written for the old numbering) were not
renumbered and should be treated as stale labels, not the current thesis structure, until it is next edited.

## 7. Reproduction

```bash
# per-event caches (about 3 min per sample; auxiliary keys and purity in extract_aux), then tables and figures
src/tools/run_truth_position.sh                                  # extract + plot
python3 src/physics/signal/truth_position_study.py --stage extract_aux --config CONFIG --signals SAMPLE
python3 src/physics/signal/truth_position_study.py --stage plot  --analysis DAYNIGHT   # or SENSITIVITY
python3 src/physics/signal/truth_position_study.py --stage faces --analysis DAYNIGHT
python3 src/physics/signal/truth_position_study.py --stage export --export_dir export_v2
python3 src/tools/export_truth_position_repo.py                  # layout for the plot repository
python3 src/tools/compare_truth_position_exports.py              # 656 acceptance checks
```

All commands run under the container described in the repository (`containers/solar_v1.0.sif`).

## 8. Source index

| topic | file |
|---|---|
| study definition and history | [solar_analyses.md §5.13](solar_analyses.md) |
| Score and χ² definitions | [solar_analyses.md §5.10](solar_analyses.md) |
| fiducial mass | [SOLARReference.md](../presentations/SOLARReference.md) |
| analysis windows and bands | [config/analysis/fiducialization.json](../../config/analysis/fiducialization.json) |
| folders | [config/analysis/folder_configs.json](../../config/analysis/folder_configs.json) |
| truth keys and consistency cut | [config/analysis/backgrounds.json](../../config/analysis/backgrounds.json) |
| mask, containment, consistency | [lib/fiducial.py](../../lib/fiducial.py) |
| scan | [src/physics/signal/01_fiducialize.py](../../src/physics/signal/01_fiducialize.py) |
| selection | [src/physics/signal/02_best_fiducial.py](../../src/physics/signal/02_best_fiducial.py) |
| best volumes | [config/analysis/fiducial/truncated/](../../config/analysis/fiducial/truncated/) |
| generated tables | [DayNight](truth_position_tables_daynight.md), [Sensitivity](truth_position_tables_sensitivity.md), [HEP](truth_position_tables_hep.md), [faces DayNight](truth_position_faces_daynight.md), [faces Sensitivity](truth_position_faces_sensitivity.md) |

---

## 9. Update 2026-09-22/23: three bugs, and what actually limits the truth-position gain

Triggered by re-examining why `fiduc_truth` was not helping HD lateral/VD as much as expected. Two independent bugs were
found and fixed, one methodological pitfall was hit and corrected, and the root physics question ("is it radiological
or gamma/neutron that actually leaks into the analysed region?") was resolved by looking at per-energy-bin spectra
rather than integrated background weight.

### 9.1 Bug 1 — radiological's configured truth-position key (`Main` → `End`)

`config/analysis/backgrounds.json`'s `BACKGROUND_SAMPLES.truth_position_keys.radiological` was `MainX/Y/Z` (the
decay's bookkeeping site — for radiological `SignalParticleSurface == -1` for every event, so `Main*` was chosen as
the fallback, same as neutron). This is why §3.7 and the old caveat above describe radiological as "unassessable" and
"99–100% of the background weight, 0% surviving the truth pipeline on HD central": `Main*` does not describe the
reconstructed cluster position for a decay product, so the truth pipeline was rejecting almost everything as a
position mismatch, and the few events with `SmoothedSignificance` computed on them were essentially noise.

`src/physics/signal/truth_position_study.py`'s existing `truth_key_check` diagnostic (already implemented for
marley/gamma/neutron, just never run for radiological) settles which key is right, using the *cached* per-event data
(`--stage extract_aux`, already run; no new heavy computation needed) — fraction of weight whose reco Y/Z lands within
30 cm of each candidate key (`Main`, `MainParent`, `End`, `SignalParticle`):

| Config | N (window) | `Main` (Y/Z match, in-box) | `End` (Y/Z match, in-box) |
|---|---|---|---|
| HD central | 31 | 29%, 58% | 65%, 90% |
| HD lateral | 14 | 71%, 93% | 86%, 100% |
| VD nominal | 1293 (largest sample) | 27%, 19% | 37%, 69% |
| VD shielded | 747 | 37%, 34% | 46%, 77% |

`End` (the decay product's actual stopping/energy-deposit point) wins at every config, most convincingly at VD nominal
(the only sample with real statistics): in-box fraction 69% vs 19%. **Fix:** `truth_position_keys.radiological` is now
`EndX/Y/Z`, same reasoning as gamma. This is a real, physically-motivated correction, not a tuning choice — see the
CSV: `output/data/solar/truth_position/tables/truth_key_check_{daynight,sensitivity}.csv`.

### 9.2 Bug 2 — the `--exclude_radiological_from_scan` scan-exclusion flag dropped 8B too

Before fixing 9.1, the working hypothesis was that radiological should be dropped from the `02_best_fiducial.py`
significance sum entirely (its position wasn't fiducializable, so summing it into the scan just adds noise). A flag
(`--exclude_radiological_from_scan`, default on with `--truth_fiducial`) was added to do this — implemented by
filtering `background_components` down to components marked `ESSENTIAL: true` in
`config/analysis/backgrounds.json` (`{gamma: true, alpha: false, neutron: true, radiological: false}`). Bug: **`8B`
has no entry in `ESSENTIAL` at all**, and `HEP`'s `background_components` is `[gamma, neutron, radiological, 8B]` —
the essential-only filter silently dropped `8B` (a real solar-8B physics background with a legitimate truth position,
nothing like radiological) alongside radiological. This confounded every HEP scan result computed under the first
version of the fix.

**Fix:** `02_best_fiducial.py` no longer reuses the `ESSENTIAL` map (which drives the unrelated MC-support gate) for
this purpose. A dedicated `NON_FIDUCIALIZABLE_TRUTH_COMPONENTS = {"radiological"}` set / `filter_fiducializable_components()`
helper drops only radiological by name; every other configured background (including `8B`) is kept. Verified by
rerunning the HEP scan alone: with `8B` restored, HD lateral's HEP scan re-converges to the exact same volume
(0/80/100) it had before either bug — confirming the apparent HEP regression seen right after bug 2 was introduced
was 100% attributable to losing `8B`, not to anything about radiological.

### 9.3 Was excluding radiological from the scan actually a good idea? Yes — tested explicitly, unanimous

With 9.1 and 9.2 both fixed, radiological is a *correctly measurable* background — so the question "should the
truth-fiducial volume scan sum it in or not?" became testable rather than assumed. Ran the scan (not the full
downstream chain) both ways, same units/exposure, for all 3 analyses on 3 configs (HD central, HD lateral, VD
nominal):

| Config | Analysis | Exclude radiological | Include radiological |
|---|---|---|---|
| HD central | DayNight | 4.428 | 4.42 |
| HD central | HEP | 2.021 | 2.02 |
| HD central | Sensitivity | 3.21 | 2.99 |
| HD lateral | DayNight | 0.967 | 0.75 |
| HD lateral | HEP | 2.016 | 2.02 |
| HD lateral | Sensitivity | 3.21 | 3.09 |
| VD nominal | DayNight | 15.11 | 1.43 |
| VD nominal | HEP | 1.04 | **no admissible volume** (MC-support gate rejects every candidate; falls back to the plain reco-default volume) |
| VD nominal | Sensitivity | 40.24 | 38.70 |

Every comparison favours excluding radiological from the scan; several (VD nominal DayNight, VD nominal HEP) are not
close. Kept `--exclude_radiological_from_scan` default-on with `--truth_fiducial`. Not run downstream (the scan-level
signal was one-sided enough not to justify ~1.5 h/config of compute to confirm it); revisit if that assumption is
ever challenged.

### 9.4 Is radiological or gamma/neutron actually leaking into the analysed region? Per-bin spectra, not integrated weight

§3.7's "radiological is 94–100% of the background weight, gamma is 0–2%" is true only *before any position cut* — it
describes the raw simulated sample, not what survives at the fiducial volume the analysis actually uses. Checked
directly with the raw per-bin `Fiducial_Scan_fiduc_truth.pkl` counts (HD lateral, HEP truth-optimal volume
X=0/Y=80/Z=100, 1 MeV bins):

| Energy (MeV) | hep signal | gamma | neutron | radiological |
|---|---|---|---|---|
| 14.5 | 0.84 | 238.9 | 1034.0 | 0 |
| 15.5 | 0.64 | 89.4 | 6.9 | 0 |
| 16.5 | 0.43 | 22.8 | 208.2 | 0 |
| 17.5 | 0.25 | 5.5 | 1.6 | 0 |
| 18.5–19.5 | ~0.1 | ~0.6 | ~0.2 | 0 |
| 20.5–29.5 | ~0 | ~0 | ~0 | 0 |

At this volume the position + containment + consistency mask rejects **every** radiological event outright (0 counts
across the whole 14–30 MeV window), while gamma and neutron survive and are the entire surviving background,
concentrated in **14–20 MeV**. Radiological dominates the raw sample (§3.7) precisely because it is the easiest
component to remove completely with the right position cut; gamma/neutron survive the same cut and are what is
actually left in the region that matters. High-energy (20–30 MeV) gamma is not a real effect here: its raw count is
~1e-13, i.e. gamma does not populate the high sub-band at all at this working point.

**Open finding, not yet resolved:** the scan itself reports the *entire* HEP significance for this volume (2.02) as
coming from the 20–30 MeV sub-band, despite raw signal and background both being ~0 there. Traced to
`smooth_histogram_with_config`'s data-driven Gaussian smoothing: in this deep zero-count tail it leaves smoothed
signal (~1e-3–1e-4, from the `hep` flux's true-energy endpoint at ~18.8 MeV smearing into the reco tail) and smoothed
background residuals that fall off at different rates, so the Asimov formula computes a non-trivial but likely
meaningless significance from the smoothing kernel's interaction with a hard kinematic cutoff, not real
signal/background separation. This may mean the HEP high sub-band's contribution to volume selection is partly
artifact-driven; not yet checked whether this changed which volume `02_best_fiducial.py` picked for any config.

### 9.5 Methodological pitfall: partial reruns of fiducial-volume-dependent data silently mix volumes

After fixing 9.1/9.2, a "targeted" rerun was used to save compute: only `01_fiducialize.py`/`03_analysis.py` for
`radiological` were rerun (its truth key changed), while `marley`/`gamma`/`neutron`/`8B`'s existing `Ref`/`Rebin` pkls
were reused on the assumption they were "unaffected." This produced a spurious result: HD lateral's HEP dropped from
5.83σ to 4.84σ, appearing to show that the corrected (honest) radiological modelling made truth-fiducialisation a
genuine loss for this config/analysis.

**This was wrong.** Every sample's `Ref`/`Rebin` data (built by `03_analysis.py --export_fiducial`) is built against
*whichever fiducial volume was current in `BestFiducials_fiduc_truth.json` at the time it was generated* — not
against the sample's own properties. `marley`'s data had last been built while `BestFiducials_fiduc_truth.json` held
a transient, bug-affected HEP volume (20/20/100, from bug 2 above) from an earlier pass; only `radiological` was
refreshed at the newly-corrected volume (0/80/100). The downstream significance was therefore computed from a
mismatched mix of per-component fiducial volumes — a real bug in the shortcut, not a physics result. A full clean
rerun (`run_studies.py --study fiduc_truth`, every sample refreshed together at the same final volume) reproduces the
original, correct 5.83σ exactly.

**Lesson:** fiducial-volume-dependent per-sample data (`Ref`/`Rebin` pkls) cannot be partially refreshed across
components after the stored best-fiducial volume changes — every sample sharing that volume must be regenerated
together, or downstream numbers silently mix inconsistent volumes.

### 9.6 Numbers after the full clean rerun of 2026-09-22 (superseded for DayNight/HEP by §9.7)

The Sensitivity column below is final. The DayNight/HEP entries of this table were computed with the background-label
bug of §9.7 (truth signal fitted against the default reco-fiducialised background) and are **not valid**; see §9.7.

| Config | Sensitivity (Δχ², 30 yr) | vs default |
|---|---|---|
| HD central | 7.33 | 5.79 |
| HD lateral | 0.62 | 0.41 |
| VD nominal | 0.91 | 0.08 |
| VD shielded | 0.92 | 0.40 |

VD shielded's first clean-rerun attempt hung ~7 h in a kaleido/headless-Chrome plot export (same failure mode as on
2026-09-21); killed and relaunched, completed 2026-09-23 00:22.

### 9.7 Bug 3 — `01_daynight.py` / `01_hep.py` fitted the truth signal against the default background

`load_available_background_dataframes(..., study_label=...)` was given the study label in `01_daynight.py` and
`01_hep.py` **only when `charge_threshold > 0`** (the comment still said "background Rebin pkls are labeled only for
charge variants"). Since the truth-fiducial and membrane-veto variants started writing labeled background Rebins
(`{config}_{sample}_{energy}_Rebin_{label}.pkl`, see `lib/study.py::study_context`), those two readers kept loading the
**default** `_Rebin.pkl` — reco positions at the default volume — while the signal came from the labeled file. Every
DayNight and HEP number of `fiduc_truth`, `fiduc_truth_refvol` and `membrane_veto_off` therefore compared a
truth-selected (or veto-off) signal with the default background. The Sensitivity stages (`01_background_template.py`,
`04_best_cuts.py`) and `significance_plot.py` already loaded the labeled background, which is why only Sensitivity showed
a consistent `fiduc_truth` gain. The same charge-only rule sat in `run_sensitivity.py`'s automatic HEP MC-threshold
choice.

**Verification before the fix.** The HEP profile likelihood was recomputed outside the pipeline with the library function
`01_hep.py` uses (`evaluate_profile_likelihood_discovery`: raw rates, `min_mc_per_bin` mask, 2% background
normalisation, conservative signal offset, 30 yr) at each study's best cut. With the default background it reproduces
the stored `PreIsotonicProfileLikelihood` exactly (e.g. VD shielded 3.9097); with the labeled truth background it gives
VD nominal 2.94 → 5.58, VD shielded 3.91 → 5.60, HD lateral 6.22 → 6.47, HD central unchanged (pre-isotonic values; the
reported value is this curve after the monotone smoothing over exposure, ~6% lower).

**Fix (2026-09-23).** All three readers use the rule of the Sensitivity stages: labeled background whenever the variant
changes the background event selection — `charge_threshold > 0 or truth_fiducial or not membrane_veto`. dm2/uncertainty
variants keep reading the default background (their backgrounds are identical). DayNight and HEP were rerun for all
three affected studies (`--no-rebin`: the labeled Rebins were already correct; `fiduc_truth` volumes re-derived and
checked identical). Results, `study_status_20260923_bkgfix.csv`, 30 yr:

| Config | Analysis | default | `fiduc_truth_refvol` (truth pos., reco vol.) | `fiduc_truth` (truth pos., truth vol.) | before fix |
|---|---|---|---|---|---|
| HD central | DayNight | 5.47 | 6.04 | **6.14** | 5.47 |
| HD central | HEP | 12.18 | 11.39 | **12.06** | 12.18 |
| HD lateral | DayNight | 1.54 | 1.70 | **2.06** | 1.75 |
| HD lateral | HEP | 5.67 | 7.18 | **6.62** | 5.83 |
| VD nominal | DayNight | 0.86 | 2.56 | **3.27** | 0.80 |
| VD nominal | HEP | 2.76 | 5.35 | **5.37** | 2.75 |
| VD shielded | DayNight | 1.91 | 4.33 | **4.34** | 1.87 |
| VD shielded | HEP | 3.66 | 5.62 | **5.62** | 3.66 |

`membrane_veto_off` (VD, held default cut) falls back onto the default within 1–2% (VD nominal DayNight 0.86, HEP 2.75;
VD shielded DayNight 1.90, HEP 3.61): its earlier small gains came from the veto-off signal against the veto-on
background. Lifting the membrane veto does not change the result.

Two consequences worth stating: (i) at the reco volume, truth positions already give most of the gain
(`fiduc_truth_refvol` ≈ `fiduc_truth` on VD), so the gain is position knowledge rather than volume choice; (ii) on HD
lateral HEP the truth-volume scan picks a worse volume (0/80/100, 6.62σ) than the reco reference (60/80/20, 7.18σ at
truth positions) — consistent with the scan-metric artefact in the HEP high band noted in §9.4.

### 9.8 What limits the gain — the argument and its figures

Figures from `src/physics/signal/fiduc_truth_limits.py` (reads existing outputs only), in
`output/images/solar/truth_position/`:

- `limits_summary` — the table of §9.7 as bars.
- `limits_composition_hep` — HEP background composition per energy bin at each study's own working point, default vs
  `fiduc_truth`, from the analysis's own `HEP_Counts.pkl` (Raw spectra, as fitted).
- `limits_migration_hep` — reconstructed `SolarEnergy` vs `MainK` of the gamma and neutron that survive the default HEP
  cut and volume.

**Pickles for the plot repository** (LOWE_RECONSTRUCTION_PUBLICATION). `scripts/sync_solar_data.sh` pulls every `*.pkl` under
`output/data/analysis/{day-night|hep|sensitivity}/{config}/marley/truncated/{label}/` for labels registered in its
`src/lib/solar_studies.py`; the macros load `{config}_{name}_{datafile}.pkl` and filter on `Name`. Written under label `fiduc_truth`:

| file (`{config}_marley_…`) | written by | content |
|---|---|---|
| `{DayNight,HEP}_FiducTruthSummary.pkl` | `fiduc_truth_limits.py --stage export` | significance per `Variant` (default / `fiduc_truth_refvol` / `fiduc_truth`), `RatioToDefault`, cut, volume |
| `{DayNight,HEP}_FiducTruthComposition.pkl` | same | the three variants' `{Analysis}_Counts` spectra in one file (`Variant`, `Component`, `SpectrumType`) |
| `HEP_FiducTruthMigration.pkl` | same | per-event `TrueEnergy` (MainK), `RecoEnergy`, `Weight`, `InWindow` for gamma and neutron at the default HEP cut and volume |
| `{DayNight,HEP,Sensitivity}_TruthPosition{Kind}.pkl` | `src/tools/export_truth_position_repo.py` | the truth-position checks of §3 (kinds in `output/data/solar/truth_position/export_repo/README.md`) |

`fiduc_truth_refvol` is not a registered label in the plot repo, so its own folder is not synced; its numbers are in
`FiducTruthSummary`.

**The argument, per detector.**

1. **VD (nominal, shielded): position was the limit, and truth position lifts it.** In the bins that carry the HEP and
   DayNight significance the default background is gamma + neutron; reco puts them inside the volume because their
   drift coordinate is wrong by hundreds of cm (§3.2, §3.4). With truth positions they essentially vanish from the signal
   bins (`limits_composition_hep`), the optimiser can loosen the topological cut (VD nominal DayNight: NHits 6/OpHits 13 →
   2/4, signal ×3–4 per bin at a lower background), and DayNight rises ×2.3–3.8, HEP ×1.5–1.9. Radiological is exactly 0 in
   the significance-carrying DayNight bins (10.5–15.5 MeV) in both pipelines, so the gain is not a few rejected
   radiological MC events.
2. **What remains after perfect position is ⁸B.** In `limits_composition_hep` the `fiduc_truth` VD panels, like both HD
   central panels, have an almost pure ⁸B background above 16 MeV. ⁸B is spatially identical to the hep signal, so no
   fiducial cut can remove it; it is separated from hep only by how sharply the ⁸B spectral endpoint is reconstructed.
   This is where energy resolution becomes the limit: on HD central from the start (HEP unchanged, 12.18 → 12.06), on
   VD once truth position has removed gamma and neutron.
3. **HD lateral: reco already localises the background.** It enters through the x = 0 plane where reco X is good
   (median |ΔX| 4–6 cm, §3.4); truth position removes part of the gamma/neutron but not all, hence the moderate gain.

**Two claims that do not hold and should not be made.**

- *"Truth energy would gain 1.7–4.6× on HEP"* (from `energy_maink`/`energy_spk`): not usable. With `MainK` as the energy
  variable the hep signal in the fixed 14–30 MeV window drops from ~225 to ~3 events (MainK is the electron only, ~5 MeV
  below `SolarEnergy`) and the raw background there is ~1e-13 — the significance comes from empty-background bins.
- *"The harmful gammas are energy-mismeasured"*: the in-window gammas have true energies of 11–14 MeV (generator
  endpoint) and `SolarEnergy` sits ~+5 MeV above `MainK` for the signal as well — it is a neutrino-energy estimator. In
  the analysis variable these gammas are genuinely signal-like; they are separable by position (and topology), not by
  better energy resolution. Neutron `MainK` shows capture lines at ~7.6–10.8 MeV from captures outside the argon, so
  the 6.1 MeV ⁴⁰Ar capture energy is not a bound for this sample.

### 9.9 `membrane_veto_off` — why lifting the VD membrane veto changes nothing (2026-09-23)

Full account, including the PDS-plane decomposition of every sample and the plot-repo pickles: [membrane_veto.md](membrane_veto.md).

The quality mask (§1.2) accepts only optical matches on plane 0 (the cathode for VD). `membrane_veto_off` also accepts
planes 1–4 (membrane and end-cap photon detectors) at the default cut and volume (`skip_best_cuts`, VD only). Its DayNight and
HEP results equal the default within 1–2% (§9.7). This is a physics result, not an artefact:

- **Event level** (truth-position caches, default HEP cut and volume, 14–30 MeV): lifting the veto adds signal +24.5% (VD nominal) /
  +20.8% (VD shielded), but gamma +63% / +131% and neutron +67% / +155%. Membrane-matched clusters are background-enriched:
  external gammas and neutrons interact near the membranes and are matched to the membrane photon detectors, which is what the veto
  exists to reject.
- **Per bin, as fitted** (`{Analysis}_Counts.pkl`, significance-weighted over the bins that carry the test statistic): DayNight
  S ×1.02–1.03 against B ×1.08; HEP S ×1.20–1.26 against B ×3.2–7.2 (neutron ×37–40 in single bins), so S/√B is flat or lower.
- **Fit**: an independent evaluation of the HEP profile likelihood (same function, raw rates, MC-support mask, 2% normalisation)
  with veto-off signal and background at the default cut gives VD nominal 2.944 → 2.943 and VD shielded 3.910 → 3.856
  (pre-isotonic), for +45% and +41% more signal events over the full spectrum. The extra signal is cancelled by the extra
  background.

Two bookkeeping bugs were found while checking this, neither of which changed a reported significance:

1. `significance_plot.py` and `exposure_plot.py` had no `--membrane_veto` argument and `run_sensitivity.py` did not forward it, so
   the synced `membrane_veto_off/*_{Counts,Significance,Exposure}.pkl` were built from the default (veto-on) Rebins and were
   bit-identical to the default ones. Fixed (argument added; forwarded in all 11 calls); the VD plots were regenerated.
2. `03_analysis.py` labeled the best-cut exports (`AnalysisMask`/`AnalysisData`/`AnalysisEnergy`/`AnalysisWeights_*_NHits*`) only for
   truth-fiducial variants. `membrane_veto_off` holds the default cut, so its export overwrote the default-named VD files with veto-off
   content. No SOLAR script reads them and they are not synced; they are labeled for veto-off runs now (`_mask_sfx`) and the
   default VD files were rewritten (verified: 0 selected events on planes 1–4).
