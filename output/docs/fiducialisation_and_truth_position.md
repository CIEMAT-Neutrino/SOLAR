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
| gamma, HD lateral | ×126 | ×24 |
| gamma, VD nominal | ×40 | ×15 |
| gamma, VD shielded | ×53 | ×19 |
| neutron, HD central / lateral | ×0.7 / ×4.2 | ×0.9 / ×1.1 |
| neutron, VD (nominal / shielded) | ×19 / ×19 | ×2.2 / ×2.4 |

Gamma piles up at the walls, signal and (in HD) neutron do not. The nearest face of the *surviving* background
(after the DayNight cut and the reco fiducial, weight share):

- HD lateral: gamma 90% X low (the $x=0$ plane), neutron 31% X low and 65% X high; the Y face, 49% of the gamma
  before the volume cut, falls to 0.9%. Radiological (4 MC events): 75% X high, 25% Z.
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

Gamma and neutron within 100 cm of the entry X face(s) (both for HD, top for VD), after the cut:

| | median \|ΔX\| gamma / neutron | \|ΔX\| > 100 cm gamma / neutron | RecoX at the volume edge |
|---|---|---|---|
| HD lateral | 4.9 / 14.5 cm | 5% / 25% | ≤ 1% |
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

1. Y and Z are wire-based and already accurate; the drift coordinate is where reco and truth differ, and the
   volume cut removes the Y-face gamma background in both.
2. **HD lateral:** the surviving background enters through the X faces and reco localises it (median
   \|ΔX\| 5–15 cm). The DayNight and HEP gain of `fiduc_truth` comes from the smaller truth volume: at the
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

- **Z margin.** `01_fiducialize.py` applies the $F_Z$ cut unconditionally in the scan, whereas
  `lib/fiducial.py: build_fiducial_spatial_mask` applies it only for `folder == "Nominal"` (hard-coded, not read from
  the `z_endcap_rejection` folder flag). For `Truncated` and `Reduced` the $F_Z$ stored in `BestFiducials.json` is
  therefore optimised with a cut that the analysis later drops. This was read from the code, not tested by a run;
  it should be resolved or stated in the text (my scans use $F_Z=0$ for this reason).
- Radiological has 1–65 MC events, so its fractions are not statistical statements; in particular the 0% share on HD
  central under the truth pipeline means all events were rejected.
- The purity checks use MC truth (`MatchedOpFlashPur`); absent or NaN purity is counted as 0.
- VD background truth X can lie outside the active box while `RecoX` is confined to it.
- Only DayNight and Sensitivity cuts were used for the event-level checks; HEP enters through the export tables.

## 6. Section numbering

`run_studies.py` numbers `fiduc_truth` 9.2.1; [solar_analyses.md](solar_analyses.md) numbers it 9.2.2; the
runbook ([thesis_plots_runbook.md](thesis_plots_runbook.md)) calls 9.2.2–9.2.3 "Energy Reconstruction / Photon
Detection". These should be reconciled before the section is written.

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
| generated tables | [DayNight](truth_position_tables_daynight.md), [Sensitivity](truth_position_tables_sensitivity.md), [faces DayNight](truth_position_faces_daynight.md), [faces Sensitivity](truth_position_faces_sensitivity.md) |
