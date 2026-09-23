# The VD membrane veto: why lifting it does not improve the analysis

Support document for the `membrane_veto` study (Chapter 9.6). Written 2026-09-23. Numbers come from:
the per-event caches of `src/physics/signal/truth_position_study.py` (via `src/physics/signal/pds_plane_decomposition.py`,
table `output/data/solar/truth_position/tables/pds_planes.csv`), the analysis's own `{Analysis}_Counts.pkl`, an independent
evaluation of the HEP profile likelihood, and `output/logs/study_status_20260923_151229.csv`. Weights are
`SignalParticleWeight` (no oscillation weighting for backgrounds; `marley` combines all solar fluxes). Folder `Truncated`.

---

## 1. What the veto is

Every cluster is matched to an optical flash; the flash's photon-detector plane is `MatchedOpFlashPlane`
(`lib/fiducial.py::accepted_flash_planes`):

| plane | meaning |
|---|---|
| −1 | no flash match (always with `MatchedOpFlashPE` = 0) |
| 0 | cathode (VD) / APA (HD) |
| 1, 2 | membranes (VD) |
| 3, 4 | end caps (VD) |

The quality mask keeps plane 0 only (`QUALITY_CUTS.OPFLASH_PLANE = 0` in `config/analysis/config.json`); this is the
membrane veto. HD only ever reports planes −1 and 0, so the veto is free there and the study runs on VD only.
`membrane_veto_off` (`--no-membrane_veto`, `lib/study.py`) accepts planes 1–4 as well, at the default cut and volume
(`skip_best_cuts`, `skip_best_sigmas`).

## 2. Result

| VD config | analysis | default | `membrane_veto_off` | record (veto off) |
|---|---|---|---|---|
| nominal | DayNight | 0.86σ | 0.86σ | 09-23 12:22 |
| nominal | HEP | 2.76σ | 2.75σ | 09-23 12:27 |
| nominal | Sensitivity (Δχ²) | 0.08 | 0.07 | 09-20 18:30 |
| shielded | DayNight | 1.91σ | 1.90σ | 09-23 12:22 |
| shielded | HEP | 3.66σ | 3.61σ | 09-23 12:27 |
| shielded | Sensitivity (Δχ²) | 0.40 | 0.34 | 09-20 18:29 |

30 yr, same cut and volume in both columns. Lifting the veto changes DayNight and HEP by 0–1.4% and lowers the
Sensitivity score. The Sensitivity rows compare runs from 09-18 (default) and 09-20 (veto off), i.e. across code versions;
their direction agrees with the rest but the size should be re-measured once the Truncated Sensitivity default is rerun.

## 3. Where the clusters are matched

All cached events (VD nominal; VD shielded is identical at this stage, it differs only after the cuts), weighted:

| sample | N MC | no match | plane 0 (cathode) | plane 1 (membrane) | plane 2 (membrane) | planes 3–4 (end caps) |
|---|---|---|---|---|---|---|
| marley (signal) | 70 092 | 1.3% | 58.7% | 19.9% | 20.0% | < 0.1% |
| gamma | 69 355 | 6.3% | 34.8% | 29.5% | 29.5% | < 0.1% |
| neutron | 199 695 | 4.3% | 40.1% | 28.2% | 27.4% | < 0.1% |
| radiological | 4 305 628 | 0.0% | 88.7% | 5.6% | 5.6% | < 0.1% |

Among matched clusters the signal is 59.5% cathode-matched, gamma 37.1–37.4% and neutron 41.9%; the external gammas and
neutrons interact near the walls and are seen by the wall (membrane) photon detectors. HD: 100% plane 0 for every sample.
End caps carry almost nothing (planes 3 and 4: 7 + 4 marley, 5 + 4 gamma, 12 + 15 neutron MC events in the whole sample).

## 4. What lifting the veto adds at the working point

Weight on planes 1–4 relative to plane 0 (= the fractional increase when the veto is lifted), per selection stage at each
analysis's default working point; in brackets the number of MC events added. From `pds_planes.csv`.

**HEP** (14–30 MeV)

| stage | VD nominal (cut 5/4/14, vol 0/0/40): marley / gamma / neutron | VD shielded (cut 7/4/15, vol 0/0/20): marley / gamma / neutron |
|---|---|---|
| matched | +68.0% / +169.6% / +138.7% | +68.0% / +167.7% / +138.7% |
| + cut | +51.8% / +131.8% / +98.1% | +49.6% / +151.0% / +137.1% |
| + window + fiducial | **+24.5%** [6307] / **+62.8%** [4286] / **+66.9%** [238] | **+20.8%** [4766] / **+131.4%** [1287] / **+155.2%** [129] |

**Sensitivity** (10–30 MeV, cut 6/4/6, vol 0/0/20)

| stage | VD nominal: marley / gamma / neutron | VD shielded: marley / gamma / neutron |
|---|---|---|
| + window + fiducial | +40.1% / +100.9% / +190.6% | +39.8% / +124.4% / +203.0% |

**DayNight** (6–18 MeV)

| stage | VD nominal (cut 6/13/6, vol 0/20/60): marley / gamma / neutron | VD shielded (cut 6/13/7, vol 20/100/20): marley / gamma / neutron |
|---|---|---|
| + cut | +3.0% / +5.0% / +0.8% | +3.2% / +10.6% / +10.2% |
| + window + fiducial | +2.7% [186] / +3.4% [99] / +0.8% [27] | +2.2% [112] / +3.9% [36] / +8.4% [19] |

Radiological adds nothing at any working point: 0 MC events after the energy window (after the cut alone: +4.0% HEP VD nominal,
0% HEP VD shielded, +4.8% Sensitivity).

Two regimes:

- **DayNight:** the default cut requires ≥ 13 optical hits, which membrane-matched flashes rarely reach; the cut already
  removes them, and lifting the veto adds ~3% of everything.
- **HEP and Sensitivity** (≥ 4 optical hits): membrane matches survive, and lifting the veto adds 21–40% more signal but
  63–203% more gamma and neutron. The added clusters are background-enriched, which is what the veto exists to reject.

## 5. In the bins that carry the significance

`{Analysis}_Counts.pkl` (raw spectra, as fitted), weighted by each bin's share of the default test statistic, veto off ÷ default:

| | signal | total background | gamma | neutron |
|---|---|---|---|---|
| DayNight VD nominal | ×1.033 | ×1.075 | ×1.03 | ×1.22 |
| DayNight VD shielded | ×1.024 | ×1.081 | ×1.04 | ×1.08 |
| HEP VD nominal | ×1.200 | ×7.19 | ×1.17 | ×39.8 |
| HEP VD shielded | ×1.262 | ×3.15 | ×1.98 | ×37.0 |

HEP VD nominal per bin: 19.5 MeV (27% of the test statistic) S 13.84 → 16.81, B 4331 → 99 950 (a few neutron MC events);
20.5 MeV (69%) S 8.24 → 9.77, B 1966 → 2416, S/√B 0.186 → 0.199.

## 6. Independent check of the HEP fit

The HEP profile likelihood recomputed with the function `01_hep.py` uses (`evaluate_profile_likelihood_discovery`: raw rates,
`min_mc_per_bin` mask, 2% background normalisation, conservative signal offset, 30 yr; equals the stored
`PreIsotonicProfileLikelihood`), veto-off signal and background, default cut:

| | signal (all energies) | PL default | PL veto off |
|---|---|---|---|
| VD nominal (5/4/14) | 91.1 → 131.8 (+45%) | 2.944 | 2.943 |
| VD shielded (7/4/15) | 37.9 → 53.3 (+41%) | 3.910 | 3.856 |

The extra signal is cancelled by the extra background. A result of this kind is expected whenever the added sample has a
lower S/B than the kept one.

## 7. Drift coordinate of membrane-matched clusters

Weighted fraction of matched clusters with |RecoX − truth X| < 10 cm (VD, all matched events):

| sample | plane 0 | plane 1 | plane 2 |
|---|---|---|---|
| marley (truth X = `SignalParticleX`; `TruthX` gives the same) | 28.6% | 10.3% | 11.1% |
| gamma (truth X = `EndX`) | 2.2% | 4.5% | 4.4% |
| neutron (truth X = `MainX`) | 9.0% | 5.5% | 7.2% |

The docstring of `accepted_flash_planes` states that membrane matches reconstruct the drift coordinate as well as plane 0
("> 94% within 10 cm, against 96.8% for the cathode"). That is **not reproduced** for the signal with `SignalParticleX` or `TruthX` over all
matched clusters; its measurement conditions are not recorded. Either way membrane matches are not better localised than cathode
matches, so drift-coordinate quality is not an argument for lifting the veto. The comment was left unchanged pending the
original measurement.

## 8. Bookkeeping bugs found while checking (fixed, commit `50aa3d4`)

None changed a reported significance.

1. The DayNight/HEP fits read the default background for this study (the background-label bug of
   `fiducialisation_and_truth_position.md` §9.7). Before that fix the study showed small gains (0.88σ, 3.79σ) from a veto-off
   signal against a veto-on background; corrected numbers are in §2.
2. `significance_plot.py` and `exposure_plot.py` had no `--membrane_veto` argument and were not forwarded it, so the synced
   `membrane_veto_off/*_{Counts,Significance,Exposure}.pkl` were built from the default Rebins (bit-identical to the default
   files). The Rebins themselves were correct: at the HEP cut the veto-off signal is +45% (MC 18 849 → 26 710) and gamma ×2.3
   (MC 3 840 → 9 289) over all energies. Fixed; plots regenerated.
3. `03_analysis.py` wrote the best-cut event exports (`AnalysisMask`/`AnalysisData`/`AnalysisEnergy`/`AnalysisWeights_*_NHits*`)
   under the default name for veto-off runs; the study holds the default cut, so it overwrote the default VD files. They are now
   labeled; the default files were rewritten. Verified at the HEP default cut:

   | | VD nominal: selected / on planes 1–4 | VD shielded: selected / on planes 1–4 |
   |---|---|---|
   | default | 19 719 / 0 | 13 241 / 0 |
   | `membrane_veto_off` | 27 745 / 8 026 | 18 338 / 5 097 |

## 9. Data for the plot repository

`src/physics/signal/pds_plane_decomposition.py` writes one pickle per config, sample and analysis into the synced tree
(label `default`, registered for every config):

`output/data/analysis/{day-night|hep|sensitivity}/{config}/marley/truncated/default/{config}_{sample}_{Analysis}_PDSPlanes.pkl`

Rows: one per (`Selection`, plane). Columns: `Geometry`, `Config`, `Name` (= sample), `Analysis`, `Study`, `Folder`,
`Selection` (`all`, `matched`, `cut`, `cut+window`, `cut+window+fiducial`), `Variable` (plane label: `No match`,
`P0 cathode/APA`, `P1 membrane`, `P2 membrane`, `P3 end cap`, `P4 end cap`), `Plane`, `PlaneRole`, `Fraction`,
`FractionError` (binomial, effective MC size), `FractionMC`, `RelativeToPlane0`, `FractionDXWithin`, `DXWithinCm`, `TruthKey`,
`Weight`, `WeightStage`, `NMC`, `NMCStage`, `NEffStage`, working point (`NHits`, `OpHits`, `AdjCl`, `EnergyMin`, `EnergyMax`,
`FiducialX/Y/Z`).

Plane decomposition table with `scripts/script_aggregate_table.py` (rows config × sample, columns planes):

```bash
python3 scripts/script_aggregate_table.py --configs hd_1x2x6_centralAPA hd_1x2x6_lateralAPA \
    vd_1x8x14_3view_30deg_nominal vd_1x8x14_3view_30deg_shielded \
    --names marley gamma neutron radiological --datafile HEP_PDSPlanes \
    --y Fraction --variables Variable --row_name Name --select Selection --save_values matched
```

Other views: `--save_values cut+window+fiducial` for the working point; `--y RelativeToPlane0` for what lifting the veto
adds; `--y FractionDXWithin` for §7; `--datafile DayNight_PDSPlanes` / `Sensitivity_PDSPlanes` for the other analyses.

## 10. Reproduction

```bash
python3 src/physics/signal/pds_plane_decomposition.py          # needs the truth_position_study.py caches
python3 src/pipelines/run_studies.py --study membrane_veto --config vd_1x8x14_3view_30deg_nominal vd_1x8x14_3view_30deg_shielded
```

All commands run under the container (`containers/solar_v1.0.sif`).
