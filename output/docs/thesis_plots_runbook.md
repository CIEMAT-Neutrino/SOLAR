# Thesis Chapter 9 — Plot Commands

*Generated 2026-08-26. Availability re-verified against PNFS 2026-09-06.*

> ## ⚠ Do not plot study variants yet — rerun pending
>
> Two changes landed 2026-09-06 that invalidate most study numbers now on disk. Plotting
> today produces figures that will not survive review.
>
> **1. Pipeline bug.** `--skip_best_cuts` gated `01_daynight.py` and `01_hep.py` as well as
> `04_best_cuts.py`, contrary to its own help text. Any DayNight/HEP variant carrying that
> flag never computed its own significance grid, and `seed_study_artifacts_from_nominal()`
> then copied nominal Results onto the study's label — so it reported **nominal physics
> under its own name** rather than failing. Both call sites removed; both scripts now always
> run. Confirmed on disk: `unc_bkg` DayNight Asimov must equal the default (σ_bkg-invariant)
> but does not — LAPA 0.915 vs 0.908, VDN 0.505 vs 0.500, VDS 0.972 vs 1.075, on grids still
> dated Aug 19–22.
>
> **2. One-knob policy.** Every variant now holds cuts and smoothing sigmas at nominal
> (`skip_best_cuts=True`, `skip_best_sigmas=True`) and changes exactly one thing. Variants
> that previously re-optimised their own cuts (`energy`, `charge`, `fiduc_truth`,
> `bkg_gamma`, `oscpoint_reactor`) produced valid physics but under a re-tuned analysis, so
> their delta vs default confounds the knob with the re-fit.
>
> Rerun commands: § Commands tab of the studies artifact, or `run_studies.py` per group.
> Only `default`, `unc_sig0/2/6` and `nuisance_*` are unaffected — those three are
> Sensitivity-only, where `--skip_best_cuts` correctly gated `04_best_cuts.py` alone.

`sync_solar_data.sh` default remote: `gae_out:/pc/choozdsk01/users/manthey/SOLAR`
Study-variant pkls auto-route to `input/data/studies/` by the sync script.
Plot scripts fall back `input/data/` → `input/data/studies/` transparently.
`--study`/`--config`/`--name`/`--folder` are repeatable (one value per flag).

Datafile stem pattern (`--configs` + `--name` supplied separately):
- `--datafile {Analysis}_{Type}` → `input/data/{config}_{name}_{Analysis}_{Type}.pkl`
- `--datafile {Analysis}_{Type}_{study}` → `input/data/studies/{config}_{name}_{...}.pkl`

**Presence legend:** `all 4` / `CAPA` etc. = configs with the pkl on PNFS 2026-09-06 · `—` = absent
**Validity legend:** `OK` = usable now · `RERUN-A` = wrong numbers (grid never computed) ·
`RERUN-B` = valid physics, re-tuned cuts, not comparable · `ORPHAN` = no longer in `STUDY_VARIANTS`

## Config Shortnames

| Shortname | Full config key |
|---|---|
| `CAPA` | `hd_1x2x6_centralAPA` |
| `LAPA` | `hd_1x2x6_lateralAPA` |
| `VDN` | `vd_1x8x14_3view_30deg_nominal` |
| `VDS` | `vd_1x8x14_3view_30deg_shielded` |

## Availability Matrix (truncated / marley — PNFS, 2026-09-06)

Presence is what is on disk. **Validity is the column that decides whether you may plot it.**

| Study variant | DayNight | HEP | Sensitivity | Validity |
|---|---|---|---|---|
| default | all 4 | all 4 | all 4 | **OK** |
| unc_sig0/2/6 | — | — | all 4 | **OK** (Sensitivity-only) |
| nuisance_nominal/sin13/escale | — | — | all 4 | **OK** (Sensitivity-only) |
| unc_bkg0/4/6 | all 4 | all 4 | all 4 | `RERUN-A` grids dated Aug 19–22 |
| unc_sig20/40 | — | all 4 | — | `RERUN-A` |
| oscpoint_solar | all 4 | all 4 | all 4 | `RERUN-A` (= default by construction) |
| fiduc (Nominal/Reduced/Truncated) | all 4 | all 4 | all 4 | `RERUN-A` |
| bkgmodel (Nominal/Reduced) | all 4 | all 4 | all 4 | `RERUN-A` |
| membrane_veto_off | VD only | VD only | VD only | `RERUN-A` |
| oscpoint_reactor | all 4 | all 4 | — *(invariant, by design)* | `RERUN-B` |
| energy_spk | all 4 | all 4 | CAPA | `RERUN-B` |
| energy_maink | all 4 | all 4 | — | `RERUN-B` |
| charge_Q50 | all 4 | all 4 | all 4 | `RERUN-B` |
| charge_Q100 | all 4 | all 4 | CAPA | `RERUN-B` |
| charge_Q500 | all 4 | all 4 | all 4 | `RERUN-B` |
| fiduc_truth | all 4 | all 4 | all 4 | `RERUN-B` |
| bkg_gamma | all 4 | all 4 | CAPA | `RERUN-B` |
| metric_raw/smoothed | CAPA | CAPA | CAPA | stale (Aug 05); LAPA/VDN/VDS never run |
| charge_Q200 | all 4 | all 4 | CAPA | `ORPHAN` — dropped from `STUDY_VARIANTS` |

**`charge_Q200` is orphaned.** The pkls exist (Aug 23) but the variant list now defines
Q50/Q100/Q500 only, so Q200 cannot be regenerated and will never be policy-compliant. The
plot commands below use **Q500** instead. Either re-add Q200 to `STUDY_VARIANTS` or drop it
from the thesis figures — do not mix it with reruns.

**`fiduc_truth` caveat.** VDN reports DN 0.500 / EG 0.357 / HEP 2.886 — identical to VDN
default on all three. Possibly real, possibly a stage that silently fell back. Verify before
using; VDS HEP (3.129) matches default too.

---

## Sync

`sync_solar_data.sh` prompts `[y/N]` — run interactively.

```bash
# DayNight (all 4 configs, all available studies)
sync_solar_data.sh \
  --analysis daynight \
  --config hd_1x2x6_centralAPA \
  --config hd_1x2x6_lateralAPA \
  --config vd_1x8x14_3view_30deg_nominal \
  --config vd_1x8x14_3view_30deg_shielded \
  --name marley \
  --folder truncated \
  --study default \
  --study metric_raw --study metric_smoothed \
  --study unc_bkg0 --study unc_bkg4 --study unc_bkg6 --study unc_bkg10 --study unc_bkg20 \
  --study oscpoint_solar --study oscpoint_reactor \
  --study energy_maink --study energy_spk \
  --study charge_Q50 --study charge_Q100 --study charge_Q500 \
  --study fiduc_truth --study bkg_gamma \
  --study fiduc --study bkgmodel

# HEP (all 4 configs, all available studies)
sync_solar_data.sh \
  --analysis hep \
  --config hd_1x2x6_centralAPA \
  --config hd_1x2x6_lateralAPA \
  --config vd_1x8x14_3view_30deg_nominal \
  --config vd_1x8x14_3view_30deg_shielded \
  --name marley \
  --folder truncated \
  --study default \
  --study metric_raw --study metric_smoothed \
  --study unc_bkg0 --study unc_bkg4 --study unc_bkg6 --study unc_bkg10 --study unc_bkg20 \
  --study unc_sig20 --study unc_sig40 \
  --study oscpoint_solar --study oscpoint_reactor \
  --study charge_Q50 --study charge_Q100 --study charge_Q500 \
  --study fiduc_truth --study bkg_gamma

# Sensitivity (all 4 configs, all available studies)
sync_solar_data.sh \
  --analysis sensitivity \
  --config hd_1x2x6_centralAPA \
  --config hd_1x2x6_lateralAPA \
  --config vd_1x8x14_3view_30deg_nominal \
  --config vd_1x8x14_3view_30deg_shielded \
  --name marley \
  --folder truncated \
  --study default \
  --study metric_raw --study metric_smoothed \
  --study unc_bkg0 --study unc_bkg4 --study unc_bkg6 --study unc_bkg10 --study unc_bkg20 \
  --study unc_sig0 --study unc_sig2 --study unc_sig6 --study unc_sig8 \
  --study oscpoint_solar --study oscpoint_reactor \
  --study energy_spk \
  --study charge_Q50 --study charge_Q100 --study charge_Q500 \
  --study fiduc_truth --study bkg_gamma

# NOTE: sync AFTER the reruns land, not before — otherwise you pull the invalid grids
# described at the top of this file into input/data/studies/ and plot them.
# Still genuinely absent on PNFS (nothing to sync until the pipeline runs):
#   metric_raw / metric_smoothed — LAPA, VDN, VDS (all three analyses)
#   Sensitivity energy_maink     — all 4 configs
#   Sensitivity energy_spk       — LAPA, VDN, VDS
#   Sensitivity charge_Q100      — LAPA, VDN, VDS
#   Sensitivity bkg_gamma        — LAPA, VDN, VDS
# Sensitivity oscpoint_reactor is absent BY DESIGN (Score invariant to dm2) — do not chase it.
```

## Common Variables

```bash
CAPA=hd_1x2x6_centralAPA
LAPA=hd_1x2x6_lateralAPA
VDN=vd_1x8x14_3view_30deg_nominal
VDS=vd_1x8x14_3view_30deg_shielded
ALL="$CAPA $LAPA $VDN $VDS"
HD="$CAPA $LAPA"
VD="$VDN $VDS"
NAME=marley
```

---

## §9.1.1 — Choice of Histogram Processing and Metric

*Figures: `fig:study_metric_dn`, `fig:study_metric_hep`, `fig:study_metric_sens`*
*[OK — all 4 configs for DN; CAPA+VDS confirmed for HEP/Sens]*

```bash
# fig:study_metric_dn
python3 scripts/script_iterable_scan.py \
  --datafile DayNight_Exposure \
  --configs $ALL --name $NAME \
  -i Study -y Significance -x Exposure \
  --select Study -s default metric_raw metric_smoothed \
  --labelx 'Exposure (kt$\cdot$yr)' --labely 'Significance ($\sigma$)' \
  --rangex 0 20 --rangey 0 6 --horizontal 3 \
  --output fig_study_metric_dn

# fig:study_metric_hep
python3 scripts/script_iterable_scan.py \
  --datafile HEP_Exposure \
  --configs $ALL --name $NAME \
  -i Study -y Significance -x Exposure \
  --select Study -s default metric_raw metric_smoothed \
  --labelx 'Exposure (kt$\cdot$yr)' --labely 'Significance ($\sigma$)' \
  --rangex 0 20 --rangey 0 12 --horizontal 3 \
  --output fig_study_metric_hep

# fig:study_metric_sens  (contour: CAPA only — 4-config overlay too cluttered)
python3 scripts/script_compare_contour.py \
  --datafile Sensitivity_Significance \
  --configs $CAPA --name $NAME \
  --select Label Variable Study -s solar sin12 default \
  -y Dm2 -x Values -z Significance \
  --labelx '$\sin^2\theta_{12}$' --labely '$\Delta m^2_{21}$ (eV$^2$)' \
  --background_smoothing_sigma 3 --contour_linestyles dotted dashed solid \
  --point 0.303 6.0e-5 --point_label 'Solar' \
  --point 0.303 7.54e-5 --point_label 'Reactor' \
  --output fig_study_metric_sens
```

## §9.1.2 — Impact of Assumed Uncertainties

*Figures: `fig:study_uncertainties_dn`, `fig:study_uncertainties_hep`, `fig:study_uncertainties_sens`*
*Plot: significance at 20 kt·yr vs $\sigma^\mathrm{c}_\mathrm{rel}$ (point scan at fixed exposure)*
*[OK — all 4 configs for DN; CAPA+VDS confirmed for HEP/Sens]*

```bash
# fig:study_uncertainties_dn
python3 scripts/script_iterable_scan.py \
  --datafile DayNight_Exposure \
  --configs $ALL --name $NAME \
  -i Study -y Significance -x Exposure \
  --select Study -s default unc_bkg0 unc_bkg4 unc_bkg6 unc_bkg10 unc_bkg20 \
  --fixed_x Exposure 20.0 \
  --labelx '$\sigma^c_\mathrm{rel}$ (%)' \
  --labely 'Significance at 20 kt$\cdot$yr ($\sigma$)' \
  --horizontal 3 --output fig_study_uncertainties_dn

# fig:study_uncertainties_hep
python3 scripts/script_iterable_scan.py \
  --datafile HEP_Exposure \
  --configs $ALL --name $NAME \
  -i Study -y Significance -x Exposure \
  --select Study -s default unc_bkg0 unc_bkg4 unc_bkg6 unc_bkg10 unc_bkg20 \
  --fixed_x Exposure 20.0 \
  --labelx '$\sigma^c_\mathrm{rel}$ (%)' \
  --labely 'Significance at 20 kt$\cdot$yr ($\sigma$)' \
  --horizontal 3 --output fig_study_uncertainties_hep

# fig:study_uncertainties_sens  (contour: CAPA only)
python3 scripts/script_iterable_scan.py \
  --datafile Sensitivity_Significance \
  --configs $CAPA --name $NAME \
  -i Study -y Score -x Exposure \
  --select Study -s default unc_bkg0 unc_bkg4 unc_bkg6 unc_bkg10 unc_bkg20 \
  --fixed_x Exposure 20.0 \
  --labelx '$\sigma^c_\mathrm{rel}$ (%)' \
  --labely 'Sensitivity Score at 20 kt$\cdot$yr' \
  --output fig_study_uncertainties_sens
```

## §9.1.3 — Impact of Oscillation Parameter Choice

*Figures: `fig:study_oscillation_dn`, `fig:study_oscillation_hep`, `fig:study_oscillation_sens`*
*[OK — all 4 configs]*

```bash
# fig:study_oscillation_dn
python3 scripts/script_iterable_scan.py \
  --datafile DayNight_Exposure \
  --configs $ALL --name $NAME \
  -i Study -y Significance -x Exposure \
  --select Study -s default oscpoint_solar oscpoint_reactor \
  --labelx 'Exposure (kt$\cdot$yr)' --labely 'Significance ($\sigma$)' \
  --rangex 0 20 --rangey 0 6 --horizontal 3 \
  --output fig_study_oscillation_dn

# fig:study_oscillation_hep
python3 scripts/script_iterable_scan.py \
  --datafile HEP_Exposure \
  --configs $ALL --name $NAME \
  -i Study -y Significance -x Exposure \
  --select Study -s default oscpoint_solar oscpoint_reactor \
  --labelx 'Exposure (kt$\cdot$yr)' --labely 'Significance ($\sigma$)' \
  --rangex 0 20 --rangey 0 12 --horizontal 3 \
  --output fig_study_oscillation_hep

# fig:study_oscillation_sens  (contour: CAPA only)
python3 scripts/script_compare_contour.py \
  --datafile Sensitivity_Significance \
  --configs $CAPA --name $NAME \
  --select Label Variable Study -s solar sin12 default \
  -y Dm2 -x Values -z Significance \
  --labelx '$\sin^2\theta_{12}$' --labely '$\Delta m^2_{21}$ (eV$^2$)' \
  --background_smoothing_sigma 3 --contour_linestyles dotted dashed solid \
  --point 0.303 6.0e-5 --point_label 'Solar' \
  --point 0.303 7.54e-5 --point_label 'Reactor' \
  --overlay_datafile Sensitivity_Significance_oscpoint_solar \
  --overlay_datafile Sensitivity_Significance_oscpoint_reactor \
  --output fig_study_oscillation_sens
```

## §9.2.1 — Impact of Energy Resolution

*Figures: `fig:study_energy_resolution_dn`, `fig:study_energy_resolution_hep`, `fig:study_energy_resolution_sens`*
*Status: DN + HEP present all 4 configs · Sensitivity CAPA only (spk), maink absent*
*Validity: `RERUN-B` — ran with re-optimised cuts. Rerun `--study energy` before plotting.*

```bash
# fig:study_energy_resolution_dn  [present all 4 — RERUN-B]
python3 scripts/script_iterable_scan.py \
  --datafile DayNight_Exposure \
  --configs $ALL --name $NAME \
  -i Study -y Significance -x Exposure \
  --select Study -s default energy_maink energy_spk \
  --labelx 'Exposure (kt$\cdot$yr)' --labely 'Significance ($\sigma$)' \
  --rangex 0 20 --rangey 0 6 --horizontal 3 \
  --output fig_study_energy_resolution_dn

# fig:study_energy_resolution_hep  [present all 4 — RERUN-B]
python3 scripts/script_iterable_scan.py \
  --datafile HEP_Exposure \
  --configs $ALL --name $NAME \
  -i Study -y Significance -x Exposure \
  --select Study -s default energy_maink energy_spk \
  --labelx 'Exposure (kt$\cdot$yr)' --labely 'Significance ($\sigma$)' \
  --rangex 0 20 --rangey 0 12 --horizontal 3 \
  --output fig_study_energy_resolution_hep

# fig:study_energy_resolution_sens  [BLOCKED — Sensitivity energy_maink absent all configs,
#   energy_spk CAPA only. Needs 04_best_cuts.py for the energy group.]
# python3 scripts/script_compare_contour.py \
#   --datafile Sensitivity_Significance \
#   --configs $CAPA --name $NAME \
#   --select Label Variable Study -s solar sin12 default \
#   -y Dm2 -x Values -z Significance \
#   --labelx '$\sin^2\theta_{12}$' --labely '$\Delta m^2_{21}$ (eV$^2$)' \
#   --background_smoothing_sigma 3 --contour_linestyles dotted dashed solid \
#   --overlay_datafile Sensitivity_Significance_energy_maink \
#   --overlay_datafile Sensitivity_Significance_energy_spk \
#   --output fig_study_energy_resolution_sens
```

## §9.2.2–9.2.3 — Energy Reconstruction / Photon Detection (truth fiducialisation)

*Figures: `fig:study_fiducialisation_dn`, `fig:study_fiducialisation_hep`, `fig:study_fiducialisation_sens`*
*Status: DN + HEP + Sensitivity present all 4 configs (completed 2026-09-04)*
*Validity: `RERUN-B` — ran with re-optimised cuts. Rerun `--study fiduc_truth` before plotting.*

> **Verify VDN first.** VDN `fiduc_truth` reports DN 0.500 / EG 0.357 / HEP 2.886 — identical
> to VDN default on all three metrics. VDS HEP (3.129) also matches its default. Confirm these
> are real before they reach a figure. The gamma/neutron *"zero MCCounts — refusing to
> overwrite"* messages in `fiduc_truth.log` are the guard working as designed (backgrounds stay
> in nominal coordinates), not the cause.

```bash
# fig:study_fiducialisation_dn  [present all 4 — RERUN-B]
python3 scripts/script_iterable_scan.py \
  --datafile DayNight_Exposure \
  --configs $ALL --name $NAME \
  -i Study -y Significance -x Exposure \
  --select Study -s default fiduc_truth \
  --labelx 'Exposure (kt$\cdot$yr)' --labely 'Significance ($\sigma$)' \
  --rangex 0 20 --rangey 0 6 --horizontal 3 \
  --output fig_study_fiducialisation_dn

# fig:study_fiducialisation_hep  [present all 4 — RERUN-B]
python3 scripts/script_iterable_scan.py \
  --datafile HEP_Exposure \
  --configs $ALL --name $NAME \
  -i Study -y Significance -x Exposure \
  --select Study -s default fiduc_truth \
  --labelx 'Exposure (kt$\cdot$yr)' --labely 'Significance ($\sigma$)' \
  --rangex 0 20 --rangey 0 12 --horizontal 3 \
  --output fig_study_fiducialisation_hep

# fig:study_fiducialisation_sens  (contour: CAPA only)
python3 scripts/script_compare_contour.py \
  --datafile Sensitivity_Significance \
  --configs $CAPA --name $NAME \
  --select Label Variable Study -s solar sin12 default \
  -y Dm2 -x Values -z Significance \
  --labelx '$\sin^2\theta_{12}$' --labely '$\Delta m^2_{21}$ (eV$^2$)' \
  --background_smoothing_sigma 3 --contour_linestyles dotted dashed solid \
  --overlay_datafile Sensitivity_Significance_fiduc_truth \
  --output fig_study_fiducialisation_sens
```

## §9.2.4 — Impact of Charge Threshold

*Figures: `fig:study_charge_threshold_dn`, `fig:study_charge_threshold_hep`, `fig:study_charge_threshold_sens`*
*Plot: significance at 20 kt·yr vs $N^\mathrm{min}_\mathrm{hits}$ threshold*
*[OK — all 4 configs]*

```bash
# fig:study_charge_threshold_dn  [OK — all 4 configs]
python3 scripts/script_iterable_scan.py \
  --datafile DayNight_Exposure \
  --configs $ALL --name $NAME \
  -i Study -y Significance -x Exposure \
  --select Study -s default charge_Q50 charge_Q100 charge_Q500 \
  --fixed_x Exposure 20.0 \
  --labelx '$N^\mathrm{min}_\mathrm{hits}$' \
  --labely 'Significance at 20 kt$\cdot$yr ($\sigma$)' \
  --horizontal 3 --output fig_study_charge_threshold_dn

# fig:study_charge_threshold_hep  [OK — all 4 configs]
python3 scripts/script_iterable_scan.py \
  --datafile HEP_Exposure \
  --configs $ALL --name $NAME \
  -i Study -y Significance -x Exposure \
  --select Study -s default charge_Q50 charge_Q100 charge_Q500 \
  --fixed_x Exposure 20.0 \
  --labelx '$N^\mathrm{min}_\mathrm{hits}$' \
  --labely 'Significance at 20 kt$\cdot$yr ($\sigma$)' \
  --horizontal 3 --output fig_study_charge_threshold_hep

# fig:study_charge_threshold_sens  (contour: CAPA only)
python3 scripts/script_iterable_scan.py \
  --datafile Sensitivity_Significance \
  --configs $CAPA --name $NAME \
  -i Study -y Score -x Exposure \
  --select Study -s default charge_Q50 charge_Q100 charge_Q500 \
  --fixed_x Exposure 20.0 \
  --labelx '$N^\mathrm{min}_\mathrm{hits}$' \
  --labely 'Sensitivity Score at 20 kt$\cdot$yr' \
  --output fig_study_charge_threshold_sens
```

## §9.2.5 — Impact of Background Model (gamma suppression)

*Figures: `fig:study_improved_bkg_dn`, `fig:study_improved_bkg_hep`, `fig:study_improved_bkg_sens`*
*Status: DN + HEP present all 4 configs · Sensitivity CAPA only*
*Validity: `RERUN-B` — ran with re-optimised cuts. Rerun `--study bkg_gamma` before plotting.*
*Uses `ClusterEnergy`, not `SolarEnergy` — datafile stems differ.*

```bash
# fig:study_improved_bkg_dn  [present all 4 — RERUN-B]
python3 scripts/script_iterable_scan.py \
  --datafile DayNight_Exposure \
  --configs $ALL --name $NAME \
  -i Study -y Significance -x Exposure \
  --select Study -s default bkg_gamma \
  --labelx 'Exposure (kt$\cdot$yr)' --labely 'Significance ($\sigma$)' \
  --rangex 0 20 --rangey 0 6 --horizontal 3 \
  --output fig_study_improved_bkg_dn

# fig:study_improved_bkg_hep  [present all 4 — RERUN-B]
python3 scripts/script_iterable_scan.py \
  --datafile HEP_Exposure \
  --configs $ALL --name $NAME \
  -i Study -y Significance -x Exposure \
  --select Study -s default bkg_gamma \
  --labelx 'Exposure (kt$\cdot$yr)' --labely 'Significance ($\sigma$)' \
  --rangex 0 20 --rangey 0 12 --horizontal 3 \
  --output fig_study_improved_bkg_hep

# fig:study_improved_bkg_sens  (contour: CAPA only — LAPA/VDN/VDS Sensitivity absent)
# The Sensitivity stage errored on 2026-09-04: "No background templates found in
# .../SENSITIVITY/{cfg}/background/truncated/ClusterEnergy". Build those templates first.
python3 scripts/script_compare_contour.py \
  --datafile Sensitivity_Significance \
  --configs $CAPA --name $NAME \
  --select Label Variable Study -s solar sin12 default \
  -y Dm2 -x Values -z Significance \
  --labelx '$\sin^2\theta_{12}$' --labely '$\Delta m^2_{21}$ (eV$^2$)' \
  --background_smoothing_sigma 3 --contour_linestyles dotted dashed solid \
  --overlay_datafile Sensitivity_Significance_bkg_gamma \
  --output fig_study_improved_bkg_sens
```

## §9.3.2 — Summary Bar Chart

*Figure: `fig:sensitivity_studies_summary`*
*Baseline (HD Central 20 kt·yr): DN=3.85σ, HEP=10.63σ, Sens score=1.31*
*Summary bar chart: CAPA only (primary detector for comparison)*
*[BLOCKED — this figure aggregates every variant, so it inherits every `RERUN-A`/`RERUN-B`
mark in the matrix. Build it last, after all reruns land. Baselines below predate the
2026-09-02 default rerun and need re-deriving too.]*

```bash
# python3 scripts/script_compare_pareto.py \
#   --datafile DayNight_Exposure HEP_Exposure Sensitivity_Significance \
#   --configs $CAPA --name $NAME \
#   --select Study -s default \
#              metric_raw metric_smoothed \
#              unc_bkg0 unc_bkg4 unc_bkg6 \
#              oscpoint_solar oscpoint_reactor \
#              energy_spk energy_maink \
#              charge_Q50 charge_Q100 charge_Q500 \
#              fiduc_truth bkg_gamma \
#   --fixed_x Exposure 20.0 \
#   --baseline_dn 3.85 --baseline_hep 10.63 --baseline_sens 1.31 \
#   --output fig_sensitivity_studies_summary
```
