# Thesis Chapter 9 — Plot Commands

*Generated: 2026-08-26. Verified against actual `sync_solar_data.sh` interface.*

`sync_solar_data.sh` default remote: `gae_out:/pc/choozdsk01/users/manthey/SOLAR`
Study-variant pkls auto-route to `input/data/studies/` by the sync script.
Plot scripts fall back `input/data/` → `input/data/studies/` transparently.
`--study`/`--config`/`--name`/`--folder` are repeatable (one value per flag).

Datafile stem pattern (`--configs` + `--name` supplied separately):
- `--datafile {Analysis}_{Type}` → `input/data/{config}_{name}_{Analysis}_{Type}.pkl`
- `--datafile {Analysis}_{Type}_{study}` → `input/data/studies/{config}_{name}_{...}.pkl`

**Status legend:** `[OK]` = pkl verified on filesystem 2026-08-26 · `[MISSING]` = pipeline not yet run

## Config Shortnames

| Shortname | Full config key |
|---|---|
| `CAPA` | `hd_1x2x6_centralAPA` |
| `LAPA` | `hd_1x2x6_lateralAPA` |
| `VDN` | `vd_1x8x14_3view_30deg_nominal` |
| `VDS` | `vd_1x8x14_3view_30deg_shielded` |

## Availability Matrix (all configs / truncated / marley)

| Study variant | DN-CAPA | DN-LAPA | DN-VDN | DN-VDS | HEP-CAPA | HEP-VDS | SENS-CAPA | SENS-VDS |
|---|---|---|---|---|---|---|---|---|
| default | OK | OK | OK | OK | OK | OK | OK | OK |
| metric_raw/smoothed | OK | OK | OK | OK | OK | ? | OK | ? |
| unc_bkg0/4/6/10/20 | OK | OK | OK | OK | OK | OK | OK | OK |
| oscpoint_solar/rct | OK | OK | OK | OK | OK | OK | OK | OK |
| energy_maink | OK | OK | OK | OK | MISSING | MISSING | MISSING | MISSING |
| energy_spk | OK | OK | OK | OK | MISSING | MISSING | OK | OK |
| charge_Q50/100/200 | OK | OK | OK | OK | OK | OK | OK | OK |
| fiduc_truth | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |
| bkg_gamma | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |

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
  --study charge_Q50 --study charge_Q100 --study charge_Q200

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
  --study charge_Q50 --study charge_Q100 --study charge_Q200

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
  --study charge_Q50 --study charge_Q100 --study charge_Q200

# After missing pipelines complete, add:
# sync_solar_data.sh --analysis hep --analysis sensitivity \
#   --config hd_1x2x6_centralAPA --config hd_1x2x6_lateralAPA \
#   --config vd_1x8x14_3view_30deg_nominal --config vd_1x8x14_3view_30deg_shielded \
#   --name marley --folder truncated --study energy_maink
# sync_solar_data.sh --analysis hep \
#   --config hd_1x2x6_centralAPA --config hd_1x2x6_lateralAPA \
#   --config vd_1x8x14_3view_30deg_nominal --config vd_1x8x14_3view_30deg_shielded \
#   --name marley --folder truncated --study energy_spk
# sync_solar_data.sh \
#   --config hd_1x2x6_centralAPA --config hd_1x2x6_lateralAPA \
#   --config vd_1x8x14_3view_30deg_nominal --config vd_1x8x14_3view_30deg_shielded \
#   --name marley --folder truncated --study fiduc_truth --study bkg_gamma
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
*Status: DN OK (all 4 configs) | HEP MISSING | Sensitivity MISSING*

```bash
# fig:study_energy_resolution_dn  [OK — all 4 configs]
python3 scripts/script_iterable_scan.py \
  --datafile DayNight_Exposure \
  --configs $ALL --name $NAME \
  -i Study -y Significance -x Exposure \
  --select Study -s default energy_maink energy_spk \
  --labelx 'Exposure (kt$\cdot$yr)' --labely 'Significance ($\sigma$)' \
  --rangex 0 20 --rangey 0 6 --horizontal 3 \
  --output fig_study_energy_resolution_dn

# fig:study_energy_resolution_hep  [MISSING — run energy --analysis HEP]
# python3 scripts/script_iterable_scan.py \
#   --datafile HEP_Exposure \
#   --configs $ALL --name $NAME \
#   -i Study -y Significance -x Exposure \
#   --select Study -s default energy_maink energy_spk \
#   --labelx 'Exposure (kt$\cdot$yr)' --labely 'Significance ($\sigma$)' \
#   --rangex 0 20 --rangey 0 12 --horizontal 3 \
#   --output fig_study_energy_resolution_hep

# fig:study_energy_resolution_sens  [MISSING — run energy --analysis Sensitivity]
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
*Status: ALL MISSING — `fiduc_truth` pipeline not yet complete*

To unblock: run `python3 src/pipelines/run_studies.py --study fiduc_truth --folder Truncated`, then sync.

```bash
# fig:study_fiducialisation_dn  [MISSING]
# python3 scripts/script_iterable_scan.py \
#   --datafile DayNight_Exposure \
#   --configs $ALL --name $NAME \
#   -i Study -y Significance -x Exposure \
#   --select Study -s default fiduc_truth \
#   --labelx 'Exposure (kt$\cdot$yr)' --labely 'Significance ($\sigma$)' \
#   --rangex 0 20 --rangey 0 6 --horizontal 3 \
#   --output fig_study_fiducialisation_dn

# fig:study_fiducialisation_hep  [MISSING — same pattern with HEP_Exposure]
# fig:study_fiducialisation_sens  [MISSING — use script_compare_contour.py, CAPA only]
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
  --select Study -s default charge_Q50 charge_Q100 charge_Q200 \
  --fixed_x Exposure 20.0 \
  --labelx '$N^\mathrm{min}_\mathrm{hits}$' \
  --labely 'Significance at 20 kt$\cdot$yr ($\sigma$)' \
  --horizontal 3 --output fig_study_charge_threshold_dn

# fig:study_charge_threshold_hep  [OK — all 4 configs]
python3 scripts/script_iterable_scan.py \
  --datafile HEP_Exposure \
  --configs $ALL --name $NAME \
  -i Study -y Significance -x Exposure \
  --select Study -s default charge_Q50 charge_Q100 charge_Q200 \
  --fixed_x Exposure 20.0 \
  --labelx '$N^\mathrm{min}_\mathrm{hits}$' \
  --labely 'Significance at 20 kt$\cdot$yr ($\sigma$)' \
  --horizontal 3 --output fig_study_charge_threshold_hep

# fig:study_charge_threshold_sens  (contour: CAPA only)
python3 scripts/script_iterable_scan.py \
  --datafile Sensitivity_Significance \
  --configs $CAPA --name $NAME \
  -i Study -y Score -x Exposure \
  --select Study -s default charge_Q50 charge_Q100 charge_Q200 \
  --fixed_x Exposure 20.0 \
  --labelx '$N^\mathrm{min}_\mathrm{hits}$' \
  --labely 'Sensitivity Score at 20 kt$\cdot$yr' \
  --output fig_study_charge_threshold_sens
```

## §9.2.5 — Impact of Background Model (gamma suppression)

*Figures: `fig:study_improved_bkg_dn`, `fig:study_improved_bkg_hep`, `fig:study_improved_bkg_sens`*
*Status: ALL MISSING — `bkg_gamma` pipeline not yet complete*

To unblock: run `python3 src/pipelines/run_studies.py --study bkg_gamma --folder Truncated`, then sync.

```bash
# fig:study_improved_bkg_dn  [MISSING]
# python3 scripts/script_iterable_scan.py \
#   --datafile DayNight_Exposure \
#   --configs $ALL --name $NAME \
#   -i Study -y Significance -x Exposure \
#   --select Study -s default bkg_gamma \
#   --labelx 'Exposure (kt$\cdot$yr)' --labely 'Significance ($\sigma$)' \
#   --rangex 0 20 --rangey 0 6 --horizontal 3 \
#   --output fig_study_improved_bkg_dn

# fig:study_improved_bkg_hep  [MISSING — same pattern with HEP_Exposure]
# fig:study_improved_bkg_sens  [MISSING — script_compare_contour.py, CAPA only]
```

## §9.3.2 — Summary Bar Chart

*Figure: `fig:sensitivity_studies_summary`*
*Baseline (HD Central 20 kt·yr): DN=3.85σ, HEP=10.63σ, Sens score=1.31*
*Summary bar chart: CAPA only (primary detector for comparison)*
*[PARTIAL — available studies only]*

```bash
# python3 scripts/script_compare_pareto.py \
#   --datafile DayNight_Exposure HEP_Exposure Sensitivity_Significance \
#   --configs $CAPA --name $NAME \
#   --select Study -s default \
#              metric_raw metric_smoothed \
#              unc_bkg0 unc_bkg20 \
#              oscpoint_solar oscpoint_reactor \
#              energy_maink \
#              charge_Q50 charge_Q100 charge_Q200 \
#   --fixed_x Exposure 20.0 \
#   --baseline_dn 3.85 --baseline_hep 10.63 --baseline_sens 1.31 \
#   --output fig_sensitivity_studies_summary
```
