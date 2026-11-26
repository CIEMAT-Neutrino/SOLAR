#!/bin/bash
# Exports of the Phase I solar paper and its technical note (frozen in solar_neutrinos_phaseI_paper_plots/input/data),
# from the production Ref pkls of output/data/results (reruns 40/41), into OUT (never the production tree):
#   OUT/cutflow   cutflow_plot.py --first_stage vetoes  (*_SolarEnergy_<Analysis>_Cutflow.pkl, the CutflowVetoes files)
#   OUT/weighted  04_weighted.py at the selected cuts    (*_Weighted_Distributions_Fiducial_<Analysis>.pkl), checked
#                 against the cut flows of OUT/cutflow
#   OUT/hep_veto_check.log  hep_veto_check.py (what the classifier vetoes remove from the hep selection)
#   OUT/analysis      01_daynight.py / 01_hep.py, production settings: Results and SignificanceBins with the per-bin
#                     significances at the drawn exposure (EVALUATION_EXPOSURE_YEARS) and, for VD, the full module
#                     (MODULE_VOLUMES of the config). With --display_exposure 30 --display_volumes 1 they reproduce the
#                     production SignificanceBins of reruns 40/41 exactly (2026-11-26 check)
#   OUT/significance  significance_plot.py from OUT/analysis: DayNight_Counts, HEP_Counts, Sensitivity_Counts (spectra
#                     of the paper's Figs. 7 and 9 and of the technote; the Sensitivity rows carry sqrt(q_i))
#   OUT/templates     02_signal_template.py --scan_mode at the selected HD-central oscillation cut, the production
#                     solar and reactor points (equal to the production templates) and the JUNO best fit
#   OUT/hypotheses/{juno,analysis}  generate_sensitivity_hypotheses.py: solar and reactor fit hypotheses of HD central,
#                     reactor at the JUNO best fit (paper Fig. 11) and at the analysis REACT_DM2 (technote)
# Called by src/pipelines/final_run.sh after the analysis (staged with the production).
# Usage: tools/paper_phaseI_exports.sh OUT   (run from the repository root; about 20 minutes)
set -u
OUT=$(realpath -m "${1:?usage: tools/paper_phaseI_exports.sh OUT}"); M=/pc/choozdsk01/users/manthey; B=$(pwd)
IMG="apptainer exec -B /pnfs,/afs,/pc,/cvmfs --home=$B/ --pwd $B/ --env SOLAR_NO_PNG=1 $M/SOLAR/containers/solar_v1.0.sif"
CONFIGS="hd_1x2x6_centralAPA hd_1x2x6_lateralAPA vd_1x8x14_3view_30deg_nominal"
SAMPLES="marley_final gamma_final neutron_final radiological_final"
# Selected NHits:OpHits:AdjCl of each analysis and config (paper Table I)
declare -A CUTS=(
  [DayNight:hd_1x2x6_centralAPA]=1:5:7 [DayNight:hd_1x2x6_lateralAPA]=2:4:5 [DayNight:vd_1x8x14_3view_30deg_nominal]=5:16:20
  [HEP:hd_1x2x6_centralAPA]=4:4:4      [HEP:hd_1x2x6_lateralAPA]=2:20:8     [HEP:vd_1x8x14_3view_30deg_nominal]=1:4:3
  [Sensitivity:hd_1x2x6_centralAPA]=1:9:4 [Sensitivity:hd_1x2x6_lateralAPA]=2:4:5 [Sensitivity:vd_1x8x14_3view_30deg_nominal]=5:15:17
)
mkdir -p $OUT/logs; echo "$(date "+%F %T") start, commit $(git rev-parse --short HEAD)$(git diff --quiet || echo +)"
for c in $CONFIGS; do for a in DayNight HEP Sensitivity; do
  $IMG python3 src/physics/signal/cutflow_plot.py --config $c --signal $SAMPLES --folder Truncated --energy SolarEnergy --analysis $a \
    --first_stage vetoes --output_dir $OUT/cutflow --rewrite --no-debug --no-plot > $OUT/logs/cutflow_${c}_$a.log 2>&1 < /dev/null &
done; done; wait
for a in DayNight HEP Sensitivity; do for c in $CONFIGS; do IFS=: read nh op ad <<< "${CUTS[$a:$c]}"; for n in $SAMPLES; do
  $IMG python3 src/physics/signal/04_weighted.py --config $c --signal $n --analysis $a --folder Truncated --nhits $nh --ophits $op --adjcls $ad \
    --summary_oscillation Mean --oscillation_backend nufast --output_dir $OUT/weighted --cutflow_dir $OUT/cutflow --check_cutflow \
    --rewrite --no-debug > $OUT/logs/weighted_${a}_${c}_$n.log 2>&1 < /dev/null &
done; done; wait; done
# Spectra with the per-bin significances at the drawn exposure (DayNight, HEP: production --mc_threshold of each config)
declare -A DN_MC=([hd_1x2x6_centralAPA]=5 [hd_1x2x6_lateralAPA]=10 [vd_1x8x14_3view_30deg_nominal]=20)
declare -A HEP_MC=([hd_1x2x6_centralAPA]=5 [hd_1x2x6_lateralAPA]=10 [vd_1x8x14_3view_30deg_nominal]=10)
COMMON="--folder Truncated --rewrite --no-debug --no-plot --signal marley_final --energy SolarEnergy"
for c in $CONFIGS; do
  $IMG python3 src/physics/daynight/01_daynight.py --config $c $COMMON --signal_uncertainty 0.0 --background_uncertainty 0.02 \
    --mc_threshold ${DN_MC[$c]} --earth_density_band 0.13 --oscillation_band 0.05 --day_fraction 0.493 --day_fraction_band 0.01 \
    --test_statistic asimov --output_dir $OUT/analysis/DAYNIGHT/truncated > $OUT/logs/01_daynight_$c.log 2>&1 < /dev/null &
  $IMG python3 src/physics/hep/01_hep.py --config $c $COMMON --signal_uncertainty 0.3 --background_uncertainty 0.02 \
    --mc_threshold ${HEP_MC[$c]} --output_dir $OUT/analysis/HEP/truncated > $OUT/logs/01_hep_$c.log 2>&1 < /dev/null &
done; wait
SIG="$IMG python3 src/physics/common/significance_plot.py $COMMON --plot --no-stacked --analysis_input_dir $OUT/analysis --output_dir $OUT/significance"
for c in $CONFIGS; do
  $SIG --config $c --signal_uncertainty 0.0 --background_uncertainty 0.02 --analysis DayNight --day_fraction 0.493 \
    > $OUT/logs/significance_DayNight_$c.log 2>&1 < /dev/null &
  $SIG --config $c --signal_uncertainty 0.3 --background_uncertainty 0.02 --analysis HEP --reference ProfileLikelihood \
    --bottom-panel-mode both > $OUT/logs/significance_HEP_$c.log 2>&1 < /dev/null &
  $SIG --config $c --signal_uncertainty 0.04 --background_uncertainty 0.02 --analysis Sensitivity \
    > $OUT/logs/significance_Sensitivity_$c.log 2>&1 < /dev/null &
done; wait
# Fit hypotheses of HD central (paper Fig. 11: reactor at the JUNO best fit; technote: at the analysis reactor point)
c=hd_1x2x6_centralAPA; IFS=: read nh op ad <<< "${CUTS[Sensitivity:$c]}"
$IMG python3 src/physics/sensitivity/02_signal_template.py --config $c --signal marley_final --folder Truncated --energy SolarEnergy \
  --scan_mode --nhits $nh --adjcls $ad --ophits $op --oscillation_backend nufast --extra_points 7.388e-05:0.3036 \
  --output_dir $OUT/templates --rewrite --no-debug --no-plot > $OUT/logs/templates_$c.log 2>&1 < /dev/null
T=$OUT/templates/$c/marley_final/truncated/SolarEnergy
for h in juno:7.388e-05_sin13_2.200e-02_sin12_3.036e-01 analysis:7.540e-05_sin13_2.200e-02_sin12_3.040e-01; do
  $IMG python3 src/physics/sensitivity/generate_sensitivity_hypotheses.py --config $c --name marley_final \
    --input_dir output/data/analysis/sensitivity/$c/marley_final/truncated/default --output_dir $OUT/hypotheses/${h%%:*} \
    --reactor $T/${c}_marley_final_NHits${nh}_AdjCl${ad}_OpHits${op}_dm2_${h#*:}.pkl > $OUT/logs/hypotheses_${h%%:*}.log 2>&1 < /dev/null
done
$IMG python3 src/physics/signal/hep_veto_check.py --config $CONFIGS > $OUT/hep_veto_check.log 2>&1
grep -l Traceback $OUT/logs/*.log | sed "s/^/FAILED: /"
grep -h "per-bin significances at" $OUT/logs/significance_*.log | sed "s/^/EXPOSURE MISMATCH: /"
for f in $OUT/logs/weighted_*.log; do tr -d " \n" < $f | grep -q "checked,0mismatched" || echo "CUT-FLOW CHECK NOT PASSED: $f"; done
echo "$(date "+%F %T") done"
