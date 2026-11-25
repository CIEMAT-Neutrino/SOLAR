#!/bin/bash
# Exports of the Phase I solar paper and its technical note (frozen in solar_neutrinos_phaseI_paper_plots/input/data),
# from the production Ref pkls of output/data/results (reruns 40/41), into OUT (never the production tree):
#   OUT/cutflow   cutflow_plot.py --first_stage vetoes  (*_SolarEnergy_<Analysis>_Cutflow.pkl, the CutflowVetoes files)
#   OUT/weighted  04_weighted.py at the selected cuts    (*_Weighted_Distributions_Fiducial_<Analysis>.pkl), checked
#                 against the cut flows of OUT/cutflow
#   OUT/hep_veto_check.log  hep_veto_check.py (what the classifier vetoes remove from the hep selection)
# Usage: tools/paper_phaseI_exports.sh OUT   (run from the repository root; about 15 minutes)
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
$IMG python3 src/physics/signal/hep_veto_check.py --config $CONFIGS > $OUT/hep_veto_check.log 2>&1
grep -l Traceback $OUT/logs/*.log | sed "s/^/FAILED: /"
for f in $OUT/logs/weighted_*.log; do tr -d "\n" < $f | tr -s " " | grep -q "checked, \?0 mismatched" || echo "CUT-FLOW CHECK NOT PASSED: $f"; done
echo "$(date "+%F %T") done"
