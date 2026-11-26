#!/bin/bash
# Cross-check of the hep working point of the S/B classifier (technical note, selection details), on the final production,
# read-only, into OUT:
#   OUT/tune_<config>.log      src/tools/hep_working_point_tune.py: gate energy and completeness scan, tuned on one half
#                              of the simulated events and checked on the other (2026-10-08 study, rerun 2026-11-25)
#   OUT/{analysis,none}/       cutflow_plot.py --analysis HEP --hep_working_point analysis|none (same cuts and volume)
#   OUT/significance.log       src/tools/hep_working_point_significance.py: 20-yr hep significance of both cut flows
# Usage: tools/hep_working_point_check.sh OUT   (run from the repository root)
set -u
OUT=$(realpath -m "${1:?usage: tools/hep_working_point_check.sh OUT}"); M=/pc/choozdsk01/users/manthey; B=$(pwd)
IMG="apptainer exec -B /pnfs,/afs,/pc,/cvmfs --home=$B/ --pwd $B/ --env SOLAR_NO_PNG=1 $M/SOLAR/containers/solar_v1.0.sif"
CONFIGS="hd_1x2x6_centralAPA hd_1x2x6_lateralAPA vd_1x8x14_3view_30deg_nominal"
mkdir -p $OUT; echo "$(date "+%F %T") start, commit $(git rev-parse --short HEAD)$(git diff --quiet || echo +)"
for c in $CONFIGS; do
  ref=output/data/analysis/hep/$c/marley_final/truncated/default/${c}_marley_final_HEP_Counts.pkl
  $IMG python3 src/tools/hep_working_point_tune.py --config $c --ref_counts $ref > $OUT/tune_$c.log 2>&1 < /dev/null &
  for v in analysis none; do
    $IMG python3 src/physics/signal/cutflow_plot.py --config $c --signal marley_final gamma_final neutron_final radiological_final \
      --folder Truncated --energy SolarEnergy --analysis HEP --hep_working_point $v --output_dir $OUT/$v \
      --rewrite --no-debug --no-plot --no-check_counts > $OUT/cutflow_${v}_$c.log 2>&1 < /dev/null &
  done
done; wait
$IMG python3 src/tools/hep_working_point_significance.py $OUT $CONFIGS > $OUT/significance.log 2>&1
grep -l Traceback $OUT/*.log | sed "s/^/FAILED: /"
echo "$(date "+%F %T") done"
