#!/bin/bash
# Day-night and hep spectra of the Phase I paper and its technical note (DayNight_Counts, HEP_Counts), with the per-bin
# significances of the lower panels evaluated at the drawn exposure (EVALUATION_EXPOSURE_YEARS, 20 yr) and, for VD, for
# the full module (two simulated drift volumes). Into OUT, never the production tree:
#   OUT/analysis      01_daynight.py / 01_hep.py, production settings of reruns 40/41, --display_exposure default,
#                     --display_volumes 2 for VD (Results and SignificanceBins)
#   OUT/significance  significance_plot.py from OUT/analysis (Counts and Significance tables)
# The 01 runs with --display_exposure 30 reproduce the production SignificanceBins exactly (2026-11-26 check).
# Usage: tools/paper_phaseI_significance_exports.sh OUT   (run from the repository root; about 5 minutes)
set -u
OUT=$(realpath -m "${1:?usage: tools/paper_phaseI_significance_exports.sh OUT}"); M=/pc/choozdsk01/users/manthey; B=$(pwd)
IMG="apptainer exec -B /pnfs,/afs,/pc,/cvmfs --home=$B/ --pwd $B/ --env SOLAR_NO_PNG=1 $M/SOLAR/containers/solar_v1.0.sif"
COMMON="--folder Truncated --rewrite --no-debug --no-plot --signal marley_final --energy SolarEnergy"
# config: DayNight and hep --mc_threshold of the production runs, simulated volumes in the paper's module
declare -A DN_MC=([hd_1x2x6_centralAPA]=5 [hd_1x2x6_lateralAPA]=10 [vd_1x8x14_3view_30deg_nominal]=20)
declare -A HEP_MC=([hd_1x2x6_centralAPA]=5 [hd_1x2x6_lateralAPA]=10 [vd_1x8x14_3view_30deg_nominal]=10)
declare -A VOLUMES=([hd_1x2x6_centralAPA]=1 [hd_1x2x6_lateralAPA]=1 [vd_1x8x14_3view_30deg_nominal]=2)
mkdir -p $OUT/logs; echo "$(date "+%F %T") start, commit $(git rev-parse --short HEAD)$(git diff --quiet || echo +)"
for c in "${!VOLUMES[@]}"; do
  $IMG python3 src/physics/daynight/01_daynight.py --config $c $COMMON --signal_uncertainty 0.0 --background_uncertainty 0.02 \
    --mc_threshold ${DN_MC[$c]} --earth_density_band 0.13 --oscillation_band 0.05 --day_fraction 0.493 --day_fraction_band 0.01 \
    --test_statistic asimov --display_volumes ${VOLUMES[$c]} --output_dir $OUT/analysis/DAYNIGHT/truncated \
    > $OUT/logs/01_daynight_$c.log 2>&1 < /dev/null &
  $IMG python3 src/physics/hep/01_hep.py --config $c $COMMON --signal_uncertainty 0.3 --background_uncertainty 0.02 \
    --mc_threshold ${HEP_MC[$c]} --display_volumes ${VOLUMES[$c]} --output_dir $OUT/analysis/HEP/truncated \
    > $OUT/logs/01_hep_$c.log 2>&1 < /dev/null &
done; wait
for c in "${!VOLUMES[@]}"; do
  $IMG python3 src/physics/common/significance_plot.py --config $c $COMMON --plot --signal_uncertainty 0.0 --background_uncertainty 0.02 \
    --no-stacked --analysis DayNight --day_fraction 0.493 --volumes ${VOLUMES[$c]} --analysis_input_dir $OUT/analysis \
    --output_dir $OUT/significance > $OUT/logs/significance_DayNight_$c.log 2>&1 < /dev/null &
  $IMG python3 src/physics/common/significance_plot.py --config $c $COMMON --plot --signal_uncertainty 0.3 --background_uncertainty 0.02 \
    --no-stacked --analysis HEP --reference ProfileLikelihood --bottom-panel-mode both --volumes ${VOLUMES[$c]} \
    --analysis_input_dir $OUT/analysis --output_dir $OUT/significance > $OUT/logs/significance_HEP_$c.log 2>&1 < /dev/null &
done; wait
grep -l Traceback $OUT/logs/*.log | sed "s/^/FAILED: /"
grep -h "per-bin significances at" $OUT/logs/significance_*.log | sed "s/^/EXPOSURE MISMATCH: /"
echo "$(date "+%F %T") done"
