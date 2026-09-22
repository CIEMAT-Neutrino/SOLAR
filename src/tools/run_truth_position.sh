#!/usr/bin/env bash
# Extract the per-event caches for src/physics/signal/truth_position_study.py, then draw the figures.
# One process per (config, sample) keeps memory bounded (~3 min each). Existing caches are skipped.
#
#   src/tools/run_truth_position.sh                    # all configs, extract + plot
#   src/tools/run_truth_position.sh plot               # plot only
#   src/tools/run_truth_position.sh extract vd_1x8x14_3view_30deg_shielded
set -euo pipefail

SOLAR=/pc/choozdsk01/users/manthey/SOLAR
IMG=(apptainer exec -B /pnfs,/afs,/pc,/cvmfs --home=$SOLAR/ --pwd $SOLAR/ $SOLAR/containers/solar_v1.0.sif)
SCRIPT=$SOLAR/src/physics/signal/truth_position_study.py

STAGE=${1:-all}
shift || true
CONFIGS=("$@")
[ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=(hd_1x2x6_centralAPA hd_1x2x6_lateralAPA vd_1x8x14_3view_30deg_nominal vd_1x8x14_3view_30deg_shielded)

if [ "$STAGE" = all ] || [ "$STAGE" = extract ]; then
  for c in "${CONFIGS[@]}"; do
    for n in marley gamma neutron radiological; do
      "${IMG[@]}" python3 "$SCRIPT" --stage extract --config "$c" --signals "$n"
    done
  done
fi
if [ "$STAGE" = all ] || [ "$STAGE" = plot ]; then
  "${IMG[@]}" python3 "$SCRIPT" --stage plot --config "${CONFIGS[@]}"
fi
