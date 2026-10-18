#!/bin/bash
# Final-run driver: the production chain behind the thesis results (calibration steps, run_sensitivity defaults,
# the Pessimistic background-model study, study_status, weighted extras, staging). Versioned copy of
# SOLAR_transfer/final_runs/final_run2.sh (2026-10-18); the repository root is taken from this file's location.
#   final_run.sh -l LABEL [-f calib|sb|analysis] [-p PARALLEL_SAMPLES] [-c "CONFIGS"] [-x 0|1]
#   -c          : configs to run (default: all four)
#   -x 0        : HD Central S/B classifier with the matched-flash inputs (only for background samples with the light overlay,
#                 e.g. the combined PIC gamma_test/neutron_test; the old-chain gamma_final/neutron_final lack it, so the default excludes them)
#   -f calib    : full calibration (01 F0 table, 02, 03 offset RF, 03b S/B RF, 04, 05, 03c, resolution, chain)
#   -f sb       : from 03b (only the S/B classifier and the steps after it change)
#   -f analysis : calibration unchanged
# Analysis: run_sensitivity Truncated (default background model) and the bkgmodel_pessimistic study, then
# study_status, copies in _runs/LABEL, weighted extras and the staging for the plotting repository.
# Logs: SOLAR-sb/output/logs/LABEL_{calib,ana,realistic}; driver log next to this script (LABEL.log).
LABEL=""; FROM=analysis; PS=3; ONLY=""; XHDC=0  # 2026-10-09: HD Central on the new PIC samples (light overlay): S/B classifier with the matched-flash inputs
while getopts l:f:p:c:x: o; do case $o in l) LABEL=$OPTARG;; f) FROM=$OPTARG;; p) PS=$OPTARG;; c) ONLY=$OPTARG;; x) XHDC=$OPTARG;; esac; done
[ -n "$LABEL" ] || { echo "usage: $0 -l LABEL [-f calib|sb|analysis] [-p N]"; exit 1; }
D=$(cd $(dirname $0) && pwd)
B=${SOLAR_ROOT:-$(cd $D/../.. && pwd)}; S=${SOLAR_CONTAINER_ROOT:-/pc/choozdsk01/users/manthey/SOLAR}
STAGE_DIR=${SOLAR_STAGE_DIR:-/pc/choozdsk01/users/manthey/SOLAR_compare}
export SOLAR_NO_PNG=1
IMG="apptainer exec -B /pnfs,/afs,/pc,/cvmfs --home=$B/ --pwd $B/ $S/containers/solar_v1.0.sif"
log(){ echo "$(date "+%F %T") $*"; }
CONFIGS="hd_1x2x6_centralAPA hd_1x2x6_lateralAPA vd_1x8x14_3view_30deg_nominal vd_1x8x14_3view_30deg_shielded"
[ -n "$ONLY" ] && CONFIGS=$ONLY
cd $B
log "== $LABEL (commit $(git rev-parse --short HEAD), from $FROM, parallel_samples $PS, configs $CONFIGS, hdc flash exclusion $XHDC)"
if [ "$FROM" != analysis ]; then
  L=$B/output/logs/${LABEL}_calib; mkdir -p $L
  run(){ local tag=$1 cfg=$2; shift 2; $IMG python3 "$@" > $L/${tag}_$cfg.log 2>&1 && log "  $cfg $tag OK" || { log "  $cfg $tag FAIL"; return 1; }; }
  for cfg in $CONFIGS; do
    case $cfg in hd_1x2x6_centralAPA) [ "$XHDC" = 1 ] && SB="--exclude_features MatchedOpFlashPE MatchedOpFlashNHits" || SB="";; *) SB="";; esac
    ( set -e
      if [ "$FROM" = calib ]; then
        run 01 $cfg src/physics/calibration/01_correction.py --config $cfg --name marley_final --rewrite --no-debug
        run 02 $cfg src/physics/calibration/02_calibration.py --config $cfg --name marley_final --rewrite --no-debug
        run 03 $cfg src/physics/calibration/03_discrimination.py --config $cfg --name marley_final --rewrite --no-debug
      fi
      run 03b $cfg src/physics/calibration/03b_signal_background.py --config $cfg $SB --no-debug
      run 04 $cfg src/physics/calibration/04_reconstruction.py --config $cfg --name marley_final --rewrite --no-debug
      run 05 $cfg src/physics/calibration/05_smearing.py --config $cfg --name marley_final --rewrite --no-debug
      run 03c $cfg src/physics/calibration/03c_signal_background_features.py --config $cfg --no-debug
      run res $cfg src/physics/detector/tpc/03_energy_resolution.py --config $cfg --name marley_final --rewrite --no-debug
      for n in marley_final gamma_final; do run chain_$n $cfg src/physics/detector/tpc/04_smearing_chain.py --config $cfg --name $n --rewrite --no-debug; done ) &
  done; wait
  if grep -l "Traceback" $L/*.log; then log "CALIB FAILED"; exit 1; fi
  mkdir -p _runs/$LABEL/calib; for cfg in $CONFIGS; do cp -a config/$cfg/marley_final/${cfg}_calib _runs/$LABEL/calib/; done
  log "CALIB DONE"
fi
L=$B/output/logs/${LABEL}_ana; R=$B/output/logs/${LABEL}_pessimistic; mkdir -p $L $R
for cfg in $CONFIGS; do
  $IMG python3 src/pipelines/run_sensitivity.py --rewrite --config $cfg --folder Truncated --parallel_samples $PS > $L/$cfg.log 2>&1 &
  $IMG python3 src/pipelines/run_studies.py --study bkgmodel --variant bkgmodel_pessimistic --config $cfg --signals marley_final --parallel_samples $PS > $R/$cfg.log 2>&1 &
done; wait
log "  failure check:"; grep -l "Traceback\|Workflow stopped" $L/*.log $R/*.log | sed "s/^/    /"
$IMG python3 src/tools/study_status.py --signal marley_final --tag ${LABEL}_$(date +%Y%m%d) > $L/study_status.log 2>&1
mkdir -p _runs/$LABEL; cp -a output/data output/logs config _runs/$LABEL/
log "ANALYSIS DONE"
for cfg in $CONFIGS; do
  ( for n in gamma_final neutron_final; do
      ( rm -f output/data/solar/weighted/$cfg/$n/truncated/*AnalysisData*.pkl
        $IMG python3 src/physics/signal/03_analysis.py --config $cfg --folder Truncated --no-rewrite --no-debug --no-plot --signal $n --analysis Sensitivity --energy SolarEnergy --oscillation_backend nufast --save_weighted ) > $L/extra_${cfg}_03_$n.log 2>&1 &
    done; wait
    for n in marley_final gamma_final neutron_final radiological_final; do
      $IMG python3 src/physics/signal/04_weighted.py --config $cfg --folder Truncated --rewrite --no-debug --signal $n --analysis DayNight HEP --oscillation_backend nufast > $L/extra_${cfg}_04_$n.log 2>&1 &
    done; wait ) &
done; wait
grep -l "Traceback" $L/extra_*.log | sed "s/^/  FAILED: /"
rm -rf $STAGE_DIR/$LABEL
$IMG python3 $B/src/tools/stage_compare.py $B $STAGE_DIR/$LABEL $LABEL > $L/stage.log 2>&1; tail -1 $L/stage.log
log "STAGED"
