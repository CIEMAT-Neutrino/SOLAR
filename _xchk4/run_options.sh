#!/bin/bash
# offline option evaluation, one config at a time, after the event tables exist
B=/pc/choozdsk01/users/manthey/SOLAR-sb; cd $B
while ! grep -q DONE _xchk4/run_events.log; do sleep 60; done
for c in hd_1x2x6_lateralAPA hd_1x2x6_centralAPA vd_1x8x14_3view_30deg_nominal vd_1x8x14_3view_30deg_shielded; do
  while [ $(free -g | awk '/Mem:/{print $7}') -lt 30 ] || [ $(cut -d. -f1 /proc/loadavg) -ge 40 ]; do sleep 120; done
  echo "######## $c $(date)"
  apptainer exec -B /pnfs,/afs,/pc,/cvmfs --home=$B/ --pwd $B/ /pc/choozdsk01/users/manthey/SOLAR/containers/solar_v1.0.sif python3 _xchk4/s7_options.py $c 2>&1 | grep -v -i "warn"
done
echo "######## weights $(date)"
apptainer exec -B /pnfs,/afs,/pc,/cvmfs --home=$B/ --pwd $B/ /pc/choozdsk01/users/manthey/SOLAR/containers/solar_v1.0.sif python3 _xchk4/s6_weights.py 2>&1 | grep -v -i "warn"
echo DONE $(date)
