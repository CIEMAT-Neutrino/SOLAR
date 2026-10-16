#!/bin/bash
# sequential per-cluster tables (one load at a time); waits if memory is short
B=/pc/choozdsk01/users/manthey/SOLAR-sb; cd $B
for c in hd_1x2x6_lateralAPA hd_1x2x6_centralAPA vd_1x8x14_3view_30deg_nominal vd_1x8x14_3view_30deg_shielded; do
 for n in radiological_final gamma_final neutron_final; do
  [ -f _xchk4/events/${c}_${n}.pkl ] && [ $n = neutron_final ] && [ $c = hd_1x2x6_lateralAPA ] && continue
  while [ $(free -g | awk '/Mem:/{print $7}') -lt 30 ] || [ $(cut -d. -f1 /proc/loadavg) -ge 40 ]; do sleep 120; done
  echo "== $c $n $(date)"
  /usr/bin/time -f "maxrss_kb %M elapsed %e" apptainer exec -B /pnfs,/afs,/pc,/cvmfs --home=$B/ --pwd $B/ /pc/choozdsk01/users/manthey/SOLAR/containers/solar_v1.0.sif python3 _xchk4/s4_events.py $c $n 2>&1 | grep -E "^vol|quality clusters|maxrss|Error|Traceback|rror:"
 done
done
echo DONE $(date)
