#!/bin/bash

# Important note: This script is meant to be run from the root of the repository!!!

# Description: Setup the environment for the project
# source /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.22.08/x86_64-centos7-gcc48-opt/bin/thisroot.sh
# source /scr/neutrinos/rodrigoa/x86_64-centos7-gcc48-opt/bin/thisroot.sh

# Mount the key directories (data, notebooks, sensitivity) to the right path in the repository
directories=("data" "notebooks" "sensitivity")

for dir in "${directories[@]}"; do
    if [ -L "$dir" ]; then
        echo "$dir symlink already exists"
    else
        echo "generating symlink for $dir..."
        sshfs manthey@gaeuidc1.ciemat.es:/pc/choozdsk01/palomare/DUNE/SOLAR/$dir "$dir"
    fi
done