#!/bin/bash

# Important note: This script is meant to be run from the root of the repository!!!

# Description: Setup the environment for the project
# source /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.22.08/x86_64-centos7-gcc48-opt/bin/thisroot.sh
# source /scr/neutrinos/rodrigoa/x86_64-centos7-gcc48-opt/bin/thisroot.sh

# Mount the key directories (notebooks, sensitivity) to the right path in the repository
# directories=("notebooks")

# for dir in "${directories[@]}"; do
#     if [ -L "$dir" ]; then
#         echo "$dir symlink already exists"
#     else
#         echo "unmounting $dir..."
# 	    fusermount -uz "$dir"
#         echo "generating symlink for $dir..."
#         sshfs manthey@gaeuidc1.ciemat.es:/pc/choozdsk01/users/manthey/SOLAR/$dir "$dir"
#     fi
# done

# Mount the key directories (data) from pnfs to the right path in the repository
directories=("data" "sensitivity")

for dir in "${directories[@]}"; do
    if [ -L "$dir" ]; then
        echo "$dir symlink already exists"
    else
        echo "unmounting $dir..."
	    fusermount -uz "$dir"
        echo "generating symlink for $dir..."
        sshfs manthey@gaeuidc1.ciemat.es:/pnfs/ciemat.es/data/neutrinos/DUNE/SOLAR/$dir "$dir"
    fi
done