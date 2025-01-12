#!/bin/bash

# Important note: This script is meant to be run from the root of the repository!!!
# Print a worning to the terminal if the script is not run from the root of the repository
if [ ! -f "src/scripts/setup.sh" ]; then
    echo "Please run this script from the root of the repository"
    exit 1
fi

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