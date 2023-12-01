#!/bin/bash

# Important note: This script is meant to be run from the root of the repository!!!

# Description: Setup the environment for the project
# source /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.22.08/x86_64-centos7-gcc48-opt/bin/thisroot.sh
# source /scr/neutrinos/rodrigoa/x86_64-centos7-gcc48-opt/bin/thisroot.sh

# Mount the key directories (data, notebooks, sensitivity) to the right path in the repository
if [ -L data ]; then
    echo "data symlink already exists"
else
    echo "data symlink does not exist"
    sshfs manthey@gaeuidc1.ciemat.es:/pc/choozdsk01/palomare/DUNE/SOLAR/data data
fi

if [ -L notebooks ]; then
    echo "notebooks symlink already exists"
else
    echo "notebooks symlink does not exist"
    sshfs manthey@gaeuidc1.ciemat.es:/pc/choozdsk01/palomare/DUNE/SOLAR/notebooks notebooks
fi

if [ -L sensitivity ]; then
    echo "sensitivity symlink already exists"
else
    echo "sensitivity symlink does not exist"
    sshfs manthey@gaeuidc1.ciemat.es:/pc/choozdsk01/palomare/DUNE/SOLAR/sensitivity sensitivity
fi