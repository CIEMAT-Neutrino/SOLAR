# Important note: This script is meant to be run from the root of the repository!!!

# Description: Setup the environment for the project
source /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.22.08/x86_64-centos7-gcc48-opt/bin/thisroot.sh

# Mount the key directories (data, notebooks, sensitivity) to the right path in the repository
ln -s /pc/choozdsk01/palomare/DUNE/SOLAR/data data
ln -s /pc/choozdsk01/palomare/DUNE/SOLAR/notebooks notebooks 
ln -s /pc/choozdsk01/palomare/DUNE/SOLAR/sensitivity sensitivity