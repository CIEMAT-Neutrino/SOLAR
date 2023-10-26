# SOLAR
SolarNuAna_module Output Library for Analysis &amp; Research

## Introduction

This module is designed to analyze the output of the [SOLAR](https://github.com/DUNE/duneana/blob/develop/duneana/SolarNuAna/SolarNuAna_module.cc) module. It is designed to be used in conjunction with the [DUNE](https://github.com/DUNE) software framework.

Read more about the module [here](https://dune-solar.readthedocs.io/en/latest/).

## Installation

To use the SolarNuAna_module, you must have a working version of the DUNE software framework. This can be done by following the instructions [here](https://dune.bnl.gov/wiki/Computing#Getting_the_DUNE_Software_Framework). Once you were able to run the module on a given set of marley (+ bkg) data, use this repository to analyse the output tree.

```bash
git clone https://github.com/CIEMAT-Neutrino/SOLAR.git
cd SOLAR
git checkout -b <branch>
source scripts/setup.sh
```

## Usage

Once you have the output tree from the SolarNuAna_module, you can use the following scripts to analyze the data.

Macros:

 - 00Processing.py: This script will process the output tree from the SolarNuAna_module and save the branches in numpy format for faster access.
 - 01Calibration.py: This script will calibrate the (lifetime-corrected) charge-to-energy conversion of the TPC based on marley electrons.
 - 02Reconstruction.py: This script will reconstruct the true neutrino energy of the events based on the previous calibration &amp; the topology of the event.
 - 03Smearing.py: This script will compute the smearing matrix of the neutrino interactions and weight it with the solar neutrino flux.
 - 04Computing.py: This script will execute the full reconstruction workflow of the detector to the solar neutrino flux based on the previous smearing matrix.
 - 05Sensitivity.py: This script will compute the sensitivity to the different oscillation parameters (dm2, sin13, sin12) based on the computed reco solar spectrum & background.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Authors

 - [**Sergio Manthey Corchado**](https://github.com/mantheys)

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Acknowledgments

 - [**DUNE Collaboration**](https://github.com/DUNE)
 - [**CIEMAT**](https://github.com/CIEMAT-Neutrino)