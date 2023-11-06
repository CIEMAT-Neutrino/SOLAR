#!/bin/bash

# This scipt should be used to clear the output of the notebooks in the repository

jupyter nbconvert --clear-output --inplace ../notebooks/*.ipynb