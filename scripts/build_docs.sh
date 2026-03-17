#!/bin/bash

# navigate to the docs directory
cd /path/to/docs

# activate virtual environment if necessary
source /path/to/venv/bin/activate

# install dependencies
pip install -r requirements.txt

# build the docs using Sphinx
make html