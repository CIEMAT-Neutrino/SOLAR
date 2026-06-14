SOLAR Documentation
===================

SOLAR is the analysis and reconstruction toolkit used to study solar-neutrino signals from DUNE `SolarNuAna_module` outputs. The repository combines staged detector workflow studies with higher-level physics analyses such as Day-Night asymmetry, HEP searches, fiducial optimization, and oscillation sensitivity scans.

The current codebase is organized around:

- staged reconstruction drivers in `src/physics/calibration/`, `src/physics/detector/`, and `src/physics/truth/`
- analysis orchestration in `src/pipelines/`
- shared utilities in `lib/`
- central defaults in `config/analysis/` (split JSON files merged at runtime)

Recent repository developments reflected in these docs include config-driven Gaussian smoothing, shared fiducial-optimization helpers, and regression tests for the smoothing pipeline.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   tutorial
   workflow
   development
   source/lib

Quick Start
-----------

Clone the repository and create a Python environment:

.. code-block:: bash

   git clone https://github.com/CIEMAT-Neutrino/SOLAR.git
   cd SOLAR
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r docs/requirements.txt

Run the full staged detector chain for one sample:

.. code-block:: bash

   python3 src/pipelines/run_all.py --config hd_1x2x6_centralAPA --name marley

Run the high-level solar analysis driver:

.. code-block:: bash

   python3 src/pipelines/run_sensitivity.py \
     --config hd_1x2x6_centralAPA \
     --names marley gamma neutron \
     --analysis DayNight HEP Sensitivity \
     --folder Reduced

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
