SOLAR Documentation
===================

SOLAR is the analysis and reconstruction toolkit used to study solar-neutrino signals from DUNE `SolarNuAna_module` outputs. The repository combines staged detector workflow studies with higher-level physics analyses such as Day-Night asymmetry, HEP searches, fiducial optimization, and oscillation sensitivity scans.

The current codebase is organized around:

- staged reconstruction drivers in `src/workflow/`, `src/preselection/`, `src/PDS/`, `src/TPC/`, and `src/vertex/`
- truth-level weighting and background preparation in `src/truth/`
- analysis orchestration in `src/analysis/`
- shared utilities in `lib/`
- central defaults in `import/analysis.json`

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

   python3 src/00All.py --config hd_1x2x6_centralAPA --name marley

Run the current high-level solar analysis driver:

.. code-block:: bash

   python3 src/analysis/10SensitivityAnalysis.py \
     --config hd_1x2x6_centralAPA \
     --names marley gamma neutron \
     --analysis DayNight HEP Sensitivity \
     --folder Reduced

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
