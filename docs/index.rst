.. SOLAR documentation master file, created by
   sphinx-quickstart on Tue Oct 24 17:15:15 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. raw:: html

   <style> 
      .tealtitle {color:#008080; font-weight:bold; font-size:60px} 
      .tealsttle {color:#008080; font-weight:bold; font-size:30px} 
      .tealstext {color:#008080; font-weight:bold; font-size:17px} 
      .tealtexts {color:#008080; font-weight:bold; font-size:12px} 
   </style>

.. role:: tealtitle
.. role:: tealsttle
.. role:: tealstext
.. role:: tealtexts

Welcome to SOLAR's documentation!
=================================

--Insert logo here!--

`Sergio Manthey Corchado <https://github.com/mantheys>`_

The **SolarNuAna_module Output Library for Analysis & Research** is a python library to analyze mc data generated with the `DUNE <https://github.com/DUNE>`_ software framework
and analysied with the `SolarNuAna <https://github.com/DUNE/duneana/blob/develop/duneana/SolarNuAna/SolarNuAna_module.cc>`_ module.

You can navigate through the documentation using the table of contents below and you can search for specific keywords using the search tab placed at left side.

---------------------------------------------------------------------------------------------------------------------------------------------

**Contents**
=================================

.. automodule:: lib.__init__
    :members:

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   tutorial
   source/modules

---------------------------------------------------------------------------------------------------------------------------------------------

.. warning::
   🚧 This project is still under development. Please, contact the authors for more information.🚧

**SUMMARY**
------------

For a quick summary or just as a reminder follow the next steps:

You need to have ``git``, ``python (>=3.6)`` and ``pip3`` installed.

0.- Clone the repository into a local directory and create your branch:

.. code-block:: bash

   git clone https://github.com/CIEMAT-Neutrino/SOLAR.git 
   cd SOLAR
   git checkout -b <your_branch_name>


1.- Install packages needed for the library to run:

* **[RECOMENDED] Work with VSCode**:

* From CIEMAT computers
   - CREATE VIRTUAL ENVIROMENT:
   - Install VSCode and some extensions: Remote-SSH, Jupyter, vscode-numpy-viewer, **Python Environment Manager**
   - CREATE VIRTUAL ENVIROMENT: **VSCode venv extension**. It will recomend you the ``/scripts/requirements.txt`` packages and it will automatically install them 😊
   
* From your local machine use conda (follow this `tutorial <https://docs.conda.io/projects/miniconda/en/latest/>`_) to create a virtual enviroment and install the packages from the ``/scripts/requirements.txt`` file:

.. code-block:: bash

      conda create -n <your_env_name> python=3.6
      conda activate <your_env_name>
      conda install -c conda-forge root
      pip install -r scripts/requirements.txt

2.- Prepare the library to be run (just the first time):

.. code-block:: bash

   cd SCINT/scripts
   sh setup.sh

To be run from the ``scripts`` folder (it will ask you for confirmation) and it will mount the ``notebooks`` folder to make your analysis. 
Additionally, if you have created your own virtual enviroment in a CIEMAT computer you need to install some packages (make sure it is activated) and answer ``y`` to the INSTALL question.
If have created the virtual enviroment with the VSCode extension you will have them installed already, answer ``n``.

3.- Make sure you have access to data to analyse:

* **[RECOMENDED] Configure VSCode SSH conection** and work from ``gaeuidc1.ciemat.es`` (you will have access to the data in ``/pc/choozdsk01/palomare/DUNE/SOLAR/``)
* Mount the folder with the data in your local machine ``sshfs pcaeXYZ:/pc/choozdsk01/palomare/DUNE/SOLAR/ data`` making sure you have an empty ``data`` folder 📂.

4.- Have a look on the ``notebooks`` folder to see how to visualize data and run the macros:

.. code-block:: bash

   cd ../notebooks
   juptyer notebook notebook.ipynb

This folder is prepared **NOT** to be synchronized with the repository so you can make your own analysis without affecting the rest of the team.
If you think your analysis can be useful for the rest of the team contact the authors to add it to the repository.

If you want to have a personal folder to store your test files locally you can create an ``scratch`` folder (it wont be synchronized with the repository).
Otherwise you can create a folder for your custom scripts and add them to the ``.gitignore`` file:

.. code-block:: bash

   mkdir <your_folder_name>
   echo "<your_folder_name/*>" >> .gitignore

5.- To run the macros:

.. code-block:: bash

   cd ../macros
   python3 XXmacro.py (--flags input)

---------------------------------------------------------------------------------------------------------------------------------------------

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _Python: http://python.org/
.. _Sphinx: http://sphinx.pocoo.org/
.. _reStructuredText: http://docutils.sourceforge.net/rst.html
.. _SOLAR: http://solar.readthedocs.io/en/latest/