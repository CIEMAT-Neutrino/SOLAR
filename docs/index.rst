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

The **SolarNuAna_module Output Library for Analysis & Research** (:tealstext:`SOLAR`) is a python library to analyze mc data generated with the [DUNE](https://github.com/DUNE) software framework
and analysied with the [SOLAR](https://github.com/DUNE/duneana/blob/develop/duneana/SolarNuAna/SolarNuAna_module.cc) module.

You can navigate through the documentation using the table of contents below and you can search for specific keywords using the search tab placed at left side.

---------------------------------------------------------------------------------------------------------------------------------------------

**Contents**
=================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   tutorial
   source/lib
   source/modules

---------------------------------------------------------------------------------------------------------------------------------------------

.. warning::
   🚧 This project is still under development. Please, contact the authors for more information.🚧

**SUMMARY**
------------

For a quick summary or just as a reminder follow the next steps:

You need to have ``git``, ``python (>=3.7)`` and ``pip3`` installed.

0.- Clone the repository into a local directory and create your branch:

.. code-block:: bash

   git clone https://github.com/CIEMAT-Neutrino/SOLAR.git 
   cd SOLAR
   git checkout -b <your_branch_name>


1.- Install packages needed for the library to run:

* **[RECOMENDED] Work with VSCode**:
   - Install VSCode and some extensions: Remote-SSH, Jupyter, vscode-numpy-viewer, **Python Environment Manager**
   - CREATE VIRTUAL ENVIROMENT: **VSCode venv extension**. It will recomend you the ``/scripts/requirements.txt`` packages and it will automatically install them :)
   
* From CIEMAT computers
   - CREATE VIRTUAL ENVIROMENT:

.. code-block:: bash
   
   mkdir venv_python3.7
   cd venv_python3.7
   /cvmfs/sft.cern.ch/lcg/releases/Python/3.7.3-f4f57/x86_64-centos7-gcc7-opt/bin/python3 -m venv .
   source bin/activate

2.- Prepare the library to be run (just the first time):

.. code-block:: bash

   cd SCINT/scripts
   sh setup.sh

To be run from the ``scripts`` folder (it will ask you for confirmation) and it will download the ``notebooks`` folder to make your analysis. 
Additionally, if you have created your own virtual enviroment in a CIEMAT computer you need to install some packages (make sure it is activated) and answer ``y`` to the INSTALL question. If have created the virtual enviroment with the VSCode extension you will have them installed already, answer ``n``.

3.- Make sure you have access to data to analyse:

* **[RECOMENDED] Configure VSCode SSH conection** and work from ``gaeuidc1.ciemat.es`` (you will have access to the data in ``/pc/choozdsk01/palomare/SCINT/folders``)
* Mount the folder with the data in your local machine ``sshfs pcaeXYZ:/pc/choozdsk01/palomare/SCINT/folder ../data`` making sure you have an empty ``data`` folder 📂.
* Copy the data to your local machine. See ``sh scripts/copy_data.sh`` for an example on how to copy the ``TUTORIAL`` data

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