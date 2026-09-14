# Generate OpHit Statistics PKL Files

This directory contains scripts to generate PKL files with OpHit statistics, replicating the logic from `OpHitSignal.ipynb`.

## Background

The notebook `src/notebooks/ophit/OpHitSignal.ipynb` creates the image:
```
output/images/ophit/hd_1x2x6_centralAPA/marley_flash/hd_1x2x6_centralAPA_marley_flash_TotalOpHit_Stats.png
```

This image shows histograms for:
- OpHitNum (Number of OpHits)
- OpHitMeanTime (Weighted Mean Time of OpHits)
- OpHitMeanR (Weighted Mean R of OpHits)

## Scripts

### 1. `generate_ophit_stats_pkl.py`

A standalone Python script that:
1. Loads data using the OPHIT workflow
2. Filters it with the same criteria as the notebook:
   - Geometry == hd_1x2x6_centralAPA
   - Version == hd_1x2x6_centralAPA
   - Name == marley_flash
   - OpHitMeanTime < 100
   - OpHitTotalPE > 0
3. Computes normalized histograms for each variable
4. Saves the data as a PKL file in DataFrame format

**Output location:**
```
output/data/workflow/ophit/hd_1x2x6_centralAPA/marley_flash/hd_1x2x6_centralAPA_marley_flash_TotalOpHit_Stats.pkl
```

**PKL file structure:**
- Each row represents a histogram bin or a summary statistic
- Columns: Config, Name, Geometry, Version, Variable, BinCenter, BinEdgeLow, BinEdgeHigh, Counts
- Variables include: OpHitNum, OpHitMeanTime, OpHitMeanR
- Summary statistics are stored with suffixes: _Mean, _STD, _99Percentile

### 2. `generate_ophit_stats_pkl_notebook_version.py`

Same as above but organized in sections for easy copying into Jupyter notebook cells.

## Usage

### From command line (if ROOT is properly configured):
```bash
cd /pc/choozdsk01/users/manthey/SOLAR
python3 src/scripts/generate_ophit_stats_pkl.py
```

### From Jupyter notebook:
1. Open a notebook in the SOLAR project
2. Copy the contents of `generate_ophit_stats_pkl_notebook_version.py` into a cell
3. Run the cell
4. The PKL file will be created at the specified location

## Notes

- The script requires the SOLAR library which depends on ROOT
- The OPHIT workflow computes OpHitMeanTime and OpHitMeanR from the raw data
- The histograms are normalized by the number of events (same as the notebook)
- The binning strategy matches the notebook:
  - OpHitNum: bins of width 1
  - OpHitMeanTime, OpHitMeanR: 75 linearly spaced bins from 0th to 99.9th percentile

## Modifying for other configurations

To generate PKL files for other configurations, modify the `configs` variable:
```python
configs = {"hd_1x2x6_centralAPA": ["marley_flash", "marley_ophit"]}
```

or for different geometries:
```python
configs = {"vd_1x8x14_3view_30deg": ["marley_ophit"]}
```
