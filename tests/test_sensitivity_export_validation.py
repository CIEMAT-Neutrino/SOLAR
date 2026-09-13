import numpy as np
import pandas as pd
import pytest

from tools.export_sensitivity_contours import build_rows


def _write_grids(path, values):
    for reference in ("solar", "react"):
        for parameter in ("sin12", "sin13"):
            filename = (
                f"marley_SolarEnergy_NHits1_AdjCl4_OpHits8_"
                f"{reference}_{parameter}_df.pkl"
            )
            pd.DataFrame(values, index=[1.0, 2.0], columns=[0.1, 0.2]).to_pickle(path / filename)


def test_export_rejects_empty_grid(tmp_path):
    _write_grids(tmp_path, [[np.nan, np.nan], [np.nan, np.nan]])

    with pytest.raises(ValueError, match="no finite values"):
        build_rows(str(tmp_path), "cfg", "marley", "SolarEnergy", "default", "full", 0.04, 0.02)


def test_export_preserves_union_of_planes(tmp_path):
    _write_grids(tmp_path, [[1.0, np.nan], [2.0, 3.0]])

    rows = build_rows(str(tmp_path), "cfg", "marley", "SolarEnergy", "default", "full", 0.04, 0.02)

    assert len(rows) == 4
    assert np.isnan(rows[0]["Significance"][0][1])


def test_export_clamps_tiny_negative_grid_values(tmp_path):
    _write_grids(tmp_path, [[-1e-10, 1.0], [2.0, 3.0]])

    rows = build_rows(str(tmp_path), "cfg", "marley", "SolarEnergy", "default", "full", 0.04, 0.02)

    assert len(rows) == 4
    assert all(min(values) >= 0.0 for row in rows for values in row["Significance"])