"""
Nadir integration of computed P_ee maps (OSC_NADIR_OVERSAMPLE) and the per-cut sampling
markers that stop 04_best_cuts.py / 06_significance.py from mixing old and new templates.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lib.defaults import load_analysis_info
from lib.oscillation import _get_oscillation_map_computed, get_oscillation_map
from lib.template_guards import (
    check_template_sampling_marker,
    template_sampling_marker_path,
    write_template_sampling_marker,
)
from src.utils import get_project_root

POINT = dict(dm2=[3.831e-5], sin13=[0.022], sin12=[0.2466])


def _map(**kwargs):
    return list(_get_oscillation_map_computed(backend="nufast", output="df", **POINT, **kwargs).values())[0]


def test_default_is_unchanged_point_sampling():
    assert np.array_equal(_map().to_numpy(), _map(nadir_oversample=1).to_numpy())


def test_oversampled_map_keeps_row_layout_and_normalisation():
    n_rows = int(load_analysis_info(str(get_project_root()))["NADIR_BINS"])
    base, integrated = _map(), _map(nadir_oversample=4)
    assert integrated.shape == base.shape == (n_rows, base.shape[1])
    assert np.allclose(integrated.index.to_numpy(), base.index.to_numpy())
    # same PDF-weighted total to well below the size of the aliasing effect
    assert abs(integrated.to_numpy().sum() / base.to_numpy().sum() - 1.0) < 0.01


def test_oversampling_converges():
    x4, x16 = _map(nadir_oversample=4).to_numpy(), _map(nadir_oversample=16).to_numpy()
    x1 = _map().to_numpy()
    assert np.abs(x4 - x16).max() < 0.2 * np.abs(x1 - x16).max()


def test_get_oscillation_map_passes_oversample_through():
    via_public = list(get_oscillation_map(backend="nufast", output="df", nadir_oversample=4, **POINT).values())[0]
    assert np.array_equal(via_public.to_numpy(), _map(nadir_oversample=4).to_numpy())


def test_oversample_rejects_unsupported_outputs():
    with pytest.raises(NotImplementedError):
        _get_oscillation_map_computed(backend="nufast", output="df", separate_day_night=True, nadir_oversample=4, **POINT)


def test_sampling_markers(tmp_path):
    args = (str(tmp_path), "cfg", "marley", 1, 4, 8)
    ok, reason = check_template_sampling_marker(*args, backend="nufast", nadir_oversample=4)
    assert not ok and "no sampling marker" in reason

    write_template_sampling_marker(*args, scope="grid", backend="nufast", nadir_oversample=4)
    assert check_template_sampling_marker(*args, backend="nufast", nadir_oversample=4)[0]
    assert not check_template_sampling_marker(*args, backend="nufast", nadir_oversample=1)[0]
    assert not check_template_sampling_marker(*args, backend="prob3", nadir_oversample=4)[0]

    # a reference-point scan with the same sampling must not downgrade the grid marker
    write_template_sampling_marker(*args, scope="scan", backend="nufast", nadir_oversample=4)
    assert check_template_sampling_marker(*args, backend="nufast", nadir_oversample=4, scopes=("grid",))[0]

    # a scan with different sampling does replace it, and then only satisfies scan consumers
    write_template_sampling_marker(*args, scope="scan", backend="nufast", nadir_oversample=2)
    assert not check_template_sampling_marker(*args, backend="nufast", nadir_oversample=2, scopes=("grid",))[0]
    assert check_template_sampling_marker(*args, backend="nufast", nadir_oversample=2, scopes=("scan", "grid"))[0]
    assert os.path.basename(template_sampling_marker_path(*args)) == "cfg_marley_NHits1_AdjCl4_OpHits8_SAMPLING.json"
