"""A changed fiducial or classifier must invalidate cached signal templates."""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lib.template_guards import check_signal_template_inputs


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_fiducial_change_invalidates_signal_template(tmp_path):
    config = "hd_test"
    analysis_config = tmp_path / "config/analysis/config.json"
    fiducial = tmp_path / "config/analysis/fiducial/truncated/BestFiducials.json"
    template = tmp_path / "signal_BASE.pkl"
    _write(analysis_config, {"SIGNAL_BACKGROUND_CLASSIFIER": False})
    _write(fiducial, {})
    template.write_bytes(b"template")
    os.utime(analysis_config, (10, 10))
    os.utime(fiducial, (10, 10))
    os.utime(template, (20, 20))
    assert check_signal_template_inputs(template, tmp_path, config, "marley_final", "Truncated")[0]
    os.utime(fiducial, (30, 30))
    current, reason = check_signal_template_inputs(template, tmp_path, config, "marley_final", "Truncated")
    assert not current and "BestFiducials.json" in reason


def test_classifier_change_invalidates_signal_template(tmp_path):
    config = "hd_test"
    analysis_config = tmp_path / "config/analysis/config.json"
    fiducial = tmp_path / "config/analysis/fiducial/truncated/BestFiducials.json"
    classifier = tmp_path / f"config/{config}/marley_final/{config}_calib/{config}_signal_background.json"
    template = tmp_path / "signal_BASE.pkl"
    _write(analysis_config, {"SIGNAL_BACKGROUND_CLASSIFIER": True})
    _write(fiducial, {})
    _write(classifier, {"SB_THRESHOLD": 0.2})
    template.write_bytes(b"template")
    for path in (analysis_config, fiducial, classifier):
        os.utime(path, (10, 10))
    os.utime(template, (20, 20))
    assert check_signal_template_inputs(template, tmp_path, config, "marley_final", "Truncated")[0]
    os.utime(classifier, (30, 30))
    current, reason = check_signal_template_inputs(template, tmp_path, config, "marley_final", "Truncated")
    assert not current and "signal_background.json" in reason


def test_reconstruction_code_change_invalidates_signal_template(tmp_path):
    config = "hd_test"
    analysis_config = tmp_path / "config/analysis/config.json"
    fiducial = tmp_path / "config/analysis/fiducial/truncated/BestFiducials.json"
    source = tmp_path / "lib/cluster.py"
    template = tmp_path / "signal_BASE.pkl"
    _write(analysis_config, {"SIGNAL_BACKGROUND_CLASSIFIER": False})
    _write(fiducial, {})
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("# reconstruction")
    template.write_bytes(b"template")
    for path in (analysis_config, fiducial, source):
        os.utime(path, (10, 10))
    os.utime(template, (20, 20))
    assert check_signal_template_inputs(template, tmp_path, config, "marley_final", "Truncated")[0]
    os.utime(source, (30, 30))
    current, reason = check_signal_template_inputs(template, tmp_path, config, "marley_final", "Truncated")
    assert not current and "cluster.py" in reason
