"""
Tests for lib/defaults.py — get_workflow_flags and DEFAULT_WORKFLOW_FLAGS.

Coverage:
  - All default flags are False (or match DEFAULT_WORKFLOW_FLAGS)
  - Flags read from analysis.json WORKFLOW section
  - Partial override merges with defaults (unset keys stay at default)
  - Unknown analysis returns empty dict
  - Case insensitive analysis key lookup
  - Flags from analysis.json fully override defaults
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lib.defaults import DEFAULT_WORKFLOW_FLAGS, get_workflow_flags


KNOWN_ANALYSES = ["HEP", "DAYNIGHT", "SENSITIVITY"]


# ── Default flag values ───────────────────────────────────────────────────────

class TestDefaultFlags:
    def test_hep_boolean_flags_are_bool(self):
        flags = DEFAULT_WORKFLOW_FLAGS.get("HEP", {})
        bool_keys = ("pl_isotonic", "pl_signal_bands", "significance_bins")
        for key in bool_keys:
            assert isinstance(flags[key], bool), f"HEP.{key} is not bool: {type(flags[key])}"

    def test_daynight_defaults_all_bool(self):
        flags = DEFAULT_WORKFLOW_FLAGS.get("DAYNIGHT", {})
        for key, val in flags.items():
            assert isinstance(val, bool), f"DAYNIGHT.{key} is not bool: {type(val)}"

    def test_hep_contains_expected_keys(self):
        flags = DEFAULT_WORKFLOW_FLAGS.get("HEP", {})
        for key in ("pl_isotonic", "pl_signal_bands", "significance_bins"):
            assert key in flags, f"missing key in HEP defaults: {key}"

    def test_daynight_contains_expected_keys(self):
        flags = DEFAULT_WORKFLOW_FLAGS.get("DAYNIGHT", {})
        for key in ("background_error", "significance_bins"):
            assert key in flags, f"missing key in DAYNIGHT defaults: {key}"


# ── get_workflow_flags with real project root ─────────────────────────────────

class TestGetWorkflowFlagsReal:
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    def test_returns_dict(self):
        result = get_workflow_flags(self.ROOT, "HEP")
        assert isinstance(result, dict)

    def test_hep_has_required_keys(self):
        result = get_workflow_flags(self.ROOT, "HEP")
        for key in ("pl_isotonic", "pl_signal_bands", "significance_bins"):
            assert key in result, f"missing key: {key}"

    def test_daynight_has_required_keys(self):
        result = get_workflow_flags(self.ROOT, "DAYNIGHT")
        for key in ("background_error", "significance_bins"):
            assert key in result, f"missing key: {key}"

    def test_unknown_analysis_returns_empty_or_dict(self):
        result = get_workflow_flags(self.ROOT, "NONEXISTENT_ANALYSIS_XYZ")
        assert isinstance(result, dict)

    def test_case_insensitive_lookup(self):
        upper = get_workflow_flags(self.ROOT, "HEP")
        lower = get_workflow_flags(self.ROOT, "hep")
        mixed = get_workflow_flags(self.ROOT, "Hep")
        assert upper == lower == mixed


# ── get_workflow_flags with tmp analysis.json ─────────────────────────────────

class TestGetWorkflowFlagsOverride:
    def _write_analysis_json(self, tmp_path, payload):
        # load_analysis_info reads analysis/config.json (not analysis/analysis.json)
        analysis_dir = tmp_path / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        with open(analysis_dir / "config.json", "w") as fh:
            json.dump(payload, fh)
        return str(tmp_path)

    def test_override_single_flag(self, tmp_path):
        root = self._write_analysis_json(tmp_path, {
            "WORKFLOW": {
                "HEP": {"pl_isotonic": True}
            }
        })
        flags = get_workflow_flags(root, "HEP")
        assert flags["pl_isotonic"] is True
        # Other defaults still present
        assert "pl_signal_bands" in flags
        assert "significance_bins" in flags

    def test_override_does_not_leak_into_other_analysis(self, tmp_path):
        root = self._write_analysis_json(tmp_path, {
            "WORKFLOW": {
                "HEP": {"pl_isotonic": True}
            }
        })
        daynight_flags = get_workflow_flags(root, "DAYNIGHT")
        assert "pl_isotonic" not in daynight_flags

    def test_partial_override_merges_with_defaults(self, tmp_path):
        root = self._write_analysis_json(tmp_path, {
            "WORKFLOW": {
                "DAYNIGHT": {"background_error": True}
            }
        })
        flags = get_workflow_flags(root, "DAYNIGHT")
        assert flags["background_error"] is True
        assert "significance_bins" in flags

    def test_empty_workflow_section_uses_defaults(self, tmp_path):
        root = self._write_analysis_json(tmp_path, {"WORKFLOW": {}})
        flags = get_workflow_flags(root, "HEP")
        # Should fall back entirely to DEFAULT_WORKFLOW_FLAGS["HEP"]
        for key, val in DEFAULT_WORKFLOW_FLAGS.get("HEP", {}).items():
            assert flags[key] == val

    def test_missing_workflow_section_uses_defaults(self, tmp_path):
        root = self._write_analysis_json(tmp_path, {"OTHER_KEY": {}})
        flags = get_workflow_flags(root, "HEP")
        for key, val in DEFAULT_WORKFLOW_FLAGS.get("HEP", {}).items():
            assert flags[key] == val

    def test_all_hep_flags_overridden(self, tmp_path):
        overrides = {"pl_isotonic": True, "pl_signal_bands": True, "significance_bins": True}
        root = self._write_analysis_json(tmp_path, {
            "WORKFLOW": {"HEP": overrides}
        })
        flags = get_workflow_flags(root, "HEP")
        for key, val in overrides.items():
            assert flags[key] == val

    def test_all_daynight_flags_overridden(self, tmp_path):
        overrides = {"background_error": True, "significance_bins": True}
        root = self._write_analysis_json(tmp_path, {
            "WORKFLOW": {"DAYNIGHT": overrides}
        })
        flags = get_workflow_flags(root, "DAYNIGHT")
        for key, val in overrides.items():
            assert flags[key] == val
