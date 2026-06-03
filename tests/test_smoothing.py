import json
from pathlib import Path

import numpy as np

from lib.smoothing import (
    get_component_smoothing_config,
    get_smoothing_config,
    should_smooth_component,
    smooth_histogram,
    smooth_histogram_with_config,
)


def test_none_returns_identical_output():
    values = np.array([0.0, 1.0, 3.0, 0.0])
    result = smooth_histogram(values, method="none")
    assert np.array_equal(result, values)


def test_gaussian_preserves_1d_integral():
    values = np.array([0.0, 0.0, 10.0, 0.0, 0.0])
    result = smooth_histogram(values, method="gaussian", sigma=0.75)
    assert result.shape == values.shape
    assert np.isclose(np.sum(result), np.sum(values))
    assert np.all(result >= 0)


def test_zero_only_input_stays_zero():
    values = np.zeros(6)
    result = smooth_histogram(values, method="gaussian", sigma=0.75)
    assert np.array_equal(result, values)


def test_gaussian_preserves_2d_integral():
    values = np.zeros((4, 4))
    values[1, 2] = 5.0
    result = smooth_histogram(values, method="gaussian", sigma_x=0.75, sigma_y=0.75)
    assert result.shape == values.shape
    assert np.isclose(np.sum(result), np.sum(values))
    assert np.all(result >= 0)


def test_config_defaults_expose_gaussian_smoothing():
    root = Path(__file__).resolve().parents[1]
    config = get_smoothing_config(str(root), analysis_name="DAYNIGHT", dimensions="1d")
    assert config["method"] == "gaussian"
    assert config["enabled"] is True
    assert "sigma" in config["params"]


def test_config_application_preserves_integral():
    root = Path(__file__).resolve().parents[1]
    config = get_smoothing_config(str(root), analysis_name="SENSITIVITY", dimensions="2d")
    values = np.zeros((3, 3))
    values[1, 1] = 9.0
    result = smooth_histogram_with_config(values, config)
    assert np.isclose(np.sum(result), 9.0)



def test_component_smoothing_selection():
    config = {
        "enabled": True,
        "method": "gaussian",
        "component_mode": "only",
        "components": ["neutron", "gamma"],
        "params": {"sigma": 0.75},
    }
    assert should_smooth_component(config, "neutron") is True
    assert should_smooth_component(config, "gamma") is True
    assert should_smooth_component(config, "hep") is False


def test_component_smoothing_config_disables_unselected_component():
    config = {
        "enabled": True,
        "method": "gaussian",
        "component_mode": "only",
        "components": ["neutron"],
        "params": {"sigma": 0.75},
    }
    neutron_config = get_component_smoothing_config(config, "neutron")
    hep_config = get_component_smoothing_config(config, "hep")
    assert neutron_config["method"] == "gaussian"
    assert hep_config["method"] == "none"


def test_per_analysis_smoothing_settings_are_independent(tmp_path):
    root = tmp_path
    (root / "analysis").mkdir(parents=True, exist_ok=True)
    analysis_payload = {
        "SMOOTHING": {
            "enabled": True,
            "method": "gaussian",
            "dimensions": {
                "1d": {"sigma": 0.4},
                "2d": {"sigma_x": 0.2, "sigma_y": 0.2},
            },
            "ENERGIES": {
                "SolarEnergy": {
                    "dimensions": {
                        "1d": {"sigma": 0.8},
                    }
                }
            },
            "ANALYSES": {
                "DAYNIGHT": {
                    "dimensions": {
                        "1d": {"sigma": 1.1},
                    }
                },
                "HEP": {
                    "dimensions": {
                        "1d": {"sigma": 2.2},
                    },
                    "ENERGIES": {
                        "SolarEnergy": {
                            "dimensions": {
                                "1d": {"sigma": 3.3},
                            }
                        }
                    },
                },
            },
        }
    }
    with open(root / "analysis" / "smoothing.json", "w") as handle:
        json.dump(analysis_payload, handle)

    daynight = get_smoothing_config(str(root), analysis_name="DAYNIGHT", dimensions="1d")
    hep = get_smoothing_config(str(root), analysis_name="HEP", dimensions="1d")
    hep_solar = get_smoothing_config(str(root), analysis_name="HEP", energy="SolarEnergy", dimensions="1d")

    assert np.isclose(daynight["params"]["sigma"], 1.1)
    assert np.isclose(hep["params"]["sigma"], 2.2)
    # Analysis-specific energy overrides take precedence over global energy overrides.
    assert np.isclose(hep_solar["params"]["sigma"], 3.3)


def test_stage_specific_smoothing_overrides(tmp_path):
    root = tmp_path
    (root / "analysis").mkdir(parents=True, exist_ok=True)
    analysis_payload = {
        "SMOOTHING": {
            "enabled": True,
            "method": "gaussian",
            "ANALYSES": {
                "SENSITIVITY": {
                    "dimensions": {
                        "1d": {"sigma": 0.25},
                    },
                    "STAGES": {
                        "FIDUCIAL": {
                            "dimensions": {
                                "1d": {"sigma": 0.10},
                            }
                        },
                        "SIGNIFICANCE": {
                            "dimensions": {
                                "1d": {"sigma": 0.35},
                            }
                        },
                    },
                }
            },
        }
    }
    with open(root / "analysis" / "smoothing.json", "w") as handle:
        json.dump(analysis_payload, handle)

    fiducial = get_smoothing_config(
        str(root), analysis_name="SENSITIVITY", dimensions="1d", stage="fiducial"
    )
    significance = get_smoothing_config(
        str(root), analysis_name="SENSITIVITY", dimensions="1d", stage="significance"
    )

    assert np.isclose(fiducial["params"]["sigma"], 0.10)
    assert np.isclose(significance["params"]["sigma"], 0.35)


def test_stage_specific_component_selection(tmp_path):
    root = tmp_path
    (root / "analysis").mkdir(parents=True, exist_ok=True)
    analysis_payload = {
        "SMOOTHING": {
            "enabled": True,
            "method": "gaussian",
            "ANALYSES": {
                "HEP": {
                    "component_mode": "only",
                    "components": ["gamma"],
                    "STAGES": {
                        "FIDUCIAL": {
                            "component_mode": "all",
                            "components": [],
                        },
                        "SIGNIFICANCE": {
                            "component_mode": "only",
                            "components": ["gamma", "neutron", "8B"],
                        },
                    },
                }
            },
        }
    }
    with open(root / "analysis" / "smoothing.json", "w") as handle:
        json.dump(analysis_payload, handle)

    fiducial = get_smoothing_config(
        str(root), analysis_name="HEP", dimensions="1d", stage="fiducial"
    )
    significance = get_smoothing_config(
        str(root), analysis_name="HEP", dimensions="1d", stage="significance"
    )

    assert should_smooth_component(fiducial, "hep") is True
    assert should_smooth_component(significance, "hep") is False
    assert should_smooth_component(significance, "gamma") is True
