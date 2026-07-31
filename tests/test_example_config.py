from pathlib import Path

import yaml


def test_example_config_does_not_double_rotate_corrected_source():
    config_path = Path(__file__).resolve().parents[1] / "config.example.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["camera"]["rotation_degrees"] == 90
    assert config["debug"]["preview_rotation"] == "none"
    assert config["debug"]["preview_width"] < config["debug"]["preview_height"]
