"""Pins the yaml config defaults to the Python defaults that back them, so
the two sources (conf/*.yaml and the code) cannot silently drift apart.
"""

import yaml

from wojtek_rl import env as wojtek_env
from wojtek_rl import paths
from wojtek_rl import randomize

CONFIG_YAML = paths.PROJECT_DIR / "wojtek_rl" / "conf" / "config.yaml"
JOYSTICK_YAML = paths.PROJECT_DIR / "wojtek_rl" / "conf" / "task" / "joystick.yaml"


def test_dr_yaml_matches_code_defaults():
    cfg = yaml.safe_load(CONFIG_YAML.read_text())
    assert cfg["dr"] == randomize._DEFAULT_DR


def test_joystick_yaml_latency_encoder_match_env_defaults():
    task_cfg = yaml.safe_load(JOYSTICK_YAML.read_text())
    default = wojtek_env.default_config()
    assert task_cfg["env"]["latency"] == default.latency.to_dict()
    assert task_cfg["env"]["encoder"] == default.encoder.to_dict()


def test_joystick_yaml_terrain_matches_env_defaults():
    task_cfg = yaml.safe_load(JOYSTICK_YAML.read_text())
    default = wojtek_env.default_config()
    assert task_cfg["env"]["terrain"] == default.terrain.to_dict()


def _listify(value):
    if isinstance(value, dict):
        return {k: _listify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_listify(v) for v in value]
    return value


def test_joystick_yaml_height_scan_matches_env_defaults():
    task_cfg = yaml.safe_load(JOYSTICK_YAML.read_text())
    default = wojtek_env.default_config()
    assert task_cfg["env"]["height_scan"] == _listify(
        default.height_scan.to_dict()
    )


# The terrain gate has no yaml mirror (nothing in the reward block does), so
# this pins the code defaults themselves: a preset can only override keys
# that exist here.
def test_terrain_gate_defaults():
    assert wojtek_env.default_config().reward.terrain_gate.to_dict() == {
        "enable": False,
        "floor": 0.15,
        "rough_ref": 0.05,
        "landing_soften": 0.5,
    }
