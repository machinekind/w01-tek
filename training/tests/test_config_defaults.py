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
