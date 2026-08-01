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


# The height scan and the terrain gate have no yaml mirror (only `dr` and the
# joystick latency/encoder/terrain blocks do), so these pin the code defaults
# themselves: a preset can only override keys that exist here.
def test_height_scan_defaults():
    assert wojtek_env.default_config().height_scan.to_dict() == {
        "enable": False,
        "x_range": (0.65, 1.45),
        "y_range": (-0.3, 0.3),
        "nx": 5,
        "ny": 5,
        "clip": 0.3,
        "hold_steps": 3,
        "delay_steps": 1,
        "dark": False,
        "mask": {
            "enable": True,
            "mount": (0.32, 0.0, 0.07),
            "pitch_deg": 15.0,
            "hfov_deg": 90.7,
            "vfov_deg": 61.2,
            "min_depth": 0.3,
            "max_depth": 3.0,
        },
        "corrupt": {
            "enable": False,
            "noise_prob": 0.6,
            "drift_prob": 0.3,
            "blackout_prob": 0.1,
            "noise_std": 0.02,
            "drift_z": 0.05,
            "drift_tilt": 0.05,
            "dropout_prob": 0.05,
        },
    }


def test_terrain_gate_defaults():
    assert wojtek_env.default_config().reward.terrain_gate.to_dict() == {
        "enable": False,
        "floor": 0.15,
        "rough_ref": 0.05,
        "landing_soften": 0.5,
    }
