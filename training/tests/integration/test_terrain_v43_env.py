"""The v4.3 additions on a real env: the roughness-gated orientation cone
resolves from config, and every other configuration keeps the static cone.

Reuses the v4.2 test arena geometry; the envs here are constructed but
never stepped, so the module pays put_model, not a step compile.
"""

import numpy as np
import pytest

from wojtek_rl import build_terrain, paths, terrain
from wojtek_rl import env as wojtek_env

ARENA = "test"


@pytest.fixture(scope="module")
def arena():
    a = terrain.generate(
        seed=0, n_rows=3, flat_row=True, pad_radius=0.3,
        stair_platform_half=0.3, stair_tread=(0.25, 0.45),
        type_caps={"pyramid_stairs": 0.7, "inverted_pyramid_stairs": 0.7},
    )
    build_terrain.write_arena(a, ARENA)
    yield a
    for p in paths.terrain_paths(ARENA).values():
        p.unlink(missing_ok=True)


def _env(arena, flat_deg, gate_enable=True):
    cfg = wojtek_env.default_config()
    cfg.terrain.enable = True
    cfg.terrain.arena = ARENA
    cfg.terrain.flat_row = True
    cfg.terrain.spawn_mode = "feature"
    cfg.terrain.spawn_grace_sec = 1.0
    cfg.terrain.stair_tread_range = (0.25, 0.45)
    cfg.terrain.type_caps = {
        "pyramid_stairs": 0.7, "inverted_pyramid_stairs": 0.7,
    }
    cfg.terrain.pad_radius = 0.3
    cfg.reward.orientation_tol_deg = 20.0
    cfg.reward.terrain_gate.enable = gate_enable
    cfg.reward.terrain_gate.orientation_tol_flat_deg = flat_deg
    cfg.sim.num_envs = 1
    return wojtek_env.WojtekJoystick(cfg)


def test_flat_cone_resolves_to_sin_squared(arena):
    env = _env(arena, flat_deg=5.0)
    assert env._orientation_tol_flat == pytest.approx(
        float(np.square(np.sin(np.radians(5.0))))
    )
    assert env._orientation_tol == pytest.approx(
        float(np.square(np.sin(np.radians(20.0))))
    )
    # The flat cone is the smaller: the interpolation opens, never shrinks.
    assert env._orientation_tol_flat < env._orientation_tol


def test_zero_flat_deg_keeps_the_static_cone(arena):
    assert _env(arena, flat_deg=0.0)._orientation_tol_flat is None


def test_gate_off_keeps_the_static_cone(arena):
    assert _env(arena, flat_deg=5.0, gate_enable=False)._orientation_tol_flat is None
