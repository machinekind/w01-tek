"""Terrain-aware joystick env, curriculum wrapper, and JAX terrain helpers.

Covers the step-3 additions: the JAX bilinear lookup matching the numpy
ground-truth, terrain-relative contact/height/clearance wiring, the legged_gym
promote/demote transition, spawn-on-tile reset, and the curriculum auto-reset
wrapper. Env/wrapper tests share a small arena written to the real sidecar
paths (like test_terrain.py) to keep put_model cheap.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jp
import numpy as np
import pytest

from wojtek_rl import build_terrain, paths, terrain, terrain_env
from wojtek_rl import env as wojtek_env
from wojtek_rl.build_model import FOOT_RADIUS
from wojtek_rl.terrain_wrapper import (
    TerrainAutoResetWrapper,
    wrap_for_terrain_brax_training,
)

SMALL_ROWS = 3


@pytest.fixture(scope="module")
def small_arena():
    """A small arena written to the real terrain sidecars, so the env loads it
    from disk the way training does. Deterministic; refreshes the artifacts."""
    a = terrain.generate(seed=0, n_rows=SMALL_ROWS)
    build_terrain.write_arena(a)
    return a


@pytest.fixture(scope="module")
def terrain_config():
    cfg = wojtek_env.default_config()
    cfg.terrain.enable = True
    return cfg


@pytest.fixture(scope="module")
def terrain_env_inst(small_arena, terrain_config):
    return wojtek_env.WojtekJoystick(terrain_config)


# -- 1. JAX bilinear sampler == numpy ground-truth lookup ---------------------


def test_bilinear_sampler_matches_lookup_height(small_arena):
    s = small_arena.spec
    cell_x = (s.x_max - s.x_min) / (s.hfield.ncol - 1)
    cell_y = (s.y_max - s.y_min) / (s.hfield.nrow - 1)
    grid = jp.asarray(small_arena.lookup, dtype=jp.float32)

    rng = np.random.default_rng(0)
    xs = rng.uniform(s.x_min, s.x_max, size=256)
    ys = rng.uniform(s.y_min, s.y_max, size=256)

    ref = terrain.lookup_height(small_arena, xs, ys)
    got = np.array(
        terrain_env.bilinear_sample(
            grid, s.x_min, cell_x, s.y_min, cell_y, jp.asarray(xs), jp.asarray(ys)
        )
    )
    # Float32 device grid vs float64 numpy reference: the surface spans a few
    # tenths of a metre, so this is machine precision, not a semantic gap.
    np.testing.assert_allclose(got, ref, atol=2e-4)


# -- 2. Terrain-relative contact / base height / clearance --------------------


def test_terrain_relative_measurements(terrain_env_inst, monkeypatch):
    """A linear ramp is bilinear-exact, so h(x, y) = a*x + b*y under the env's
    own lookup grid. The three helpers must subtract exactly that surface."""
    env = terrain_env_inst
    nrow, ncol = env._terrain_lookup.shape
    xs = env._terrain_x_min + env._terrain_cell_x * np.arange(ncol)
    ys = env._terrain_y_min + env._terrain_cell_y * np.arange(nrow)
    a, b = 0.1, -0.05
    synth = (a * xs[None, :] + b * ys[None, :].T).astype(np.float32)
    monkeypatch.setattr(env, "_terrain_lookup", jp.asarray(synth))

    def h(x, y):
        return a * x + b * y

    ngeom = env.mj_model.ngeom
    foot_xy = np.array([[0.3, 0.2], [-0.4, 0.5], [1.0, -0.7], [-0.9, -0.3]])
    foot_z = np.array([0.05, 0.20, 0.047, 0.30])
    geom_xpos = np.zeros((ngeom, 3), dtype=np.float32)
    geom_xpos[env._foot_geom_ids, :2] = foot_xy
    geom_xpos[env._foot_geom_ids, 2] = foot_z
    base_xy = np.array([0.6, -0.2])
    qpos = np.zeros(env.mj_model.nq, dtype=np.float32)
    qpos[0:2] = base_xy
    qpos[2] = 0.30
    data = SimpleNamespace(geom_xpos=jp.asarray(geom_xpos), qpos=jp.asarray(qpos))

    exp_h = h(foot_xy[:, 0], foot_xy[:, 1])
    exp_contact = (foot_z - exp_h) < FOOT_RADIUS + 0.005
    np.testing.assert_array_equal(np.array(env._foot_contact(data)), exp_contact)
    np.testing.assert_allclose(
        np.array(env._foot_clearance(data)), foot_z - FOOT_RADIUS - exp_h, atol=1e-5
    )
    np.testing.assert_allclose(
        float(env._base_height(data)), 0.30 - h(base_xy[0], base_xy[1]), atol=1e-5
    )


def test_flat_helpers_are_world_z():
    """With terrain disabled the helpers return the flat world-z expressions,
    so the flat reward/termination path is unchanged."""
    flat = wojtek_env.WojtekJoystick()
    assert flat._terrain_enabled is False
    ngeom = flat.mj_model.ngeom
    geom_xpos = np.zeros((ngeom, 3), dtype=np.float32)
    geom_xpos[flat._foot_geom_ids, 2] = np.array([0.05, 0.20, 0.047, 0.30])
    qpos = np.zeros(flat.mj_model.nq, dtype=np.float32)
    qpos[2] = 0.123
    data = SimpleNamespace(geom_xpos=jp.asarray(geom_xpos), qpos=jp.asarray(qpos))
    np.testing.assert_allclose(
        np.array(flat._foot_clearance(data)),
        np.array([0.05, 0.20, 0.047, 0.30]) - FOOT_RADIUS,
        atol=1e-6,  # float32 device math vs float64 reference
    )
    assert float(flat._base_height(data)) == pytest.approx(0.123)


# -- 3. Curriculum transition (pure function) ---------------------------------

N_ROWS = 10
TILE = 3.0
DEMOTE = 0.5


def _curr(level, walked, commanded, key=0):
    lvl, _ = terrain_env.curriculum_step(
        jp.int32(level), jp.float32(walked), jp.float32(commanded),
        jax.random.PRNGKey(key), N_ROWS, TILE, DEMOTE,
    )
    return int(lvl)


def test_curriculum_promote_on_crossing():
    # walked past half a tile edge, commanded satisfied -> +1
    assert _curr(2, walked=2.0, commanded=3.0) == 3


def test_curriculum_demote_on_short_distance():
    # covered < half the commanded distance, didn't cross -> -1
    assert _curr(5, walked=0.2, commanded=3.0) == 4


def test_curriculum_stand_is_neutral():
    # a standing episode clears neither threshold
    assert _curr(5, walked=0.0, commanded=0.0) == 5
    assert _curr(5, walked=0.01, commanded=0.0) == 5


def test_curriculum_clips_at_floor():
    assert _curr(0, walked=0.1, commanded=3.0) == 0


def test_curriculum_hold_at_ceiling_without_promote():
    # at the top level but not crossing -> hold (no promote, so no random row)
    assert _curr(N_ROWS - 1, walked=0.8, commanded=1.0) == N_ROWS - 1


def test_curriculum_max_level_random_respawn():
    """Promoting from the top level respawns on a uniformly random row (not a
    clip to the top), so easy terrain is revisited."""
    keys = jax.random.split(jax.random.PRNGKey(0), 200)

    def one(k):
        lvl, _ = terrain_env.curriculum_step(
            jp.int32(N_ROWS - 1), jp.float32(2.0), jp.float32(3.0),
            k, N_ROWS, TILE, DEMOTE,
        )
        return lvl

    rows = np.array(jax.vmap(one)(keys))
    assert rows.min() >= 0 and rows.max() < N_ROWS
    assert len(np.unique(rows)) > 3  # genuinely random, not pinned to the top
    assert np.any(rows < N_ROWS - 1)


# -- 4. Env integration: reset + steps on the terrain scene -------------------


def test_terrain_env_actor_obs_still_54(terrain_env_inst):
    state = jax.jit(terrain_env_inst.reset)(jax.random.PRNGKey(0))
    assert state.obs["state"].shape == (wojtek_env.OBS_SIZE,)
    assert state.obs["privileged_state"].shape == (wojtek_env.PRIVILEGED_SIZE,)


def test_terrain_reset_spawns_on_pad(terrain_env_inst):
    env = terrain_env_inst
    origins = np.array(env._terrain_origin_xy)
    pads = np.array(env._terrain_pad_h)
    for seed in range(6):
        state = jax.jit(env.reset)(jax.random.PRNGKey(seed))
        info = state.info
        lvl = int(info["terrain_level"])
        tt = int(info["terrain_type"])
        assert 0 <= lvl < env._terrain_n_rows
        assert 0 <= tt < env._terrain_n_types
        xy = np.array(state.data.qpos[0:2])
        # within the jitter box of the tile's pad centre
        assert np.all(np.abs(xy - origins[lvl, tt]) <= env._terrain_pad_jitter + 1e-5)
        z = float(state.data.qpos[2])
        assert z == pytest.approx(pads[lvl, tt] + float(info["spawn_height"]), abs=1e-5)


def test_terrain_step_finite(terrain_env_inst):
    state = jax.jit(terrain_env_inst.reset)(jax.random.PRNGKey(0))
    step = jax.jit(terrain_env_inst.step)
    for _ in range(5):
        state = step(state, jp.zeros(12))
    assert np.isfinite(float(state.reward))
    assert np.all(np.isfinite(np.array(state.obs["state"])))
    assert np.all(np.isfinite(np.array(state.obs["privileged_state"])))
    assert float(state.metrics["terrain_level_per_step"]) == float(
        state.info["terrain_level"]
    )


def test_terrain_initial_level_in_lower_half(terrain_env_inst):
    env = terrain_env_inst
    keys = jax.random.split(jax.random.PRNGKey(1), 32)
    levels = np.array(jax.jit(jax.vmap(env.reset))(keys).info["terrain_level"])
    init_rows = max(1, round(env._terrain_n_rows * env._terrain_init_level_frac))
    assert levels.max() < init_rows
    assert levels.min() >= 0


# -- 5. Curriculum auto-reset wrapper -----------------------------------------


def test_wrapper_teleports_to_pads_and_bounds_levels(terrain_env_inst):
    env = terrain_env_inst
    wrapped = wrap_for_terrain_brax_training(env, episode_length=3, action_repeat=1)
    assert isinstance(wrapped, TerrainAutoResetWrapper)
    rng = jax.random.split(jax.random.PRNGKey(0), 4)
    state = jax.jit(wrapped.reset)(rng)
    step = jax.jit(wrapped.step)

    origins = np.array(env._terrain_origin_xy).reshape(-1, 2)
    saw_done = False
    for _ in range(10):
        state = step(state, jp.zeros((4, 12)))
        done = np.array(state.done).astype(bool)
        levels = np.array(state.info["terrain_level"])
        assert np.all((levels >= 0) & (levels < env._terrain_n_rows))
        assert np.all(np.isfinite(np.array(state.obs["state"])))
        if done.any():
            saw_done = True
            xy = np.array(state.data.qpos[done, 0:2])
            dist = np.linalg.norm(xy[:, None, :] - origins[None, :, :], axis=-1).min(1)
            # teleported onto some tile's pad (centre + jitter)
            assert np.all(dist <= env._terrain_pad_jitter + 1e-4)
    assert saw_done  # episode_length=3 must have forced truncation dones


def test_flat_env_uses_stock_wrapper():
    """A flat env carries no terrain state, so train.py leaves it on brax's
    stock wrap_for_brax_training (the terrain wrapper is never built)."""
    from mujoco_playground import wrapper as mp_wrapper

    flat = wojtek_env.WojtekJoystick()
    assert getattr(flat, "_terrain_enabled", False) is False
    state = jax.jit(flat.reset)(jax.random.PRNGKey(0))
    assert "terrain_level" not in state.info
    assert "terrain_rng" not in state.info
    # the stock stack still resets/steps a flat env
    wrapped = mp_wrapper.wrap_for_brax_training(flat, episode_length=5)
    s = jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(0), 2))
    s = jax.jit(wrapped.step)(s, jp.zeros((2, 12)))
    assert np.all(np.isfinite(np.array(s.obs["state"])))


# -- 6. Missing-assets error path ---------------------------------------------


def test_missing_terrain_assets_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "TERRAIN_SCENE_XML", tmp_path / "nope_scene.xml")
    monkeypatch.setattr(paths, "TERRAIN_SPEC_JSON", tmp_path / "nope_spec.json")
    monkeypatch.setattr(paths, "TERRAIN_LOOKUP_NPZ", tmp_path / "nope_lookup.npz")
    cfg = wojtek_env.default_config()
    cfg.terrain.enable = True
    with pytest.raises(FileNotFoundError, match="build-terrain"):
        wojtek_env.WojtekJoystick(cfg)
