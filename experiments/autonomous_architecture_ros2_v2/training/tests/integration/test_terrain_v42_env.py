"""The v4.2 terrain package end to end on a real (small) MJX env: feature
spawns, band promotion state, strikes, pinning, and the wrapper teleport.

One module-scoped env pays the compile; the arena is written to the `test`
file set with the v4.2 geometry (tread range, caps, summit pads, flat row)
so the build/config agreement checks run for real.
"""

import jax
import jax.numpy as jp
import numpy as np
import pytest

from wojtek_rl import build_terrain, paths, terrain, terrain_env, terrain_suite
from wojtek_rl import env as wojtek_env
from wojtek_rl.terrain_wrapper import wrap_for_terrain_brax_training

ROWS = 3
ARENA = "test"
TREADS = (0.25, 0.45)
CAPS = {"pyramid_stairs": 0.7, "inverted_pyramid_stairs": 0.7}
PAD = 0.3


@pytest.fixture(scope="module")
def v42_arena():
    a = terrain.generate(
        seed=0, n_rows=ROWS, flat_row=True, pad_radius=PAD,
        stair_platform_half=PAD, stair_tread=TREADS, type_caps=CAPS,
    )
    build_terrain.write_arena(a, ARENA)
    yield a
    for p in paths.terrain_paths(ARENA).values():
        p.unlink(missing_ok=True)


@pytest.fixture(scope="module")
def v42_env(v42_arena):
    cfg = wojtek_env.default_config()
    cfg.terrain.enable = True
    cfg.terrain.arena = ARENA
    cfg.terrain.flat_row = True
    cfg.terrain.spawn_mode = "feature"
    cfg.terrain.spawn_grace_sec = 1.0
    cfg.terrain.pinned_frac = 0.5
    cfg.terrain.demote_strikes = 3
    cfg.terrain.stair_tread_range = TREADS
    cfg.terrain.type_caps = dict(CAPS)
    cfg.terrain.pad_radius = PAD
    cfg.command.terrain_bias.enable = True
    cfg.no_progress.enable = True
    cfg.no_progress.terrain_grace_sec = 4.0
    cfg.no_progress.terrain_p_max_scale = 0.5
    return wojtek_env.WojtekJoystick(cfg)


def test_eval_deep_arena_builds_and_loads():
    """The deep measurement course must load through the real require
    checks, not just check_arena: the summit rule ties its platform to its
    pad, and a drift between builder and loader makes the arena unloadable
    (the review's finding 1). Writes the real eval_deep slot; the build is
    deterministic, so this refreshes rather than perturbs it."""
    arena = terrain.generate(**terrain_suite.deep_arena_kwargs())
    assert arena.spec.stair_platform_half == terrain.summit_platform_half(
        terrain_suite.EVAL_PAD_RADIUS
    )
    build_terrain.write_arena(arena, "eval_deep")
    cfg = wojtek_env.default_config().terrain
    cfg.enable = True
    cfg.arena = "eval_deep"
    cfg.pad_jitter = 0.0
    cfg.spawn_yaw = False
    loaded = terrain_env.load(cfg)  # raises on any geometry disagreement
    assert loaded.kind == "eval_deep"
    assert loaded.n_rows == len(terrain_suite.DIFFICULTIES)
    # A v4.2 run's terrain keys must not poison an eval_deep load either:
    # measurement arenas skip the build/config stair-geometry agreement.
    cfg.stair_tread_range = [0.25, 0.45]
    cfg.type_caps = {"pyramid_stairs": 0.7}
    terrain_env.load(cfg)


def test_measurement_env_pins_legacy_no_progress_patience(v42_env):
    """Finding 2: the run's terrain patience must not follow the policy
    onto the measurement course -- baselines were scanned at 2 s grace and
    full hazard. The training-kind env keeps the patience; an env on an
    eval arena resolves to none."""
    assert v42_env._npg_patience == (4.0, 0.5)
    cfg = wojtek_env.default_config()
    cfg.no_progress.enable = True
    cfg.no_progress.terrain_grace_sec = 4.0
    cfg.no_progress.terrain_p_max_scale = 0.5
    cfg.terrain.enable = True
    cfg.terrain.arena = "eval_deep"
    cfg.terrain.pad_jitter = 0.0
    cfg.terrain.spawn_yaw = False
    env = wojtek_env.WojtekJoystick(cfg)
    assert env._npg_patience == (0.0, 1.0)


def test_geometry_mismatch_is_refused(v42_arena):
    cfg = wojtek_env.default_config()
    cfg.terrain.enable = True
    cfg.terrain.arena = ARENA
    cfg.terrain.flat_row = True
    cfg.terrain.spawn_mode = "feature"
    # The arena on disk was built with treads/caps; a preset without them
    # must be refused, like a flat-row mismatch.
    with pytest.raises(ValueError, match="stair"):
        wojtek_env.WojtekJoystick(cfg)


def test_feature_reset_spawns_on_the_terrain(v42_env):
    env = v42_env
    origins = np.array(env._terrain.origin_xy)
    reach = env._terrain.tile_size / 2
    off_pad = 0
    for seed in range(12):
        state = jax.jit(env.reset)(jax.random.PRNGKey(seed))
        info = state.info
        lvl = int(info["terrain_level"])
        tt = int(info["terrain_type"])
        xy = np.array(state.data.qpos[0:2])
        r = np.max(np.abs(xy - origins[lvl, tt]))
        assert r <= reach  # on its own tile
        if lvl == 0:
            assert r <= env._terrain.pad_jitter + 1e-5  # flat row keeps pads
        elif r > env._terrain.pad_radius:
            off_pad += 1
        # Base height rides the terrain under the spawn, not the pad table.
        ground = float(env._terrain.height(jp.asarray(xy)))
        z = float(state.data.qpos[2])
        assert z == pytest.approx(ground + float(info["spawn_height"]), abs=2e-3)
        # Band-tracking state seeded at the spawn radius.
        assert float(info["cheby_min"]) == pytest.approx(r, abs=1e-5)
        assert float(info["cheby_max"]) == pytest.approx(r, abs=1e-5)
        assert int(info["curriculum_strikes"]) == 0
    assert off_pad > 0  # the draws genuinely leave the pad


def test_feature_step_finite_and_tracks_the_band(v42_env):
    env = v42_env
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    step = jax.jit(env.step)
    for _ in range(5):
        state = step(state, jp.zeros(12))
    assert np.isfinite(float(state.reward))
    assert np.all(np.isfinite(np.array(state.obs["state"])))
    assert float(state.info["cheby_min"]) <= float(state.info["cheby_max"])


def test_wrapper_respawn_pins_and_clears_band_state(v42_env):
    env = v42_env
    wrapped = wrap_for_terrain_brax_training(env, episode_length=3, action_repeat=1)
    n = 4
    rng = jax.random.split(jax.random.PRNGKey(0), n)
    state = jax.jit(wrapped.reset)(rng)
    step = jax.jit(wrapped.step)
    for _ in range(8):  # crosses at least two 3-step episode boundaries
        state = step(state, jp.zeros((n, 12)))
    levels = np.array(state.info["terrain_level"])
    # pinned_frac=0.5 of 4 envs: envs 0 and 1 sit on rungs 0 and 1.
    assert levels[0] == 0 and levels[1] == 1
    assert np.all(levels >= 0) and np.all(levels < env._terrain.n_rows)
    assert np.all(np.array(state.info["curriculum_strikes"]) >= 0)
    # The band state was re-seeded at the new spawns: min <= max, both finite.
    cmin = np.array(state.info["cheby_min"])
    cmax = np.array(state.info["cheby_max"])
    assert np.all(np.isfinite(cmin)) and np.all(cmin <= cmax + 1e-6)
    # tile_origin matches each env's current tile.
    origins = np.array(env._terrain.origin_xy)
    types = np.array(state.info["terrain_type"])
    want = origins[levels, types]
    np.testing.assert_allclose(np.array(state.info["tile_origin"]), want, atol=1e-5)
