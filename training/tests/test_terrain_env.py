"""Tests for training on the terrain arena.

What is covered:

- The height lookup the env uses on the GPU returns the same surface as the
  numpy one the arena was built with.
- Height, foot contact and foot clearance are measured from the terrain surface
  under the robot, not from z = 0.
- Moving an env to a harder or easier tile after each episode: the rule itself,
  and the same rule driven through a real `env.step`.
- Spawning on a tile, and the deterministic spawn the terrain scan uses.
- Refusing to start when the generated arena files are missing or were built by
  an older version of the generator.

The env and wrapper tests share one small 3-row arena, written to the `test`
file set so a test run can never overwrite the arena a policy trained on. Small
because building the MuJoCo model is the slow part.
"""

import dataclasses
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
TEST_ARENA = "test"


@pytest.fixture(scope="module")
def small_arena():
    """A small arena written to the `test` file set, so the env loads it from
    disk the way training does without touching the arena a policy trained on.
    Deterministic, and removed afterwards."""
    a = terrain.generate(seed=0, n_rows=SMALL_ROWS)
    build_terrain.write_arena(a, TEST_ARENA)
    yield a
    for p in paths.terrain_paths(TEST_ARENA).values():
        p.unlink(missing_ok=True)


@pytest.fixture(scope="module")
def terrain_config():
    cfg = wojtek_env.default_config()
    cfg.terrain.enable = True
    cfg.terrain.arena = TEST_ARENA
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
    nrow, ncol = env._terrain.lookup.shape
    xs = env._terrain.x_min + env._terrain.cell_x * np.arange(ncol)
    ys = env._terrain.y_min + env._terrain.cell_y * np.arange(nrow)
    a, b = 0.1, -0.05
    synth = (a * xs[None, :] + b * ys[None, :].T).astype(np.float32)
    # Arena is frozen, so swap the whole thing for a copy with the synthetic grid.
    monkeypatch.setattr(
        env, "_terrain", dataclasses.replace(env._terrain, lookup=jp.asarray(synth))
    )

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
EPISODE = 1000


def _curr(level, walked, commanded, key=0, steps_lived=EPISODE, episode=EPISODE):
    """`steps_lived == episode` by default: a timeout, where the projection
    factor is 1 and the threshold is the pre-projection one."""
    lvl, _ = terrain_env.curriculum_step(
        jp.int32(level), jp.float32(walked), jp.float32(commanded),
        jp.int32(steps_lived), episode,
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


def test_curriculum_timeout_threshold_is_unprojected():
    """A timeout lived the whole episode, so the projection factor is 1 and the
    rule is exactly what it was before the projection existed."""
    # 1.2 walked of a 2.0 commanded distance: above the 1.0 threshold, and
    # below the 1.5 half-tile crossing, so the level holds
    assert _curr(5, walked=1.2, commanded=2.0) == 5
    # 0.9 is below the same threshold, so it demotes
    assert _curr(5, walked=0.9, commanded=2.0) == 4


def test_curriculum_early_fall_demotes():
    """A fall at step 50 of 1000 gets a threshold twenty times the distance
    commanded so far, so almost any fall demotes -- the escape valve legged_gym
    has. Without the projection this same episode holds its level."""
    walked, commanded = 0.2, 0.15  # 50 steps at 0.15 m/s commanded
    assert _curr(5, walked, commanded, steps_lived=50, episode=1000) == 4
    # the unprojected threshold is 0.5 * 0.15 = 0.075, below the 0.2 walked
    assert _curr(5, walked, commanded, steps_lived=1000, episode=1000) == 5


def test_curriculum_promote_wins_over_demote():
    """Both conditions firing is a promote, matching legged_gym's
    `move_down * ~move_up`."""
    # crossed half a tile, and the projected threshold is far above that
    lvl = _curr(5, walked=2.0, commanded=0.02, steps_lived=2, episode=1000)
    assert lvl == 6
    # the same episode without the crossing demotes, so both really do fire
    assert _curr(5, walked=1.4, commanded=0.02, steps_lived=2, episode=1000) == 4


def test_curriculum_first_step_division_is_guarded():
    """steps_lived can be 0 before the first step has run; the level must stay
    finite rather than come back as a nan-propagated garbage index."""
    for level in (0, 5, N_ROWS - 1):
        assert 0 <= _curr(level, walked=0.0, commanded=0.0, steps_lived=0) < N_ROWS


def test_curriculum_max_level_random_respawn():
    """Promoting from the top level respawns on a uniformly random row (not a
    clip to the top), so easy terrain is revisited."""
    keys = jax.random.split(jax.random.PRNGKey(0), 200)

    def one(k):
        lvl, _ = terrain_env.curriculum_step(
            jp.int32(N_ROWS - 1), jp.float32(2.0), jp.float32(3.0),
            jp.int32(EPISODE), EPISODE,
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
    origins = np.array(env._terrain.origin_xy)
    pads = np.array(env._terrain.pad_h)
    for seed in range(6):
        state = jax.jit(env.reset)(jax.random.PRNGKey(seed))
        info = state.info
        lvl = int(info["terrain_level"])
        tt = int(info["terrain_type"])
        assert 0 <= lvl < env._terrain.n_rows
        assert 0 <= tt < env._terrain.n_types
        xy = np.array(state.data.qpos[0:2])
        # within the jitter box of the tile's pad centre
        assert np.all(np.abs(xy - origins[lvl, tt]) <= env._terrain.pad_jitter + 1e-5)
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
    init_rows = max(1, round(env._terrain.n_rows * env._terrain.init_level_frac))
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

    origins = np.array(env._terrain.origin_xy).reshape(-1, 2)
    saw_done = False
    for _ in range(10):
        state = step(state, jp.zeros((4, 12)))
        done = np.array(state.done).astype(bool)
        levels = np.array(state.info["terrain_level"])
        assert np.all((levels >= 0) & (levels < env._terrain.n_rows))
        assert np.all(np.isfinite(np.array(state.obs["state"])))
        if done.any():
            saw_done = True
            xy = np.array(state.data.qpos[done, 0:2])
            # Teleported onto some tile's pad (centre + jitter). Bounded per
            # axis, not by a Euclidean radius: the jitter is drawn on a square,
            # so a corner draw is jitter*sqrt(2) from the centre and a radial
            # bound rejects it. Re-running this loop over seeds 0..39 hits
            # 0.1968 against a 0.15 jitter; it passed only because PRNGKey(0)
            # happened to land inside the inscribed circle every time.
            off = np.abs(xy[:, None, :] - origins[None, :, :])
            nearest = off.max(axis=-1).argmin(axis=-1)
            per_axis = off[np.arange(len(xy)), nearest]
            assert np.all(per_axis <= env._terrain.pad_jitter + 1e-4), per_axis
    assert saw_done  # episode_length=3 must have forced truncation dones


def _drive_to_a_done(env, episode_length, walked, n_envs=2, level=1, key=3):
    """Two steps through the wrapper, ending on a fall done, with `walked`
    metres between the spawn and where the base ends up.

    `spawn_xy` is what the wrapper measures distance from and the env never
    touches it mid-episode, so writing it is how a specific walked distance is
    staged; `last_xy` would be overwritten by the next step. The fall is staged
    by dropping the base to just above the local surface, below
    `fall.min_height` -- that is the termination the demote rule exists for.
    """
    wrapped = wrap_for_terrain_brax_training(env, episode_length=episode_length)
    step = jax.jit(wrapped.step)
    state = jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(key), n_envs))
    # A live forward command, so commanded_dist accumulates at a known rate.
    state.info["command"] = jp.tile(jp.array([0.5, 0.0, 0.0, 0.125]), (n_envs, 1))
    state = step(state, jp.zeros((n_envs, 12)))
    assert not np.any(np.array(state.done)), "the first step must not terminate"
    state.info["terrain_level"] = jp.full((n_envs,), level, dtype=jp.int32)
    xy = state.data.qpos[:, 0:2]
    state.info["spawn_xy"] = xy + jp.array([walked, 0.0])
    qpos = state.data.qpos.at[:, 2].set(env._terrain.height(xy) + 0.01)
    state = state.replace(data=state.data.replace(qpos=qpos))
    state = step(state, jp.zeros((n_envs, 12)))
    assert np.all(np.array(state.done) == 1.0), "the staged fall must terminate"
    return np.array(state.info["terrain_level"])


def test_wrapper_demotes_after_an_early_fall(terrain_env_inst):
    """The behaviour the projected demote threshold adds, through env.step.

    Two steps of a 0.5 m/s command accumulate 0.02 m of commanded distance; the
    fall lands at step 2 of 1000, so the threshold is 0.5 * 0.02 * 1000/2 =
    5 m and the 0.02 m walked is far below it. Unprojected, the threshold would
    be 0.01 m and this episode would hold its level.
    """
    levels = _drive_to_a_done(terrain_env_inst, episode_length=1000, walked=0.02)
    assert np.all(levels == 0), levels


def test_wrapper_promote_beats_demote_through_step(terrain_env_inst):
    """Same staged fall, but the base is 2 m from its spawn: a crossing. Both
    conditions fire (the projected threshold is 5 m) and promotion wins."""
    levels = _drive_to_a_done(terrain_env_inst, episode_length=1000, walked=2.0)
    assert np.all(levels == 2), levels


def test_domain_randomization_composes_with_the_curriculum_wrapper(terrain_env_inst):
    """The DR wrapper actually resets and steps under the terrain auto-reset.

    Nothing else in the repo executes the DR wrapper -- `make_domain_randomize`
    is tested as a pure function -- and `domain_rand: true` is the default
    training path, so this composition is what a real run does and what nothing
    covered. The rng binding mirrors brax's ppo.train, which partials the rng in
    before handing the callable to wrap_env_fn.
    """
    import functools

    from wojtek_rl.randomize import make_domain_randomize

    n = 2
    randomization_fn = functools.partial(
        make_domain_randomize(
            terrain_env_inst.mj_model,
            {"foot_friction": {"enable": True}, "joint_gains": {"enable": True}},
        ),
        rng=jax.random.split(jax.random.PRNGKey(0), n),
    )
    wrapped = wrap_for_terrain_brax_training(
        terrain_env_inst, episode_length=3, randomization_fn=randomization_fn
    )
    step = jax.jit(wrapped.step)
    state = jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(1), n))
    saw_done = False
    for _ in range(8):
        state = step(state, jp.zeros((n, 12)))
        saw_done = saw_done or bool(np.any(np.array(state.done)))
        assert np.all(np.isfinite(np.array(state.obs["state"])))
        levels = np.array(state.info["terrain_level"])
        assert np.all((levels >= 0) & (levels < terrain_env_inst._terrain.n_rows))
    # the teleport-on-done path ran with a per-env randomized model
    assert saw_done


def test_scan_reset_places_a_run_on_its_tile(terrain_env_inst):
    """The terrain scan's deterministic spawn, on the env the scan would use.

    Lives here rather than in test_terrain_scan.py because it needs a built
    terrain env, and this module already has one. Everything else about the scan
    is pure and tested there.
    """
    from wojtek_rl import terrain, terrain_scan, terrain_suite

    env = terrain_env_inst
    cell = terrain_suite.Cell(
        name="probe", terrain_type="pyramid_stairs", difficulty=0.5, row=1, bar=None
    )
    centre, spawn, yaw, pad_h = terrain_scan._spawn_table(env, cell)
    # the tile the cell names, from the env's own spawn table
    j = terrain.TYPES.index(cell.terrain_type)
    np.testing.assert_allclose(centre[0], np.array(env._terrain.origin_xy)[cell.row, j])
    assert pad_h[0] == pytest.approx(float(np.array(env._terrain.pad_h)[cell.row, j]))

    run = terrain_suite.COURSE[5]
    command = jp.array([0.4, 0.0, 0.0, terrain_scan.COMMAND_HEIGHT])
    # Eager, not jitted: the env is a Python object, so it can only ever be a
    # static argument. make_cell_runner binds it in a closure for exactly that
    # reason (functools.partial(scan_reset, env)).
    state = terrain_scan.scan_reset(
        env, jax.random.PRNGKey(0), jp.asarray(spawn[5]), pad_h[5],
        jp.float32(run.yaw), command,
    )

    qpos = np.array(state.data.qpos)
    np.testing.assert_allclose(qpos[0:2], spawn[5], atol=1e-6)
    # base at the commanded height above the tile's pad, not above world zero
    assert qpos[2] == pytest.approx(pad_h[5] + terrain_scan.COMMAND_HEIGHT, abs=1e-6)
    assert float(env._base_height(state.data)) == pytest.approx(
        terrain_scan.COMMAND_HEIGHT, abs=2e-3
    )
    # facing the course heading
    w, x, y, z = qpos[3:7]
    assert np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)) == pytest.approx(
        run.yaw, abs=1e-5
    )
    # joints on the commanded height's stance anchor, with none of reset's noise
    np.testing.assert_allclose(
        qpos[np.array(env._qadr)], np.array(env._height_ctrl(command[3])), atol=1e-6
    )
    np.testing.assert_allclose(np.array(state.data.qvel), 0.0, atol=0.0)

    # the OBSERVED command is the course's, not the one env.reset sampled
    np.testing.assert_allclose(np.array(state.info["command"]), np.array(command))
    names = env.actor_obs_names
    catalog = env._obs_catalog(state.data, state.info)
    offset = sum(catalog[n].shape[0] for n in names[: names.index("command")])
    np.testing.assert_allclose(
        np.array(state.obs["state"])[offset : offset + 4], np.array(command), atol=1e-6
    )


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


@pytest.mark.parametrize(
    "missing", ["scene", "hfield", "spec", "lookup"]
)
def test_missing_terrain_assets_raises(monkeypatch, tmp_path, missing):
    """Each of the four generated files, heightfield included. Without the
    heightfield in the check, a missing .bin is a raw MuJoCo compile error
    instead of the message that names build-terrain."""
    files = {}
    for role in ("scene", "hfield", "spec", "lookup"):
        p = tmp_path / f"terrain_{role}"
        if role != missing:
            p.touch()
        files[role] = p
    monkeypatch.setattr(paths, "terrain_paths", lambda kind="train": files)
    cfg = wojtek_env.default_config()
    cfg.terrain.enable = True
    with pytest.raises(FileNotFoundError, match="build-terrain") as excinfo:
        wojtek_env.WojtekJoystick(cfg)
    assert files[missing].name in str(excinfo.value)


def test_a_stale_arena_is_refused(monkeypatch, tmp_path, small_arena):
    """An arena built before the stair flight went from four steps to six has
    four-step stairs, is entirely self-consistent, and would train silently while
    run.json claimed six. The stair geometry is a code constant, so a mismatch is
    always staleness."""
    import json

    from wojtek_rl import build_terrain

    files = paths.terrain_paths(TEST_ARENA)
    build_terrain.write_arena(small_arena, TEST_ARENA)
    spec = json.loads(files["spec"].read_text())
    spec["n_steps"] = 4  # what the generator built before this change
    stale = tmp_path / "stale_spec.json"
    stale.write_text(json.dumps(spec))
    monkeypatch.setattr(
        paths, "terrain_paths", lambda kind="train": {**files, "spec": stale}
    )
    cfg = wojtek_env.default_config()
    cfg.terrain.enable = True
    cfg.terrain.arena = TEST_ARENA
    with pytest.raises(ValueError, match="different generator") as excinfo:
        wojtek_env.WojtekJoystick(cfg)
    assert "n_steps" in str(excinfo.value)
    assert "build-terrain" in str(excinfo.value)


def test_contact_floor_is_derived_from_the_model(terrain_env_inst):
    """The warp contact-budget warning fires against a floor computed from the
    robot's own collision set, not a rule of thumb. Warp allows four contacts per
    geom-heightfield pair, so 21 geoms put 84 contacts in the pool before a single
    box is touched -- which is why the flat default of 32 is not close."""
    n = terrain_env_inst._count_ground_colliding_geoms()
    assert n == 21, n  # base box + 4 feet + 16 per-leg proxies
    assert 4 * n == 84
    # the same count on the flat scene: it keys on body, so neither the floor
    # plane nor the generator's terrain geoms are mistaken for robot geoms
    assert wojtek_env.WojtekJoystick()._count_ground_colliding_geoms() == n
    # and the flat default really is below the floor
    assert wojtek_env.default_config().sim.naconmax_per_env < 4 * n
