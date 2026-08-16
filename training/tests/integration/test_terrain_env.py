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
from wojtek_rl.base import BASE_CONTACT_TOL
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
    lvl, _, _ = terrain_env.curriculum_step(
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


def test_curriculum_first_step_holds_the_level_exactly():
    """steps_lived is 0 only before the first step has run, where
    commanded_dist is 0 too, so the projected threshold is a guarded 0/0.
    The observable contract is that such an episode holds its level exactly:
    no demote off a degenerate threshold, no promote. (The returned level is
    an int32 by construction -- a nan can only reach the boolean comparisons
    -- so "stays in range" would hold even with the guard deleted; equality
    is the assertion that means something.)"""
    for level in (0, 5, N_ROWS - 1):
        assert _curr(level, walked=0.0, commanded=0.0, steps_lived=0) == level
        # a small drift below the promote threshold must not demote either
        assert _curr(level, walked=0.3, commanded=0.0, steps_lived=0) == level


def test_curriculum_max_level_random_respawn():
    """Promoting from the top level respawns on a uniformly random row (not a
    clip to the top), so easy terrain is revisited."""
    keys = jax.random.split(jax.random.PRNGKey(0), 200)

    def one(k):
        lvl, _, _ = terrain_env.curriculum_step(
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


def test_base_terrain_contact_reads_the_lowest_corner(terrain_env_inst, monkeypatch):
    """The base collides as a chessboard of small boxes (ce9a464). The
    diagnostic has to use each box's extent along world z rather than its
    centre height. A chessboard cell is taller than it is wide (hz > hx),
    so at the same centre height a level cell touches the surface while a
    cell rotated onto its end still clears it."""
    env = terrain_env_inst
    flat = jp.zeros_like(env._terrain.lookup)  # surface at z = 0 everywhere
    monkeypatch.setattr(
        env, "_terrain", dataclasses.replace(env._terrain, lookup=flat)
    )
    hx, _, hz = np.array(env._base_geom_half)[0]
    assert hz > hx + BASE_CONTACT_TOL, "the orientation contrast needs hz > hx"
    level = np.eye(3)
    # Rotated 90 degrees about y, world z runs along the box's x axis, so
    # the reach below the centre is hx instead of hz.
    on_end = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])

    def contact(z_all, z_one=None, rot_one=level):
        """Place every base cell level at z_all, with one cell optionally
        moved or rotated."""
        ngeom = env.mj_model.ngeom
        xpos = np.zeros((ngeom, 3), dtype=np.float32)
        xmat = np.zeros((ngeom, 3, 3), dtype=np.float32)
        ids = np.asarray(env._base_geom_ids)
        xpos[ids, 2] = z_all
        xmat[ids] = level
        if z_one is not None:
            xpos[ids[0], 2] = z_one
            xmat[ids[0]] = rot_one
        data = SimpleNamespace(geom_xpos=jp.asarray(xpos), geom_xmat=jp.asarray(xmat))
        return bool(env._base_terrain_contact(data))

    assert not contact(hz + 0.05)  # every cell well clear
    assert contact(hz + 0.05, z_one=hz + 0.005)  # one cell down is contact
    # The orientation contrast, both at centre height hz. A level cell
    # reaches hz below its centre and touches. The same cell on its end
    # reaches only hx and stays clear.
    assert contact(hz + 0.05, z_one=hz)
    assert not contact(hz + 0.05, z_one=hz, rot_one=on_end)


# -- 5. Curriculum auto-reset wrapper -----------------------------------------


def test_wrapper_teleports_to_pads_and_bounds_levels(terrain_env_inst):
    env = terrain_env_inst
    wrapped = wrap_for_terrain_brax_training(env, episode_length=3, action_repeat=1)
    assert isinstance(wrapped, TerrainAutoResetWrapper)
    rng = jax.random.split(jax.random.PRNGKey(0), 4)
    state = jax.jit(wrapped.reset)(rng)
    step = jax.jit(wrapped.step)

    origins = np.array(env._terrain.origin_xy)  # (rows, types, 2)
    pads = np.array(env._terrain.pad_h)
    saw_done = False
    for _ in range(10):
        state = step(state, jp.zeros((4, 12)))
        done = np.array(state.done).astype(bool)
        levels = np.array(state.info["terrain_level"])
        assert np.all((levels >= 0) & (levels < env._terrain.n_rows))
        assert np.all(np.isfinite(np.array(state.obs["state"])))
        if done.any():
            saw_done = True
            # The pad of the tile the wrapper SAYS it moved to -- the
            # post-transition level and the env's own type -- not just some
            # pad. Indexed on [level, type], this is what catches a respawn
            # drawn from the pre-transition level.
            lvl = levels[done]
            ttype = np.array(state.info["terrain_type"])[done]
            xy = np.array(state.data.qpos[done, 0:2])
            # Bounded per axis, not by a Euclidean radius: the jitter is drawn
            # on a square, so a corner draw is jitter*sqrt(2) from the centre
            # and a radial bound rejects it. Re-running this loop over seeds
            # 0..39 hits 0.1968 against a 0.15 jitter; it passed only because
            # PRNGKey(0) happened to land inside the inscribed circle.
            off = np.abs(xy - origins[lvl, ttype])
            assert np.all(off <= env._terrain.pad_jitter + 1e-4), off
            # and the base-height write really happened, at the new tile's pad
            z = np.array(state.data.qpos[done, 2])
            spawn_h = np.array(state.info["spawn_height"])[done]
            np.testing.assert_allclose(z, pads[lvl, ttype] + spawn_h, atol=1e-4)
    assert saw_done  # episode_length=3 must have forced truncation dones


def test_base_contact_metrics_reach_the_brax_episode_accumulator(terrain_env_inst):
    """brax's EpisodeWrapper builds its per-episode accumulator from the env's
    metric keys at reset and indexes it by key on every step, so a metric added
    above that wrapper raises KeyError on the first step. These two are added in
    the env, below it, which is what makes them land in `episode/`."""
    env = terrain_env_inst
    wrapped = wrap_for_terrain_brax_training(env, episode_length=3, action_repeat=1)
    state = jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(0), 2))
    names = ("base_contact_alive_per_step", "base_contact_at_done")
    assert all(n in state.metrics for n in names)
    assert all(n in state.info["episode_metrics"] for n in names)

    step = jax.jit(wrapped.step)
    for _ in range(4):
        state = step(state, jp.zeros((2, 12)))
        for n in names:
            assert np.all(np.isfinite(np.array(state.metrics[n])))
            assert np.all(np.array(state.info["episode_metrics"][n]) >= 0.0)
    # A live env is never counted as terminated and vice versa: the two are
    # disjoint by construction, which is what makes them readable apart.
    both = np.array(state.metrics["base_contact_alive_per_step"]) * np.array(
        state.metrics["base_contact_at_done"]
    )
    assert not np.any(both)


def _drive_to_a_done(env, episode_length, walked, n_envs=2, level=1, key=3):
    """Two steps through the wrapper, ending on a fall done, with `walked`
    metres between the spawn and where the base ends up.

    `spawn_xy` is what the wrapper measures distance from and the env never
    touches it mid-episode, so writing it is how a specific walked distance is
    staged; `last_xy` would be overwritten by the next step. The fall is staged
    by rolling the base 90 degrees, past `fall.max_tilt_gz` -- one control
    step of physics cannot right that. (Staging a height fall by dropping the
    base into the ground injects a violent penetration impulse, and whether
    the ejection leaves the base above or below `min_height` after 0.02 s is
    knife-edge: it flipped when the arena apron changed the scene's contact
    ordering.)
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
    roll90 = jp.array([jp.cos(jp.pi / 4), jp.sin(jp.pi / 4), 0.0, 0.0])
    qpos = state.data.qpos.at[:, 3:7].set(roll90)
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
    centre, spawn, yaw, pad_h = terrain_scan.spawn_table(env, cell)
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


COUNTERS = ("crossings", "fell", "counted", "steps")


def test_the_batched_rollout_matches_the_per_cell_one(monkeypatch):
    """The terrain scan's two rollouts, on the same runs.

    `--per-cell` dispatches 64 worlds per cell per speed and is the reference;
    the default folds the cell axis into the batch and dispatches once per
    speed. This is the pair actually built against MJX: three cells in one
    192-world batch against three 64-world dispatches, with each cell on its own
    tile and its own deadlines, and every run's outcome compared. The key
    streams and the batch layout are pinned exactly, and much more cheaply, in
    tests/unit/test_terrain_scan.py.

    The flat scene, because the heightfield one costs about four minutes per
    program to trace and compile and nothing under test here is
    terrain-specific; `spawn_table` reads nothing but the arena's tile
    origins, so a stand-in arena stands in for the eval one.

    The per-step metrics are NOT compared, and the reason is not this code: one
    mjx step is not batch-shape invariant on the jax CPU backend. Identical
    states stepped as 64 worlds and as the first 64 of 192 part in the sixth
    decimal of qpos -- identical lanes inside one batched call do too, in
    blocks of 32 -- and a legged robot amplifies that by roughly 10x per
    step. The same comparison on the warp backend is pending a GPU run.
    """
    from wojtek_rl import terrain_scan, terrain_suite

    env = wojtek_env.WojtekJoystick()
    runs = terrain_suite.RUNS_PER_CELL_SPEED
    cells = [
        terrain_suite.Cell(
            name=f"probe_{ttype}", terrain_type=ttype, difficulty=0.5,
            row=row, bar=None,
        )
        for row, ttype in enumerate(terrain.TYPES[:3])
    ]
    origin = np.zeros((len(cells), len(terrain.TYPES), 2), dtype=np.float32)
    for row, _ in enumerate(cells):
        origin[row, row] = (4.0 * row, 0.0)
    arena = SimpleNamespace(
        _terrain=SimpleNamespace(
            origin_xy=origin,
            pad_h=np.zeros((len(cells), len(terrain.TYPES)), dtype=np.float32),
        )
    )

    def inf(obs, key):
        """Stand-in policy, deterministic in (obs, key)."""
        state = obs["state"]
        noise = jax.random.uniform(
            key, (state.shape[0], env.action_size), minval=-1.0, maxval=1.0
        )
        return 0.5 * jp.tanh(state[:, : env.action_size]) + 0.5 * noise, {}

    # The real settle window is 50 steps and this rollout is 6, so without this
    # nothing would be measured at all. Read when the rollout is traced, below.
    monkeypatch.setattr(terrain_suite, "SETTLE_STEPS", 0)
    budget = 6
    speed, height = 0.4, terrain_scan.COMMAND_HEIGHT
    # A different deadline per cell, so runs stop at different steps inside the
    # one batch and a slice taken from the wrong cell lands on the wrong count.
    deadlines = [np.full(runs, 2 + row, np.int32) for row, _ in enumerate(cells)]
    tables = [terrain_scan.spawn_table(arena, c) for c in cells]
    centre, spawn, yaw, pad_h = [np.concatenate(x) for x in zip(*tables)]

    reference = terrain_scan.make_cell_runner(env, inf)
    want = []
    for row, cell in enumerate(cells):
        centres, spawns, yaws, pads = tables[row]
        want.append(jax.tree.map(np.asarray, reference(
            terrain_scan.cell_key(cell), centres, spawns, pads, yaws,
            speed, height, deadlines[row], budget=budget,
        )))

    batched = terrain_scan.make_batch_runner(env, inf, len(cells))
    got = jax.tree.map(np.asarray, batched(
        jp.stack([terrain_scan.cell_key(c) for c in cells]),
        centre, spawn, pad_h, yaw, speed, height,
        np.concatenate(deadlines), budget=budget,
    ))

    for row, cell in enumerate(cells):
        part = slice(row * runs, (row + 1) * runs)
        for key in COUNTERS:
            np.testing.assert_array_equal(
                got[key][part], want[row][key], err_msg=f"{cell.name} {key}"
            )
        # each cell ran to its own deadline, and every run was measured
        assert np.all(got["steps"][part] == 2 + row)
        assert np.all(got["counted"][part] > 0)


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


def test_spawn_scatter_must_fit_the_pad(small_arena):
    """The eval arena's pads are 0.40 m by design, which the default 0.15 m
    pad_jitter plus the 0.36 m standing footprint does not fit -- a training
    run pointed at it would spawn with feet on the features. The guard turns
    that into a message naming both numbers."""
    cfg = wojtek_env.default_config()
    cfg.terrain.enable = True
    cfg.terrain.arena = TEST_ARENA
    cfg.terrain.pad_jitter = 0.3  # 0.3 + 0.36 > the test arena's 0.6 m pad
    with pytest.raises(ValueError, match="pad_jitter"):
        wojtek_env.WojtekJoystick(cfg)


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
    geom-heightfield pair, so 29 geoms put 116 contacts in the pool before a
    single box is touched. The terrain presets carry that number as
    sim.naconmax_per_env."""
    n = terrain_env_inst._count_ground_colliding_geoms()
    assert n == 29, n  # 9 chessboard base cells + 4 feet + 16 per-leg proxies
    assert 4 * n == 116
    # the same count on the flat scene: it keys on body, so neither the floor
    # plane nor the generator's terrain geoms are mistaken for robot geoms
    assert wojtek_env.WojtekJoystick()._count_ground_colliding_geoms() == n
    # and the flat default really is below the floor
    assert wojtek_env.default_config().sim.naconmax_per_env < 4 * n
