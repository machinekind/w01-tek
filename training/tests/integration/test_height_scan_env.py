"""The height scan inside the env, the reward gate it feeds, and the
curriculum wrapper's episode-boundary handling.

The scan is the first observation that depends on where in the world the base
is, which is what the wrapper's cached-observation auto-reset assumed away.
The arena is the same small `test` file set the terrain env tests use.
"""

import jax
import jax.numpy as jp
import numpy as np
import pytest

from wojtek_rl import build_terrain, height_scan, paths, symmetry, terrain
from wojtek_rl import env as wojtek_env
from wojtek_rl.build_model import FOOT_RADIUS
from wojtek_rl.terrain_wrapper import wrap_for_terrain_brax_training

TEST_ARENA = "test"
TERRAIN_ROWS = 2  # plus the flat row, which becomes level 0

ACTOR_INCLUDE = (
    "gyro", "gravity", "joint_pos", "joint_vel", "last_act", "command",
    "height_scan",
)
PRIVILEGED = (
    "gyro", "gravity", "joint_pos", "joint_vel", "last_act", "command",
    "phase", "linvel", "foot_contact", "height_scan_clean",
)
ACTOR_SIZE = 71
CRITIC_SIZE = 86

GATE_FLOOR = 0.15
CLEARANCE = 0.04  # foot clearance staged for the gate probe


def _config(*, terrain_on=True, scan=True, dark=False):
    cfg = wojtek_env.default_config()
    cfg.gait.swing_height = 0.08
    cfg.reward.terrain_gate.enable = True
    cfg.reward.terrain_gate.floor = GATE_FLOOR
    cfg.symmetry.enable = True
    if scan:
        cfg.obs.state = tuple(cfg.obs.state) + ("height_scan",)
        cfg.obs.include = ACTOR_INCLUDE
        cfg.obs.privileged = PRIVILEGED
        cfg.height_scan.enable = True
        cfg.height_scan.dark = dark
        cfg.height_scan.corrupt.enable = True
    if terrain_on:
        cfg.terrain.enable = True
        cfg.terrain.arena = TEST_ARENA
        cfg.terrain.flat_row = True
    return cfg


@pytest.fixture(scope="module")
def arena():
    a = terrain.generate(seed=0, n_rows=TERRAIN_ROWS, flat_row=True)
    build_terrain.write_arena(a, TEST_ARENA)
    yield a
    for p in paths.terrain_paths(TEST_ARENA).values():
        p.unlink(missing_ok=True)


@pytest.fixture(scope="module")
def scan_env(arena):
    return wojtek_env.WojtekJoystick(_config())


@pytest.fixture(scope="module")
def flat_env():
    """The exporter path: an enabled scan with no terrain to sample."""
    return wojtek_env.WojtekJoystick(_config(terrain_on=False))


@pytest.fixture(scope="module")
def dark_env(arena):
    return wojtek_env.WojtekJoystick(_config(dark=True))


@pytest.fixture(scope="module")
def blind_env(arena):
    """Terrain without a scan: the wrapper's no-op path."""
    return wojtek_env.WojtekJoystick(_config(scan=False))


def _actor_slice(env):
    offset = 0
    for name in env.actor_obs_names:
        if name == "height_scan":
            return slice(offset, offset + height_scan.SIZE)
        offset += symmetry.COMPONENT_SIZES[name]
    raise AssertionError("actor observation has no height_scan")


def _critic_slice(env):
    offset = 0
    for name in env._config.obs.privileged:
        if name == "height_scan_clean":
            return slice(offset, offset + height_scan.SIZE)
        offset += symmetry.COMPONENT_SIZES[name]
    raise AssertionError("privileged observation has no height_scan_clean")


# -- 1. observation layout ----------------------------------------------------


def test_obs_layout_with_the_scan(scan_env):
    state = jax.jit(scan_env.reset)(jax.random.PRNGKey(0))
    assert state.obs["state"].shape == (ACTOR_SIZE,)
    assert state.obs["privileged_state"].shape == (CRITIC_SIZE,)
    assert np.all(np.isfinite(np.array(state.obs["state"])))
    assert np.all(np.isfinite(np.array(state.obs["privileged_state"])))
    # A live arena has to reach both blocks, or every assertion below about
    # zeros is vacuous.
    assert np.any(np.array(state.obs["privileged_state"])[_critic_slice(scan_env)])


def test_flat_scene_zeroes_both_scan_blocks(flat_env):
    state = jax.jit(flat_env.reset)(jax.random.PRNGKey(0))
    assert state.obs["state"].shape == (ACTOR_SIZE,)
    assert state.obs["privileged_state"].shape == (CRITIC_SIZE,)
    actor = np.array(state.obs["state"])[_actor_slice(flat_env)]
    critic = np.array(state.obs["privileged_state"])[_critic_slice(flat_env)]
    np.testing.assert_array_equal(actor, np.zeros(height_scan.SIZE))
    np.testing.assert_array_equal(critic, np.zeros(height_scan.SIZE))
    assert "scan_hold" not in state.info


def test_dark_blinds_the_actor_only(dark_env):
    state = jax.jit(dark_env.reset)(jax.random.PRNGKey(0))
    actor = np.array(state.obs["state"])[_actor_slice(dark_env)]
    critic = np.array(state.obs["privileged_state"])[_critic_slice(dark_env)]
    np.testing.assert_array_equal(actor, np.zeros(height_scan.SIZE))
    assert np.any(critic)


# -- 2. sample and hold -------------------------------------------------------


def test_scan_hold_only_moves_on_promote_ticks(scan_env):
    """The buffer the actor reads changes on the promote tick and nowhere
    else, and what lands there is the frame captured delay_steps earlier
    (capture happens at phase 0, promotion at phase delay_steps, and nothing
    writes pending in between)."""
    hs = scan_env._config.height_scan
    step = jax.jit(scan_env.step)
    state = jax.jit(scan_env.reset)(jax.random.PRNGKey(1))
    # A blackout episode holds zeros throughout, which would make "the buffer
    # moved" unobservable.
    assert int(state.info["scan_regime"]) == height_scan.NOISE
    moved = 0
    for _ in range(3 * hs.hold_steps):
        phase = int(state.info["scan_step"]) % hs.hold_steps
        before_hold = np.array(state.info["scan_hold"])
        before_pending = np.array(state.info["scan_pending"])
        state = step(state, jp.zeros(12))
        after = np.array(state.info["scan_hold"])
        if phase == hs.delay_steps:
            np.testing.assert_array_equal(after, before_pending)
            moved += int(np.any(after != before_hold))
        else:
            np.testing.assert_array_equal(after, before_hold)
    assert moved, "no capture ever reached the actor buffer"


# -- 3. roughness gate on the lift shaping ------------------------------------


def _gate_probe(env, state, xy):
    """The reward terms the gate scales, with the robot placed at `xy`.

    Everything the terms depend on except the terrain under the robot is
    staged: heading zero, a fixed foot clearance, all four feet in swing and
    landing this step, and a swing that reached the apex target.
    """
    data = state.data
    delta = jp.asarray(xy) - data.qpos[0:2]
    qpos = data.qpos.at[0:2].set(jp.asarray(xy))
    qpos = qpos.at[3:7].set(jp.array([1.0, 0.0, 0.0, 0.0]))
    geom_xpos = data.geom_xpos.at[:, 0:2].add(delta)
    feet = env._foot_geom_ids
    ground = env._terrain.height(geom_xpos[feet][:, 0:2])
    geom_xpos = geom_xpos.at[feet, 2].set(ground + CLEARANCE + FOOT_RADIUS)
    adr = env._sensor_adr["orientation"]
    sensordata = data.sensordata.at[adr : adr + 4].set(
        jp.array([1.0, 0.0, 0.0, 0.0])
    )
    data = data.replace(qpos=qpos, geom_xpos=geom_xpos, sensordata=sensordata)
    info = dict(state.info)
    info["command"] = jp.array([0.5, 0.0, 0.0, 0.125])
    info["swing_apex"] = jp.full(4, env._config.reward.apex_target)
    rewards, _ = env._get_reward(
        data, info, jp.zeros(12), jp.zeros(12),
        jp.ones(4, dtype=bool), jp.zeros(4, dtype=bool),
    )
    return {k: float(rewards[k]) for k in ("feet_apex", "high_step")}


def _tile_xy(env, level, ttype):
    index = terrain.TYPES.index(ttype)
    return np.array(env._terrain.origin_xy)[level, index]


def test_terrain_gate_scales_the_lift_shaping(scan_env):
    """Flat ground pays the floor fraction of feet_apex and high_step; a
    stair tile pays more. Ungated, both terms are the bare expressions."""
    state = jax.jit(scan_env.reset)(jax.random.PRNGKey(0))
    flat_xy = _tile_xy(scan_env, 0, "pyramid_stairs")
    stairs_xy = _tile_xy(scan_env, TERRAIN_ROWS, "pyramid_stairs")

    gate = scan_env._config.reward.terrain_gate
    gated_flat = _gate_probe(scan_env, state, flat_xy)
    # The spawn pad is flat by construction, so the gate can only rise off it.
    offsets = np.linspace(0.7, 1.4, 4)
    gated_stairs = [
        _gate_probe(scan_env, state, stairs_xy + np.array([d, 0.0]))
        for d in offsets
    ]
    gate.enable = False
    try:
        ungated_flat = _gate_probe(scan_env, state, flat_xy)
        ungated_stairs = [
            _gate_probe(scan_env, state, stairs_xy + np.array([d, 0.0]))
            for d in offsets
        ]
    finally:
        gate.enable = True

    # Four feet at the apex target, and 0.04 m of clearance against a 0.08 m
    # swing target: the ungated terms are exactly these numbers.
    assert ungated_flat["feet_apex"] == pytest.approx(4.0)
    assert ungated_flat["high_step"] == pytest.approx(CLEARANCE / 0.08)
    for term in ("feet_apex", "high_step"):
        assert gated_flat[term] == pytest.approx(
            ungated_flat[term] * GATE_FLOOR, rel=1e-5
        )
        factors = [
            g[term] / u[term] for g, u in zip(gated_stairs, ungated_stairs)
        ]
        assert max(factors) > GATE_FLOOR + 0.05, factors
        assert max(factors) <= 1.0 + 1e-5


# -- 4. the curriculum wrapper at an episode boundary -------------------------


def _drive_to_a_done(env, n, key=0):
    """Wrapped rollout ending on a step where the rolled-over envs fell.

    Returns the mask too: one step of physics can right a base on a slope, so
    which envs terminate depends on the tile they spawned on.
    """
    wrapped = wrap_for_terrain_brax_training(env, episode_length=1000)
    step = jax.jit(wrapped.step)
    state = jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(key), n))
    state = step(state, jp.zeros((n, 12)))
    assert not np.any(np.array(state.done))
    roll90 = jp.array([jp.cos(jp.pi / 4), jp.sin(jp.pi / 4), 0.0, 0.0])
    qpos = state.data.qpos.at[:, 3:7].set(roll90)
    before = state.replace(data=state.data.replace(qpos=qpos))
    after = step(before, jp.zeros((n, 12)))
    done = np.array(after.done).astype(bool)
    assert done.any(), "the staged fall terminated nothing"
    return wrapped, before, after, done


def _mirrored(values, mirror):
    perm = height_scan.mirror_map()[0]
    return np.where(np.asarray(mirror)[:, None], values[:, perm], values)


def test_wrapper_serves_the_scan_of_the_new_spawn(scan_env):
    """The observation a done step returns carries the scan of the tile the
    env was teleported to, not the one the cached spawn observation was built
    on."""
    n = 4
    _, _, state, done = _drive_to_a_done(scan_env, n)
    critic = np.array(state.obs["privileged_state"])[:, _critic_slice(scan_env)]
    actor = np.array(state.obs["state"])[:, _actor_slice(scan_env)]

    hs = scan_env._config.height_scan
    grid = np.array(scan_env._scan_grid)
    qpos = np.array(state.data.qpos)
    first = state.info["first_data"]
    # The teleport is a rigid move of the cached pose, so the feet keep their
    # height above the base.
    dz = qpos[:, 2] - np.array(first.qpos)[:, 2]
    ref_z = np.min(
        np.array(first.geom_xpos)[:, scan_env._foot_geom_ids, 2], axis=1
    ) + dz
    expected = np.stack([
        np.clip(
            np.array(
                scan_env._terrain.height(
                    height_scan.world_xy(
                        grid, qpos[i, 0:2],
                        height_scan.yaw_from_quat(qpos[i, 3:7]),
                    )
                )
            )
            - ref_z[i],
            -hs.clip, hs.clip,
        )
        for i in range(n)
    ])
    mirror = np.array(state.info["mirror"])
    np.testing.assert_allclose(
        critic[done], _mirrored(expected, mirror)[done], atol=1e-5
    )
    np.testing.assert_allclose(
        actor[done],
        _mirrored(np.array(state.info["scan_hold"]), mirror)[done],
        atol=1e-6,
    )
    cached = np.array(state.info["first_obs"]["state"])[:, _actor_slice(scan_env)]
    assert np.any(np.abs(actor[done] - cached[done]) > 1e-6)


def test_wrapper_redraws_the_corruption_regime(scan_env):
    n = 16
    _, before, after, done = _drive_to_a_done(scan_env, n, key=1)
    was = np.array(before.info["scan_regime"])
    now = np.array(after.info["scan_regime"])
    assert set(np.unique(now)) <= {0, 1, 2}
    assert np.any(was[done] != now[done])
    np.testing.assert_array_equal(was[~done], now[~done])
    steps = np.array(after.info["scan_step"])[done]
    assert np.all((steps >= 0) & (steps < scan_env._config.height_scan.hold_steps))


def test_wrapper_without_a_scan_is_untouched(blind_env):
    """No scan, no scan work: the wrapper serves the cached spawn observation
    on done and adds nothing to the env's info."""
    n = 4
    wrapped, _, state, done = _drive_to_a_done(blind_env, n, key=2)
    assert wrapped._scan_live is False
    assert not [k for k in state.info if k.startswith("scan_")]
    for key in ("state", "privileged_state"):
        np.testing.assert_array_equal(
            np.array(state.obs[key])[done],
            np.array(state.info["first_obs"][key])[done],
        )


# -- 5. occlusion over a descending edge --------------------------------------


def _teleport(env, state, xy, yaw):
    """The reset pose moved rigidly to `xy` and turned to face `yaw`."""
    data = state.data
    quat = jp.array([jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)])
    xy = jp.asarray(xy)
    dz = env._terrain.height(xy) - env._terrain.height(data.qpos[0:2])
    delta = xy - data.qpos[0:2]
    qpos = data.qpos.at[0:2].set(xy).at[2].add(dz).at[3:7].set(quat)
    geom_xpos = data.geom_xpos.at[:, 0:2].add(delta).at[:, 2].add(dz)
    adr = env._sensor_adr["orientation"]
    return data.replace(
        qpos=qpos,
        geom_xpos=geom_xpos,
        sensordata=data.sensordata.at[adr : adr + 4].set(quat),
    )


def _both_masks(env, data):
    """(mask, frustum-only mask) at one pose, as 5x5 grids.

    Also checks that the clean grid the critic reads is the same either way:
    occlusion is a mask, not a measurement.
    """
    m = env._config.height_scan.mask
    assert m.occlusion, "the default has occlusion on"
    clean, seen = env._scan_raw(data)
    m.occlusion = False
    try:
        clean_off, frustum = env._scan_raw(data)
    finally:
        m.occlusion = True
    np.testing.assert_array_equal(np.array(clean_off), np.array(clean))
    return np.array(seen).reshape(5, 5), np.array(frustum).reshape(5, 5)


def test_occlusion_hides_the_treads_below_a_stair_rim(scan_env):
    """Standing near the edge of the stair plateau and facing out, the grid
    runs down the flight. The rim between the camera and the lower treads
    takes them out of the mask; the frustum alone keeps every one of them."""
    state = jax.jit(scan_env.reset)(jax.random.PRNGKey(0))
    stairs_xy = _tile_xy(scan_env, TERRAIN_ROWS, "pyramid_stairs")
    data = _teleport(scan_env, state, stairs_xy + np.array([-0.25, 0.0]), 0.0)

    grid_xy = height_scan.world_xy(
        scan_env._scan_grid, data.qpos[0:2],
        height_scan.yaw_from_quat(scan_env._quat(data)),
    )
    ground = np.array(scan_env._terrain.height(grid_xy)).reshape(5, 5)
    # the pose is only a test of occlusion if the grid really runs off an edge
    assert np.all(np.diff(ground.mean(axis=1)) < -0.03), ground

    seen, frustum = _both_masks(scan_env, data)
    assert frustum.all(), frustum
    assert seen[0].all(), seen  # the plateau the robot stands on
    assert not seen[-1].any(), seen  # the bottom treads, behind the rim


def test_flat_ground_sees_everything_the_frustum_does(scan_env):
    """No edge, no occlusion: on the arena's flat row the two masks agree."""
    state = jax.jit(scan_env.reset)(jax.random.PRNGKey(0))
    data = _teleport(scan_env, state, _tile_xy(scan_env, 0, "rough_uniform"), 0.0)
    seen, frustum = _both_masks(scan_env, data)
    np.testing.assert_array_equal(seen, frustum)
    assert frustum.any()
