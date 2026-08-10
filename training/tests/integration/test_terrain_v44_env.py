"""The v4.4 additions on a real env: the flat_pitch posture term with its
roughness/row gates, sticky flat spins, and the pinned-flat wrapper slice.

Reuses the v4.2 test arena geometry. Most envs here are constructed and
reset but not stepped; the wrapper tests pay one small step compile like
the v4.2 module. The pinned_flat_frac=0.0 wrapper path (mask and rungs
unchanged) is also exercised every run by
test_terrain_v42_env.py::test_wrapper_respawn_pins_and_clears_band_state,
which runs pinned_frac=0.5 with the flat pin at its default.
"""

import jax
import jax.numpy as jp
import numpy as np
import pytest

from wojtek_rl import build_terrain, paths, terrain, terrain_env
from wojtek_rl import env as wojtek_env
from wojtek_rl.terrain_wrapper import wrap_for_terrain_brax_training

ARENA = "test"
ROWS = 3
TREADS = (0.25, 0.45)
CAPS = {"pyramid_stairs": 0.7, "inverted_pyramid_stairs": 0.7}
PAD = 0.3

METRIC_KEYS = (
    "pitch_down_deg_per_step",
    "flat_frac_per_step",
    "flat_pitch_down_deg_per_step",
    "flat_pitch_on_flat_per_step",
    "cmd_spin_per_step",
)


@pytest.fixture(scope="module")
def arena():
    a = terrain.generate(
        seed=0, n_rows=ROWS, flat_row=True, pad_radius=PAD,
        stair_platform_half=PAD, stair_tread=TREADS, type_caps=CAPS,
    )
    build_terrain.write_arena(a, ARENA)
    yield a
    for p in paths.terrain_paths(ARENA).values():
        p.unlink(missing_ok=True)


def _terrain_cfg():
    cfg = wojtek_env.default_config()
    cfg.terrain.enable = True
    cfg.terrain.arena = ARENA
    cfg.terrain.flat_row = True
    cfg.terrain.spawn_mode = "feature"
    cfg.terrain.stair_tread_range = TREADS
    cfg.terrain.type_caps = dict(CAPS)
    cfg.terrain.pad_radius = PAD
    return cfg


@pytest.fixture(scope="module")
def gated_env(arena):
    """Terrain gate on, flat_pitch priced, every spawn on the flat row
    (rough = 0 under the feet and the strip)."""
    cfg = _terrain_cfg()
    cfg.terrain.spawn_level = 0
    cfg.reward.terrain_gate.enable = True
    cfg.reward.scales.flat_pitch = -10.0
    return wojtek_env.WojtekJoystick(cfg)


@pytest.fixture(scope="module")
def gated_reset(gated_env):
    return jax.jit(gated_env.reset)(jax.random.PRNGKey(0))


@pytest.fixture(scope="module")
def flat_env():
    return wojtek_env.WojtekJoystick()


@pytest.fixture(scope="module")
def flat_reset(flat_env):
    return jax.jit(flat_env.reset)(jax.random.PRNGKey(0))


def _pitched_data(env, state, pitch_deg):
    """state's data with the base pitched about +y; positive = nose down
    (gravity_body[0] = +sin(pitch), per battery.pitch_down_deg).
    _gravity_body reads the orientation sensor, so the quat goes into
    sensordata."""
    half = np.radians(pitch_deg) / 2.0
    quat = jp.array([np.cos(half), 0.0, np.sin(half), 0.0])
    adr = env._sensor_adr["orientation"]
    sensordata = state.data.sensordata.at[adr : adr + 4].set(quat)
    return state.data.replace(sensordata=sensordata)


def _rewards(env, data, info):
    zeros4 = jp.zeros(4, dtype=bool)
    rewards, _ = env._get_reward(
        data, info, jp.zeros(12), jp.zeros(12), zeros4, zeros4
    )
    return rewards


def _sin(deg):
    return float(np.sin(np.radians(deg)))


# -- flat_pitch reward term ----------------------------------------------------


def test_flat_pitch_prices_a_pitched_flat_row_state(gated_env, gated_reset):
    """On the flat row (rough 0, gate 1) a 10 deg nose-down state pays the
    linear sin-space hinge past the 2 deg cone; upright and nose-up pay
    exactly zero."""
    info = gated_reset.info
    pitched = _pitched_data(gated_env, gated_reset, 10.0)
    assert float(_rewards(gated_env, pitched, info)["flat_pitch"]) == (
        pytest.approx(_sin(10.0) - _sin(2.0), abs=1e-6)
    )
    upright = _pitched_data(gated_env, gated_reset, 0.0)
    assert float(_rewards(gated_env, upright, info)["flat_pitch"]) == 0.0
    nose_up = _pitched_data(gated_env, gated_reset, -10.0)
    assert float(_rewards(gated_env, nose_up, info)["flat_pitch"]) == 0.0


def test_flat_pitch_scale_carries_the_price_into_the_summed_reward(
    gated_env, gated_reset
):
    """The -10 scale prices the hinge in the same sum expression step()
    uses; at scale 0 the identical sum is bit-exact without the term."""
    pitched = _pitched_data(gated_env, gated_reset, 10.0)
    rewards = _rewards(gated_env, pitched, gated_reset.info)
    scales = gated_env._config.reward.scales
    total = sum(rewards[k] * scales[k] for k in rewards)
    without = sum(rewards[k] * scales[k] for k in rewards if k != "flat_pitch")
    assert float(total) == pytest.approx(
        float(without) - 10.0 * float(rewards["flat_pitch"]), abs=1e-6
    )
    assert float(rewards["flat_pitch"]) > 0.1


def test_flat_pitch_scale_zero_is_bit_exact_off(flat_env, flat_reset):
    """Default scale 0.0: on a canned pitched state where the raw term is
    live, the summed reward equals the flat_pitch-free sum bit for bit
    (`+ raw*0.0` is IEEE-exact)."""
    assert flat_env._config.reward.scales.flat_pitch == 0.0
    pitched = _pitched_data(flat_env, flat_reset, 10.0)
    rewards = _rewards(flat_env, pitched, flat_reset.info)
    assert float(rewards["flat_pitch"]) > 0.1  # the raw term is live
    scales = flat_env._config.reward.scales
    total = sum(rewards[k] * scales[k] for k in rewards)
    without = sum(rewards[k] * scales[k] for k in rewards if k != "flat_pitch")
    assert float(total) == float(without)


def test_flat_pitch_gate_open_when_terrain_is_off(flat_env, flat_reset):
    """Terrain off = the whole arena is flat: the hinge is charged in
    full, no roughness gate in the path."""
    pitched = _pitched_data(flat_env, flat_reset, 10.0)
    rewards = _rewards(flat_env, pitched, flat_reset.info)
    assert float(rewards["flat_pitch"]) == pytest.approx(
        _sin(10.0) - _sin(2.0), abs=1e-6
    )


def test_flat_pitch_row_only_gate_follows_the_row_id(arena, gated_reset):
    """flat_pitch_row_only: the gate is the flat-row id, so a terrain-row
    state pays nothing whatever the local roughness reads."""
    cfg = _terrain_cfg()
    cfg.terrain.spawn_level = 0
    cfg.reward.terrain_gate.enable = True
    cfg.reward.terrain_gate.flat_pitch_row_only = True
    cfg.reward.scales.flat_pitch = -10.0
    env = wojtek_env.WojtekJoystick(cfg)
    pitched = _pitched_data(env, gated_reset, 10.0)
    info = dict(gated_reset.info)
    info["terrain_level"] = jp.array(0)
    assert float(_rewards(env, pitched, info)["flat_pitch"]) == pytest.approx(
        _sin(10.0) - _sin(2.0), abs=1e-6
    )
    info["terrain_level"] = jp.array(1)
    assert float(_rewards(env, pitched, info)["flat_pitch"]) == 0.0


def test_flat_pitch_spin_exempt_frees_pure_spin_windows(arena, gated_reset):
    """flat_pitch_spin_exempt (v4.5): a pure-spin command window pays no
    flat_pitch on the flat row; walk, arc and stand windows keep the full
    hinge."""
    cfg = _terrain_cfg()
    cfg.terrain.spawn_level = 0
    cfg.reward.terrain_gate.enable = True
    cfg.reward.terrain_gate.flat_pitch_spin_exempt = True
    cfg.reward.scales.flat_pitch = -10.0
    env = wojtek_env.WojtekJoystick(cfg)
    pitched = _pitched_data(env, gated_reset, 10.0)
    hinge = _sin(10.0) - _sin(2.0)

    def priced(command):
        info = dict(gated_reset.info)
        info["command"] = jp.array(command)
        return float(_rewards(env, pitched, info)["flat_pitch"])

    assert priced([0.0, 0.0, 1.0, 0.125]) == 0.0  # pure spin
    assert priced([0.4, 0.0, 0.0, 0.125]) == pytest.approx(hinge, abs=1e-6)
    assert priced([0.3, 0.0, 1.0, 0.125]) == pytest.approx(hinge, abs=1e-6)
    assert priced([0.0, 0.0, 0.0, 0.125]) == pytest.approx(hinge, abs=1e-6)


def test_flat_pitch_spin_exempt_default_off_keeps_spins_priced(
    gated_env, gated_reset
):
    """Without the key (default False) a pure-spin window pays the hinge
    exactly as before v4.5."""
    pitched = _pitched_data(gated_env, gated_reset, 10.0)
    info = dict(gated_reset.info)
    info["command"] = jp.array([0.0, 0.0, 1.0, 0.125])
    assert float(_rewards(gated_env, pitched, info)["flat_pitch"]) == (
        pytest.approx(_sin(10.0) - _sin(2.0), abs=1e-6)
    )


def test_flat_pitch_reads_zero_when_terrain_gate_is_off(arena, gated_reset):
    """Terrain on with terrain_gate off: no roughness signal exists, so
    the term is a constant 0 and only the key survives (parity with
    scales and both metrics dicts)."""
    cfg = _terrain_cfg()
    cfg.terrain.spawn_level = 0
    cfg.reward.scales.flat_pitch = -10.0
    env = wojtek_env.WojtekJoystick(cfg)
    pitched = _pitched_data(env, gated_reset, 10.0)
    rewards = _rewards(env, pitched, gated_reset.info)
    assert "flat_pitch" in rewards
    assert float(rewards["flat_pitch"]) == 0.0


# -- command.pure_wz_sticky ------------------------------------------------------


V42_MIX = dict(zero_prob=0.25, pure_wz_prob=0.25, pure_vy_prob=0.2,
               arc_prob=0.2, pure_slow_prob=0.1, pure_back_prob=0.2)


def _cmd_env(**command_overrides):
    cfg = wojtek_env.default_config()
    cfg.command.update(command_overrides)
    return wojtek_env.WojtekJoystick(cfg)


@pytest.fixture(scope="module")
def sticky_on():
    return _cmd_env(**V42_MIX, pure_wz_sticky=True)


@pytest.fixture(scope="module")
def sticky_off():
    return _cmd_env(**V42_MIX)


def _draws(env, keys, on_flat):
    if on_flat is None:
        return np.array(jax.vmap(env._sample_command)(keys))
    return np.array(
        jax.vmap(lambda k: env._sample_command(k, on_flat))(keys)
    )


def _is_spin(cmds):
    return (np.abs(cmds[:, 2]) > 0) & (cmds[:, 0] == 0) & (cmds[:, 1] == 0)


def test_sticky_off_matches_keyless_config(sticky_off):
    """pure_wz_sticky=False consumes no randomness and moves no draw: the
    commands are bit-identical to a config that never had the key, on every
    on_flat path."""
    cfg = wojtek_env.default_config()
    cfg.command.update(V42_MIX)
    del cfg.command.pure_wz_sticky
    keyless = wojtek_env.WojtekJoystick(cfg)
    keys = jax.random.split(jax.random.PRNGKey(0), 256)
    for flat in (None, jp.array(True), jp.array(False)):
        np.testing.assert_array_equal(
            _draws(sticky_off, keys, flat), _draws(keyless, keys, flat)
        )


def test_sticky_restores_every_flat_spin_and_nothing_else(sticky_on, sticky_off):
    """Sticky on the flat row: every base pure_wz draw survives the
    overwrite chain, every changed draw is a clean spin, and everything
    else (heights included) is bit-identical."""
    keys = jax.random.split(jax.random.PRNGKey(1), 512)
    a = _draws(sticky_on, keys, jp.array(True))
    b = _draws(sticky_off, keys, jp.array(True))
    spin_a, spin_b = _is_spin(a), _is_spin(b)
    changed = np.any(a != b, axis=1)
    assert np.all(spin_a[changed])  # sticky only ever restores spins
    np.testing.assert_array_equal(a[~changed], b[~changed])
    np.testing.assert_array_equal(a[:, 3], b[:, 3])  # heights untouched
    assert np.all(spin_a[spin_b])  # surviving spins stay bit-identical
    # Survival goes ~0.35 -> 1.0 of the 0.25 draw prob.
    assert spin_a.mean() == pytest.approx(0.25, abs=0.05)
    assert spin_a.mean() > 2.0 * spin_b.mean()


def test_sticky_leaves_terrain_rows_untouched(sticky_on, sticky_off):
    """Sticky is masked by on_flat: terrain-row draws are bit-identical
    with the knob on."""
    keys = jax.random.split(jax.random.PRNGKey(2), 256)
    np.testing.assert_array_equal(
        _draws(sticky_on, keys, jp.array(False)),
        _draws(sticky_off, keys, jp.array(False)),
    )
    # on_flat=None (terrain off) is treated as flat.
    np.testing.assert_array_equal(
        _draws(sticky_on, keys, None), _draws(sticky_on, keys, jp.array(True))
    )


# -- terrain.pinned_flat_frac ----------------------------------------------------


def test_flat_pin_defaults_to_zero_for_keyless_configs(arena):
    """A pre-v4.4 config (no pinned_flat_frac key) loads with the pin at
    0.0, same as the explicit default."""
    cfg = _terrain_cfg()
    del cfg.terrain.pinned_flat_frac
    assert terrain_env.load(cfg.terrain).pinned_flat_frac == 0.0


def test_flat_pin_zero_rolls_out_identically_to_a_keyless_config(arena):
    """pinned_flat_frac=0.0 leaves the pinned mask and every level
    bit-identical: a rollout under the explicit 0.0 matches a config that
    never had the key, teleports included."""
    def rollout(delete_key):
        cfg = _terrain_cfg()
        cfg.terrain.pinned_frac = 0.5
        if delete_key:
            del cfg.terrain.pinned_flat_frac
        env = wojtek_env.WojtekJoystick(cfg)
        wrapped = wrap_for_terrain_brax_training(
            env, episode_length=3, action_repeat=1
        )
        n = 8
        rng = jax.random.split(jax.random.PRNGKey(0), n)
        state = jax.jit(wrapped.reset)(rng)
        step = jax.jit(wrapped.step)
        levels, qpos = [], []
        for _ in range(8):
            state = step(state, jp.zeros((n, 12)))
            levels.append(np.array(state.info["terrain_level"]))
            qpos.append(np.array(state.data.qpos))
        return np.stack(levels), np.stack(qpos)

    levels_zero, qpos_zero = rollout(delete_key=False)
    levels_keyless, qpos_keyless = rollout(delete_key=True)
    np.testing.assert_array_equal(levels_zero, levels_keyless)
    np.testing.assert_array_equal(qpos_zero, qpos_keyless)


def test_flat_pin_pins_exactly_the_extra_slice_at_level_zero(arena):
    """pinned_frac=0.5 + pinned_flat_frac=0.25 of 8 envs: envs 0-3 hold
    rungs 0-3 as before (the flat row plus ROWS terrain rows), envs 4-5
    hold the flat row, the rest ride the ladder."""
    cfg = _terrain_cfg()
    cfg.terrain.pinned_frac = 0.5
    cfg.terrain.pinned_flat_frac = 0.25
    env = wojtek_env.WojtekJoystick(cfg)
    wrapped = wrap_for_terrain_brax_training(
        env, episode_length=3, action_repeat=1
    )
    n = 8
    rng = jax.random.split(jax.random.PRNGKey(0), n)
    state = jax.jit(wrapped.reset)(rng)
    step = jax.jit(wrapped.step)
    for _ in range(8):  # crosses at least two 3-step episode boundaries
        state = step(state, jp.zeros((n, 12)))
    levels = np.array(state.info["terrain_level"])
    assert list(levels[:4]) == [0, 1, 2, 3]  # per-rung slice, untouched
    assert levels[4] == 0 and levels[5] == 0  # the flat slice
    assert np.all(levels >= 0) and np.all(levels < env._terrain.n_rows)


# -- telemetry metrics -----------------------------------------------------------


def test_new_metrics_exist_and_read_flat_on_a_flat_env(flat_env, flat_reset):
    for key in METRIC_KEYS:
        assert key in flat_reset.metrics
    state = jax.jit(flat_env.step)(flat_reset, jp.zeros(12))
    for key in METRIC_KEYS:
        assert np.isfinite(float(state.metrics[key]))
    # Terrain off: every env counts as flat, and the flat-conditioned cost
    # numerator is exactly the raw gated hinge.
    assert float(state.metrics["flat_frac_per_step"]) == 1.0
    assert float(state.metrics["flat_pitch_on_flat_per_step"]) == float(
        state.metrics["reward/flat_pitch"]
    )


def test_cmd_spin_metric_flags_pure_spin_commands(flat_env, flat_reset):
    def stepped_with(command):
        info = dict(flat_reset.info)
        info["command"] = jp.array(command)
        state = flat_reset.replace(info=info)
        return jax.jit(flat_env.step)(state, jp.zeros(12))

    spin = stepped_with([0.0, 0.0, 1.0, 0.125])
    assert float(spin.metrics["cmd_spin_per_step"]) == 1.0
    arc = stepped_with([0.3, 0.0, 1.0, 0.125])
    assert float(arc.metrics["cmd_spin_per_step"]) == 0.0
    stand = stepped_with([0.0, 0.0, 0.0, 0.125])
    assert float(stand.metrics["cmd_spin_per_step"]) == 0.0


def test_flat_frac_metric_follows_the_row_on_terrain(gated_env, gated_reset):
    step = jax.jit(gated_env.step)
    on_flat = step(gated_reset, jp.zeros(12))
    assert float(on_flat.metrics["flat_frac_per_step"]) == 1.0
    info = dict(gated_reset.info)
    info["terrain_level"] = jp.array(1)
    off_flat = step(gated_reset.replace(info=info), jp.zeros(12))
    assert float(off_flat.metrics["flat_frac_per_step"]) == 0.0
    assert float(off_flat.metrics["flat_pitch_down_deg_per_step"]) == 0.0
