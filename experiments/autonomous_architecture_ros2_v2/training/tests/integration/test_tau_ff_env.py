"""Feed-forward torque head, on the real env.

The disabled default's bitwise guarantee is carried by
test_latency.py::test_golden_bitwise_regression (the tau_ff refactor left
the disabled step path byte-identical); here we prove the ENABLED head
actually reaches the physics under both delay paths, that the action/obs
contracts hold, and that the reward terms price it.
"""

import jax
import jax.numpy as jp
import numpy as np
import pytest

from wojtek_rl import env as wojtek_env


def _make_env(**tau_overrides):
    cfg = wojtek_env.default_config()
    cfg.tau_ff.enable = True
    for k, v in tau_overrides.items():
        setattr(cfg.tau_ff, k, v)
    return wojtek_env.WojtekJoystick(config=cfg)


@pytest.fixture(scope="module")
def env():
    return _make_env()


def test_action_and_obs_contract(env):
    assert env.action_size == 24
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    # Obs sizes must NOT grow with the head: last_act stays the 12-wide
    # position half (deployment obs contract).
    assert state.obs["state"].shape == (wojtek_env.OBS_SIZE,)
    assert state.obs["privileged_state"].shape == (
        wojtek_env.PRIVILEGED_SIZE,
    )
    assert state.info["last_act"].shape == (12,)
    assert state.info["filtered_act"].shape == (24,)
    assert state.info["tau_ff_prev"].shape == (12,)
    # Both pricing terms exist in the metrics stream.
    assert "reward/tau_ff_swing" in state.metrics
    assert "reward/tau_ff" in state.metrics


def test_head_reaches_the_physics_through_the_delay(env):
    """With the default action_delay=1, this step's tau_ff lands on the
    NEXT step's physics (like the position targets): identical first
    steps, diverged second steps."""
    step = jax.jit(env.step)
    s0 = jax.jit(env.reset)(jax.random.PRNGKey(1))
    zero = jp.zeros(24)
    kick = jp.zeros(24).at[12:].set(1.0)  # +tau_ff.scale Nm on every joint

    a_zero = step(s0, zero)
    a_kick = step(s0, kick)
    np.testing.assert_array_equal(
        np.asarray(a_zero.data.qpos), np.asarray(a_kick.data.qpos)
    )

    b_zero = step(a_zero, zero)
    b_kick = step(a_kick, zero)  # the kick was queued; both act zero now
    assert not np.array_equal(
        np.asarray(b_zero.data.qpos), np.asarray(b_kick.data.qpos)
    )


def test_head_reaches_the_physics_immediately_without_delay():
    cfg = wojtek_env.default_config()
    cfg.tau_ff.enable = True
    cfg.latency.enable = True
    cfg.latency.min_substeps = 0
    cfg.latency.max_substeps = 0  # d=0: new ctrl (and tau_ff) at once
    env = wojtek_env.WojtekJoystick(config=cfg)
    step = jax.jit(env.step)
    s0 = jax.jit(env.reset)(jax.random.PRNGKey(2))
    zero = jp.zeros(24)
    kick = jp.zeros(24).at[12:].set(1.0)
    assert not np.array_equal(
        np.asarray(step(s0, zero).data.qpos),
        np.asarray(step(s0, kick).data.qpos),
    )


def test_swing_penalty_prices_the_current_action(env):
    """The tau_ff_swing metric reads the CURRENT action's head against the
    POST-step contacts. With the base lifted well off the floor every leg
    is airborne through the step, so a full-head action prices all four
    legs; grounded, the same action prices only airborne legs (zero when
    all feet are down)."""
    step = jax.jit(env.step)
    s0 = jax.jit(env.reset)(jax.random.PRNGKey(3))
    kick = jp.zeros(24).at[12:].set(1.0)
    zero = jp.zeros(24)

    lifted = s0.replace(
        data=s0.data.replace(qpos=s0.data.qpos.at[2].add(0.3))
    )
    m_air = step(lifted, kick).metrics["reward/tau_ff_swing"]
    # 1.0 raw * scale Nm * 3 joints * 4 airborne legs.
    expected = float(env._config.tau_ff.scale) * 3.0 * 4.0
    np.testing.assert_allclose(float(m_air), expected, rtol=1e-5)

    # A zero head costs nothing anywhere; a full head on the ground costs
    # only whatever legs the step left airborne (bounded by the full price).
    assert float(step(lifted, zero).metrics["reward/tau_ff_swing"]) == 0.0
    m_ground = step(s0, kick).metrics["reward/tau_ff_swing"]
    assert 0.0 <= float(m_ground) <= expected
    assert float(step(s0, kick).metrics["reward/tau_ff"]) > 0.0


def test_head_clip_bounds_authority(env):
    """Head outputs past +-1 saturate at +-scale Nm before the physics."""
    step = jax.jit(env.step)
    s0 = jax.jit(env.reset)(jax.random.PRNGKey(4))
    big = jp.zeros(24).at[12:].set(100.0)
    one = jp.zeros(24).at[12:].set(1.0)
    q_big = np.asarray(step(step(s0, big), big).data.qpos)
    q_one = np.asarray(step(step(s0, one), one).data.qpos)
    np.testing.assert_allclose(q_big, q_one, atol=1e-9)
