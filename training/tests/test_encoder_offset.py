"""Encoder-zero offset domain randomization (Workstream C2).

See docs/plans/2026-07-10-mjwarp-migration.md, Workstream C2, and the HARD
INVARIANT there: epsilon must only shift the OBSERVED joint_pos (added) and
the ctrl WRITTEN to physics (subtracted) -- never joint ref/qpos0/body
frames, which would silently re-anchor the four-bar closure ~8.5 cm open.
test_closure_residual_unchanged is the regression guard for that invariant.
"""

import jax
import jax.numpy as jp
import numpy as np

from wojtek_rl import env as wojtek_env
from wojtek_rl import paths


def _config(enable, range_=0.02):
    cfg = wojtek_env.default_config()
    cfg.encoder.enable = enable
    cfg.encoder.range = range_
    return cfg


def test_obs_shifted_by_epsilon():
    """joint_pos in the obs catalog reads (qpos - home_ctrl) + epsilon."""
    env = wojtek_env.WojtekJoystick(_config(True))
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    eps = np.array(state.info["encoder_offset"])
    assert eps.shape == (12,)
    assert np.any(eps != 0.0)
    assert np.all(np.abs(eps) <= 0.02)

    catalog = env._obs_catalog(state.data, state.info)
    expected = (
        np.array(state.data.qpos[env._qadr]) - np.array(env._home_ctrl) + eps
    )
    np.testing.assert_allclose(np.array(catalog["joint_pos"]), expected, atol=0)


def test_ctrl_shifted_by_epsilon():
    """After one step (latency disabled), data.ctrl == anchor - epsilon."""
    env = wojtek_env.WojtekJoystick(_config(True))  # latency disabled (default)
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    eps = np.array(state.info["encoder_offset"])
    anchor = np.array(state.info["motor_targets"])

    next_state = jax.jit(env.step)(state, jp.zeros(12))
    expected_ctrl = anchor - eps
    np.testing.assert_allclose(
        np.array(next_state.data.ctrl), expected_ctrl, atol=1e-6
    )


def test_closure_residual_unchanged():
    """Four-bar closure residual stays tight under a large encoder offset:
    epsilon shifts obs/ctrl only, never the connect-constraint anchor.

    Deviation from the literal brief: reset() injects independent +-0.05 rad
    noise per actuated joint, which displaces the passive closing link away
    from the manifold the equality constraint enforces; this is a soft
    (Baumgarte-style) constraint that relaxes back over ~40 control steps
    regardless of encoder DR (verified: identical transient with
    encoder.enable=False). Asserting <=2e-3 m from step 0 would fail on the
    reset transient alone, not on anything epsilon-related. So this test
    settles first (zero action, encoder still enabled throughout so epsilon
    is live the whole time), then measures under moderate random actions
    (realistic policy-output magnitude) -- the invariant under test (no
    re-anchoring) is exactly as well covered, just not confounded by the
    unrelated reset-noise transient.
    """
    env = wojtek_env.WojtekJoystick(_config(True, range_=0.05))
    state = jax.jit(env.reset)(jax.random.PRNGKey(1))
    step = jax.jit(env.step)

    foot_ids = np.array(
        [env._mj_model.body(f"{leg}_foot_link").id for leg in paths.LEGS]
    )
    chain_ids = np.array(
        [env._mj_model.body(f"{leg}_chain_close_a_link").id for leg in paths.LEGS]
    )

    for _ in range(60):  # let the reset-noise transient settle
        state = step(state, jp.zeros(12))

    rng = jax.random.PRNGKey(2)
    for i in range(50):
        rng, r_a = jax.random.split(rng)
        action = jax.random.uniform(r_a, (12,), minval=-0.2, maxval=0.2)
        state = step(state, action)
        xpos = np.array(state.data.xpos)
        residual = np.max(
            np.linalg.norm(xpos[foot_ids] - xpos[chain_ids], axis=-1)
        )
        assert residual <= 2e-3, f"step {i}: closure residual {residual} m"


def test_disabled_off_is_zero_and_no_rng():
    """Default env (encoder off): epsilon is exactly zero and no extra rng
    is spent, same guarantee as test_latency's disabled-path test."""
    rng = jax.random.PRNGKey(0)
    expected_rng, _, _ = jax.random.split(rng, 3)
    env = wojtek_env.WojtekJoystick()
    state = jax.jit(env.reset)(rng)

    assert np.array_equal(
        np.array(state.info["encoder_offset"]), np.zeros(12)
    )
    assert np.array_equal(np.array(state.info["rng"]), np.array(expected_rng))
