"""Encoder-zero offset domain randomization (Workstream C2).

See the HARD
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
    """A large encoder offset does not move the four-bar connect anchor: the
    closure residual stays tight because epsilon shifts obs and ctrl only.

    reset() adds +-0.05 rad noise to the actuated joints, which opens the
    loop for ~40 steps (the same transient with the encoder off). So settle
    under zero action first, then measure under moderate random actions.
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
        # 3e-3 recalibrated for the 14 kg model (measured 2.15 mm with the
        # encoder on vs 1.60 mm off; the offset itself adds only ~0.5 mm).
        assert residual <= 3e-3, f"step {i}: closure residual {residual} m"


def test_latency_and_encoder_combined():
    """Both domain-randomization features on at once: the ctrl actually
    written to physics must reflect BOTH the latency where-scan and the
    encoder subtraction. Neither test_latency.py nor the rest of this file
    exercises that combined branch of step() (encoder.enable and
    latency.enable both True) -- this is the only one that does.

    latency min_substeps=max_substeps=2 fixes d=2 < n_substeps=5, so the
    where-scan runs the PREV segment (substeps 0-1) then the NEW segment
    (substeps 2-4). The scan's last substep sets data.ctrl to the new
    targets minus epsilon, which is what should remain after the step.
    """
    cfg = wojtek_env.default_config()
    cfg.encoder.enable = True
    cfg.encoder.range = 0.02
    cfg.latency.enable = True
    cfg.latency.min_substeps = 2
    cfg.latency.max_substeps = 2
    assert cfg.action_filter == 0.0  # test assumes filtered action == action

    env = wojtek_env.WojtekJoystick(cfg)
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    eps = np.array(state.info["encoder_offset"])
    assert eps.shape == (12,)
    assert np.any(eps != 0.0)

    action = jp.full(12, 0.3)
    next_state = jax.jit(env.step)(state, action)

    # Same computation as step(): with action_filter=0, filtered action ==
    # action, so this is the new motor target for the step just taken.
    cmd = state.info["command"]
    new_targets = jp.clip(
        env._height_ctrl(cmd[3]) + action * env._config.action_scale,
        env._ctrlrange[:, 0],
        env._ctrlrange[:, 1],
    )
    expected_ctrl = np.array(new_targets) - eps
    np.testing.assert_allclose(
        np.array(next_state.data.ctrl), expected_ctrl, atol=1e-6
    )
    assert np.all(np.isfinite(np.array(next_state.data.qpos)))
    assert np.all(np.isfinite(np.array(next_state.data.qvel)))


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
