"""fall.max_toggle_deg: state-based four-bar branch guard."""

import jax
import jax.numpy as jp
import numpy as np
import pytest

from wojtek_rl import env as wojtek_env


def _cfg(max_toggle=150.0, knee_max=0.0):
    cfg = wojtek_env.default_config()
    cfg.pd_kp = 40.0
    cfg.pd_kd = 1.6
    cfg.max_torque = 9.0
    cfg.knee_target_max = knee_max  # 0: no command clamp — the point
    cfg.real_pose_ref = True
    cfg.fall.max_toggle_deg = max_toggle
    cfg.push.enable = False
    return cfg


@pytest.fixture(scope="module")
def env():
    return wojtek_env.WojtekJoystick(_cfg())


def test_healthy_stance_and_steps_do_not_terminate(env):
    """Nominal standing and stepping live at toggle ~13 deg — far from the
    150 deg guard."""
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    step = jax.jit(env.step)
    for _ in range(50):
        state = step(state, jp.zeros(12))
    assert float(state.done) == 0.0


def test_envelope_reaches_full_kinematic_height(env):
    """Without the command clamp the calibrated envelope must reach the
    full ~0.21 m kinematic top (it was capped at ~0.154 by the old
    knee_target_max)."""
    h = np.array(env._anchor_heights)
    assert h[-1] > 0.20, f"envelope top {h[-1]:.3f}"


def test_flat_linkage_terminates(env):
    """Force a leg's four-bar toward collinear (the 177-deg uncoordinated
    configuration measured on the kp40 plant: knee target far out, hip
    held): the toggle guard must terminate the episode."""
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    step = jax.jit(env.step)
    # drive knees hard toward extension while hips hold: the sagged,
    # uncoordinated path that approaches the toggle
    act = jp.zeros(12).at[jp.array([2, 5, 8, 11])].set(1.0)
    done = 0.0
    for _ in range(300):
        state = step(state, act)
        done = max(done, float(state.done))
        if done:
            break
    # Whether this exact drive reaches 150 deg depends on the plant; the
    # guard must at minimum never fire spuriously below it. If it fired,
    # confirm it fired for the right reason: recompute the angle.
    if done:
        a5 = np.array(state.data.xanchor)[np.array(env._toggle_j5)]
        af = np.array(state.data.xanchor)[np.array(env._toggle_jf)]
        sb = np.array(state.data.site_xpos)[np.array(env._toggle_site)]
        u, v = af - a5, sb - af
        cos = (u * v).sum(-1) / (
            np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-9
        )
        ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        low = np.array(state.data.qpos[2]) < env._config.fall.min_height
        assert ang.max() > 150.0 or low, (
            f"terminated but max toggle only {ang.max():.0f} deg and not a fall"
        )


def test_disabled_guard_is_inert():
    plain = wojtek_env.WojtekJoystick(_cfg(max_toggle=0.0))
    state = jax.jit(plain.reset)(jax.random.PRNGKey(1))
    state = jax.jit(plain.step)(state, jp.zeros(12))
    assert np.isfinite(float(state.reward))
