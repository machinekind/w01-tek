"""Unified locomotion mechanics: height command, gait blending, stand."""

import jax
import jax.numpy as jp
import numpy as np
import pytest

from fbb_rl import env as fbb_env


@pytest.fixture(scope="module")
def env():
    return fbb_env.FourBarBotJoystick()


def _cmd(vx=0.0, vy=0.0, wz=0.0, h=0.125):
    return jp.array([vx, vy, wz, h])


def test_command_is_4d_and_obs_sizes(env):
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    assert state.info["command"].shape == (4,)
    assert state.obs["state"].shape == (fbb_env.OBS_SIZE,)
    assert state.obs["privileged_state"].shape == (fbb_env.PRIVILEGED_SIZE,)


def test_zero_command_keeps_height(env):
    heights = []
    for s in range(40):
        cmd = env._sample_command(jax.random.PRNGKey(s))
        c = env._config.command
        assert c.height[0] - 1e-6 <= float(cmd[3]) <= c.height[1] + 1e-6
        heights.append(float(cmd[3]))
    assert np.std(heights) > 0.005  # actually varies


def test_height_ctrl_monotone(env):
    lo = env._height_ctrl(jp.array(0.09))
    mid = env._height_ctrl(jp.array(0.125))
    hi = env._height_ctrl(jp.array(0.17))
    # second joints (idx 1,4,7,10) extend with height
    for i in (1, 4, 7, 10):
        assert float(lo[i]) < float(mid[i]) < float(hi[i])
    # abduction joints unchanged
    for i in (0, 3, 6, 9):
        assert abs(float(hi[i]) - float(lo[i])) < 1e-6


def test_gait_blend_walk_to_trot(env):
    slow = {"command": _cmd(vx=0.1), "phase": jp.array(0.0)}
    fast = {"command": _cmd(vx=0.8), "phase": jp.array(0.0)}
    walk = np.array(env._leg_phases(slow))
    trot = np.array(env._leg_phases(fast))
    wrap = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
    np.testing.assert_allclose(walk, wrap(np.array(fbb_env.WALK_PHASE)), atol=1e-5)
    np.testing.assert_allclose(trot, wrap(np.array(fbb_env.TROT_PHASE)), atol=1e-5)


def test_clock_freezes_when_standing(env):
    assert float(env._phase_dt(_cmd(vx=0.0))) == 0.0
    assert float(env._phase_dt(_cmd(vx=0.5))) > 0.0
    # faster command -> faster clock
    assert float(env._phase_dt(_cmd(vx=1.0))) > float(env._phase_dt(_cmd(vx=0.2)))


def test_step_finite_and_height_reward_present(env):
    state = jax.jit(env.reset)(jax.random.PRNGKey(1))
    state = jax.jit(env.step)(state, jp.zeros(12))
    assert jp.isfinite(state.reward)
    assert "reward/height_tracking" in state.metrics
    assert jp.isfinite(state.obs["state"]).all()


def test_reset_starts_near_commanded_height(env):
    for seed in range(4):
        state = jax.jit(env.reset)(jax.random.PRNGKey(seed))
        h_cmd = float(state.info["command"][3])
        assert abs(float(state.data.qpos[2]) - h_cmd) < 0.02
