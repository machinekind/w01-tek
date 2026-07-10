import jax
import jax.numpy as jp
import numpy as np
import pytest

from wojtek_rl import env as wojtek_env


@pytest.fixture(scope="module")
def env():
    return wojtek_env.WojtekJoystick()


@pytest.fixture(scope="module")
def reset_state(env):
    return jax.jit(env.reset)(jax.random.PRNGKey(0))


def test_obs_shapes(env, reset_state):
    assert reset_state.obs["state"].shape == (wojtek_env.OBS_SIZE,)
    assert reset_state.obs["privileged_state"].shape == (wojtek_env.PRIVILEGED_SIZE,)


def test_action_size(env):
    assert env.action_size == 12


def test_step_produces_finite_reward(env, reset_state):
    state = jax.jit(env.step)(reset_state, jp.zeros(12))
    assert np.isfinite(float(state.reward))
    assert float(state.done) in (0.0, 1.0)
    for v in state.metrics.values():
        assert np.isfinite(float(v))


def test_pd_hold_keeps_robot_up(env, reset_state):
    step = jax.jit(env.step)
    state = reset_state
    for _ in range(50):  # 1 s of sim time
        state = step(state, jp.zeros(12))
    assert float(state.done) == 0.0
    assert float(state.data.qpos[2]) > 0.07


def test_tilted_robot_terminates(env, reset_state):
    qpos = reset_state.data.qpos.at[3:7].set(jp.array([0.0, 1.0, 0.0, 0.0]))
    data = reset_state.data.replace(qpos=qpos)
    state = reset_state.replace(data=data)
    state = jax.jit(env.step)(state, jp.zeros(12))
    assert float(state.done) == 1.0
