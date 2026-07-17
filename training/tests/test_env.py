import jax
import jax.numpy as jp
import numpy as np
import pytest

from wojtek_rl import env as wojtek_env
from wojtek_rl.base import ABDUCTION_ACTUATORS, KNEE_ACTUATORS


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


# -- abduction_ctrl_limit -----------------------------------------------------


def test_abduction_ctrl_limit_zero_leaves_ctrlrange_untouched(env):
    """Default (0 = off): abduction ctrlrange stays the model's +-pi, same
    as every other unclamped joint."""
    idx = np.array(ABDUCTION_ACTUATORS)
    ctrlrange = np.array(env.mj_model.actuator_ctrlrange[idx])
    np.testing.assert_allclose(ctrlrange[:, 0], -3.1416)
    np.testing.assert_allclose(ctrlrange[:, 1], 3.1416)


def test_abduction_ctrl_limit_clamps_ctrlrange_when_set():
    cfg = wojtek_env.default_config()
    cfg.abduction_ctrl_limit = 0.44
    limited = wojtek_env.WojtekJoystick(cfg)
    idx = np.array(ABDUCTION_ACTUATORS)
    ctrlrange = np.array(limited.mj_model.actuator_ctrlrange[idx])
    np.testing.assert_allclose(ctrlrange[:, 0], -0.44)
    np.testing.assert_allclose(ctrlrange[:, 1], 0.44)
    # Knee (non-abduction) ctrlrange is untouched by the abduction knob.
    knee_idx = np.array(KNEE_ACTUATORS)
    knee_range = np.array(limited.mj_model.actuator_ctrlrange[knee_idx])
    np.testing.assert_allclose(knee_range[:, 0], 0.425)
    np.testing.assert_allclose(knee_range[:, 1], 5.8)


# -- knee_target_max -----------------------------------------------------------


def test_knee_target_max_zero_matches_ctrlrange(env):
    """Default (0 = off): the motor-target bounds are exactly self._ctrlrange,
    the pre-existing goldens' clip path."""
    np.testing.assert_array_equal(
        np.array(env._target_hi), np.array(env._ctrlrange[:, 1])
    )
    np.testing.assert_array_equal(
        np.array(env._target_lo), np.array(env._ctrlrange[:, 0])
    )


def test_knee_target_max_lowers_upper_bound_when_set():
    """The knee (third-joint) ctrlrange upper bound is 5.8 rad, well past
    KNEE_SINGULARITY (3.2); knee_target_max=3.15 must actually lower it, not
    no-op against an already-tighter range."""
    cfg = wojtek_env.default_config()
    cfg.knee_target_max = 3.15
    clipped = wojtek_env.WojtekJoystick(cfg)
    knee_idx = np.array(KNEE_ACTUATORS)
    hi = np.array(clipped._target_hi)
    np.testing.assert_allclose(hi[knee_idx], 3.15)
    lo = np.array(clipped._target_lo)
    np.testing.assert_array_equal(lo, np.array(clipped._ctrlrange[:, 0]))
    # Non-knee actuators keep the model's ctrlrange upper bound.
    other_idx = np.array(ABDUCTION_ACTUATORS)
    np.testing.assert_array_equal(
        hi[other_idx], np.array(clipped._ctrlrange[other_idx, 1])
    )


def test_knee_target_max_binds_in_step():
    """End-to-end: a large action driving the knee actuators toward the raw
    ctrlrange ceiling (5.8) still gets clipped at knee_target_max."""
    cfg = wojtek_env.default_config()
    cfg.knee_target_max = 3.15
    clipped = wojtek_env.WojtekJoystick(cfg)
    state = jax.jit(clipped.reset)(jax.random.PRNGKey(0))
    action = jp.zeros(12).at[np.array(KNEE_ACTUATORS)].set(10.0)
    state = jax.jit(clipped.step)(state, action)
    knee_targets = np.array(state.info["motor_targets"])[np.array(KNEE_ACTUATORS)]
    assert np.all(knee_targets <= 3.15 + 1e-6)


# -- torque_limit reward term ---------------------------------------------------


def _reward_for_actuator_force(env, reset_state, force):
    data = reset_state.data.replace(actuator_force=force)
    info = reset_state.info
    contact = jp.zeros(4, dtype=bool)
    first_contact = jp.zeros(4, dtype=bool)
    rewards, _ = env._get_reward(
        data, info, jp.zeros(12), jp.zeros(12), first_contact, contact
    )
    return rewards["torque_limit"]


def test_torque_limit_zero_below_cap(env, reset_state):
    below = 0.5 * env._torque_cap  # well under torque_limit_frac (0.85) * cap
    assert float(_reward_for_actuator_force(env, reset_state, below)) == 0.0


def test_torque_limit_positive_above_cap(env, reset_state):
    above = 0.95 * env._torque_cap  # above torque_limit_frac (0.85) * cap
    assert float(_reward_for_actuator_force(env, reset_state, above)) > 0.0


# -- pd_kp / pd_kd ---------------------------------------------------------


def test_pd_gains_zero_leaves_xml_gains_untouched(env):
    """Default (0 = off): actuator gains stay the XML baseline
    (gainprm[:,0]=20, biasprm[:,1]=-20, biasprm[:,2]=-1)."""
    gainprm = np.array(env.mj_model.actuator_gainprm)
    biasprm = np.array(env.mj_model.actuator_biasprm)
    np.testing.assert_allclose(gainprm[:, 0], 20.0)
    np.testing.assert_allclose(biasprm[:, 1], -20.0)
    np.testing.assert_allclose(biasprm[:, 2], -1.0)


def test_pd_gains_override_both_kp_and_kd():
    cfg = wojtek_env.default_config()
    cfg.pd_kp = 40.0
    cfg.pd_kd = 1.4
    stiff = wojtek_env.WojtekJoystick(cfg)
    gainprm = np.array(stiff.mj_model.actuator_gainprm)
    biasprm = np.array(stiff.mj_model.actuator_biasprm)
    np.testing.assert_allclose(gainprm[:, 0], 40.0)
    np.testing.assert_allclose(biasprm[:, 1], -40.0)
    np.testing.assert_allclose(biasprm[:, 2], -1.4)
    np.testing.assert_allclose(biasprm[:, 0], 0.0)


def test_pd_gains_kp_only_keeps_xml_kd():
    cfg = wojtek_env.default_config()
    cfg.pd_kp = 40.0
    kp_only = wojtek_env.WojtekJoystick(cfg)
    biasprm = np.array(kp_only.mj_model.actuator_biasprm)
    np.testing.assert_allclose(biasprm[:, 2], -1.0)
