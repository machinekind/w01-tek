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


def test_tracking_far_is_live_under_relative_kernels(env, reset_state):
    """The far mix-in must blend into the relative kernels too. It used to
    apply only in the absolute branch, so terrain_blind_v3 (tracking_relative
    with tracking_far dropped as inert) lost the far-field gradient that
    fixed stiff_b's dead spin. A robot at rest under a pure-spin command
    must score visibly higher tracking_ang_vel with the blend on."""
    r = env._config.reward
    saved_rel, saved_w = r.tracking_relative, r.tracking_far_weight
    info = dict(reset_state.info)
    info["command"] = jp.array([0.0, 0.0, 1.5, 0.125])

    def k_ang():
        rewards, _ = env._get_reward(
            reset_state.data, info, jp.zeros(12), jp.zeros(12),
            jp.zeros(4, dtype=bool), jp.zeros(4, dtype=bool),
        )
        return float(rewards["tracking_ang_vel"])

    try:
        r.tracking_relative = True
        r.tracking_far_weight = 0.0
        bare = k_ang()
        r.tracking_far_weight = 0.25
        blended = k_ang()
    finally:
        r.tracking_relative = saved_rel
        r.tracking_far_weight = saved_w
    # at rest: bare relative kernel ~exp(-1/rel_sigma), far term
    # ~exp(-2.25/2.5) -- the blend must lift the reward well clear
    assert blended > 3.0 * bare
    assert blended <= 1.0


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


# -- command.arc_prob ---------------------------------------------------------


def test_sample_command_arc_mode_holds_forward_curve():
    """arc_prob=1: every draw is a forward arc -- vx inside arc_vx, vy
    zeroed, wz kept live inside its range."""
    cfg = wojtek_env.default_config()
    cfg.command.arc_prob = 1.0
    cfg.command.zero_prob = 0.0
    arc_env = wojtek_env.WojtekJoystick(cfg)
    keys = jax.random.split(jax.random.PRNGKey(0), 64)
    cmds = np.array(jax.vmap(arc_env._sample_command)(keys))
    np.testing.assert_allclose(cmds[:, 1], 0.0)
    assert np.all(cmds[:, 0] >= cfg.command.arc_vx[0])
    assert np.all(cmds[:, 0] <= cfg.command.arc_vx[1])
    assert np.all(cmds[:, 2] >= cfg.command.wz[0])
    assert np.all(cmds[:, 2] <= cfg.command.wz[1])


def test_sample_command_arc_default_off(env):
    """Default arc_prob=0 keeps the plain box: vy is drawn, not zeroed."""
    keys = jax.random.split(jax.random.PRNGKey(0), 64)
    cmds = np.array(jax.vmap(env._sample_command)(keys))
    assert np.any(np.abs(cmds[:, 1]) > 1e-6)


# -- command.pure_back_prob ---------------------------------------------------


@pytest.fixture(scope="module")
def env_without_back_keys():
    """Env whose command block predates pure_back_prob/back_vx, so the
    sampler falls back through `.get`."""
    cfg = wojtek_env.default_config()
    del cfg.command.pure_back_prob
    del cfg.command.back_vx
    return wojtek_env.WojtekJoystick(cfg)


def test_sample_command_back_mode_holds_clean_reverse():
    """pure_back_prob=1: every draw is a clean reverse -- vx inside back_vx
    (negative throughout), vy and wz zeroed."""
    cfg = wojtek_env.default_config()
    cfg.command.pure_back_prob = 1.0
    cfg.command.zero_prob = 0.0
    back_env = wojtek_env.WojtekJoystick(cfg)
    keys = jax.random.split(jax.random.PRNGKey(0), 64)
    cmds = np.array(jax.vmap(back_env._sample_command)(keys))
    np.testing.assert_allclose(cmds[:, 1], 0.0)
    np.testing.assert_allclose(cmds[:, 2], 0.0)
    assert np.all(cmds[:, 0] >= cfg.command.back_vx[0])
    assert np.all(cmds[:, 0] <= cfg.command.back_vx[1])
    assert np.all(cmds[:, 0] < 0.0)


def test_sample_command_back_overrides_fast():
    """pure_back applies after pure_fast, so a both-fire draw is backward."""
    cfg = wojtek_env.default_config()
    cfg.command.pure_fast_prob = 1.0
    cfg.command.pure_back_prob = 1.0
    cfg.command.zero_prob = 0.0
    both = wojtek_env.WojtekJoystick(cfg)
    keys = jax.random.split(jax.random.PRNGKey(0), 32)
    cmds = np.array(jax.vmap(both._sample_command)(keys))
    assert np.all(cmds[:, 0] < 0.0)


def test_sample_command_back_default_off(env):
    """Default pure_back_prob=0 keeps the plain box: vy and wz are drawn,
    not zeroed, and forward commands still appear."""
    keys = jax.random.split(jax.random.PRNGKey(0), 64)
    cmds = np.array(jax.vmap(env._sample_command)(keys))
    assert np.any(np.abs(cmds[:, 1]) > 1e-6)
    assert np.any(np.abs(cmds[:, 2]) > 1e-6)
    assert np.any(cmds[:, 0] > 0.0)


def test_sample_command_back_off_matches_keyless_config(env, env_without_back_keys):
    """pure_back_prob=0 consumes no randomness: the drawn commands are
    bit-identical to a config that never had the keys."""
    keys = jax.random.split(jax.random.PRNGKey(0), 64)
    cmds = np.array(jax.vmap(env._sample_command)(keys))
    legacy = np.array(jax.vmap(env_without_back_keys._sample_command)(keys))
    np.testing.assert_array_equal(cmds, legacy)


# -- reward.orientation_tol_deg -----------------------------------------------


def _tilted_data(env, reset_state, tilt_deg):
    """reset_state's data with the base pitched tilt_deg from vertical.

    _gravity_body reads the orientation sensor, not qpos, so the quaternion
    goes into sensordata.
    """
    half = np.radians(tilt_deg) / 2.0
    quat = jp.array([np.cos(half), 0.0, np.sin(half), 0.0])
    adr = env._sensor_adr["orientation"]
    sensordata = reset_state.data.sensordata.at[adr : adr + 4].set(quat)
    return reset_state.data.replace(sensordata=sensordata)


def _sin2(deg):
    return float(np.square(np.sin(np.radians(deg))))


def _orientation_reward(env, data, info):
    zeros4 = jp.zeros(4, dtype=bool)
    rewards, _ = env._get_reward(
        data, info, jp.zeros(12), jp.zeros(12), zeros4, zeros4
    )
    return float(rewards["orientation"])


def test_orientation_default_matches_legacy_expression(env, reset_state):
    """Default orientation_tol_deg=0: the term is exactly the pre-existing
    sum(square(gravity[:2])), no cone arithmetic in the path at all."""
    assert env._orientation_tol == 0.0
    for tilt in (0.0, 10.0, 25.0):
        data = _tilted_data(env, reset_state, tilt)
        legacy = float(jp.sum(jp.square(env._gravity_body(data)[:2])))
        assert _orientation_reward(env, data, reset_state.info) == legacy


def test_orientation_cone_zero_inside_and_offset_outside(env, reset_state):
    """A 15 deg cone: nothing inside it, and outside the penalty is the tilt's
    sin^2 less the cone's, continuous at the edge. Scored on the shared reset
    state -- the cone config leaves the model identical, so this is the same
    physics with the cone on."""
    cfg = wojtek_env.default_config()
    cfg.reward.orientation_tol_deg = 15.0
    coned = wojtek_env.WojtekJoystick(cfg)
    assert coned._orientation_tol == pytest.approx(_sin2(15.0))

    inside = _tilted_data(coned, reset_state, 10.0)
    assert _orientation_reward(coned, inside, reset_state.info) == 0.0

    edge = _tilted_data(coned, reset_state, 15.0)
    assert _orientation_reward(coned, edge, reset_state.info) == pytest.approx(
        0.0, abs=1e-7
    )

    outside = _tilted_data(coned, reset_state, 25.0)
    assert _orientation_reward(coned, outside, reset_state.info) == pytest.approx(
        _sin2(25.0) - _sin2(15.0), abs=1e-6
    )
    # The anti-nosedive gradient survives: a big tilt still costs.
    nosedive = _tilted_data(coned, reset_state, 60.0)
    assert _orientation_reward(coned, nosedive, reset_state.info) > 0.5
