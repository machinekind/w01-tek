"""Eval-only sim2real robustness-grid perturbations: Kt miscalibration
(alpha) and actuator-bandwidth lag (lag_tau). See battery.py's
apply_kt_miscalibration/make_lagged_rollout_fns and
training/docs/configuration.md's "Robustness grid (eval-only)".
"""

import jax
import numpy as np
import pytest

from wojtek_rl import env as wojtek_env
from wojtek_rl.battery import (
    apply_kt_miscalibration,
    lag_coeff,
    lag_update,
    make_lagged_rollout_fns,
)

# -- apply_kt_miscalibration --------------------------------------------


def test_alpha_scales_gains_and_torque_cap():
    env = wojtek_env.WojtekJoystick()
    m = env.mj_model
    kp0 = m.actuator_gainprm[:, 0].copy()
    bias1_0 = m.actuator_biasprm[:, 1].copy()
    bias2_0 = m.actuator_biasprm[:, 2].copy()
    force0 = m.actuator_forcerange.copy()

    alpha = 1.58
    apply_kt_miscalibration(m, alpha)

    np.testing.assert_allclose(m.actuator_gainprm[:, 0], kp0 * alpha)
    np.testing.assert_allclose(m.actuator_biasprm[:, 1], bias1_0 * alpha)
    np.testing.assert_allclose(m.actuator_biasprm[:, 2], bias2_0 * alpha)
    np.testing.assert_allclose(m.actuator_forcerange, force0 * alpha)


def test_alpha_one_is_bitwise_noop():
    env = wojtek_env.WojtekJoystick()
    m = env.mj_model
    gainprm0 = m.actuator_gainprm.copy()
    biasprm0 = m.actuator_biasprm.copy()
    force0 = m.actuator_forcerange.copy()

    apply_kt_miscalibration(m, 1.0)

    assert np.array_equal(m.actuator_gainprm, gainprm0)
    assert np.array_equal(m.actuator_biasprm, biasprm0)
    assert np.array_equal(m.actuator_forcerange, force0)


def test_alpha_works_on_default_pd_kp_zero_config():
    """default_config's pd_kp/pd_kd/max_torque are all 0 (kept XML kp=20/
    kd=1/forcerange=+-9); apply_kt_miscalibration must scale those
    XML-default EFFECTIVE values the same as it would an explicit pd_kp --
    the stiffness runs this grid probes range from pd_kp=40 up, but a
    kp=0 (XML-default) config must not be a silent no-scale special case."""
    cfg = wojtek_env.default_config()
    assert cfg.pd_kp == 0.0
    assert cfg.max_torque == 0.0
    env = wojtek_env.WojtekJoystick(cfg)
    m = env.mj_model
    assert m.actuator_gainprm[0, 0] == 20.0  # confirms pd_kp=0 kept the XML default
    assert m.actuator_biasprm[0, 2] == -1.0
    assert m.actuator_forcerange[0, 1] == 9.0

    apply_kt_miscalibration(m, 2.0)

    assert m.actuator_gainprm[0, 0] == 40.0
    assert m.actuator_biasprm[0, 1] == -40.0
    assert m.actuator_biasprm[0, 2] == -2.0
    assert m.actuator_forcerange[0, 1] == 18.0
    assert m.actuator_forcerange[0, 0] == -18.0


# -- lag_coeff / lag_update ------------------------------------------------


def test_lag_coeff_tau_to_zero_is_pass_through():
    assert lag_coeff(0.004, 1e-9) == pytest.approx(1.0, abs=1e-9)


def test_lag_update_tau_to_zero_passes_through_in_one_substep():
    coeff = lag_coeff(0.004, 1e-9)
    assert float(lag_update(0.0, 5.0, coeff)) == pytest.approx(5.0)


def test_lag_update_step_response_matches_closed_form():
    """A constant-coefficient EMA's step response after k updates from 0
    is target*(1 - (1-coeff)^k); with coeff = 1-exp(-dt/tau), that is
    target*(1 - exp(-k*dt/tau)) -- exactly the substep-loop's per-step
    filter, decoupled here from any physics."""
    dt, tau, target = 0.004, 0.02, 5.0
    coeff = lag_coeff(dt, tau)
    x = 0.0
    for k in range(1, 11):
        x = float(lag_update(x, target, coeff))
        expected = target * (1 - np.exp(-k * dt / tau))
        assert x == pytest.approx(expected, rel=1e-6)


# -- explicit-PD pipeline vs native, tau -> 0 ---------------------------


def _rollout_n(reset, step, seed, actions):
    state = reset(jax.random.PRNGKey(seed))
    qpos, qvel, force = [], [], []
    for a in actions:
        state = step(state, a)
        qpos.append(np.array(state.data.qpos))
        qvel.append(np.array(state.data.qvel))
        force.append(np.array(state.data.actuator_force))
    return np.array(qpos), np.array(qvel), np.array(force)


# Single-substep agreement between the two pipelines is float32-machine-
# precision (~1e-6 qpos / ~1e-7 actuator_force, checked by hand while
# building this test against a fresh reset). The tolerances below are much
# looser because this is a contact-rich, chaotic system: kp=20 (default)
# drives fast, stiff-contact dynamics, so a ULP-level difference in one
# substep's torque shifts exactly when a contact activates a step or two
# later, and the resulting qpos/qvel divergence compounds across the
# 8-step*5-substep rollout -- not a sign of a formula mismatch (that would
# show up already at substep 1, not growing steadily over 40 substeps).
# Measured max |diff| over the 8 control steps below: default-config qpos
# 0.0171 / qvel 1.037 / actuator_force 0.999; latency-enabled qpos 0.0106 /
# qvel 0.724 / actuator_force 0.905. The full-battery keeper equivalence
# run (run_battery's own gate, aggregate track_err_rms/vel_err/vibration
# over ~700-step scenarios) is the tolerance that actually matters; these
# short-rollout checks exist to catch a gross formula bug (e.g. the
# ctrllimited-vs-forcerange mixup this test caught during development,
# which produced 3-4x larger divergence over the same 8 steps) fast, in a
# unit test.
_QPOS_ATOL = 0.025
_QVEL_ATOL = 1.5
_FORCE_ATOL = 1.3


def test_explicit_pd_matches_native_at_tiny_tau_default_config():
    """Default config (latency disabled, action_delay=1): the explicit-PD
    substep loop at lag_tau=1e-9 must track the native pipeline closely
    over a short rollout. Not bitwise -- the PD force is recomputed in JAX
    here instead of the model's internal gain/bias actuator; see the
    tolerance note above."""
    env = wojtek_env.WojtekJoystick()
    seed = 0
    n_steps = 8
    actions = jax.random.uniform(
        jax.random.PRNGKey(1), (n_steps, 12), minval=-0.3, maxval=0.3
    )

    reset_n, step_n = jax.jit(env.reset), jax.jit(env.step)
    reset_l, step_l = make_lagged_rollout_fns(env, 1e-9)
    reset_l, step_l = jax.jit(reset_l), jax.jit(step_l)

    qpos_n, qvel_n, force_n = _rollout_n(reset_n, step_n, seed, actions)
    qpos_l, qvel_l, force_l = _rollout_n(reset_l, step_l, seed, actions)

    assert np.all(np.isfinite(qpos_l)) and np.all(np.isfinite(qvel_l))
    np.testing.assert_allclose(qpos_n, qpos_l, atol=_QPOS_ATOL)
    np.testing.assert_allclose(qvel_n, qvel_l, atol=_QVEL_ATOL)
    np.testing.assert_allclose(force_n, force_l, atol=_FORCE_ATOL)


def test_explicit_pd_matches_native_at_tiny_tau_latency_enabled():
    """Same equivalence check with latency.enable=True (the keeper run's
    actual config) -- exercises _explicit_pd_substeps' prev/new ctrl
    switch-at-substep-d path, the branch the default-config test above
    does not reach. See the tolerance note above."""
    cfg = wojtek_env.default_config()
    cfg.latency.enable = True
    cfg.latency.min_substeps = 0
    cfg.latency.max_substeps = 5
    env = wojtek_env.WojtekJoystick(cfg)
    seed = 0
    n_steps = 8
    actions = jax.random.uniform(
        jax.random.PRNGKey(1), (n_steps, 12), minval=-0.3, maxval=0.3
    )

    reset_n, step_n = jax.jit(env.reset), jax.jit(env.step)
    reset_l, step_l = make_lagged_rollout_fns(env, 1e-9)
    reset_l, step_l = jax.jit(reset_l), jax.jit(step_l)

    qpos_n, qvel_n, force_n = _rollout_n(reset_n, step_n, seed, actions)
    qpos_l, qvel_l, force_l = _rollout_n(reset_l, step_l, seed, actions)

    assert np.all(np.isfinite(qpos_l)) and np.all(np.isfinite(qvel_l))
    np.testing.assert_allclose(qpos_n, qpos_l, atol=_QPOS_ATOL)
    np.testing.assert_allclose(qvel_n, qvel_l, atol=_QVEL_ATOL)
    np.testing.assert_allclose(force_n, force_l, atol=_FORCE_ATOL)


def test_lag_tau_changes_the_trajectory():
    """Sanity check that lag_tau actually does something: a real
    (non-tiny) lag must diverge from the tau->0 trajectory."""
    env = wojtek_env.WojtekJoystick()
    seed = 0
    n_steps = 8
    actions = jax.random.uniform(
        jax.random.PRNGKey(1), (n_steps, 12), minval=-0.3, maxval=0.3
    )

    reset_tiny, step_tiny = make_lagged_rollout_fns(env, 1e-9)
    reset_lag, step_lag = make_lagged_rollout_fns(env, 0.02)
    qpos_tiny, _, _ = _rollout_n(
        jax.jit(reset_tiny), jax.jit(step_tiny), seed, actions
    )
    qpos_lag, _, _ = _rollout_n(
        jax.jit(reset_lag), jax.jit(step_lag), seed, actions
    )
    assert not np.allclose(qpos_tiny, qpos_lag, atol=1e-4)
