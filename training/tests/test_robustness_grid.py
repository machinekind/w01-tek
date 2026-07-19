"""Eval-only sim2real robustness-grid perturbations: Kt miscalibration
(alpha), actuator-bandwidth lag (lag_tau), and the speed-dependent
torque envelope (torque_envelope). See battery.py's
apply_kt_miscalibration/make_lagged_rollout_fns/apply_torque_envelope and
training/docs/configuration.md's "Robustness grid (eval-only)".
"""

import jax
import jax.numpy as jp
import numpy as np
import pytest

from wojtek_rl import env as wojtek_env
from wojtek_rl.battery import (
    apply_kt_miscalibration,
    apply_torque_envelope,
    lag_coeff,
    lag_update,
    make_lagged_rollout_fns,
    parse_torque_envelope,
    torque_envelope_limit,
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


def test_lag_coeff_exact_zero_is_pass_through():
    """lag_tau == 0 exactly (not just close to it) must not hit the
    dt_sub/lag_tau ZeroDivisionError -- a --torque-envelope-only grid cell
    passes lag_tau=0 straight through to here (see run_battery)."""
    assert lag_coeff(0.004, 0.0) == 1.0


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


# -- torque_envelope_limit / apply_torque_envelope ------------------------


def test_torque_envelope_limit_below_omega_b_is_static_cap():
    cap, omega_b, omega_0 = 9.0, 15.0, 28.0
    for w in (0.0, 10.0, omega_b, -10.0):  # sign-agnostic: back-EMF cares
        # about speed magnitude, not rotation direction
        assert float(torque_envelope_limit(jp.array(w), cap, omega_b, omega_0)) == (
            pytest.approx(cap)
        )


def test_torque_envelope_limit_midpoint_is_linear():
    cap, omega_b, omega_0 = 9.0, 15.0, 28.0
    mid = (omega_b + omega_0) / 2
    lim = torque_envelope_limit(jp.array(mid), cap, omega_b, omega_0)
    assert float(lim) == pytest.approx(cap / 2)


def test_torque_envelope_limit_above_omega_0_is_zero():
    cap, omega_b, omega_0 = 9.0, 15.0, 28.0
    for w in (omega_0, 50.0, -50.0):
        assert float(torque_envelope_limit(jp.array(w), cap, omega_b, omega_0)) == (
            pytest.approx(0.0)
        )


def test_torque_envelope_limit_vectorized_over_actuators():
    """The battery calls this once per substep over all 12 actuators at
    once -- confirm it broadcasts elementwise rather than only working on
    scalars."""
    cap, omega_b, omega_0 = 9.0, 15.0, 28.0
    qvel = jp.array([0.0, (omega_b + omega_0) / 2, 50.0])
    lim = torque_envelope_limit(qvel, cap, omega_b, omega_0)
    np.testing.assert_allclose(np.asarray(lim), [cap, cap / 2, 0.0])


def test_apply_torque_envelope_driving_quadrant_clamped_beyond_omega_0():
    cap, omega_b, omega_0 = 9.0, 15.0, 28.0
    qvel = jp.array(50.0)  # far beyond omega_0
    tau_driving = jp.array(cap)  # tau*qvel >= 0
    out = apply_torque_envelope(tau_driving, qvel, cap, omega_b, omega_0)
    assert float(out) == pytest.approx(0.0)


def test_apply_torque_envelope_braking_quadrant_keeps_static_cap():
    """Braking (tau*qvel < 0) is regenerative -- not limited by available
    bus voltage the way driving is -- so only the flat cap applies, even
    at a joint speed where the driving envelope would be fully zeroed."""
    cap, omega_b, omega_0 = 9.0, 15.0, 28.0
    qvel = jp.array(50.0)  # far beyond omega_0
    tau_braking = jp.array(-cap)  # tau*qvel < 0
    out = apply_torque_envelope(tau_braking, qvel, cap, omega_b, omega_0)
    assert float(out) == pytest.approx(-cap)


# -- parse_torque_envelope -------------------------------------------------


def test_parse_torque_envelope_none_passes_through():
    assert parse_torque_envelope(None) is None


def test_parse_torque_envelope_valid_spec():
    assert parse_torque_envelope("15,28") == (15.0, 28.0)


def test_parse_torque_envelope_bad_format_raises():
    with pytest.raises(ValueError):
        parse_torque_envelope("15")


def test_parse_torque_envelope_non_numeric_raises():
    with pytest.raises(ValueError):
        parse_torque_envelope("a,b")


def test_parse_torque_envelope_requires_omega_0_greater_than_omega_b():
    with pytest.raises(ValueError):
        parse_torque_envelope("28,15")
    with pytest.raises(ValueError):
        parse_torque_envelope("15,15")


def test_parse_torque_envelope_rejects_negative_omega_b():
    with pytest.raises(ValueError):
        parse_torque_envelope("-1,10")


# -- torque_envelope through the explicit-PD rollout -----------------------


def test_make_lagged_rollout_fns_requires_lag_or_envelope():
    """lag_tau<=0 with no torque_envelope means native -- run_battery
    branches on that before ever reaching here (see run_battery); this
    guards the branch's other half."""
    env = wojtek_env.WojtekJoystick()
    with pytest.raises(AssertionError):
        make_lagged_rollout_fns(env, 0.0)


def test_lag_zero_with_envelope_uses_explicit_pd():
    """A --torque-envelope-only cell (lag_tau=0) must not hit the
    lag_tau>0 assert -- the envelope alone justifies the explicit-PD
    path, with the lag filter as an exact passthrough (lag_coeff(_, 0) ==
    1.0)."""
    env = wojtek_env.WojtekJoystick()
    reset_fn, step_fn = make_lagged_rollout_fns(env, 0.0, torque_envelope=(15.0, 28.0))
    reset, step = jax.jit(reset_fn), jax.jit(step_fn)
    state = reset(jax.random.PRNGKey(0))
    action = jax.random.uniform(jax.random.PRNGKey(1), (12,), minval=-0.3, maxval=0.3)
    state = step(state, action)
    assert np.all(np.isfinite(np.asarray(state.data.qpos)))
    assert np.all(np.isfinite(np.asarray(state.data.qvel)))


def test_torque_envelope_none_is_bitwise_same_as_omitted_kwarg():
    """`torque_envelope=None` -- the default -- must be a genuine no-op:
    the clamp is skipped by a Python-level `if envelope is not None` (a
    static branch, not a traced one), so the resulting rollout is
    bit-for-bit identical whether the kwarg is spelled out or left off,
    i.e. adding the feature did not perturb the existing explicit-PD
    path at all when unused."""
    env = wojtek_env.WojtekJoystick()
    seed = 0
    n_steps = 8
    actions = jax.random.uniform(
        jax.random.PRNGKey(1), (n_steps, 12), minval=-0.3, maxval=0.3
    )

    reset_a, step_a = make_lagged_rollout_fns(env, 0.02)
    reset_b, step_b = make_lagged_rollout_fns(env, 0.02, torque_envelope=None)
    qpos_a, qvel_a, force_a = _rollout_n(
        jax.jit(reset_a), jax.jit(step_a), seed, actions
    )
    qpos_b, qvel_b, force_b = _rollout_n(
        jax.jit(reset_b), jax.jit(step_b), seed, actions
    )
    np.testing.assert_array_equal(qpos_a, qpos_b)
    np.testing.assert_array_equal(qvel_a, qvel_b)
    np.testing.assert_array_equal(force_a, force_b)


def test_torque_envelope_plateau_composes_with_alpha():
    """The envelope's cap term is whatever `limit` make_lagged_rollout_fns
    reads off `env.mj_model` at construction time -- if a caller applies
    apply_kt_miscalibration first (as run_battery does for alpha != 1.0),
    the plateau below omega_b is the alpha-scaled cap, not the model's
    original one."""
    env = wojtek_env.WojtekJoystick()
    base_cap = float(env.mj_model.actuator_forcerange[0, 1])
    alpha = 1.58
    apply_kt_miscalibration(env.mj_model, alpha)
    scaled_cap = float(env.mj_model.actuator_forcerange[0, 1])
    assert scaled_cap == pytest.approx(base_cap * alpha)

    omega_b, omega_0 = 15.0, 28.0
    plateau = torque_envelope_limit(jp.array(0.0), scaled_cap, omega_b, omega_0)
    assert float(plateau) == pytest.approx(scaled_cap)
    assert float(plateau) != pytest.approx(base_cap)


def test_torque_envelope_harsh_envelope_changes_trajectory():
    """Sanity check that the envelope actually constrains the plant: a
    harsh envelope (omega_b/omega_0 near zero, so almost any nonzero
    qvel saturates it) must diverge from the flat-cap trajectory."""
    env = wojtek_env.WojtekJoystick()
    seed = 0
    n_steps = 8
    actions = jax.random.uniform(
        jax.random.PRNGKey(1), (n_steps, 12), minval=-0.3, maxval=0.3
    )

    reset_flat, step_flat = make_lagged_rollout_fns(env, 0.02)
    reset_env, step_env = make_lagged_rollout_fns(
        env, 0.02, torque_envelope=(0.05, 0.1)
    )
    qpos_flat, _, _ = _rollout_n(
        jax.jit(reset_flat), jax.jit(step_flat), seed, actions
    )
    qpos_env, _, _ = _rollout_n(
        jax.jit(reset_env), jax.jit(step_env), seed, actions
    )
    assert not np.allclose(qpos_flat, qpos_env, atol=1e-4)
