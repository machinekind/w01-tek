"""Model-free tests of the gyro-vib sensor model. They check that the
resonator rings at half the control rate, that the config default keeps it
off, that a gain of 0 leaves both the state and the observation exactly as
they were, and that _build_obs adds the state to the actor's gyro only.
The update inside env.step needs a live env, so tests/integration covers
it.
"""

import numpy as np

from wojtek_rl import env as wojtek_env
from wojtek_rl.base import VIB_MIX, gyro_vib_step


def test_default_config_disables_gyro_vib():
    cfg = wojtek_env.default_config().obs_noise
    assert cfg.gyro_vib == 0.0
    assert 0.0 < cfg.gyro_vib_decay < 1.0


def test_vib_mix_rows_are_unit_and_independent():
    assert VIB_MIX.shape == (3, 12)
    np.testing.assert_allclose(np.linalg.norm(VIB_MIX, axis=1), 1.0, rtol=1e-6)
    assert np.linalg.matrix_rank(VIB_MIX) == 3


# A torque-change pattern that matches VIB_MIX row 0. A uniform pattern
# would sum to zero against the row's alternating signs and drive nothing.
TAU_X = np.repeat([1.0, -1.0, 1.0, -1.0], 3)


def test_dc_drive_is_attenuated():
    # A drive that never changes settles to a constant gain*d/(1+decay).
    # The resonator damps steady input. Only input near its ringing
    # frequency gets amplified.
    drive = float(VIB_MIX[0] @ TAU_X)
    x = np.zeros(3)
    for _ in range(200):
        x = gyro_vib_step(x, TAU_X, gain=0.1, decay=0.9)
    np.testing.assert_allclose(x[0], 0.1 * drive / 1.9, rtol=1e-3)


def test_resonance_amplifies_alternating_drive():
    # A drive that flips sign every step sits exactly at the resonant
    # frequency. The state then flips sign every step too and settles at
    # amplitude gain*d/(1-decay), 19 times the constant-drive response at
    # decay 0.9. That amplification is what lets the corruption feed on
    # the policy's own reaction.
    drive = float(VIB_MIX[0] @ TAU_X)
    x = np.zeros(3)
    hist = []
    for i in range(400):
        x = gyro_vib_step(x, ((-1.0) ** i) * TAU_X, gain=0.1, decay=0.9)
        hist.append(x[0])
    assert np.sign(hist[-1]) == -np.sign(hist[-2]) != 0
    np.testing.assert_allclose(abs(x[0]), 0.1 * drive / 0.1, rtol=1e-2)


def test_zero_gain_is_exactly_zero():
    # env.step runs the recurrence on every step now, gain or no gain, so
    # that a sweep can change the gain without a recompile. A run with the
    # loop off must therefore still see a state of exactly zero. A
    # rounding-sized value would already be a changed observation.
    rng = np.random.default_rng(0)
    x = np.zeros(3)
    for _ in range(50):
        x = gyro_vib_step(x, rng.standard_normal(12), gain=0.0, decay=0.9)
        assert np.all(np.asarray(x) == 0.0)


def test_zero_gain_clears_a_ringing_state():
    # Switching the gain off mid-episode (a bisection probe does exactly
    # that between rollouts) must not leave the resonator ringing on.
    x = np.array([0.4, -0.2, 0.1])
    x = gyro_vib_step(x, TAU_X, gain=0.0, decay=0.9)
    assert np.all(np.asarray(x) == 0.0)


def test_zero_gain_leaves_the_observation_untouched():
    # Checkpoints have to keep working. At gain 0 the actor's gyro must
    # read the same as it did when the update was skipped altogether,
    # which is what an env with no gyro_vib key in info produces.
    import jax.numpy as jp

    from test_gyro_bias import _make_stub

    zero = gyro_vib_step(jp.zeros(3), jp.zeros(12), gain=0.0, decay=0.9)
    with_key = _make_stub()._build_obs(None, {"gyro_vib": zero})
    without_key = _make_stub()._build_obs(None, {})
    np.testing.assert_array_equal(with_key["state"], without_key["state"])
    np.testing.assert_array_equal(
        with_key["privileged_state"], without_key["privileged_state"]
    )


def test_build_obs_adds_vib_to_actor_only():
    from test_gyro_bias import GYRO, FOO, _make_stub
    import jax.numpy as jp

    vib = jp.array([0.3, -0.1, 0.2])
    obs = _make_stub()._build_obs(None, {"gyro_vib": vib})
    np.testing.assert_allclose(obs["state"][:3], GYRO + vib, rtol=1e-6)
    np.testing.assert_allclose(obs["state"][3:], FOO, rtol=1e-6)
    np.testing.assert_allclose(
        obs["privileged_state"], jp.concatenate([GYRO, FOO]), rtol=1e-6
    )


def test_build_obs_sums_bias_and_vib():
    from test_gyro_bias import GYRO, _make_stub
    import jax.numpy as jp

    bias = jp.array([0.05, 0.0, -0.05])
    vib = jp.array([0.3, -0.1, 0.2])
    obs = _make_stub()._build_obs(None, {"gyro_bias": bias, "gyro_vib": vib})
    np.testing.assert_allclose(obs["state"][:3], GYRO + bias + vib, rtol=1e-6)
