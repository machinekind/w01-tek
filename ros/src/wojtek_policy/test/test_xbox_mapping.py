"""JoyCommandMapper: deadzone shaping, asymmetric scaling, height rate."""

import numpy as np

from wojtek_policy.xbox_mapping import JoyCommandMapper

# fbb_loco_v8 ranges (policy_meta.json)
V8 = dict(
    cmd_low=(-0.8, -0.5, -1.0),
    cmd_high=(1.2, 0.5, 1.0),
    height_range=(0.09, 0.17),
)


def make(**kw):
    return JoyCommandMapper(**{**V8, **kw})


def test_deadzone_zeroes_small_inputs():
    m = make(deadzone=0.15)
    assert np.all(m.velocity(0.1, -0.14, 0.05) == 0.0)


def test_deadzone_is_continuous_and_saturates():
    m = make(deadzone=0.15)
    just_over = m.velocity(0.1501, 0.0, 0.0)[0]
    assert 0.0 < just_over < 0.01
    assert np.allclose(m.velocity(1.0, 1.0, 1.0), [1.2, 0.5, 1.0])


def test_asymmetric_vx_scaling():
    m = make(deadzone=0.0)
    assert np.isclose(m.velocity(1.0, 0, 0)[0], 1.2)  # full forward
    assert np.isclose(m.velocity(-1.0, 0, 0)[0], -0.8)  # full back
    assert np.isclose(m.velocity(0.5, 0, 0)[0], 0.6)


def test_height_integrates_slowly_and_clips():
    m = make(deadzone=0.0, height_rate=0.02, initial_height=0.13)
    for _ in range(50):  # 1 s of full stick up at 50 Hz
        h = m.step_height(1.0, 0.02)
    assert np.isclose(h, 0.15)
    for _ in range(500):  # 10 s: must clip at the trained max
        h = m.step_height(1.0, 0.02)
    assert np.isclose(h, 0.17)
    for _ in range(5000):
        h = m.step_height(-1.0, 0.02)
    assert np.isclose(h, 0.09)


def test_height_stick_deadzone_holds():
    m = make(deadzone=0.15, initial_height=0.13)
    for _ in range(100):
        h = m.step_height(0.1, 0.02)
    assert h == 0.13


def test_initial_height_clipped_to_range():
    assert make(initial_height=0.5).height == 0.17
