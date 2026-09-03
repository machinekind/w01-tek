"""Model-free tests of the drive gate (the dead-man behind /cmd_vel)."""
import pytest

from wojtek_deck.drive import DEADMAN, IDLE, LIVE, DriveGate


def make(**kw):
    return DriveGate(cmd_low=(-0.6, -0.4, -0.7), cmd_high=(1.2, 0.4, 0.7),
                     height_range=(0.09, 0.17), height_default=0.125, **kw)


def test_silent_until_the_pad_speaks():
    g = make()
    assert g.tick(0.0) is None
    assert g.tick(100.0) is None
    assert g.state == IDLE


def test_sticks_scale_into_the_asymmetric_box():
    g = make()
    g.command(1.0, vx=1.0, vy=-0.5, yaw=0.5)
    vx, vy, yaw, h = g.tick(1.1)
    assert g.state == LIVE
    assert vx == pytest.approx(1.2)      # +1 -> box high
    assert vy == pytest.approx(-0.2)     # -0.5 -> half of box low
    assert yaw == pytest.approx(0.35)
    assert h == pytest.approx(0.125)     # default stance until set


def test_stick_values_are_clipped_to_unit():
    g = make()
    g.command(0.0, vx=7.0, vy=0.0, yaw=-9.0)
    vx, _, yaw, _ = g.tick(0.0)
    assert vx == pytest.approx(1.2)
    assert yaw == pytest.approx(-0.7)


def test_deadman_zeros_then_goes_silent():
    g = make(timeout_s=0.5, silence_after_s=2.0)
    g.command(0.0, vx=1.0, vy=0.0, yaw=0.0, height=0.15)
    assert g.tick(0.4)[0] == pytest.approx(1.2)
    # link gone: zeros, height held
    out = g.tick(0.6)
    assert g.state == DEADMAN
    assert out == pytest.approx((0.0, 0.0, 0.0, 0.15))
    assert g.tick(2.4) == pytest.approx((0.0, 0.0, 0.0, 0.15))
    # burst over: silence, so another source can take /cmd_vel
    assert g.tick(2.6) is None
    assert g.state == IDLE


def test_fresh_frame_recovers_from_deadman():
    g = make()
    g.command(0.0, vx=1.0, vy=0.0, yaw=0.0)
    g.tick(1.0)
    assert g.state == DEADMAN
    g.command(1.1, vx=0.5, vy=0.0, yaw=0.0)
    assert g.tick(1.2)[0] == pytest.approx(0.6)
    assert g.state == LIVE


def test_explicit_stop_starts_the_burst_immediately():
    g = make(timeout_s=0.5, silence_after_s=2.0)
    g.command(10.0, vx=1.0, vy=0.0, yaw=0.0, height=0.11)
    g.stop(10.1)
    out = g.tick(10.1)
    assert g.state == DEADMAN
    assert out == pytest.approx((0.0, 0.0, 0.0, 0.11))
    assert g.tick(12.05) is not None     # still bursting
    assert g.tick(12.2) is None          # 2 s after the (backdated) stamp


def test_stop_before_any_frame_stays_silent():
    g = make()
    g.stop(5.0)
    assert g.tick(5.0) is None
    assert g.state == IDLE


def test_height_is_clamped_and_held():
    g = make()
    g.command(0.0, 0.0, 0.0, 0.0, height=0.5)
    assert g.tick(0.0)[3] == pytest.approx(0.17)
    g.command(0.1, 0.0, 0.0, 0.0)        # no height in the frame: held
    assert g.tick(0.1)[3] == pytest.approx(0.17)
    assert g.step_height(-0.005) == pytest.approx(0.165)
    assert g.step_height(-1.0) == pytest.approx(0.09)
