"""Model-free tests of the drive gate (the dead-man behind /cmd_vel)."""
import pytest

from wojtek_deck.drive import (BRIDGE, BRIDGE_MAY, DEADMAN, IDLE, LIVE, NOBODY,
                               OPERATOR, DriveGate)


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


# -- the bridge and a person's pad share the gate; the person wins -----------

def test_owner_is_nobody_until_someone_drives():
    g = make()
    assert g.owner(0.0) == NOBODY


def test_the_bridge_drives_when_no_person_does():
    g = make()
    assert g.command(1.0, vx=1.0, vy=0.0, yaw=0.0, bridge=True) is True
    vx, _, _, _ = g.tick(1.1)
    assert vx == pytest.approx(1.2)
    assert g.owner(1.1) == BRIDGE


def test_owner_falls_back_to_nobody_when_the_bridge_goes_quiet():
    g = make(timeout_s=0.5)
    g.command(0.0, vx=1.0, vy=0.0, yaw=0.0, bridge=True)
    assert g.owner(0.4) == BRIDGE
    assert g.owner(0.6) == NOBODY      # its frames stopped: the dead-man has it


def test_a_person_takes_the_robot_from_the_bridge_at_once():
    g = make(human_wins_s=1.0)
    g.command(0.0, vx=1.0, vy=0.0, yaw=0.0, bridge=True)
    g.command(0.1, vx=0.0, vy=0.0, yaw=1.0)          # a pad frame
    assert g.owner(0.1) == OPERATOR
    # The bridge keeps sending; its frames are dropped.
    assert g.command(0.2, vx=1.0, vy=0.0, yaw=0.0, bridge=True) is False
    vx, _, yaw, _ = g.tick(0.25)
    assert vx == pytest.approx(0.0)
    assert yaw == pytest.approx(0.7)


def test_the_bridge_gets_the_robot_back_after_the_person_lets_go():
    g = make(human_wins_s=1.0)
    g.command(0.0, vx=0.0, vy=0.0, yaw=1.0)
    assert g.command(0.9, vx=1.0, vy=0.0, yaw=0.0, bridge=True) is False
    assert g.owner(0.9) == OPERATOR
    assert g.command(1.1, vx=1.0, vy=0.0, yaw=0.0, bridge=True) is True
    assert g.owner(1.1) == BRIDGE


def test_a_person_holding_the_sticks_at_zero_still_owns_the_robot():
    g = make(human_wins_s=1.0)
    for i in range(30):                               # 3 s of zero frames
        g.command(i * 0.1, vx=0.0, vy=0.0, yaw=0.0)
    assert g.owner(2.95) == OPERATOR
    assert g.command(2.95, vx=1.0, vy=0.0, yaw=0.0, bridge=True) is False


def test_the_bridge_cannot_change_the_height():
    g = make()
    g.command(0.0, vx=0.0, vy=0.0, yaw=0.0, height=0.15)          # a person set it
    g.command(2.0, vx=0.5, vy=0.0, yaw=0.0, height=0.09, bridge=True)
    _, _, _, h = g.tick(2.0)
    assert h == pytest.approx(0.15)


def test_the_bridge_may_only_move_the_sticks_and_stop():
    assert BRIDGE_MAY == {"cmd", "stop"}
    # What the page can send and the bridge must not: arm and the service
    # calls, the height step.
    assert not {"call", "height"} & BRIDGE_MAY


def test_a_stop_is_taken_from_either_sender():
    g = make(timeout_s=0.5)
    g.command(0.0, vx=1.0, vy=0.0, yaw=0.0)           # a person is driving
    g.stop(0.1)                                       # the bridge's stop arrives
    assert g.tick(0.1) == (0.0, 0.0, 0.0, 0.125)
    assert g.state == DEADMAN


def test_the_bridge_leaving_stops_what_the_bridge_was_driving():
    g = make()
    g.command(0.0, vx=1.0, vy=0.0, yaw=0.0, bridge=True)
    g.client_gone(0.1, bridge=True, pads_left=1)
    assert g.tick(0.1) == (0.0, 0.0, 0.0, 0.125)


def test_the_bridge_leaving_does_not_stop_a_person():
    g = make()
    g.command(0.0, vx=1.0, vy=0.0, yaw=0.0)
    g.client_gone(0.1, bridge=True, pads_left=1)
    vx, _, _, _ = g.tick(0.1)
    assert vx == pytest.approx(1.2)


def test_the_last_pad_leaving_stops_what_a_person_was_driving():
    g = make()
    g.command(0.0, vx=1.0, vy=0.0, yaw=0.0)
    g.client_gone(0.1, bridge=False, pads_left=0)     # the bridge may still be connected
    assert g.tick(0.1) == (0.0, 0.0, 0.0, 0.125)


def test_a_pad_leaving_does_not_stop_the_bridge_or_another_pad():
    g = make()
    g.command(0.0, vx=1.0, vy=0.0, yaw=0.0, bridge=True)
    g.client_gone(0.1, bridge=False, pads_left=0)     # an idle page closed
    vx, _, _, _ = g.tick(0.1)
    assert vx == pytest.approx(1.2)
    g2 = make()
    g2.command(0.0, vx=1.0, vy=0.0, yaw=0.0)
    g2.client_gone(0.1, bridge=False, pads_left=1)    # another pad is still there
    vx, _, _, _ = g2.tick(0.1)
    assert vx == pytest.approx(1.2)
