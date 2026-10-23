"""Model-free tests of the pad's drive gate (quiet when idle, zero on stop).

Pure python -- no rclpy, no ROS graph. Run anywhere:
    pytest ros/src/wojtek_teleop/test/test_pad_drive.py
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wojtek_teleop.pad_drive import IDLE, LIVE, ZEROING, PadDrive  # noqa: E402

TIMEOUT = 0.5
SILENCE = 2.0


def make():
    return PadDrive(cmd_low=(-0.6, -0.4, -0.7), cmd_high=(1.2, 0.4, 0.7),
                    height_range=(0.09, 0.17), height_default=0.125,
                    timeout_s=TIMEOUT, silence_after_s=SILENCE)


def idle_joy(g, t0, t1, step=0.05):
    """The joy driver's autorepeat: centred sticks, frame after frame."""
    t = t0
    while t <= t1:
        g.joy(t, 0.0, 0.0, 0.0)
        t += step


def test_an_idle_pad_never_publishes():
    g = make()
    assert g.tick(0.0) is None
    idle_joy(g, 0.0, 30.0)
    assert g.tick(30.0) is None
    assert g.state == IDLE


def test_sticks_scale_into_the_asymmetric_box():
    g = make()
    g.joy(1.0, vx=1.0, vy=-0.5, yaw=0.5)
    vx, vy, yaw, h = g.tick(1.1)
    assert g.state == LIVE
    assert vx == pytest.approx(1.2)      # +1 -> box high
    assert vy == pytest.approx(-0.2)     # -0.5 -> half of box low
    assert yaw == pytest.approx(0.35)
    assert h == pytest.approx(0.125)     # default stance until stepped


def test_release_zeros_for_a_while_then_goes_quiet():
    g = make()
    g.joy(0.0, 1.0, 0.0, 0.0)
    assert g.tick(0.1)[0] == pytest.approx(1.2)
    # Sticks back to centre; the driver keeps repeating the centred state.
    idle_joy(g, 0.2, 5.0)
    # Within the timeout the command already reads zero, still LIVE.
    assert g.tick(0.3) == (0.0, 0.0, 0.0, 0.125)
    assert g.state == LIVE
    # The clock runs from the last deflected frame (t=0), not from the first
    # centred one. Past the timeout: the zeroing burst, sticks centred, pad
    # still here.
    assert g.tick(TIMEOUT + 0.1) == (0.0, 0.0, 0.0, 0.125)
    assert g.state == ZEROING
    assert not g.pad_lost(TIMEOUT + 0.1)
    assert g.tick(TIMEOUT + SILENCE - 0.1) == (0.0, 0.0, 0.0, 0.125)
    # Then silence, however long the driver keeps repeating.
    assert g.tick(TIMEOUT + SILENCE + 0.1) is None
    assert g.state == IDLE
    assert g.tick(5.0) is None


def test_a_lost_pad_zeros_then_goes_quiet():
    g = make()
    g.joy(0.0, 1.0, 0.0, 0.0)
    # No more joy frames at all: for the timeout the last command holds.
    assert g.tick(TIMEOUT - 0.1)[0] == pytest.approx(1.2)
    # Then the burst, and the node can tell the pad itself is gone.
    assert g.tick(TIMEOUT + 0.1) == (0.0, 0.0, 0.0, 0.125)
    assert g.state == ZEROING
    assert g.pad_lost(TIMEOUT + 0.1)
    assert g.tick(TIMEOUT + SILENCE + 0.1) is None
    assert g.state == IDLE


def test_driving_again_reopens_the_gate():
    g = make()
    g.joy(0.0, 1.0, 0.0, 0.0)
    idle_joy(g, 0.1, 10.0)
    assert g.tick(10.0) is None
    g.joy(10.05, 0.0, 0.0, -1.0)
    assert g.tick(10.1) == (0.0, 0.0, pytest.approx(-0.7), 0.125)
    assert g.state == LIVE


def test_height_step_is_input_and_rides_the_burst():
    g = make()
    idle_joy(g, 0.0, 1.0)
    assert g.tick(1.0) is None
    assert g.step_height(1.0, 0.005) == pytest.approx(0.13)
    # Centred sticks, but the new set-point has to reach the policy.
    assert g.tick(1.1) == (0.0, 0.0, 0.0, pytest.approx(0.13))
    assert g.tick(1.0 + TIMEOUT + 1.0) == (0.0, 0.0, 0.0, pytest.approx(0.13))
    assert g.state == ZEROING
    assert g.tick(1.0 + TIMEOUT + SILENCE + 0.1) is None


def test_height_is_clamped_and_survives_stops():
    g = make()
    for _ in range(20):
        g.step_height(0.0, 0.005)
    assert g.height == pytest.approx(0.17)
    g.joy(0.0, 0.5, 0.0, 0.0)
    assert g.tick(0.1)[3] == pytest.approx(0.17)
    g.joy(0.2, 0.0, 0.0, 0.0)
    assert g.tick(0.2 + TIMEOUT + 0.1)[3] == pytest.approx(0.17)
