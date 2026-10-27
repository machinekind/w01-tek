"""The node around the pad gate: Joy messages in, /cmd_vel out, no graph.

Feeds sensor_msgs/Joy straight into the callback, drives the 20 Hz tick by
hand on a fake clock, and reads what the node would have published. Needs
rclpy (skipped on a host without ROS); run in the dev container:

    pytest ros/src/wojtek_teleop/test/test_gamepad_teleop_node.py
"""
import os
import sys
from pathlib import Path

import pytest

# Never let a test node reach the robot's graph: own domain, this host only.
os.environ.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")
os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")

rclpy = pytest.importorskip("rclpy")
from sensor_msgs.msg import Joy  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wojtek_teleop.gamepad_teleop import GamepadTeleop  # noqa: E402

TEST_DOMAIN = 88
TICK = 0.05
AX_VX, AX_YAW, AX_VY = 1, 0, 3
BTN_LB, BTN_RB = 4, 5


@pytest.fixture(scope="module", autouse=True)
def _ros():
    rclpy.init(domain_id=TEST_DOMAIN)
    yield
    rclpy.shutdown()


class FakeClock:
    t = 0.0


@pytest.fixture
def node():
    n = GamepadTeleop()
    clock = FakeClock()
    n._now = lambda: clock.t
    published = []
    n._pub_cmd.publish = published.append   # capture instead of publishing
    n.published = published
    n.clock = clock
    yield n
    n.destroy_node()


def joy(vx=0.0, vy=0.0, yaw=0.0, lb=0, rb=0):
    msg = Joy()
    msg.axes = [0.0] * 8
    msg.axes[AX_VX], msg.axes[AX_VY], msg.axes[AX_YAW] = vx, vy, yaw
    msg.buttons = [0] * 11
    msg.buttons[BTN_LB], msg.buttons[BTN_RB] = lb, rb
    return msg


def run(node, seconds, frame):
    """The joy driver's autorepeat and the node's tick, interleaved."""
    end = node.clock.t + seconds
    while node.clock.t < end:
        node._on_joy(frame())
        node._tick()
        node.clock.t += TICK


def test_idle_pad_publishes_nothing(node):
    run(node, 5.0, lambda: joy())
    assert node.published == []


def test_a_slightly_off_centre_idle_pad_is_still_idle(node):
    # Bluetooth pads rest a little off centre; that is what the deadzone is
    # for, and it must not turn a resting pad into a driver.
    run(node, 3.0, lambda: joy(vx=0.05, yaw=-0.05))
    assert node.published == []


def test_full_stick_reaches_the_scaled_box_not_the_trained_edge(node):
    """speed_scale (default 0.4) shrinks every axis of the command box: the
    trained edge is the policy's top speed, too fast to drive indoors."""
    assert node.drive_high == pytest.approx([0.4 * v for v in node.cmd_high])
    assert node.drive_low == pytest.approx([0.4 * v for v in node.cmd_low])


def test_deflected_stick_drives_and_release_zeros_then_stops(node):
    run(node, 1.0, lambda: joy(vx=1.0))
    assert node.published, "a deflected stick must publish"
    assert all(m.linear.x == pytest.approx(node.drive_high[0])
               for m in node.published)
    n_live = len(node.published)
    # Release: the driver keeps repeating the centred state.
    run(node, 5.0, lambda: joy())
    tail = node.published[n_live:]
    assert tail, "release must publish the zeroing burst"
    assert all(m.linear.x == 0.0 and m.angular.z == 0.0 for m in tail)
    # About 0.5 s of zero-valued LIVE plus the 2 s burst, then nothing.
    assert 2.0 / TICK <= len(tail) <= 3.0 / TICK
    assert all(m.linear.z == pytest.approx(node.height_default) for m in tail)


def test_height_step_reaches_the_policy_with_sticks_at_rest(node):
    run(node, 1.0, lambda: joy())
    assert node.published == []
    node._on_joy(joy(rb=1))       # press RB
    node._on_joy(joy())           # release RB
    run(node, 5.0, lambda: joy())
    assert node.published, "a height step must publish"
    assert all(m.linear.x == 0.0 for m in node.published)
    assert all(m.linear.z == pytest.approx(node.height_default + 0.005)
               for m in node.published)
    assert len(node.published) <= 3.0 / TICK


def test_lost_pad_zeros_then_stops(node):
    run(node, 1.0, lambda: joy(vx=1.0))
    n_live = len(node.published)
    # No more joy frames at all: pad off, out of range, driver gone.
    end = node.clock.t + 5.0
    while node.clock.t < end:
        node._tick()
        node.clock.t += TICK
    tail = node.published[n_live:]
    assert tail
    # For the timeout the last command holds, then zeros, then nothing.
    assert tail[-1].linear.x == 0.0
    assert len(tail) <= 3.0 / TICK
