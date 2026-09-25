"""Two drive sources on one /cmd_vel: the pad must not talk over the deck.

The bug this guards, seen on the robot: the joy driver repeats an idle
pad's state at 20 Hz, gamepad_teleop took that as input and published a
zero command 20 times a second, and policy_node -- which keeps whichever
/cmd_vel message came last -- alternated between the deck's command and
that zero. The robot stuttered.

A real graph in one process: the gamepad_teleop node under test, plus one
helper node that plays the joy driver (a centred pad, repeated at 20 Hz)
and the deck gateway (a command on /cmd_vel at 20 Hz), and listens on
/cmd_vel the way policy_node does. Needs rclpy (skipped on a host without
ROS); run in the dev container:

    pytest ros/src/wojtek_teleop/test/test_two_sources.py
"""
import os
import sys
import time
from pathlib import Path

import pytest

# Never let a test node reach the robot's graph: own domain, this host only.
os.environ.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")
os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")

rclpy = pytest.importorskip("rclpy")
from geometry_msgs.msg import Twist  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from sensor_msgs.msg import Joy  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wojtek_teleop.gamepad_teleop import GamepadTeleop  # noqa: E402

TEST_DOMAIN = 88
RATE_HZ = 20.0
DECK_VX = 0.3
AX_VX = 1


class DeckAndJoy(Node):
    """Stands in for the joy driver, the deck gateway, and policy_node's ear."""

    def __init__(self):
        super().__init__("fake_deck_and_joy")
        self._joy = self.create_publisher(Joy, "joy", 10)
        self._cmd = self.create_publisher(Twist, "cmd_vel", 10)
        self.seen = []                  # (monotonic s, Twist) as they arrive
        self.create_subscription(
            Twist, "cmd_vel",
            lambda m: self.seen.append((time.monotonic(), m)), 10)
        self.stick_vx = 0.0             # what the pad's left stick reads
        self.deck_vx = None             # None = deck not driving
        self.create_timer(1.0 / RATE_HZ, self._tick)

    def _tick(self):
        j = Joy()
        j.axes = [0.0] * 8
        j.axes[AX_VX] = self.stick_vx
        j.buttons = [0] * 11
        self._joy.publish(j)            # the driver's autorepeat
        if self.deck_vx is not None:
            t = Twist()
            t.linear.x = self.deck_vx
            self._cmd.publish(t)


@pytest.fixture(scope="module", autouse=True)
def _ros():
    rclpy.init(domain_id=TEST_DOMAIN)
    yield
    rclpy.shutdown()


@pytest.fixture
def graph():
    teleop = GamepadTeleop()
    helper = DeckAndJoy()
    ex = SingleThreadedExecutor()
    ex.add_node(teleop)
    ex.add_node(helper)

    def spin(seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            ex.spin_once(timeout_sec=0.02)

    spin(1.0)                           # discovery; the pad is idle meanwhile
    yield teleop, helper, spin
    ex.shutdown()
    teleop.destroy_node()
    helper.destroy_node()


def since(helper, t0):
    return [m for t, m in helper.seen if t >= t0]


def test_idle_pad_leaves_cmd_vel_to_the_deck(graph):
    teleop, helper, spin = graph
    assert helper.seen == [], "an idle pad must publish nothing"
    helper.deck_vx = DECK_VX
    t0 = time.monotonic() + 0.3         # let the first messages settle
    spin(1.8)
    got = since(helper, t0)
    assert got, "the deck's own commands must arrive"
    # Every message on /cmd_vel is the deck's: no zero from the pad among
    # them, and the rate is one source's, not two sources' interleaved.
    assert all(m.linear.x == pytest.approx(DECK_VX) for m in got)
    assert 0.7 * RATE_HZ * 1.5 <= len(got) <= 1.3 * RATE_HZ * 1.5


def test_pad_drives_when_deflected_and_hands_back_on_release(graph):
    teleop, helper, spin = graph
    helper.stick_vx = 1.0
    t0 = time.monotonic() + 0.3
    spin(1.3)
    got = since(helper, t0)
    assert got, "a deflected stick must drive"
    assert all(m.linear.x == pytest.approx(teleop.drive_high[0]) for m in got)
    # Release. Zeros for a while, then nothing at all, so another source
    # can take /cmd_vel without being fought.
    helper.stick_vx = 0.0
    t_rel = time.monotonic()
    spin(4.0)
    tail = since(helper, t_rel + 0.6)
    assert tail, "release must send the zeroing burst"
    assert all(m.linear.x == 0.0 for m in tail)
    last = max(t for t, _ in helper.seen)
    assert last - t_rel < 3.0, "the pad must go quiet after the burst"
    n = len(helper.seen)
    spin(1.0)
    assert len(helper.seen) == n, "an idle pad stays quiet"
