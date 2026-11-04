"""ROS graph checks for cancel ordering and text-command handover.

Runs in a separate localhost-only domain. No robot or simulator is used.
"""

import os
from pathlib import Path
import sys
import time

import pytest

os.environ.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")
os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")
rclpy = pytest.importorskip("rclpy")
from geometry_msgs.msg import Twist  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from std_msgs.msg import Empty, String  # noqa: E402
from std_srvs.srv import SetBool  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wojtek_teleop.command_mux_node import CommandMuxNode  # noqa: E402
from wojtek_teleop.text_commander import TextCommander  # noqa: E402


@pytest.fixture
def graph():
    rclpy.init(
        args=["--ros-args", "-r", "cmd_vel:=/wojtek/cmd_vel/text"],
        domain_id=91,
    )
    mux = CommandMuxNode()
    text = TextCommander()
    helper = Node("mux_test_client")
    nav_command = helper.create_publisher(String, "/wojtek/nav_command", 10)
    pad = helper.create_publisher(Twist, "/wojtek/cmd_vel/pad", 10)
    cancel = helper.create_publisher(Empty, "/wojtek/nav/cancel", 10)
    enable = helper.create_client(SetBool, "/wojtek/autonomy_enable")
    seen = []
    helper.create_subscription(
        Twist, "/wojtek/cmd_vel/selected",
        lambda msg: seen.append((time.monotonic(), msg)), 10,
    )
    executor = SingleThreadedExecutor()
    for node in (mux, text, helper):
        executor.add_node(node)

    def spin(seconds):
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            executor.spin_once(timeout_sec=0.01)

    def until(predicate, timeout=2.0):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.01)
        assert predicate(), "ROS graph did not reach the expected state"

    def set_enabled(value):
        request = SetBool.Request()
        request.data = value
        future = enable.call_async(request)
        until(future.done)
        return future.result()

    try:
        until(lambda: enable.service_is_ready()
              and nav_command.get_subscription_count() == 1
              and pad.get_subscription_count() == 1
              and cancel.get_subscription_count() == 2
              and mux._pub.get_subscription_count() == 1)
        yield mux, text, nav_command, pad, cancel, seen, spin, until, set_enabled
    finally:
        executor.shutdown()
        for node in (mux, text, helper):
            node.destroy_node()
        rclpy.shutdown()


def test_enable_waits_for_cancel_to_settle_and_external_cancel_still_wins(graph):
    mux, _, _, _, cancel, _, spin, until, set_enabled = graph
    # Call callbacks directly so the timer cannot publish the cancel between
    # the two requests. This is the disable/enable ordering regression.
    assert mux._on_enable(SetBool.Request(data=False), SetBool.Response()).success
    assert not mux._on_enable(SetBool.Request(data=True), SetBool.Response()).success
    spin(0.1)
    assert not set_enabled(True).success
    spin(0.6)
    assert set_enabled(True).success
    spin(0.1)
    assert mux._mux.autonomy_enabled, "an old self-cancel must not revoke success"
    cancel.publish(Empty())
    until(lambda: not mux._mux.autonomy_enabled)
    assert not set_enabled(True).success
    spin(0.6)
    assert set_enabled(True).success


def test_manual_takeover_clears_text_before_autonomy_is_reenabled(graph):
    mux, text, command, pad, _, seen, spin, until, set_enabled = graph
    assert set_enabled(True).success
    command.publish(String(data="forward"))
    until(lambda: any(msg.linear.x > 0 for _, msg in seen))
    turn = Twist()
    turn.angular.z = -0.4
    pad.publish(turn)
    until(lambda: mux._mux.owner == "pad" and text._state._twist is None)
    spin(0.7)
    assert set_enabled(True).success
    enabled_at = time.monotonic()
    spin(0.2)
    assert not any(
        msg.linear.x != 0 or msg.angular.z != 0
        for received_at, msg in seen if received_at >= enabled_at
    ), "cancelled text motion must not resume before its old two-second deadline"
    command.publish(String(data="left"))
    until(lambda: any(received_at >= enabled_at and msg.angular.z > 0
                      for received_at, msg in seen))
