#!/usr/bin/env python3
"""Short text commands to Twist, with timeout and cancellation.

`wojtek/nav_command` accepts forward, left, right and stop. Motion repeats
on `cmd_vel` at 20 Hz; remap this output to `/wojtek/cmd_vel/text` when using
the command mux. A stop, timeout or `/wojtek/nav/cancel` clears the command,
emits one zero Twist, then leaves the source silent. Only a new text command
can restart it. linear.z stays zero to request the policy's default height.
"""
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Empty, String

from wojtek_teleop.text_command_state import CommandState

DRIVE_TICK_HZ = 20.0     # /cmd_vel publish rate while active (same as consoles)


class TextCommander(Node):
    def __init__(self):
        super().__init__("text_commander")
        v_forward = self.declare_parameter("v_forward", 0.3).value
        w_turn = self.declare_parameter("w_turn", 0.5).value
        timeout = self.declare_parameter("command_timeout", 2.0).value
        self._state = CommandState(v_forward, w_turn, timeout)
        self._pub_cmd = self.create_publisher(Twist, "cmd_vel", 10)
        self.create_subscription(String, "wojtek/nav_command", self._on_command, 10)
        self.create_subscription(Empty, "/wojtek/nav/cancel", self._on_cancel, 10)
        self.create_timer(1.0 / DRIVE_TICK_HZ, self._tick)
        self.get_logger().info(
            "text commander up -- "
            f"{'/'.join(self._state.known_commands)} on wojtek/nav_command, "
            f"v={v_forward} m/s, w={w_turn} rad/s, dead-man {timeout} s"
        )

    def _now(self):
        return self.get_clock().now().nanoseconds / 1e9

    def _on_command(self, msg):
        if not self._state.handle(msg.data, self._now()):
            self.get_logger().warning(
                f"unknown nav command {msg.data[:64]!r} -- stopping "
                f"(known: {', '.join(self._state.known_commands)})")

    def _on_cancel(self, _msg):
        self._state.cancel()

    def _tick(self):
        twist = self._state.twist_to_publish(self._now())
        if twist is None:
            return
        t = Twist()
        t.linear.x, t.angular.z = twist  # CommandState only emits floats
        # linear.z stays 0.0: policy_node reads that as "default height".
        self._pub_cmd.publish(t)


def main():
    rclpy.init()
    node = TextCommander()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
