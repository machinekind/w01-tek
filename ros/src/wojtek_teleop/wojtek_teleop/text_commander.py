#!/usr/bin/env python3
"""Text commands -> /cmd_vel: the ROS-side contract for the future VLM (#92).

    ros2 run wojtek_teleop text_commander
    ros2 topic pub -1 /wojtek/nav_command std_msgs/String "data: forward"

Subscribes `wojtek/nav_command` (std_msgs/String: forward / left / right /
stop) and publishes the SAME /cmd_vel surface the gamepad and the consoles
use, so nothing in wojtek_bringup / wojtek_policy changes. A VLM watching
/camera/camera/color/image_raw later needs to publish only nav_command --
this node freezes that contract; everything below /cmd_vel stays as is.

Speeds are parameters (`v_forward`, `w_turn`), and a dead-man stops the
robot `command_timeout` seconds after the last command: the VLM must keep
talking to keep Wojtek walking. On stop/timeout the node publishes exactly
ONE zero Twist and then goes silent -- the zero is mandatory because
policy_node latches the last received command forever, and the silence lets
another drive source (web console, gamepad) take /cmd_vel without being
fought. linear.z stays 0.0 = "use the default height" for policy_node;
height is the operator consoles' business, not the VLM's.
"""
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String

DRIVE_TICK_HZ = 20.0     # /cmd_vel publish rate while active (same as consoles)


class CommandState:
    """Pure command core -- all behaviour, no rclpy (tested without ROS).

    `handle` arms a command, `twist_to_publish` is the timer's question:
    (vx, wz) while active, exactly one (0, 0) on stop/timeout, then None.
    """

    def __init__(self, v_forward, w_turn, command_timeout):
        self._timeout = float(command_timeout)
        # The whole vocabulary: motion commands map to (vx, wz); "stop"
        # (and, defensively, anything unknown) clears instead of mapping.
        self._map = {
            "forward": (float(v_forward), 0.0),
            "left": (0.0, float(w_turn)),
            "right": (0.0, -float(w_turn)),
        }
        self._twist = None        # (vx, wz) while a command is active
        self._deadline = 0.0
        self._stop_sent = True    # nothing to zero out yet

    @property
    def known_commands(self):
        return (*self._map, "stop")

    def handle(self, command, now):
        """Arm `command`; False for unknown ones (caller logs, we stop)."""
        command = command.strip().lower()
        twist = self._map.get(command)
        if twist is not None:
            self._twist = twist
            self._deadline = now + self._timeout
            self._stop_sent = False
            return True
        # "stop" and unknown commands both stop -- but only if a command
        # of ours is active; a zero burst from idle would shout over
        # whichever other source currently drives /cmd_vel.
        self._twist = None
        return command == "stop"

    def twist_to_publish(self, now):
        if self._twist is not None and now <= self._deadline:
            return self._twist
        self._twist = None        # timed out (or stopped by handle)
        if self._stop_sent:
            return None
        self._stop_sent = True
        return (0.0, 0.0)


class TextCommander(Node):
    def __init__(self):
        super().__init__("text_commander")
        v_forward = self.declare_parameter("v_forward", 0.3).value
        w_turn = self.declare_parameter("w_turn", 0.5).value
        timeout = self.declare_parameter("command_timeout", 2.0).value
        self._state = CommandState(v_forward, w_turn, timeout)
        self._pub_cmd = self.create_publisher(Twist, "cmd_vel", 10)
        self.create_subscription(String, "wojtek/nav_command", self._on_command, 10)
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
