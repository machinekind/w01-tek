"""The walker's ROS shell: the robot's legs, played by a person.

    python3 -m wojtek_vlm_gui.walker_node        # on the PC, the bench's ROS domain

Subscribes /cmd_vel (goto's setpoint driving and the brain's turns),
wojtek/nav/goal (goto's setpoint) and wojtek/nav/cancel; broadcasts
odom -> base_link at TICK_HZ from the integrated command (walker.Walker)
and publishes the walking guide as JSON on limits.GUIDE_TOPIC for the
page. Nothing else: it moves nothing, it only says what the robot would do.

Two roles (parameter `role`):
  * walker   -- the bench: a person carries the camera and walks what the
                page's Walk panel says.
  * odometry -- the real robot running without leg odometry (use_imu:=false
                leg_odom:=false, the fallback while the IMU lead is loose):
                the same dead reckoning of the command is then the robot's
                only pose, so goto and the pixel resolver have an odom frame.
                It drifts with every step the gait does not deliver as
                commanded; the brain looks again after every move. The
                robot's drop-in runs WITH leg odometry since fork PR #14, so
                this role is only for a session started with that fallback.

Guarded in both roles (walker.TfGuard): the node listens on /tf for
GUARD_LISTEN_S before it broadcasts anything, and stops broadcasting for
good the moment an odom -> base_link it did not send shows up (the robot's
leg odometry, a simulator). The guide then carries `tf_conflict: true` and
the log says why. Two sources of one transform would make goto and the
pixel resolver steer by a pose that jumps between them.
"""

from __future__ import annotations

import json
import math
import time

import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from rclpy.node import Node
from std_msgs.msg import Empty, String
from tf2_msgs.msg import TFMessage
from tf2_ros import TransformBroadcaster

from wojtek_vlm_gui import limits
from wojtek_vlm_gui.walker import TfGuard, Walker

TICK_HZ = 20.0
GUIDE_HZ = 10.0


class WalkerNode(Node):
    def __init__(self) -> None:
        super().__init__("wojtek_walker")
        self.declare_parameter("yaw_gain", 0.76)
        self.declare_parameter("speed_gain", 1.0)
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("role", "walker")
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.walker = Walker(yaw_gain=g("yaw_gain"), speed_gain=g("speed_gain"))
        self.odom, self.base = g("odom_frame"), g("base_frame")
        self.role = str(g("role"))
        if self.role not in ("walker", "odometry"):
            raise ValueError(f"role must be walker or odometry, not {self.role!r}")
        self.guard = TfGuard()
        self.guard.start(time.monotonic())
        self._conflict_logged = False
        # Before the broadcaster: whatever arrives while listening is not ours.
        self.create_subscription(TFMessage, "/tf", self._on_tf, 50)
        self.tf = TransformBroadcaster(self)
        self.pub_guide = self.create_publisher(String, limits.GUIDE_TOPIC, 1)
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd, 10)
        self.create_subscription(PoseStamped, limits.GOAL_TOPIC, self._on_goal, 10)
        self.create_subscription(Empty, limits.CANCEL_TOPIC, lambda _m: self.walker.clear_goal(), 10)
        self._last = time.monotonic()
        self._next_guide = 0.0
        self.create_timer(1.0 / TICK_HZ, self._tick)
        self.get_logger().info(f"walker up ({self.role}): /cmd_vel -> {self.odom}->{self.base} and {limits.GUIDE_TOPIC}")

    def _on_cmd(self, msg: Twist) -> None:
        self.walker.command(msg.linear.x, msg.linear.y, msg.angular.z, time.monotonic())

    def _on_goal(self, msg: PoseStamped) -> None:
        frame = msg.header.frame_id or self.odom
        if frame == self.odom:
            self.walker.set_goal_odom(msg.pose.position.x, msg.pose.position.y)
        elif frame == self.base:
            self.walker.set_goal_base(msg.pose.position.x, msg.pose.position.y)
        # Any other frame: goto transforms it itself; the guide just has no vector.

    def _on_tf(self, msg: TFMessage) -> None:
        for t in msg.transforms:
            if t.header.frame_id != self.odom or t.child_frame_id != self.base:
                continue
            stamp_ns = t.header.stamp.sec * 1_000_000_000 + t.header.stamp.nanosec
            if self.guard.seen(stamp_ns) and not self._conflict_logged:
                self._conflict_logged = True
                self.get_logger().error(
                    f"another node publishes {self.odom} -> {self.base} (leg odometry? a sim?): "
                    "the walker does not broadcast it, the guide goes on without a pose")

    def _tick(self) -> None:
        now = time.monotonic()
        self.walker.step(now, now - self._last)
        self._last = now
        w = self.walker
        if self.guard.may_broadcast(now):
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id, t.child_frame_id = self.odom, self.base
            t.transform.translation.x, t.transform.translation.y = w.x, w.y
            t.transform.rotation.z, t.transform.rotation.w = _yaw_quat(w.yaw)
            self.guard.sent(t.header.stamp.sec * 1_000_000_000 + t.header.stamp.nanosec)
            self.tf.sendTransform(t)
        if now >= self._next_guide:
            self.pub_guide.publish(String(data=json.dumps(
                {**w.guide(now), "role": self.role, "tf_conflict": self.guard.foreign})))
            self._next_guide = now + 1.0 / GUIDE_HZ


def _yaw_quat(yaw: float):
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def main() -> None:
    rclpy.init()
    node = WalkerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
