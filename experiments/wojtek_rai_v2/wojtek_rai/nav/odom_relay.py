"""nav_msgs/Odometry from what the simulation already publishes.

The MuJoCo plant broadcasts ground-truth TF `odom -> base_link` and a Twist
on `/odom_vel` (linear velocity in the odom frame, angular in the body
frame), but Nav2's controllers and slam_toolbox want `/odom`. This node
samples the TF at `rate_hz`, rotates the linear velocity into the body
frame, and publishes `nav_msgs/Odometry` with fixed small covariances.

    python3 -m wojtek_rai.nav.odom_relay
"""

from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener


def _yaw(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class OdomRelay(Node):
    def __init__(self) -> None:
        super().__init__("odom_relay")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("vel_topic", "/odom_vel")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("rate_hz", 30.0)
        self._odom_frame = self.get_parameter("odom_frame").value
        self._base_frame = self.get_parameter("base_frame").value
        self._buf = Buffer()
        self._listener = TransformListener(self._buf, self)
        self._vel = Twist()
        self.create_subscription(Twist, self.get_parameter("vel_topic").value, self._on_vel, 10)
        self._pub = self.create_publisher(Odometry, self.get_parameter("odom_topic").value, 10)
        self.create_timer(1.0 / float(self.get_parameter("rate_hz").value), self._tick)
        self._warned = False

    def _on_vel(self, msg: Twist) -> None:
        self._vel = msg

    def _tick(self) -> None:
        try:
            tf = self._buf.lookup_transform(self._odom_frame, self._base_frame, rclpy.time.Time())
        except Exception as exc:  # noqa: BLE001 -- tf2 raises several unrelated types
            if not self._warned:
                self.get_logger().warn(f"waiting for TF {self._odom_frame}->{self._base_frame}: {exc}")
                self._warned = True
            return
        t, q = tf.transform.translation, tf.transform.rotation
        yaw = _yaw(q)
        vx_w, vy_w = self._vel.linear.x, self._vel.linear.y
        c, s = math.cos(yaw), math.sin(yaw)
        odom = Odometry()
        odom.header.stamp = tf.header.stamp
        odom.header.frame_id = self._odom_frame
        odom.child_frame_id = self._base_frame
        odom.pose.pose.position.x = t.x
        odom.pose.pose.position.y = t.y
        odom.pose.pose.position.z = t.z
        odom.pose.pose.orientation = q
        odom.twist.twist.linear.x = c * vx_w + s * vy_w
        odom.twist.twist.linear.y = -s * vx_w + c * vy_w
        odom.twist.twist.angular.z = self._vel.angular.z
        for i in range(6):
            odom.pose.covariance[i * 7] = 1e-4
            odom.twist.covariance[i * 7] = 1e-3
        self._pub.publish(odom)


def main() -> None:
    rclpy.init()
    node = OdomRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
