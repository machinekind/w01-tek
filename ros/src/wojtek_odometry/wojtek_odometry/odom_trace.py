"""Two RViz trails: where the robot really went vs where the odometry
thinks it went.

Publishes nav_msgs/Path on /wojtek/odom_trace/ground_truth (from the
simulator's TF odom->base_link) and /wojtek/odom_trace/estimate (from
/wojtek/odom). The estimate is re-anchored into the ground-truth frame
at the first sample -- position and yaw aligned -- so both trails start
at the same point and the gap between them IS the accumulated drift.
Restart the node to re-anchor (e.g. after teleporting or restarting the
odometry).

Sim-only, like odom_vs_ground_truth: it needs a ground-truth transform
to compare against.
"""

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener


def _yaw(qx, qy, qz, qw):
    return float(np.arctan2(
        2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))


class OdomTrace(Node):
    def __init__(self):
        super().__init__("odom_trace")
        self.declare_parameter("rate_hz", 5.0)
        self._tf = Buffer()
        self._listener = TransformListener(self._tf, self)
        self._odom = None
        self.create_subscription(Odometry, "wojtek/odom", self._on_odom, 10)
        self._pub_gt = self.create_publisher(Path, "wojtek/odom_trace/ground_truth", 1)
        self._pub_est = self.create_publisher(Path, "wojtek/odom_trace/estimate", 1)
        self._gt_path = Path()
        self._est_path = Path()
        self._gt_path.header.frame_id = "odom"
        self._est_path.header.frame_id = "odom"
        self._anchor = None  # (gt_xy, gt_yaw, od_xy, od_yaw)
        self.create_timer(1.0 / self.get_parameter("rate_hz").value, self._tick)
        self.get_logger().info(
            "tracing TF odom->base_link vs /wojtek/odom; add both "
            "/wojtek/odom_trace/* Path displays in RViz")

    def _on_odom(self, msg):
        self._odom = msg

    def _tick(self):
        try:
            tf = self._tf.lookup_transform("odom", "base_link", rclpy.time.Time())
        except Exception:  # noqa: BLE001 -- TF not up yet
            return
        if self._odom is None:
            return
        t, r = tf.transform.translation, tf.transform.rotation
        p, o = self._odom.pose.pose.position, self._odom.pose.pose.orientation
        gt_xy = np.array([t.x, t.y])
        od_xy = np.array([p.x, p.y])
        if self._anchor is None:
            self._anchor = (gt_xy, _yaw(r.x, r.y, r.z, r.w),
                            od_xy, _yaw(o.x, o.y, o.z, o.w))
        gt0, gt_yaw0, od0, od_yaw0 = self._anchor

        # Estimate displacement, rotated from its own start heading into the
        # ground truth's, re-rooted at the ground-truth start point.
        a = gt_yaw0 - od_yaw0
        c, s = np.cos(a), np.sin(a)
        d = od_xy - od0
        est = gt0 + np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])

        stamp = self.get_clock().now().to_msg()
        for path, xy, z in ((self._gt_path, gt_xy, t.z), (self._est_path, est, t.z)):
            pose = PoseStamped()
            pose.header.frame_id = "odom"
            pose.header.stamp = stamp
            pose.pose.position.x = float(xy[0])
            pose.pose.position.y = float(xy[1])
            pose.pose.position.z = float(z)
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
            path.header.stamp = stamp
        self._pub_gt.publish(self._gt_path)
        self._pub_est.publish(self._est_path)


def main(args=None):
    rclpy.init(args=args)
    node = OdomTrace()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
