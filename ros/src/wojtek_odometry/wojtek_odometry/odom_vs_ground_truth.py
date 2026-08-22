"""Drift meter: leg odometry vs the simulator's ground truth.

Run beside sim.launch.py (hw:=mujoco) and the leg_odometry_node. The
MuJoCo plugin broadcasts the true odom->base_link on TF at 100 Hz; this
node samples both poses once a second and prints the position error, the
yaw error, and the drift as a percentage of the true distance travelled.
Ctrl+C prints the final summary.

Both poses are reduced to displacement-since-start in their own initial
heading frame, so the meter can be (re)started at any time, wherever the
robot happens to be -- it compares how far each says the robot moved,
not where each puts it.
"""

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener


def _yaw(qx, qy, qz, qw):
    return float(np.arctan2(
        2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))


class OdomVsGroundTruth(Node):
    def __init__(self):
        super().__init__("odom_vs_ground_truth")
        self._tf = Buffer()
        self._listener = TransformListener(self._tf, self)
        self._odom = None
        self.create_subscription(Odometry, "wojtek/odom", self._on_odom, 10)
        self._gt_prev = None
        self._distance = 0.0
        self.create_timer(1.0, self._tick)
        self.get_logger().info("comparing /wojtek/odom against TF odom->base_link")

    def _on_odom(self, msg):
        self._odom = msg

    @staticmethod
    def _relative(xy, yaw, origin):
        xy0, yaw0 = origin
        c, s = np.cos(-yaw0), np.sin(-yaw0)
        d = xy - xy0
        return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]]), yaw - yaw0

    def _tick(self):
        try:
            tf = self._tf.lookup_transform("odom", "base_link", rclpy.time.Time())
        except Exception:  # noqa: BLE001 -- not up yet; keep waiting
            return
        if self._odom is None:
            return
        t = tf.transform.translation
        r = tf.transform.rotation
        p = self._odom.pose.pose.position
        o = self._odom.pose.pose.orientation
        gt_xy = np.array([t.x, t.y])
        gt_yaw = _yaw(r.x, r.y, r.z, r.w)
        od_xy = np.array([p.x, p.y])
        od_yaw = _yaw(o.x, o.y, o.z, o.w)

        if self._gt_prev is None:  # first sample defines both origins
            self._gt_origin = (gt_xy, gt_yaw)
            self._od_origin = (od_xy, od_yaw)
            self._gt_prev = gt_xy
            return
        self._distance += float(np.linalg.norm(gt_xy - self._gt_prev))
        self._gt_prev = gt_xy

        gt_d, gt_dyaw = self._relative(gt_xy, gt_yaw, self._gt_origin)
        od_d, od_dyaw = self._relative(od_xy, od_yaw, self._od_origin)
        err = float(np.linalg.norm(gt_d - od_d))
        dyaw = np.degrees((od_dyaw - gt_dyaw + np.pi) % (2 * np.pi) - np.pi)
        drift = 100.0 * err / self._distance if self._distance > 0.05 else float("nan")
        self.get_logger().info(
            f"gt ({gt_d[0]:+.3f} {gt_d[1]:+.3f})  odom ({od_d[0]:+.3f} {od_d[1]:+.3f})  "
            f"err {err:.3f} m  yaw_err {dyaw:+.1f} deg  "
            f"dist {self._distance:.2f} m  drift {drift:.1f}%")


def main(args=None):
    rclpy.init(args=args)
    node = OdomVsGroundTruth()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
