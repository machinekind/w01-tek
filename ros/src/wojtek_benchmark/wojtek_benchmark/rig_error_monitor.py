"""Sim-only judge of the benchmark rig: tracked pose vs exact ground truth.

The tracker's output travelled camera pixels -> detection -> calibration;
/sim/qpos is the exact state the render was drawn from.  sim_rig.yaml
pins where the bench world frame sits in the sim world and where the tag
geom sits on the base, so both poses land in the same frame and their
difference is the END-TO-END error of the whole rig -- the number that
says whether real-world benchmark scores can be trusted.

  subscribes  /benchmark/robot_tag_pose  (PoseStamped from the tracker)
              /sim/qpos                  (Float64MultiArray, ground truth)
  publishes   /benchmark/pose_error_mm   (Float32)
              /benchmark/yaw_error_deg   (Float32)
              static TF bench_world -> odom

The static TF is the display bridge: the sim's odom frame IS the sim world
(the hardware plugin broadcasts ground-truth odom -> base_link), and
sim_rig.yaml pins where bench_world sits in it.  Publishing that edge
joins the rig's TF tree to the robot's, so one 3D scene (display frame
bench_world) shows the ground-truth robot model, the tag-measured pose,
and the trajectory -- the visible gap between model and measurement is the
rig error.  Sim-only on purpose: reality has no ground-truth odom, which
is the whole reason the rig exists.

Timing: images are stamped with their qpos arrival time (sim_camera_node's
documented deviation), so the monitor buffers recent qpos and matches by
nearest stamp.  Residual mismatch shows up as extra error while the robot
moves and vanishes at rest -- watch the standing value for the rig's floor.
"""

from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float32, Float64MultiArray
from tf2_ros import StaticTransformBroadcaster

from wojtek_benchmark import tracker
from wojtek_benchmark.sim_rig import (
    DEFAULT_RIG_CONFIG,
    _rpy_quat,
    load_rig_config,
    world_frame_in_sim,
)


def _quat_to_mat(w, x, y, z):
    n = w * w + x * x + y * y + z * z
    s = 0.0 if n < 1e-15 else 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ])


class RigErrorMonitor(Node):
    def __init__(self):
        super().__init__("wojtek_rig_error_monitor")
        self.declare_parameter("rig_config", str(DEFAULT_RIG_CONFIG))
        self.declare_parameter("qpos_buffer_s", 1.0)
        self.declare_parameter("log_period_s", 2.0)
        self.declare_parameter("world_frame", "bench_world")
        self.declare_parameter("odom_frame", "odom")

        rig_cfg = load_rig_config(self.get_parameter("rig_config").value)
        self._T_bench_sim = tracker.inv_se3(world_frame_in_sim(rig_cfg))
        mount = rig_cfg["robot_tag"]
        w, x, y, z = _rpy_quat(mount.get("rpy_deg", [0.0, 0.0, 0.0]))
        self._T_base_tag = tracker.se3(_quat_to_mat(w, x, y, z), mount["pos"])

        self._qpos_buf = deque()
        self.create_subscription(
            Float64MultiArray, "sim/qpos", self._on_qpos, qos_profile_sensor_data
        )
        self.create_subscription(
            PoseStamped, "/benchmark/robot_tag_pose", self._on_pose, 10
        )
        self._pub_pos = self.create_publisher(Float32, "/benchmark/pose_error_mm", 10)
        self._pub_yaw = self.create_publisher(Float32, "/benchmark/yaw_error_deg", 10)
        self._last_log = None

        # Display bridge (see module docstring): pose of odom (= the sim
        # world) expressed in bench_world, exact from sim_rig.yaml.
        T_bench_sim = self._T_bench_sim
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = self.get_parameter("world_frame").value
        tf.child_frame_id = self.get_parameter("odom_frame").value
        tf.transform.translation.x, tf.transform.translation.y, \
            tf.transform.translation.z = map(float, T_bench_sim[:3, 3])
        R = T_bench_sim[:3, :3]
        # bench_world differs from the sim world by a planar pose only.
        yaw = np.arctan2(R[1, 0], R[0, 0])
        tf.transform.rotation.w = float(np.cos(yaw / 2.0))
        tf.transform.rotation.z = float(np.sin(yaw / 2.0))
        # Kept on self: a static broadcaster's latched publisher must outlive
        # this constructor or late subscribers never receive the transform.
        self._tf_static = StaticTransformBroadcaster(self)
        self._tf_static.sendTransform(tf)
        self.get_logger().info("comparing tracked pose against /sim/qpos")

    def _on_qpos(self, msg):
        now = self.get_clock().now().nanoseconds
        self._qpos_buf.append((now, np.asarray(msg.data[:7], dtype=float)))
        horizon = int(self.get_parameter("qpos_buffer_s").value * 1e9)
        while self._qpos_buf and now - self._qpos_buf[0][0] > horizon:
            self._qpos_buf.popleft()

    def _on_pose(self, msg):
        if not self._qpos_buf:
            return
        t = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        _, qpos = min(self._qpos_buf, key=lambda item: abs(item[0] - t))

        T_sim_base = tracker.se3(_quat_to_mat(*qpos[3:7]), qpos[0:3])
        T_bench_tag_gt = self._T_bench_sim @ T_sim_base @ self._T_base_tag

        p = msg.pose.position
        o = msg.pose.orientation
        T_bench_tag = tracker.se3(
            _quat_to_mat(o.w, o.x, o.y, o.z), [p.x, p.y, p.z]
        )

        pos_err_mm = 1000.0 * float(
            np.linalg.norm(T_bench_tag[:3, 3] - T_bench_tag_gt[:3, 3])
        )
        R_err = T_bench_tag_gt[:3, :3].T @ T_bench_tag[:3, :3]
        yaw_err = tracker.yaw_deg(tracker.se3(R_err, [0, 0, 0]))

        self._pub_pos.publish(Float32(data=pos_err_mm))
        self._pub_yaw.publish(Float32(data=yaw_err))

        now = self.get_clock().now().nanoseconds
        period = int(self.get_parameter("log_period_s").value * 1e9)
        if self._last_log is None or now - self._last_log > period:
            self._last_log = now
            self.get_logger().info(
                f"rig error: {pos_err_mm:.1f} mm, yaw {yaw_err:+.2f} deg"
            )


def main():
    rclpy.init()
    node = RigErrorMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
