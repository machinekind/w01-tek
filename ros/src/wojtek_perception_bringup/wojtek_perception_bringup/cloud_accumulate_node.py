"""Primitive persistent map: depth points accumulated in the odom frame.

Each depth frame is deprojected (subsampled), transformed into odom with
TF, and merged into one ever-growing plain point cloud, republished at a
slow rate as an unordered PointCloud2 in odom -- drive around and the
world stays behind you. The points are the raw measurements; the only
processing is THINNING: a 3D grid keeps the first point that lands in
each min_point_spacing cell, so re-seeing the same patch of floor a
thousand times does not grow the cloud a thousandfold. This is
deliberately the simplest thing that maps:

  * no ray clearing -- a point, once seen, is permanent (restart or call
    ~/clear to forget); dynamic obstacles will leave trails,
  * no occupancy semantics -- it is a point cloud, nothing more,
  * anchored in odom, so the map inherits the odometry's drift; that is
    the experiment, per the nav plan: measured drift decides whether this
    stage is usable or SLAM has to take over.

The pose each sub-cloud is pasted at comes from the ROBOT'S OWN odometry
(/wojtek/odom, the leg-kinematics + IMU estimate), NOT from TF: in the
simulator TF odom->base_link carries the ground truth, and a map anchored
to it would be dishonestly perfect. The real robot knows nothing beyond
its odometry, so the map must inherit the odometry's drift -- in sim you
can SEE that drift as the pasted walls shearing away from where the
(ground-truth-posed) robot model stands. Only the static base->camera
extrinsics chain is taken from TF, which is the URDF on both sides.

Deprojection is from the depth IMAGE (not the driver's cloud) for the same
reason: a full PointCloud2 would serialise ~100k points a frame before
we kept one in fifty.
"""

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener


def _quat_to_matrix(x, y, z, w):
    n = x * x + y * y + z * z + w * w
    s = 2.0 / n if n > 1e-12 else 0.0
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array([
        [1 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1 - (xx + yy)],
    ])


class CloudAccumulateNode(Node):
    def __init__(self):
        super().__init__("cloud_accumulate")
        self.declare_parameter("target_frame", "odom")
        # The odometry whose pose each sub-cloud is pasted at. The robot's
        # own estimate on purpose -- see the module docstring.
        self.declare_parameter("odom_topic", "wojtek/odom")
        self.declare_parameter("base_frame", "base_link")
        # Thinning cell edge [m]: at most one stored point per cell of this
        # size (the point itself is the raw measurement, not a cell centre).
        # 2 cm reads as a normal dense cloud; memory is what bounds it, so
        # the cap below is the guard.
        self.declare_parameter("min_point_spacing", 0.02)
        # Every Nth pixel in u and v. 2 keeps ~25k rays of the 102k --
        # comfortable on the PC; loosen on the Pi if the CPU budget says so.
        self.declare_parameter("pixel_stride", 2)
        # Every Nth depth frame. 3 -> ~5 Hz processing at the camera's 15.
        self.declare_parameter("frame_stride", 3)
        self.declare_parameter("min_range_m", 0.3)
        self.declare_parameter("max_range_m", 3.0)
        # Points above this height in the target frame are dropped: the
        # ceiling maps to nothing the planner uses.
        self.declare_parameter("max_height_m", 1.0)
        self.declare_parameter("publish_rate_hz", 1.0)
        # Hard cap on stored points; exists so a runaway TF cannot eat the
        # host's RAM (1M points ~= 12 MB of payload plus dict overhead).
        self.declare_parameter("max_points", 1000000)

        self._frame = self.get_parameter("target_frame").value
        self._voxel = self.get_parameter("min_point_spacing").value
        self._px_stride = int(self.get_parameter("pixel_stride").value)
        self._frame_stride = int(self.get_parameter("frame_stride").value)
        self._min_r = self.get_parameter("min_range_m").value
        self._max_r = self.get_parameter("max_range_m").value
        self._max_h = self.get_parameter("max_height_m").value
        self._max_points = int(self.get_parameter("max_points").value)

        self._base_frame = self.get_parameter("base_frame").value
        self._tf = Buffer()
        self._listener = TransformListener(self._tf, self)
        self._info = None
        self._odom = None  # latest (position ndarray, rotation matrix)
        self._frame_count = 0
        self._points_by_cell = {}  # (i, j, k) -> (x, y, z) float32 point
        self._full_warned = False

        self.create_subscription(
            Odometry, self.get_parameter("odom_topic").value, self._on_odom, 10)
        self.create_subscription(
            CameraInfo, "depth/camera_info", self._on_info,
            qos_profile_sensor_data)
        self.create_subscription(
            Image, "depth/image", self._on_depth, qos_profile_sensor_data)
        self._pub = self.create_publisher(PointCloud2, "cloud_map/points", 1)
        self.create_service(Trigger, "~/clear", self._on_clear)
        self.create_timer(
            1.0 / self.get_parameter("publish_rate_hz").value, self._publish)
        self.get_logger().info(
            f"accumulating a thinned raw cloud in '{self._frame}' "
            f"(>= {self._voxel} m spacing); ~/clear resets")

    def _on_clear(self, _req, res):
        n = len(self._points_by_cell)
        self._points_by_cell.clear()
        self._full_warned = False
        res.success = True
        res.message = f"cleared {n} points"
        return res

    def _on_info(self, msg):
        self._info = msg

    def _on_odom(self, msg):
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        self._odom = (np.array([p.x, p.y, p.z]),
                      _quat_to_matrix(o.x, o.y, o.z, o.w))

    def _on_depth(self, msg):
        self._frame_count += 1
        if self._info is None or self._frame_count % self._frame_stride:
            return
        if msg.encoding != "16UC1":
            self.get_logger().error(
                f"expected 16UC1 depth, got {msg.encoding}", once=True)
            return
        if len(self._points_by_cell) >= self._max_points:
            if not self._full_warned:
                self._full_warned = True
                self.get_logger().warn(
                    f"point cap ({self._max_points}) reached -- map frozen, "
                    "call ~/clear to start over")
            return
        if self._odom is None:
            self.get_logger().warn(
                "no odometry yet -- dropping depth frames", once=True)
            return
        # TF is used ONLY for the static base->camera extrinsics (the URDF
        # chain, identical on robot and sim); the odom->base pose comes from
        # the odometry topic so the map honestly inherits its drift.
        try:
            tf = self._tf.lookup_transform(
                self._base_frame, msg.header.frame_id, rclpy.time.Time())
        except Exception as e:  # noqa: BLE001 -- extrinsics not up yet
            self.get_logger().warn(f"no TF, dropping frame: {e}", once=True)
            return

        depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(
            msg.height, msg.width)[::self._px_stride, ::self._px_stride]
        k = self._info.k
        fx, fy, cx, cy = k[0], k[4], k[2], k[5]
        vs, us = np.mgrid[0:msg.height:self._px_stride,
                          0:msg.width:self._px_stride]
        z = depth.astype(np.float32) * 1e-3
        valid = (z > self._min_r) & (z < self._max_r)
        z = z[valid]
        if z.size == 0:
            return
        x = (us[valid] - cx) * z / fx
        y = (vs[valid] - cy) * z / fy
        pts_optical = np.stack([x, y, z], axis=1)

        t = tf.transform.translation
        q = tf.transform.rotation
        rot_bc = _quat_to_matrix(q.x, q.y, q.z, q.w)
        pts_base = pts_optical @ rot_bc.T + np.array([t.x, t.y, t.z])
        odom_p, odom_rot = self._odom
        pts = pts_base @ odom_rot.T + odom_p
        pts = pts[pts[:, 2] < self._max_h]

        idx = np.floor(pts / self._voxel).astype(np.int32)
        for key, p in zip(map(tuple, idx), pts.astype(np.float32)):
            self._points_by_cell.setdefault(key, p)

    def _publish(self):
        if not self._points_by_cell:
            return
        pts = np.stack(list(self._points_by_cell.values()))
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame
        msg.height = 1
        msg.width = pts.shape[0]
        msg.fields = [
            PointField(name=n, offset=4 * i,
                       datatype=PointField.FLOAT32, count=1)
            for i, n in enumerate("xyz")
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12 * pts.shape[0]
        msg.is_dense = True
        msg.data = pts.tobytes()
        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CloudAccumulateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
