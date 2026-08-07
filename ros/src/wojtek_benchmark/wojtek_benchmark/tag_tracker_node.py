"""AprilTag ground-truth tracker: images in, world-frame robot pose out.

The real rig's node, sim-agnostic by construction: it consumes an image
topic, a CameraInfo, and the tag inventory (apriltags.yaml) -- nothing in
it can tell the sim render from a webcam.  Point image_topic/info_topic at
a real camera driver and it is the deployment tracker.

Two phases:

1. CALIBRATION.  Averages the three floor-tag centers over calib_frames
   detections, then solves the camera->world transform.  It REFUSES unless
   the vision-measured leg lengths match the tape-measured expected_leg_x/y
   parameters (1 % default) and the L is square -- an unmeasured or
   misplaced course must produce no numbers at all.  On success the world
   frame is latched and published as a static TF.
2. TRACKING.  Every frame with the robot tag visible publishes its pose in
   the bench world frame (PoseStamped + TF).  The pose is the tag SURFACE
   frame (x forward on the robot, z out of the tag face), i.e. the
   detected tag frame times tracker.TAG_TO_SURFACE.

  subscribes  image_topic  (sensor_msgs/Image: mono8, rgb8 or bgr8)
              info_topic   (sensor_msgs/CameraInfo)
  publishes   /benchmark/robot_tag_pose  (PoseStamped, world frame)
              TF world_frame -> robot_tag_frame, static world_frame -> camera
  services    /benchmark/recalibrate    (std_srvs/Trigger)
"""

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import Trigger
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from wojtek_benchmark import tracker
from wojtek_benchmark.sim_rig import DEFAULT_TAGS_CONFIG, load_tags_config

POSE_TOPIC = "/benchmark/robot_tag_pose"


def _quat_wxyz(R):
    """Rotation matrix -> (w, x, y, z), Shepperd's method."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return np.array([s / 4, (R[2, 1] - R[1, 2]) / s,
                         (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0) * 2
    q = np.empty(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = s / 4
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


def _fill_transform(msg, T):
    msg.transform.translation.x, msg.transform.translation.y, \
        msg.transform.translation.z = map(float, T[:3, 3])
    w, x, y, z = _quat_wxyz(T[:3, :3])
    msg.transform.rotation.w = float(w)
    msg.transform.rotation.x = float(x)
    msg.transform.rotation.y = float(y)
    msg.transform.rotation.z = float(z)


class TagTrackerNode(Node):
    def __init__(self):
        super().__init__("wojtek_tag_tracker")
        self.declare_parameter("image_topic", "/benchmark/camera/image_raw")
        self.declare_parameter("info_topic", "/benchmark/camera/camera_info")
        self.declare_parameter("tags_config", str(DEFAULT_TAGS_CONFIG))
        # Tape-measured center-to-center legs of the L. NaN = unmeasured,
        # and calibration will refuse: record the course before tracking.
        self.declare_parameter("expected_leg_x_m", float("nan"))
        self.declare_parameter("expected_leg_y_m", float("nan"))
        self.declare_parameter("leg_tol_frac", tracker.DEFAULT_LEG_TOL_FRAC)
        self.declare_parameter("angle_tol_deg", tracker.DEFAULT_ANGLE_TOL_DEG)
        self.declare_parameter("calib_frames", 30)
        self.declare_parameter("world_frame", "bench_world")
        self.declare_parameter("robot_tag_frame", "benchmark_robot_tag")

        tags = load_tags_config(self.get_parameter("tags_config").value)
        self._sizes_by_id = {t["id"]: t["size_m"] for t in tags.values()}
        self._roles_by_id = {t["id"]: role for role, t in tags.items()}
        self._robot_id = tags["robot"]["id"]
        self._detector = tracker.make_detector()

        self._intrinsics = None
        self._T_cam_world = None
        self._calib_frames = []
        self._world = self.get_parameter("world_frame").value
        self._tag_frame = self.get_parameter("robot_tag_frame").value

        self.create_subscription(
            CameraInfo, self.get_parameter("info_topic").value,
            self._on_info, qos_profile_sensor_data,
        )
        self.create_subscription(
            Image, self.get_parameter("image_topic").value,
            self._on_image, qos_profile_sensor_data,
        )
        self._pub_pose = self.create_publisher(PoseStamped, POSE_TOPIC, 10)
        self._tf = TransformBroadcaster(self)
        self._tf_static = StaticTransformBroadcaster(self)
        self.create_service(Trigger, "/benchmark/recalibrate", self._on_recalibrate)
        self.get_logger().info(
            f"tracking tags {sorted(self._sizes_by_id)} on "
            f"{self.get_parameter('image_topic').value}; waiting for "
            f"CameraInfo, then calibrating from "
            f"{self.get_parameter('calib_frames').value} frames"
        )

    # -- inputs ---------------------------------------------------------------
    def _on_info(self, msg):
        self._intrinsics = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])
        if any(abs(d) > 1e-12 for d in msg.d):
            self.get_logger().warning(
                "camera reports distortion but the tracker assumes rectified "
                "images -- feed image_rect or expect biased poses",
                once=True,
            )

    def _gray(self, msg):
        if msg.encoding == "mono8":
            return np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width)
        if msg.encoding in ("rgb8", "bgr8"):
            rgb = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
            w = [0.299, 0.587, 0.114] if msg.encoding == "rgb8" else [0.114, 0.587, 0.299]
            return (rgb @ np.array(w)).astype(np.uint8)
        self.get_logger().error(f"unsupported encoding {msg.encoding!r}", once=True)
        return None

    def _on_image(self, msg):
        if self._intrinsics is None:
            return
        gray = self._gray(msg)
        if gray is None:
            return
        dets = tracker.detect(
            self._detector, gray, self._intrinsics, self._sizes_by_id
        )
        if self._T_cam_world is None:
            self._calibrate(dets, msg.header)
        elif self._robot_id in dets:
            self._track(dets[self._robot_id], msg.header)

    # -- phases ---------------------------------------------------------------
    def _calibrate(self, dets, header):
        floor = {self._roles_by_id[i]: d.t for i, d in dets.items()
                 if i != self._robot_id}
        if floor:
            self._calib_frames.append(floor)
        need = int(self.get_parameter("calib_frames").value)
        if len(self._calib_frames) < need:
            return
        legs = (self.get_parameter("expected_leg_x_m").value,
                self.get_parameter("expected_leg_y_m").value)
        legs = tuple(None if np.isnan(v) else v for v in legs)
        try:
            self._T_cam_world = tracker.solve_world_from_floor(
                tracker.averaged_centers(self._calib_frames), legs,
                leg_tol_frac=self.get_parameter("leg_tol_frac").value,
                angle_tol_deg=self.get_parameter("angle_tol_deg").value,
            )
        except tracker.CalibrationError as e:
            # Refuse loudly, drop the window, try again with fresh frames:
            # a misplaced tag needs a human, not a retry, and the log says so.
            self.get_logger().error(f"calibration REFUSED: {e}")
            self._calib_frames = []
            return
        self._calib_frames = []
        tf = TransformStamped()
        tf.header.stamp = header.stamp
        tf.header.frame_id = self._world
        tf.child_frame_id = header.frame_id or "benchmark_camera_optical_frame"
        _fill_transform(tf, tracker.inv_se3(self._T_cam_world))
        self._tf_static.sendTransform(tf)
        self.get_logger().info(
            f"calibrated: {self._world} frame latched from floor tags"
        )

    def _track(self, det, header):
        T_world_tag = tracker.inv_se3(self._T_cam_world) @ tracker.se3(det.R, det.t)
        T_world_tag[:3, :3] = T_world_tag[:3, :3] @ tracker.TAG_TO_SURFACE

        pose = PoseStamped()
        pose.header.stamp = header.stamp
        pose.header.frame_id = self._world
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = \
            map(float, T_world_tag[:3, 3])
        w, x, y, z = _quat_wxyz(T_world_tag[:3, :3])
        pose.pose.orientation.w = float(w)
        pose.pose.orientation.x = float(x)
        pose.pose.orientation.y = float(y)
        pose.pose.orientation.z = float(z)
        self._pub_pose.publish(pose)

        tf = TransformStamped()
        tf.header.stamp = header.stamp
        tf.header.frame_id = self._world
        tf.child_frame_id = self._tag_frame
        _fill_transform(tf, T_world_tag)
        self._tf.sendTransform(tf)

    def _on_recalibrate(self, _req, resp):
        self._T_cam_world = None
        self._calib_frames = []
        resp.success = True
        resp.message = "calibration dropped; re-solving from fresh frames"
        self.get_logger().info(resp.message)
        return resp


def main():
    rclpy.init()
    node = TagTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
