"""ROS shell for the pixel-goal resolver: the VLM's pixel in, goto's setpoint out.

Inputs
  wojtek/nav/pixel_goal   geometry_msgs/PointStamped. header.stamp is the
                          stamp of the colour picture the VLM looked at
                          (the depth image of that moment is what the pixel
                          is read against). point.x, point.y: the pixel in
                          the colour image, normalised to [0, 1] (what a
                          VLM adapter emits: Qwen's 0-1000 / 1000); a value
                          above 1 is taken as an absolute colour pixel.
                          point.z: standoff in metres, 0 = the parameter.
  camera depth image + camera_info, colour camera_info
                          the depth stream is kept in a ring keyed by
                          stamp; the colour image itself is not needed
                          here, only its intrinsics.
  wojtek/nav/status       goto's latched status word.
  TF odom->camera frames  at the picture's stamp (leg_odometry + URDF).

Outputs
  wojtek/nav/goal         geometry_msgs/PoseStamped in odom, the standoff
                          setpoint, re-sent every repeat_s while the goal
                          is live (goto's dead-man is 3 s).
  wojtek/nav/pixel_target geometry_msgs/PointStamped in odom: the object
                          point itself, for the map view and the eval.
  wojtek/nav/pixel_status std_msgs/String, latched: resolving / no_frame /
                          no_depth / no_tf / sent / reached / blocked /
                          timeout / replaced -- what the VLM loop reads.

A new pixel goal replaces the one in flight. The resolver never talks to
/cmd_vel: goto keeps the veto and the reflexes.
"""

import bisect
import math

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs  # noqa: F401 -- registers PointStamped with tf2

from wojtek_nav.pixel_goal import (
    GoalTracker,
    colour_to_depth_pixel,
    deproject,
    depth_at,
    nearest_frame,
    standoff,
)

TICK_HZ = 10.0
# The leg odometry publishes at ~25 Hz: a picture stamped after its latest
# sample cannot be transformed yet. Retry on the tick for this long.
RESOLVE_RETRY_S = 0.5


def _stamp_s(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def _yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class PixelGoalNode(Node):
    def __init__(self):
        super().__init__("wojtek_pixel_goal")
        p = self.declare_parameter
        p("depth_topic", "/camera/camera/depth/image_rect_raw")
        p("depth_info_topic", "/camera/camera/depth/camera_info")
        p("colour_info_topic", "/camera/camera/color/camera_info")
        p("odom_frame", "odom")
        p("base_frame", "base_link")
        p("standoff_m", 0.7)
        p("patch_radius", 4)
        p("z_min", 0.2)
        p("z_max", 3.0)
        # Picture-to-depth pairing tolerance. The sim's two streams run on
        # separate timers (~67 ms apart at worst); the D435 pairs them.
        p("max_skew_s", 0.1)
        # How far back a picture may be: the model's answer comes seconds after
        # the picture, a cold model (first call) took 10 s in the sim. 15 s of
        # 424x240 depth at 15 Hz is ~45 MB.
        p("ring_s", 15.0)
        p("repeat_s", 1.0)
        p("blocked_hold_s", 5.0)
        p("max_goal_s", 60.0)
        g = lambda name: self.get_parameter(name).value  # noqa: E731
        self._odom = g("odom_frame")
        self._base = g("base_frame")

        self._tf = Buffer(cache_time=Duration(seconds=30.0))
        self._listener = TransformListener(self._tf, self)
        self._ring = []          # [(stamp_s, Image)], oldest first
        self._ring_s = float(g("ring_s"))
        self._depth_info = None
        self._colour_info = None
        self._goto_status = None
        self._pending = None     # (PointStamped, first_try_s) awaiting TF
        self._goal = None        # PoseStamped in odom, the live setpoint
        self._tracker = GoalTracker(g("repeat_s"), g("blocked_hold_s"), g("max_goal_s"))

        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Image, g("depth_topic"), self._on_depth, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, g("depth_info_topic"),
                                 lambda m: setattr(self, "_depth_info", m), qos_profile_sensor_data)
        self.create_subscription(CameraInfo, g("colour_info_topic"),
                                 lambda m: setattr(self, "_colour_info", m), qos_profile_sensor_data)
        self.create_subscription(String, "wojtek/nav/status",
                                 lambda m: setattr(self, "_goto_status", m.data), latched)
        self.create_subscription(PointStamped, "wojtek/nav/pixel_goal", self._on_pixel, 10)
        self._pub_goal = self.create_publisher(PoseStamped, "wojtek/nav/goal", 10)
        self._pub_target = self.create_publisher(PointStamped, "wojtek/nav/pixel_target", latched)
        self._pub_status = self.create_publisher(String, "wojtek/nav/pixel_status", latched)
        self.create_timer(1.0 / TICK_HZ, self._tick)
        self._set_status("idle")
        self.get_logger().info(
            "pixel goal up: PointStamped on wojtek/nav/pixel_goal -> setpoints on "
            f"wojtek/nav/goal, standoff {g('standoff_m'):.2f} m"
        )

    # -- inputs ----------------------------------------------------------

    def _on_depth(self, msg):
        t = _stamp_s(msg.header)
        self._ring.append((t, msg))
        cutoff = t - self._ring_s
        while self._ring and self._ring[0][0] < cutoff:
            self._ring.pop(0)

    def _on_pixel(self, msg):
        if self._goal is not None and not self._tracker.done:
            self._set_status("replaced")
        self._goal = None
        self._pending = (msg, self._now_s())
        self._set_status("resolving")
        self._try_resolve()

    # -- the resolver ----------------------------------------------------

    def _try_resolve(self):
        msg, first_try = self._pending
        if self._depth_info is None or self._colour_info is None:
            return self._fail("no_frame", "camera intrinsics not received yet")
        stamp = _stamp_s(msg.header)
        depth = nearest_frame(self._ring, stamp, self.get_parameter("max_skew_s").value)
        if depth is None:
            return self._fail("no_frame", f"no depth image within max_skew of stamp {stamp:.3f}")
        # The pixel: normalised [0, 1] of the colour image, or absolute.
        u_c, v_c = msg.point.x, msg.point.y
        if 0.0 <= u_c <= 1.0 and 0.0 <= v_c <= 1.0:
            u_c, v_c = u_c * self._colour_info.width, v_c * self._colour_info.height
        u_d, v_d = colour_to_depth_pixel(u_c, v_c, self._colour_info.k, self._depth_info.k)
        arr = _depth_array(depth)
        z, share = depth_at(arr, u_d, v_d, int(self.get_parameter("patch_radius").value),
                            self.get_parameter("z_min").value, self.get_parameter("z_max").value)
        if z is None:
            return self._fail("no_depth", f"depth unusable at ({u_d:.0f}, {v_d:.0f}): {share:.0%} of the patch")
        x, y, zc = deproject(u_d, v_d, z, self._depth_info.k)
        pt = PointStamped()
        pt.header.frame_id = depth.header.frame_id
        pt.header.stamp = depth.header.stamp
        pt.point.x, pt.point.y, pt.point.z = float(x), float(y), float(zc)
        try:
            target = self._tf.transform(pt, self._odom, timeout=Duration(seconds=0.0))
            robot = self._tf.lookup_transform(self._odom, self._base, Time.from_msg(depth.header.stamp))
        except Exception as exc:  # noqa: BLE001 -- tf2 raises several unrelated types
            if self._now_s() - first_try < RESOLVE_RETRY_S:
                return None  # the odometry sample after the picture is still on its way
            return self._fail("no_tf", f"{type(exc).__name__}: {exc}")
        self._pending = None
        rx, ry = robot.transform.translation.x, robot.transform.translation.y
        dist = msg.point.z if msg.point.z > 0.0 else self.get_parameter("standoff_m").value
        gx, gy, yaw = standoff((target.point.x, target.point.y), (rx, ry), dist)
        goal = PoseStamped()
        goal.header.frame_id = self._odom
        goal.pose.position.x, goal.pose.position.y = float(gx), float(gy)
        goal.pose.orientation.z, goal.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        self._goal = goal
        self._pub_target.publish(target)
        self._tracker.start(self._now_s())
        self.get_logger().info(
            f"pixel ({u_c:.0f}, {v_c:.0f}) @ {stamp:.3f}: depth {z:.2f} m -> object "
            f"({target.point.x:.2f}, {target.point.y:.2f}) odom, setpoint ({gx:.2f}, {gy:.2f})"
        )
        self._set_status("sent")
        return None

    def _fail(self, word, why):
        self._pending = None
        self.get_logger().warning(f"pixel goal dropped ({word}): {why}")
        self._set_status(word)

    # -- the loop --------------------------------------------------------

    def _tick(self):
        if self._pending is not None:
            self._try_resolve()
        if self._goal is None or self._tracker.done:
            return
        action = self._tracker.step(self._goto_status, self._now_s())
        if action == "send":
            self._goal.header.stamp = self.get_clock().now().to_msg()
            self._pub_goal.publish(self._goal)
        elif action is not None:
            self._set_status(action)

    def _now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _set_status(self, word):
        self._pub_status.publish(String(data=word))
        self.get_logger().info(word)


def _depth_array(msg):
    if msg.encoding == "16UC1":
        return np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
    if msg.encoding == "32FC1":
        return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
    raise ValueError(f"unsupported depth encoding {msg.encoding}")


def main():
    rclpy.init()
    node = PixelGoalNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
