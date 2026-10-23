"""Leg-kinematics + IMU odometry.

Principle: a stance foot is pinned to the floor, so in the base frame
    0 = v_base + omega x r_foot + J q_dot
and every stance leg is an independent measurement of
    v_base = -(omega x r_foot + J q_dot).
The measurements are averaged over the stance set, rotated into the odom
frame with the IMU orientation (yaw zeroed at startup), and integrated
into a planar pose. Roll/pitch in the published pose come straight from
the IMU; z stays 0 -- Nav2 is planar and integrated z would only drift.

Inputs
  /wojtek/joint_states_abs   absolute URDF angles + velocities, 200 Hz
                             (real_io_node; the same topic also carries
                             the 8 passive four-bar joints for RViz --
                             those messages are filtered out by name)
  /joint_states              effort only (activation-relative frame does
                             not matter: torque carries no offset);
                             knee torque gates unloaded legs out of the
                             stance set
  /imu_sensor_broadcaster/imu  orientation (ESKF / sim ground truth) and
                             gyro; the mount is upright (rpy 0 0 0), so
                             sensor axes == base axes

Outputs
  /wojtek/odom               nav_msgs/Odometry (twist in base frame)
  /wojtek/odom/debug         Float64MultiArray, layout [4x knee |tau|,
                             4x stance flag] in wojtek_policy.poses leg
                             order -- for threshold tuning
  TF odom->base_link         only with publish_tf:=true. Off by default:
                             in simulation the MuJoCo plugin publishes
                             the ground-truth transform, and on the real
                             robot launch_common's static identity owns
                             the edge until it is deliberately replaced.

Contact detection is by foot height: the stance feet share the ground
plane, so after levelling the base with the IMU's roll/pitch, a foot
within contact_z_delta of the lowest one is in stance. Measured on the
walking policy (kp=40), knee torque alone does NOT separate stance from
swing -- the swing-tracking torques smear 0-6 N*m over the stance range
-- so |knee torque| serves only as a floor (contact_tau_floor) that
rejects unloaded legs, e.g. the robot carried in the air, where the
height test alone would call all four feet "stance" and integrate the
stepping motion.
"""

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
)
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray, String
from tf2_ros import TransformBroadcaster

from wojtek_odometry.leg_kinematics import LEGS, LegKinematics


def _quat_to_matrix(x, y, z, w):
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array([
        [1 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1 - (xx + yy)],
    ])


def _yaw_of(rot):
    return float(np.arctan2(rot[1, 0], rot[0, 0]))


def _rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class LegOdometryNode(Node):
    def __init__(self):
        super().__init__("wojtek_leg_odometry")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("publish_tf", False)
        self.declare_parameter("publish_rate_hz", 50.0)
        # A foot within this many metres of the lowest (IMU-levelled) foot
        # counts as stance. Tight on purpose: the quiet gait's rear feet
        # clear the floor by barely 2 cm, and measured against ground truth
        # the velocity error is small only in the 0-2 mm band (-0.03 m/s)
        # while already at 5-10 mm the "foot" is airborne and reads -0.3.
        # An empty stance set at gait transitions is fine -- the estimate
        # freezes for those samples instead of averaging in swing legs.
        self.declare_parameter("contact_z_delta", 0.003)
        # |knee torque| floor [N*m]: legs below it never count as stance,
        # which keeps a carried robot's stepping out of the integrator.
        self.declare_parameter("contact_tau_floor", 0.8)
        # First-order low-pass on the fused base velocity [Hz]; 0 disables.
        # Stance transitions kick the raw estimate, this smooths them without
        # adding meaningful lag at gait timescales.
        self.declare_parameter("velocity_lowpass_hz", 8.0)
        # EFFECTIVE rolling radius of the foot sphere [m]. Geometrically the
        # sphere is 0.046 m, but the contact does not roll ideally: fitted
        # against the simulator's ground truth the scale bias crosses zero
        # at ~0.031 m (0.046 over-corrects to +7 %, 0 under-reads -16 %).
        # A calibration constant in the joint_map tradition, not a guess.
        self.declare_parameter("rolling_radius", 0.031)
        # Process every Nth joint_states_abs message. The per-message numpy
        # kinematics (4 Jacobians) cannot keep 200 Hz on the Pi 4's A72 --
        # measured: the executor floods and the 50 Hz publish starves to
        # ~6 Hz. The integrator is stride-safe by construction (velocity is
        # sampled at the processed instant and integrated over the actual
        # stamp dt), so 50 Hz processing (stride 4) loses nothing at gait
        # timescales. 1 = every message (PC-class hosts).
        self.declare_parameter("input_stride", 1)

        self._odom_frame = self.get_parameter("odom_frame").value
        self._base_frame = self.get_parameter("base_frame").value
        self._z_delta = self.get_parameter("contact_z_delta").value
        self._tau_floor = self.get_parameter("contact_tau_floor").value
        self._lp_hz = self.get_parameter("velocity_lowpass_hz").value
        self._roll_r = self.get_parameter("rolling_radius").value
        self._input_stride = max(1, int(self.get_parameter("input_stride").value))
        self._input_count = 0

        self._legs = None  # built when robot_description arrives
        self._actuated = None  # flat joint-name list, LEGS order
        self._stance = np.zeros(4, dtype=bool)
        self._tau_knee = np.zeros(4)
        self._rot_imu = None  # base -> world, straight from the IMU
        self._rot_yaw0 = None  # world -> odom (yaw zero at first IMU sample)
        self._gyro = np.zeros(3)
        self._pos = np.zeros(2)
        self._vel_base = np.zeros(3)
        self._last_stamp = None

        # robot_description is latched by robot_state_publisher.
        self.create_subscription(
            String, "robot_description", self._on_urdf,
            QoSProfile(
                depth=1,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                history=QoSHistoryPolicy.KEEP_LAST,
            ),
        )
        self.create_subscription(
            JointState, "wojtek/joint_states_abs", self._on_joints_abs, 10)
        self.create_subscription(JointState, "joint_states", self._on_efforts, 10)
        self.create_subscription(Imu, "imu_sensor_broadcaster/imu", self._on_imu, 10)

        self._pub_odom = self.create_publisher(Odometry, "wojtek/odom", 10)
        self._pub_debug = self.create_publisher(
            Float64MultiArray, "wojtek/odom/debug", 10)
        self._tf = TransformBroadcaster(self) \
            if self.get_parameter("publish_tf").value else None
        self._publish_period = 1.0 / self.get_parameter("publish_rate_hz").value
        self._last_publish = None

        self.get_logger().info(
            "leg odometry up, waiting for robot_description "
            f"(contact z_delta {self._z_delta:g} m, |tau| floor "
            f"{self._tau_floor:g} N*m, "
            f"publish_tf={'on' if self._tf else 'off'})")

    # ------------------------------------------------------------ inputs
    def _on_urdf(self, msg):
        if self._legs is not None:
            return
        self._legs = [LegKinematics(msg.data, leg) for leg in LEGS]
        self._actuated = [name for k in self._legs for name in k.actuated]
        self.get_logger().info("kinematics built from robot_description")

    def _on_efforts(self, msg):
        if self._actuated is None or len(msg.effort) != len(msg.name):
            return
        idx = {n: i for i, n in enumerate(msg.name)}
        for li in range(4):
            knee = self._actuated[3 * li + 2]
            if knee in idx:
                self._tau_knee[li] = abs(msg.effort[idx[knee]])

    def _on_imu(self, msg):
        o = msg.orientation
        rot = _quat_to_matrix(o.x, o.y, o.z, o.w)
        if self._rot_yaw0 is None:
            if abs(o.x) + abs(o.y) + abs(o.z) + abs(o.w) < 1e-9:
                return  # "no fusion yet" all-zero marker from the ESKF
            self._rot_yaw0 = _rot_z(-_yaw_of(rot))
            self.get_logger().info("IMU alive, odom yaw zeroed")
        self._rot_imu = rot
        g = msg.angular_velocity
        self._gyro = np.array([g.x, g.y, g.z])

    # ------------------------------------------------------- integration
    def _on_joints_abs(self, msg):
        if self._legs is None or self._rot_imu is None:
            return
        if len(msg.velocity) != len(msg.name):
            return  # the passive-joint helper message has no velocities
        self._input_count += 1
        if self._input_count % self._input_stride:
            return  # deliberate decimation -- see the input_stride parameter
        idx = {n: i for i, n in enumerate(msg.name)}
        if not all(n in idx for n in self._actuated):
            return

        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        dt = 0.0 if self._last_stamp is None else stamp - self._last_stamp
        self._last_stamp = stamp
        if dt <= 0.0 or dt > 0.1:
            return

        feet = []
        for kin in self._legs:
            q = np.array([msg.position[idx[n]] for n in kin.actuated])
            qd = np.array([msg.velocity[idx[n]] for n in kin.actuated])
            r_foot, jac, omega_leg = kin.foot_state(q, qd)
            feet.append((r_foot, jac @ qd, omega_leg))

        # Stance = levelled foot height within z_delta of the lowest foot,
        # and the knee actually loaded. Yaw does not move z, so the raw IMU
        # rotation levels just as well as the yaw-zeroed one.
        z = np.array([(self._rot_imu @ r)[2] for r, _, _ in feet])
        self._stance = (z < z.min() + self._z_delta) \
            & (self._tau_knee > self._tau_floor)

        # A stance foot is a rolling sphere: its CONTACT POINT is pinned,
        # its centre translates at omega_shank x (R * up). Without this term
        # the estimate under-reads by the gait's pitch rate times R (~15 %).
        up_b = self._rot_imu.T @ np.array([0.0, 0.0, 1.0])
        samples = [
            -(np.cross(self._gyro, r_foot) + v_foot_rel)
            + np.cross(self._gyro + omega_leg, self._roll_r * up_b)
            for li, (r_foot, v_foot_rel, omega_leg) in enumerate(feet)
            if self._stance[li]
        ]

        # No stance leg: the robot is airborne or lifted; freeze rather than
        # coast on a stale velocity.
        v_raw = np.mean(samples, axis=0) if samples else np.zeros(3)
        if self._lp_hz > 0.0:
            alpha = 1.0 - np.exp(-2.0 * np.pi * self._lp_hz * dt)
            self._vel_base = self._vel_base + alpha * (v_raw - self._vel_base)
        else:
            self._vel_base = v_raw

        rot_ob = self._rot_yaw0 @ self._rot_imu  # base -> odom
        v_odom = rot_ob @ self._vel_base
        self._pos += v_odom[:2] * dt

        if (self._last_publish is None
                or stamp - self._last_publish >= self._publish_period):
            self._last_publish = stamp
            self._publish(msg.header.stamp, rot_ob)

    def _publish(self, stamp, rot_ob):
        # Full 3D orientation (roll/pitch from the IMU), planar position.
        # Quaternion straight from the rotation matrix.
        tr = np.trace(rot_ob)
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2
            qw, qx, qy, qz = 0.25 * s, (rot_ob[2, 1] - rot_ob[1, 2]) / s, \
                (rot_ob[0, 2] - rot_ob[2, 0]) / s, (rot_ob[1, 0] - rot_ob[0, 1]) / s
        else:
            i = int(np.argmax(np.diag(rot_ob)))
            j, k = (i + 1) % 3, (i + 2) % 3
            s = np.sqrt(1.0 + rot_ob[i, i] - rot_ob[j, j] - rot_ob[k, k]) * 2
            q = [0.0, 0.0, 0.0]
            q[i] = 0.25 * s
            q[j] = (rot_ob[j, i] + rot_ob[i, j]) / s
            q[k] = (rot_ob[k, i] + rot_ob[i, k]) / s
            qw = (rot_ob[k, j] - rot_ob[j, k]) / s
            qx, qy, qz = q

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self._odom_frame
        odom.child_frame_id = self._base_frame
        odom.pose.pose.position.x = float(self._pos[0])
        odom.pose.pose.position.y = float(self._pos[1])
        odom.pose.pose.orientation.x = float(qx)
        odom.pose.pose.orientation.y = float(qy)
        odom.pose.pose.orientation.z = float(qz)
        odom.pose.pose.orientation.w = float(qw)
        odom.twist.twist.linear.x = float(self._vel_base[0])
        odom.twist.twist.linear.y = float(self._vel_base[1])
        odom.twist.twist.linear.z = float(self._vel_base[2])
        odom.twist.twist.angular.x = float(self._gyro[0])
        odom.twist.twist.angular.y = float(self._gyro[1])
        odom.twist.twist.angular.z = float(self._gyro[2])
        # Rough constants: leg odometry drifts in x/y/yaw, is direct in the
        # rest. Enough for a downstream EKF to weigh it sensibly.
        odom.pose.covariance[0] = odom.pose.covariance[7] = 1e-3
        odom.pose.covariance[35] = 1e-3
        odom.twist.covariance[0] = odom.twist.covariance[7] = 1e-3
        odom.twist.covariance[35] = 1e-4
        self._pub_odom.publish(odom)

        debug = Float64MultiArray()
        debug.data = list(self._tau_knee) + [float(s) for s in self._stance]
        self._pub_debug.publish(debug)

        if self._tf is not None:
            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id = self._odom_frame
            tf.child_frame_id = self._base_frame
            tf.transform.translation.x = float(self._pos[0])
            tf.transform.translation.y = float(self._pos[1])
            tf.transform.rotation.x = float(qx)
            tf.transform.rotation.y = float(qy)
            tf.transform.rotation.z = float(qz)
            tf.transform.rotation.w = float(qw)
            self._tf.sendTransform(tf)


def main(args=None):
    rclpy.init(args=args)
    node = LegOdometryNode()
    # The wait-set executor rebuilds its wait set on EVERY wake, and this
    # node wakes ~200x/s (4 subscriptions) -- profiled on the Pi 4 that
    # plumbing alone ate ~40% of the node's CPU. Jazzy's (experimental)
    # EventsExecutor drives callbacks from listener events instead and
    # skips that entirely; fall back to the classic spin where it is
    # missing so nothing breaks on older distros.
    try:
        from rclpy.experimental import EventsExecutor
        executor = EventsExecutor()
    except ImportError:
        executor = None
    try:
        if executor is None:
            rclpy.spin(node)
        else:
            executor.add_node(node)
            executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
