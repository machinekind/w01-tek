"""50 Hz policy node: sensors in, joint position targets out.

Identical for simulation and the real robot -- only topic remaps and
parameters differ:

  subscribes  /joint_states  (sensor_msgs/JointState, URDF convention, absolute)
              /imu/data      (sensor_msgs/Imu)
              /cmd_vel       (geometry_msgs/Twist)
              /wojtek/cmd_height (std_msgs/Float32; 4-D-command policies only,
                               overrides the command_height parameter)
  publishes   /wojtek/joint_targets (sensor_msgs/JointState, URDF convention,
                                  absolute; 12 actuated joints)
  services    /wojtek/enable (std_srvs/SetBool), /wojtek/reset (std_srvs/Trigger)

The policy itself (wojtek_policy.policy.WojtekPolicy) works in the MuJoCo/training
convention; this node converts on both edges using joint_map.yaml.
"""

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float32
from std_srvs.srv import SetBool, Trigger

from wojtek_policy.joint_map import JointMap
from wojtek_policy.policy import WojtekPolicy, gravity_from_quat

# Training command ranges: never ask the policy for more than it was trained
# to track. Fallback for metas without cmd_low/cmd_high (fbb_v3, env.py
# default_config); newer exports carry their own ranges.
CMD_LOW = np.array([-0.6, -0.4, -0.7])
CMD_HIGH = np.array([0.6, 0.4, 0.7])


def rpy_to_mat(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return (
        np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    )


class PolicyNode(Node):
    def __init__(self):
        super().__init__("wojtek_policy")
        share = get_package_share_directory("wojtek_policy")
        self.declare_parameter("policy_dir", f"{share}/config")
        self.declare_parameter("joint_map_yaml", f"{share}/config/joint_map.yaml")
        # Rotation of the IMU frame expressed in base_link (URDF imu_joint
        # rpy). Sim publishes IMU already in base_link -> zeros.
        self.declare_parameter("imu_mount_rpy", [0.0, 0.0, 0.0])
        self.declare_parameter("clamp_knee", False)
        self.declare_parameter("auto_enable", True)
        self.declare_parameter("soft_start_s", 1.0)
        self.declare_parameter("watchdog_timeout_s", 0.2)
        # Use the accelerometer+gyro complementary filter instead of the IMU
        # orientation quaternion (for IMUs without usable fusion).
        self.declare_parameter("gravity_from_accel", False)
        # Commanded standing height (m) for 4-D-command policies (v8+);
        # live-settable: ros2 param set /wojtek_policy command_height 0.15
        self.declare_parameter("command_height", 0.13)
        # EMA low-pass on motor targets (wojtek_rl.env action_filter mirror);
        # 0 = off. Anti-vibration knob, live-settable:
        #   ros2 param set /wojtek_policy action_ema 0.3
        self.declare_parameter("action_ema", 0.0)

        pdir = self.get_parameter("policy_dir").value
        self.policy = WojtekPolicy(
            f"{pdir}/policy.npz",
            clamp_knee=self.get_parameter("clamp_knee").value,
        )
        self.jmap = JointMap(self.get_parameter("joint_map_yaml").value)
        self.joint_names = self.policy.joint_names  # policy actuator order
        self.imu_mount = rpy_to_mat(*self.get_parameter("imu_mount_rpy").value)
        m = self.policy.meta
        self._cmd_low = np.array(m.get("cmd_low", CMD_LOW), np.float64)
        self._cmd_high = np.array(m.get("cmd_high", CMD_HIGH), np.float64)
        self._height_range = m.get("cmd_height_range")

        self._q_urdf = None  # (12,) latest, actuator order
        self._dq_urdf = None
        self._gyro_base = None
        self._gravity_base = None
        self._grav_filt = np.array([0.0, 0.0, -1.0])
        self._cmd = np.zeros(3)
        self._cmd_height = None  # latest wojtek/cmd_height, overrides the param
        self._joints_stamp = None
        self._imu_stamp = None
        self._enabled = self.get_parameter("auto_enable").value
        self._enable_time = None
        self._was_running = False

        self.create_subscription(JointState, "joint_states", self._on_joints, 10)
        self.create_subscription(Imu, "imu/data", self._on_imu, 10)
        self.create_subscription(Twist, "cmd_vel", self._on_cmd, 10)
        self.create_subscription(Float32, "wojtek/cmd_height", self._on_height, 10)
        self._pub = self.create_publisher(JointState, "wojtek/joint_targets", 10)
        self.create_service(SetBool, "wojtek/enable", self._srv_enable)
        self.create_service(Trigger, "wojtek/reset", self._srv_reset)
        self.create_timer(self.policy.ctrl_dt, self._tick)
        self.get_logger().info(
            f"policy {self.policy.meta['run_name']} loaded, "
            f"{1.0 / self.policy.ctrl_dt:.0f} Hz, enabled={self._enabled}"
        )

    # -- inputs --------------------------------------------------------------
    def _on_joints(self, msg):
        idx = {n: i for i, n in enumerate(msg.name)}
        try:
            order = [idx[n] for n in self.joint_names]
        except KeyError:
            return  # partial message (e.g. passive-joints-only publisher)
        self._q_urdf = np.array([msg.position[i] for i in order])
        if len(msg.velocity) == len(msg.name):
            self._dq_urdf = np.array([msg.velocity[i] for i in order])
        else:
            self._dq_urdf = np.zeros(12)
        self._joints_stamp = self.get_clock().now()

    def _on_imu(self, msg):
        gyro = np.array(
            [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
        )
        self._gyro_base = self.imu_mount @ gyro
        q = msg.orientation
        have_quat = (
            not self.get_parameter("gravity_from_accel").value
            and msg.orientation_covariance[0] >= 0.0
            and (q.w, q.x, q.y, q.z) != (0.0, 0.0, 0.0, 0.0)
        )
        if have_quat:
            g_imu = gravity_from_quat(q.w, q.x, q.y, q.z)
            self._gravity_base = self.imu_mount @ g_imu
        else:
            # Complementary filter on the gravity direction in the IMU frame:
            # propagate with gyro (g_dot = -w x g), correct with accelerometer
            # (specific force ~ -g when not accelerating).
            acc = np.array(
                [
                    msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z,
                ]
            )
            g = self._grav_filt
            g = g - self.policy.ctrl_dt * np.cross(gyro, g)
            if np.linalg.norm(acc) > 1e-3:
                g_meas = -acc / np.linalg.norm(acc)
                g = 0.98 * g + 0.02 * g_meas
            self._grav_filt = g / np.linalg.norm(g)
            self._gravity_base = self.imu_mount @ self._grav_filt
        self._imu_stamp = self.get_clock().now()

    def _on_cmd(self, msg):
        cmd = np.array([msg.linear.x, msg.linear.y, msg.angular.z])
        self._cmd = np.clip(cmd, self._cmd_low, self._cmd_high)

    def _on_height(self, msg):
        self._cmd_height = float(msg.data)

    def _full_cmd(self):
        """Velocity command plus, for 4-D-command policies, the height."""
        if self.policy.cmd_dim < 4:
            return self._cmd
        h = self._cmd_height
        if h is None:
            h = self.get_parameter("command_height").value
        if self._height_range is not None:
            h = float(np.clip(h, self._height_range[0], self._height_range[1]))
        return np.concatenate([self._cmd, [h]])

    # -- services ------------------------------------------------------------
    def _srv_enable(self, req, res):
        if req.data and not self._enabled:
            self.policy.reset()
            self._enable_time = self.get_clock().now()
        self._enabled = req.data
        res.success = True
        res.message = f"policy {'enabled' if req.data else 'disabled'}"
        self.get_logger().info(res.message)
        return res

    def _srv_reset(self, req, res):
        self.policy.reset()
        self._enable_time = self.get_clock().now()
        res.success = True
        res.message = "policy state reset"
        return res

    # -- control loop ----------------------------------------------------------
    def _fresh(self, stamp):
        if stamp is None:
            return False
        timeout = self.get_parameter("watchdog_timeout_s").value
        return (self.get_clock().now() - stamp).nanoseconds < timeout * 1e9

    def _tick(self):
        if not self._enabled:
            self._was_running = False
            return
        fresh = self._fresh(self._joints_stamp)
        if self.policy.needs_imu:
            fresh = fresh and self._fresh(self._imu_stamp)
        if not fresh:
            if self._was_running:
                self.get_logger().warning(
                    "sensor data stale -- holding (no targets published)",
                    throttle_duration_sec=1.0,
                )
            self._was_running = False
            return
        # A single NaN in the inputs would poison the policy permanently
        # (last_action feeds back into the observation), so treat non-finite
        # input like stale data: hold, and reset the policy state on
        # recovery via the _was_running edge. Seen in practice from the IMU
        # broadcaster's NaN placeholders right after activation. IMU terms
        # only exist (and only matter) for policies that observe them.
        inputs = [self._q_urdf, self._dq_urdf, self._cmd]
        if self.policy.needs_imu:
            inputs = [self._gyro_base, self._gravity_base, *inputs]
        inputs = np.concatenate(inputs)
        if not np.all(np.isfinite(inputs)):
            self.get_logger().warning(
                "non-finite sensor input -- holding (no targets published)",
                throttle_duration_sec=1.0,
            )
            self._was_running = False
            return
        if not self._was_running:
            # (Re)entering RUN: restart the gait clock and the soft-start ramp.
            self.policy.reset()
            self._enable_time = self.get_clock().now()
            self._was_running = True

        self.policy.action_ema = float(
            np.clip(self.get_parameter("action_ema").value, 0.0, 0.95)
        )
        q_mjc = self.jmap.to_mjc(self.joint_names, self._q_urdf)
        dq_mjc = self.jmap.vel_to_mjc(self.joint_names, self._dq_urdf)
        targets_mjc = self.policy.step(
            self._gyro_base, self._gravity_base, q_mjc, dq_mjc, self._full_cmd()
        )

        # Soft start: blend from the measured pose to the policy output.
        soft = self.get_parameter("soft_start_s").value
        beta = 1.0
        if soft > 0 and self._enable_time is not None:
            t = (self.get_clock().now() - self._enable_time).nanoseconds * 1e-9
            beta = min(1.0, t / soft)
        targets_mjc = beta * targets_mjc + (1.0 - beta) * q_mjc

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.jmap.to_urdf(self.joint_names, targets_mjc).tolist()
        self._pub.publish(msg)


def main():
    rclpy.init()
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
