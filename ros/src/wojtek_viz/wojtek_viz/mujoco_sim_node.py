"""Real-time MuJoCo simulator node for wojtek.

Steps scene_mjx.xml (the exact physics the policy was trained on: position
servos kp=20/kd=1, forcerange +/-6, dt=0.004) paced to wall clock and bridges
it to ROS:

  subscribes  /wojtek/joint_targets (JointState, URDF convention)
  publishes   /joint_states (URDF convention, actuated + passive four-bar
              joints -> drives robot_state_publisher / RViz)
              /imu_sensor_broadcaster/imu (orientation, gyro, accel of the
              base, base_link frame; named like the real broadcaster's topic)
              TF odom -> base_link (ground truth base pose)
              /odom_vel (Twist, ground-truth base velocity, debugging)
  services    /sim/reset (std_srvs/Trigger) -- back to the initial pose

The initial_pose parameter selects where the robot spawns: "home" (the home
keyframe, standing) or "folded" (the real robot's boot/zeroing pose: lying
flat, hips straight, knees on the mechanical stop -- same definition as
real_io_node, see wojtek_policy.poses). The folded pose is reached by
settling from the home keyframe under gravity with folded ctrl targets, so
the four-bar closure stays consistent.
"""

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped, Twist
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster

from wojtek_policy.joint_map import JointMap
from wojtek_policy import poses

import mujoco  # noqa: isort  (heavier import last)

from ament_index_python.packages import get_package_share_directory


def _prepare_model_xml():
    """Copy the MJX scene next to the installed meshes so includes resolve.

    wojtek_viz ships the MJX XMLs in its share/config; the robot file
    expects meshdir ../meshes relative to itself, so rewrite it to the
    meshes installed by wojtek_description.
    """
    import re
    import tempfile
    from pathlib import Path

    share = Path(get_package_share_directory("wojtek_viz")) / "config"
    meshes = Path(get_package_share_directory("wojtek_description")) / "meshes"
    tmp = Path(tempfile.mkdtemp(prefix="wojtek_mj_"))
    robot = (share / "wojtek_mjx.xml").read_text()
    robot = re.sub(r'meshdir="[^"]*"', f'meshdir="{meshes}"', robot)
    (tmp / "wojtek_mjx.xml").write_text(robot)
    (tmp / "scene_mjx.xml").write_text((share / "scene_mjx.xml").read_text())
    return str(tmp / "scene_mjx.xml")


class MujocoSimNode(Node):
    def __init__(self):
        super().__init__("wojtek_mujoco_sim")
        policy_share = get_package_share_directory("wojtek_policy")
        self.declare_parameter("model_xml", "")
        self.declare_parameter("joint_map_yaml", f"{policy_share}/config/joint_map.yaml")
        self.declare_parameter("publish_rate_hz", 100.0)
        self.declare_parameter("realtime_factor", 1.0)
        self.declare_parameter("initial_pose", "home")
        self.declare_parameter("folded_knee_rad", poses.FOLDED_KNEE_RAD)

        xml = self.get_parameter("model_xml").value or _prepare_model_xml()
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)
        self.jmap = JointMap(self.get_parameter("joint_map_yaml").value)

        self.actuators = [self.model.actuator(i).name for i in range(self.model.nu)]
        # All hinge joints that exist in the URDF (first..fifth per leg; the
        # MJCF-only foot joints are skipped).
        mapped = set(self.jmap.names())
        self.pub_joints = [
            self.model.joint(i).name
            for i in range(self.model.njnt)
            if self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
            and self.model.joint(i).name in mapped
        ]
        self._qadr = np.array(
            [self.model.joint(n).qposadr[0] for n in self.pub_joints]
        )
        self._vadr = np.array([self.model.joint(n).dofadr[0] for n in self.pub_joints])
        # Actuator torque -> /joint_states effort: actuator index per
        # published joint (-1 for the passive four-bar joints, reported as
        # zero effort). Torques flip sign like velocities (URDF convention).
        act_idx = {n: i for i, n in enumerate(self.actuators)}
        self._jnt_act = list(zip(
            [act_idx.get(n, -1) for n in self.pub_joints],
            self.jmap.sign(self.pub_joints),
        ))
        self._sensor_adr = {
            n: self.model.sensor(n).adr[0]
            for n in ("orientation", "angular-velocity", "linear-acceleration")
        }

        self._reset()

        self.create_subscription(
            JointState, "wojtek/joint_targets", self._on_targets, 10
        )
        self._pub_js = self.create_publisher(JointState, "joint_states", 10)
        # Same topic name the real robot's imu_sensor_broadcaster publishes,
        # so policy/console subscribe one canonical name in sim and real.
        self._pub_imu = self.create_publisher(Imu, "imu_sensor_broadcaster/imu", 10)
        self._pub_vel = self.create_publisher(Twist, "odom_vel", 10)
        self._tf = TransformBroadcaster(self)
        self.create_service(Trigger, "sim/reset", self._srv_reset)

        self._wall_start = self.get_clock().now()
        period = 1.0 / self.get_parameter("publish_rate_hz").value
        self.create_timer(period, self._tick)
        self.get_logger().info(f"simulating {xml} at dt={self.model.opt.timestep}")

    def _reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        mujoco.mj_forward(self.model, self.data)
        initial_pose = self.get_parameter("initial_pose").value
        if initial_pose == "home":
            return
        if initial_pose != "folded":
            raise ValueError(
                f"initial_pose must be 'home' or 'folded', got {initial_pose!r}"
            )
        # Settle from standing into the folded boot pose: hold folded ctrl
        # targets and step until the robot has come to rest on the ground.
        # This keeps the passive four-bar joints consistent with the closure
        # instead of hand-writing a full qpos keyframe.
        self.data.ctrl[:] = poses.folded_ctrl(
            self.actuators, self.get_parameter("folded_knee_rad").value
        )
        for _ in range(int(round(2.0 / self.model.opt.timestep))):
            mujoco.mj_step(self.model, self.data)
        self.data.qvel[:] = 0.0
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _srv_reset(self, req, res):
        self._reset()
        self._wall_start = self.get_clock().now()
        res.success = True
        res.message = f"sim reset to {self.get_parameter('initial_pose').value} pose"
        return res

    def _on_targets(self, msg):
        idx = {n: i for i, n in enumerate(msg.name)}
        try:
            q_urdf = np.array([msg.position[idx[n]] for n in self.actuators])
        except KeyError as e:
            self.get_logger().warning(f"target message missing joint {e}")
            return
        self.data.ctrl[:] = self.jmap.to_mjc(self.actuators, q_urdf)

    def _tick(self):
        # Step physics up to wall-clock time (bounded, so a stall cannot
        # trigger a death spiral).
        rt = self.get_parameter("realtime_factor").value
        elapsed = (self.get_clock().now() - self._wall_start).nanoseconds * 1e-9 * rt
        steps = 0
        while self.data.time < elapsed and steps < 200:
            mujoco.mj_step(self.model, self.data)
            steps += 1

        now = self.get_clock().now().to_msg()

        js = JointState()
        js.header.stamp = now
        js.name = self.pub_joints
        js.position = self.jmap.to_urdf(
            self.pub_joints, self.data.qpos[self._qadr]
        ).tolist()
        js.velocity = self.jmap.vel_to_urdf(
            self.pub_joints, self.data.qvel[self._vadr]
        ).tolist()
        js.effort = [
            float(s * self.data.actuator_force[i]) if i >= 0 else 0.0
            for i, s in self._jnt_act
        ]
        self._pub_js.publish(js)

        imu = Imu()
        imu.header.stamp = now
        imu.header.frame_id = "base_link"
        adr = self._sensor_adr["orientation"]
        w, x, y, z = self.data.sensordata[adr : adr + 4]
        imu.orientation.w, imu.orientation.x = float(w), float(x)
        imu.orientation.y, imu.orientation.z = float(y), float(z)
        adr = self._sensor_adr["angular-velocity"]
        gx, gy, gz = self.data.sensordata[adr : adr + 3]
        imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = (
            float(gx), float(gy), float(gz),
        )
        adr = self._sensor_adr["linear-acceleration"]
        ax, ay, az = self.data.sensordata[adr : adr + 3]
        imu.linear_acceleration.x, imu.linear_acceleration.y = float(ax), float(ay)
        imu.linear_acceleration.z = float(az)
        self._pub_imu.publish(imu)

        tf = TransformStamped()
        tf.header.stamp = now
        tf.header.frame_id = "odom"
        tf.child_frame_id = "base_link"
        px, py, pz = self.data.qpos[0:3]
        qw, qx, qy, qz = self.data.qpos[3:7]
        tf.transform.translation.x, tf.transform.translation.y = float(px), float(py)
        tf.transform.translation.z = float(pz)
        tf.transform.rotation.w, tf.transform.rotation.x = float(qw), float(qx)
        tf.transform.rotation.y, tf.transform.rotation.z = float(qy), float(qz)
        self._tf.sendTransform(tf)

        vel = Twist()
        vel.linear.x, vel.linear.y, vel.linear.z = map(float, self.data.qvel[0:3])
        vel.angular.x, vel.angular.y, vel.angular.z = map(float, self.data.qvel[3:6])
        self._pub_vel.publish(vel)


def main():
    rclpy.init()
    node = MujocoSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
