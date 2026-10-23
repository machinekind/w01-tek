"""Real-time MuJoCo simulator node for wojtek.

Steps scene_mjx.xml paced to wall clock and bridges it to ROS. With the
`policy` parameter set, the servo gains and torque cap come from that
policy's contract (policy_meta.json pd block), so the simulated plant is
exactly the one the policy trained against; without it the XML defaults
(kp=20/kd=1, forcerange +/-6) apply. Topics:

  subscribes  /wojtek/joint_targets (JointState, URDF convention)
  publishes   /joint_states (URDF convention, actuated + passive four-bar
              joints -> drives robot_state_publisher / RViz)
              /imu_sensor_broadcaster/imu (orientation, gyro, accel of the
              base, base_link frame; named like the real broadcaster's topic)
              TF odom -> base_link (ground truth base pose)
              /odom_vel (Twist, ground-truth base velocity, debugging)
              /sim/rtf (Float32, sim-time/wall-time ratio over 1 s windows)
              with camera:=true (default) a D435-compatible virtual camera
              rendered offscreen from the robot's head pose (issue #91):
              /camera/camera/depth/image_rect_raw  16UC1 mm, 424x240, ~15 Hz
              /camera/camera/depth/camera_info     matching intrinsics
              /camera/camera/color/image_raw       rgb8, ~5 Hz (for the VLM)
              /camera/camera/color/camera_info
              (sensor-data QoS; stamps match the odom->base_link TF; see
              wojtek_pc.camera_spec for the contract, depth_camera for the
              renderer, and machinekind/wojtek#91 for the requirements)
  services    /sim/reset (std_srvs/Trigger) -- back to the initial pose

The initial_pose parameter selects where the robot spawns: "home" (the home
keyframe, standing) or "folded" (the real robot's boot/zeroing pose: lying
flat, hips straight, knees on the mechanical stop -- same definition as
real_io_node, see wojtek_policy.poses). The folded pose is reached by
settling from the home keyframe under gravity with folded ctrl targets, so
the four-bar closure stays consistent.
"""

import os
import sys
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped, Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, Imu, JointState
from std_msgs.msg import Float32
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster

from wojtek_policy.joint_map import JointMap
from wojtek_policy import poses
from wojtek_pc import camera_spec

# `import mujoco` RAISES if MUJOCO_GL names a backend whose library is
# missing (no libEGL in a slim container, say). Physics needs no GL at all,
# so a broken backend must degrade to camera-off, never kill the sim: try
# the preferred backend, and on failure fall back to MUJOCO_GL=disabled.
# A failed import leaves partial modules behind -- clear them before retrying.
_PREFERRED_GL = os.environ.get("MUJOCO_GL") or (
    "egl" if sys.platform.startswith("linux") else "glfw"
)
os.environ["MUJOCO_GL"] = _PREFERRED_GL
try:
    import mujoco  # noqa: isort  (heavier import last)

    GL_BACKEND = None if _PREFERRED_GL == "disabled" else _PREFERRED_GL
except Exception:
    for _m in [m for m in sys.modules
               if m == "mujoco" or m.startswith("mujoco.")]:
        del sys.modules[_m]
    os.environ["MUJOCO_GL"] = "disabled"
    import mujoco  # noqa: isort

    GL_BACKEND = None

from ament_index_python.packages import get_package_share_directory


def _prepare_model_xml():
    """Copy the MJX scene next to the installed meshes so includes resolve.

    wojtek_pc ships the MJX XMLs in its share/config; the robot file
    expects meshdir ../meshes relative to itself, so rewrite it to the
    meshes installed by wojtek_description.
    """
    import re
    import tempfile
    from pathlib import Path

    share = Path(get_package_share_directory("wojtek_pc")) / "config"
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
        # Same reference policy_node gets. When set, the servo gains and
        # torque cap from the policy's contract are written into the model,
        # so the simulated plant is the one the policy trained against (the
        # XML carries the kp=20/kd=1/±6 defaults, wrong for e.g. a kp80
        # policy).
        self.declare_parameter("policy", "")
        # D435-compatible virtual camera (issue #91): topics, encodings and
        # rates mirror the real perception stack so depth consumers/VLM
        # run unchanged. `camera:=false` on the launch is the off-switch for
        # weak machines; rendering runs on its own thread, never in _tick.
        self.declare_parameter("camera", True)
        self.declare_parameter("camera_depth_hz", 15.0)
        self.declare_parameter("camera_color_hz", 5.0)
        self.declare_parameter("depth_min_m", camera_spec.DEPTH_MIN_M)
        self.declare_parameter("depth_max_m", camera_spec.DEPTH_MAX_M)

        xml = self.get_parameter("model_xml").value or _prepare_model_xml()
        self._camera_on = bool(self.get_parameter("camera").value)
        if self._camera_on and GL_BACKEND is None:
            self.get_logger().warning(
                "no MuJoCo GL backend available (MUJOCO_GL fell back to "
                "'disabled') -- camera off, physics unaffected"
            )
            self._camera_on = False
        self.model = None
        if self._camera_on:
            from wojtek_pc.depth_camera import load_model_with_camera

            try:
                self.model = load_model_with_camera(xml)
            except Exception as e:
                self.get_logger().error(
                    f"camera injection failed ({e}) -- camera off"
                )
                self._camera_on = False
        if self.model is None:
            self.model = mujoco.MjModel.from_xml_path(xml)
        policy_ref = self.get_parameter("policy").value
        if policy_ref:
            from wojtek_policy.policy_source import load_meta, pd_settings
            meta, source = load_meta(policy_ref)
            pd = pd_settings(meta)
            self.model.actuator_gainprm[:, 0] = pd["kp"]
            self.model.actuator_biasprm[:, 1] = -pd["kp"]
            self.model.actuator_biasprm[:, 2] = -pd["kd"]
            self.model.actuator_forcerange[:, 0] = -pd["max_torque"]
            self.model.actuator_forcerange[:, 1] = pd["max_torque"]
            self.get_logger().info(
                f"plant matched to {meta['run_name']} ({source}): "
                f"kp={pd['kp']:g} kd={pd['kd']:g} "
                f"max_torque={pd['max_torque']:g}"
            )
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
        # Actuated-joint dof addresses, actuator order: where the tau_ff
        # feed-forward torque (joint_targets.effort) is applied as
        # qfrc_applied -- the same mechanism the training env uses.
        self._act_vadr = np.array([
            self.model.jnt_dofadr[self.model.actuator_trnid[i, 0]]
            for i in range(self.model.nu)
        ])
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

        # Real-time factor telemetry: sim-time advance / wall advance over
        # ~1 s windows. Makes "the camera does not slow physics down" a
        # measurement instead of an opinion.
        self._pub_rtf = self.create_publisher(Float32, "sim/rtf", 10)
        self._rtf_ref = None  # (wall ns, sim time) of the last window edge
        self._warned_catchup = False

        # Camera: _tick stores a (stamp, qpos, qvel) snapshot under a lock
        # (a memcpy, sub-microsecond); the render thread consumes the newest
        # one. Images carry the SNAPSHOT stamp -- the same stamp the
        # odom->base_link TF was broadcast with -- so tf2 lookups at the
        # image time are exact. GL contexts are thread-affine: the render
        # thread creates, uses and destroys them, nothing else touches GL.
        self._snap = None
        self._snap_lock = threading.Lock()
        self._render_stop = threading.Event()
        self._render_thread = None
        if self._camera_on:
            from wojtek_pc.depth_camera import SimDepthCamera

            self._cam = SimDepthCamera(
                self.model,
                min_m=self.get_parameter("depth_min_m").value,
                max_m=self.get_parameter("depth_max_m").value,
            )
            self._pub_depth = self.create_publisher(
                Image, camera_spec.DEPTH_TOPIC, qos_profile_sensor_data
            )
            self._pub_depth_info = self.create_publisher(
                CameraInfo, camera_spec.DEPTH_INFO_TOPIC, qos_profile_sensor_data
            )
            self._pub_color = self.create_publisher(
                Image, camera_spec.COLOR_TOPIC, qos_profile_sensor_data
            )
            self._pub_color_info = self.create_publisher(
                CameraInfo, camera_spec.COLOR_INFO_TOPIC, qos_profile_sensor_data
            )
            self._render_thread = threading.Thread(
                target=self._render_loop, name="d435_render", daemon=True
            )
            self._render_thread.start()

        self._wall_start = self.get_clock().now()
        period = 1.0 / self.get_parameter("publish_rate_hz").value
        self.create_timer(period, self._tick)
        self.get_logger().info(
            f"simulating {xml} at dt={self.model.opt.timestep} "
            f"(camera={'on, GL=' + str(GL_BACKEND) if self._camera_on else 'off'})"
        )

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
        # tau_ff policies ride a feed-forward torque on the same message
        # (URDF sign convention, like velocities); applied on top of the PD
        # actuators exactly as in training (qfrc_applied). Cleared when the
        # message carries no effort, so switching policies cannot leave a
        # stale torque behind.
        self.data.qfrc_applied[self._act_vadr] = 0.0
        if len(msg.effort) == len(msg.name):
            tau_urdf = np.array([msg.effort[idx[n]] for n in self.actuators])
            self.data.qfrc_applied[self._act_vadr] = self.jmap.vel_to_mjc(
                self.actuators, tau_urdf
            )

    def _tick(self):
        # Step physics up to wall-clock time (bounded, so a stall cannot
        # trigger a death spiral).
        rt = self.get_parameter("realtime_factor").value
        elapsed = (self.get_clock().now() - self._wall_start).nanoseconds * 1e-9 * rt
        steps = 0
        while self.data.time < elapsed and steps < 200:
            mujoco.mj_step(self.model, self.data)
            steps += 1
        if steps >= 200 and self.data.time < elapsed and not self._warned_catchup:
            self._warned_catchup = True
            self.get_logger().warning(
                "physics hit the 200-step catch-up cap -- sim is running "
                "slower than wall clock (check sim/rtf)"
            )

        now = self.get_clock().now().to_msg()

        # Newest-state snapshot for the render thread; stamp identical to
        # the TF broadcast below so image-time TF lookups are exact.
        if self._camera_on:
            with self._snap_lock:
                self._snap = (now, self.data.qpos.copy(), self.data.qvel.copy())

        wall_ns = self.get_clock().now().nanoseconds
        if self._rtf_ref is None:
            self._rtf_ref = (wall_ns, self.data.time)
        elif wall_ns - self._rtf_ref[0] >= 1_000_000_000:
            dt_wall = (wall_ns - self._rtf_ref[0]) * 1e-9
            self._pub_rtf.publish(
                Float32(data=float((self.data.time - self._rtf_ref[1]) / dt_wall))
            )
            self._rtf_ref = (wall_ns, self.data.time)

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

    # -- camera render thread -------------------------------------------------
    def _image_msg(self, arr, stamp, frame_id, encoding):
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.height, msg.width = arr.shape[:2]
        msg.encoding = encoding
        msg.is_bigendian = False
        msg.step = arr.shape[1] * arr.itemsize * (
            arr.shape[2] if arr.ndim == 3 else 1
        )
        msg.data = arr.tobytes()
        return msg

    def _render_loop(self):
        """Owns every GL object: create, use and destroy on this one thread.

        A dedicated thread rather than a timer because GL contexts are
        thread-affine and a MultiThreadedExecutor may move a timer callback
        between workers; it also keeps a future mujoco.viewer.launch_passive
        (which owns its own context) safe. If the renderer cannot come up,
        log once and return -- physics keeps running.
        """
        try:
            self._cam.start()
        except Exception as e:
            self.get_logger().error(
                f"offscreen renderer unavailable ({e}) -- camera off, "
                "physics unaffected"
            )
            return
        depth_period = 1.0 / self.get_parameter("camera_depth_hz").value
        color_hz = self.get_parameter("camera_color_hz").value
        color_period = 1.0 / color_hz if color_hz > 0 else None
        try:
            next_depth = time.monotonic()
            next_color = next_depth if color_period is not None else None
            while not self._render_stop.is_set():
                due = min(t for t in (next_depth, next_color) if t is not None)
                wait = due - time.monotonic()
                if wait > 0:
                    self._render_stop.wait(wait)
                    continue
                with self._snap_lock:
                    snap = self._snap
                if snap is None:  # physics has not ticked yet
                    self._render_stop.wait(0.05)
                    continue
                stamp, qpos, qvel = snap
                now_m = time.monotonic()
                if now_m >= next_depth:
                    depth = self._cam.render_depth(qpos, qvel)
                    self._pub_depth.publish(self._image_msg(
                        depth, stamp, camera_spec.DEPTH_FRAME_ID, "16UC1"
                    ))
                    info = camera_spec.camera_info_msg(
                        depth.shape[1], depth.shape[0], camera_spec.DEPTH_FRAME_ID
                    )
                    info.header.stamp = stamp
                    self._pub_depth_info.publish(info)
                    # If rendering fell behind, resync instead of bursting.
                    next_depth = max(next_depth + depth_period, now_m)
                if next_color is not None and now_m >= next_color:
                    rgb = np.ascontiguousarray(self._cam.render_color(qpos, qvel))
                    self._pub_color.publish(self._image_msg(
                        rgb, stamp, camera_spec.COLOR_FRAME_ID,
                        camera_spec.COLOR_ENCODING
                    ))
                    info = camera_spec.camera_info_msg(
                        rgb.shape[1], rgb.shape[0], camera_spec.COLOR_FRAME_ID
                    )
                    info.header.stamp = stamp
                    self._pub_color_info.publish(info)
                    next_color = max(next_color + color_period, now_m)
        finally:
            self._cam.close()

    def destroy_node(self):
        self._render_stop.set()
        if self._render_thread is not None:
            self._render_thread.join(timeout=2.0)
        super().destroy_node()


def main():
    rclpy.init()
    node = MujocoSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
