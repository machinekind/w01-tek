"""Adapter between the policy topics and the real robot's ros2_control stack.

The MD80 hardware interface reports joint states relative to the pose the
controllers were activated in, and (since the write() fix in our fork) sends
IMPEDANCE position commands in that same activation-relative frame. The
policy works in absolute URDF angles, so this node shifts by an offset in
both directions. The offset starts as "the robot was in the boot_pose at
activation" and can be corrected at runtime with /wojtek/zero.

The boot_pose parameter names the reference pose (the robot's position 0):

  boot_pose="home" (default): the home standing pose (wojtek_policy.poses
    HOME_CTRL, where /wojtek/stand_up ends).
  boot_pose="folded": robot lying flat on its base, hips straight and knees
    folded against the mechanical stop (folded_knee_rad, policy convention --
    see wojtek_policy.poses). Reproducible by hand without holding the
    robot -- call /wojtek/stand_up afterwards to ramp slowly into the home
    standing pose, then /wojtek/arm.

  subscribes  /joint_states           (boot-relative, from joint_state_broadcaster)
              /wojtek/joint_targets      (absolute URDF, from policy_node)
  publishes   /wojtek/joint_states_abs   (absolute URDF -> policy_node + RViz rsp;
              the 12 actuated joints, plus a separate passive-joints-only
              message with the four-bar fourth/fifth joints computed from the
              knee angles -- robot_state_publisher merges by name, policy_node
              skips the partial message)
              /forward_position_controller/commands (Float64MultiArray,
              boot-relative positions, actuator order -> MD80 IMPEDANCE mode)
              /forward_effort_controller/commands (Float64MultiArray, N*m,
              actuator order -> MD80 impedance-mode feed-forward torque.
              Published only when the incoming target carries an effort
              field (tau_ff policies); torque is frame-offset-free, so no
              boot shift applies. The launch spawns that controller only
              for tau_ff policies -- without it the publication is inert.)
  services    /wojtek/arm      (std_srvs/SetBool)  -- gate actually sending commands
              /wojtek/stand_up (std_srvs/Trigger)  -- slow ramp current -> home
              /wojtek/lie_down (std_srvs/Trigger)  -- slow ramp current -> folded
              /wojtek/zero     (std_srvs/Trigger)  -- re-zero: declare that the
              robot is physically in the boot_pose RIGHT NOW and recompute
              the offset from the current measurement. Lets you power and
              activate the motors in any pose: launch, physically pose the
              robot to match what RViz shows (the boot_pose), then call this.

Safety:
  * starts DISARMED: consumes and logs targets but publishes no commands.
  * arming is refused while a ramp is running, and if any joint is more than
    max_arm_jump_rad away from the home pose the policy expects.
  * stand_up/lie_down are refused while armed, so the policy and the ramp
    can never fight over the command topic.
  * if targets go stale the last command is simply not refreshed (MD80
    impedance holds the last target); an explicit disarm stops updates.

NOTE: md80_hardware_interface's write() used to send IMPEDANCE commands in
the drive's raw frame while read() reported activation-relative states; our
fork now shifts commands by the same activation offset, so both directions
share one frame and /wojtek/zero is exact. Still verify the first powered test
with dry_run:=true + /wojtek/arm false.
"""

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import SetBool, Trigger

from wojtek_policy.joint_map import JointMap

from wojtek_policy import poses


class RealIoNode(Node):
    def __init__(self):
        super().__init__("wojtek_real_io")
        share = get_package_share_directory("wojtek_policy")
        self.declare_parameter("joint_map_yaml", f"{share}/config/joint_map.yaml")
        self.declare_parameter("max_arm_jump_rad", 0.15)
        self.declare_parameter("dry_run", False)
        self.declare_parameter("boot_pose", "home")
        self.declare_parameter("folded_knee_rad", poses.FOLDED_KNEE_RAD)
        self.declare_parameter("ramp_duration_s", 4.0)
        self.declare_parameter("ramp_rate_hz", 50.0)

        # Joint order and the home pose are robot constants (poses.py), not
        # properties of whichever policy is loaded -- arming must work the
        # same with no policy at all.
        self.jmap = JointMap(self.get_parameter("joint_map_yaml").value)
        self.joint_names = list(poses.ACTUATOR_NAMES)

        # Named poses in URDF convention.
        folded = poses.folded_ctrl(
            self.joint_names, self.get_parameter("folded_knee_rad").value
        )
        self.home_urdf = self.jmap.to_urdf(self.joint_names, poses.HOME_CTRL)
        self.folded_urdf = self.jmap.to_urdf(self.joint_names, folded)
        boot_pose = self.get_parameter("boot_pose").value
        if boot_pose not in ("folded", "home"):
            raise ValueError(f"boot_pose must be 'folded' or 'home', got {boot_pose!r}")
        self.boot_urdf = self.folded_urdf if boot_pose == "folded" else self.home_urdf
        # Measured (activation-relative) -> absolute URDF. Starts assuming the
        # robot was in the boot pose at activation; /wojtek/zero recomputes it.
        self._offset_urdf = self.boot_urdf.copy()

        # Passive four-bar joints for RViz: (name, poly coeffs, index of that
        # leg's knee in joint_names).
        self._passive = [
            (name, coeffs, self.joint_names.index(
                name.replace("fourth_joint", "third_joint").replace(
                    "fifth_joint", "third_joint")))
            for name, coeffs in poses.PASSIVE_FROM_KNEE.items()
        ]

        self._armed = False
        self._q_rel = None
        self._ramp_timer = None

        self.create_subscription(JointState, "joint_states", self._on_joints, 10)
        self.create_subscription(
            JointState, "wojtek/joint_targets", self._on_targets, 10
        )
        self._pub_abs = self.create_publisher(JointState, "wojtek/joint_states_abs", 10)
        self._pub_cmd = self.create_publisher(
            Float64MultiArray, "forward_position_controller/commands", 10
        )
        self._pub_effort = self.create_publisher(
            Float64MultiArray, "forward_effort_controller/commands", 10
        )
        self.create_service(SetBool, "wojtek/arm", self._srv_arm)
        self.create_service(
            Trigger, "wojtek/stand_up", lambda req, res: self._srv_ramp(
                res, self.home_urdf, "stand_up"
            )
        )
        self.create_service(
            Trigger, "wojtek/lie_down", lambda req, res: self._srv_ramp(
                res, self.folded_urdf, "lie_down"
            )
        )
        self.create_service(Trigger, "wojtek/zero", self._srv_zero)
        self.get_logger().info(
            f"real io up, DISARMED, boot_pose={boot_pose}. Assuming the robot "
            f"was in the {boot_pose} pose at activation; if not, pose it there "
            "now and call: ros2 service call /wojtek/zero std_srvs/srv/Trigger. "
            "Then /wojtek/stand_up if lying, and arm with: ros2 service call "
            "/wojtek/arm std_srvs/srv/SetBool '{data: true}'"
        )

    def _on_joints(self, msg):
        idx = {n: i for i, n in enumerate(msg.name)}
        try:
            order = [idx[n] for n in self.joint_names]
        except KeyError:
            return
        self._q_rel = np.array([msg.position[i] for i in order])

        q_abs = self._q_rel + self._offset_urdf
        out = JointState()
        out.header = msg.header
        out.name = self.joint_names
        out.position = q_abs.tolist()
        if len(msg.velocity) == len(msg.name):
            out.velocity = [msg.velocity[i] for i in order]
        self._pub_abs.publish(out)

        # Passive four-bar joints follow from the knee angles; without them
        # robot_state_publisher has no TF for the lower-leg links and RViz
        # draws the legs broken.
        passive = JointState()
        passive.header = msg.header
        passive.name = [name for name, _, _ in self._passive]
        passive.position = [
            float(np.polyval(coeffs, q_abs[knee_idx]))
            for _, coeffs, knee_idx in self._passive
        ]
        self._pub_abs.publish(passive)

    def _on_targets(self, msg):
        idx = {n: i for i, n in enumerate(msg.name)}
        try:
            target_abs = np.array([msg.position[idx[n]] for n in self.joint_names])
        except KeyError as e:
            self.get_logger().warning(f"target missing joint {e}")
            return
        effort = None
        if len(msg.effort) == len(msg.name):
            # tau_ff policies: feed-forward torque riding the same message.
            effort = np.array([msg.effort[idx[n]] for n in self.joint_names])
        if not self._armed:
            return
        self._send_cmd(target_abs - self._offset_urdf, effort)  # boot-relative

    def _send_cmd(self, cmd_rel, effort=None):
        # Last line of defense: a non-finite target must never reach the
        # drives (tau = kp*(NaN - q) ...). Upstream guards exist in
        # policy_node, but this topic is open to anyone. A bad effort drops
        # the WHOLE command -- position without its feed-forward would
        # sag-jerk on a tau_ff policy.
        finite = np.all(np.isfinite(cmd_rel)) and (
            effort is None or np.all(np.isfinite(effort))
        )
        if not finite:
            self.get_logger().error(
                "non-finite motor command -- dropping", throttle_duration_sec=1.0
            )
            return
        if self.get_parameter("dry_run").value:
            eff = (
                f" effort {np.array2string(effort, precision=2)}"
                if effort is not None else ""
            )
            self.get_logger().info(
                f"dry_run cmd: {np.array2string(cmd_rel, precision=3)}{eff}",
                throttle_duration_sec=1.0,
            )
            return
        out = Float64MultiArray()
        out.data = cmd_rel.tolist()
        self._pub_cmd.publish(out)
        if effort is not None:
            tau = Float64MultiArray()
            tau.data = effort.tolist()
            self._pub_effort.publish(tau)

    def _srv_ramp(self, res, target_urdf, name):
        if self._armed:
            res.success, res.message = False, "armed -- disarm before ramping"
            return res
        if self._ramp_timer is not None:
            res.success, res.message = False, "a ramp is already running"
            return res
        if self._q_rel is None:
            res.success, res.message = False, "no joint states yet"
            return res
        self._ramp_from = self._q_rel + self._offset_urdf
        self._ramp_to = target_urdf
        self._ramp_name = name
        self._ramp_t0 = self.get_clock().now()
        period = 1.0 / self.get_parameter("ramp_rate_hz").value
        self._ramp_timer = self.create_timer(period, self._ramp_tick)
        duration = self.get_parameter("ramp_duration_s").value
        self.get_logger().info(f"{name}: ramping over {duration:.1f} s")
        res.success, res.message = True, f"{name} ramp started"
        return res

    def _ramp_tick(self):
        duration = self.get_parameter("ramp_duration_s").value
        t = (self.get_clock().now() - self._ramp_t0).nanoseconds * 1e-9
        s = min(t / duration, 1.0)
        alpha = 0.5 - 0.5 * np.cos(np.pi * s)  # smooth start and stop
        q = (1.0 - alpha) * self._ramp_from + alpha * self._ramp_to
        self._send_cmd(q - self._offset_urdf)
        if s >= 1.0:
            self.destroy_timer(self._ramp_timer)
            self._ramp_timer = None
            self.get_logger().info(f"{self._ramp_name}: ramp complete")

    def _srv_zero(self, req, res):
        if self._armed:
            res.success, res.message = False, "armed -- disarm before zeroing"
            return res
        if self._ramp_timer is not None:
            res.success, res.message = False, "ramp running -- refusing to zero"
            return res
        if self._q_rel is None:
            res.success, res.message = False, "no joint states yet"
            return res
        shift = np.abs(self.boot_urdf - self._q_rel - self._offset_urdf)
        self._offset_urdf = self.boot_urdf - self._q_rel
        res.success = True
        res.message = (
            f"zeroed: current pose is now the boot pose "
            f"(largest correction {np.max(shift):.3f} rad)"
        )
        self.get_logger().info(res.message)
        return res

    def _srv_arm(self, req, res):
        if not req.data:
            self._armed = False
            res.success, res.message = True, "disarmed"
            return res
        if self._ramp_timer is not None:
            res.success, res.message = False, "ramp running -- refusing to arm"
            return res
        if self._q_rel is None:
            res.success, res.message = False, "no joint states yet -- refusing to arm"
            return res
        # The first policy commands will be near the home pose; guard against
        # a jump from the measured pose.
        jump = np.abs(self._q_rel + self._offset_urdf - self.home_urdf)
        limit = self.get_parameter("max_arm_jump_rad").value
        if np.max(jump) > limit:
            res.success = False
            res.message = (
                f"joint displacement from home pose up to {np.max(jump):.3f} rad "
                f"> {limit} -- robot is not standing in the home pose, refusing "
                "to arm (run /wojtek/stand_up first)"
            )
            return res
        self._armed = True
        res.success, res.message = True, "armed"
        self.get_logger().warn("ARMED -- policy commands now reach the motors")
        return res


def main():
    rclpy.init()
    node = RealIoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
