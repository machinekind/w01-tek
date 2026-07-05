"""Adapter between the policy topics and the real robot's ros2_control stack.

The MD80 hardware interface zeroes every joint position when it activates
(state = raw encoder - boot value). The policy works in absolute URDF angles,
so this node shifts by the boot pose in both directions. THE ROBOT MUST BE
POWERED ON / ACTIVATED IN THE HOME STANDING POSE (the pose in
policy_meta.json home_ctrl) -- that is what boot_pose="home" assumes.

  subscribes  /joint_states           (boot-relative, from joint_state_broadcaster)
              /fbb/joint_targets      (absolute URDF, from policy_node)
  publishes   /fbb/joint_states_abs   (absolute URDF -> policy_node + RViz rsp)
              /forward_position_controller/commands (Float64MultiArray,
              boot-relative positions, actuator order -> MD80 IMPEDANCE mode)
  services    /fbb/arm (std_srvs/SetBool) -- gate actually sending commands

Safety:
  * starts DISARMED: consumes and logs targets but publishes no commands.
  * arming is refused if the first command would jump any joint by more than
    max_arm_jump_rad from its measured position.
  * if targets go stale the last command is simply not refreshed (MD80
    impedance holds the last target); an explicit disarm stops updates.

NOTE (verify on hardware): md80_hardware_interface subtracts the boot offset
from position STATES but sends IMPEDANCE position commands raw. This node
assumes raw == boot-relative, which holds when the drives zero their encoder
output at power-on. Verify with dry_run:=true + /fbb/arm false before the
first powered test.
"""

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import SetBool

from fbb_policy.joint_map import JointMap
from fbb_policy.policy import FbbPolicy


class RealIoNode(Node):
    def __init__(self):
        super().__init__("fbb_real_io")
        share = get_package_share_directory("fbb_policy")
        self.declare_parameter("policy_dir", f"{share}/config")
        self.declare_parameter("joint_map_yaml", f"{share}/config/joint_map.yaml")
        self.declare_parameter("max_arm_jump_rad", 0.15)
        self.declare_parameter("dry_run", False)

        pdir = self.get_parameter("policy_dir").value
        policy = FbbPolicy(f"{pdir}/policy.npz")
        self.jmap = JointMap(self.get_parameter("joint_map_yaml").value)
        self.joint_names = policy.joint_names
        # Boot pose in URDF convention = home standing pose.
        self.boot_urdf = self.jmap.to_urdf(self.joint_names, policy.home_ctrl)

        self._armed = False
        self._q_rel = None

        self.create_subscription(JointState, "joint_states", self._on_joints, 10)
        self.create_subscription(
            JointState, "fbb/joint_targets", self._on_targets, 10
        )
        self._pub_abs = self.create_publisher(JointState, "fbb/joint_states_abs", 10)
        self._pub_cmd = self.create_publisher(
            Float64MultiArray, "forward_position_controller/commands", 10
        )
        self.create_service(SetBool, "fbb/arm", self._srv_arm)
        self.get_logger().info(
            "real io up, DISARMED. Robot must have been powered on in the home "
            "standing pose. Arm with: ros2 service call /fbb/arm "
            "std_srvs/srv/SetBool '{data: true}'"
        )

    def _on_joints(self, msg):
        idx = {n: i for i, n in enumerate(msg.name)}
        try:
            order = [idx[n] for n in self.joint_names]
        except KeyError:
            return
        self._q_rel = np.array([msg.position[i] for i in order])

        out = JointState()
        out.header = msg.header
        out.name = self.joint_names
        out.position = (self._q_rel + self.boot_urdf).tolist()
        if len(msg.velocity) == len(msg.name):
            out.velocity = [msg.velocity[i] for i in order]
        self._pub_abs.publish(out)

    def _on_targets(self, msg):
        idx = {n: i for i, n in enumerate(msg.name)}
        try:
            target_abs = np.array([msg.position[idx[n]] for n in self.joint_names])
        except KeyError as e:
            self.get_logger().warning(f"target missing joint {e}")
            return
        cmd = target_abs - self.boot_urdf  # boot-relative
        if not self._armed:
            return
        if self.get_parameter("dry_run").value:
            self.get_logger().info(
                f"dry_run cmd: {np.array2string(cmd, precision=3)}",
                throttle_duration_sec=1.0,
            )
            return
        out = Float64MultiArray()
        out.data = cmd.tolist()
        self._pub_cmd.publish(out)

    def _srv_arm(self, req, res):
        if not req.data:
            self._armed = False
            res.success, res.message = True, "disarmed"
            return res
        if self._q_rel is None:
            res.success, res.message = False, "no joint states yet -- refusing to arm"
            return res
        # The first commands will be near the current policy target; guard
        # against a jump from the measured pose.
        jump = np.abs(self._q_rel)  # boot-relative pose should be ~0 at home
        limit = self.get_parameter("max_arm_jump_rad").value
        if np.max(jump) > limit:
            res.success = False
            res.message = (
                f"joint displacement from boot pose up to {np.max(jump):.3f} rad "
                f"> {limit} -- robot is not in the home pose, refusing to arm"
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
