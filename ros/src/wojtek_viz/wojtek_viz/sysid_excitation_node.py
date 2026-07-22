"""Excitation trajectory publisher for actuator-parameter identification.

The rig is the AIR rig: robot held off the ground, belly resting on a
raised support so the legs hang and swing freely. Publishes
/wojtek/joint_targets
(JointState, URDF convention) around the pose the robot is in when the
node starts, running a fixed program designed so each physics parameter
has a phase that pins it:

  settle     hold the anchor pose (lets a recording establish a baseline)
  ramps      per-joint-type slow triangle waves: constant low velocity, so
             Coulomb friction shows up as a clean constant torque offset —
             the classic frictionloss/stiction measurement
  chirps     per-joint-type frequency sweeps in paired (amplitude, top
             frequency) passes — big-and-slow, medium, small-and-fast: the
             amplitude spread separates Coulomb friction from viscous
             kd/damping, the fast pass runs through the closed-loop servo
             resonance so armature becomes visible, and pairing keeps the
             peak joint velocity roughly constant across passes
  steps      per-joint-type +/- position steps with holds; nothing pins
             command latency and rise time better
  multisine  small simultaneous multi-frequency motion on all joints
             (cross-validation data, exercises inter-joint coupling)
  hold       anchor again, forever (stop the node/recording here)

Excitation is computed in MuJoCo convention (so legs move symmetrically)
and converted through the joint map, exactly like mujoco_sim_node. Any
phase can be skipped by setting its *_sec parameter to 0.

Works against the MuJoCo sim node for pipeline-validation bags and
against the real robot; for a first real run, start with a reduced
amplitude_scale and be ready on the kill switch.

Record with:
  ros2 bag record /wojtek/joint_targets /joint_states /imu/data

then fit with `./training/run.sh sysid --bag <dir>`. A level
belly-on-support rig is the model's default upright orientation, so no
IMU and no --base-quat is needed; pass --base-quat only if the rig tilts
the robot. See training/docs/sysid.md.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from ament_index_python.packages import get_package_share_directory
from wojtek_policy.joint_map import JointMap

LEGS = ("rear_left", "rear_right", "front_right", "front_left")
TYPES = ("first", "second", "third")


class SysidExcitationNode(Node):
    def __init__(self):
        super().__init__("wojtek_sysid_excitation")
        policy_share = get_package_share_directory("wojtek_policy")
        self.declare_parameter("joint_map_yaml", f"{policy_share}/config/joint_map.yaml")
        self.declare_parameter("rate_hz", 100.0)
        self.declare_parameter("amplitude_scale", 1.0)
        self.declare_parameter("settle_sec", 3.0)
        # Slow triangles: |velocity| = 4*amp/period ~ 0.05-0.075 rad/s,
        # quasi-static so friction dominates the torque balance.
        self.declare_parameter("ramp_sec", 8.0)  # one triangle per type
        self.declare_parameter("ramp_amp_rad", [0.10, 0.12, 0.15])
        self.declare_parameter("chirp_sec", 8.0)
        self.declare_parameter("chirp_f0_hz", 0.3)
        # Per joint type (first/abduction, second, third/knee).
        self.declare_parameter("chirp_amp_rad", [0.08, 0.10, 0.12])
        # Paired passes (amp scale, top frequency): the amplitude spread
        # separates Coulomb from viscous friction, the fast small pass
        # crosses the servo resonance (~2-4 Hz at kp=20) for armature, and
        # the pairing keeps peak joint velocity roughly constant.
        self.declare_parameter("chirp_amp_scales", [2.0, 1.0, 0.5])
        self.declare_parameter("chirp_f1_hz", [1.5, 4.0, 8.0])
        self.declare_parameter("step_amp_rad", [0.06, 0.08, 0.10])
        self.declare_parameter("step_hold_sec", 0.8)
        self.declare_parameter("multisine_sec", 12.0)
        self.declare_parameter("multisine_amp_rad", 0.06)
        # Cosine fade at both ends of chirp/multisine phases: they end
        # mid-cycle, and without the envelope the snap back to the anchor
        # is a ~10 rad/s command spike at every phase boundary.
        self.declare_parameter("fade_sec", 0.5)

        self.names = [f"{leg}_{t}_joint" for leg in LEGS for t in TYPES]
        self.jmap = JointMap(self.get_parameter("joint_map_yaml").value)
        self._anchor_mjc = None  # set from the first full /joint_states
        self._t0 = None
        self._phase_logged = ""

        # Fixed multisine table: 3 incommensurate frequencies and phases
        # per joint, deterministic so reruns are comparable.
        rng = np.random.default_rng(0)
        self._ms_freq = rng.uniform(0.3, 5.0, size=(12, 3))
        self._ms_phase = rng.uniform(0.0, 2 * np.pi, size=(12, 3))

        self.create_subscription(JointState, "joint_states", self._on_state, 10)
        self._pub = self.create_publisher(JointState, "wojtek/joint_targets", 10)
        self.create_timer(1.0 / self.get_parameter("rate_hz").value, self._tick)
        self.get_logger().info("waiting for /joint_states to anchor the pose")

    def _on_state(self, msg):
        if self._anchor_mjc is not None:
            return
        idx = {n: i for i, n in enumerate(msg.name)}
        if any(n not in idx for n in self.names):
            return
        q_urdf = np.array([msg.position[idx[n]] for n in self.names])
        self._anchor_mjc = self.jmap.to_mjc(self.names, q_urdf)
        self._t0 = self.get_clock().now()
        self.get_logger().info("anchored at current pose, starting program")

    def _fade(self, t, duration):
        """Cosine envelope: 0 -> 1 over fade_sec, 1 -> 0 at the end."""
        fade = self.get_parameter("fade_sec").value
        if fade <= 0:
            return 1.0
        edge = min(t, duration - t)
        if edge >= fade:
            return 1.0
        return 0.5 * (1 - np.cos(np.pi * max(edge, 0.0) / fade))

    def _program(self, t):
        """(phase name, delta in MuJoCo convention) at program time t."""
        scale = self.get_parameter("amplitude_scale").value
        delta = np.zeros(12)

        t -= self.get_parameter("settle_sec").value
        if t < 0:
            return "settle", delta

        ramp_sec = self.get_parameter("ramp_sec").value
        ramp_amps = self.get_parameter("ramp_amp_rad").value
        for k, typ in enumerate(TYPES):
            if t < ramp_sec:
                # Triangle 0 -> +amp -> -amp -> 0 over ramp_sec: constant
                # |velocity| except at the three turnarounds.
                u = t / ramp_sec
                tri = 4 * u if u < 0.25 else (2 - 4 * u if u < 0.75 else 4 * u - 4)
                delta[np.arange(12) % 3 == k] = scale * ramp_amps[k] * tri
                return f"ramp_{typ}", delta
            t -= ramp_sec

        chirp_sec = self.get_parameter("chirp_sec").value
        amps = self.get_parameter("chirp_amp_rad").value
        f0 = self.get_parameter("chirp_f0_hz").value
        f1s = self.get_parameter("chirp_f1_hz").value
        for amp_scale, f1 in zip(self.get_parameter("chirp_amp_scales").value, f1s):
            for k, typ in enumerate(TYPES):
                if t < chirp_sec:
                    phi = 2 * np.pi * (
                        f0 * t + (f1 - f0) * t * t / (2 * chirp_sec)
                    )
                    delta[np.arange(12) % 3 == k] = (
                        scale * amp_scale * amps[k]
                        * self._fade(t, chirp_sec) * np.sin(phi)
                    )
                    return f"chirp_{typ}_x{amp_scale:g}_{f1:g}Hz", delta
                t -= chirp_sec

        hold_sec = self.get_parameter("step_hold_sec").value
        step_amps = self.get_parameter("step_amp_rad").value
        levels = (1.0, 0.0, -1.0, 0.0)
        for k, typ in enumerate(TYPES):
            block = hold_sec * len(levels)
            if t < block:
                level = levels[int(t // hold_sec)] if hold_sec > 0 else 0.0
                delta[np.arange(12) % 3 == k] = scale * step_amps[k] * level
                return f"steps_{typ}", delta
            t -= block

        ms_sec = self.get_parameter("multisine_sec").value
        if t < ms_sec:
            amp = scale * self.get_parameter("multisine_amp_rad").value
            waves = np.sin(2 * np.pi * self._ms_freq * t + self._ms_phase)
            delta[:] = amp * self._fade(t, ms_sec) * waves.mean(axis=1)
            return "multisine", delta

        return "hold", delta

    def _tick(self):
        if self._anchor_mjc is None:
            return
        t = (self.get_clock().now() - self._t0).nanoseconds * 1e-9
        phase, delta = self._program(t)
        if phase != self._phase_logged:
            self.get_logger().info(f"phase: {phase}")
            self._phase_logged = phase
            if phase == "hold":
                self.get_logger().info("program done, holding anchor (stop the bag)")

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.names
        msg.position = self.jmap.to_urdf(self.names, self._anchor_mjc + delta).tolist()
        self._pub.publish(msg)


def main():
    rclpy.init()
    node = SysidExcitationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
