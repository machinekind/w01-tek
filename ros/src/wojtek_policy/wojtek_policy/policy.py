"""Numpy runtime for exported wojtek locomotion policies.

Loads the .npz + policy_meta.json written by training/wojtek_rl/
export_policy.py and reproduces the training-time observation/action
pipeline. The observation vector is assembled from meta["obs_layout"], so
one runtime serves policies with different sensor suites -- wojtek_v3
(gyro + gravity + gait clock, 3-D command) and the springy family
(proprioception only, 4-D command with a fixed height) both load here:

  obs = concat(obs_layout components)
  motor_targets = clip(anchor + tanh_mlp(obs) * action_scale,
                       target_low, target_high)

anchor is the stance for the trained standing height: home_ctrl when the
command is 3-D, home_ctrl shifted by the measured height table below when
the command carries a height. target_low/high are the bounds training
clipped motor targets to -- the model ctrlrange narrowed by the clamps in
meta["deploy"] when present.

Joint order everywhere is the actuator order from policy_meta.json:
rear_left, rear_right, front_right, front_left x (first, second, third).
No ROS imports here so it is unit-testable without a ROS install.
"""

import json
from pathlib import Path

import numpy as np

# Stance ctrl vs commanded standing height, measured table copied from
# training/wojtek_rl/env.py (HEIGHT_TABLE / DSECOND_TABLE). dsecond shifts
# the hip (x1) and knee (x2) actuators off the home pose.
HEIGHT_TABLE = (0.084, 0.094, 0.106, 0.121, 0.139, 0.160, 0.182)
DSECOND_TABLE = (-0.45, -0.30, -0.15, 0.0, 0.15, 0.30, 0.45)


def height_anchor(home_ctrl, height, ctrl_low, ctrl_high):
    """Ctrl anchor for a commanded standing height (env.py _height_ctrl).

    Clipped to the model ctrlrange only; training clips the anchor+action
    sum to the (narrower) target bounds afterwards, so the anchor itself
    may sit above target_high and the order of the two clips matters.
    """
    dsecond = np.interp(height, HEIGHT_TABLE, DSECOND_TABLE)
    offset = np.tile(np.array([0.0, 1.0, 2.0]), 4) * dsecond
    return np.clip(home_ctrl + offset, ctrl_low, ctrl_high).astype(np.float32)


class WojtekPolicy:
    def __init__(self, npz_path, meta_path=None, clamp_knee=False):
        npz_path = Path(npz_path)
        meta_path = Path(meta_path) if meta_path else npz_path.with_name(
            "policy_meta.json"
        )
        self.meta = json.loads(meta_path.read_text())
        data = np.load(npz_path)
        self._norm_mean = data["norm_mean"]
        self._norm_std = data["norm_std"]
        self._layers = []
        i = 0
        while f"hidden_{i}_kernel" in data:
            self._layers.append((data[f"hidden_{i}_kernel"], data[f"hidden_{i}_bias"]))
            i += 1

        m = self.meta
        self.joint_names = list(m["actuator_names"])
        self.home_ctrl = np.array(m["home_ctrl"], np.float32)
        self.ctrl_low = np.array(m["ctrl_low"], np.float32)
        self.ctrl_high = np.array(m["ctrl_high"], np.float32)
        self.ctrl_dt = float(m["ctrl_dt"])
        self.knee_singularity = float(m["knee_singularity"])
        self.clamp_knee = clamp_knee
        self.gait_freq_hz = float(m["gait_freq_hz"])
        self._trot_phase = np.array(m["trot_phase"], np.float32)

        self.layout = [
            (name, int(width))
            for name, width in (e.split(":") for e in m["obs_layout"])
        ]
        names = [n for n, _ in self.layout]
        self.uses_imu = "gyro" in names or "gravity" in names
        self._uses_phase = "phase" in names or "phase_cos_sin" in names
        self._cmd_width = dict(self.layout).get("command", 3)

        # Scalar, or per-joint-type [abduction, hip, knee] tiled over the legs.
        scale = np.asarray(m["action_scale"], np.float32)
        self.action_scale = np.tile(scale, 4) if scale.size == 3 else scale

        deploy = m.get("deploy", {})
        self.command_height = deploy.get("command_height", 0.0)
        self.command_low = np.array(
            deploy.get("command_low", [-0.6, -0.4, -0.7]), np.float32
        )
        self.command_high = np.array(
            deploy.get("command_high", [0.6, 0.4, 0.7]), np.float32
        )
        low, high = self.ctrl_low.copy(), self.ctrl_high.copy()
        limit = deploy.get("abduction_ctrl_limit", 0.0)
        if limit:
            low[0::3] = np.maximum(low[0::3], -limit)
            high[0::3] = np.minimum(high[0::3], limit)
        ktm = deploy.get("knee_target_max", 0.0)
        if ktm:
            high[2::3] = np.minimum(high[2::3], ktm)
        self.target_low, self.target_high = low, high

        if self._cmd_width >= 4:
            self.anchor_ctrl = height_anchor(
                self.home_ctrl, self.command_height, self.ctrl_low, self.ctrl_high
            )
        else:
            self.anchor_ctrl = self.home_ctrl

        self.reset()

    def reset(self):
        self.last_action = np.zeros(12, np.float32)
        self.last_obs = None
        self.phase = self._trot_phase.copy()

    def _mlp(self, obs):
        x = (obs - self._norm_mean) / self._norm_std
        for kernel, bias in self._layers[:-1]:
            x = x @ kernel + bias
            x = x * np.exp(-np.logaddexp(0.0, -x))  # SiLU
        kernel, bias = self._layers[-1]
        x = x @ kernel + bias
        return np.tanh(x[: len(self.home_ctrl)])

    def _assemble_obs(self, gyro, gravity_body, joint_pos, joint_vel, command):
        parts = []
        for name, width in self.layout:
            if name == "gyro":
                parts.append(np.asarray(gyro, np.float32))
            elif name == "gravity":
                parts.append(np.asarray(gravity_body, np.float32))
            elif name in ("joint_pos", "qpos-home"):
                parts.append(np.asarray(joint_pos, np.float32) - self.home_ctrl)
            elif name in ("joint_vel", "qvel"):
                parts.append(np.asarray(joint_vel, np.float32))
            elif name == "last_act":
                parts.append(self.last_action)
            elif name == "command":
                cmd = np.asarray(command, np.float32)
                if width == 4 and cmd.size == 3:
                    cmd = np.append(cmd, np.float32(self.command_height))
                parts.append(cmd)
            elif name in ("phase", "phase_cos_sin"):
                parts.append(np.cos(self.phase))
                parts.append(np.sin(self.phase))
            else:
                raise KeyError(f"unknown obs component {name!r} in obs_layout")
        obs = np.concatenate(parts).astype(np.float32)
        if obs.size != self.meta["obs_size"]:
            raise ValueError(
                f"assembled obs has {obs.size} elements, meta says "
                f"{self.meta['obs_size']}"
            )
        return obs

    def step(self, gyro, gravity_body, joint_pos, joint_vel, command):
        """One 50 Hz control step. Returns motor position targets (12,).

        All arrays in actuator order; gyro/gravity in the base (IMU) frame
        (ignored by policies whose obs_layout omits them), command =
        [vx, vy, wz] in the base frame.
        """
        obs = self._assemble_obs(gyro, gravity_body, joint_pos, joint_vel, command)
        self.last_obs = obs
        action = self._mlp(obs)
        self.last_action = action.astype(np.float32)
        if self._uses_phase:
            self.phase += 2.0 * np.pi * self.ctrl_dt * self.gait_freq_hz
            self.phase = np.mod(self.phase + np.pi, 2.0 * np.pi) - np.pi

        targets = self.anchor_ctrl + action * self.action_scale
        targets = np.clip(targets, self.target_low, self.target_high)
        if self.clamp_knee:
            # Real-robot safety: never command the far branch of the four-bar
            # (snap-through can break the linkage). Redundant when the meta
            # carries a knee_target_max below the singularity.
            targets[2::3] = np.minimum(targets[2::3], self.knee_singularity)
        return targets


def gravity_from_quat(qw, qx, qy, qz):
    """World -z rotated into the body frame (what the training env observed)."""
    # R^T @ [0, 0, -1] for unit quaternion (w, x, y, z)
    return np.array(
        [
            -2.0 * (qx * qz - qw * qy),
            -2.0 * (qy * qz + qw * qx),
            -(1.0 - 2.0 * (qx * qx + qy * qy)),
        ],
        np.float32,
    )
