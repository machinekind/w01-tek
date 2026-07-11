"""Numpy runtime for the exported wojtek locomotion policy.

Loads the .npz written by 4_wojtek_rl/wojtek_rl/export_policy.py and
reproduces the training-time observation/action pipeline:

  obs (53) = gyro(3) + gravity_body(3) + (qpos - home)(12) + qvel(12)
           + last_action(12) + command(3) + [cos(phase), sin(phase)](8)
  motor_targets = clip(home_ctrl + tanh_mlp(obs) * action_scale, ctrlrange)

Joint order everywhere is the actuator order from policy_meta.json:
rear_left, rear_right, front_right, front_left x (first, second, third).
No ROS imports here so it is unit-testable without a ROS install.
"""

import json
from pathlib import Path

import numpy as np


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
        self.action_scale = float(m["action_scale"])
        self.ctrl_dt = float(m["ctrl_dt"])
        self.knee_singularity = float(m["knee_singularity"])
        self.clamp_knee = clamp_knee
        self.gait_freq_hz = float(m["gait_freq_hz"])
        self._trot_phase = np.array(m["trot_phase"], np.float32)

        self.reset()

    def reset(self):
        self.last_action = np.zeros(12, np.float32)
        self.phase = self._trot_phase.copy()

    def _mlp(self, obs):
        x = (obs - self._norm_mean) / self._norm_std
        for kernel, bias in self._layers[:-1]:
            x = x @ kernel + bias
            x = x * np.exp(-np.logaddexp(0.0, -x))  # SiLU
        kernel, bias = self._layers[-1]
        x = x @ kernel + bias
        return np.tanh(x[: len(self.home_ctrl)])

    def step(self, gyro, gravity_body, joint_pos, joint_vel, command):
        """One 50 Hz control step. Returns motor position targets (12,).

        All arrays in actuator order; gyro/gravity in the base (IMU) frame,
        command = [vx, vy, wz] in the base frame.
        """
        obs = np.concatenate(
            [
                np.asarray(gyro, np.float32),
                np.asarray(gravity_body, np.float32),
                np.asarray(joint_pos, np.float32) - self.home_ctrl,
                np.asarray(joint_vel, np.float32),
                self.last_action,
                np.asarray(command, np.float32),
                np.cos(self.phase),
                np.sin(self.phase),
            ]
        ).astype(np.float32)
        action = self._mlp(obs)
        self.last_action = action.astype(np.float32)
        self.phase += 2.0 * np.pi * self.ctrl_dt * self.gait_freq_hz
        self.phase = np.mod(self.phase + np.pi, 2.0 * np.pi) - np.pi

        targets = self.home_ctrl + action * self.action_scale
        targets = np.clip(targets, self.ctrl_low, self.ctrl_high)
        if self.clamp_knee:
            # Real-robot safety: never command the far branch of the four-bar
            # (snap-through can break the linkage). Off in sim to match the
            # training env, which clipped only to ctrlrange.
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
