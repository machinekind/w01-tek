"""Adapter: APT-RL (SkillQuad) controller behind the WojtekPolicy interface.

A policy dir whose policy_meta.json says "kind": "apt" holds the APT export
(policy.npz = the A*_deploy.npz written by apt_np.py export). The whole step
(history, VAE prior pose, residual, torque decoder) lives in apt_np.AptNumpy;
this class only maps the node's inputs onto it and its outputs onto the
driver contract:

  targets   = AptNumpy qc (absolute rad, MuJoCo/actuator convention)
  last_tau_ff = tau_ff_true / S   (REPORTED N*m, what the MD80 takes),
                clipped to +-tau_ff_limit (reported N*m)

Extra inputs vs schema 2: the IMU specific force (acc, body frame) and the
torque the drives reported on the previous step (tau_rep, reported N*m).
No action filter, no carrot, 3-D command.
"""

import json
from pathlib import Path

import numpy as np

from wojtek_policy.apt_np import AptNumpy


class AptPolicy:
    needs_acc_tau = True
    uses_imu = True
    uses_carrot = False
    tau_ff_enabled = True

    def __init__(self, npz_path, meta_path=None, clamp_knee=False, robot=None,
                 tau_ff_limit=None):
        npz_path = Path(npz_path)
        meta_path = Path(meta_path) if meta_path else npz_path.with_name(
            "policy_meta.json")
        self.meta = m = json.loads(meta_path.read_text())
        if m.get("kind") != "apt":
            raise ValueError(f"{meta_path} is not an apt contract")
        self.ctl = AptNumpy(str(npz_path))
        self.S = float(self.ctl.S)
        self.joint_names = list(m["actuator_names"])
        self.ctrl_dt = float(self.ctl.m["ctrl_dt"])
        self.command_low = np.array(m["command_low"], np.float32)
        self.command_high = np.array(m["command_high"], np.float32)
        self.command_fill = np.zeros(0, np.float32)
        self.command_width = 3
        tff = m.get("tau_ff") or {}
        self.tau_ff_scale = float(tff.get("scale", 0.0))
        if tau_ff_limit is not None and float(tau_ff_limit) > 0:
            self.tau_ff_limit = float(tau_ff_limit)
        else:
            self.tau_ff_limit = float(tff.get("limit", self.tau_ff_scale))
        ks = m.get("knee_singularity")
        self.knee_singularity = None if ks is None else float(ks)
        if clamp_knee and self.knee_singularity is None:
            raise ValueError("clamp_knee: apt contract has no knee_singularity")
        self.clamp_knee = clamp_knee
        # First-order low-pass on the OUTPUT (qc and tau_ff together),
        # y += alpha * (x - y) per tick; 1.0 = off. Settable live via the
        # policy_node parameter output_lpf_alpha (0.72 = 10 Hz at 50 Hz).
        self.output_lpf_alpha = 1.0
        self.reset()

    def reset(self):
        self.ctl.reset()
        self.last_tau_ff = np.zeros(12, np.float32)
        self.last_tau_ff_true = np.zeros(12)
        self._y_q = self._y_ff = None
        self.last_obs = None

    def step(self, gyro, gravity_body, joint_pos, joint_vel, command,
             carrot=None, acc=None, tau_rep=None):
        if acc is None:
            raise ValueError("apt policy needs the IMU specific force (acc)")
        if tau_rep is None or self.ctl.t == 0:
            tau_rep = np.zeros(12)
        cmd = np.asarray(command, np.float64)[:3]
        qc, ff_true = self.ctl.step(
            np.asarray(gyro, np.float64), np.asarray(gravity_body, np.float64),
            np.asarray(acc, np.float64), np.asarray(joint_pos, np.float64),
            np.asarray(joint_vel, np.float64), np.asarray(tau_rep, np.float64),
            cmd,
        )
        a = float(self.output_lpf_alpha)
        if self._y_q is None or a >= 1.0:
            self._y_q, self._y_ff = np.array(qc, float), np.array(ff_true, float)
        else:
            self._y_q += a * (qc - self._y_q)
            self._y_ff += a * (ff_true - self._y_ff)
        qc, ff_true = self._y_q, self._y_ff
        self.last_tau_ff_true = ff_true
        self.last_tau_ff = np.clip(
            ff_true / self.S, -self.tau_ff_limit, self.tau_ff_limit
        ).astype(np.float32)
        targets = np.asarray(qc, np.float32)
        if self.clamp_knee:
            targets[2::3] = np.minimum(targets[2::3], self.knee_singularity)
        return targets


def is_apt_meta(meta_path) -> bool:
    try:
        return json.loads(Path(meta_path).read_text()).get("kind") == "apt"
    except Exception:
        return False
