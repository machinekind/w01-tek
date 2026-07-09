"""Numpy runtime for the exported fbb locomotion policy.

Loads the .npz written by training/wojtek_rl/export_policy.py and
reproduces the training-time observation/action pipeline. The observation
is assembled from policy_meta.json's obs_layout, so one runtime serves
every exported policy generation:

  fbb_v3 (53):      gyro(3) + gravity(3) + (qpos-home)(12) + qvel(12)
                    + last_action(12) + command(3) + cos/sin phase(8)
  fbb_loco_v8 (48): (qpos-home)(12) + qvel(12) + last_action(12)
                    + command(4: vx,vy,wz,height) + cos/sin phase(8)

Action pipeline (matches wojtek_rl.env.WojtekJoystick.step):

  anchor = height_ctrl(command[3]) if the command carries a height, else home
  motor_targets = clip(anchor + tanh_mlp(obs) * action_scale, ctrlrange)

Gait clock: v3 advanced a fixed-frequency trot clock. v8 uses a master
clock whose frequency scales with command speed (frozen when standing)
plus walk<->trot phase offsets blended across trot_band -- enabled when
the meta carries gait_freq_band (see export_policy.py / README).

Joint order everywhere is the actuator order from policy_meta.json:
rear_left, rear_right, front_right, front_left x (first, second, third).
No ROS imports here so it is unit-testable without a ROS install.
"""

import json
from pathlib import Path

import numpy as np

# wojtek_rl.env constants for policies whose meta predates their export.
TROT_PHASE = (0.0, np.pi, 0.0, np.pi)
WALK_PHASE = (0.0, np.pi, 1.5 * np.pi, 0.5 * np.pi)
HEIGHT_TABLE = (0.084, 0.094, 0.106, 0.121, 0.139, 0.160, 0.182)
DSECOND_TABLE = (-0.45, -0.30, -0.15, 0.0, 0.15, 0.30, 0.45)
CMD_WZ_WEIGHT = 0.3  # _cmd_speed: |v_xy| + 0.3 * |wz|
STAND_SPEED = 0.05  # below this command speed the clock freezes


class FbbPolicy:
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
        # EMA low-pass on the action driving the motors, mirroring
        # wojtek_rl.env action_filter (kills hardware tremble the rigid-body
        # sim doesn't show). Exactly as in training, the RAW action feeds
        # last_act in the observation; only the motor targets are smoothed.
        # 0 = off (matches v8 training); try 0.2-0.5 on a vibrating robot.
        self.action_ema = 0.0

        # Observation layout: ["gyro:3", ..., "command:4", "phase:8"].
        layout = m.get(
            "obs_layout",
            ["gyro:3", "gravity:3", "joint_pos:12", "joint_vel:12",
             "last_act:12", "command:3", "phase:8"],
        )
        self._obs_names = [e.split(":")[0] for e in layout]
        self.cmd_dim = next(
            int(e.split(":")[1]) for e in layout if e.startswith("command")
        )
        self.needs_imu = "gyro" in self._obs_names or "gravity" in self._obs_names

        # Gait clock. gait_freq_band => v8-style speed-scaled master clock
        # with walk<->trot offset blending; otherwise the fixed v3 clock.
        self._trot_phase = np.array(m["trot_phase"], np.float32)
        band = m.get("gait_freq_band")
        self._freq_band = None if band is None else (float(band[0]), float(band[1]))
        # Fixed-clock (v3-era) policies trot at every speed; only the
        # speed-scaled clock generation walks below the trot band.
        default_walk = WALK_PHASE if band is not None else m["trot_phase"]
        self._walk_phase = np.array(m.get("walk_phase", default_walk), np.float32)
        self.gait_freq_hz = float(m["gait_freq_hz"])
        self._trot_band = tuple(m.get("trot_band", (0.35, 0.55)))
        self._cmd_vmax = float(m.get("cmd_vmax", 1.2))

        # Commanded-height stance anchor (4-D commands only).
        self._height_table = np.array(
            m.get("height_table", HEIGHT_TABLE), np.float32
        )
        self._dsecond_table = np.array(
            m.get("dsecond_table", DSECOND_TABLE), np.float32
        )
        # Height whose stance anchor is exactly home_ctrl (dsecond = 0);
        # used to pad 3-D velocity commands for 4-D-command policies.
        self.default_height = float(
            np.interp(0.0, self._dsecond_table, self._height_table)
        )

        self.reset()

    def reset(self):
        self.last_action = np.zeros(12, np.float32)
        self._filtered_act = np.zeros(12, np.float32)
        # Master clock; per-leg phases add blended offsets.
        self.phase = np.float32(0.0)

    # -- gait clock (wojtek_rl.env: _cmd_speed/_gait_frac/_leg_phases/_phase_dt) --
    def _cmd_speed(self, command):
        return float(
            np.linalg.norm(command[:2]) + CMD_WZ_WEIGHT * abs(command[2])
        )

    def _gait_frac(self, command):
        lo, hi = self._trot_band
        return float(np.clip((self._cmd_speed(command) - lo) / (hi - lo), 0.0, 1.0))

    def _leg_phases(self, command):
        f = self._gait_frac(command)
        offsets = (1.0 - f) * self._walk_phase + f * self._trot_phase
        phase = self.phase + offsets
        return np.mod(phase + np.pi, 2.0 * np.pi) - np.pi

    def _phase_dt(self, command):
        if self._freq_band is None:
            return 2.0 * np.pi * self.ctrl_dt * self.gait_freq_hz
        speed = self._cmd_speed(command)
        if speed <= STAND_SPEED:
            return 0.0
        frac = np.clip(speed / self._cmd_vmax, 0.0, 1.0)
        f0, f1 = self._freq_band
        return 2.0 * np.pi * self.ctrl_dt * (f0 + (f1 - f0) * frac)

    def _height_ctrl(self, height):
        """Ctrl anchor for a commanded standing height (measured table)."""
        dsecond = np.interp(height, self._height_table, self._dsecond_table)
        offset = np.tile(np.array([0.0, 1.0, 2.0], np.float32), 4) * dsecond
        return np.clip(self.home_ctrl + offset, self.ctrl_low, self.ctrl_high)

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

        All arrays in actuator order; gyro/gravity in the base (IMU) frame
        (may be None for policies that don't observe them), command =
        [vx, vy, wz] or [vx, vy, wz, height] in the base frame.
        """
        command = np.asarray(command, np.float32)
        if self.cmd_dim >= 4 and command.shape[0] == 3:
            command = np.concatenate([command, [self.default_height]])
        leg_phase = self._leg_phases(command)
        catalog = {
            "gyro": None if gyro is None else np.asarray(gyro, np.float32),
            "gravity": None
            if gravity_body is None
            else np.asarray(gravity_body, np.float32),
            "joint_pos": np.asarray(joint_pos, np.float32) - self.home_ctrl,
            "joint_vel": np.asarray(joint_vel, np.float32),
            "last_act": self.last_action,
            "command": command,
            "phase": np.concatenate([np.cos(leg_phase), np.sin(leg_phase)]),
        }
        obs = np.concatenate([catalog[n] for n in self._obs_names]).astype(
            np.float32
        )
        action = self._mlp(obs)
        self.last_action = action.astype(np.float32)
        self.phase += self._phase_dt(command)
        self.phase = np.mod(self.phase + np.pi, 2.0 * np.pi) - np.pi

        af = self.action_ema
        self._filtered_act = af * self._filtered_act + (1.0 - af) * self.last_action
        anchor = (
            self._height_ctrl(command[3]) if self.cmd_dim >= 4 else self.home_ctrl
        )
        targets = anchor + self._filtered_act * self.action_scale
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
