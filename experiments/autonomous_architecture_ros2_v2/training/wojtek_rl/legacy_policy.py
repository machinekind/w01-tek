"""TEMPORARY: numpy runtime for schema-1 exports that carry a gait clock.

Delete this module once a schema-2 keeper that actually walks is published.

The deployment runtime (`wojtek_policy.policy.WojtekPolicy`) speaks the
schema-2 contract and its `KNOWN_COMPONENTS` has no `phase`. The last export
known to walk in the room demo -- `fbb_loco_v8` -- is schema-1 and its
observation *ends* with the phase clock:

    joint_pos:12  joint_vel:12  last_act:12  command:4  phase:8   = 48

so it cannot be loaded at all, while the current keeper
(`<HF_ORGANIZATION>/wojtek-springy-locomotion`) loads and does not walk: it was
trained with an obs preset that dropped both the IMU and the phase clock, and
a memoryless MLP with no clock settles on a fixed point instead of a gait
(5 mm in 5 s at a commanded 0.4 m/s, flat scene and room alike).

This bridge lives in `training/` on purpose. It is a sim-only crutch for the
demo and the SCAN-Planner physics tier; nothing in `ros/` changes, so the
robot's deployment path cannot pick it up by accident.

The clock, the walk/trot blend and the standing-height anchor are mirrored
from `wojtek_rl.env` (`_phase_dt`, `_leg_phases`, `_height_ctrl`) with the
constants read from the export's own meta.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from loguru import logger


def is_legacy_meta(meta: dict) -> bool:
    """True for a schema-1 export the deployment runtime cannot load."""
    return meta.get("schema_version") is None and any(
        entry.startswith("phase:") for entry in meta.get("obs_layout", [])
    )


class LegacyPhasePolicy:
    """Same call surface as WojtekPolicy: reset() / step(...) -> targets."""

    def __init__(self, npz_path, meta_path=None, clamp_knee: bool = False):
        npz_path = Path(npz_path)
        meta_path = Path(meta_path) if meta_path else npz_path.with_name(
            "policy_meta.json"
        )
        self.meta = m = json.loads(meta_path.read_text())
        data = np.load(npz_path)
        self._norm_mean = data["norm_mean"]
        self._norm_std = data["norm_std"]
        self._layers = []
        i = 0
        while f"hidden_{i}_kernel" in data:
            self._layers.append((data[f"hidden_{i}_kernel"], data[f"hidden_{i}_bias"]))
            i += 1

        self.joint_names = list(m["actuator_names"])
        self.home_ctrl = np.array(m["home_ctrl"], np.float32)
        self.ctrl_low = np.array(m["ctrl_low"], np.float32)
        self.ctrl_high = np.array(m["ctrl_high"], np.float32)
        self.action_scale = np.array(m["action_scale"], np.float32)
        self.action_filter = float(m.get("action_filter", 0.0))
        self.ctrl_dt = float(m["ctrl_dt"])
        self.knee_singularity = float(m["knee_singularity"])
        self.clamp_knee = clamp_knee

        # Gait clock (env.py: gait.freq band, gait.trot_band, WALK/TROT_PHASE).
        self.freq_band = tuple(m.get("gait_freq_band", (m["gait_freq_hz"],) * 2))
        self.trot_band = tuple(m["trot_band"])
        self._walk_phase = np.array(m["walk_phase"], np.float32)
        self._trot_phase = np.array(m["trot_phase"], np.float32)
        self.cmd_vmax = float(m["cmd_vmax"])
        self._height_table = np.array(m["height_table"], np.float32)
        self._dsecond_table = np.array(m["dsecond_table"], np.float32)
        self.command_fill = np.array(
            [float(np.mean(m["cmd_height_range"]))], np.float32
        )

        self.layout = [
            (name, int(width))
            for name, width in (e.split(":") for e in m["obs_layout"])
        ]
        names = [n for n, _ in self.layout]
        self.uses_imu = "gyro" in names or "gravity" in names
        self._cmd_width = dict(self.layout).get("command", 3)
        self.reset()

    @property
    def command_width(self) -> int:
        return self._cmd_width

    def reset(self) -> None:
        self.last_action = np.zeros(12, np.float32)
        self.filtered_action = np.zeros(12, np.float32)
        self.phase = 0.0  # master clock; per-leg offsets in _leg_phases
        self.last_obs = None

    # -- env.py mirrors ----------------------------------------------------

    def _cmd_speed(self, command) -> float:
        return float(np.linalg.norm(command[:2]) + 0.3 * abs(command[2]))

    def _gait_frac(self, speed: float) -> float:
        lo, hi = self.trot_band
        return float(np.clip((speed - lo) / (hi - lo), 0.0, 1.0))

    def _leg_phases(self, speed: float) -> np.ndarray:
        f = self._gait_frac(speed)
        offsets = (1.0 - f) * self._walk_phase + f * self._trot_phase
        return np.mod(self.phase + offsets + np.pi, 2 * np.pi) - np.pi

    def _advance_phase(self, speed: float) -> None:
        f0, f1 = self.freq_band
        frac = float(np.clip(speed / self.cmd_vmax, 0.0, 1.0))
        freq = f0 + (f1 - f0) * frac
        if speed > 0.05:  # a standing command freezes the clock
            self.phase += 2 * np.pi * self.ctrl_dt * freq
            self.phase = float(np.mod(self.phase + np.pi, 2 * np.pi) - np.pi)

    def _height_anchor(self, height: float) -> np.ndarray:
        dsecond = float(np.interp(height, self._height_table, self._dsecond_table))
        offset = np.tile(np.array([0.0, 1.0, 2.0], np.float32), 4) * dsecond
        return np.clip(self.home_ctrl + offset, self.ctrl_low, self.ctrl_high)

    # -- runtime -----------------------------------------------------------

    def _mlp(self, obs: np.ndarray) -> np.ndarray:
        x = (obs - self._norm_mean) / self._norm_std
        for kernel, bias in self._layers[:-1]:
            x = x @ kernel + bias
            x = x * np.exp(-np.logaddexp(0.0, -x))  # SiLU
        kernel, bias = self._layers[-1]
        x = x @ kernel + bias
        return np.tanh(x[: len(self.home_ctrl)])

    def step(self, gyro, gravity_body, joint_pos, joint_vel, command):
        """One control step (ctrl_dt) -> motor position targets (12,)."""
        command = np.asarray(command, np.float32)
        if command.size == 3 and self.command_fill.size:
            command = np.concatenate([command, self.command_fill])
        speed = self._cmd_speed(command)
        leg_phase = self._leg_phases(speed)

        parts = []
        for name, _ in self.layout:
            if name == "gyro":
                parts.append(np.asarray(gyro, np.float32))
            elif name == "gravity":
                parts.append(np.asarray(gravity_body, np.float32))
            elif name == "joint_pos":
                parts.append(np.asarray(joint_pos, np.float32) - self.home_ctrl)
            elif name == "joint_vel":
                parts.append(np.asarray(joint_vel, np.float32))
            elif name == "last_act":
                parts.append(self.last_action)
            elif name == "command":
                parts.append(command)
            elif name == "phase":
                parts.append(np.concatenate([np.cos(leg_phase), np.sin(leg_phase)]))
            else:
                raise ValueError(f"legacy runtime cannot assemble obs {name!r}")
        obs = np.concatenate(parts).astype(np.float32)
        if obs.size != self.meta["obs_size"]:
            raise ValueError(
                f"assembled obs has {obs.size} elements, meta says "
                f"{self.meta['obs_size']}"
            )
        self.last_obs = obs

        action = self._mlp(obs).astype(np.float32)
        self.last_action = action
        af = self.action_filter
        self.filtered_action = af * self.filtered_action + (1.0 - af) * action

        anchor = self._height_anchor(float(command[3]) if command.size > 3
                                     else float(self.command_fill[0]))
        targets = np.clip(
            anchor + self.filtered_action * self.action_scale,
            self.ctrl_low, self.ctrl_high,
        )
        if self.clamp_knee:
            # Real-robot safety only: never command the far branch of the
            # four-bar. Off in sim, matching the training env.
            targets[2::3] = np.minimum(targets[2::3], self.knee_singularity)
        self._advance_phase(speed)
        return targets.astype(np.float32)


def load(npz_path, meta_path=None, clamp_knee: bool = False) -> LegacyPhasePolicy:
    pol = LegacyPhasePolicy(npz_path, meta_path, clamp_knee)
    logger.warning(
        f"loaded {pol.meta['run_name']} through the TEMPORARY schema-1 phase "
        "bridge (wojtek_rl.legacy_policy) -- sim only, not a deployment path"
    )
    return pol
