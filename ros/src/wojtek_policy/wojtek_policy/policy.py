"""Numpy runtime for exported wojtek locomotion policies (schema 2).

Loads the policy.npz + policy_meta.json written by training/wojtek_rl/
export_policy.py. The meta is a complete deployment contract of RESOLVED
numbers -- final 12-vectors computed by the training env at export time --
so this runtime is a plain interpreter with no training knowledge:

  obs  = concat(obs_layout components)
  filt = action_filter * filt + (1 - action_filter) * tanh_mlp(obs)
  motor_targets = clip(anchor_ctrl + filt * action_scale,
                       target_low, target_high)

The one runtime-computed exception is the live standing height: a contract
whose command box carries a real height range (4th dim, low < high) makes
the stance anchor a function of the commanded height, so it cannot be a
single resolved vector. Such contracts ship ctrl_low/ctrl_high and the
runtime re-anchors with height_anchor() below, exactly as the training env
does every step.

A meta without schema_version 2 is refused: re-export the policy from its
run dir (./training/run.sh export --run ...) or, for a published keeper,
regenerate the meta with training/wojtek_rl/migrate_keeper_meta.py.

Joint order everywhere is the actuator order from policy_meta.json:
rear_left, rear_right, front_right, front_left x (first, second, third).
No ROS imports here so it is unit-testable without a ROS install.
"""

import json
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 2

# Obs components this interpreter can produce, mapped to the step() inputs.
# A layout naming anything else fails at load, not mid-flight.
KNOWN_COMPONENTS = (
    "gyro", "gravity", "joint_pos", "joint_vel", "last_act", "command"
)

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
        self.meta = m = json.loads(meta_path.read_text())
        version = m.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"policy_meta.json at {meta_path} has schema_version "
                f"{version!r}, this runtime needs {SCHEMA_VERSION} -- "
                "re-export the policy (./training/run.sh export --run ...) "
                "or regenerate a keeper's meta with "
                "training/wojtek_rl/migrate_keeper_meta.py"
            )
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
        self.anchor_ctrl = np.array(m["anchor_ctrl"], np.float32)
        self.action_scale = np.array(m["action_scale"], np.float32)
        self.target_low = np.array(m["target_low"], np.float32)
        self.target_high = np.array(m["target_high"], np.float32)
        self.command_low = np.array(m["command_low"], np.float32)
        self.command_high = np.array(m["command_high"], np.float32)
        self.command_fill = np.array(m["command_fill"], np.float32)
        self.action_filter = float(m["action_filter"])
        self.ctrl_dt = float(m["ctrl_dt"])
        self.knee_singularity = float(m["knee_singularity"])
        self.clamp_knee = clamp_knee

        self.layout = [
            (name, int(width))
            for name, width in (e.split(":") for e in m["obs_layout"])
        ]
        unknown = [n for n, _ in self.layout if n not in KNOWN_COMPONENTS]
        if unknown:
            raise ValueError(
                f"obs_layout components {unknown} are not supported by this "
                f"runtime (supported: {list(KNOWN_COMPONENTS)})"
            )
        names = [n for n, _ in self.layout]
        self.uses_imu = "gyro" in names or "gravity" in names
        self._cmd_width = dict(self.layout).get("command", 3)
        if self._cmd_width != 3 + self.command_fill.size:
            raise ValueError(
                f"command obs is {self._cmd_width}-D but command_fill has "
                f"{self.command_fill.size} entries "
                f"(expected {self._cmd_width - 3})"
            )

        # Live standing height: the anchor must track command[3] (the env
        # re-anchors every step), which needs the model ctrlrange -- the
        # anchor clips there, not at the narrower target bounds. A pinned
        # height (low == high) keeps the resolved anchor_ctrl untouched.
        self._anchor_height = None
        live = (
            self._cmd_width >= 4
            and float(self.command_high[3]) > float(self.command_low[3])
        )
        if live and not ("ctrl_low" in m and "ctrl_high" in m):
            raise ValueError(
                f"contract at {meta_path} trains a live standing-height "
                f"range [{self.command_low[3]}, {self.command_high[3]}] m "
                "but carries no ctrl_low/ctrl_high for the runtime to "
                "re-anchor with -- re-export the policy "
                "(./training/run.sh export --run ...)"
            )
        self._live_height = live
        if live:
            self.ctrl_low = np.array(m["ctrl_low"], np.float32)
            self.ctrl_high = np.array(m["ctrl_high"], np.float32)
            self._anchor_height = float(self.command_fill[0])

        self.reset()

    @property
    def command_width(self):
        return self._cmd_width

    def reset(self):
        self.last_action = np.zeros(12, np.float32)
        self.filtered_action = np.zeros(12, np.float32)
        self.last_obs = None

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
                parts.append(np.asarray(command, np.float32))
        obs = np.concatenate(parts).astype(np.float32)
        if obs.size != self.meta["obs_size"]:
            raise ValueError(
                f"assembled obs has {obs.size} elements, meta says "
                f"{self.meta['obs_size']}"
            )
        return obs

    def step(self, gyro, gravity_body, joint_pos, joint_vel, command):
        """One control step (ctrl_dt). Returns motor position targets (12,).

        All arrays in actuator order; gyro/gravity in the base (IMU) frame
        (ignored by policies whose obs_layout omits them). command is
        [vx, vy, wz] in the base frame -- dims the contract trained beyond
        that are filled from command_fill -- or the full trained command
        vector, whose 4th element is the commanded standing height (m).
        """
        command = np.asarray(command, np.float32)
        if command.size == 3 and self.command_fill.size:
            command = np.concatenate([command, self.command_fill])
        if self._live_height:
            # Training re-anchors the stance to the commanded height every
            # step (env.py uses _height_ctrl(command[3])); mirror that so
            # "zero action" means "stand at the commanded height".
            height = float(command[3])
            if height != self._anchor_height:
                self._anchor_height = height
                self.anchor_ctrl = height_anchor(
                    self.home_ctrl, height, self.ctrl_low, self.ctrl_high
                )
        obs = self._assemble_obs(gyro, gravity_body, joint_pos, joint_vel, command)
        self.last_obs = obs
        action = self._mlp(obs).astype(np.float32)
        self.last_action = action
        af = self.action_filter
        self.filtered_action = af * self.filtered_action + (1.0 - af) * action

        targets = self.anchor_ctrl + self.filtered_action * self.action_scale
        targets = np.clip(targets, self.target_low, self.target_high)
        if self.clamp_knee:
            # Real-robot safety: never command the far branch of the four-bar
            # (snap-through can break the linkage). Redundant when target_high
            # already caps the knees below the singularity.
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
