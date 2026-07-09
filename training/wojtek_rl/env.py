"""Joystick velocity-tracking environment for four_bar_bot.

Skeleton copied from 3_jaxpot_robotics/jaxpot_robotics/race_env.py; rewards
ported from the mujoco_playground Go1 joystick task. Actor observations use
only signals the real robot has (IMU + joint encoders + own history).
"""

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from wojtek_rl.base import WojtekEnv
from wojtek_rl.build_model import FOOT_RADIUS

# 3 gyro + 3 gravity + 12 qpos + 12 qvel + 12 last_act + 4 cmd + 8 phase
OBS_SIZE = 54
PRIVILEGED_SIZE = 61  # OBS_SIZE + 3 linvel + 4 contacts

# Gait clock offsets per leg, paths.LEGS order (rear_left, rear_right,
# front_right, front_left): trot = diagonal pairs in antiphase; walk =
# 4-beat lateral sequence (RL, FL, RR, FR a quarter cycle apart). The
# command speed blends walk->trot offsets across gait.trot_band.
TROT_PHASE = (0.0, np.pi, 0.0, np.pi)
WALK_PHASE = (0.0, np.pi, 1.5 * np.pi, 0.5 * np.pi)

# Settled standing height vs uniform leg-extension offset, measured with a
# 2 s PD-hold settle on the current model (kp=20/kd=1): dsecond in rad on
# every second joint, dthird = 2*dsecond on every third joint.
HEIGHT_TABLE = (0.084, 0.094, 0.106, 0.121, 0.139, 0.160, 0.182)
DSECOND_TABLE = (-0.45, -0.30, -0.15, 0.0, 0.15, 0.30, 0.45)


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.004,
        episode_length=1000,
        action_scale=0.5,
        obs_noise=config_dict.create(
            gyro=0.2, gravity=0.05, joint_pos=0.01, joint_vel=1.5
        ),
        # Declarative observation spec: ordered lists of catalog names (see
        # WojtekEnv._obs_catalog + this env's command/phase additions).
        # `state` is the actor (sensors the real robot has), `privileged`
        # the critic. Changing either changes obs sizes, so checkpoints
        # don't transfer across obs configs.
        obs=config_dict.create(
            include=(),  # obs presets whitelist actor sensors; () = all
            state=(
                "gyro",
                "gravity",
                "joint_pos",
                "joint_vel",
                "last_act",
                "command",
                "phase",
            ),
            privileged=(
                "gyro",
                "gravity",
                "joint_pos",
                "joint_vel",
                "last_act",
                "command",
                "phase",
                "linvel",
                "foot_contact",
            ),
        ),
        command=config_dict.create(
            vx=(-0.6, 0.6),
            vy=(-0.4, 0.4),
            wz=(-0.7, 0.7),
            # Commanded standing height; the measured stable envelope is
            # 0.084-0.182 m, kept inside with margin for gait dynamics.
            height=(0.09, 0.17),
            resample_steps=250,  # 5 s
            # Velocity part zeroes with this prob (stand training); the
            # height command stays live so the robot stands AT the height.
            zero_prob=0.15,
        ),
        push=config_dict.create(enable=True, interval_steps=200, vel=0.4),
        action_delay=1,  # control steps of latency between policy and motors
        # EMA low-pass on actions before the PD targets (0 = off). Kills the
        # noise-driven standing limit cycle structurally; mirror the same
        # one-liner on the robot when deploying a policy trained with it.
        action_filter=0.0,
        # NOTE: design standing height is ~0.10 m (Task 3 correction); the
        # plan's original 0.10 min_height would terminate almost every step.
        fall=config_dict.create(min_height=0.06, max_tilt_gz=-0.4),
        # Trot clock: fbb_v2 skated at 7 Hz instead of stepping (duty factor
        # ~1.0); the phase reward makes periodic swings the only way to score.
        gait=config_dict.create(
            # Clock frequency band: freq[0] at the slowest walk, freq[1] at
            # max commanded speed (speed-scaled; the fbb_run_v1 lesson —
            # stride length is capped by the 0.21 m legs, fast commands
            # need a faster clock, not a frozen one).
            freq=(1.4, 3.0),  # Hz
            swing_height=0.03,  # target peak foot clearance, m
            # Walk->trot blend band on planar command speed (m/s): below the
            # band the offsets are a 4-beat walk, above it a diagonal trot,
            # inside they interpolate (an amble, like a real dog's).
            trot_band=(0.35, 0.55),
            # Stance fraction of the cycle, blended walk->trot with the same
            # band. 50% duty at walking speeds reads as a soft trot (v1:
            # diag-corr 0.82 in the walk band); a real walk is ~70% stance.
            duty=(0.7, 0.5),
        ),
        reward=config_dict.create(
            tracking_sigma=0.25,
            phase_sigma=0.002,
            height_sigma=1e-3,  # m^2: 1 cm err -> 0.90, 3 cm -> 0.41
            # Pose anchors to the commanded-height stance; leg joints get a
            # lighter weight than abduction so the gait may flex around it.
            pose_leg_weight=0.25,
            scales=config_dict.create(
                tracking_lin_vel=1.5,
                tracking_ang_vel=0.8,
                height_tracking=1.0,
                lin_vel_z=-2.0,
                ang_vel_xy=-0.05,
                orientation=-5.0,
                torques=-2e-4,
                action_rate=-0.25,
                action_accel=-0.1,
                energy=-2e-3,
                pose=-0.5,
                feet_air_time=2.0,
                feet_slip=-0.25,
                feet_phase=1.0,
                contact_match=1.0,
                stand_still=-0.5,
                termination=-1.0,
            ),
        ),
    )


class WojtekJoystick(WojtekEnv):
    def __init__(self, config=None, config_overrides=None):
        super().__init__(config or default_config(), config_overrides)

    def _sample_command(self, rng):
        """(vx, vy, wz, height). Zeroing (stand training) keeps the height."""
        r1, r2, r3, r4, r5 = jax.random.split(rng, 5)
        c = self._config.command
        vel = jp.array(
            [
                jax.random.uniform(r1, minval=c.vx[0], maxval=c.vx[1]),
                jax.random.uniform(r2, minval=c.vy[0], maxval=c.vy[1]),
                jax.random.uniform(r3, minval=c.wz[0], maxval=c.wz[1]),
            ]
        )
        zero = jax.random.bernoulli(r4, c.zero_prob)
        vel = jp.where(zero, jp.zeros(3), vel)
        height = jax.random.uniform(r5, minval=c.height[0], maxval=c.height[1])
        return jp.concatenate([vel, height[None]])

    def _cmd_speed(self, command):
        """Planar speed the gait clock should serve; turning counts too."""
        return jp.linalg.norm(command[:2]) + 0.3 * jp.abs(command[2])

    def _height_ctrl(self, height):
        """Ctrl anchor for a commanded standing height (measured table)."""
        dsecond = jp.interp(
            height, jp.array(HEIGHT_TABLE), jp.array(DSECOND_TABLE)
        )
        offset = jp.tile(jp.array([0.0, 1.0, 2.0]), 4) * dsecond
        return jp.clip(
            self._home_ctrl + offset, self._ctrlrange[:, 0], self._ctrlrange[:, 1]
        )

    def _gait_frac(self, command):
        """0 = walk, 1 = trot, blended across gait.trot_band."""
        band = self._config.gait.trot_band
        return jp.clip(
            (self._cmd_speed(command) - band[0]) / (band[1] - band[0]), 0.0, 1.0
        )

    def _leg_phases(self, info):
        """Per-leg clock: master phase + speed-blended walk/trot offsets."""
        f = self._gait_frac(info["command"])
        offsets = (1.0 - f) * jp.array(WALK_PHASE) + f * jp.array(TROT_PHASE)
        phase = info["phase"] + offsets
        return jp.fmod(phase + jp.pi, 2 * jp.pi) - jp.pi

    def _gait_targets(self, info):
        """(target foot clearance, stance mask) from the duty-aware clock.

        theta in [0,1) is the normalized leg phase; the swing window is the
        first (1-duty) of the cycle with a sinusoidal clearance bump, the
        rest is stance on the ground.
        """
        g = self._config.gait
        f = self._gait_frac(info["command"])
        duty = (1.0 - f) * g.duty[0] + f * g.duty[1]
        theta = jp.fmod(self._leg_phases(info) + 2 * jp.pi, 2 * jp.pi) / (
            2 * jp.pi
        )
        swing_frac = 1.0 - duty
        in_swing = theta < swing_frac
        rz = g.swing_height * jp.sin(jp.pi * theta / swing_frac) * in_swing
        return rz, ~in_swing

    def _phase_dt(self, command):
        """Clock increment: speed-scaled; frozen when told to stand."""
        g = self._config.gait
        c = self._config.command
        vmax = self._cmd_speed(
            jp.array([max(abs(c.vx[0]), abs(c.vx[1])), 0.0, 0.0, 0.0])
        )
        speed = self._cmd_speed(command)
        frac = jp.clip(speed / vmax, 0.0, 1.0)
        freq = g.freq[0] + (g.freq[1] - g.freq[0]) * frac
        return jp.where(speed > 0.05, 2 * jp.pi * self.dt * freq, 0.0)

    # -- reset / step -------------------------------------------------------
    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, r_cmd, r_pos = jax.random.split(rng, 3)
        command = self._sample_command(r_cmd)
        # Start posed for the commanded height: legs on the height anchor
        # (base z follows via the same measured table), plus joint noise.
        anchor = self._height_ctrl(command[3])
        qpos = self._home_qpos.at[self._qadr].set(
            anchor + jax.random.uniform(r_pos, (12,), minval=-0.05, maxval=0.05)
        )
        qpos = qpos.at[2].set(command[3])
        data = mjx.make_data(self._mjx_model)
        data = data.replace(qpos=qpos, qvel=jp.zeros(self._mj_model.nv), ctrl=anchor)
        data = mjx.forward(self._mjx_model, data)
        info = {
            "rng": rng,
            "command": command,
            "last_act": jp.zeros(12),
            "last_last_act": jp.zeros(12),
            "filtered_act": jp.zeros(12),
            "steps_since_cmd": jp.array(0),
            "feet_air_time": jp.zeros(4),
            "last_contact": jp.zeros(4, dtype=bool),
            "motor_targets": anchor,
            "step_count": jp.array(0),
            "phase": jp.array(0.0),  # master clock; per-leg via _leg_phases
        }
        metrics = {f"reward/{k}": jp.zeros(()) for k in self._config.reward.scales}
        obs = self._get_obs(data, info)
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        info = dict(state.info)
        rng, r_noise, r_cmd, r_push = jax.random.split(info["rng"], 4)
        info["rng"] = rng

        # Actions center on the commanded-height stance, so "zero action"
        # means "stand at the commanded height" at any height.
        af = self._config.action_filter
        filt = af * info["filtered_act"] + (1.0 - af) * action
        info["filtered_act"] = filt
        motor_targets = jp.clip(
            self._height_ctrl(info["command"][3])
            + filt * self._config.action_scale,
            self._ctrlrange[:, 0],
            self._ctrlrange[:, 1],
        )
        # one control step of latency: the motors receive last step's targets
        applied_targets = (
            info["motor_targets"] if self._config.action_delay > 0 else motor_targets
        )
        data = state.data
        # random horizontal push, every push.interval_steps
        if self._config.push.enable:
            push_now = (info["step_count"] % self._config.push.interval_steps) == (
                self._config.push.interval_steps - 1
            )
            push = jax.random.uniform(r_push, (2,), minval=-1.0, maxval=1.0)
            push = push / (jp.linalg.norm(push) + 1e-6) * self._config.push.vel
            qvel = data.qvel.at[:2].add(jp.where(push_now, push, jp.zeros(2)))
            data = data.replace(qvel=qvel)

        data = mjx_env.step(self._mjx_model, data, applied_targets, self.n_substeps)

        contact = self._foot_contact(data)
        contact_filt = contact | info["last_contact"]
        first_contact = (info["feet_air_time"] > 0) & contact_filt
        rewards, done = self._get_reward(
            data, info, action, motor_targets, first_contact, contact
        )
        info["feet_air_time"] = jp.where(
            contact_filt, 0.0, info["feet_air_time"] + self.dt
        )
        info["last_contact"] = contact
        info["last_last_act"] = info["last_act"]
        info["last_act"] = action
        info["motor_targets"] = motor_targets
        # Master clock advances at the speed the CURRENT command asks for
        # (frozen when standing); per-leg phases come from _leg_phases.
        phase = info["phase"] + self._phase_dt(info["command"])
        info["phase"] = jp.fmod(phase + jp.pi, 2 * jp.pi) - jp.pi
        info["step_count"] = info["step_count"] + 1
        info["steps_since_cmd"] = info["steps_since_cmd"] + 1
        resample = info["steps_since_cmd"] >= self._config.command.resample_steps
        info["command"] = jp.where(
            resample, self._sample_command(r_cmd), info["command"]
        )
        info["steps_since_cmd"] = jp.where(resample, 0, info["steps_since_cmd"])

        reward = sum(
            rewards[k] * self._config.reward.scales[k] for k in rewards
        )
        reward = jp.clip(reward * self.dt, -100.0, 100.0)
        # Merge over the incoming metrics: brax's EvalWrapper injects keys
        # (e.g. "reward") that must survive the step for scan carry parity.
        metrics = {
            **state.metrics,
            **{f"reward/{k}": v for k, v in rewards.items()},
        }
        obs = self._get_obs(data, info, r_noise)
        return mjx_env.State(data, obs, reward, done.astype(jp.float32), metrics, info)

    # -- observations -------------------------------------------------------
    def _obs_catalog(self, data, info):
        catalog = super()._obs_catalog(data, info)
        catalog["command"] = info["command"]
        leg_phase = self._leg_phases(info)
        catalog["phase"] = jp.concatenate([jp.cos(leg_phase), jp.sin(leg_phase)])
        return catalog

    def _get_obs(self, data, info, rng=None):
        return self._build_obs(data, info, rng)

    # -- rewards ------------------------------------------------------------
    def _get_reward(self, data, info, action, motor_targets, first_contact, contact):
        cmd = info["command"]
        linvel = self._local_linvel(data)
        gyro = self._gyro(data)
        gravity = self._gravity_body(data)
        sigma = self._config.reward.tracking_sigma
        moving = self._cmd_speed(cmd) > 0.05
        anchor = self._height_ctrl(cmd[3])
        # Abduction full weight; leg joints lighter (gait flexes around it).
        pose_w = jp.tile(
            jp.array([1.0, self._config.reward.pose_leg_weight,
                      self._config.reward.pose_leg_weight]), 4
        )

        fall = (data.qpos[2] < self._config.fall.min_height) | (
            gravity[2] > self._config.fall.max_tilt_gz
        )

        # Foot clearance above the floor vs the duty-aware swing profile,
        # plus explicit contact/stance schedule matching.
        foot_clearance = data.geom_xpos[self._foot_geom_ids][:, 2] - FOOT_RADIUS
        target_clearance, stance_mask = self._gait_targets(info)
        phase_err = jp.sum(jp.square(foot_clearance - target_clearance))
        contact_match = jp.mean(
            (contact == stance_mask).astype(jp.float32)
        )

        # Horizontal foot speed while in contact = skating.
        feet_vel = data.sensordata[self._foot_linvel_adr]
        slip = jp.sum(jp.sum(jp.square(feet_vel[:, :2]), axis=-1) * contact)

        qvel = data.qvel[self._vadr]

        rewards = {
            "tracking_lin_vel": jp.exp(
                -jp.sum(jp.square(cmd[:2] - linvel[:2])) / sigma
            ),
            "tracking_ang_vel": jp.exp(-jp.square(cmd[2] - gyro[2]) / sigma),
            "height_tracking": jp.exp(
                -jp.square(data.qpos[2] - cmd[3])
                / self._config.reward.height_sigma
            ),
            "lin_vel_z": jp.square(linvel[2]),
            "ang_vel_xy": jp.sum(jp.square(gyro[:2])),
            "orientation": jp.sum(jp.square(gravity[:2])),
            "torques": jp.sum(jp.square(data.actuator_force)),
            "action_rate": jp.sum(jp.square(action - info["last_act"])),
            "action_accel": jp.sum(
                jp.square(action - 2 * info["last_act"] + info["last_last_act"])
            ),
            "energy": jp.sum(jp.abs(qvel) * jp.abs(data.actuator_force)),
            "pose": jp.sum(pose_w * jp.square(data.qpos[self._qadr] - anchor)),
            "feet_air_time": jp.sum(
                (info["feet_air_time"] - 0.1) * first_contact
            )
            * moving,
            "feet_slip": slip * moving,
            "feet_phase": jp.exp(-phase_err / self._config.reward.phase_sigma)
            * moving,
            "contact_match": contact_match * moving,
            # Position pull to the anchor plus velocity damping: L1 position
            # alone caused bang-bang fidgeting around the anchor (v2).
            "stand_still": (
                jp.sum(jp.abs(data.qpos[self._qadr] - anchor))
                + 0.2 * jp.sum(jp.abs(qvel))
            )
            * (~moving),
            "termination": fall.astype(jp.float32),
        }
        return rewards, fall
