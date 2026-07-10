"""Commanded jump task for four_bar_bot.

Episode timeline: stand at home, then at a randomized step a jump command
ramps in (visible to the policy as a countdown), the robot launches, and
after touchdown it must recover to a quiet stand. Reward maximizes peak
base height during flight while penalizing tilt, lateral drift and the
impact spikes that would hurt the printed linkage on landing.
"""

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from wojtek_rl.base import KNEE_ACTUATORS, KNEE_SINGULARITY, WojtekEnv

# 3 gyro + 3 gravity + 12 qpos + 12 qvel + 12 last_act + 2 jump signal
OBS_SIZE = 44


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.004,
        # Physics backend. auto picks warp on a CUDA host and jax elsewhere.
        # naconmax_per_env scales with the training batch; njmax is per
        # world. See docs/plans/mjwarp-phase0-report.md section 4.
        sim=config_dict.create(
            backend="auto", naconmax_per_env=32, njmax=320, num_envs=1
        ),
        episode_length=250,  # 5 s: stand, jump, land, recover
        action_scale=0.5,
        jump_at_steps=(50, 120),  # command arrives 1.0-2.4 s in
        countdown_steps=25,  # 0.5 s ramp the policy can pre-load against
        knee_target_max=3.15,  # below the 3.2 snap-through singularity
        # Wind-up target: hold the base this far below stand height through
        # the countdown. v4's open-ended "deeper is better" crouch got
        # unlearned once the smoothness costs outweighed it; a target depth
        # pays only for actually holding a real crouch.
        crouch_depth=0.06,
        crouch_sigma=0.02,
        # Smoothness/energy cost multiplier while prepping/launching; full
        # cost stands and after landing. The jump is allowed to be dynamic
        # for the half second it needs.
        active_cost_scale=0.25,
        # Motor torque cap for THIS task only (model default is 6 Nm). The
        # MD80 drives deliver 15 Nm; 6 Nm barely exceeds body weight through
        # the linkage, which capped fbb_jump_v2 at a 2 cm hop.
        forcerange=6.0,
        stand_height=0.0,  # 0.0 -> settled home keyframe height at env init
        obs_noise=config_dict.create(
            gyro=0.2, gravity=0.05, joint_pos=0.01, joint_vel=1.5
        ),
        # Declarative observation spec: ordered catalog names, see
        # WojtekEnv._obs_catalog + jump_signal. state = actor,
        # privileged = critic.
        obs=config_dict.create(
            include=(),  # obs presets whitelist actor sensors; () = all
            state=(
                "gyro",
                "gravity",
                "joint_pos",
                "joint_vel",
                "last_act",
                "jump_signal",
            ),
            privileged=(
                "gyro",
                "gravity",
                "joint_pos",
                "joint_vel",
                "last_act",
                "jump_signal",
                "linvel",
                "base_height",
                "actuator_force",
                "foot_contact",
            ),
        ),
        fall=config_dict.create(min_height=0.05, max_tilt_gz=-0.4),
        reward=config_dict.create(
            scales=config_dict.create(
                jump_peak=40.0,
                flight=10.0,
                # Dense launch shaping: the sparse flight/peak rewards only
                # pay after takeoff, which exploration never found in
                # fbb_jump_v1 (flight reward stayed exactly 0 for 36M steps
                # while the policy farmed stand rewards). Grade the attempt:
                # upward base velocity during the jump window.
                launch_vel=5.0,
                crouch=2.0,
                stand_height=1.0,
                posture=1.0,
                orientation=-5.0,
                lateral_vel=-1.0,
                torques=-2e-4,
                action_rate=-0.25,
                action_accel=-0.1,
                energy=-2e-3,
                dof_acc=-2.5e-7,
                dof_vel=-0.1,
                knee_singularity=-1.0,
                termination=-1.0,
            ),
        ),
    )


class WojtekJump(WojtekEnv):
    def __init__(self, config=None, config_overrides=None):
        super().__init__(config or default_config(), config_overrides)
        if not self._config.stand_height:
            self._config.stand_height = float(self._home_qpos[2])
        self._knee_idx = jp.array(KNEE_ACTUATORS)
        self._target_lo = self._ctrlrange[:, 0]
        self._target_hi = self._ctrlrange[:, 1].at[self._knee_idx].set(
            self._config.knee_target_max
        )

    def _customize_model(self, m):
        fr = float(self._config.forcerange)
        m.actuator_forcerange[:] = [-fr, fr]

    # -- reset / step -------------------------------------------------------
    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, r_pos, r_jump = jax.random.split(rng, 3)
        qpos = self._home_qpos.at[self._qadr].add(
            jax.random.uniform(r_pos, (12,), minval=-0.05, maxval=0.05)
        )
        data = self._make_data()
        data = data.replace(
            qpos=qpos, qvel=jp.zeros(self._mj_model.nv), ctrl=self._home_ctrl
        )
        data = mjx.forward(self._mjx_model, data)

        lo, hi = self._config.jump_at_steps
        info = {
            "rng": rng,
            "last_act": jp.zeros(12),
            "last_last_act": jp.zeros(12),
            "step_count": jp.array(0),
            "jump_at": jax.random.randint(r_jump, (), lo, hi),
            "max_z": jp.array(self._config.stand_height),
            "been_airborne": jp.array(False),
            "landed": jp.array(False),
        }
        metrics = {f"reward/{k}": jp.zeros(()) for k in self._config.reward.scales}
        metrics["jump/peak_height"] = jp.zeros(())
        obs = self._get_obs(data, info)
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def _phase(self, info):
        """pre (waiting) -> prep (countdown: crouch) -> jump -> post (landed)."""
        commanded = info["step_count"] >= info["jump_at"]
        in_jump = commanded & ~info["landed"]
        prep = ~commanded & (
            info["step_count"] >= info["jump_at"] - self._config.countdown_steps
        )
        return commanded, in_jump, prep

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        info = dict(state.info)
        rng, r_noise = jax.random.split(info["rng"])
        info["rng"] = rng

        motor_targets = jp.clip(
            self._home_ctrl + action * self._config.action_scale,
            self._target_lo,
            self._target_hi,
        )
        data = mjx_env.step(self._mjx_model, state.data, motor_targets, self.n_substeps)

        contact = self._foot_contact(data)
        airborne = ~jp.any(contact)
        commanded, in_jump, prep = self._phase(info)
        info["been_airborne"] = info["been_airborne"] | (in_jump & airborne)
        info["landed"] = info["landed"] | (
            info["been_airborne"] & jp.any(contact)
        )

        rewards, done = self._get_reward(data, info, action, airborne, in_jump, prep)
        # Track the flight apex after the command; reward pays the increments.
        new_max = jp.maximum(info["max_z"], data.qpos[2])
        info["max_z"] = jp.where(in_jump, new_max, info["max_z"])

        info["last_last_act"] = info["last_act"]
        info["last_act"] = action
        info["step_count"] = info["step_count"] + 1

        reward = sum(
            rewards[k] * self._config.reward.scales[k] for k in rewards
        )
        reward = jp.clip(reward * self.dt, -100.0, 100.0)
        metrics = {
            **state.metrics,
            **{f"reward/{k}": v for k, v in rewards.items()},
            "jump/peak_height": info["max_z"] - self._config.stand_height,
        }
        obs = self._get_obs(data, info, r_noise)
        return mjx_env.State(data, obs, reward, done.astype(jp.float32), metrics, info)

    # -- observations -------------------------------------------------------
    def _obs_catalog(self, data, info):
        catalog = super()._obs_catalog(data, info)
        countdown = jp.clip(
            (info["jump_at"] - info["step_count"])
            / self._config.countdown_steps,
            0.0,
            1.0,
        )
        _, in_jump, _ = self._phase(info)
        catalog["jump_signal"] = jp.array(
            [countdown, in_jump.astype(jp.float32)]
        )
        return catalog

    def _get_obs(self, data, info, rng=None):
        return self._build_obs(data, info, rng)

    # -- rewards ------------------------------------------------------------
    def _get_reward(self, data, info, action, airborne, in_jump, prep):
        gravity = self._gravity_body(data)
        height = data.qpos[2]
        qpos = data.qpos[self._qadr]
        qvel = data.qvel[self._vadr]
        linvel = self._local_linvel(data)
        z0 = self._config.stand_height

        fall = (height < self._config.fall.min_height) | (
            gravity[2] > self._config.fall.max_tilt_gz
        )
        # Crouching must not cost reward during the wind-up: fbb_jump_v2/v3
        # twitched instead of crouching because stand_height/posture stayed
        # active through the countdown.
        standing_phase = ~in_jump & ~prep
        knees = qpos[self._knee_idx]
        # Smoothness costs relax while the maneuver is on: v4 unlearned the
        # crouch because the down-up swing cost more in action_rate/energy
        # than the crouch reward paid.
        cost_scale = jp.where(
            prep | in_jump, self._config.active_cost_scale, 1.0
        )
        z_crouch = z0 - self._config.crouch_depth

        return {
            # Pay only for new apex height, so the total equals the peak.
            "jump_peak": in_jump * jp.maximum(height - info["max_z"], 0.0),
            "flight": in_jump * airborne * jp.maximum(height - z0, 0.0),
            "launch_vel": in_jump * jp.maximum(data.qvel[2], 0.0),
            # Wind-up: hold the target crouch depth through the countdown.
            "crouch": prep
            * jp.exp(-jp.square((height - z_crouch) / self._config.crouch_sigma)),
            "stand_height": standing_phase * jp.exp(-40.0 * jp.abs(height - z0)),
            "posture": standing_phase
            * jp.exp(-0.5 * jp.sum(jp.square(qpos - self._home_ctrl))),
            "orientation": jp.sum(jp.square(gravity[:2])),
            "lateral_vel": jp.sum(jp.square(linvel[:2])),
            "torques": jp.sum(jp.square(data.actuator_force)),
            "action_rate": cost_scale * jp.sum(jp.square(action - info["last_act"])),
            "action_accel": cost_scale
            * jp.sum(
                jp.square(action - 2 * info["last_act"] + info["last_last_act"])
            ),
            "energy": cost_scale
            * jp.sum(jp.abs(qvel) * jp.abs(data.actuator_force)),
            "dof_acc": jp.sum(jp.square(data.qacc[self._vadr])),
            "dof_vel": jp.sum(
                jp.square(jp.maximum(jp.abs(qvel) - 2 * jp.pi, 0.0))
            ),
            "knee_singularity": jp.sum(
                jp.maximum(knees - KNEE_SINGULARITY, 0.0)
            ),
            "termination": fall.astype(jp.float32),
        }, fall
