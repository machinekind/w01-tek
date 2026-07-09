"""Safe fall-recovery task for four_bar_bot.

Rewards ported from the mujoco_playground Go1 Getup task, with four-bar
specifics on top:
  - reset keeps the kinematic loops closed: instead of sampling raw qpos
    (which would violate the connect equalities and can blow up the
    2-iteration Newton solve), the robot is dropped from a random
    orientation while the PD motors drive random targets during a settle
    phase, producing a physically consistent crumpled pose.
  - knee (third joint) motor targets are clipped below the snap-through
    singularity at ~3.2 rad, and being past it is penalized: on hardware
    crossing the singular branch under load can break the printed linkage.
  - "safe" costs: lateral base momentum, joint velocity/acceleration and
    torque spikes are penalized so the recovery is gentle on the legs.
"""

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from fbb_rl.base import KNEE_ACTUATORS, KNEE_SINGULARITY, FourBarBotEnv

# 3 gyro + 3 gravity + 12 qpos + 12 qvel + 12 last_act
OBS_SIZE = 42


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.004,
        episode_length=300,  # 6 s
        action_scale=0.5,
        drop_from_height_prob=0.8,
        drop_height=0.35,
        settle_time=0.7,
        knee_target_max=3.15,  # hard clip, just below the 3.2 singularity
        obs_noise=config_dict.create(
            gyro=0.2, gravity=0.05, joint_pos=0.01, joint_vel=1.5
        ),
        # Declarative observation spec: ordered catalog names, see
        # FourBarBotEnv._obs_catalog. state = actor, privileged = critic.
        obs=config_dict.create(
            include=(),  # obs presets whitelist actor sensors; () = all
            state=("gyro", "gravity", "joint_pos", "joint_vel", "last_act"),
            privileged=(
                "gyro",
                "gravity",
                "joint_pos",
                "joint_vel",
                "last_act",
                "linvel",
                "base_height",
                "actuator_force",
                "foot_contact",
            ),
        ),
        # 0.0 -> resolved to the settled home keyframe height at env init.
        stand_height=0.0,
        reward=config_dict.create(
            scales=config_dict.create(
                orientation=1.0,
                torso_height=1.0,
                posture=1.0,
                stand_still=1.0,
                action_rate=-0.001,
                torques=-1e-4,
                dof_acc=-2.5e-7,
                dof_vel=-0.1,
                dof_pos_limits=-0.1,
                lateral_vel=-0.5,
                knee_singularity=-1.0,
            ),
        ),
    )


class FourBarBotGetup(FourBarBotEnv):
    def __init__(self, config=None, config_overrides=None):
        super().__init__(config or default_config(), config_overrides)
        if not self._config.stand_height:
            self._config.stand_height = float(self._home_qpos[2])
        self._settle_steps = int(self._config.settle_time / self.sim_dt)
        self._knee_idx = jp.array(KNEE_ACTUATORS)
        # Per-actuator target bounds with the knees capped at the safe branch.
        self._target_lo = self._ctrlrange[:, 0]
        self._target_hi = self._ctrlrange[:, 1].at[self._knee_idx].set(
            self._config.knee_target_max
        )
        lo, hi = self._ctrlrange[:, 0], self._ctrlrange[:, 1]
        c, r = (lo + hi) / 2, hi - lo
        self._soft_lo = c - 0.5 * r * 0.95
        self._soft_hi = c + 0.5 * r * 0.95
        self._up = jp.array([0.0, 0.0, -1.0])

    # -- reset / step -------------------------------------------------------
    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, r_drop, r_quat, r_vel, r_ctrl = jax.random.split(rng, 5)
        drop = jax.random.bernoulli(r_drop, self._config.drop_from_height_prob)

        quat = jax.random.normal(r_quat, (4,))
        quat = quat / (jp.linalg.norm(quat) + 1e-6)
        qpos = self._home_qpos
        qpos = jp.where(
            drop,
            qpos.at[2].set(self._config.drop_height).at[3:7].set(quat),
            qpos,
        )
        qvel = jp.zeros(self._mj_model.nv)
        qvel = jp.where(
            drop,
            qvel.at[0:6].set(jax.random.uniform(r_vel, (6,), minval=-0.5, maxval=0.5)),
            qvel,
        )
        # Crumple the legs on the way down by driving random PD targets; the
        # loop closures stay physically consistent throughout.
        settle_ctrl = jax.random.uniform(
            r_ctrl, (self._mj_model.nu,), minval=self._target_lo, maxval=self._target_hi
        )
        settle_ctrl = jp.where(drop, settle_ctrl, self._home_ctrl)

        data = mjx.make_data(self._mjx_model)
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=settle_ctrl)
        data = mjx.forward(self._mjx_model, data)
        data = mjx_env.step(self._mjx_model, data, settle_ctrl, self._settle_steps)
        data = data.replace(time=0.0)

        info = {
            "rng": rng,
            "last_act": jp.zeros(12),
            "last_last_act": jp.zeros(12),
        }
        metrics = {f"reward/{k}": jp.zeros(()) for k in self._config.reward.scales}
        obs = self._get_obs(data, info)
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        info = dict(state.info)
        rng, r_noise = jax.random.split(info["rng"])
        info["rng"] = rng

        # Targets are relative to the CURRENT joint angles (playground getup
        # finding: home-relative targets cripple the initial range of motion
        # when the robot starts far from home).
        motor_targets = jp.clip(
            state.data.qpos[self._qadr] + action * self._config.action_scale,
            self._target_lo,
            self._target_hi,
        )
        data = mjx_env.step(self._mjx_model, state.data, motor_targets, self.n_substeps)

        rewards = self._get_reward(data, info, action)
        info["last_last_act"] = info["last_act"]
        info["last_act"] = action

        reward = sum(
            rewards[k] * self._config.reward.scales[k] for k in rewards
        )
        reward = jp.clip(reward * self.dt, 0.0, 10000.0)
        metrics = {
            **state.metrics,
            **{f"reward/{k}": v for k, v in rewards.items()},
        }
        obs = self._get_obs(data, info, r_noise)
        return mjx_env.State(data, obs, reward, jp.zeros(()), metrics, info)

    # -- observations -------------------------------------------------------
    def _get_obs(self, data, info, rng=None):
        return self._build_obs(data, info, rng)

    # -- rewards ------------------------------------------------------------
    def _get_reward(self, data, info, action):
        gravity = self._gravity_body(data)
        height = data.qpos[2]
        qpos = data.qpos[self._qadr]
        qvel = data.qvel[self._vadr]
        z_des = self._config.stand_height

        ori_err = jp.sum(jp.square(self._up - gravity))
        is_upright = ori_err < 0.01
        capped = jp.minimum(height, z_des)
        is_at_height = (z_des - capped) < 0.01
        gate = is_upright & is_at_height

        knees = qpos[self._knee_idx]
        world_vxy = data.qvel[0:2]

        return {
            "orientation": jp.exp(-2.0 * ori_err),
            "torso_height": jp.exp(capped) - 1.0,
            "posture": is_upright
            * jp.exp(-0.5 * jp.sum(jp.square(qpos - self._home_ctrl))),
            "stand_still": gate * jp.exp(-0.5 * jp.sum(jp.square(action))),
            "action_rate": jp.sum(jp.square(action - info["last_act"]))
            + jp.sum(
                jp.square(action - 2 * info["last_act"] + info["last_last_act"])
            ),
            "torques": jp.sqrt(jp.sum(jp.square(data.actuator_force)))
            + jp.sum(jp.abs(data.actuator_force)),
            "dof_acc": jp.sum(jp.square(data.qacc[self._vadr])),
            "dof_vel": jp.sum(
                jp.square(jp.maximum(jp.abs(qvel) - 2 * jp.pi, 0.0))
            ),
            "dof_pos_limits": jp.sum(
                -jp.clip(qpos - self._soft_lo, None, 0.0)
                + jp.clip(qpos - self._soft_hi, 0.0, None)
            ),
            # Safety: no sideways momentum while righting (world-frame xy).
            "lateral_vel": jp.sum(jp.square(world_vxy)),
            # Safety: get off the singular knee branch, gently.
            "knee_singularity": jp.sum(
                jp.maximum(knees - KNEE_SINGULARITY, 0.0)
            ),
        }
