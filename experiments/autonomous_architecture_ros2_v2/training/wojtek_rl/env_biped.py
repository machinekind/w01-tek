"""Bipedal stand: rear up onto the hind legs and hold.

The getup recipe pointed at a different attitude target: gravity in the
body frame must reach [-1, 0, 0] (nose-up 90 deg, verified numerically),
the base must rise to biped_height, the HIND feet must carry the robot
and the FRONT feet must leave the ground. Balance is genuinely dynamic —
the feet are spheres, so the support is a line — which is why (unlike
the locomotion family) the ACTOR observes the IMU (gyro + gravity):
tilt about the foot line is unobservable from encoders alone.

Same no-termination philosophy as getup: a fall just stops earning; the
policy learns recovery for free from the drop-reset distribution.
"""

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from wojtek_rl.base import WojtekEnv
from wojtek_rl.env_getup import (
    KNEE_ACTUATORS,
    KNEE_SINGULARITY,
    default_config as _getup_config,
)

# Physically consistent TRUE-stand snapshot (loop closures satisfied),
# harvested from a plain-MuJoCo rollout: pitch -90 deg, hind targets
# (hip -1.4, knee 2.8), fronts tucked.  Unlike the v2 snapshot — which
# turned out to SIT on the hind third links (heel-rest tripod,
# statically stable, visually a sit) — this pose stands on the two foot
# spheres only, shins >5 cm clear, base at 0.39 m.  Knees rest at 2.71,
# leaving +-0.4 rad of authority below the 3.15 target clip (the v3
# snapshot pinned them at 3.11, +0.04 from the ceiling — no headroom to
# balance with).  Genuinely unstable (~0.3 s passive); holding it
# requires active balance.
BIPED_QPOS = [
    5.9133e-02, 0.0, 3.9274e-01, 6.2015e-01, 0.0, -7.8448e-01, 0.0,
    -6.2138e-02, -1.3589e+00, 2.7092e+00, 7.3703e-01, -7.2669e-01,
    -1.8611e-02, 6.2138e-02, -1.3589e+00, 2.7092e+00, -7.3703e-01,
    7.2669e-01, 1.8611e-02, -7.2670e-03, -9.9163e-01, 1.0290e+00,
    2.1845e-01, -2.1700e-01, -2.3897e-01, 7.2670e-03, -9.9163e-01,
    1.0290e+00, -2.1845e-01, 2.1700e-01, 2.3897e-01,
]
BIPED_CTRL = [
    0.0, -1.4, 2.8, 0.0, -1.4, 2.8, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0,
]


def default_config() -> config_dict.ConfigDict:
    cfg = _getup_config()
    cfg.episode_length = 500  # 10 s: rear up AND hold
    # Base height floor while reared (m).  The true tiptoe stand puts the
    # base at 0.35-0.40; the old 1.6x-quad value (0.206) was satisfiable
    # by the heel-rest SIT, which is exactly what must not pay.
    cfg.biped_height = 0.32
    # Reverse-curriculum mix: fraction of resets that start ON the
    # quad->reared corridor (learn to hold first; the drop starts then
    # learn to reach it).  Each such reset samples alpha ~ U(0,1) and
    # interpolates home -> reared (pitch alpha*90deg, joints/ctrl lerped)
    # so the whole transition path is covered, not just its endpoint.
    cfg.biped_init_prob = 0.5
    # Lower edge of the corridor's alpha range: 0.0 spans the whole
    # quad->reared path; ~0.9 confines training to the hold itself
    # (capability-probe mode).
    cfg.biped_alpha_min = 0.0
    # Short settle for corridor inits: just resolve interpolation
    # inconsistencies without collapsing the transient pose.
    cfg.biped_settle_time = 0.1
    cfg.reward = config_dict.create(
        scales=config_dict.create(
            # Attitude toward nose-up; the dominant shaping term.
            biped_orientation=2.0,
            # Base height, exp-capped at biped_height (getup pattern).
            biped_height=1.0,
            # Hind feet planted, front feet clear — gated by attitude
            # progress so lying on the back with legs up pays nothing.
            hind_contact=1.0,
            front_clear=1.0,
            # Sitting on the hind shins/heels while reared is the cheap
            # local optimum v2 converged to; make it pay negative.
            sit_contact=-2.0,
            # Holding: minimal action once reared and at height.
            stand_still=1.0,
            # Balance quality while reared: body rates hurt.
            body_rates=-0.05,
            # Safety/effort, straight from getup.
            action_rate=-0.001,
            torques=-1e-4,
            dof_acc=-2.5e-7,
            dof_vel=-0.1,
            dof_pos_limits=-0.1,
            # Softer than getup's -0.5: protective steps translate the
            # base, and taxing that suppresses the catch behavior.
            lateral_vel=-0.1,
            knee_singularity=-1.0,
        ),
    )
    return cfg


class WojtekBiped(WojtekEnv):
    """Rear-up-and-hold task. Reset/settle logic is inherited verbatim
    from the getup env by composition of the same config fields."""

    def __init__(self, config=None, config_overrides=None):
        super().__init__(config or default_config(), config_overrides)
        self._settle_steps = int(self._config.settle_time / self.sim_dt)
        self._biped_settle_steps = int(
            self._config.biped_settle_time / self.sim_dt
        )
        self._knee_idx = jp.array(KNEE_ACTUATORS)
        self._target_lo = self._ctrlrange[:, 0]
        self._target_hi = self._ctrlrange[:, 1].at[self._knee_idx].set(
            self._config.knee_target_max
        )
        lo, hi = self._ctrlrange[:, 0], self._ctrlrange[:, 1]
        c, r = (lo + hi) / 2, hi - lo
        self._soft_lo = c - 0.5 * r * 0.95
        self._soft_hi = c + 0.5 * r * 0.95
        # Nose-up attitude: gravity lands on -x in the body frame
        # (verified against the model: pitch -90deg -> [-1, 0, 0]).
        self._biped_up = jp.array([-1.0, 0.0, 0.0])
        self._biped_qpos = jp.array(BIPED_QPOS)
        self._biped_ctrl = jp.array(BIPED_CTRL)
        # Hind-leg link collision geoms ("*_link_floor"): any of them near
        # the floor while reared means the robot is sitting, not standing.
        # Lowest-point height from center/orientation (capsule axis is the
        # local z), same style as the base-contact helper.
        import mujoco as _mj

        m = self._mj_model
        sit_ids, sit_r, sit_hl = [], [], []
        for i in range(m.ngeom):
            n = _mj.mj_id2name(m, _mj.mjtObj.mjOBJ_GEOM, i) or ""
            if n.startswith("rear") and n.endswith("_link_floor"):
                sit_ids.append(i)
                sit_r.append(float(m.geom_size[i, 0]))
                sit_hl.append(
                    float(m.geom_size[i, 1])
                    if m.geom_type[i] == _mj.mjtGeom.mjGEOM_CAPSULE
                    else 0.0
                )
        self._sit_geom_ids = jp.array(sit_ids)
        self._sit_geom_r = jp.array(sit_r)
        self._sit_geom_hl = jp.array(sit_hl)

    # Step/observation logic is getup's verbatim; the method is reused
    # unbound.  Reset extends getup's drop/settle with the reared branch.
    from wojtek_rl.env_getup import WojtekGetup as _G

    step = _G.step
    _get_obs = _G._get_obs

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, r_mode, r_drop, r_quat, r_vel, r_ctrl, r_jnt, r_bvel = (
            jax.random.split(rng, 8)
        )
        biped = jax.random.bernoulli(r_mode, self._config.biped_init_prob)
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
            qvel.at[0:6].set(
                jax.random.uniform(r_vel, (6,), minval=-0.5, maxval=0.5)
            ),
            qvel,
        )
        settle_ctrl = jax.random.uniform(
            r_ctrl,
            (self._mj_model.nu,),
            minval=self._target_lo,
            maxval=self._target_hi,
        )
        settle_ctrl = jp.where(drop, settle_ctrl, self._home_ctrl)

        # Corridor branch: interpolate home -> reared by alpha, with joint
        # noise and a body-rate kick.  Intermediate poses are transient
        # (that is the point: the policy must finish the rear-up from
        # there), so they get only a short settle to resolve contacts.
        rng, r_alpha = jax.random.split(rng)
        # sqrt biases the corridor toward the reared end (pdf 2x): the
        # hold is where balance experience is scarce; the low end is
        # already covered by the drop starts.
        a_min = self._config.biped_alpha_min
        alpha = a_min + (1.0 - a_min) * jp.sqrt(jax.random.uniform(r_alpha))
        pitch = alpha * (-jp.pi / 2)
        biped_qpos = (
            (1.0 - alpha) * self._home_qpos + alpha * self._biped_qpos
        )
        biped_qpos = biped_qpos.at[3:7].set(
            jp.array(
                [jp.cos(pitch / 2), 0.0, jp.sin(pitch / 2), 0.0]
            )
        )
        biped_qpos = biped_qpos.at[7:].add(
            jax.random.uniform(
                r_jnt, (self._mj_model.nq - 7,), minval=-0.03, maxval=0.03
            )
        )
        biped_qvel = (
            jp.zeros(self._mj_model.nv)
            .at[3:6]
            .set(jax.random.uniform(r_bvel, (3,), minval=-0.3, maxval=0.3))
        )
        biped_ctrl = (
            (1.0 - alpha) * self._home_ctrl + alpha * self._biped_ctrl
        )

        data_g = self._make_data().replace(
            qpos=qpos, qvel=qvel, ctrl=settle_ctrl
        )
        data_g = mjx.forward(self._mjx_model, data_g)
        data_g = mjx_env.step(
            self._mjx_model, data_g, settle_ctrl, self._settle_steps
        )
        data_b = self._make_data().replace(
            qpos=biped_qpos, qvel=biped_qvel, ctrl=biped_ctrl
        )
        data_b = mjx.forward(self._mjx_model, data_b)
        data_b = mjx_env.step(
            self._mjx_model, data_b, biped_ctrl, self._biped_settle_steps
        )
        # Select only the minimal state and rebuild derived quantities in
        # a fresh Data: a whole-pytree where() drags Warp's internal
        # contact/index buffers through tracers and leaks them (observed
        # UnexpectedTracerError on the warp backend under pmap).
        data = self._make_data().replace(
            qpos=jp.where(biped, data_b.qpos, data_g.qpos),
            qvel=jp.where(biped, data_b.qvel, data_g.qvel),
            ctrl=jp.where(biped, biped_ctrl, settle_ctrl),
        )
        data = mjx.forward(self._mjx_model, data)

        info = {
            "rng": rng,
            "last_act": jp.zeros(12),
            "last_last_act": jp.zeros(12),
        }
        metrics = {
            f"reward/{k}": jp.zeros(()) for k in self._config.reward.scales
        }
        obs = self._get_obs(data, info)
        return mjx_env.State(
            data, obs, jp.zeros(()), jp.zeros(()), metrics, info
        )

    def _rear_link_down(self, data):
        """Per-geom flag: hind shin/heel link within 15 mm of the floor."""
        z = data.geom_xpos[self._sit_geom_ids, 2]
        rot = data.geom_xmat[self._sit_geom_ids].reshape(-1, 3, 3)
        low = z - (jp.abs(rot[:, 2, 2]) * self._sit_geom_hl + self._sit_geom_r)
        return low < 0.015

    # -- rewards ------------------------------------------------------------
    def _get_reward(self, data, info, action):
        gravity = self._gravity_body(data)
        height = data.qpos[2]
        qpos = data.qpos[self._qadr]
        qvel = data.qvel[self._vadr]
        z_des = self._config.biped_height

        ori_err = jp.sum(jp.square(self._biped_up - gravity))
        ori_kernel = jp.exp(-2.0 * ori_err)
        capped = jp.minimum(height, z_des)
        sit = self._rear_link_down(data).astype(jp.float32)
        clean = jp.prod(1.0 - sit)  # 1 iff no hind link touches
        gate = (ori_err < 0.05) & ((z_des - capped) < 0.02) & (clean > 0.5)

        contact = self._foot_contact(data).astype(jp.float32)
        clearance = self._foot_clearance(data)
        # Foot order: [rear_left, rear_right, front_right, front_left].
        hind = contact[:2]
        front_clr = clearance[2:]

        knees = qpos[self._knee_idx]
        world_vxy = data.qvel[0:2]
        gyro = self._gyro(data)

        return {
            "biped_orientation": ori_kernel,
            "biped_height": jp.exp(capped) - 1.0,
            # max, not mean: at least one hind foot planted.  Point-feet
            # bipeds catch falls by STEPPING (the v4 policy railed the
            # knee ceiling trying to catch a backward fall with both
            # feet planted); a both-feet-down reward taxes every
            # protective step.
            "hind_contact": ori_kernel * clean * jp.max(hind),
            "front_clear": ori_kernel
            * jp.mean(jp.clip(front_clr / 0.05, 0.0, 1.0)),
            # Binary near-reared gate, NOT the smooth kernel: a kernel
            # gate taxes the crouched pass-through states of every
            # rear-up attempt and suppresses trying (the slip-penalty
            # lesson).  Transitional states must pay nothing.
            "sit_contact": (ori_err < 0.2) * jp.sum(sit),
            "stand_still": gate * jp.exp(-0.5 * jp.sum(jp.square(action))),
            "body_rates": jp.sum(jp.square(gyro)),
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
            "lateral_vel": jp.sum(jp.square(world_vxy)),
            "knee_singularity": jp.sum(
                jp.maximum(knees - KNEE_SINGULARITY, 0.0)
            ),
        }
