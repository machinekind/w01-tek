"""Shared plumbing for four_bar_bot MJX tasks.

Model loading, actuator address tables and IMU/foot helpers, extracted from
the joystick env so getup/jump tasks reuse them. Task envs subclass this and
provide their own config, reset, step, observations and rewards.

On the terrain scene this class holds one ``terrain_env.Arena`` (``self._terrain``)
and asks it for surface heights. Reading the generated files, checking they are
the arena the code expects, and the height lookup itself all live in
``terrain_env``; nothing about the terrain file layout belongs here.
"""

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from brax import math as brax_math
from mujoco import mjx
from mujoco_playground._src import mjx_env

from wojtek_rl import paths, terrain_env
from wojtek_rl.build_model import BASE_BOX_NAMES, FOOT_RADIUS

# Actuator indices of the knee cranks (third joints), paths.LEGS order.
KNEE_ACTUATORS = (2, 5, 8, 11)
# Actuator indices of the abduction joints (first joints), paths.LEGS order.
ABDUCTION_ACTUATORS = (0, 3, 6, 9)
# The four-bar snaps through its singular branch past this third-joint angle
# (the foot flips above the trunk). Recovery/jump policies must not command
# targets on the far branch; crossing it under load can break the linkage.
KNEE_SINGULARITY = 3.2
# A base half-box counts as touching the terrain when its lowest corner sits
# within this of the surface. Looser than the foot threshold because the
# surface comes from the bilinear lookup, which is smoother than the
# heightfield prisms the physics actually collides against.
BASE_CONTACT_TOL = 0.01


def resolve_backend(backend: str) -> str:
    """Resolve a sim.backend value to "jax" or "warp".

    "auto" picks warp when jax runs on a GPU and the vendored MJWarp
    imports, and jax otherwise. Explicit values pass through. "warp" on a
    host without CUDA fails later in put_model, and that failure should
    stay loud.
    """
    if backend in ("jax", "warp"):
        return backend
    if backend != "auto":
        raise ValueError(f"sim.backend must be jax, warp or auto, got {backend!r}")
    try:
        from mujoco.mjx import warp as mjxw

        warp_ok = bool(mjxw.WARP_INSTALLED)
    except Exception:
        warp_ok = False
    return "warp" if warp_ok and jax.default_backend() == "gpu" else "jax"


def data_budget_kwargs(
    backend: str, naconmax_per_env: int, njmax: int, num_envs: int
) -> dict:
    """make_data buffer kwargs for the resolved backend.

    Warp reserves fixed buffer space for contacts and constraints before the
    simulation runs. The jax backend sizes its own buffers on the fly, so it
    takes no kwargs.

    naconmax sizes one shared contact pool for the whole batch of envs. It
    is naconmax_per_env multiplied by num_envs.

    njmax sizes the constraint rows for a single world. Every env in the
    batch gets its own njmax rows, so this number never multiplies by
    num_envs.

    If a buffer is too small, warp drops the overflow silently instead of
    raising an error. The defaults come from measured peaks; re-measure with
    `check-terrain` when the arena or the batch geometry changes.
    """
    if backend != "warp":
        return {}
    return {
        "naconmax": int(naconmax_per_env) * int(num_envs),
        "njmax": int(njmax),
    }


def make_data_fn(backend, mj_model, mjx_model, naconmax_per_env, njmax, num_envs):
    """Return a zero-argument callable that builds a fresh mjx.Data on the backend.

    The warp branch applies the buffer budgets from data_budget_kwargs. The
    jax branch stays byte-for-byte the call the envs made before the backend
    flag existed.
    """
    if backend == "warp":
        kwargs = data_budget_kwargs("warp", naconmax_per_env, njmax, num_envs)
        return lambda: mjx.make_data(mj_model, impl="warp", **kwargs)
    return lambda: mjx.make_data(mjx_model)


class WojtekEnv(mjx_env.MjxEnv):
    def __init__(self, config, config_overrides=None):
        super().__init__(config, config_overrides)
        # Terrain is opt-in and joystick-only. getup/jump have no terrain key, so
        # `.get` leaves them on the flat scene. terrain_env owns the loading,
        # validation and lookup; this class only asks it for heights.
        terrain_cfg = self._config.get("terrain")
        enabled = bool(terrain_cfg is not None and terrain_cfg.enable)
        self._terrain = terrain_env.load(terrain_cfg) if enabled else None
        scene_xml = self._terrain.files["scene"] if enabled else paths.SCENE_XML
        self._mj_model = mujoco.MjModel.from_xml_path(str(scene_xml))
        self._mj_model.opt.timestep = self.sim_dt
        self._customize_model(self._mj_model)
        sim = self._config.sim
        self._backend = resolve_backend(sim.backend)
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._backend)
        self._make_data_fn = make_data_fn(
            self._backend,
            self._mj_model,
            self._mjx_model,
            sim.naconmax_per_env,
            sim.njmax,
            sim.num_envs,
        )

        key = self._mj_model.key("home")
        self._home_qpos = jp.array(key.qpos)
        self._home_ctrl = jp.array(key.ctrl)

        m = self._mj_model
        self._qadr = jp.array(
            [m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)]
        )
        self._vadr = jp.array(
            [m.jnt_dofadr[m.actuator_trnid[i, 0]] for i in range(m.nu)]
        )
        self._ctrlrange = jp.array(m.actuator_ctrlrange)
        self._foot_geom_ids = np.array(
            [m.geom(f"{leg}_foot_sphere").id for leg in paths.LEGS]
        )
        self._base_geom_ids = np.array([m.geom(n).id for n in BASE_BOX_NAMES])
        self._base_geom_half = jp.array(
            m.geom_size[self._base_geom_ids][:, :3]
        )
        self._sensor_adr = {
            name: m.sensor(name).adr[0]
            for name in ("orientation", "angular-velocity", "linear-acceleration")
        }
        self._foot_linvel_adr = jp.array(
            [
                [m.sensor(f"{leg}_foot_linvel").adr[0] + i for i in range(3)]
                for leg in paths.LEGS
            ]
        )

        if self._terrain_enabled:
            self._warn_on_small_contact_budget()

    @property
    def _terrain_enabled(self) -> bool:
        return self._terrain is not None

    def _warn_on_small_contact_budget(self) -> None:
        # Warp has a fixed contact pool and drops overflow silently, so an
        # undersized budget is wrong physics with no error. Warn against a floor
        # derived from the model rather than a rule of thumb: warp allows four
        # contacts per geom-heightfield pair, so every collision geom on the
        # robot can put four contacts in the pool before a single box is touched.
        # That is a floor, not a budget -- measure the real number with
        # `check-terrain --backend warp` (the jax backend never applies it).
        if self._backend == "warp":
            floor = 4 * self._count_ground_colliding_geoms()
            if self._config.sim.naconmax_per_env < floor:
                print(
                    "WARNING: terrain.enable on the warp backend with "
                    f"sim.naconmax_per_env="
                    f"{self._config.sim.naconmax_per_env}, below the "
                    f"heightfield-only floor of {floor} "
                    f"(4 contacts x {floor // 4} colliding geoms). Warp drops "
                    "the overflow silently. Measure it: "
                    "`./training/run.sh check-terrain --backend warp "
                    f"--arena {self._terrain.kind}`."
                )

    def _count_ground_colliding_geoms(self) -> int:
        """Robot geoms that can pair with the terrain.

        Keyed on body, not name: the ground is whatever sits in the worldbody
        (the flat scene's floor plane, or the generator's heightfield and boxes),
        and everything on a real body is the robot. 21 on this model -- the base
        box, the four feet, and four per-leg collision proxies."""
        m = self._mj_model
        return sum(
            1
            for i in range(m.ngeom)
            if m.geom_bodyid[i] != 0 and (m.geom_contype[i] or m.geom_conaffinity[i])
        )

    def _customize_model(self, m: mujoco.MjModel) -> None:
        """Task-specific tweaks applied before the model is put on device."""

    def _make_data(self):
        """mjx.make_data on the resolved backend, with warp budgets applied."""
        return self._make_data_fn()

    # -- MjxEnv plumbing -------------------------------------------------
    @property
    def xml_path(self) -> str:
        return str(
            self._terrain.files["scene"] if self._terrain_enabled else paths.SCENE_XML
        )

    @property
    def action_size(self) -> int:
        return self._mj_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    # -- helpers ----------------------------------------------------------
    def _quat(self, data):
        adr = self._sensor_adr["orientation"]
        return data.sensordata[adr : adr + 4]

    def _gyro(self, data):
        adr = self._sensor_adr["angular-velocity"]
        return data.sensordata[adr : adr + 3]

    def _gravity_body(self, data):
        return brax_math.rotate(
            jp.array([0.0, 0.0, -1.0]), brax_math.quat_inv(self._quat(data))
        )

    def _local_linvel(self, data):
        return brax_math.rotate(data.qvel[:3], brax_math.quat_inv(self._quat(data)))

    def _foot_contact(self, data):
        """Per-foot contact flag, from a height lookup rather than a contact
        force.

        It has a blind band on terrain, in two places, and both are one-sided
        (a foot that IS touching reads as airborne):

        - A foot pressed against the vertical face of a riser sits above the
          surface the lookup returns for its own xy, so it reads as airborne.
        - The lookup rasterises a box onto the node grid, so within about one
          cell of a box edge (0.04 m, and up to ~0.05 m after bilinear
          interpolation) it returns the box top for ground next to the box, and
          a foot on that ground reads as airborne.

        Both land on the stair and step tiles the curriculum aims at. Replacing
        this with real contact forces is the fix; it has not been done."""
        foot = data.geom_xpos[self._foot_geom_ids]
        z = foot[:, 2]
        if self._terrain_enabled:
            z = z - self._terrain.height(foot[:, :2])
        return z < FOOT_RADIUS + 0.005

    def _base_terrain_contact(self, data):
        """Whether any base collision box is down on the terrain.

        Feeds the base-contact metrics, and termination when
        ``fall.on_base_contact`` is set. No observation reads it. Like
        ``_foot_contact`` it works off the height lookup rather than contact
        forces, and inherits that method's blind bands. A box has no single
        lowest point, so it takes the lowest corner -- the centre minus the
        box's own extent along world z -- and compares it against the surface
        under the centre, which is the height ``_base_height`` already uses.

        Terrain only; the flat scene has no lookup to read.
        """
        centre = data.geom_xpos[self._base_geom_ids]
        # geom_xmat rows are the geom frame's axes in world coordinates, so
        # row 2 against the half-sizes is the box's half-extent along world z.
        rot = data.geom_xmat[self._base_geom_ids].reshape(-1, 3, 3)
        reach = jp.sum(jp.abs(rot[:, 2, :]) * self._base_geom_half, axis=-1)
        gap = (centre[:, 2] - reach) - self._terrain.height(centre[:, :2])
        return jp.any(gap < BASE_CONTACT_TOL)

    def _base_height(self, data):
        """Base height above the ground. Flat: ``data.qpos[2]``, unchanged.
        Terrain: minus the surface height under the base."""
        if self._terrain_enabled:
            return data.qpos[2] - self._terrain.height(data.qpos[0:2])
        return data.qpos[2]

    def _foot_clearance(self, data):
        """Height of each foot's bottom above the ground. Flat: the old
        ``geom_xpos[..., 2] - FOOT_RADIUS``, unchanged."""
        foot = data.geom_xpos[self._foot_geom_ids]
        clearance = foot[:, 2] - FOOT_RADIUS
        if self._terrain_enabled:
            clearance = clearance - self._terrain.height(foot[:, :2])
        return clearance

    def _noisy(self, rng, clean, scales):
        noise = jax.random.uniform(rng, clean.shape, minval=-1.0, maxval=1.0)
        return clean + noise * scales

    def _step_with_latency(self, data, prev, new, d, qfrc_prev=None, qfrc_new=None):
        """Run n_substeps physics steps: ctrl is `prev` while the substep
        index is below d, `new` from d onward. d=n_substeps holds `prev` for
        the whole period; d=0 applies `new` immediately.

        `qfrc_prev`/`qfrc_new` (optional, full nv-wide vectors) ride the
        same switching into data.qfrc_applied — the joystick env's
        feed-forward torque head uses this so tau_ff sees the same latency
        as the position targets. None leaves qfrc_applied untouched.

        A single `jp.where` handles every d. A per-lane lax.cond would be
        wrong under the training vmap: `d` is batched, so cond runs every
        branch and selects, tripling the substep physics.
        """

        def _substep(data, i):
            ctrl = jp.where(i < d, prev, new)
            data = data.replace(ctrl=ctrl)
            if qfrc_prev is not None:
                data = data.replace(
                    qfrc_applied=jp.where(i < d, qfrc_prev, qfrc_new)
                )
            return mjx.step(self._mjx_model, data), None

        return jax.lax.scan(_substep, data, jp.arange(self.n_substeps))[0]

    def _obs_catalog(self, data, info):
        """name -> observation vector. Task envs extend with their signals."""
        return {
            "gyro": self._gyro(data),
            "gravity": self._gravity_body(data),
            # The foot geoms' sliding-friction coefficients, read from the
            # LIVE model. Under brax domain randomization the playground
            # wrapper rebinds self._mjx_model to the per-env randomized
            # model inside every vmapped reset/step, so with
            # dr.foot_friction enabled this IS the env's own mu draw (and,
            # since that DR gives feet contact priority, the contact
            # friction the robot actually walks on). PRIVILEGED-ONLY in
            # spirit: no physical sensor measures mu, so a policy whose
            # ACTOR observes it (a mu-teacher) can never deploy — the ROS
            # runtime refuses the component. Without foot_friction DR it
            # degenerates to the model constant (0.9), observed variance
            # zero.
            "foot_friction": self._mjx_model.geom_friction[
                self._foot_geom_ids, 0
            ],
            "joint_pos": data.qpos[self._qadr] - self._home_ctrl,
            "joint_vel": data.qvel[self._vadr],
            "last_act": info["last_act"],
            # Sim-only signals, meant for the privileged critic list:
            "linvel": self._local_linvel(data),
            # Terrain-relative, like the reward and termination paths. On the
            # flat scene _base_height IS qpos[2], so getup/jump (which observe
            # this and have no terrain key) are unchanged.
            "base_height": jp.atleast_1d(self._base_height(data)),
            "actuator_force": data.actuator_force,
            "foot_contact": self._foot_contact(data).astype(jp.float32),
        }

    @property
    def actor_obs_names(self):
        """Resolved actor observation list: the task's ordered obs.state,
        filtered by the obs.include whitelist when one is set (sensor-suite
        presets name what the robot HAS; task signals it doesn't list are
        dropped too, so presets must include them explicitly)."""
        include = self._config.obs.get("include", ())
        names = list(self._config.obs.state)
        if include:
            names = [n for n in names if n in include]
        if not names:
            raise ValueError(
                f"obs.include {list(include)} leaves no actor observations "
                f"(task obs.state: {list(self._config.obs.state)})"
            )
        return names

    def _build_obs(self, data, info, rng=None):
        """Observations declared by the env config.

        `obs.state` (actor: sensors the real robot exposes) and
        `obs.privileged` (critic: anything the sim knows) are ordered lists
        of catalog names. Actor noise scales come from `obs_noise` by
        component name (no entry = noise-free).
        """
        catalog = self._obs_catalog(data, info)

        def gather(names):
            missing = [n for n in names if n not in catalog]
            if missing:
                raise KeyError(
                    f"unknown obs component(s) {missing}; available: "
                    f"{sorted(catalog)}"
                )
            return jp.concatenate([catalog[n] for n in names])

        state_names = self.actor_obs_names
        state = gather(state_names)
        if rng is not None:
            noise = self._config.obs_noise
            scales = jp.concatenate(
                [jp.full(catalog[n].shape, noise.get(n, 0.0)) for n in state_names]
            )
            state = self._noisy(rng, state, scales)
        return {
            "state": state,
            "privileged_state": gather(self._config.obs.privileged),
        }
