"""Shared plumbing for four_bar_bot MJX tasks.

Model loading, actuator address tables and IMU/foot helpers, extracted from
the joystick env so getup/jump tasks reuse them. Task envs subclass this and
provide their own config, reset, step, observations and rewards.
"""

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from brax import math as brax_math
from mujoco import mjx
from mujoco_playground._src import mjx_env

from wojtek_rl import paths
from wojtek_rl.build_model import FOOT_RADIUS

# Actuator indices of the knee cranks (third joints), paths.LEGS order.
KNEE_ACTUATORS = (2, 5, 8, 11)
# The four-bar snaps through its singular branch past this third-joint angle
# (the foot flips above the trunk). Recovery/jump policies must not command
# targets on the far branch; crossing it under load can break the linkage.
KNEE_SINGULARITY = 3.2


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
    raising an error. The measured numbers behind the defaults are in
    docs/plans/mjwarp-phase0-report.md §4.
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
        self._mj_model = mujoco.MjModel.from_xml_path(str(paths.SCENE_XML))
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

    def _customize_model(self, m: mujoco.MjModel) -> None:
        """Task-specific tweaks applied before the model is put on device."""

    def _make_data(self):
        """mjx.make_data on the resolved backend, with warp budgets applied."""
        return self._make_data_fn()

    # -- MjxEnv plumbing -------------------------------------------------
    @property
    def xml_path(self) -> str:
        return str(paths.SCENE_XML)

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
        z = data.geom_xpos[self._foot_geom_ids][:, 2]
        return z < FOOT_RADIUS + 0.005

    def _noisy(self, rng, clean, scales):
        noise = jax.random.uniform(rng, clean.shape, minval=-1.0, maxval=1.0)
        return clean + noise * scales

    def _step_with_latency(self, data, prev, new, d):
        """Substep scan: ctrl = `prev` targets while substep index < d,
        `new` targets from d onward. d=n_substeps holds `prev` the whole
        period (today's action_delay=1); d=0 applies `new` immediately.

        One per-substep `jp.where` covers every d in [0, n_substeps]. This
        path is reached only when latency DR is deliberately enabled; the
        disabled default stays on the stock mjx_env.step for the bitwise
        golden guarantee, so a couple of float ULPs at the d=0/d=n_substeps
        boundaries here are immaterial. A per-lane lax.cond would be wrong
        anyway: under the training vmap `d` is batched, so cond runs *every*
        branch and selects, tripling the substep physics for no benefit
        (jaxpr-verified). `jp.where` on the input ctrl is an exact element
        select, so no blending occurs -- boundaries apply prev/new exactly.
        """

        def _substep(data, i):
            ctrl = jp.where(i < d, prev, new)
            data = data.replace(ctrl=ctrl)
            return mjx.step(self._mjx_model, data), None

        return jax.lax.scan(_substep, data, jp.arange(self.n_substeps))[0]

    def _obs_catalog(self, data, info):
        """name -> observation vector. Task envs extend with their signals."""
        return {
            "gyro": self._gyro(data),
            "gravity": self._gravity_body(data),
            "joint_pos": data.qpos[self._qadr] - self._home_ctrl,
            "joint_vel": data.qvel[self._vadr],
            "last_act": info["last_act"],
            # Sim-only signals, meant for the privileged critic list:
            "linvel": self._local_linvel(data),
            "base_height": data.qpos[2:3],
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
