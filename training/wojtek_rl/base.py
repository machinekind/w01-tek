"""Shared plumbing for four_bar_bot MJX tasks.

Model loading, actuator address tables and IMU/foot helpers, extracted from
the joystick env so getup/jump tasks reuse them. Task envs subclass this and
provide their own config, reset, step, observations and rewards.
"""

import json

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from brax import math as brax_math
from mujoco import mjx
from mujoco_playground._src import mjx_env

from wojtek_rl import paths, terrain, terrain_env
from wojtek_rl.build_model import FOOT_RADIUS

# Actuator indices of the knee cranks (third joints), paths.LEGS order.
KNEE_ACTUATORS = (2, 5, 8, 11)
# Actuator indices of the abduction joints (first joints), paths.LEGS order.
ABDUCTION_ACTUATORS = (0, 3, 6, 9)
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
        # Terrain is a joystick-only opt-in block; getup/jump configs have no
        # `terrain` key, so `.get` leaves them on the flat scene untouched.
        terrain_cfg = self._config.get("terrain")
        self._terrain_enabled = bool(
            terrain_cfg is not None and terrain_cfg.get("enable", False)
        )
        scene_xml = paths.TERRAIN_SCENE_XML if self._terrain_enabled else paths.SCENE_XML
        if self._terrain_enabled:
            self._require_terrain_assets()
        self._mj_model = mujoco.MjModel.from_xml_path(str(scene_xml))
        self._mj_model.opt.timestep = self.sim_dt
        self._customize_model(self._mj_model)
        if self._terrain_enabled and terrain_cfg.get("feet_only", True):
            self._collide_feet_only(self._mj_model)
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

        if self._terrain_enabled:
            self._load_terrain(terrain_cfg)

    def _require_terrain_assets(self) -> None:
        missing = [
            p
            for p in (
                paths.TERRAIN_SCENE_XML,
                paths.TERRAIN_SPEC_JSON,
                paths.TERRAIN_LOOKUP_NPZ,
            )
            if not p.exists()
        ]
        if missing:
            names = ", ".join(p.name for p in missing)
            raise FileNotFoundError(
                f"terrain.enable=true but the generated terrain assets are "
                f"missing ({names}). Run `./training/run.sh build-terrain` "
                f"first; the sidecars are gitignored and built on demand."
            )

    def _collide_feet_only(self, m: mujoco.MjModel) -> None:
        """Drop the leg links out of terrain collision so only the four foot
        spheres (and the base box, as on flat) pair with the terrain geoms.

        The `*_link_floor` capsules/spheres carry the flat floor's leg-contact
        semantics; on rough terrain and stair edges they would multiply
        contacts, so the plan collides feet only (following mujoco_playground's
        rough-terrain tasks). Patched on the loaded model, never in the
        generated XML."""
        for i in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
            if name.endswith("_link_floor"):
                m.geom_contype[i] = 0
                m.geom_conaffinity[i] = 0

    def _load_terrain(self, terrain_cfg) -> None:
        """Build the device-side terrain lookup grid and the per-(row, type)
        spawn tables from the generated sidecars, plus the curriculum params
        the reset and the auto-reset wrapper read."""
        npz = np.load(paths.TERRAIN_LOOKUP_NPZ)
        x_min, x_max = float(npz["x_min"]), float(npz["x_max"])
        y_min, y_max = float(npz["y_min"]), float(npz["y_max"])
        ncol, nrow = int(npz["ncol"]), int(npz["nrow"])
        self._terrain_lookup = jp.asarray(npz["lookup"], dtype=jp.float32)
        self._terrain_x_min = x_min
        self._terrain_y_min = y_min
        self._terrain_cell_x = (x_max - x_min) / (ncol - 1)
        self._terrain_cell_y = (y_max - y_min) / (nrow - 1)

        spec = json.loads(paths.TERRAIN_SPEC_JSON.read_text())
        origin_xy, pad_h = terrain_env.tables_from_spec(spec, terrain.TYPES)
        self._terrain_origin_xy = jp.asarray(origin_xy)
        self._terrain_pad_h = jp.asarray(pad_h)
        self._terrain_n_rows = int(spec["n_rows"])
        self._terrain_n_types = len(terrain.TYPES)
        self._terrain_tile_size = float(spec["tile_size"])
        self._terrain_pad_jitter = float(terrain_cfg.get("pad_jitter", 0.15))
        self._terrain_spawn_yaw = bool(terrain_cfg.get("spawn_yaw", True))
        self._terrain_demote_fraction = float(terrain_cfg.get("demote_fraction", 0.5))
        self._terrain_init_level_frac = float(terrain_cfg.get("init_level_frac", 0.5))

        # Warp reserves one shared contact pool sized from naconmax_per_env;
        # feet-on-terrain contacts far exceed the flat default, and warp drops
        # the overflow silently (see check_terrain / docs). The real number is
        # measured on GPU in step 5; warn loudly if a warp terrain run kept the
        # untouched flat default.
        if self._backend == "warp" and self._config.sim.naconmax_per_env <= 32:
            print(
                "WARNING: terrain.enable on the warp backend with "
                f"sim.naconmax_per_env={self._config.sim.naconmax_per_env} "
                "(the flat default). Terrain contacts overflow this pool "
                "silently; set ++task.env.sim.naconmax_per_env to ~2x the flat "
                "default (measure with check-terrain on GPU)."
            )

    def _terrain_height(self, xy):
        """Ground-truth terrain surface height under world ``xy`` (``(..., 2)``)
        by clamped bilinear lookup. Only called when terrain is enabled."""
        return terrain_env.bilinear_sample(
            self._terrain_lookup,
            self._terrain_x_min, self._terrain_cell_x,
            self._terrain_y_min, self._terrain_cell_y,
            xy[..., 0], xy[..., 1],
        )

    def _customize_model(self, m: mujoco.MjModel) -> None:
        """Task-specific tweaks applied before the model is put on device."""

    def _make_data(self):
        """mjx.make_data on the resolved backend, with warp budgets applied."""
        return self._make_data_fn()

    # -- MjxEnv plumbing -------------------------------------------------
    @property
    def xml_path(self) -> str:
        return str(paths.TERRAIN_SCENE_XML if self._terrain_enabled else paths.SCENE_XML)

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
        foot = data.geom_xpos[self._foot_geom_ids]
        z = foot[:, 2]
        if self._terrain_enabled:
            z = z - self._terrain_height(foot[:, :2])
        return z < FOOT_RADIUS + 0.005

    def _base_height(self, data):
        """Base height above the local ground: world z on flat, height over the
        terrain surface under the base when terrain is enabled. The flat return
        is ``data.qpos[2]`` verbatim, so flat rewards/terminations are
        unchanged."""
        if self._terrain_enabled:
            return data.qpos[2] - self._terrain_height(data.qpos[0:2])
        return data.qpos[2]

    def _foot_clearance(self, data):
        """Per-foot clearance of the sphere bottom above the local ground. Flat
        return matches ``geom_xpos[..., 2] - FOOT_RADIUS`` verbatim."""
        foot = data.geom_xpos[self._foot_geom_ids]
        clearance = foot[:, 2] - FOOT_RADIUS
        if self._terrain_enabled:
            clearance = clearance - self._terrain_height(foot[:, :2])
        return clearance

    def _noisy(self, rng, clean, scales):
        noise = jax.random.uniform(rng, clean.shape, minval=-1.0, maxval=1.0)
        return clean + noise * scales

    def _step_with_latency(self, data, prev, new, d):
        """Run n_substeps physics steps: ctrl is `prev` while the substep
        index is below d, `new` from d onward. d=n_substeps holds `prev` for
        the whole period; d=0 applies `new` immediately.

        A single `jp.where` handles every d. A per-lane lax.cond would be
        wrong under the training vmap: `d` is batched, so cond runs every
        branch and selects, tripling the substep physics.
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
