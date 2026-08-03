"""Auto-reset wrapper that runs the terrain curriculum.

Playground's stock auto-reset has two modes. The default restores one cached
state on done and never calls ``env.reset`` again, so a level held in env info
would stay frozen at its reset value. ``full_reset=True`` does re-run
``env.reset`` on done and preserves info for exactly this kind of curriculum
-- but it pays a full reset (kinematics, obs rebuild) inside every training
step. This wrapper takes the cheap path instead: restore the cached state,
then teleport the base to the new curriculum spawn. Timeout dones appear only
above EpisodeWrapper, so the tile switch has to live here, in a replacement
for the top wrapper.

On done, the wrapper moves the env's level (``curriculum_step``), picks a
spawn on the new tile, and writes that pose into the restored cached state.
It reads and updates the ``terrain_*`` info keys the env creates at reset.

``wrap_for_terrain_brax_training`` has the same signature as playground's
``wrap_for_brax_training``, so train.py can pass either one. ``full_reset``
is refused rather than ignored: the teleport IS this wrapper's reset, and
swallowing the flag would read as supporting both modes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jp
from brax.envs.wrappers import training as brax_training
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.wrapper import (
    BraxDomainRandomizationVmapWrapper,
    Wrapper,
)

from wojtek_rl import height_scan, symmetry, terrain_env


def _component_slice(names, target):
    """Where `target` sits in a concatenated observation, or None."""
    offset = 0
    for name in names:
        size = symmetry.COMPONENT_SIZES[name]
        if name == target:
            return slice(offset, offset + size)
        offset += size
    return None


def wrap_for_terrain_brax_training(
    env: mjx_env.MjxEnv,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Callable[[mjx.Model], tuple[mjx.Model, mjx.Model]] | None = None,
    full_reset: bool = False,
) -> Wrapper:
    """Playground training stack with the terrain curriculum auto-reset."""
    if full_reset:
        raise ValueError(
            "wrap_for_terrain_brax_training does not support full_reset: the "
            "curriculum teleport replaces the reset. Use playground's "
            "wrap_for_brax_training if a true full reset is wanted."
        )
    if randomization_fn is None:
        env = brax_training.VmapWrapper(env)
    else:
        env = BraxDomainRandomizationVmapWrapper(env, randomization_fn)
    env = brax_training.EpisodeWrapper(env, episode_length, action_repeat)
    env = TerrainAutoResetWrapper(env)
    return env


class TerrainAutoResetWrapper(Wrapper):
    """Restores the cached first state on done, like ``BraxAutoResetWrapper``,
    then moves the base to the new curriculum spawn. Every cached observation
    but the height scan stays correct after the move: nothing else carries
    world position or heading. The scan is recomputed for the new pose below,
    and only when the env has one."""

    def __init__(self, env: Any):
        super().__init__(env)
        base = env.unwrapped
        self._base = base
        self._origin_xy = base._terrain.origin_xy
        self._pad_h = base._terrain.pad_h
        self._n_rows = base._terrain.n_rows
        self._tile_size = base._terrain.tile_size
        self._pad_jitter = base._terrain.pad_jitter
        self._spawn_yaw = base._terrain.spawn_yaw
        self._demote_fraction = base._terrain.demote_fraction
        # v4.2 curriculum state (all inert at the legacy defaults):
        # feature spawns, band promotion, strike demotion, pinned coverage.
        self._spawn_mode = base._terrain.spawn_mode
        self._feature_r = base._terrain.feature_r
        self._pad_radius = base._terrain.pad_radius
        self._flat_row = base._terrain.flat_row
        self._height = base._terrain.height
        self._grace_steps = int(round(base._terrain.spawn_grace_sec / base.dt))
        self._demote_strikes = base._terrain.demote_strikes
        self._pinned_frac = base._terrain.pinned_frac
        # For reseeding the no-progress meter on respawn (see step below).
        self._cmd_speed = base._cmd_speed
        # EpisodeWrapper's length, for projecting the demote threshold onto a
        # full episode. Without one below us there are no truncation dones and
        # no "steps" key either, so the env's own length is the right fallback.
        self._episode_length = int(
            getattr(env, "episode_length", base._config.episode_length)
        )
        # Static, so an env without a live scan adds no scan work to its step.
        self._scan_live = bool(getattr(base, "_scan_live", False))
        if self._scan_live:
            hs = base._config.height_scan
            self._scan_corrupt = hs.corrupt
            self._hold_steps = int(hs.hold_steps)
            self._quat_adr = int(base._sensor_adr["orientation"])
            self._actor_scan = (
                None if hs.dark
                else _component_slice(base.actor_obs_names, "height_scan")
            )
            self._critic_scan = _component_slice(
                base._config.obs.privileged, "height_scan_clean"
            )
            self._scan_mirror = (
                jp.array(height_scan.mirror_map()[0])
                if base._config.symmetry.enable else None
            )

    def reset(self, rng: jax.Array) -> mjx_env.State:
        state = self.env.reset(rng)
        state.info["first_data"] = state.data
        state.info["first_obs"] = state.obs
        return state

    def _respawn(self, done, level, ttype, spawn_xy, last_xy, commanded, spawn_h,
                 steps_lived, cached_qpos, rng, strikes, cheby_min, cheby_max,
                 pinned, pinned_level):
        """One env's teleport. Everything returned is gated on ``done``; a
        live env keeps its values."""
        walked = jp.linalg.norm(last_xy - spawn_xy)
        crossed = None
        if self._spawn_mode == "feature":
            # Promotion means crossing the tile's feature band: the episode's
            # Chebyshev range around the tile centre reached inside the
            # platform AND past the feature's outer ring. The flat row has no
            # band, so it keeps the legacy walked-distance rule.
            f_r = self._feature_r[level, ttype]
            band = (
                (cheby_min <= self._pad_radius) & (cheby_max >= f_r) & (f_r > 0)
            )
            on_flat = self._flat_row & (level == 0)
            crossed = jp.where(on_flat, walked > 0.5 * self._tile_size, band)
        new_level, new_strikes, rng2 = terrain_env.curriculum_step(
            level, walked, commanded, steps_lived, self._episode_length, rng,
            self._n_rows, self._tile_size, self._demote_fraction,
            strikes=strikes, demote_strikes=self._demote_strikes,
            crossed=crossed, grace_steps=self._grace_steps, pinned=pinned,
        )
        # A pinned env holds its own rung, whatever level its first reset
        # happened to draw; its first respawn moves it there for good.
        new_level = jp.where(pinned, pinned_level, new_level)
        rng2, r_spawn = jax.random.split(rng2)
        if self._spawn_mode == "feature":
            on_flat_new = self._flat_row & (new_level == 0)
            new_xy, pad_h, quat = terrain_env.sample_feature_spawn(
                r_spawn, ttype, new_level, on_flat_new,
                self._origin_xy, self._pad_h, self._pad_jitter,
                self._tile_size, self._spawn_yaw,
            )
            ground = jp.where(on_flat_new, pad_h, self._height(new_xy))
        else:
            new_xy, ground, quat = terrain_env.sample_tile_spawn(
                r_spawn, ttype, new_level,
                self._origin_xy, self._pad_h, self._pad_jitter, self._spawn_yaw,
            )
        qpos = cached_qpos
        qpos = qpos.at[0:2].set(jp.where(done, new_xy, qpos[0:2]))
        qpos = qpos.at[2].set(jp.where(done, ground + spawn_h, qpos[2]))
        qpos = qpos.at[3:7].set(jp.where(done, quat, qpos[3:7]))
        level_out = jp.where(done, new_level, level)
        spawn_out = jp.where(done, new_xy, spawn_xy)
        strikes_out = jp.where(done, new_strikes, strikes)
        rng_out = jp.where(done, rng2, rng)
        origin_out = self._origin_xy[level_out, ttype]
        r0 = jp.max(jp.abs(spawn_out - origin_out))
        return qpos, level_out, spawn_out, rng_out, strikes_out, origin_out, r0

    def _scan_pose(self, teleport_data, new_qpos, cached_qpos):
        """The teleported pose as the scan reads it.

        The next physics step is what recomputes kinematics and sensors, so
        the base orientation and the foot heights the scan needs are written
        here. Spawn quaternions are yaw-only, and a yaw rotation about the
        base leaves every geom's height above it unchanged, so shifting z by
        the base's own shift is exact.
        """
        adr = self._quat_adr
        dz = (new_qpos[:, 2] - cached_qpos[:, 2])[:, None]
        return teleport_data.replace(
            sensordata=teleport_data.sensordata.at[:, adr : adr + 4].set(
                new_qpos[:, 3:7]
            ),
            geom_xpos=teleport_data.geom_xpos.at[:, :, 2].add(dz),
        )

    def _respawn_scan(self, info, done, scan_data):
        """Redraw the episode's sensor corruption and refill the camera
        buffers from the new spawn. Returns the actor and clean scans there,
        for the observation this step serves."""
        keys = jax.vmap(lambda k: jax.random.split(k, 4))(info["scan_rng"])
        info["scan_rng"] = keys[:, 0]
        c = self._scan_corrupt
        regime, drift, cam_jit = jax.vmap(
            lambda k: height_scan.sample_corruption(
                k, c.noise_prob, c.drift_prob, c.blackout_prob,
                c.drift_z, c.drift_tilt, c.pitch_jitter_deg, c.mount_jitter,
            )
        )(keys[:, 1])
        regime = jp.where(done, regime, info["scan_regime"])
        drift = jp.where(done[:, None], drift, info["scan_drift"])
        cam_jit = jp.where(done[:, None], cam_jit, info["scan_cam_jit"])
        phase = jax.vmap(
            lambda k: jax.random.randint(k, (), 0, self._hold_steps)
        )(keys[:, 2])
        scan = jax.vmap(self._base._scan_actor)(
            scan_data, keys[:, 3], regime, drift, cam_jit
        )
        clean = jax.vmap(lambda d: self._base._scan_raw(d)[0])(scan_data)
        info["scan_regime"] = regime
        info["scan_drift"] = drift
        info["scan_cam_jit"] = cam_jit
        info["scan_step"] = jp.where(done, phase, info["scan_step"])
        info["scan_hold"] = jp.where(done[:, None], scan, info["scan_hold"])
        info["scan_pending"] = jp.where(
            done[:, None], scan, info["scan_pending"]
        )
        return scan, clean

    def _splice_scan(self, vector, where, values, done, mirror):
        """Write `values` into one observation component for the done envs."""
        if self._scan_mirror is not None:
            values = jp.where(
                mirror[:, None], values[:, self._scan_mirror], values
            )
        return vector.at[:, where].set(
            jp.where(done[:, None], values, vector[:, where])
        )

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        reset_data = state.info["first_data"]
        reset_obs = state.info["first_obs"]
        if "steps" in state.info:
            steps = jp.where(
                state.done, jp.zeros_like(state.info["steps"]), state.info["steps"]
            )
            state.info.update(steps=steps)
        state = state.replace(done=jp.zeros_like(state.done))
        state = self.env.step(state, action)
        done = state.done  # fall or timeout; both are set below this wrapper

        info = state.info
        # Steps in the episode that just ended: zeroed on done above, before
        # EpisodeWrapper incremented it for this step. A timeout therefore holds
        # exactly episode_length here.
        steps_lived = info.get(
            "steps", jp.full_like(done, self._episode_length, dtype=jp.int32)
        )
        # The pinned coverage slice: the first pinned_frac of the batch sits
        # one env per rung, round-robin, and never rides the ladder. Batch
        # size is trace-time static, so this costs nothing per step.
        n_envs = done.shape[0]
        if self._pinned_frac > 0:
            idx = jp.arange(n_envs)
            pinned = idx < int(round(self._pinned_frac * n_envs))
            pinned_level = idx % self._n_rows
        else:
            pinned = jp.zeros(n_envs, dtype=bool)
            pinned_level = jp.zeros(n_envs, dtype=jp.int32)
        (
            new_qpos, new_level, new_spawn_xy, new_rng,
            new_strikes, new_origin, new_r0,
        ) = jax.vmap(self._respawn)(
            done,
            info["terrain_level"],
            info["terrain_type"],
            info["spawn_xy"],
            info["last_xy"],
            info["commanded_dist"],
            info["spawn_height"],
            steps_lived,
            reset_data.qpos,
            info["terrain_rng"],
            info["curriculum_strikes"],
            info["cheby_min"],
            info["cheby_max"],
            pinned,
            pinned_level,
        )
        teleport_data = reset_data.replace(qpos=new_qpos)

        def where_done(x, y):
            d = done
            if d.shape and d.shape[0] != x.shape[0]:
                return y
            if d.shape:
                d = jp.reshape(d, [x.shape[0]] + [1] * (len(x.shape) - 1))
            return jp.where(d, x, y)

        data = jax.tree.map(where_done, teleport_data, state.data)
        obs = jax.tree.map(where_done, reset_obs, state.obs)
        if self._scan_live:
            scan, clean = self._respawn_scan(
                info, done, self._scan_pose(teleport_data, new_qpos, reset_data.qpos)
            )
            if self._actor_scan is not None:
                obs["state"] = self._splice_scan(
                    obs["state"], self._actor_scan, scan, done, info["mirror"]
                )
            if self._critic_scan is not None:
                obs["privileged_state"] = self._splice_scan(
                    obs["privileged_state"], self._critic_scan, clean, done,
                    info["mirror"],
                )

        info["terrain_level"] = new_level
        info["spawn_xy"] = new_spawn_xy
        info["last_xy"] = jp.where(done[:, None], new_spawn_xy, info["last_xy"])
        info["commanded_dist"] = jp.where(
            done, jp.zeros_like(info["commanded_dist"]), info["commanded_dist"]
        )
        info["terrain_rng"] = new_rng
        info["curriculum_strikes"] = new_strikes
        info["tile_origin"] = jp.where(
            done[:, None], new_origin, info["tile_origin"]
        )
        info["cheby_min"] = jp.where(done, new_r0, info["cheby_min"])
        info["cheby_max"] = jp.where(done, new_r0, info["cheby_max"])
        if "progress_ema" in info:
            # A respawn keeps the dead episode's command and clocks (the
            # auto-reset contract here), so the no-progress meter restarts
            # by hand: reseed to ratio 1, exactly like reset does. Left
            # stale, a cut env would respawn with a sub-threshold EMA and
            # no grace (steps_since_cmd carries over) and be cut again
            # within a second.
            demand = jax.vmap(self._cmd_speed)(info["command"])
            info["progress_ema"] = jp.where(done, demand, info["progress_ema"])
        return state.replace(data=data, obs=obs, info=info)
