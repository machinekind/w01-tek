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

from wojtek_rl import terrain_env


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
    then moves the base to the new curriculum spawn. The cached obs stays
    correct after the move: no observation contains world position or
    heading."""

    def __init__(self, env: Any):
        super().__init__(env)
        base = env.unwrapped
        self._origin_xy = base._terrain.origin_xy
        self._pad_h = base._terrain.pad_h
        self._n_rows = base._terrain.n_rows
        self._tile_size = base._terrain.tile_size
        self._pad_jitter = base._terrain.pad_jitter
        self._spawn_yaw = base._terrain.spawn_yaw
        self._demote_fraction = base._terrain.demote_fraction
        # For reseeding the no-progress meter on respawn (see step below).
        self._cmd_speed = base._cmd_speed
        # EpisodeWrapper's length, for projecting the demote threshold onto a
        # full episode. Without one below us there are no truncation dones and
        # no "steps" key either, so the env's own length is the right fallback.
        self._episode_length = int(
            getattr(env, "episode_length", base._config.episode_length)
        )

    def reset(self, rng: jax.Array) -> mjx_env.State:
        state = self.env.reset(rng)
        state.info["first_data"] = state.data
        state.info["first_obs"] = state.obs
        return state

    def _respawn(self, done, level, ttype, spawn_xy, last_xy, commanded, spawn_h,
                 steps_lived, cached_qpos, rng):
        """One env's teleport. Everything returned is gated on ``done``; a
        live env keeps its values."""
        walked = jp.linalg.norm(last_xy - spawn_xy)
        new_level, rng2 = terrain_env.curriculum_step(
            level, walked, commanded, steps_lived, self._episode_length, rng,
            self._n_rows, self._tile_size, self._demote_fraction,
        )
        rng2, r_spawn = jax.random.split(rng2)
        new_xy, pad_h, quat = terrain_env.sample_tile_spawn(
            r_spawn, ttype, new_level,
            self._origin_xy, self._pad_h, self._pad_jitter, self._spawn_yaw,
        )
        qpos = cached_qpos
        qpos = qpos.at[0:2].set(jp.where(done, new_xy, qpos[0:2]))
        qpos = qpos.at[2].set(jp.where(done, pad_h + spawn_h, qpos[2]))
        qpos = qpos.at[3:7].set(jp.where(done, quat, qpos[3:7]))
        level_out = jp.where(done, new_level, level)
        spawn_out = jp.where(done, new_xy, spawn_xy)
        rng_out = jp.where(done, rng2, rng)
        return qpos, level_out, spawn_out, rng_out

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
        new_qpos, new_level, new_spawn_xy, new_rng = jax.vmap(self._respawn)(
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

        info["terrain_level"] = new_level
        info["spawn_xy"] = new_spawn_xy
        info["last_xy"] = jp.where(done[:, None], new_spawn_xy, info["last_xy"])
        info["commanded_dist"] = jp.where(
            done, jp.zeros_like(info["commanded_dist"]), info["commanded_dist"]
        )
        info["terrain_rng"] = new_rng
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
