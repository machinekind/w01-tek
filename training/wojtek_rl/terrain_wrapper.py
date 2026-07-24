"""Curriculum auto-reset wrapper for the terrain-aware joystick env.

The playground / brax training stack builds
``DR-vmap|Vmap -> Episode -> BraxAutoResetWrapper``. Its auto-reset restores a
cached first state on done and never re-runs ``env.reset``, and truncation-done
is only set inside ``EpisodeWrapper`` — so a curriculum that must react to every
episode end (fall or timeout) and teleport the robot to a new tile has to live
in a replacement for the top auto-reset wrapper.

``wrap_for_terrain_brax_training`` builds the same stack but swaps in
``TerrainAutoResetWrapper``. On done it applies the legged_gym promote/demote
rule per env, samples a fresh spawn on the env's terrain type at the new level,
and overwrites the cached first state's base pose (xy, z, yaw) for the done
envs — the cached qvel is already zeros. Everything it needs rides in the env's
info (see ``env.py`` reset): ``terrain_type``, ``terrain_level``, ``spawn_xy``,
``spawn_height``, ``last_xy``, ``commanded_dist``, ``terrain_rng``. It reads the
tile tables and curriculum params off the unwrapped env, closed over once at
construction.

The signature matches ``mujoco_playground.wrapper.wrap_for_brax_training`` so
train.py can pass it as ``wrap_env_fn``. ``full_reset`` is accepted for
signature parity but ignored: the terrain teleport is the reset.
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
    del full_reset  # the terrain respawn is the reset
    if randomization_fn is None:
        env = brax_training.VmapWrapper(env)
    else:
        env = BraxDomainRandomizationVmapWrapper(env, randomization_fn)
    env = brax_training.EpisodeWrapper(env, episode_length, action_repeat)
    env = TerrainAutoResetWrapper(env)
    return env


class TerrainAutoResetWrapper(Wrapper):
    """Auto-reset that promotes/demotes and teleports on every episode end.

    Mirrors ``BraxAutoResetWrapper`` (cache first data/obs at reset, restore on
    done) but, for done envs, rewrites the restored base pose to a fresh
    curriculum spawn instead of the original one. The cached obs is reused
    unchanged because it carries no world pose (gravity is yaw-invariant), which
    is exactly what makes teleporting valid."""

    def __init__(self, env: Any):
        super().__init__(env)
        base = env.unwrapped
        self._origin_xy = base._terrain_origin_xy
        self._pad_h = base._terrain_pad_h
        self._n_rows = base._terrain_n_rows
        self._tile_size = base._terrain_tile_size
        self._pad_jitter = base._terrain_pad_jitter
        self._spawn_yaw = base._terrain_spawn_yaw
        self._demote_fraction = base._terrain_demote_fraction

    def reset(self, rng: jax.Array) -> mjx_env.State:
        state = self.env.reset(rng)
        state.info["first_data"] = state.data
        state.info["first_obs"] = state.obs
        return state

    def _respawn(self, done, level, ttype, spawn_xy, last_xy, commanded, spawn_h,
                 cached_qpos, rng):
        """Per-env teleport. Returns the new base-pose qpos row, the new level,
        the new spawn xy, and the advanced rng — each gated on ``done`` so a
        surviving env keeps its level, spawn and rng untouched."""
        walked = jp.linalg.norm(last_xy - spawn_xy)
        new_level, rng2 = terrain_env.curriculum_step(
            level, walked, commanded, rng,
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
        done = state.done  # fall OR truncation, both visible above EpisodeWrapper

        info = state.info
        new_qpos, new_level, new_spawn_xy, new_rng = jax.vmap(self._respawn)(
            done,
            info["terrain_level"],
            info["terrain_type"],
            info["spawn_xy"],
            info["last_xy"],
            info["commanded_dist"],
            info["spawn_height"],
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
        return state.replace(data=data, obs=obs, info=info)
