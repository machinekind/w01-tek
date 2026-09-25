"""Batched MJX replay of recorded commands under candidate parameters.

One losses() call evaluates a whole CMA-ES population: the model is
vmapped over candidates (space.batched_model) and the rollout over
windows, so P candidates x W windows step as one batch. Latency is
applied as a fractional shift of the command stream on the control grid
(linear interpolation between neighboring commands), which keeps the
delay parameter continuous for the optimizer.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from mujoco import mjx

from wojtek_rl.base import data_budget_kwargs, resolve_backend

# Fitness assigned to diverged rollouts (NaN/inf). RMS position error is
# O(0.01..1) rad, so this dominates without breaking CMA-ES ranking.
DIVERGED = 1e3


@dataclass
class Evaluator:
    losses: callable  # (P, dim) genomes -> (P,) RMS position error, rad
    rollout: callable  # (dim,) genome -> (W, T, nu) simulated joint positions
    backend: str
    n_windows: int
    n_steps: int


def make_evaluator(mj_model, space, ds, backend="auto", popsize=64):
    """Build the population evaluator for one dataset.

    popsize only sizes the warp backend's shared contact pool (jax sizes
    its buffers on the fly); losses() accepts any population size, but on
    warp it should not exceed the popsize given here.
    """
    backend = resolve_backend(backend)
    mjx_model = mjx.put_model(mj_model, impl=backend)
    W, T, _ = ds.cmd.shape
    kwargs = data_budget_kwargs(backend, 32, 320, popsize * W)
    if backend == "warp":
        data0 = mjx.make_data(mj_model, impl="warp", **kwargs)
    else:
        data0 = mjx.make_data(mjx_model)

    cmd = jnp.asarray(ds.cmd)
    meas = jnp.asarray(ds.meas)
    qpos0 = jnp.asarray(ds.qpos0)
    qvel0 = jnp.asarray(ds.qvel0)
    qadr = jnp.asarray(space.qadr)
    n_sub = ds.n_substeps
    wu = ds.warmup_steps

    def delayed(cmd_w, latency):
        t = jnp.arange(T, dtype=jnp.float32)
        src = t - latency
        lo = jnp.clip(jnp.floor(src), 0, T - 1)
        frac = jnp.clip(src - lo, 0.0, 1.0)[:, None]
        lo = lo.astype(jnp.int32)
        hi = jnp.clip(lo + 1, 0, T - 1)
        return cmd_w[lo] * (1.0 - frac) + cmd_w[hi] * frac

    def traj(model, latency, qpos0_w, qvel0_w, cmd_w):
        d = data0.replace(qpos=qpos0_w, qvel=qvel0_w, ctrl=cmd_w[0])
        d = mjx.forward(model, d)

        def step(d, ctrl):
            d = d.replace(ctrl=ctrl)
            d = jax.lax.fori_loop(0, n_sub, lambda _, dd: mjx.step(model, dd), d)
            return d, d.qpos[qadr]

        _, q = jax.lax.scan(step, d, delayed(cmd_w, latency))
        return q  # (T, nu)

    def window_mse(model, latency, qpos0_w, qvel0_w, cmd_w, meas_w):
        e = (traj(model, latency, qpos0_w, qvel0_w, cmd_w) - meas_w)[wu:]
        return jnp.mean(e * e)

    per_window = jax.vmap(window_mse, in_axes=(None, None, 0, 0, 0, 0))

    @jax.jit
    def losses(U):
        model_v, in_axes, latency = space.batched_model(mjx_model, U)
        mse = jax.vmap(
            lambda m, lat: jnp.mean(per_window(m, lat, qpos0, qvel0, cmd, meas)),
            in_axes=(in_axes, 0),
        )(model_v, latency)
        rmse = jnp.sqrt(mse)
        return jnp.where(jnp.isfinite(rmse), rmse, DIVERGED)

    @jax.jit
    def rollout(u):
        model, latency = space.apply(mjx_model, u)
        return jax.vmap(traj, in_axes=(None, None, 0, 0, 0))(
            model, latency, qpos0, qvel0, cmd
        )

    return Evaluator(
        losses=losses, rollout=rollout, backend=backend, n_windows=W, n_steps=T
    )
