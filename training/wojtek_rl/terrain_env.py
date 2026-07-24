"""JAX helpers for the terrain arena: height lookup, spawn tables, curriculum.

Everything reads the files written by ``./run.sh build-terrain``. No brax
imports here, so base.py can use it without the training stack.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
import numpy as np


def bilinear_sample(grid, x_min, cell_x, y_min, cell_y, x, y):
    """Sample ``grid`` at world ``(x, y)``. Same math as ``terrain._bilinear``,
    clamped at the edges. Rows index y, columns index x."""
    nr, nc = grid.shape
    fx = jp.clip((x - x_min) / cell_x, 0.0, nc - 1)
    fy = jp.clip((y - y_min) / cell_y, 0.0, nr - 1)
    cx0 = jp.floor(fx).astype(jp.int32)
    ry0 = jp.floor(fy).astype(jp.int32)
    cx1 = jp.minimum(cx0 + 1, nc - 1)
    ry1 = jp.minimum(ry0 + 1, nr - 1)
    tx = fx - cx0
    ty = fy - ry0
    g00 = grid[ry0, cx0]
    g01 = grid[ry0, cx1]
    g10 = grid[ry1, cx0]
    g11 = grid[ry1, cx1]
    return (
        g00 * (1 - tx) * (1 - ty)
        + g01 * tx * (1 - ty)
        + g10 * (1 - tx) * ty
        + g11 * tx * ty
    )


def tables_from_spec(spec: dict, types) -> tuple[np.ndarray, np.ndarray]:
    """Tile-centre xy and pad height for every (row, type) pair.

    Indexed by type, not column, because each row shuffles its column order.
    Shapes: ``(n_rows, n_types, 2)`` and ``(n_rows, n_types)``."""
    n_rows = int(spec["n_rows"])
    type_idx = {t: i for i, t in enumerate(types)}
    origin_xy = np.zeros((n_rows, len(types), 2), dtype=np.float32)
    pad_h = np.zeros((n_rows, len(types)), dtype=np.float32)
    for t in spec["tiles"]:
        r = int(t["row"])
        j = type_idx[t["type"]]
        origin_xy[r, j] = (t["origin"][0], t["origin"][1])
        pad_h[r, j] = t["pad_height"]
    return origin_xy, pad_h


def sample_tile_spawn(rng, terrain_type, level, origin_xy, pad_h, pad_jitter, yaw_enable):
    """Pick a spawn pose on one tile.

    Returns the pad centre plus a small xy jitter, the pad's height, and a
    yaw quaternion. ``pad_jitter`` and ``yaw_enable`` are plain Python
    values, so the function can be vmapped over (rng, terrain_type, level)."""
    xy0 = origin_xy[level, terrain_type]
    ph = pad_h[level, terrain_type]
    rj, ry = jax.random.split(rng)
    jitter = jax.random.uniform(rj, (2,), minval=-pad_jitter, maxval=pad_jitter)
    spawn_xy = xy0 + jitter
    if yaw_enable:
        yaw = jax.random.uniform(ry, minval=-jp.pi, maxval=jp.pi)
        quat = jp.array([jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)])
    else:
        quat = jp.array([1.0, 0.0, 0.0, 0.0])
    return spawn_xy, ph, quat


def curriculum_step(level, walked, commanded_dist, rng, n_rows, tile_size, demote_fraction):
    """Move one env's level after an episode, legged_gym style.

    Walked more than half a tile: one level up. Covered less than
    ``demote_fraction`` of the commanded distance: one level down. Otherwise
    stay, which covers standing episodes too. Levels stay inside
    ``[0, n_rows-1]``, except going up from the top level lands on a random
    row, so easy terrain stays in training. The rng always advances, so the
    caller can gate everything on ``done`` with one where."""
    promote = walked > 0.5 * tile_size
    demote = walked < demote_fraction * commanded_dist
    delta = jp.where(promote, 1, jp.where(demote, -1, 0))
    stepped = jp.clip(level + delta, 0, n_rows - 1)
    at_max = level >= (n_rows - 1)
    rng, sub = jax.random.split(rng)
    rand_row = jax.random.randint(sub, (), 0, n_rows)
    new_level = jp.where(promote & at_max, rand_row, stepped)
    return new_level, rng
