"""JAX terrain helpers shared by the env and the curriculum wrapper.

Everything here reads the generated terrain sidecars (``terrain_lookup.npz``
and ``terrain_spec.json``, written by ``./run.sh build-terrain``). The module
stays free of brax so base.py can import it without the training stack.

``bilinear_sample`` reproduces ``terrain._bilinear`` exactly, clamped edges
included, so heights sampled in the env match the lookup grid the physics
geometry was validated against.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
import numpy as np


def bilinear_sample(grid, x_min, cell_x, y_min, cell_y, x, y):
    """Bilinear sample of ``grid`` (rows index y, cols index x) at world
    ``(x, y)``, clamped to the grid edges. Matches ``terrain._bilinear``
    exactly."""
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
    """Per-(row, terrain-type) tile-origin xy and pad height from the spec dict.

    The tables are indexed by terrain type because the arena shuffles the
    column order per row. Both arrays are ``(n_rows, n_types)``, aligned to
    the ``types`` order."""
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
    """Spawn pose on the ``(level, terrain_type)`` tile.

    Returns the pad centre plus xy jitter, the tile's pad height, and a base
    yaw quaternion. ``pad_jitter`` (metres) and ``yaw_enable`` are static
    Python values, so the function vmaps over ``(rng, terrain_type, level)``.
    The jitter is small enough that the reset settle transient stays on the
    flat pad."""
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
    """legged_gym promote/demote for one finished episode.

    An episode that walked more than half a tile edge promotes one level. An
    episode that covered less than ``demote_fraction`` of its commanded
    distance demotes one. Anything else holds. A standing episode has a
    near-zero commanded distance, clears neither threshold, and holds too.
    Levels clip to ``[0, n_rows-1]``. A promotion from the top level lands on
    a uniformly random row instead, so easy terrain keeps being trained. The
    rng advances on every call, so the caller can gate the whole result on
    ``done`` with one where."""
    promote = walked > 0.5 * tile_size
    demote = walked < demote_fraction * commanded_dist
    delta = jp.where(promote, 1, jp.where(demote, -1, 0))
    stepped = jp.clip(level + delta, 0, n_rows - 1)
    at_max = level >= (n_rows - 1)
    rng, sub = jax.random.split(rng)
    rand_row = jax.random.randint(sub, (), 0, n_rows)
    new_level = jp.where(promote & at_max, rand_row, stepped)
    return new_level, rng
