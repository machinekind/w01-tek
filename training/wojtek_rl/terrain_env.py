"""JAX-side terrain helpers for the terrain-aware joystick env.

Pure functions and table builders shared by the env (base.py/env.py) and the
curriculum auto-reset wrapper (terrain_wrapper.py). Kept free of brax so
base.py can import it without pulling the training stack in.

The lookup grid, tile-origin table and pad-height table all come from the
generated terrain sidecars (``terrain_lookup.npz`` + ``terrain_spec.json``,
written by ``./run.sh build-terrain``). ``bilinear_sample`` reproduces
``terrain.lookup_height`` / ``terrain._bilinear`` node-for-node with clamped
edges, so a foot's terrain height in the env matches the physics geometry the
same lookup was validated against.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
import numpy as np


def bilinear_sample(grid, x_min, cell_x, y_min, cell_y, x, y):
    """Clamped bilinear sample of ``grid`` (row indexes y, col indexes x) at
    world ``(x, y)``. Matches ``terrain._bilinear`` with ``dx=cell_x`` exactly:
    fractional indices clamp to the grid, then the four corner nodes blend."""
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

    Indexed by terrain TYPE, not column: the arena shuffles the column order
    per row, so a fixed terrain type sits in a different column each row. The
    returned arrays are ``(n_rows, n_types)`` shaped, aligned to ``types``
    order (``terrain.TYPES``)."""
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
    """Spawn pose on the ``(level, terrain_type)`` tile: pad centre plus xy
    jitter, the tile's pad height, and a base yaw quaternion.

    ``terrain_type`` and ``level`` are integer scalars; ``origin_xy`` is
    ``(n_rows, n_types, 2)`` and ``pad_h`` ``(n_rows, n_types)``. ``pad_jitter``
    (metres) and ``yaw_enable`` are static, closed over by the caller so this
    stays vmap-friendly on ``(rng, terrain_type, level)``. Jitter stays inside
    ``pad_jitter`` of the centre so the settle transient never leaves the pad;
    yaw is uniform about +z when enabled, else identity."""
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
    """legged_gym promote/demote for one finished episode. Returns
    ``(new_level, rng_out)``.

    Promote (+1) when the robot crossed its tile — walked more than half a tile
    edge; demote (-1) when it covered less than ``demote_fraction`` of the
    distance its command asked for; otherwise hold. Levels clip to
    ``[0, n_rows-1]``. An env promoting from the top level respawns on a
    uniformly random row instead, so easy terrain is not forgotten. A standing
    episode (``commanded_dist`` ~ 0) clears neither threshold, so it is
    neutral. ``rng`` advances whether or not the random-row branch is taken, so
    the caller can gate the whole draw on ``done`` with a single where."""
    promote = walked > 0.5 * tile_size
    demote = walked < demote_fraction * commanded_dist
    delta = jp.where(promote, 1, jp.where(demote, -1, 0))
    stepped = jp.clip(level + delta, 0, n_rows - 1)
    at_max = level >= (n_rows - 1)
    rng, sub = jax.random.split(rng)
    rand_row = jax.random.randint(sub, (), 0, n_rows)
    new_level = jp.where(promote & at_max, rand_row, stepped)
    return new_level, rng
