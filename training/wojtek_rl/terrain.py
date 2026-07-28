"""Procedural terrain arena for Wojtek locomotion training.

One shared arena, deterministic from a seed: a grid of ~3 m tiles with rows of
increasing difficulty (0..1) and one tile of every terrain type per row, plus a
flat border around the grid. By default each row's column order is shuffled and
each interior row's difficulty is jittered within half a row gap; ``ordered=True``
restores the sorted-column, exact-difficulty layout, and ``difficulties=`` gives
the rows explicitly (what the measurement arena in ``terrain_suite`` does).
``flat_row=True`` prepends one flat row below all of them, as level 0.
Rough, slopes, and the inverted-stairs pit are carved into a single heightfield
covering the whole arena; stairs and discrete obstacles are static box geoms
(crisp edges, cheap contacts) sitting on the heightfield. A modality overlay of
low rough noise rides the slope and box-tile grounds so those tiles are not
perfectly smooth. Every tile reaches flat ground (z = 0) at its border so
neighbours join seamlessly, and every tile has a flat circular spawn pad at its
centre.

The heightfield geom and all box geoms live in the worldbody, so MuJoCo's
same-body exclusion drops every terrain-vs-terrain contact for free; only the
robot pairs with the terrain.

A dense lookup grid stores the ground-truth surface height (the heightfield
with the boxes rasterised in) on the same node grid as the heightfield. Later
PRs read terrain-relative heights and scandots from it by bilinear
interpolation, with no raycast (raycasts do not exist on the JAX backend), so
it must match the physics geometry node-for-node.

MuJoCo normalises heightfield data to [0, 1] on compile: the physical surface is
``pos_z + data * elevation_z``. We store data already normalised to its own
full range and set ``elevation_z`` to the height range and ``pos_z`` to the
minimum height, which reproduces the intended heights exactly (checked against
mj_ray in the tests).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Column order of the arena; one column per terrain type.
TYPES: tuple[str, ...] = (
    "rough_uniform",
    "pyramid_slope",
    "inverted_pyramid_slope",
    "pyramid_stairs",
    "inverted_pyramid_stairs",
    "discrete_obstacles",
    "random_grid",
    "wave",
)

DEFAULT_SEED = 0
DEFAULT_N_ROWS = 10
TILE_SIZE = 3.0
BORDER = 2.0
# 0.04 m keeps the elevation file and lookup npz in the low single MB while
# still resolving a 3 cm stair riser. It has nothing to do with the MJWarp
# per-pair contact cap: a GPU run on 2026-07-25 measured the base box over the
# cap at this cell size. Splitting the base collision box (build_model.py) is
# what keeps each geom pair under it.
CELL_SIZE = 0.04
# Thickness of the heightfield's solid base, measured downward from the geom
# frame -- which the scene generator places at the arena's minimum height
# (pos_z = hmin), so the solid already extends below the deepest pit whatever
# this value is. The only real constraint is that it is positive, and the
# MJCF compiler enforces that; pit depth never has to be added here.
HFIELD_BASE_Z = 1.0

# Flat circular spawn pad at every tile centre (>= 0.5 m per the brief). The
# measurement arena passes a smaller radius: spawn jitter is zero there, so the
# pad only has to hold the standing footprint.
PAD_RADIUS = 0.6
# A standing robot's reach: the feet sit 0.31 m from the base centre (0.257
# fore/aft, 0.174 lateral) and the foot radius is 0.046 m. This much flat ground
# under the base, plus whatever spawn jitter is in play, is what a spawn needs;
# it is the floor under every pad radius in this file.
FOOTPRINT_REACH = 0.36
# Rough noise ramps from 0 at the pad edge to full over this width, so the pad
# has no cliff at its rim.
PAD_TAPER = 0.25
# Rough noise and the modality overlay ramp back to exactly 0 over this width
# inward from the tile edge, so borders stay seamless. The pad taper alone left
# rough proud at the rim (a several-cm one-cell step across a row seam).
EDGE_TAPER = 0.25

# Slope tiles: flat square platform (holds the pad), then a linear ramp.
SLOPE_PLATFORM_HALF = 0.6
# Stair tiles: flat square platform, then concentric 0.13 m treads.
#
# Six risers, not the physical rig's four. N_STEPS counts risers; the flight
# lays N_STEPS - 1 treads between the platform and the tile ground. Four
# risers give 3 x 0.13 = 0.39 m of tread run against the 0.514 m fore/aft
# foot spacing (feet sit 0.257 m fore and aft of the base centre), so the
# robot always has feet on flat ground at one end or the other -- four steps
# measures entering and leaving a staircase, never walking one. Six risers
# give 0.65 m of run: a 0.136 m window with all four feet on treads. The
# five treads end 1.25 m from the tile centre on a 1.50 m half-tile; the
# platform cannot shrink below 0.51 m while spawn jitter stays at 0.15 m
# (jitter plus the 0.36 m standing footprint).
STAIR_PLATFORM_HALF = 0.6
TREAD = 0.13
N_STEPS = 6


def stair_pit_half(stair_platform_half: float = STAIR_PLATFORM_HALF) -> float:
    """Outer half-size of the stair terrace region, a CHEBYSHEV half-size: the
    terraces are concentric squares. The inverted pit is carved to this square
    and both stair types reach flat ground (0) at the rim."""
    return stair_platform_half + (N_STEPS - 1) * TREAD

# Rough tiles: white noise sampled on this coarse pitch then bilinearly
# upsampled, so features are foot-scale rolling ground, not per-cell spikes.
ROUGH_COARSE_STEP = 0.15
# Modality overlay: coarse rough noise at this fraction of the rough amplitude
# laid over slope and box-tile grounds, so those tiles are not perfectly smooth.
OVERLAY_FRACTION = 0.3

DISCRETE_N = 12
DISCRETE_HALF_RANGE = (0.10, 0.25)
DISCRETE_EDGE_MARGIN = 0.1

# random_grid (rubble): a cell grid; each cell may raise one small box.
GRID_PITCH = 0.25
GRID_FILL_PROB = 0.7
GRID_HALF_RANGE = (0.04, 0.11)

# wave: two integer half-periods per axis, so the surface is exactly 0 on all
# four tile edges (see _wave_patch).
WAVE_HALF_PERIODS = (2, 3)


def rough_amplitude(d: float) -> float:
    return 0.005 + 0.035 * d


def slope_angle(d: float) -> float:
    return 0.4 * d


def stair_riser(d: float) -> float:
    return 0.02 + 0.07 * d


def discrete_max_height(d: float) -> float:
    return 0.01 + 0.07 * d


def wave_amplitude(d: float) -> float:
    return 0.01 + 0.05 * d


# Inverses of the five ramps above. The measurement suite asks for physical
# dimensions ("a 5 cm riser") and needs the difficulty that realizes them, so
# it inverts the generator's own ramp instead of carrying a second table that
# could drift. test_terrain.py pins each pair as an exact round trip.


def rough_difficulty(amplitude: float) -> float:
    return (amplitude - 0.005) / 0.035


def slope_difficulty(angle_rad: float) -> float:
    return angle_rad / 0.4


def stair_difficulty(riser: float) -> float:
    return (riser - 0.02) / 0.07


def discrete_difficulty(height: float) -> float:
    return (height - 0.01) / 0.07


def wave_difficulty(amplitude: float) -> float:
    return (amplitude - 0.01) / 0.05


@dataclass(frozen=True)
class Box:
    """Static box: world centre, half-sizes (m), and yaw about +z (rad)."""

    pos: tuple[float, float, float]
    half: tuple[float, float, float]
    yaw: float = 0.0


@dataclass(frozen=True)
class TileSpec:
    row: int
    col: int
    terrain_type: str
    difficulty: float
    origin: tuple[float, float, float]  # tile-centre ground point (z = 0)
    pad_radius: float
    pad_height: float  # world z of the flat spawn pad at the tile centre


@dataclass(frozen=True)
class HFieldSpec:
    nrow: int  # rows index y
    ncol: int  # cols index x
    radius_x: float
    radius_y: float
    elevation_z: float
    base_z: float
    pos_z: float


@dataclass(frozen=True)
class TerrainSpec:
    seed: int
    n_rows: int  # arena rows, the prepended flat row included when there is one
    n_cols: int
    tile_size: float
    border: float
    cell_size: float
    ordered: bool
    pad_radius: float
    stair_platform_half: float
    n_steps: int
    types: tuple[str, ...]
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    hfield: HFieldSpec
    tiles: tuple[TileSpec, ...]
    flat_row: bool = False


@dataclass
class Arena:
    spec: TerrainSpec
    boxes: tuple[Box, ...]
    hfield_data: np.ndarray  # (nrow, ncol) float32 in [0, 1], row-major over y
    lookup: np.ndarray  # (nrow, ncol) float32 physical surface height (m)


def _bilinear(
    grid: np.ndarray, x0: float, dx: float, y0: float, dy: float,
    x: np.ndarray, y: np.ndarray, xp=np,
) -> np.ndarray:
    """Sample grid (row indexes y, col indexes x) at world (x, y), clamped.

    `xp` is numpy or jax.numpy: the generator and its tests sample with numpy,
    while the env samples the same grid on device (terrain_env.bilinear_sample
    binds jax.numpy). One body, so the two cannot drift -- this function is
    the ground truth that base height, foot contact and fall termination all
    read through."""
    nr, nc = grid.shape
    fx = xp.clip((x - x0) / dx, 0.0, nc - 1)
    fy = xp.clip((y - y0) / dy, 0.0, nr - 1)
    cx0 = xp.floor(fx).astype(xp.int32)
    ry0 = xp.floor(fy).astype(xp.int32)
    cx1 = xp.minimum(cx0 + 1, nc - 1)
    ry1 = xp.minimum(ry0 + 1, nr - 1)
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


def _tile_rng(seed: int, row: int, col: int) -> np.random.Generator:
    """Independent deterministic stream per tile."""
    return np.random.default_rng([seed, row, col])


def _cheby(lx: np.ndarray, ly: np.ndarray) -> np.ndarray:
    return np.maximum(np.abs(lx), np.abs(ly))


def _pad_taper(lx: np.ndarray, ly: np.ndarray, pad_radius: float) -> np.ndarray:
    """0 inside the spawn pad, ramping to 1 by PAD_TAPER past its rim."""
    r = np.sqrt(lx**2 + ly**2)
    return np.clip((r - pad_radius) / PAD_TAPER, 0.0, 1.0)


def _edge_taper(lx: np.ndarray, ly: np.ndarray) -> np.ndarray:
    """1 in the tile interior, ramping to exactly 0 at the tile perimeter."""
    return np.clip((TILE_SIZE / 2 - _cheby(lx, ly)) / EDGE_TAPER, 0.0, 1.0)


def _coarse_noise(lx: np.ndarray, ly: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """White noise on ROUGH_COARSE_STEP pitch, bilinearly upsampled to (lx, ly)."""
    n = int(round(TILE_SIZE / ROUGH_COARSE_STEP)) + 1
    half = TILE_SIZE / 2
    coarse = rng.uniform(-1.0, 1.0, size=(n, n))
    step = TILE_SIZE / (n - 1)
    return _bilinear(coarse, -half, step, -half, step, lx, ly)


def _rough_patch(
    lx: np.ndarray, ly: np.ndarray, d: float, rng: np.random.Generator,
    pad_radius: float,
) -> np.ndarray:
    amp = rough_amplitude(d)
    return (
        amp * _coarse_noise(lx, ly, rng)
        * _pad_taper(lx, ly, pad_radius) * _edge_taper(lx, ly)
    )


def _overlay(
    lx: np.ndarray, ly: np.ndarray, d: float, rng: np.random.Generator,
    pad_radius: float,
) -> np.ndarray:
    """Coarse rough noise for slope and box-tile grounds, its own rng draw. Zero
    at the pad and at the border, so pads stay flat and borders stay seamless."""
    amp = OVERLAY_FRACTION * rough_amplitude(d)
    return (
        amp * _coarse_noise(lx, ly, rng)
        * _pad_taper(lx, ly, pad_radius) * _edge_taper(lx, ly)
    )


def slope_plateau_height(d: float) -> float:
    """Height of a slope tile's central platform above its edge (which is 0)."""
    return float(np.tan(slope_angle(d)) * (TILE_SIZE / 2 - SLOPE_PLATFORM_HALF))


def _slope_patch(lx: np.ndarray, ly: np.ndarray, d: float, sign: float) -> np.ndarray:
    # Frustum: flat platform in the middle, ramping to 0 at the tile edge, so
    # the shared border with the next tile stays seamless. sign +1 is a hill
    # (platform up), -1 a pit (platform down).
    slope = np.tan(slope_angle(d))
    reach = TILE_SIZE / 2 - np.maximum(_cheby(lx, ly), SLOPE_PLATFORM_HALF)
    return sign * slope * reach


def _pyramid_stair_boxes(
    cx: float, cy: float, d: float, stair_half: float
) -> tuple[list[Box], float]:
    """High central plateau; treads descend to ground (0) at the rim, so the
    tile border stays seamless. Boxes sit on the flat heightfield."""
    riser = stair_riser(d)
    top = N_STEPS * riser
    boxes = [Box((cx, cy, top / 2), (stair_half, stair_half, top / 2))]
    for k in range(1, N_STEPS):
        r_in = stair_half + (k - 1) * TREAD
        r_out = stair_half + k * TREAD
        boxes.extend(_frame_boxes(cx, cy, r_in, r_out, top=(N_STEPS - k) * riser))
    return boxes, top


def _pit_carve(
    lx: np.ndarray, ly: np.ndarray, d: float, stair_half: float
) -> np.ndarray:
    """Heightfield patch for an inverted-stairs pit: -H inside the pit square,
    0 outside, so the rim is flush with the tile ground."""
    depth = N_STEPS * stair_riser(d)
    return np.where(_cheby(lx, ly) <= stair_pit_half(stair_half), -depth, 0.0)


def _inverted_stair_boxes(
    cx: float, cy: float, d: float, stair_half: float
) -> tuple[list[Box], float]:
    """Terraces inside the carved pit: each ring sits on the pit floor (-H) and
    rises one riser more outward. The topmost riser (-riser -> 0) is the carve
    wall itself, beveled over at most one cell (CELL_SIZE) by the heightfield.
    Central pad is the carved floor at -H, no plateau box."""
    riser = stair_riser(d)
    depth = N_STEPS * riser
    boxes: list[Box] = []
    for j in range(1, N_STEPS):
        r_in = stair_half + (j - 1) * TREAD
        r_out = stair_half + j * TREAD
        boxes.extend(_frame_boxes(cx, cy, r_in, r_out, top=-depth + j * riser, base=-depth))
    return boxes, -depth


def _frame_boxes(
    cx: float, cy: float, r_in: float, r_out: float, top: float, base: float = 0.0
) -> list[Box]:
    mid = (r_in + r_out) / 2
    band = (r_out - r_in) / 2
    cz = (base + top) / 2
    hz = (top - base) / 2
    return [
        Box((cx, cy + mid, cz), (r_out, band, hz)),  # top strip
        Box((cx, cy - mid, cz), (r_out, band, hz)),  # bottom strip
        Box((cx - mid, cy, cz), (band, r_in, hz)),  # left strip
        Box((cx + mid, cy, cz), (band, r_in, hz)),  # right strip
    ]


def _discrete_boxes(
    cx: float, cy: float, d: float, rng: np.random.Generator, pad_radius: float
) -> list[Box]:
    max_h = discrete_max_height(d)
    # Keep obstacles clear of the tile edge so the border stays seamless with
    # the neighbour tile (a box at the rim would be a step across it).
    reach = TILE_SIZE / 2 - DISCRETE_EDGE_MARGIN
    boxes: list[Box] = []
    for _ in range(DISCRETE_N):
        hx = float(rng.uniform(*DISCRETE_HALF_RANGE))
        hy = float(rng.uniform(*DISCRETE_HALF_RANGE))
        yaw = float(rng.uniform(0.0, np.pi))
        # Clamp on the rotated footprint's AABB, so a turned box still clears
        # the tile edge by DISCRETE_EDGE_MARGIN and the border stays seamless.
        hax = hx * abs(np.cos(yaw)) + hy * abs(np.sin(yaw))
        hay = hx * abs(np.sin(yaw)) + hy * abs(np.cos(yaw))
        u = float(rng.uniform(-reach + hax, reach - hax))
        v = float(rng.uniform(-reach + hay, reach - hay))
        # Clear the pad by the box's corner radius, so no corner pokes in.
        if np.hypot(u, v) < pad_radius + np.hypot(hx, hy) + 0.05:
            continue
        h = float(rng.uniform(0.5, 1.0)) * max_h
        boxes.append(Box((cx + u, cy + v, h / 2), (hx, hy, h / 2), yaw))
    return boxes


def _random_grid_boxes(
    cx: float, cy: float, d: float, rng: np.random.Generator, pad_radius: float
) -> list[Box]:
    """Rubble: a regular cell grid, each cell raising one small yawed box with
    probability GRID_FILL_PROB. Boxes only rise above grade -- unlike Isaac's
    random grid we cannot also sink cells, since boxes cannot carve. Each box is
    kept inside its cell so neighbours never fuse into slabs, and held off the
    pad (corner rule) and the tile edge (margin) to keep borders seamless."""
    max_h = discrete_max_height(d)
    reach = TILE_SIZE / 2 - DISCRETE_EDGE_MARGIN
    n = int(2 * reach / GRID_PITCH)
    start = -(n - 1) * GRID_PITCH / 2
    half_cell = GRID_PITCH / 2
    boxes: list[Box] = []
    for iy in range(n):
        for ix in range(n):
            if rng.uniform() > GRID_FILL_PROB:
                continue
            hx = float(rng.uniform(*GRID_HALF_RANGE))
            hy = float(rng.uniform(*GRID_HALF_RANGE))
            yaw = float(rng.uniform(0.0, np.pi))
            # The box's bounding circle (radius = half-diagonal) plus its jitter
            # must stay inside the cell, whatever the yaw, so neighbours never
            # fuse. Shrink an oversized box, then jitter within the slack.
            diag = np.hypot(hx, hy)
            if diag > half_cell:
                s = 0.98 * half_cell / diag
                hx *= s
                hy *= s
                diag = np.hypot(hx, hy)
            jitter = max(0.0, half_cell - diag)
            jr = jitter * np.sqrt(rng.uniform())
            jth = rng.uniform(0.0, 2 * np.pi)
            u = start + ix * GRID_PITCH + jr * np.cos(jth)
            v = start + iy * GRID_PITCH + jr * np.sin(jth)
            if np.hypot(u, v) < pad_radius + np.hypot(hx, hy) + 0.05:
                continue
            h = float(rng.uniform(0.3, 1.0)) * max_h
            boxes.append(Box((cx + u, cy + v, h / 2), (hx, hy, h / 2), yaw))
    return boxes


def _wave_patch(
    lx: np.ndarray, ly: np.ndarray, d: float, rng: np.random.Generator,
    pad_radius: float,
) -> np.ndarray:
    # Integer half-periods in both axes make the product of sines exactly 0 on
    # all four tile edges (lx or ly = +/- half), so the border invariant holds
    # by construction. The pad is flattened with the same radial taper as rough.
    amp = wave_amplitude(d)
    half = TILE_SIZE / 2
    kx = int(rng.choice(WAVE_HALF_PERIODS))
    ky = int(rng.choice(WAVE_HALF_PERIODS))
    wave = (
        np.sin(kx * np.pi * (lx + half) / TILE_SIZE)
        * np.sin(ky * np.pi * (ly + half) / TILE_SIZE)
    )
    return amp * wave * _pad_taper(lx, ly, pad_radius)


def _footprint_nodes(
    xs: np.ndarray, ys: np.ndarray, px: float, py: float,
    hx: float, hy: float, yaw: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """(row, col) indices of grid nodes under a box: its axis-aligned cell block
    when yaw == 0, else the nodes inside the rotated rectangle (node offsets
    rotated by -yaw, tested against |u| <= hx, |v| <= hy). None if no node."""
    if yaw == 0.0:
        c0 = int(np.searchsorted(xs, px - hx))
        c1 = int(np.searchsorted(xs, px + hx, side="right"))
        r0 = int(np.searchsorted(ys, py - hy))
        r1 = int(np.searchsorted(ys, py + hy, side="right"))
        if c1 <= c0 or r1 <= r0:
            return None
        rr, cc = np.mgrid[r0:r1, c0:c1]
        return rr.ravel(), cc.ravel()
    hax = hx * abs(np.cos(yaw)) + hy * abs(np.sin(yaw))
    hay = hx * abs(np.sin(yaw)) + hy * abs(np.cos(yaw))
    c0 = int(np.searchsorted(xs, px - hax))
    c1 = int(np.searchsorted(xs, px + hax, side="right"))
    r0 = int(np.searchsorted(ys, py - hay))
    r1 = int(np.searchsorted(ys, py + hay, side="right"))
    if c1 <= c0 or r1 <= r0:
        return None
    gx, gy = np.meshgrid(xs[c0:c1] - px, ys[r0:r1] - py)
    u = gx * np.cos(yaw) + gy * np.sin(yaw)
    v = -gx * np.sin(yaw) + gy * np.cos(yaw)
    rr, cc = np.nonzero((np.abs(u) <= hx) & (np.abs(v) <= hy))
    if rr.size == 0:
        return None
    return r0 + rr, c0 + cc


def _seat_boxes(
    tile_boxes: list[Box], heights: np.ndarray, xs: np.ndarray, ys: np.ndarray
) -> list[Box]:
    """Drop each box onto the undulating tile ground: its base sits at the max
    ground height under its footprint, so it embeds at worst and never floats."""
    seated: list[Box] = []
    for b in tile_boxes:
        px, py, _ = b.pos
        hx, hy, hz = b.half
        nodes = _footprint_nodes(xs, ys, px, py, hx, hy, b.yaw)
        if nodes is None:
            r = int(np.clip(np.searchsorted(ys, py), 0, heights.shape[0] - 1))
            c = int(np.clip(np.searchsorted(xs, px), 0, heights.shape[1] - 1))
            base = float(heights[r, c])
        else:
            base = float(heights[nodes].max())
        seated.append(Box((px, py, base + hz), b.half, b.yaw))
    return seated


def generate(
    seed: int = DEFAULT_SEED,
    n_rows: int = DEFAULT_N_ROWS,
    tile_size: float = TILE_SIZE,
    border: float = BORDER,
    cell_size: float = CELL_SIZE,
    ordered: bool = False,
    difficulties: tuple[float, ...] | None = None,
    pad_radius: float = PAD_RADIUS,
    stair_platform_half: float = STAIR_PLATFORM_HALF,
    flat_row: bool = False,
) -> Arena:
    """Build one arena.

    ``difficulties`` replaces the 0..1 ramp with an explicit per-row list, and
    then sets the row count. Rows take those values exactly -- no jitter, and
    no clamp at 1.0, so the measurement arena can hold rows harder than
    anything training reaches. The column shuffle still draws from the same
    per-row stream in the same order, so a seed picks the same tile contents
    with or without an explicit difficulty list.

    ``pad_radius`` sizes the flat spawn pad. It cannot go below 0.36 m: the
    feet sit 0.31 m from the base centre and the foot radius is 0.046 m, so
    that is the flat ground a standing robot needs. Anything above the
    training default's spawn jitter (0.15 m) plus that footprint is safe.

    ``flat_row`` prepends one genuinely flat row, which becomes level 0. A
    difficulty-0 row is not flat -- its stairs still have a 2 cm riser -- so
    flat ground is a row of its own rather than the ramps turned down. The row
    is additive: every terrain row keeps the difficulty and the seed streams it
    has without the flag and moves up one row index and one tile.
    """
    n_cols = len(TYPES)
    if difficulties is not None:
        difficulties = tuple(float(d) for d in difficulties)
        n_rows = len(difficulties)
    n_terrain_rows = n_rows
    n_rows += int(flat_row)
    total_x = n_cols * tile_size + 2 * border
    total_y = n_rows * tile_size + 2 * border
    x_min, x_max = -total_x / 2, total_x / 2
    y_min, y_max = -total_y / 2, total_y / 2
    ncol = int(round(total_x / cell_size)) + 1
    nrow = int(round(total_y / cell_size)) + 1
    xs = np.linspace(x_min, x_max, ncol)
    ys = np.linspace(y_min, y_max, nrow)

    heights = np.zeros((nrow, ncol), dtype=np.float64)  # heightfield surface
    grid_x0 = -n_cols * tile_size / 2
    grid_y0 = -n_rows * tile_size / 2

    boxes: list[Box] = []
    tiles: list[TileSpec] = []
    if flat_row:
        # Ground stays at 0 here and no box is placed, so the row is flat to
        # the bit. It still carries one tile of every type, because the spawn
        # table is indexed by (row, type) and the env draws a type uniformly:
        # a row missing a type would spawn those envs at the arena origin. On
        # this row the type only names the column.
        cy = grid_y0 + 0.5 * tile_size
        for j, ttype in enumerate(TYPES):
            cx = grid_x0 + (j + 0.5) * tile_size
            tiles.append(
                TileSpec(
                    row=0, col=j, terrain_type=ttype, difficulty=0.0,
                    origin=(float(cx), float(cy), 0.0),
                    pad_radius=pad_radius, pad_height=0.0,
                )
            )
    for i in range(n_terrain_rows):
        # The generator index stays the row's identity: it drives the seed
        # streams and the difficulty ramp, so a row holds the same tiles with
        # and without the flat row. Only where it sits moves.
        row = i + int(flat_row)
        d = (
            difficulties[i] if difficulties is not None
            else i / (n_terrain_rows - 1)
        )
        if ordered:
            row_types = TYPES
        else:
            # One rng stream per row drives both the column shuffle and the
            # difficulty jitter. Each row still holds every type exactly once.
            row_rng = np.random.default_rng([seed, i])
            row_types = tuple(TYPES[k] for k in row_rng.permutation(n_cols))
            if difficulties is None and 0 < i < n_terrain_rows - 1:
                # Jitter under half a row gap (+-0.4 of 1/(n-1)) keeps realized
                # difficulty strictly increasing; rows 0 and n-1 stay exact so
                # the curriculum floor and ceiling are preserved. An explicit
                # difficulty list is taken as given, so no draw happens.
                d = float(np.clip(
                    d + row_rng.uniform(-0.4, 0.4) / (n_terrain_rows - 1),
                    0.0, 1.0))
        cy = grid_y0 + (row + 0.5) * tile_size
        ri0 = int(np.searchsorted(ys, cy - tile_size / 2))
        ri1 = int(np.searchsorted(ys, cy + tile_size / 2))
        for j, ttype in enumerate(row_types):
            cx = grid_x0 + (j + 0.5) * tile_size
            ci0 = int(np.searchsorted(xs, cx - tile_size / 2))
            ci1 = int(np.searchsorted(xs, cx + tile_size / 2))
            lx, ly = np.meshgrid(xs[ci0:ci1] - cx, ys[ri0:ri1] - cy)
            rng = _tile_rng(seed, i, j)
            pad_height = 0.0
            if ttype == "rough_uniform":
                heights[ri0:ri1, ci0:ci1] = _rough_patch(lx, ly, d, rng, pad_radius)
            elif ttype == "pyramid_slope":
                heights[ri0:ri1, ci0:ci1] = (
                    _slope_patch(lx, ly, d, +1.0) + _overlay(lx, ly, d, rng, pad_radius)
                )
                pad_height = slope_plateau_height(d)
            elif ttype == "inverted_pyramid_slope":
                heights[ri0:ri1, ci0:ci1] = (
                    _slope_patch(lx, ly, d, -1.0) + _overlay(lx, ly, d, rng, pad_radius)
                )
                pad_height = -slope_plateau_height(d)
            elif ttype == "pyramid_stairs":
                tile_boxes, pad_height = _pyramid_stair_boxes(
                    cx, cy, d, stair_platform_half
                )
                boxes.extend(tile_boxes)
            elif ttype == "inverted_pyramid_stairs":
                heights[ri0:ri1, ci0:ci1] = _pit_carve(
                    lx, ly, d, stair_platform_half
                )
                tile_boxes, pad_height = _inverted_stair_boxes(
                    cx, cy, d, stair_platform_half
                )
                boxes.extend(tile_boxes)
            elif ttype == "discrete_obstacles":
                heights[ri0:ri1, ci0:ci1] = _overlay(lx, ly, d, rng, pad_radius)
                boxes.extend(_seat_boxes(
                    _discrete_boxes(cx, cy, d, rng, pad_radius), heights, xs, ys
                ))
            elif ttype == "random_grid":
                heights[ri0:ri1, ci0:ci1] = _overlay(lx, ly, d, rng, pad_radius)
                boxes.extend(_seat_boxes(
                    _random_grid_boxes(cx, cy, d, rng, pad_radius), heights, xs, ys
                ))
            elif ttype == "wave":
                heights[ri0:ri1, ci0:ci1] = _wave_patch(lx, ly, d, rng, pad_radius)
            tiles.append(
                TileSpec(
                    row=row, col=j, terrain_type=ttype, difficulty=d,
                    origin=(float(cx), float(cy), 0.0),
                    pad_radius=pad_radius, pad_height=float(pad_height),
                )
            )

    lookup = heights.copy()
    for b in boxes:
        px, py, pz = b.pos
        hx, hy, hz = b.half
        top = pz + hz
        if b.yaw == 0.0:
            c0 = int(np.searchsorted(xs, px - hx))
            c1 = int(np.searchsorted(xs, px + hx, side="right"))
            r0 = int(np.searchsorted(ys, py - hy))
            r1 = int(np.searchsorted(ys, py + hy, side="right"))
            np.maximum(lookup[r0:r1, c0:c1], top, out=lookup[r0:r1, c0:c1])
        else:
            nodes = _footprint_nodes(xs, ys, px, py, hx, hy, b.yaw)
            if nodes is not None:
                np.maximum.at(lookup, nodes, top)

    hmin = float(heights.min())
    hmax = float(heights.max())
    if hmax - hmin < 1e-6:
        elevation_z, pos_z = 1e-3, 0.0
        hfield_data = np.zeros((nrow, ncol), dtype=np.float32)
    else:
        elevation_z, pos_z = hmax - hmin, hmin
        hfield_data = ((heights - hmin) / (hmax - hmin)).astype(np.float32)

    spec = TerrainSpec(
        seed=seed, n_rows=n_rows, n_cols=n_cols, tile_size=tile_size,
        border=border, cell_size=cell_size, ordered=ordered,
        pad_radius=pad_radius, stair_platform_half=stair_platform_half,
        n_steps=N_STEPS, types=TYPES,
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        hfield=HFieldSpec(
            nrow=nrow, ncol=ncol, radius_x=total_x / 2, radius_y=total_y / 2,
            elevation_z=elevation_z, base_z=HFIELD_BASE_Z, pos_z=pos_z,
        ),
        tiles=tuple(tiles),
        flat_row=flat_row,
    )
    return Arena(
        spec=spec, boxes=tuple(boxes),
        hfield_data=hfield_data, lookup=lookup.astype(np.float32),
    )


def lookup_height(arena: Arena, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinearly sample the ground-truth surface height at world (x, y)."""
    s = arena.spec
    cell_x = (s.x_max - s.x_min) / (s.hfield.ncol - 1)
    cell_y = (s.y_max - s.y_min) / (s.hfield.nrow - 1)
    return _bilinear(
        arena.lookup, s.x_min, cell_x, s.y_min, cell_y,
        np.asarray(x, dtype=float), np.asarray(y, dtype=float),
    )


def write_hfield_bin(path: Path, hfield_data: np.ndarray) -> None:
    """MuJoCo raw hfield: int32 nrow, int32 ncol, then nrow*ncol float32 (y-major)."""
    nrow, ncol = hfield_data.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", nrow, ncol))
        f.write(np.ascontiguousarray(hfield_data, dtype=np.float32).tobytes())


def spec_to_dict(spec: TerrainSpec) -> dict:
    """JSON-ready view of the spec: grid meta, heightfield meta, per-tile info.

    ``flat_row`` is written only when it is on, so an arena built without it is
    byte-identical to one built before the flag existed -- the eval arena's
    spec is the measurement fingerprint, and readers use ``spec.get``."""
    hf = spec.hfield
    out = {
        "seed": spec.seed,
        "n_rows": spec.n_rows,
        "n_cols": spec.n_cols,
        "tile_size": spec.tile_size,
        "border": spec.border,
        "cell_size": spec.cell_size,
        "ordered": spec.ordered,
        "pad_radius": spec.pad_radius,
        "stair_platform_half": spec.stair_platform_half,
        "n_steps": spec.n_steps,
        "types": list(spec.types),
        "extent": {
            "x_min": spec.x_min, "x_max": spec.x_max,
            "y_min": spec.y_min, "y_max": spec.y_max,
        },
        "hfield": {
            "nrow": hf.nrow, "ncol": hf.ncol,
            "radius_x": hf.radius_x, "radius_y": hf.radius_y,
            "elevation_z": hf.elevation_z, "base_z": hf.base_z, "pos_z": hf.pos_z,
        },
        "tiles": [
            {
                "row": t.row, "col": t.col, "type": t.terrain_type,
                "difficulty": t.difficulty, "origin": list(t.origin),
                "pad_radius": t.pad_radius, "pad_height": t.pad_height,
            }
            for t in spec.tiles
        ],
    }
    if spec.flat_row:
        out["flat_row"] = True
    return out
