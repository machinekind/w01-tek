"""Procedural terrain arena for Wojtek locomotion training.

One shared arena, deterministic from a seed: a grid of ~3 m tiles with rows =
difficulty (0..1) and columns = terrain type, plus a flat border around the
grid. Rough, slopes, and the inverted-stairs pit are carved into a single
heightfield covering the whole arena; stairs and discrete obstacles are static
box geoms (crisp edges, cheap contacts) sitting on the heightfield. Every tile
reaches flat ground (z = 0) at its border so neighbours join seamlessly, and
every tile has a flat circular spawn pad at its centre.

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
)

DEFAULT_SEED = 0
DEFAULT_N_ROWS = 10
TILE_SIZE = 3.0
BORDER = 2.0
# 0.04 m keeps the elevation file and lookup npz in the low single MB, and
# keeps the base box's footprint (0.34 x 0.16 m) over few enough heightfield
# sub-cells that the MJWarp per-pair contact accumulator does not overflow;
# finer cells are the documented failure mode there.
CELL_SIZE = 0.04
# The base box lifts to this below the arena floor, so nothing falls through
# even where slope/stair tiles dig below z = 0.
HFIELD_BASE_Z = 0.5

# Flat circular spawn pad at every tile centre (>= 0.5 m per the brief).
PAD_RADIUS = 0.6
# Rough noise ramps from 0 at the pad edge to full over this width, so the pad
# has no cliff at its rim.
PAD_TAPER = 0.25

# Slope tiles: flat square platform (holds the pad), then a linear ramp.
SLOPE_PLATFORM_HALF = 0.6
# Stair tiles: flat square platform, then concentric 0.13 m treads. Four steps
# matches the physical mini-staircase rig (4 steps, 13 cm tread).
STAIR_PLATFORM_HALF = 0.7
TREAD = 0.13
N_STEPS = 4
# Outer half-size of the stair terrace region; the inverted pit is carved to
# this square and both stair types reach flat ground (0) at the rim.
STAIR_PIT_HALF = STAIR_PLATFORM_HALF + (N_STEPS - 1) * TREAD

# Rough tiles: white noise sampled on this coarse pitch then bilinearly
# upsampled, so features are foot-scale rolling ground, not per-cell spikes.
ROUGH_COARSE_STEP = 0.15

DISCRETE_N = 12
DISCRETE_HALF_RANGE = (0.10, 0.25)
DISCRETE_EDGE_MARGIN = 0.1


def rough_amplitude(d: float) -> float:
    return 0.005 + 0.035 * d


def slope_angle(d: float) -> float:
    return 0.4 * d


def stair_riser(d: float) -> float:
    return 0.02 + 0.07 * d


def discrete_max_height(d: float) -> float:
    return 0.01 + 0.07 * d


@dataclass(frozen=True)
class Box:
    """Axis-aligned static box, world centre and half-sizes (m)."""

    pos: tuple[float, float, float]
    half: tuple[float, float, float]


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
    n_rows: int
    n_cols: int
    tile_size: float
    border: float
    cell_size: float
    types: tuple[str, ...]
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    hfield: HFieldSpec
    tiles: tuple[TileSpec, ...]


@dataclass
class Arena:
    spec: TerrainSpec
    boxes: tuple[Box, ...]
    hfield_data: np.ndarray  # (nrow, ncol) float32 in [0, 1], row-major over y
    lookup: np.ndarray  # (nrow, ncol) float32 physical surface height (m)


def _bilinear(
    grid: np.ndarray, x0: float, dx: float, y0: float, dy: float,
    x: np.ndarray, y: np.ndarray,
) -> np.ndarray:
    """Sample grid (row indexes y, col indexes x) at world (x, y), clamped."""
    nr, nc = grid.shape
    fx = np.clip((x - x0) / dx, 0.0, nc - 1)
    fy = np.clip((y - y0) / dy, 0.0, nr - 1)
    cx0 = np.floor(fx).astype(int)
    ry0 = np.floor(fy).astype(int)
    cx1 = np.minimum(cx0 + 1, nc - 1)
    ry1 = np.minimum(ry0 + 1, nr - 1)
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


def _rough_patch(
    lx: np.ndarray, ly: np.ndarray, d: float, rng: np.random.Generator
) -> np.ndarray:
    amp = rough_amplitude(d)
    n = int(round(TILE_SIZE / ROUGH_COARSE_STEP)) + 1
    half = TILE_SIZE / 2
    coarse = rng.uniform(-1.0, 1.0, size=(n, n))
    noise = _bilinear(coarse, -half, TILE_SIZE / (n - 1), -half,
                      TILE_SIZE / (n - 1), lx, ly)
    r = np.sqrt(lx**2 + ly**2)
    taper = np.clip((r - PAD_RADIUS) / PAD_TAPER, 0.0, 1.0)
    return amp * noise * taper


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


def _pyramid_stair_boxes(cx: float, cy: float, d: float) -> tuple[list[Box], float]:
    """High central plateau; treads descend to ground (0) at the rim, so the
    tile border stays seamless. Boxes sit on the flat heightfield."""
    riser = stair_riser(d)
    top = N_STEPS * riser
    boxes = [Box((cx, cy, top / 2), (STAIR_PLATFORM_HALF, STAIR_PLATFORM_HALF, top / 2))]
    for k in range(1, N_STEPS):
        r_in = STAIR_PLATFORM_HALF + (k - 1) * TREAD
        r_out = STAIR_PLATFORM_HALF + k * TREAD
        boxes.extend(_frame_boxes(cx, cy, r_in, r_out, top=(N_STEPS - k) * riser))
    return boxes, top


def _pit_carve(lx: np.ndarray, ly: np.ndarray, d: float) -> np.ndarray:
    """Heightfield patch for an inverted-stairs pit: -H inside the pit square,
    0 outside, so the rim is flush with the tile ground."""
    depth = N_STEPS * stair_riser(d)
    return np.where(_cheby(lx, ly) <= STAIR_PIT_HALF, -depth, 0.0)


def _inverted_stair_boxes(cx: float, cy: float, d: float) -> tuple[list[Box], float]:
    """Terraces inside the carved pit: each ring sits on the pit floor (-H) and
    rises one riser more outward. The topmost riser (-riser -> 0) is the carve
    wall itself, beveled over at most one cell (CELL_SIZE) by the heightfield.
    Central pad is the carved floor at -H, no plateau box."""
    riser = stair_riser(d)
    depth = N_STEPS * riser
    boxes: list[Box] = []
    for j in range(1, N_STEPS):
        r_in = STAIR_PLATFORM_HALF + (j - 1) * TREAD
        r_out = STAIR_PLATFORM_HALF + j * TREAD
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


def _discrete_boxes(cx: float, cy: float, d: float, rng: np.random.Generator) -> list[Box]:
    max_h = discrete_max_height(d)
    # Keep obstacles clear of the tile edge so the border stays seamless with
    # the neighbour tile (a box at the rim would be a step across it).
    reach = TILE_SIZE / 2 - DISCRETE_EDGE_MARGIN
    boxes: list[Box] = []
    for _ in range(DISCRETE_N):
        hx = float(rng.uniform(*DISCRETE_HALF_RANGE))
        hy = float(rng.uniform(*DISCRETE_HALF_RANGE))
        u = float(rng.uniform(-reach + hx, reach - hx))
        v = float(rng.uniform(-reach + hy, reach - hy))
        # Clear the pad by the box's corner radius, so no corner pokes in.
        if np.hypot(u, v) < PAD_RADIUS + np.hypot(hx, hy) + 0.05:
            continue
        h = float(rng.uniform(0.5, 1.0)) * max_h
        boxes.append(Box((cx + u, cy + v, h / 2), (hx, hy, h / 2)))
    return boxes


def generate(
    seed: int = DEFAULT_SEED,
    n_rows: int = DEFAULT_N_ROWS,
    tile_size: float = TILE_SIZE,
    border: float = BORDER,
    cell_size: float = CELL_SIZE,
) -> Arena:
    n_cols = len(TYPES)
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
    for i in range(n_rows):
        d = i / (n_rows - 1)
        cy = grid_y0 + (i + 0.5) * tile_size
        ri0 = int(np.searchsorted(ys, cy - tile_size / 2))
        ri1 = int(np.searchsorted(ys, cy + tile_size / 2))
        for j, ttype in enumerate(TYPES):
            cx = grid_x0 + (j + 0.5) * tile_size
            ci0 = int(np.searchsorted(xs, cx - tile_size / 2))
            ci1 = int(np.searchsorted(xs, cx + tile_size / 2))
            lx, ly = np.meshgrid(xs[ci0:ci1] - cx, ys[ri0:ri1] - cy)
            rng = _tile_rng(seed, i, j)
            pad_height = 0.0
            if ttype == "rough_uniform":
                heights[ri0:ri1, ci0:ci1] = _rough_patch(lx, ly, d, rng)
            elif ttype == "pyramid_slope":
                heights[ri0:ri1, ci0:ci1] = _slope_patch(lx, ly, d, +1.0)
                pad_height = slope_plateau_height(d)
            elif ttype == "inverted_pyramid_slope":
                heights[ri0:ri1, ci0:ci1] = _slope_patch(lx, ly, d, -1.0)
                pad_height = -slope_plateau_height(d)
            elif ttype == "pyramid_stairs":
                tile_boxes, pad_height = _pyramid_stair_boxes(cx, cy, d)
                boxes.extend(tile_boxes)
            elif ttype == "inverted_pyramid_stairs":
                heights[ri0:ri1, ci0:ci1] = _pit_carve(lx, ly, d)
                tile_boxes, pad_height = _inverted_stair_boxes(cx, cy, d)
                boxes.extend(tile_boxes)
            elif ttype == "discrete_obstacles":
                boxes.extend(_discrete_boxes(cx, cy, d, rng))
            tiles.append(
                TileSpec(
                    row=i, col=j, terrain_type=ttype, difficulty=d,
                    origin=(float(cx), float(cy), 0.0),
                    pad_radius=PAD_RADIUS, pad_height=float(pad_height),
                )
            )

    lookup = heights.copy()
    for b in boxes:
        px, py, pz = b.pos
        hx, hy, hz = b.half
        c0 = int(np.searchsorted(xs, px - hx))
        c1 = int(np.searchsorted(xs, px + hx, side="right"))
        r0 = int(np.searchsorted(ys, py - hy))
        r1 = int(np.searchsorted(ys, py + hy, side="right"))
        np.maximum(lookup[r0:r1, c0:c1], pz + hz, out=lookup[r0:r1, c0:c1])

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
        border=border, cell_size=cell_size, types=TYPES,
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        hfield=HFieldSpec(
            nrow=nrow, ncol=ncol, radius_x=total_x / 2, radius_y=total_y / 2,
            elevation_z=elevation_z, base_z=HFIELD_BASE_Z, pos_z=pos_z,
        ),
        tiles=tuple(tiles),
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
    """JSON-ready view of the spec: grid meta, heightfield meta, per-tile info."""
    hf = spec.hfield
    return {
        "seed": spec.seed,
        "n_rows": spec.n_rows,
        "n_cols": spec.n_cols,
        "tile_size": spec.tile_size,
        "border": spec.border,
        "cell_size": spec.cell_size,
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
