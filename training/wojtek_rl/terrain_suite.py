"""The fixed terrain measurement suite: arena, cells, bars, and course.

This is the definition half of `./run.sh terrain-scan`. Nothing here touches
jax or a checkpoint, so the tables are cheap to import and easy to test.

Everything is fixed. No difficulty, tile, heading or offset is sampled, so two
scans of one checkpoint return the same numbers. The pieces:

- `DIFFICULTIES` -- the arena's rows, derived by inverting the generator's own
  ramps (`terrain.rough_difficulty` and friends) from the physical dimensions
  the terrain plan asks for, deduplicated and sorted. Two frontier rows (1.2
  and 1.4) sit past anything training reaches; they are measured, never gated.
- `CELLS` -- one (terrain type, difficulty) pair per measured cell, with the
  pass-rate bar the terrain plan sets for it, or `None` for a tracked cell.
- `COURSE` -- the 32 runs every cell is scored on: eight headings by four
  start offsets.

Cell names are stable identifiers. They are what a gate compares against a
baseline, so renaming one silently retires its history.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from wojtek_rl import terrain

# Arena identity. Seed 0 and sorted columns (`ordered=True`), so the course is
# reproducible from the row table alone.
EVAL_SEED = 0
EVAL_ORDERED = True
# 0.40 m against training's 0.60, so a crossing spends less of its distance on
# flat ground. It cannot shrink further: a standing robot needs 0.36 m of flat
# ground (see terrain.generate). Spawn jitter is zero here, which is what lets
# the pad shrink at all.
EVAL_PAD_RADIUS = 0.40

# Rows are keyed on the difficulty rounded to this many decimals, so two
# inversions that mean the same row (1 cm rough and a 3 cm riser are both
# d = 1/7) land on one row instead of two that differ in the 16th bit.
ROW_DECIMALS = 6

# Rows past the plan's ceiling. Tracked, never gated: the suite should be
# harder than necessary, so a policy's headroom is visible.
FRONTIER_DIFFICULTIES = (1.2, 1.4)

# Pass-rate bars from the terrain plan, easiest to hardest within a family.
BARS = (0.95, 0.80, 0.60)
# The commanded speed the plan's bars were written for. The same bars apply at
# the other two speeds, tagged `provisional` -- the plan sets no numbers there,
# and inventing a shape across speeds without data would be worse than keeping
# them flat. See `threshold()`.
PLAN_SPEED = 0.4
SPEEDS = (0.2, PLAN_SPEED, 0.7)

# The course: eight headings by four start offsets, 32 runs per cell and speed.
N_HEADINGS = 8
# Offsets along the heading, m. Small on purpose: the pad is 0.40 m and the
# standing footprint reaches 0.36 m, so this is the room there is while all
# four feet stay on flat ground (test_terrain_suite pins it). It still sweeps
# most of a 13 cm tread, which is the foot-placement variable a riser is
# sensitive to.
START_OFFSETS = (-0.03, -0.01, 0.01, 0.03)
RUNS_PER_CELL_SPEED = N_HEADINGS * len(START_OFFSETS)

# One crossing is a walk out to OUT_RADIUS or a walk back to BACK_RADIUS.
# OUT_RADIUS clears the outermost stair tread (1.25 m) and the scattered boxes
# (1.40 m), so a crossing cannot be completed without meeting the obstacle.
CROSSINGS = 4
OUT_RADIUS = 1.45
BACK_RADIUS = 0.30
# Skipped at the start of every run, so tracking error and clearance do not
# measure acceleration from standstill. Same 50 steps the flat battery skips.
SETTLE_STEPS = 50
# Step budget = the course distance at the commanded speed, times this. A run
# that averages below 1/1.6 = 62% of its commanded speed times out and counts
# as "did not finish the crossings", which is the distinction the pass rule is
# built on: it did not fall, it could not climb.
BUDGET_SLACK = 1.6


@dataclass(frozen=True)
class Cell:
    """One measured cell: a terrain type at a difficulty, with its bar."""

    name: str
    terrain_type: str
    difficulty: float
    row: int
    bar: float | None  # pass rate at PLAN_SPEED; None = tracked, never gated

    @property
    def tracked(self) -> bool:
        return self.bar is None


# Primary physical dimension per terrain type: the ramp that reads as "how
# hard is this tile", and the unit its cell name carries. random_grid (rubble)
# shares the discrete-obstacle height ramp.
_PRIMARY = {
    "rough_uniform": ("cm", lambda d: terrain.rough_amplitude(d) * 100),
    "pyramid_slope": ("deg", lambda d: math.degrees(terrain.slope_angle(d))),
    "inverted_pyramid_slope": ("deg", lambda d: math.degrees(terrain.slope_angle(d))),
    "pyramid_stairs": ("cm", lambda d: terrain.stair_riser(d) * 100),
    "inverted_pyramid_stairs": ("cm", lambda d: terrain.stair_riser(d) * 100),
    "discrete_obstacles": ("cm", lambda d: terrain.discrete_max_height(d) * 100),
    "random_grid": ("cm", lambda d: terrain.discrete_max_height(d) * 100),
    "wave": ("cm", lambda d: terrain.wave_amplitude(d) * 100),
}


def realized_dimension(terrain_type: str, difficulty: float) -> tuple[float, str]:
    """(value, unit) of the type's primary dimension at this difficulty."""
    unit, ramp = _PRIMARY[terrain_type]
    return float(ramp(difficulty)), unit


def cell_name(terrain_type: str, difficulty: float) -> str:
    """Stable identifier: type plus realized dimension, e.g.
    `pyramid_stairs_5cm`, `pyramid_slope_15deg`."""
    value, unit = realized_dimension(terrain_type, difficulty)
    text = f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{terrain_type}_{text}{unit}"


def _row_key(difficulty: float) -> float:
    return round(float(difficulty), ROW_DECIMALS)


# The suite, as physical dimensions. Each entry is (types, difficulties, bars);
# `bars` is per difficulty, `None` for a tracked cell. Difficulties come from
# the ramp inverses, so this table cannot drift from the terrain it describes.
_ROUGH = (0.01, 0.025, 0.04)  # amplitude, m
_SLOPE = (8.0, 15.0, 22.0)  # degrees
_RISER = (0.03, 0.05, 0.07, 0.09)  # m
_STEP = (0.025, 0.05, 0.065, 0.08)  # m
# Rubble and wave get the same three rungs the rough ladder uses. The plan sets
# no bar for either, so all six cells are tracked.
_RUBBLE_WAVE_ROWS = (_ROUGH[0], _ROUGH[1], _ROUGH[2])

_FAMILIES: tuple[tuple[tuple[str, ...], tuple[float, ...], tuple[float | None, ...]], ...] = (
    (
        ("rough_uniform",),
        tuple(terrain.rough_difficulty(a) for a in _ROUGH),
        BARS,
    ),
    (
        ("pyramid_slope", "inverted_pyramid_slope"),
        tuple(terrain.slope_difficulty(math.radians(a)) for a in _SLOPE),
        BARS,
    ),
    (
        # 9 cm is the plan's frontier riser: measured, never gated.
        ("pyramid_stairs", "inverted_pyramid_stairs"),
        tuple(terrain.stair_difficulty(r) for r in _RISER),
        BARS + (None,),
    ),
    (
        # The steps family mirrors the stairs family, because both are vertical
        # walls. The plan gates its 8 cm step at 60%, but 8 cm is 0.64 of this
        # robot's 12.5 cm hip height, above the 0.5-0.6 the same document calls
        # the blind limit -- so 8 cm becomes tracked and a 6.5 cm cell takes
        # the 60% bar.
        ("discrete_obstacles",),
        tuple(terrain.discrete_difficulty(h) for h in _STEP),
        BARS + (None,),
    ),
    (
        ("random_grid", "wave"),
        tuple(terrain.rough_difficulty(a) for a in _RUBBLE_WAVE_ROWS),
        (None, None, None),
    ),
)


def _build() -> tuple[tuple[float, ...], tuple[Cell, ...]]:
    """Rows and cells, together: a cell's `row` is an index into the rows, so
    the two have to be built from one pass over the family table."""
    keys: list[float] = []
    for _, difficulties, _ in _FAMILIES:
        for d in difficulties:
            if _row_key(d) not in keys:
                keys.append(_row_key(d))
    for d in FRONTIER_DIFFICULTIES:
        if _row_key(d) not in keys:
            keys.append(_row_key(d))
    rows = tuple(sorted(keys))
    index = {d: i for i, d in enumerate(rows)}

    cells: list[Cell] = []
    for types, difficulties, bars in _FAMILIES:
        for ttype in types:
            for d, bar in zip(difficulties, bars):
                key = _row_key(d)
                cells.append(
                    Cell(
                        name=cell_name(ttype, key),
                        terrain_type=ttype,
                        difficulty=key,
                        row=index[key],
                        bar=bar,
                    )
                )
    # Every type at every frontier row, tracked.
    for d in FRONTIER_DIFFICULTIES:
        key = _row_key(d)
        for ttype in terrain.TYPES:
            cells.append(
                Cell(
                    name=cell_name(ttype, key),
                    terrain_type=ttype,
                    difficulty=key,
                    row=index[key],
                    bar=None,
                )
            )
    return rows, tuple(cells)


DIFFICULTIES, CELLS = _build()
CELLS_BY_NAME = {c.name: c for c in CELLS}
# Bumped when the cell set or the arena changes in a way that makes old numbers
# incomparable. The gate refuses a baseline from a different version.
CELLS_VERSION = "v1"


def eval_arena_kwargs() -> dict:
    """`terrain.generate` kwargs for the measurement arena."""
    return {
        "seed": EVAL_SEED,
        "ordered": EVAL_ORDERED,
        "difficulties": DIFFICULTIES,
        "pad_radius": EVAL_PAD_RADIUS,
    }


def arena_fingerprint() -> dict:
    """What a scan records so two scans can be checked for comparability."""
    return {
        "seed": EVAL_SEED,
        "rows": len(DIFFICULTIES),
        "pad_radius": EVAL_PAD_RADIUS,
        "stair_platform_half": terrain.STAIR_PLATFORM_HALF,
        "n_steps": terrain.N_STEPS,
        "cells": CELLS_VERSION,
    }


def threshold(cell: Cell, speed: float) -> tuple[int | None, str]:
    """(pass count out of RUNS_PER_CELL_SPEED, provenance tag) for one cell at
    one commanded speed. `None` means tracked, so nothing is gated."""
    if cell.bar is None:
        return None, "tracked"
    tag = "plan" if speed == PLAN_SPEED else "provisional"
    return bar_count(cell.bar), tag


def bar_count(bar: float, runs: int = RUNS_PER_CELL_SPEED) -> int:
    """A pass-rate bar as a count of runs. 0.95/0.80/0.60 of 32 are 31/26/20."""
    return math.ceil(bar * runs)


@dataclass(frozen=True)
class Run:
    """One scored run: where on the pad it starts and which way it faces."""

    index: int
    heading_index: int
    yaw: float  # rad, the fixed heading it walks and returns along
    offset: float  # m along the heading from the tile centre


def course() -> tuple[Run, ...]:
    """The 32 runs, in a fixed order: heading outer, start offset inner."""
    runs = []
    for h in range(N_HEADINGS):
        yaw = 2.0 * math.pi * h / N_HEADINGS
        for offset in START_OFFSETS:
            runs.append(
                Run(index=len(runs), heading_index=h, yaw=yaw, offset=offset)
            )
    return tuple(runs)


COURSE = course()


def course_distance() -> float:
    """Metres of commanded travel one run asks for: out to OUT_RADIUS from the
    worst start offset, then three legs between BACK_RADIUS and OUT_RADIUS."""
    return (OUT_RADIUS + max(abs(o) for o in START_OFFSETS)) + (CROSSINGS - 1) * (
        OUT_RADIUS - BACK_RADIUS
    )


def episode_budget(speed: float, ctrl_dt: float) -> int:
    """Control steps one run gets at this commanded speed."""
    if speed == 0:
        raise ValueError(
            "a commanded speed of 0 has no step budget: the course is defined by "
            "distance, and a standing robot never completes a crossing"
        )
    travel_s = BUDGET_SLACK * course_distance() / abs(speed)
    return SETTLE_STEPS + math.ceil(travel_s / ctrl_dt)
