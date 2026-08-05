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
- `COURSE` -- the 64 runs every cell is scored on: eight headings by four
  start offsets, on two draws of the policy's noise.

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
# 0.7, tagged `provisional` -- the plan sets no numbers there, and inventing a
# shape across speeds without data would be worse than keeping them flat. See
# `threshold()`.
PLAN_SPEED = 0.4
# 0.2 m/s was measured and dropped on 2026-07-27. Finishing the course inside
# BUDGET_SLACK needs a run to average 62% of its commanded speed, every policy
# measured so far tracks about 60% at 0.2, and all of them scored 0 there. The
# bars gate at 0.4.
SPEEDS = (PLAN_SPEED, 0.7)

# The course: eight headings by four start offsets, each start run on two draws
# of the policy's noise -- 64 runs per cell and speed.
N_HEADINGS = 8
# Offsets along the heading, m. Small on purpose: the pad is 0.40 m and the
# standing footprint reaches 0.36 m, so this is the room there is while all
# four feet stay on flat ground (test_terrain_suite pins it). The 0.06 m span
# sweeps 46% of a 13 cm tread on an axis heading; on a diagonal it projects to
# 0.06/sqrt(2) = 0.042 m of Chebyshev radius, a third of a tread. That phase
# is the foot-placement variable a riser is sensitive to.
START_OFFSETS = (-0.03, -0.01, 0.01, 0.03)
# The same 32 physical starts, scored on two draws of the policy's noise. The
# scan's measured test-retest wobble moved one cell/speed pair by up to 6 of 32
# runs across --eval-seed draws (2026-07-27 validation), and that spread is
# noise-draw variance: a second draw of the same starts attacks exactly that
# axis without changing what is being tested. The draws differ only in the
# rollout key each run is handed (terrain_scan splits the cell key per run), so
# no physical dimension of the course moves.
N_NOISE_DRAWS = 2
RUNS_PER_CELL_SPEED = N_HEADINGS * len(START_OFFSETS) * N_NOISE_DRAWS

# One crossing is a walk out to OUT_RADIUS or a walk back to BACK_RADIUS, both
# measured as CHEBYSHEV distance from the tile centre -- max(|dx|, |dy|), not the
# Euclidean radius.
#
# Chebyshev because that is how the terrain is built. Every feature is a
# concentric square: the slope frustum and the stair pit are carved against
# terrain._cheby, and the scattered boxes are held inside a square reach. A
# Euclidean radius therefore means something different on every heading. At
# Euclidean 1.45 on a 45 degree heading the base is only 1.03 from the centre in
# Chebyshev -- still on tread 4 of 6 -- so half of the eight headings would have
# completed a "crossing" without climbing the last two risers, and the pass rate
# would blend two different tests.
#
# OUT_RADIUS clears the outermost stair tread (Chebyshev 1.25 m) and the
# scattered boxes (1.40 m), so a crossing cannot be completed without meeting the
# obstacle.
#
# Known limitation, accepted: at 1.45 the base is 0.05 m from the tile border, so
# the leading feet reach 0.21 m into a neighbouring tile at the turnaround (0.255
# m on a diagonal, where the foot corner leads). That is unavoidable with a
# six-step flight on a 3 m tile -- clearing the last riser needs the base at
# 1.25 + 0.257 = 1.51 m. Which neighbour depends on the heading: along x it
# shares the row (same difficulty, different type); along y it is the SAME type
# one row over, i.e. a harder or easier difficulty -- the worst case; on a
# diagonal it is the corner tile, which differs in both. Tiles are flat to within
# 2 cm at the border itself (test_tile_borders_seamless), but 0.21 m in the
# neighbour's own features have begun: worst case is a steep inverted slope,
# 10 cm below grade at the hardest gated row and 15 cm at the frontier rows. The
# crossing is already scored when the base reaches the ring, so this is the state
# at the end of a leg rather than terrain the robot had to cross.
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
    # Which measurement arena the cell lives on. The legacy cells share the
    # `eval` arena; the deep-tread stair cells need their own geometry and
    # so their own file set.
    arena: str = "eval"

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
# Bumped when the cell set, the speed set or the arena changes in a way that
# makes old numbers incomparable. The gate refuses a baseline from a different
# version. v2 dropped the 0.2 m/s column. v3 doubled the runs per cell to 64
# with a second noise draw, so every out-of-32 number is incomparable.
CELLS_VERSION = "v3"

# The deep-tread stair cells: real stair geometry (Blondel-plausible treads)
# on the same rows, headings and budgets. They live on their own arena
# (`eval_deep`) so the legacy arena -- and every score ever taken on it --
# stays byte-identical; suite version 2 is the legacy suite plus these.
# Tracked, not gated: the plan has no bars for the geometry yet. Grand
# totals are only comparable within one suite version; cross-version
# comparisons happen per cell family.
SUITE_VERSION = 2
DEEP_TREAD = 0.30
_DEEP_RISER = (0.03, 0.05, 0.07, 0.09)  # m


def _build_deep() -> tuple[Cell, ...]:
    index = {d: i for i, d in enumerate(DIFFICULTIES)}
    cells = []
    for ttype in ("pyramid_stairs", "inverted_pyramid_stairs"):
        for riser in _DEEP_RISER:
            key = _row_key(terrain.stair_difficulty(riser))
            text = f"{riser * 100:.1f}".rstrip("0").rstrip(".")
            cells.append(
                Cell(
                    name=f"{ttype}_deep_{text}cm",
                    terrain_type=ttype,
                    difficulty=key,
                    row=index[key],
                    bar=None,
                    arena="eval_deep",
                )
            )
    return tuple(cells)


DEEP_CELLS = _build_deep()
ALL_CELLS = CELLS + DEEP_CELLS
ALL_CELLS_BY_NAME = {c.name: c for c in ALL_CELLS}


def eval_arena_kwargs() -> dict:
    """`terrain.generate` kwargs for the measurement arena."""
    return {
        "seed": EVAL_SEED,
        "ordered": EVAL_ORDERED,
        "difficulties": DIFFICULTIES,
        "pad_radius": EVAL_PAD_RADIUS,
    }


def deep_arena_kwargs() -> dict:
    """`terrain.generate` kwargs for the deep-tread measurement arena: the
    legacy course's rows, order and pads, on real stair treads."""
    return {**eval_arena_kwargs(), "stair_tread": DEEP_TREAD}


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


def deep_arena_fingerprint() -> dict:
    """The deep arena's own fingerprint, recorded separately so the legacy
    `arena` fingerprint -- what the gate compares -- never changes shape.
    Extended stair geometry sizes its platform by the summit rule, so the
    deep course's platform follows its pad."""
    platform = terrain.summit_platform_half(EVAL_PAD_RADIUS)
    return {
        **arena_fingerprint(),
        "stair_tread": DEEP_TREAD,
        "stair_platform_half": platform,
        "n_steps": terrain.stair_steps(DEEP_TREAD, platform),
    }


def threshold(cell: Cell, speed: float) -> tuple[int | None, str]:
    """(pass count out of RUNS_PER_CELL_SPEED, provenance tag) for one cell at
    one commanded speed. `None` means tracked, so nothing is gated."""
    if cell.bar is None:
        return None, "tracked"
    tag = "plan" if speed == PLAN_SPEED else "provisional"
    return bar_count(cell.bar), tag


def bar_count(bar: float, runs: int = RUNS_PER_CELL_SPEED) -> int:
    """A pass-rate bar as a count of runs. 0.95/0.80/0.60 of 64 are 61/52/39."""
    return math.ceil(bar * runs)


@dataclass(frozen=True)
class Run:
    """One scored run: where on the pad it starts and which way it faces."""

    index: int
    heading_index: int
    yaw: float  # rad, the fixed heading it walks and returns along
    offset: float  # m along the heading from the tile centre
    draw: int  # which noise draw; both draws start from the same pose


def course() -> tuple[Run, ...]:
    """The 64 runs, in a fixed order: draw outer, heading, start offset inner.

    Draw is outermost, so runs 0..31 are the 32-run course unchanged and 32..63
    repeat those starts. What separates the two is the rollout key each run is
    handed -- terrain_scan splits the cell's key into RUNS_PER_CELL_SPEED of
    them -- so nothing physical distinguishes a draw.
    """
    runs = []
    for draw in range(N_NOISE_DRAWS):
        for h in range(N_HEADINGS):
            yaw = 2.0 * math.pi * h / N_HEADINGS
            for offset in START_OFFSETS:
                runs.append(
                    Run(
                        index=len(runs), heading_index=h, yaw=yaw,
                        offset=offset, draw=draw,
                    )
                )
    return tuple(runs)


COURSE = course()


def heading_stretch(yaw: float) -> float:
    """How far a heading walks to reach a given Chebyshev radius, per unit radius.

    1 along an axis, sqrt(2) along a diagonal. The radii are Chebyshev, so this
    is what makes a diagonal run's course longer than an axis run's."""
    return 1.0 / max(abs(math.cos(yaw)), abs(math.sin(yaw)))


def run_distance(run: "Run") -> float:
    """Metres this particular run walks: out to OUT_RADIUS from its start offset,
    then three legs between BACK_RADIUS and OUT_RADIUS, at its own heading.

    The offset is signed: a positive offset spawns the robot ahead along the
    heading it walks out on, so its first leg is SHORTER by the offset. Getting
    the sign wrong hands the +offset runs more time than their course needs,
    and at a speed where the timeout is the deciding factor that turns the
    start offset into a pass/fail variable."""
    stretch = heading_stretch(run.yaw)
    out_leg = OUT_RADIUS * stretch - run.offset
    return_leg = (OUT_RADIUS - BACK_RADIUS) * stretch
    return out_leg + (CROSSINGS - 1) * return_leg


def course_distance() -> float:
    """The longest run's distance, which is what the batch's hard stop is sized
    on. Individual runs get their own deadline (`episode_budget`), so an axis
    heading is not handed the diagonal's extra slack -- that would make the
    timeout threshold, and so the effective difficulty, heading-dependent."""
    return max(run_distance(r) for r in COURSE)


def episode_budget(speed: float, ctrl_dt: float, distance: float | None = None) -> int:
    """Control steps a run gets at this commanded speed.

    `distance` defaults to the longest run's, which is the batch's hard stop;
    pass `run_distance(run)` for one run's own deadline."""
    if speed == 0:
        raise ValueError(
            "a commanded speed of 0 has no step budget: the course is defined by "
            "distance, and a standing robot never completes a crossing"
        )
    travel_s = BUDGET_SLACK * (course_distance() if distance is None else distance)
    return SETTLE_STEPS + math.ceil(travel_s / abs(speed) / ctrl_dt)


def run_deadlines(speed: float, ctrl_dt: float) -> tuple[int, ...]:
    """Each run's own step deadline, in COURSE order."""
    return tuple(episode_budget(speed, ctrl_dt, run_distance(r)) for r in COURSE)
