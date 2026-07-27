"""The fixed terrain measurement suite: row table, cells, bars, course, arena.

Pins the numbers the terrain measurement plan specifies, so a change to a ramp,
a bar or the course order has to be a deliberate edit here rather than a silent
drift in what "pyramid_stairs_5cm at 80%" means.
"""

import math

import mujoco
import numpy as np
import pytest

from test_terrain import pad_flatness  # sibling test module, same directory

from wojtek_rl import build_terrain, paths, terrain, terrain_suite

TEST_ARENA = "test"


@pytest.fixture(scope="module")
def eval_arena():
    return terrain.generate(**terrain_suite.eval_arena_kwargs())


# -- 1. Row table --------------------------------------------------------------

# The dimensions the terrain plan asks for, and the difficulty each row is
# supposed to realize. Derived independently of terrain_suite (the plan's table,
# transcribed) so the two have to agree.
PLAN_ROWS = (
    (0.142857, "rough 1 cm, riser 3 cm"),
    (0.214286, "steps 2.5 cm"),
    (0.349066, "slope 8 deg"),
    (0.428571, "riser 5 cm"),
    (0.571429, "rough 2.5 cm, steps 5 cm"),
    (0.654498, "slope 15 deg"),
    (0.714286, "riser 7 cm"),
    (0.785714, "steps 6.5 cm"),
    (0.959931, "slope 22 deg"),
    (1.0, "rough 4 cm, riser 9 cm, steps 8 cm"),
    (1.2, "frontier"),
    (1.4, "frontier"),
)


def test_row_table_matches_the_plan():
    assert len(terrain_suite.DIFFICULTIES) == 12
    for got, (want, label) in zip(terrain_suite.DIFFICULTIES, PLAN_ROWS):
        assert got == pytest.approx(want, abs=5e-7), label


def test_rows_are_sorted_and_deduplicated():
    rows = terrain_suite.DIFFICULTIES
    assert list(rows) == sorted(rows)
    assert len(set(rows)) == len(rows)
    # 1 cm rough and a 3 cm riser are both d = 1/7, and the two inversions do
    # not produce the same float; they must still be one row.
    assert terrain.rough_difficulty(0.01) != terrain.stair_difficulty(0.03)
    assert len({c.row for c in terrain_suite.CELLS if c.difficulty == rows[0]}) == 1


def test_realized_dimensions_match_the_plan():
    """Each row realizes the physical dimension it was derived from."""
    d = dict(zip(terrain_suite.DIFFICULTIES, range(12)))
    r = terrain_suite.DIFFICULTIES
    assert terrain.rough_amplitude(r[0]) == pytest.approx(0.01, abs=1e-7)
    assert terrain.stair_riser(r[0]) == pytest.approx(0.03, abs=1e-7)
    assert terrain.discrete_max_height(r[1]) == pytest.approx(0.025, abs=1e-7)
    assert math.degrees(terrain.slope_angle(r[2])) == pytest.approx(8.0, abs=1e-4)
    assert terrain.stair_riser(r[3]) == pytest.approx(0.05, abs=1e-7)
    assert terrain.rough_amplitude(r[4]) == pytest.approx(0.025, abs=1e-7)
    assert terrain.discrete_max_height(r[4]) == pytest.approx(0.05, abs=1e-7)
    assert math.degrees(terrain.slope_angle(r[5])) == pytest.approx(15.0, abs=1e-4)
    assert terrain.stair_riser(r[6]) == pytest.approx(0.07, abs=1e-7)
    assert terrain.discrete_max_height(r[7]) == pytest.approx(0.065, abs=1e-7)
    assert math.degrees(terrain.slope_angle(r[8])) == pytest.approx(22.0, abs=1e-4)
    assert terrain.rough_amplitude(r[9]) == pytest.approx(0.04, abs=1e-7)
    assert terrain.stair_riser(r[9]) == pytest.approx(0.09, abs=1e-7)
    assert terrain.discrete_max_height(r[9]) == pytest.approx(0.08, abs=1e-7)
    assert len(d) == 12


# -- 2. Cells and bars ---------------------------------------------------------


def test_cell_count():
    assert len(terrain_suite.CELLS) == 43
    gated = [c for c in terrain_suite.CELLS if not c.tracked]
    assert len(gated) == 18
    assert len(terrain_suite.CELLS) - len(gated) == 25


def test_cell_names_are_unique_and_stable():
    names = [c.name for c in terrain_suite.CELLS]
    assert len(set(names)) == len(names)
    # A gate compares these against a baseline, so renaming one retires its
    # history. Spot-check one cell per family.
    for name in (
        "rough_uniform_1cm",
        "pyramid_slope_15deg",
        "inverted_pyramid_slope_22deg",
        "pyramid_stairs_5cm",
        "inverted_pyramid_stairs_5cm",
        "pyramid_stairs_9cm",
        "discrete_obstacles_6.5cm",
        "discrete_obstacles_8cm",
        "random_grid_5cm",
        "wave_6cm",
        "pyramid_stairs_11.8cm",
    ):
        assert name in terrain_suite.CELLS_BY_NAME, name


def test_bar_counts_out_of_32():
    assert [terrain_suite.bar_count(b) for b in terrain_suite.BARS] == [31, 26, 20]
    assert terrain_suite.RUNS_PER_CELL_SPEED == 32


def test_gated_families_take_the_plan_ladder():
    for family, names in {
        "rough": ("rough_uniform_1cm", "rough_uniform_2.5cm", "rough_uniform_4cm"),
        "slope_up": ("pyramid_slope_8deg", "pyramid_slope_15deg", "pyramid_slope_22deg"),
        "stairs_up": ("pyramid_stairs_3cm", "pyramid_stairs_5cm", "pyramid_stairs_7cm"),
        "steps": (
            "discrete_obstacles_2.5cm",
            "discrete_obstacles_5cm",
            "discrete_obstacles_6.5cm",
        ),
    }.items():
        bars = [terrain_suite.CELLS_BY_NAME[n].bar for n in names]
        assert bars == list(terrain_suite.BARS), family


def test_frontier_and_blind_limit_cells_are_tracked():
    """Rows past the plan's ceiling, the 9 cm risers, and the 8 cm step are
    measured and never gated. 8 cm is 0.64 of the 12.5 cm hip height, above the
    0.5-0.6 the plan itself calls the blind limit."""
    for name in (
        "pyramid_stairs_9cm",
        "inverted_pyramid_stairs_9cm",
        "discrete_obstacles_8cm",
        "random_grid_5cm",
        "wave_6cm",
    ):
        assert terrain_suite.CELLS_BY_NAME[name].tracked, name
    frontier = [
        c for c in terrain_suite.CELLS
        if c.difficulty in terrain_suite.FRONTIER_DIFFICULTIES
    ]
    assert len(frontier) == 16
    assert all(c.tracked for c in frontier)
    assert {c.terrain_type for c in frontier} == set(terrain.TYPES)


def test_threshold_provenance():
    """The plan sets bars at 0.4 m/s only; the same number at 0.7 carries a
    provisional tag, so a failure there reads as a prompt to check the bar."""
    gated = terrain_suite.CELLS_BY_NAME["pyramid_stairs_5cm"]
    tracked = terrain_suite.CELLS_BY_NAME["pyramid_stairs_9cm"]
    assert terrain_suite.SPEEDS == (0.4, 0.7)
    assert terrain_suite.threshold(gated, 0.4) == (26, "plan")
    assert terrain_suite.threshold(gated, 0.7) == (26, "provisional")
    for speed in terrain_suite.SPEEDS:
        assert terrain_suite.threshold(tracked, speed) == (None, "tracked")


# -- 3. The course -------------------------------------------------------------


def test_course_is_fixed_and_deterministic():
    a, b = terrain_suite.course(), terrain_suite.course()
    assert a == b == terrain_suite.COURSE
    assert len(a) == 32
    assert [r.index for r in a] == list(range(32))
    # heading outer, offset inner
    assert [r.heading_index for r in a[:8]] == [0, 0, 0, 0, 1, 1, 1, 1]
    assert len({(r.heading_index, r.offset) for r in a}) == 32
    assert {r.yaw for r in a} == {
        2 * math.pi * h / 8 for h in range(8)
    }


def test_course_offsets_keep_every_foot_on_the_pad():
    """The start offsets are small because the pad is: a standing robot reaches
    0.36 m, and the measurement pad is 0.40 m."""
    foot = np.array([[0.257, 0.174], [0.257, -0.174], [-0.257, 0.174], [-0.257, -0.174]])
    foot_radius = 0.046
    for run in terrain_suite.COURSE:
        step = np.array([math.cos(run.yaw), math.sin(run.yaw)]) * run.offset
        reach = np.linalg.norm(foot + step, axis=1).max() + foot_radius
        assert reach < terrain_suite.EVAL_PAD_RADIUS, (run, reach)
    # and the sweep is worth having: 46% of a 13 cm tread on an axis heading,
    # a third on a diagonal (the span projects to Chebyshev by 1/sqrt(2))
    span = max(terrain_suite.START_OFFSETS) - min(terrain_suite.START_OFFSETS)
    assert 0.3 * terrain.TREAD < span / math.sqrt(2) < span < terrain.TREAD


def test_crossing_radii_clear_the_obstacles():
    """A crossing cannot be completed on the flat pad: the turnaround radius is
    outside the outermost stair tread and the scattered boxes."""
    assert terrain_suite.OUT_RADIUS > terrain.stair_pit_half()
    box_reach = terrain.TILE_SIZE / 2 - terrain.DISCRETE_EDGE_MARGIN
    assert terrain_suite.OUT_RADIUS > box_reach
    assert terrain_suite.OUT_RADIUS < terrain.TILE_SIZE / 2
    assert terrain_suite.BACK_RADIUS < terrain_suite.EVAL_PAD_RADIUS


def test_episode_budget_covers_the_course():
    ctrl_dt = 0.02
    for speed in terrain_suite.SPEEDS:
        budget = terrain_suite.episode_budget(speed, ctrl_dt)
        walked = (budget - terrain_suite.SETTLE_STEPS) * ctrl_dt * speed
        assert walked >= terrain_suite.course_distance()
        # the slack is what makes a timeout mean "could not climb"
        assert walked == pytest.approx(
            terrain_suite.BUDGET_SLACK * terrain_suite.course_distance(), rel=0.01
        )
    # faster commands get less time, and every budget is a whole number of steps
    budgets = [terrain_suite.episode_budget(s, ctrl_dt) for s in terrain_suite.SPEEDS]
    assert budgets == sorted(budgets, reverse=True)
    assert all(isinstance(b, int) for b in budgets)


def test_every_run_gets_the_same_speed_fraction_not_the_same_budget():
    """A diagonal heading walks sqrt(2) further to the same Chebyshev radius. One
    shared budget would hand an axis run that extra slack, so the speed fraction
    a run has to sustain -- and with it the effective difficulty -- would depend
    on its heading."""
    ctrl_dt = 0.02
    for speed in terrain_suite.SPEEDS:
        deadlines = terrain_suite.run_deadlines(speed, ctrl_dt)
        assert len(deadlines) == len(terrain_suite.COURSE)
        for run, deadline in zip(terrain_suite.COURSE, deadlines):
            walked = (deadline - terrain_suite.SETTLE_STEPS) * ctrl_dt * speed
            # the same sustained-speed fraction on every heading, 1/BUDGET_SLACK,
            # give or take the single step the deadline is rounded up by
            assert terrain_suite.run_distance(run) / walked == pytest.approx(
                1.0 / terrain_suite.BUDGET_SLACK, rel=2e-3
            ), (run, deadline)
        # the axis headings really do get less time than the diagonals
        assert min(deadlines) < max(deadlines)
        # and the hard stop is the longest run's deadline
        assert max(deadlines) == terrain_suite.episode_budget(speed, ctrl_dt)


def test_heading_stretch_is_one_on_axes_and_root_two_on_diagonals():
    for run in terrain_suite.COURSE:
        stretch = terrain_suite.heading_stretch(run.yaw)
        expected = 1.0 if run.heading_index % 2 == 0 else math.sqrt(2.0)
        assert stretch == pytest.approx(expected)


def test_total_scan_size():
    """The scan is 2752 runs; the cluster job's header is sized off this."""
    runs = len(terrain_suite.CELLS) * terrain_suite.RUNS_PER_CELL_SPEED * len(
        terrain_suite.SPEEDS
    )
    assert runs == 2752
    steps = sum(
        terrain_suite.episode_budget(s, 0.02) for s in terrain_suite.SPEEDS
    ) * len(terrain_suite.CELLS) * terrain_suite.RUNS_PER_CELL_SPEED
    assert 2.5e6 < steps < 3.5e6, steps


# -- 4. Arena kinds ------------------------------------------------------------


def test_arena_kinds_have_distinct_paths():
    seen = {}
    for kind in paths.TERRAIN_KINDS:
        files = paths.terrain_paths(kind)
        assert set(files) == {"scene", "hfield", "spec", "lookup"}
        for role, p in files.items():
            assert (role, p) not in seen.items()
            assert p not in seen.values(), (kind, p)
            seen[(kind, role)] = p
    assert len(set(seen.values())) == 4 * len(paths.TERRAIN_KINDS)
    # train keeps the original unsuffixed names
    assert paths.terrain_paths("train")["scene"].name == "scene_terrain.xml"
    assert paths.terrain_paths("eval")["scene"].name == "scene_terrain_eval.xml"
    with pytest.raises(ValueError, match="arena kind"):
        paths.terrain_paths("nope")


# -- 5. The measurement arena --------------------------------------------------


def test_eval_arena_realizes_every_cell(eval_arena):
    """Every cell names a tile that exists, at exactly its difficulty."""
    by_key = {
        (t.terrain_type, t.row): t for t in eval_arena.spec.tiles
    }
    assert eval_arena.spec.n_rows == 12
    for cell in terrain_suite.CELLS:
        tile = by_key[(cell.terrain_type, cell.row)]
        assert tile.difficulty == pytest.approx(cell.difficulty, abs=1e-9), cell.name
        assert tile.pad_radius == terrain_suite.EVAL_PAD_RADIUS


def test_eval_arena_is_sorted_not_shuffled(eval_arena):
    """Sorted columns, so the course is reproducible from the row table. The
    training arena shuffles, which is why measurement tiles are not the tiles a
    policy trained on even at the same seed -- that is the point."""
    s = eval_arena.spec
    assert s.ordered is True
    for r in range(s.n_rows):
        cols = sorted((t for t in s.tiles if t.row == r), key=lambda t: t.col)
        assert [t.terrain_type for t in cols] == list(terrain.TYPES)
    train = terrain.generate(seed=terrain_suite.EVAL_SEED)
    assert [(t.row, t.col, t.terrain_type) for t in train.spec.tiles] != [
        (t.row, t.col, t.terrain_type) for t in s.tiles
    ]


def test_eval_arena_pads_flat_under_the_smaller_radius(eval_arena):
    spread, offset = pad_flatness(eval_arena, spawn_jitter=0.0)
    assert spread < 1e-3, spread
    assert offset < 1e-3, offset


def test_eval_arena_holds_the_deepest_pit(eval_arena):
    """The frontier row digs a 71 cm pit, and what absorbs it is the geom
    frame: pos_z sits at the arena minimum, so the hfield's solid base extends
    below the pit floor for any positive base_z. base_z itself only has to be
    positive (the MJCF compiler enforces that); pit depth never adds to it."""
    hf = eval_arena.spec.hfield
    deepest = terrain.N_STEPS * terrain.stair_riser(max(terrain_suite.DIFFICULTIES))
    assert deepest == pytest.approx(0.708, abs=1e-3)
    assert hf.pos_z == pytest.approx(-deepest, abs=1e-6)
    assert hf.base_z > 0


def test_eval_arena_seams_have_no_cliff(eval_arena):
    """Same invariant as the training arena's, at a bound the frontier rows
    allow: rough noise is bilinearly upsampled from a 0.15 m coarse grid, so one
    0.04 m cell can step by 2*cell/coarse = 0.53 of the amplitude, plus 0.16 of
    it from the edge taper. The training arena's 2.5 cm bound is that same 0.7 x
    amplitude at its 4 cm ceiling; here the ceiling is 5.4 cm."""
    s = eval_arena.spec
    L = eval_arena.lookup
    xs = np.linspace(s.x_min, s.x_max, s.hfield.ncol)
    ys = np.linspace(s.y_min, s.y_max, s.hfield.nrow)
    grid_x0 = -s.n_cols * s.tile_size / 2
    grid_y0 = -s.n_rows * s.tile_size / 2
    band = 2 * s.cell_size + 1e-9
    worst = 0.0
    for j in range(1, s.n_cols):
        cols = np.nonzero(np.abs(xs - (grid_x0 + j * s.tile_size)) <= band)[0]
        block = L[:, cols.min():cols.max() + 1]
        worst = max(worst, float(np.abs(np.diff(block, axis=1)).max()))
    for i in range(1, s.n_rows):
        rows = np.nonzero(np.abs(ys - (grid_y0 + i * s.tile_size)) <= band)[0]
        block = L[rows.min():rows.max() + 1, :]
        worst = max(worst, float(np.abs(np.diff(block, axis=0)).max()))
    amp = max(terrain.rough_amplitude(d) for d in terrain_suite.DIFFICULTIES)
    assert worst <= 0.7 * amp, (worst, amp)


def test_rebuilt_arena_is_not_served_from_cache():
    """Two different arenas written to one file set inside the same second, and
    the second compile has to be the second arena. MuJoCo caches heightfields by
    filename at one-second mtime resolution, so without write_arena's mtime bump
    this silently loads the first arena's elevation data into a scene whose
    lookup grid describes the second -- physics and heights disagreeing with no
    error. See build_terrain._force_new_mtime."""
    scene = paths.terrain_paths(TEST_ARENA)["scene"]
    small = terrain.generate(seed=0, n_rows=10)
    big = terrain.generate(seed=0, n_rows=12)
    assert small.spec.hfield.nrow != big.spec.hfield.nrow
    try:
        for first, second in ((small, big), (big, small)):
            # in place, and with a delete in between -- deleting does NOT clear
            # MuJoCo's cache entry for the name, so a fresh file with a fresh
            # same-second mtime collides with what the deleted one left behind.
            # That is what the fixtures in this suite do at teardown.
            for delete_between in (False, True):
                build_terrain.write_arena(first, TEST_ARENA)
                got_first = mujoco.MjModel.from_xml_path(str(scene)).hfield_nrow[0]
                if delete_between:
                    for p in paths.terrain_paths(TEST_ARENA).values():
                        p.unlink(missing_ok=True)
                build_terrain.write_arena(second, TEST_ARENA)
                got_second = mujoco.MjModel.from_xml_path(str(scene)).hfield_nrow[0]
                assert got_first == first.spec.hfield.nrow
                assert got_second == second.spec.hfield.nrow, (
                    f"stale cached heightfield (delete_between={delete_between})"
                )
    finally:
        for p in paths.terrain_paths(TEST_ARENA).values():
            p.unlink(missing_ok=True)


def test_eval_arena_compiles():
    """The scene MuJoCo actually loads, written to the `test` file set so the
    real eval arena is never touched by a test run."""
    arena = terrain.generate(**terrain_suite.eval_arena_kwargs())
    build_terrain.write_arena(arena, TEST_ARENA)
    try:
        m = mujoco.MjModel.from_xml_path(
            str(paths.terrain_paths(TEST_ARENA)["scene"])
        )
        assert m.nhfield == 1
        assert m.hfield_nrow[0] == arena.spec.hfield.nrow
        n_boxes = sum(
            1 for i in range(m.ngeom)
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith(
                "terrain_box_"
            )
        )
        assert n_boxes == len(arena.boxes)
        assert m.key("home").id >= 0
    finally:
        for p in paths.terrain_paths(TEST_ARENA).values():
            p.unlink(missing_ok=True)


def test_a_zero_speed_has_no_budget():
    """The course is defined by distance, so a standing robot never completes a
    crossing -- better a clear error than a division by zero."""
    with pytest.raises(ValueError, match="commanded speed of 0"):
        terrain_suite.episode_budget(0.0, 0.02)
    # a negative command is the same course walked the other way round
    assert terrain_suite.episode_budget(-0.4, 0.02) == terrain_suite.episode_budget(
        0.4, 0.02
    )
