"""Terrain-scan reduction, gates, and baseline handling.

Everything here is pure: dicts and numpy arrays in, dicts out. No checkpoint,
no jax rollout -- the rollout itself is exercised by running the scan.
"""

import json

import numpy as np
import pytest

from wojtek_rl import terrain_scan, terrain_suite

N = terrain_suite.RUNS_PER_CELL_SPEED
CROSSINGS = terrain_suite.CROSSINGS


def _out(crossings, fell):
    return {
        "crossings": np.array(crossings),
        "fell": np.array(fell, dtype=bool),
        "saturation": np.full(len(crossings), 0.1),
        "track_err": np.full(len(crossings), 0.05),
        "clearance": np.full(len(crossings), 0.01),
        "counted": np.full(len(crossings), 100.0),
        "steps": 500,
        "nacon_max": 77,
    }


# -- reduction -----------------------------------------------------------------


def test_pass_requires_all_four_crossings_and_no_fall():
    crossings = [CROSSINGS, CROSSINGS, CROSSINGS - 1, 0]
    fell = [False, True, False, True]
    r = terrain_scan.reduce_runs(_out(crossings, fell))
    assert r.passed == 1  # only the first
    assert r.of == 4
    assert r.falls == 2
    assert r.timeouts == 1  # ran out of budget mid-course without falling
    assert r.crossings_mean == pytest.approx((4 + 4 + 3 + 0) / 4)
    assert r.nacon_max == 77


def test_a_fall_fails_the_run():
    """Falling during the course fails it, whatever the crossing count. The
    rollout is what makes sure a fall AFTER the fourth crossing is never
    recorded -- see fall_progress."""
    r = terrain_scan.reduce_runs(_out([CROSSINGS - 1], [True]))
    assert r.passed == 0
    assert r.falls == 1
    assert r.timeouts == 0


def test_cell_entry_carries_bar_and_provenance():
    cell = terrain_suite.CELLS_BY_NAME["pyramid_stairs_5cm"]
    r = terrain_scan.reduce_runs(_out([CROSSINGS] * N, [False] * N))
    at_plan = terrain_scan.cell_entry(cell, 0.4, r)
    assert at_plan["passed"] == N and at_plan["of"] == N
    assert at_plan["bar"] == 26 and at_plan["provenance"] == "plan"
    assert at_plan["bar_fraction"] == 0.8
    off_plan = terrain_scan.cell_entry(cell, 0.7, r)
    assert off_plan["bar"] == 26 and off_plan["provenance"] == "provisional"
    tracked = terrain_scan.cell_entry(
        terrain_suite.CELLS_BY_NAME["pyramid_stairs_9cm"], 0.4, r
    )
    assert tracked["bar"] is None and tracked["provenance"] == "tracked"


# -- absolute gate -------------------------------------------------------------


def _cells(entries):
    """{cell: {speed: entry}} from (cell_name, speed, passed) triples."""
    out = {}
    for name, speed, passed in entries:
        cell = terrain_suite.CELLS_BY_NAME[name]
        r = terrain_scan.reduce_runs(
            _out([CROSSINGS] * passed + [0] * (N - passed), [False] * N)
        )
        out.setdefault(name, {})[str(speed)] = terrain_scan.cell_entry(cell, speed, r)
    return out


def test_absolute_gate_passes_on_the_bar():
    """The bar is a floor, not a strict inequality: exactly 26 of 32 passes."""
    gate = terrain_scan.absolute_gate(_cells([("pyramid_stairs_5cm", 0.4, 26)]))
    assert gate["verdict"] == "pass"
    assert gate["checked"] == 1


def test_a_partial_scan_is_incomplete_not_a_pass():
    """"The four cells I measured are fine" must not read as "the policy
    passed"."""
    cells = _cells([("pyramid_stairs_5cm", 0.4, 32)])
    assert terrain_scan.absolute_gate(cells)["verdict"] == "pass"
    gate = terrain_scan.absolute_gate(cells, expect_gated=terrain_scan.gated_pairs())
    assert gate["verdict"] == "incomplete"
    assert gate["checked"] == 1 and gate["expected"] == 54
    # a real failure still outranks incompleteness
    bad = terrain_scan.absolute_gate(
        _cells([("pyramid_stairs_5cm", 0.4, 1)]), expect_gated=terrain_scan.gated_pairs()
    )
    assert bad["verdict"] == "fail"


def test_gated_pairs_counts_every_gated_cell_at_every_speed():
    assert terrain_scan.gated_pairs() == 18 * 3
    assert terrain_scan.gated_pairs(speeds=(0.4,)) == 18


def test_absolute_gate_fails_one_below():
    gate = terrain_scan.absolute_gate(_cells([("pyramid_stairs_5cm", 0.4, 25)]))
    assert gate["verdict"] == "fail"
    assert gate["failures"] == [
        {
            "cell": "pyramid_stairs_5cm",
            "speed": "0.4",
            "passed": 25,
            "bar": 26,
            "provenance": "plan",
        }
    ]


def test_absolute_gate_ignores_tracked_cells():
    """A tracked cell with zero passes is data, not a gate failure."""
    gate = terrain_scan.absolute_gate(
        _cells([("pyramid_stairs_9cm", 0.4, 0), ("discrete_obstacles_8cm", 0.4, 0)])
    )
    assert gate["verdict"] == "pass"
    assert gate["checked"] == 0


def test_absolute_gate_reports_provisional_provenance():
    """A provisional failure has to be readable as one: the plan sets no bar
    away from 0.4 m/s."""
    gate = terrain_scan.absolute_gate(_cells([("pyramid_stairs_5cm", 0.2, 10)]))
    assert gate["verdict"] == "fail"
    assert gate["failures"][0]["provenance"] == "provisional"


# -- relative gate -------------------------------------------------------------


def _scan(cells, engine="warp", arena=None):
    return {
        "run": "candidate",
        "checkpoint": "1",
        "engine": engine,
        "arena": arena if arena is not None else terrain_suite.arena_fingerprint(),
        "cells": cells,
    }


def test_relative_gate_without_a_baseline_says_so():
    gate = terrain_scan.relative_gate(_scan({}), None)
    assert gate["verdict"] == "no baseline"


def test_relative_gate_allows_a_small_drop():
    now = _scan(_cells([("pyramid_stairs_5cm", 0.4, 29)]))
    base = _scan(_cells([("pyramid_stairs_5cm", 0.4, 32)]))
    # 32/32 -> 29/32 is 9.4 points, inside the 10-point limit
    gate = terrain_scan.relative_gate(now, base)
    assert gate["verdict"] == "pass", gate
    assert gate["drops"] == []


def test_relative_gate_fails_a_big_drop():
    now = _scan(_cells([("pyramid_stairs_5cm", 0.4, 28)]))
    base = _scan(_cells([("pyramid_stairs_5cm", 0.4, 32)]))
    gate = terrain_scan.relative_gate(now, base)  # 12.5 points
    assert gate["verdict"] == "fail"
    assert gate["drops"][0]["drop"] == pytest.approx(12.5)


def test_relative_gate_gains_are_never_failures():
    now = _scan(_cells([("pyramid_stairs_5cm", 0.4, 32)]))
    base = _scan(_cells([("pyramid_stairs_5cm", 0.4, 10)]))
    assert terrain_scan.relative_gate(now, base)["verdict"] == "pass"


def test_relative_gate_refuses_a_different_arena():
    """Scores from two terrains are not comparable, so the gate refuses rather
    than reporting a difference."""
    other = dict(terrain_suite.arena_fingerprint(), rows=10)
    now = _scan(_cells([("pyramid_stairs_5cm", 0.4, 5)]))
    base = _scan(_cells([("pyramid_stairs_5cm", 0.4, 32)]), arena=other)
    gate = terrain_scan.relative_gate(now, base)
    assert gate["verdict"] == "refused"
    assert "arena" in gate["notes"][0]


def test_relative_gate_refuses_a_different_engine():
    now = _scan(_cells([("pyramid_stairs_5cm", 0.4, 5)]), engine="warp")
    base = _scan(_cells([("pyramid_stairs_5cm", 0.4, 32)]), engine="jax")
    gate = terrain_scan.relative_gate(now, base)
    assert gate["verdict"] == "refused"
    assert "engine" in gate["notes"][0]


def test_a_new_cell_has_nothing_to_compare_against():
    """Otherwise the 6.5 cm steps cell fails its own first gate."""
    now = _scan(
        _cells([("pyramid_stairs_5cm", 0.4, 30), ("discrete_obstacles_6.5cm", 0.4, 0)])
    )
    base = _scan(_cells([("pyramid_stairs_5cm", 0.4, 30)]))
    gate = terrain_scan.relative_gate(now, base)
    assert gate["verdict"] == "pass"
    assert gate["unmatched"] == ["discrete_obstacles_6.5cm@0.4"]


def test_a_cell_missing_at_one_speed_only_is_unmatched_at_that_speed():
    now = _scan(
        _cells([("pyramid_stairs_5cm", 0.4, 10), ("pyramid_stairs_5cm", 0.2, 10)])
    )
    base = _scan(_cells([("pyramid_stairs_5cm", 0.4, 32)]))
    gate = terrain_scan.relative_gate(now, base)
    assert gate["verdict"] == "fail"  # the 0.4 pair dropped
    assert gate["unmatched"] == ["pyramid_stairs_5cm@0.2"]


# -- baseline loading ----------------------------------------------------------


def test_load_baseline_from_a_file_and_a_directory(tmp_path):
    doc = {"run": "keeper", "cells": {}}
    path = tmp_path / "terrain_scan.json"
    path.write_text(json.dumps(doc))
    assert terrain_scan.load_baseline(str(path)) == doc
    assert terrain_scan.load_baseline(str(tmp_path)) == doc
    assert terrain_scan.load_baseline(None) is None
    assert terrain_scan.load_baseline("") is None


# -- command box ---------------------------------------------------------------


def test_command_box_warnings_flag_extrapolation():
    """The code default trains vx +-0.6, so the 0.7 cells are extrapolation
    there; every real preset trains -0.8 to 1.2 and stays quiet."""
    tight = {"env_config": {"command": {"vx": [-0.6, 0.6], "height": [0.09, 0.17]}}}
    assert any("0.7" in w for w in terrain_scan.command_box_warnings(tight))
    real = {"env_config": {"command": {"vx": [-0.8, 1.2], "height": [0.125, 0.125]}}}
    assert terrain_scan.command_box_warnings(real) == []


def test_command_box_warnings_flag_an_unreachable_height():
    run = {"env_config": {"command": {"vx": [-0.8, 1.2], "height": [0.15, 0.17]}}}
    warnings = terrain_scan.command_box_warnings(run)
    assert any("height" in w for w in warnings)


def test_command_box_warnings_tolerate_an_old_run():
    assert terrain_scan.command_box_warnings({}) == []


# -- the crossing rule ---------------------------------------------------------

OUT = terrain_suite.OUT_RADIUS
BACK = terrain_suite.BACK_RADIUS


def _advance(crossings, radius, running=True):
    return int(
        terrain_scan.crossing_progress(
            np.array([crossings]), np.array([radius]), np.array([running])
        )[0]
    )


def test_standing_on_the_pad_earns_no_crossing():
    """The failure this rule exists for: half the commanded distance can be
    walked on the flat pad without ever meeting the obstacle."""
    assert _advance(0, 0.0) == 0
    assert _advance(0, terrain_suite.EVAL_PAD_RADIUS) == 0
    # and not even reaching the outermost stair tread counts
    assert _advance(0, 1.25) == 0
    assert _advance(0, OUT - 1e-6) == 0


def test_outbound_leg_completes_at_the_turnaround_radius():
    assert _advance(0, OUT) == 1
    assert _advance(0, OUT + 0.3) == 1
    assert _advance(2, OUT) == 3  # even counts are outbound


def test_inbound_leg_completes_near_the_centre():
    """After an outbound leg the robot is past OUT_RADIUS, so the inbound test
    cannot fire on the same spot -- no double count."""
    assert _advance(1, OUT) == 1
    assert _advance(1, BACK + 1e-6) == 1
    assert _advance(1, BACK) == 2
    assert _advance(3, 0.0) == 4


def test_a_finished_run_never_advances():
    assert _advance(2, OUT, running=False) == 2
    assert _advance(0, OUT, running=False) == 0


def test_the_count_stops_at_four():
    assert _advance(terrain_suite.CROSSINGS, 0.0) == terrain_suite.CROSSINGS
    assert _advance(terrain_suite.CROSSINGS, OUT) == terrain_suite.CROSSINGS


def test_a_single_step_advances_by_at_most_one():
    """Otherwise a fast run could bank two crossings in one step."""
    for crossings in range(terrain_suite.CROSSINGS):
        for radius in (0.0, BACK, 1.0, OUT, 5.0):
            assert _advance(crossings, radius) - crossings in (0, 1)


def test_leg_sign_alternates_out_and_back():
    signs = [float(terrain_scan.leg_sign(np.array([c]))[0]) for c in range(4)]
    assert signs == [1.0, -1.0, 1.0, -1.0]


def test_a_fall_after_the_course_is_not_recorded():
    """A run keeps being stepped after its fourth crossing -- the batch is one
    program over 32 envs -- and its command still says walk out, so it walks on.
    Without the running gate, falling over a hundred steps after passing would
    turn the pass into a fall."""
    finished = np.array([False])  # no longer running: crossings == CROSSINGS
    assert not bool(
        terrain_scan.fall_progress(np.array([False]), np.array([True]), finished)[0]
    )
    # still on course: the fall counts
    assert bool(
        terrain_scan.fall_progress(np.array([False]), np.array([True]), np.array([True]))[0]
    )
    # and the flag is sticky once set, even after the run stops
    assert bool(
        terrain_scan.fall_progress(np.array([True]), np.array([False]), finished)[0]
    )


# -- arena validation ----------------------------------------------------------


def _eval_spec(**overrides):
    """A spec dict shaped like terrain_spec.json for the measurement arena."""
    from wojtek_rl import terrain

    spec = {
        "seed": terrain_suite.EVAL_SEED,
        "n_rows": len(terrain_suite.DIFFICULTIES),
        "ordered": terrain_suite.EVAL_ORDERED,
        "pad_radius": terrain_suite.EVAL_PAD_RADIUS,
        "n_steps": terrain.N_STEPS,
        "stair_platform_half": terrain.STAIR_PLATFORM_HALF,
        "tiles": [
            {"row": r, "col": 0, "difficulty": d}
            for r, d in enumerate(terrain_suite.DIFFICULTIES)
        ],
    }
    spec.update(overrides)
    return spec


def test_check_arena_accepts_the_measurement_course():
    terrain_scan.check_arena(_eval_spec())


@pytest.mark.parametrize(
    "overrides",
    [
        {"seed": 5},
        {"n_rows": 10},
        {"ordered": False},
        {"pad_radius": 0.6},
        {"n_steps": 4},
        {"stair_platform_half": 0.7},
    ],
    ids=["seed", "rows", "shuffled", "pad", "steps", "platform"],
)
def test_check_arena_refuses_a_different_arena(overrides):
    """The fingerprint the scan records comes from the suite's constants and the
    gate compares two scans on it, so an arena built with other parameters would
    file its numbers under a description of something else."""
    with pytest.raises(ValueError, match="not the measurement course"):
        terrain_scan.check_arena(_eval_spec(**overrides))


def test_check_arena_catches_a_shifted_row_table():
    """Same row count, different difficulties: the cells are defined by row
    index, so this is a different terrain, not a rescale."""
    rows = list(terrain_suite.DIFFICULTIES)
    rows[3] = rows[3] + 0.05
    spec = _eval_spec(
        tiles=[{"row": r, "col": 0, "difficulty": d} for r, d in enumerate(rows)]
    )
    with pytest.raises(ValueError, match="difficulties"):
        terrain_scan.check_arena(spec)


def test_check_arena_tolerates_a_spec_without_tiles():
    """An older spec has no per-tile difficulty to check; the scalar checks still
    apply and must not raise on the missing list."""
    terrain_scan.check_arena(_eval_spec(tiles=[]))


def test_metrics_average_only_over_runs_that_were_measured():
    """A run that fell inside the settle window has no metric steps, so its
    saturation, tracking error and clearance come back as zero. Averaging those
    in drags a cell's numbers toward zero exactly where falls are common -- the
    hard cells -- and in the direction that makes hard terrain look easy. On this
    case the diluted tracking error reads 2.7x better than the survivors'."""
    n_fell, n_ok = 20, 12
    counted = np.array([0.0] * n_fell + [900.0] * n_ok)
    out = {
        "crossings": np.array([0] * n_fell + [CROSSINGS] * n_ok),
        "fell": np.array([True] * n_fell + [False] * n_ok),
        "saturation": np.where(counted > 0, 0.30, 0.0),
        "track_err": np.where(counted > 0, 0.22, 0.0),
        "clearance": np.where(counted > 0, 0.02, 0.0),
        "counted": counted,
        "steps": 1036,
        "nacon_max": 90,
    }
    r = terrain_scan.reduce_runs(out)
    assert r.track_err == pytest.approx(0.22)
    assert r.saturation == pytest.approx(0.30)
    assert r.clearance == pytest.approx(0.02)
    assert r.measured == n_ok
    # outcome counts still cover every run: a fall is a result, not a gap
    assert r.of == n_fell + n_ok
    assert r.falls == n_fell
    assert r.passed == n_ok
    # and the count is reported, so a thin sample is visible
    cell = terrain_suite.CELLS_BY_NAME["pyramid_stairs_5cm"]
    assert terrain_scan.cell_entry(cell, 0.4, r)["measured"] == n_ok


def test_a_cell_where_every_run_died_early_reports_zero_not_nan():
    counted = np.zeros(4)
    out = {
        "crossings": np.zeros(4, dtype=int), "fell": np.ones(4, dtype=bool),
        "saturation": counted, "track_err": counted, "clearance": counted,
        "counted": counted, "steps": 60, "nacon_max": 0,
    }
    r = terrain_scan.reduce_runs(out)
    assert r.measured == 0
    assert r.track_err == 0.0 and r.saturation == 0.0
    assert r.falls == 4 and r.passed == 0
