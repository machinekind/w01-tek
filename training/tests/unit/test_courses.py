"""Course benchmark: geometry, the frozen follower, and the score.

Everything here is pure numpy/math -- no checkpoint, no MJX -- so the parts
that define what a score MEANS are pinned by fast tests. The score is a
recorded number compared across months of runs, so the follower constants
and the physical normalizers are asserted explicitly: a test failure here is
the intended alarm when someone "improves" a constant.
"""

import math
import sys

import jax
import jax.numpy as jp
import numpy as np
import pytest

from wojtek_rl.courses import (
    GOAL_RADIUS_M,
    HEIGHT_CMD,
    LOOKAHEAD_M,
    NOMINAL_HEIGHT_M,
    SPIN_ENTER_RAD,
    SPIN_EXIT_RAD,
    STANCE_HALFWIDTH_M,
    SUBSCORE_CAP,
    YAW_MAX,
    Course,
    Pursuit,
    SpinCourse,
    arc,
    join,
    line,
    sine_slalom,
    aggregate,
    course_catalogue,
    friction_geom_ids,
    seed_result,
    spin_seed_result,
    step_budget,
)
from wojtek_rl.courses import runner


# -- geometry --------------------------------------------------------------


def test_line_from_origin_along_heading():
    assert np.allclose(line(10.0), [[0, 0], [10, 0]])
    assert np.allclose(line(2.0, start=(1.0, 1.0), heading=math.pi / 2), [[1, 1], [1, 3]])


def test_arc_leaves_tangent_to_heading_and_turns_left_for_positive_sweep():
    a = arc(2.0, math.pi / 2, n=64)
    assert np.allclose(a[0], [0.0, 0.0])
    # quarter left turn of radius 2 lands at (2, 2)
    assert np.allclose(a[-1], [2.0, 2.0], atol=1e-9)
    # first step is essentially along +x (tangent to heading 0)
    assert a[1][0] > 0 and abs(a[1][1]) < abs(a[1][0])
    # y grows monotonically: a left turn never dips right
    assert np.all(np.diff(a[:, 1]) >= -1e-12)


def test_arc_negative_sweep_turns_right():
    a = arc(2.0, -math.pi / 2, n=64)
    assert np.allclose(a[-1], [2.0, -2.0], atol=1e-9)


def test_full_circle_arc_closes_on_itself():
    a = arc(1.5, 2 * math.pi, n=96)
    assert np.allclose(a[-1], a[0], atol=1e-9)


def test_join_drops_duplicated_shared_endpoints():
    j = join(line(3.0), line(2.0, start=(3.0, 0.0)))
    assert len(j) == 3  # 2 + 2 minus the shared (3, 0)
    assert np.allclose(j, [[0, 0], [3, 0], [5, 0]])


def test_sine_slalom_amplitude_and_endpoints():
    s = sine_slalom(9.0, 0.5, 3.0, n=181)
    assert np.allclose(s[0], [0.0, 0.0])
    assert s[-1][0] == pytest.approx(9.0)
    assert np.abs(s[:, 1]).max() == pytest.approx(0.5, abs=1e-3)


# -- catalogue -------------------------------------------------------------


def test_catalogue_is_the_documented_twenty():
    assert set(course_catalogue()) == {
        "straight_10m", "arc_r3_90deg", "circle_r2", "circle_r075",
        "figure_eight_r15", "square_3m", "slalom_05m", "u_turn",
        "straight_slow", "straight_fast", "circle_r2_fast",
        "speed_steps_straight", "straight_slippery", "circle_r1_slippery",
        "straight_push", "straight_push_fast",
        "spin_left", "spin_right", "spin_slow", "spin_fast",
    }


def _path_courses():
    return {n: c for n, c in course_catalogue().items() if isinstance(c, Course)}


def test_every_course_starts_at_the_robot_origin():
    """Waypoints are in the START frame, so the first one must be (0, 0) --
    otherwise the course is laid out offset from wherever the robot is."""
    for name, c in _path_courses().items():
        assert np.allclose(c.waypoints[0], [0.0, 0.0]), name


def test_segment_speeds_broadcast_or_match_segment_count():
    for name, c in _path_courses().items():
        assert len(c.segment_speeds) == len(c.waypoints) - 1, name
        assert np.all(c.segment_speeds > 0.0), name


def test_speed_steps_is_the_only_course_with_a_varying_command():
    varying = {
        n for n, c in _path_courses().items()
        if len(np.unique(c.segment_speeds)) > 1
    }
    assert varying == {"speed_steps_straight"}


def test_speed_steps_blocks():
    c = course_catalogue()["speed_steps_straight"]
    assert np.allclose(c.segment_speeds, [0.2, 0.6, 1.0, 0.4])
    assert np.allclose(c.waypoints[:, 0], [0.0, 2.5, 5.0, 7.5, 10.0])


def test_only_the_slippery_rows_change_friction():
    friction = {
        n: f for n, c in course_catalogue().items()
        if (f := getattr(c, "friction", None)) is not None
    }
    assert set(friction) == {"straight_slippery", "circle_r1_slippery"}
    assert set(friction.values()) == {0.4}


def test_only_the_push_rows_carry_a_disturbance():
    pushed = {
        n: p for n, c in course_catalogue().items()
        if (p := getattr(c, "push_at_m", None)) is not None
    }
    assert set(pushed) == {"straight_push", "straight_push_fast"}
    assert set(pushed.values()) == {5.0}


def test_push_rows_differ_from_their_baselines_in_exactly_one_thing():
    """straight_push vs straight_10m: same geometry and speed, only the
    impulse. straight_push_fast vs straight_fast: likewise."""
    cat = course_catalogue()
    for push, base in [
        ("straight_push", "straight_10m"),
        ("straight_push_fast", "straight_fast"),
    ]:
        assert np.allclose(cat[push].waypoints, cat[base].waypoints)
        assert np.allclose(cat[push].segment_speeds, cat[base].segment_speeds)
        assert cat[push].friction == cat[base].friction
        assert cat[base].push_at_m is None


def test_slippery_rows_share_geometry_with_their_dry_baseline():
    cat = course_catalogue()
    assert np.allclose(
        cat["straight_slippery"].waypoints, cat["straight_10m"].waypoints
    )
    assert np.allclose(
        cat["straight_slippery"].segment_speeds,
        cat["straight_10m"].segment_speeds,
    )


def test_speed_rows_share_geometry_with_their_nominal_baseline():
    cat = course_catalogue()
    for fast, base in [
        ("straight_slow", "straight_10m"),
        ("straight_fast", "straight_10m"),
        ("circle_r2_fast", "circle_r2"),
    ]:
        assert np.allclose(cat[fast].waypoints, cat[base].waypoints), fast


def test_mismatched_speed_count_is_rejected():
    c = Course("bad", "", np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]), (0.5, 0.5, 0.5))
    with pytest.raises(ValueError, match="3 speeds for 2 segments"):
        _ = c.segment_speeds


# -- step budget -----------------------------------------------------------


def test_step_budget_scales_with_ideal_time():
    """10 m at 0.5 m/s is 20 s ideal -> 2.5x + 2 s = 52 s = 2600 steps."""
    c = course_catalogue()["straight_10m"]
    assert step_budget(c, 0.02) == 2600


def test_slow_course_gets_a_bigger_budget_than_the_fast_one():
    cat = course_catalogue()
    dt = 0.02
    assert step_budget(cat["straight_slow"], dt) > step_budget(cat["straight_10m"], dt)
    assert step_budget(cat["straight_fast"], dt) < step_budget(cat["straight_10m"], dt)


def test_step_budget_respects_the_hard_ceiling():
    huge = Course("huge", "", np.array([[0.0, 0.0], [10_000.0, 0.0]]), (0.1,))
    assert step_budget(huge, 0.02) == 6000


# -- follower --------------------------------------------------------------


def _straight_pursuit(speed=0.5, length=10.0):
    c = Course("t", "", line(length), (speed,))
    return Pursuit.from_course(c, 0.0, 0.0, 0.0)


def test_pursuit_resamples_to_the_full_course_length():
    p = _straight_pursuit()
    assert p.total_length == pytest.approx(10.0, abs=1e-6)
    assert len(p.pts) == len(p.speed) == len(p.cum_s) == len(p.tangents)


def test_on_path_and_on_heading_commands_straight_ahead():
    p = _straight_pursuit()
    cmd, xte, s, reached = p.command(1.0, 0.0, 0.0)
    assert float(cmd[0]) == pytest.approx(0.5, abs=1e-6)  # full commanded speed
    assert float(cmd[1]) == 0.0  # never strafes: the follower is non-holonomic
    assert float(cmd[2]) == pytest.approx(0.0, abs=1e-6)  # no yaw needed
    assert float(cmd[3]) == HEIGHT_CMD
    assert xte == pytest.approx(0.0, abs=1e-9)
    assert s == pytest.approx(1.0, abs=0.02)
    assert not reached


def test_cross_track_error_sign_is_positive_to_the_left():
    p = _straight_pursuit()
    _, left, _, _ = p.command(1.0, 0.3, 0.0)
    assert left > 0
    p = _straight_pursuit()
    _, right, _, _ = p.command(1.0, -0.3, 0.0)
    assert right < 0
    assert abs(left) == pytest.approx(0.3, abs=1e-6)


def test_offset_left_of_the_path_steers_right():
    """Robot 0.2 m left of a straight path, heading along it -> negative
    (clockwise) yaw rate to come back."""
    p = _straight_pursuit()
    cmd, _, _, _ = p.command(1.0, 0.2, 0.0)
    assert float(cmd[2]) < 0.0


# cmd is a jp.array, i.e. float32 in this project (no x64 config anywhere),
# so a clipped 1.2 reads back as 1.2000000476837158. Same tolerance dance as
# test_battery.py's _close.
F32 = 1e-5


def test_heading_error_beyond_the_spin_threshold_yaws_in_place():
    p = _straight_pursuit()
    # facing 90 deg away from a path that runs along +x
    cmd, _, _, _ = p.command(1.0, 0.0, math.pi / 2)
    assert float(cmd[0]) == 0.0  # no forward motion while pivoting
    assert abs(float(cmd[2])) > 0.0
    assert abs(float(cmd[2])) <= YAW_MAX + F32


def test_yaw_rate_never_exceeds_the_frozen_cap():
    p = Pursuit.from_course(Course("t", "", arc(0.4, math.pi), (1.0,)), 0.0, 0.0, 0.0)
    for yaw in np.linspace(-math.pi, math.pi, 25):
        cmd, _, _, _ = p.command(0.1, 0.0, float(yaw))
        assert abs(float(cmd[2])) <= YAW_MAX + F32
        p.i = 0  # re-probe from the course start


def test_forward_speed_is_reduced_by_heading_error():
    """Inside the spin threshold, vx = v_target * cos(alpha) -- a moderate
    heading error slows the robot rather than cutting the corner."""
    straight_cmd, _, _, _ = _straight_pursuit().command(1.0, 0.0, 0.0)
    skew_cmd, _, _, _ = _straight_pursuit().command(1.0, 0.0, 0.5)
    assert 0.0 < float(skew_cmd[0]) < float(straight_cmd[0])


def _walk_index_to(p, x):
    """Advance the monotone progress index as a real rollout would, by
    querying the follower along the path up to arclength/x-position `x`.
    Teleporting the probe robot straight to the goal is NOT representative:
    the window caps how far the index moves per call."""
    for k in range(0, len(p.pts), 25):
        if p.pts[k][0] > x:
            break
        p.command(float(p.pts[k][0]), float(p.pts[k][1]), 0.0)


def test_reaching_the_final_point_reports_completion():
    p = _straight_pursuit()
    _walk_index_to(p, 10.0 - GOAL_RADIUS_M / 2)
    _, _, _, reached = p.command(10.0 - GOAL_RADIUS_M / 2, 0.0, 0.0)
    assert reached


def test_just_outside_the_goal_radius_is_not_completion():
    p = _straight_pursuit()
    _walk_index_to(p, 10.0 - GOAL_RADIUS_M * 1.5)
    _, _, _, reached = p.command(10.0 - GOAL_RADIUS_M * 1.5, 0.0, 0.0)
    assert not reached


def test_a_closed_course_does_not_complete_at_its_start():
    """A circle's final waypoint IS its start point. Distance-only completion
    'finished' every closed course 1.9 s in, on the lead-in -- measured on
    the stiff_b keeper before GOAL_MIN_PROGRESS_M existed. Standing right on
    the start/end point with zero progress must not read as done."""
    for name in ("circle_r2", "circle_r075", "figure_eight_r15", "square_3m"):
        c = course_catalogue()[name]
        p = Pursuit.from_course(c, 0.0, 0.0, 0.0)
        end = p.pts[-1]
        _, _, _, reached = p.command(float(end[0]), float(end[1]), 0.0)
        assert not reached, name


def test_a_closed_course_completes_after_walking_the_loop():
    c = course_catalogue()["circle_r2"]
    p = Pursuit.from_course(c, 0.0, 0.0, 0.0)
    # drive the index around the whole loop the way a rollout does
    for k in range(0, len(p.pts), 25):
        p.command(float(p.pts[k][0]), float(p.pts[k][1]), 0.0)
    end = p.pts[-1]
    _, _, _, reached = p.command(float(end[0]), float(end[1]), 0.0)
    assert reached


def test_progress_is_monotone_so_a_crossing_cannot_snap_backwards():
    """figure_eight crosses itself at the origin. After a lap the closest
    point must stay ahead, not jump back to the start of the course."""
    c = course_catalogue()["figure_eight_r15"]
    p = Pursuit.from_course(c, 0.0, 0.0, 0.0)
    seen = []
    # walk the polyline itself, which passes through the crossing twice
    for k in range(0, len(p.pts), 5):
        x, y = p.pts[k]
        _, _, s, _ = p.command(float(x), float(y), 0.0)
        seen.append(s)
    assert all(b >= a - 1e-9 for a, b in zip(seen, seen[1:]))


def test_course_is_laid_out_in_the_robots_start_frame():
    """A robot starting at (5, -2) facing +y gets the same course rotated
    and translated onto it, not a course anchored at the world origin."""
    c = Course("t", "", line(3.0), (0.5,))
    p = Pursuit.from_course(c, 5.0, -2.0, math.pi / 2)
    assert np.allclose(p.pts[0], [5.0, -2.0], atol=1e-9)
    assert np.allclose(p.pts[-1], [5.0, 1.0], atol=1e-6)  # 3 m along +y


def test_lookahead_point_is_the_frozen_distance_ahead():
    p = _straight_pursuit()
    # A robot exactly on the path sees the lookahead point straight ahead;
    # place it laterally and the geometry must match LOOKAHEAD_M.
    cmd, _, _, _ = p.command(0.0, LOOKAHEAD_M, 0.0)
    alpha = math.atan2(-LOOKAHEAD_M, LOOKAHEAD_M)  # 45 deg right, by construction
    expected = np.clip(
        2.0 * (0.5 * math.cos(alpha)) * math.sin(alpha) / LOOKAHEAD_M,
        -YAW_MAX, YAW_MAX,
    )
    assert float(cmd[2]) == pytest.approx(float(expected), abs=1e-6)


def test_spin_thresholds_and_yaw_cap_are_the_frozen_values():
    assert SPIN_ENTER_RAD == pytest.approx(math.radians(60.0), abs=0.01)
    assert SPIN_EXIT_RAD == pytest.approx(math.radians(20.0), abs=0.01)
    # Inside the trained wz box [-1, 1]: commanding wz beyond it froze the
    # stiff_b keeper solid (60 s of wz=-1.2 moved yaw by under a degree).
    assert YAW_MAX == 1.0


def test_spin_branch_has_hysteresis_so_it_cannot_chatter():
    """The slalom deadlock: alpha sitting ON a single spin threshold flips
    the command between walk and spin every step, and the policy's action
    filter averages that to a permanent stand. With hysteresis, once
    spinning the follower keeps spinning until alpha is small."""
    p = _straight_pursuit()
    # 90 deg off-heading: enters the spin branch
    cmd, _, _, _ = p.command(1.0, 0.0, math.pi / 2)
    assert p.spinning and float(cmd[0]) == 0.0
    # heading improved to 40 deg -- between exit (20) and enter (60):
    # WITHOUT hysteresis this would flip back to walking
    cmd, _, _, _ = p.command(1.0, 0.0, math.radians(40.0))
    assert p.spinning and float(cmd[0]) == 0.0
    # below the exit threshold: walking resumes
    cmd, _, _, _ = p.command(1.0, 0.0, math.radians(10.0))
    assert not p.spinning and float(cmd[0]) > 0.0


def test_walk_branch_does_not_enter_spin_below_the_enter_threshold():
    p = _straight_pursuit()
    cmd, _, _, _ = p.command(1.0, 0.0, math.radians(40.0))
    assert not p.spinning and float(cmd[0]) > 0.0


# -- model swapping: the two bugs that made scenarios silently meaningless --


def test_friction_geoms_include_the_feet_not_just_the_floor():
    """Equal-priority contacts take the element-wise MAX of the two geoms'
    friction, so lowering only the floor leaves the feet's value winning and
    the slippery scenarios measure nothing."""
    ids = friction_geom_ids(0, np.array([26, 41, 56, 71]))
    assert list(ids) == [0, 26, 41, 56, 71]


def test_friction_geoms_dedupe_and_accept_scalars():
    assert list(friction_geom_ids(3, np.array([3, 7]))) == [3, 7]


def test_rejitting_after_a_model_swap_picks_up_the_new_value():
    """run_courses mutates env._mjx_model between friction groups and
    re-jits. The trace bakes the model in as constants, so this asserts the
    re-jit really does retrace rather than reuse the old executable."""

    class _FakeEnv:
        def __init__(self):
            self.scale = 2.0  # stands in for _mjx_model

        def step(self, state, action):
            return state * self.scale + action

    env = _FakeEnv()
    assert float(jax.jit(env.step)(jp.array(1.0), jp.array(0.0))) == 2.0
    env.scale = 10.0
    assert float(jax.jit(env.step)(jp.array(1.0), jp.array(0.0))) == 10.0


# -- scoring ---------------------------------------------------------------


def _rec(n=600, xte=0.0, v_err=0.0, h_err=0.0, slip=0.0, cmd_v=0.5, dt=0.02):
    """A synthetic rollout with hand-set error levels.

    qvel is a pure 1 Hz sine so the >5 Hz vibration fraction is ~0 and the
    smoothness sub-score sits at the cap unless a test says otherwise.
    """
    t = np.arange(n) * dt
    return {
        "xy": np.stack([cmd_v * t, np.zeros(n)], -1),
        "s": cmd_v * t,
        "xte": np.full(n, xte),
        "cmd_v": np.full(n, cmd_v),
        "v_fwd": np.full(n, cmd_v - v_err),
        "v_planar": np.full(n, cmd_v),
        "h": np.full(n, HEIGHT_CMD + h_err),
        "qvel": np.sin(2 * np.pi * 1.0 * t)[:, None] * np.ones((1, 12)),
        "slip_speed": np.full(n, slip),
    }


def _info(n=600, completed=True, fell_at=None, length=10.0):
    return {
        "fell_at": fell_at,
        "completed": completed,
        "steps": n,
        "frames": [],
        "total_length": length,
    }


def test_tracking_subscore_is_stance_halfwidth_over_rms_crosstrack():
    r = seed_result(_rec(xte=STANCE_HALFWIDTH_M), _info(), 0.02)
    assert r["subscores"]["tracking"] == pytest.approx(1.0, abs=1e-3)
    r = seed_result(_rec(xte=STANCE_HALFWIDTH_M / 10), _info(), 0.02)
    assert r["subscores"]["tracking"] == pytest.approx(10.0, rel=1e-2)


def test_height_subscore_is_nominal_height_over_rms_height_error():
    r = seed_result(_rec(h_err=NOMINAL_HEIGHT_M), _info(), 0.02)
    assert r["subscores"]["height"] == pytest.approx(1.0, abs=1e-3)


def test_speed_subscore_is_commanded_speed_over_rms_speed_error():
    r = seed_result(_rec(cmd_v=0.5, v_err=0.5), _info(), 0.02)
    assert r["subscores"]["speed"] == pytest.approx(1.0, abs=1e-3)
    r = seed_result(_rec(cmd_v=0.5, v_err=0.05), _info(), 0.02)
    assert r["subscores"]["speed"] == pytest.approx(10.0, rel=1e-2)


def test_grip_subscore_is_one_when_feet_slide_as_far_as_the_body_moves():
    r = seed_result(_rec(cmd_v=0.5, slip=0.5), _info(), 0.02)
    assert r["subscores"]["grip"] == pytest.approx(1.0, abs=1e-3)


def test_subscores_are_unbounded_above_and_capped_only_for_json():
    r = seed_result(_rec(), _info(), 0.02)  # zero error everywhere
    assert r["subscores"]["tracking"] == SUBSCORE_CAP
    assert r["score"] == SUBSCORE_CAP


def test_score_is_the_weakest_subscore_not_an_average():
    """One bad axis (tracking at 1.0) cannot be diluted by four perfect
    ones -- that is the whole point of the min."""
    r = seed_result(_rec(xte=STANCE_HALFWIDTH_M), _info(), 0.02)
    assert r["binding"] == "tracking"
    assert r["score"] == pytest.approx(r["subscores"]["tracking"], abs=1e-6)
    assert r["score"] < min(
        v for k, v in r["subscores"].items() if k != "tracking"
    )


def test_a_fall_scores_zero_however_good_the_metrics_look():
    r = seed_result(_rec(n=300), _info(n=300, completed=False, fell_at=299), 0.02)
    assert r["score"] == 0.0
    assert r["fell_at"] == 299
    # metrics are still reported, so the failure is diagnosable
    assert r["subscores"] is not None
    assert r["raw"]["xte_rms_m"] == 0.0


def test_an_unfinished_course_scores_zero_even_without_a_fall():
    r = seed_result(_rec(n=400), _info(n=400, completed=False), 0.02)
    assert r["score"] == 0.0
    assert r["fell_at"] is None


def test_a_rollout_too_short_to_measure_scores_zero_with_no_subscores():
    r = seed_result({}, _info(n=0, completed=False), 0.02)
    assert r["score"] == 0.0
    assert r["subscores"] is None
    assert r["progress_m"] == 0.0


def test_speed_is_scored_only_where_forward_motion_was_commanded():
    """The yaw-in-place branch commands vx = 0; charging speed error there
    would penalise a corner twice."""
    rec = _rec(n=600, v_err=0.0)
    rec["cmd_v"] = np.concatenate([np.zeros(300), np.full(300, 0.5)])
    rec["v_fwd"] = np.zeros(600)  # never moves forward at all
    r = seed_result(rec, _info(), 0.02)
    # error is judged over the moving half only: |0.5 - 0| = 0.5 vs mean cmd 0.5
    assert r["subscores"]["speed"] == pytest.approx(1.0, abs=1e-3)


def test_vibration_binds_the_score_when_joints_buzz():
    rec = _rec()
    t = np.arange(len(rec["s"])) * 0.02
    rec["qvel"] = np.sin(2 * np.pi * 20.0 * t)[:, None] * np.ones((1, 12))
    r = seed_result(rec, _info(), 0.02)
    assert r["binding"] == "smoothness"
    assert r["subscores"]["smoothness"] < 2.0


# -- aggregation -----------------------------------------------------------


def test_aggregate_reports_median_and_worst_seed():
    seeds = [
        seed_result(_rec(xte=x), _info(), 0.02)
        for x in (0.01, 0.02, 0.02, 0.5)
    ]
    agg = aggregate(seeds)
    assert agg["seeds"] == 4
    assert agg["score_worst"] < agg["score_median"]
    assert agg["score_worst"] == min(s["score"] for s in seeds)
    assert agg["falls"] == 0
    assert agg["completed"] == 4


def test_aggregate_counts_falls_and_keeps_worst_at_zero():
    good = seed_result(_rec(xte=0.02), _info(), 0.02)
    bad = seed_result(_rec(n=300), _info(n=300, completed=False, fell_at=299), 0.02)
    agg = aggregate([good, good, good, bad])
    assert agg["falls"] == 1
    assert agg["completed"] == 3
    assert agg["score_worst"] == 0.0
    assert agg["score_median"] > 0.0  # median survives a single bad seed
    assert len(agg["per_seed"]) == 4


def test_aggregate_survives_seeds_that_produced_no_subscores():
    dead = seed_result({}, _info(n=0, completed=False), 0.02)
    agg = aggregate([dead, dead])
    assert agg["score_median"] == 0.0
    assert "subscore_median" not in agg


# -- spin scenarios ----------------------------------------------------------


def _spin_courses():
    return {n: c for n, c in course_catalogue().items() if isinstance(c, SpinCourse)}


def test_spin_family_covers_both_directions_and_rates():
    spins = _spin_courses()
    assert set(spins) == {"spin_left", "spin_right", "spin_slow", "spin_fast"}
    # the chirality pair differs ONLY in sign
    assert spins["spin_left"].wz == -spins["spin_right"].wz
    assert spins["spin_left"].wz > 0  # + is CCW/left
    # the rate rows share the left direction so rate is the only variable
    assert spins["spin_slow"].wz > 0 and spins["spin_fast"].wz > 0
    assert spins["spin_slow"].wz < spins["spin_left"].wz < spins["spin_fast"].wz
    assert all(c.turns == 1.0 for c in spins.values())


def test_spin_step_budget_scales_with_commanded_rate():
    """One turn at 0.8 rad/s is 7.85 s ideal -> 2.5x + 2 s = 21.6 s."""
    spins = _spin_courses()
    dt = 0.02
    assert step_budget(spins["spin_left"], dt) == round((2.5 * 2 * math.pi / 0.8 + 2) / dt)
    assert step_budget(spins["spin_slow"], dt) > step_budget(spins["spin_left"], dt)
    assert step_budget(spins["spin_fast"], dt) < step_budget(spins["spin_left"], dt)
    # direction must not change the budget
    assert step_budget(spins["spin_left"], dt) == step_budget(spins["spin_right"], dt)


def _spin_rec(n=600, wz_cmd=0.8, wz_err=0.0, drift=0.0, h_err=0.0, dt=0.02):
    """Synthetic spin rollout at hand-set error levels; qvel is a 1 Hz sine
    so smoothness sits at the cap unless a test says otherwise."""
    t = np.arange(n) * dt
    wz = np.full(n, wz_cmd - math.copysign(wz_err, wz_cmd))
    return {
        "yaw_progress": np.abs(wz).cumsum() * dt,
        "wz": wz,
        "drift": np.full(n, drift),
        "h": np.full(n, HEIGHT_CMD + h_err),
        "qvel": np.sin(2 * np.pi * 1.0 * t)[:, None] * np.ones((1, 12)),
    }


def _spin_info(n=600, completed=True, fell_at=None):
    return {
        "fell_at": fell_at,
        "completed": completed,
        "steps": n,
        "frames": [],
        "total_length": 2 * math.pi,
    }


def test_spin_rotation_subscore_is_command_over_rate_error():
    r = spin_seed_result(_spin_rec(wz_cmd=0.8, wz_err=0.8), _spin_info(), 0.02, 0.8)
    assert r["subscores"]["rotation"] == pytest.approx(1.0, abs=1e-3)
    r = spin_seed_result(_spin_rec(wz_cmd=0.8, wz_err=0.08), _spin_info(), 0.02, 0.8)
    assert r["subscores"]["rotation"] == pytest.approx(10.0, rel=1e-2)


def test_spin_drift_subscore_is_stance_halfwidth_over_max_drift():
    r = spin_seed_result(
        _spin_rec(drift=STANCE_HALFWIDTH_M), _spin_info(), 0.02, 0.8
    )
    assert r["subscores"]["drift"] == pytest.approx(1.0, abs=1e-3)


def test_spin_has_no_path_subscores():
    r = spin_seed_result(_spin_rec(), _spin_info(), 0.02, 0.8)
    assert set(r["subscores"]) == {"rotation", "drift", "height", "smoothness"}


def test_spin_incomplete_rotation_scores_zero():
    """A robot that stands instead of spinning (the shipped stiff_b keeper's
    right spin, measured 4 deg in 5 s) times out and must score 0."""
    r = spin_seed_result(
        _spin_rec(n=600, wz_cmd=-0.8, wz_err=0.8),
        _spin_info(completed=False),
        0.02,
        -0.8,
    )
    assert r["score"] == 0.0
    assert r["subscores"] is not None  # still diagnosable
    assert r["subscores"]["rotation"] == pytest.approx(1.0, abs=1e-3)


def test_spin_fall_scores_zero():
    r = spin_seed_result(
        _spin_rec(n=300), _spin_info(n=300, completed=False, fell_at=299), 0.02, 0.8
    )
    assert r["score"] == 0.0 and r["fell_at"] == 299


def test_spin_right_command_is_negative_and_scored_by_magnitude():
    """RMS error is direction-agnostic; the normalizer must be |wz_cmd| so a
    right spin cannot score negative."""
    r = spin_seed_result(
        _spin_rec(wz_cmd=-0.8, wz_err=0.08), _spin_info(), 0.02, -0.8
    )
    assert r["subscores"]["rotation"] == pytest.approx(10.0, rel=1e-2)
    assert r["score"] > 0


# -- CLI: --seed-base flag passthrough --------------------------------------
#
# courses runs CPU-deterministic (run.sh sets JAX_PLATFORMS=cpu for the
# `courses` subcommand), so without --seed-base a rescan reuses seed=0..N-1
# bit-for-bit and measures exactly zero replicate delta. These tests mock
# `run_courses` (the function that owns the seed loop) rather than running a
# real checkpoint, per the tests/unit env-free guard.


def _fake_run_courses(calls):
    def fake(run_dir, **kwargs):
        calls["run_dir"] = run_dir
        calls.update(kwargs)
        return {"run": "r", "checkpoint": "c", "seeds": kwargs["seeds"], "courses": {}}

    return fake


def test_seed_base_flag_reaches_run_courses(tmp_path, monkeypatch):
    calls = {}
    monkeypatch.setattr(runner, "run_courses", _fake_run_courses(calls))
    monkeypatch.setattr(
        sys, "argv",
        ["courses", "--run", str(tmp_path), "--seed-base", "100"],
    )
    runner.main()
    assert calls["run_dir"] == tmp_path
    # argparse must coerce via type=int: a string "100" would still reach
    # run_courses without error but break `seed_base + s` arithmetic in the
    # seed loop (str + int raises, or worse, silently concatenates under a
    # different signature).
    assert calls["seed_base"] == 100
    assert isinstance(calls["seed_base"], int)


def test_seed_base_defaults_to_zero(tmp_path, monkeypatch):
    calls = {}
    monkeypatch.setattr(runner, "run_courses", _fake_run_courses(calls))
    monkeypatch.setattr(sys, "argv", ["courses", "--run", str(tmp_path)])
    runner.main()
    assert calls["seed_base"] == 0
