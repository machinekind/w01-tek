"""SCAN local planner: map, footprint, guide, optimiser, executor.

Everything except the last test is pure numpy on a synthetic map -- no
MuJoCo, no scene assets, no policy -- so the planner's behaviour is pinned
independently of whatever the renderer and the locomotion checkpoint are
doing that week.
"""

import math

import numpy as np
import pytest

from wojtek_rl.midlevel import Forward, Turn
from wojtek_rl.navigation import NavConfig
from wojtek_rl.scan import Footprint, MapConfig, ScanExecutor, ScanPlanner, SlidingOccupancyMap
from wojtek_rl.scan.guide import GuideSearch
from wojtek_rl.scan.traj import (
    BSpline,
    TrajectoryOptimizer,
    build_rebound_pairs,
    init_from_path,
    optimize_with_rebound,
)

FLOOR_Z = 0.0
OBSTACLE_Z = 0.20


def floor_points(x0=-3.0, x1=3.0, y0=-3.0, y1=3.0, step=0.025):
    xs, ys = np.meshgrid(np.arange(x0, x1, step), np.arange(y0, y1, step))
    return np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, FLOOR_Z)], axis=1)


def box_points(x0, x1, y0, y1, step=0.025):
    xs, ys = np.meshgrid(np.arange(x0, x1 + 1e-9, step), np.arange(y0, y1 + 1e-9, step))
    return np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, OBSTACLE_Z)], axis=1)


def build_map(boxes=(), sensor=(0.0, 0.0), cfg=None) -> SlidingOccupancyMap:
    """One fused frame: floor everywhere, obstacles where asked.

    Floor and obstacles go in together because that is what one depth frame
    is -- fusing them as separate frames would let the floor rays free the
    obstacle cells they pass through, which a real ray (which stops at the
    obstacle) never does.
    """
    omap = SlidingOccupancyMap(cfg or MapConfig())
    pts = [floor_points()] + [box_points(*b) for b in boxes]
    omap.integrate_points(np.concatenate(pts), sensor)
    return omap


# -- map -------------------------------------------------------------------


def test_hits_occupy_and_rays_free():
    omap = build_map([(1.0, 1.4, -0.4, 0.4)])
    assert omap.is_inflated_occupied(1.2, 0.0)
    # A cell the rays crossed on the way to the floor is known and free.
    assert not omap.is_unknown(0.5, 0.0)
    assert not omap.is_inflated_occupied(0.2, 0.0)
    # Nothing was ever observed outside the sensed patch.
    assert omap.is_unknown(3.6, 3.6)


def test_band_only_obstacles():
    """Below z_step is a lip to step over, above z_up is clearance to walk
    under; neither belongs in the grid."""
    omap = SlidingOccupancyMap()
    low = box_points(1.0, 1.2, -0.2, 0.2)
    low[:, 2] = 0.02
    high = box_points(1.0, 1.2, -0.2, 0.2)
    high[:, 2] = 0.9
    omap.integrate_points(np.concatenate([low, high]), (0.0, 0.0))
    assert omap.occupied.sum() == 0


def test_logodds_forgets_a_moved_obstacle():
    omap = build_map([(1.0, 1.2, -0.2, 0.2)])
    assert omap.occupied.sum() > 0
    for _ in range(12):  # the chair moved: rays now reach the floor behind it
        omap.integrate_points(floor_points(0.5, 2.0, -1.0, 1.0), (0.0, 0.0))
    assert omap.occupied.sum() == 0


def test_sliding_window_clears_incoming_cells():
    omap = build_map([(1.0, 1.2, -0.2, 0.2)])
    known_before = int(omap.known.sum())
    omap.recenter(6.0, 0.0)  # slide most of the window off the observed area
    assert int(omap.known.sum()) < known_before
    assert omap.is_unknown(6.0, 0.0)
    # Memory is bounded: the buffer never grows with distance travelled.
    assert omap.logodds.shape == (omap.n, omap.n)


def test_footprint_disc_marks_standing_ground_free():
    omap = SlidingOccupancyMap()
    assert omap.is_unknown(0.0, 0.0)
    omap.mark_free_disc(0.0, 0.0, 0.3)
    assert not omap.is_unknown(0.0, 0.0)
    assert not omap.is_inflated_occupied(0.25, 0.0)


# -- footprint -------------------------------------------------------------


def test_footprint_is_yaw_aware():
    """A slot the body fits through lengthwise but not broadside."""
    # Sized against the current radius (0.20 = 4 cells): walls at +-0.26 sit
    # one cell beyond the inflation disc from the centreline, so the lengthwise
    # cylinders (on the centreline) clear while the broadside ones (at
    # +-d_off = +-0.25) land inside the walls.
    gap = 0.52
    omap = build_map([(0.6, 1.6, gap / 2, 1.2), (0.6, 1.6, -1.2, -gap / 2)])
    fp = Footprint()
    assert not fp.collides(omap, 1.1, 0.0, 0.0), "should fit driving along the slot"
    assert fp.collides(omap, 1.1, 0.0, math.pi / 2), "should not fit broadside"


def test_footprint_length_matches_body():
    fp = Footprint()
    # Visually tuned 2026-07-27 (/tune): discs on the leg pairs, 0.90 m
    # nose-to-tail. See the footprint module docstring for the waist trade.
    assert 0.85 <= fp.length <= 0.95


# -- guide -----------------------------------------------------------------


def test_guide_detours_around_a_blocking_box():
    omap = build_map([(1.0, 1.5, -0.5, 0.5)])
    search = GuideSearch(omap, Footprint())
    path, status = search.search((0.0, 0.0), (2.5, 0.0))
    assert status == "reached"
    pts = np.asarray(path)
    assert np.abs(pts[:, 1]).max() > 0.5, "path did not go around the box"
    yaws = np.arctan2(np.gradient(pts[:, 1]), np.gradient(pts[:, 0]))
    assert not Footprint().collides(omap, pts[:, 0], pts[:, 1], yaws).any()


def test_guide_keeps_clearance_when_it_is_cheap():
    """The shortest path grazes the inflated boundary; the cost layer should
    buy a little room when there is room to buy."""
    omap = build_map([(1.0, 1.5, -3.0, 0.0)])  # obstacle occupying y < 0
    search = GuideSearch(omap, Footprint())
    path, status = search.search((0.0, 0.6), (2.5, 0.6))
    assert status == "reached"
    ys = np.asarray(path)[:, 1]
    assert ys.min() > 0.25, f"path hugged the obstacle (min y {ys.min():.2f})"


def test_guide_reports_dead_end_with_boundary_fallback():
    """Goal walled off inside the window -> fallback toward the boundary,
    not a planning failure."""
    omap = build_map([(1.0, 1.3, -4.0, 4.0)])  # wall across the whole window
    search = GuideSearch(omap, Footprint())
    path, status = search.search((0.0, 0.0), (2.5, 0.0))
    assert status == "fallback"
    assert path, "fallback must still hand back an executable path"


# -- trajectory ------------------------------------------------------------


def test_bspline_boundary_state_is_pinned():
    path = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
    p0, v0 = np.array([0.0, 0.0]), np.array([0.3, 0.0])
    spline = init_from_path(path, 0.35, (p0, v0, np.zeros(2)), v_nom=0.4)
    assert np.allclose(spline.sample(0.0)[0], p0, atol=1e-9)
    assert np.allclose(spline.velocity(0.0)[0], v0, atol=1e-9)


def test_induced_yaw_follows_the_tangent():
    Q = np.stack([np.linspace(0, 2, 10), np.zeros(10)], axis=1)
    assert np.allclose(BSpline(Q, 0.35).induced_yaw(), 0.0)
    Q = np.stack([np.zeros(10), np.linspace(0, 2, 10)], axis=1)
    assert np.allclose(BSpline(Q, 0.35).induced_yaw(), math.pi / 2)


def test_rebound_optimisation_pushes_control_points_out():
    """A trajectory that clips an obstacle must be pushed back into free space.

    The realistic case the rebound term is for: the guide path is collision-free,
    but the smooth curve through it cuts the corner (here simulated by
    shoving the interior control points at the box). No ESDF is built --
    obstacles reach the optimiser only as {anchor, direction} pairs.
    """
    omap = build_map([(1.0, 1.5, -0.5, 0.5)])
    fp = Footprint()
    guide, status = GuideSearch(omap, fp).search((0.0, 0.0), (2.5, 0.0))
    assert status == "reached"
    spline = init_from_path(
        guide, 0.35, (np.zeros(2), np.zeros(2), np.zeros(2)), v_nom=0.4
    )
    # Corner-cut: drag the free control points toward the obstacle centre.
    free = slice(TrajectoryOptimizer.FIXED, len(spline.Q) - TrajectoryOptimizer.FIXED)
    spline.Q[free] += 0.6 * (np.array([1.25, 0.0]) - spline.Q[free])

    pairs = build_rebound_pairs(spline, guide, omap, fp)
    assert len(pairs), "a clipped trajectory must produce rebound pairs"
    before = fp.collides(omap, spline.Q[:, 0], spline.Q[:, 1], spline.induced_yaw()).sum()
    assert before, "the perturbation was supposed to cause collisions"

    out, _ = optimize_with_rebound(spline, guide, omap, fp, TrajectoryOptimizer())
    after = fp.collides(omap, out.Q[:, 0], out.Q[:, 1], out.induced_yaw()).sum()
    assert after == 0, f"still colliding: {before} -> {after}"


def test_feasibility_penalty_prices_overspeed():
    """The one-sided v/a penalty must charge for exceeding the caps and leave
    a feasible trajectory alone (Eq. 13)."""
    from wojtek_rl.scan.traj import OptConfig, ReboundPairs

    dt, opt = 0.35, TrajectoryOptimizer(OptConfig(v_max=0.4, lambda_s=0.0))
    fast = np.stack([np.arange(10) * 0.6, np.zeros(10)], axis=1)   # 1.7 m/s
    slow = np.stack([np.arange(10) * 0.10, np.zeros(10)], axis=1)  # 0.29 m/s
    j_fast, g_fast = opt.cost(fast, dt, ReboundPairs())
    j_slow, g_slow = opt.cost(slow, dt, ReboundPairs())
    assert j_fast > 1.0 and j_slow == 0.0
    # ... and its gradient pulls the fast control points together.
    assert np.linalg.norm(g_fast) > 0 and np.linalg.norm(g_slow) == 0


# -- planner / executor ----------------------------------------------------


def drive(planner_or_exec, goal=None, pose=(0.0, 0.0, 0.0), dt=0.02, steps=1500):
    """Integrate the commanded body velocity kinematically (no physics)."""
    x, y, yaw = pose
    trail = [(x, y)]
    for _ in range(steps):
        if goal is not None:
            vx, vy, wz = planner_or_exec.update(x, y, yaw, dt)
            done = planner_or_exec.status in ("reached", "blocked")
        else:
            vx, vy, wz = planner_or_exec.update(x, y, yaw)
            done = not planner_or_exec.active
        c, s = math.cos(yaw), math.sin(yaw)
        x += (c * vx - s * vy) * dt
        y += (s * vx + c * vy) * dt
        yaw += wz * dt
        trail.append((x, y))
        if done:
            break
    return (x, y, yaw), trail


def test_planner_reaches_a_goal_behind_an_obstacle_without_touching_it():
    omap = build_map([(1.0, 1.5, -0.5, 0.5)])
    planner = ScanPlanner(omap)
    planner.set_goal(2.5, 0.0)
    pose, trail = drive(planner, goal=True)
    assert planner.status == "reached", planner.debug.status
    assert math.hypot(pose[0] - 2.5, pose[1]) < 0.2
    pts = np.asarray(trail)
    yaws = np.arctan2(np.gradient(pts[:, 1]), np.gradient(pts[:, 0]))
    assert not Footprint().collides(omap, pts[:, 0], pts[:, 1], yaws).any()
    assert np.abs(pts[:, 1]).max() > 0.4, "expected a lateral detour"


def test_planner_reports_blocked_when_walled_in():
    omap = build_map([
        (0.6, 0.9, -4.0, 4.0),    # wall between the robot and the goal
        (-4.0, 4.0, 1.0, 1.3),    # and a closed box around it, so the
        (-4.0, 4.0, -1.3, -1.0),  # boundary fallback has nowhere to escape
        (-1.3, -1.0, -4.0, 4.0),
    ])
    planner = ScanPlanner(omap)
    planner.set_goal(2.5, 0.0)
    _, _ = drive(planner, goal=True)
    assert planner.status == "blocked"


def test_executor_matches_the_midlevel_contract():
    omap = build_map([(1.0, 1.5, -0.5, 0.5)])
    ex = ScanExecutor(NavConfig(vx_max=0.4, vy_max=0.25, yaw_max=0.7), omap, dt=0.02)
    assert not ex.active
    ex.submit(Turn(math.radians(45)))
    assert ex.active
    pose, _ = drive(ex)
    assert abs(math.degrees(pose[2]) - 45) < 8, pose
    assert not ex.active
    assert ex.blocked == 0


def test_executor_forward_is_planned_not_straight():
    omap = build_map([(1.0, 1.5, -0.5, 0.5)])
    ex = ScanExecutor(NavConfig(vx_max=0.4, vy_max=0.25, yaw_max=0.7), omap, dt=0.02)
    ex.submit(Forward(2.5))
    pose, trail = drive(ex)
    assert math.hypot(pose[0] - 2.5, pose[1]) < 0.3, pose
    pts = np.asarray(trail)
    assert not Footprint().collides(
        omap, pts[:, 0], pts[:, 1], np.zeros(len(pts))
    ).any()


def test_executor_truncates_into_a_wall_and_reports_blocked():
    """'forward 2' at a wall 0.5 m away: stop short, tell the navigator."""
    omap = build_map([(0.7, 1.1, -3.0, 3.0)])
    ex = ScanExecutor(NavConfig(vx_max=0.4, vy_max=0.25, yaw_max=0.7), omap, dt=0.02)
    ex.submit(Forward(2.0))
    pose, _ = drive(ex)
    assert pose[0] < 0.6, f"walked into the wall: x={pose[0]:.2f}"
    assert ex.blocked >= 1, "a truncated move must be reported as blocked"


# -- legacy policy bridge (temporary, see wojtek_rl.legacy_policy) ---------


def test_legacy_meta_detection():
    from wojtek_rl.legacy_policy import is_legacy_meta

    phase = {"obs_layout": ["joint_pos:12", "command:4", "phase:8"]}
    assert is_legacy_meta(phase)
    assert not is_legacy_meta({**phase, "schema_version": 2})
    assert not is_legacy_meta({"obs_layout": ["joint_pos:12", "command:4"]})


def _legacy_policy_dir():
    from wojtek_rl import paths

    d = paths.PROJECT_DIR / "runs/legacy_policy"
    return d if (d / "policy.npz").exists() else None


@pytest.mark.skipif(_legacy_policy_dir() is None, reason="no local legacy policy")
def test_legacy_policy_gait_clock_runs():
    """The bridge must reproduce the phase clock: a walk command advances it,
    a stand command freezes it, and the targets stay inside ctrlrange."""
    from wojtek_rl.np_policy import load_policy_runtime

    pol = load_policy_runtime(str(_legacy_policy_dir()))
    qpos, qvel = pol.home_ctrl.copy(), np.zeros(12, np.float32)

    pol.reset()
    for _ in range(10):
        pol.step(None, None, qpos, qvel, np.array((0.0, 0.0, 0.0), np.float32))
    assert pol.phase == 0.0, "a standing command must freeze the gait clock"

    pol.reset()
    seen = []
    for _ in range(10):
        t = pol.step(None, None, qpos, qvel, np.array((0.4, 0.0, 0.0), np.float32))
        seen.append(t)
        assert np.all(t >= pol.ctrl_low - 1e-6) and np.all(t <= pol.ctrl_high + 1e-6)
    assert pol.phase != 0.0, "a walk command must advance the gait clock"
    assert np.ptp(np.array(seen), axis=0).max() > 0.01, "targets are static"


# -- integration (needs the scanned scene + a GL backend) ------------------


def _scene_available(name="room") -> bool:
    from wojtek_rl import paths

    return (paths.scene_dir(name) / "occupancy.npz").exists() and paths.scene_xml(
        name
    ).exists()


@pytest.mark.skipif(not _scene_available(), reason="room assets not built")
def test_kinsim_planner_beats_the_straight_march():
    """The headline claim, on the real scanned room: same oracle-VLM guidance,
    straight march collides, planner arrives."""
    import os

    os.environ.setdefault("MUJOCO_GL", "cgl")
    from wojtek_eval.gridmap import GridMap
    from wojtek_rl import paths
    from wojtek_rl.scan_bench import generate_episodes, run_episode

    grid = GridMap.load(paths.scene_dir("room") / "occupancy.npz")
    episodes = generate_episodes(grid, 2, seed=1)
    if not episodes:
        pytest.skip("no blocked-line episodes in this scene")
    for ep in episodes:
        straight, _ = run_episode("room", ep, planner=False)
        scan, _ = run_episode("room", ep, planner=True)
        assert straight.collisions > 0, "episode was supposed to be blocked"
        assert scan.collisions == 0, f"planner collided on ep{ep.idx}"
        assert scan.final_dist_m < straight.final_dist_m
