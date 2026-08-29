"""SCAN-Planner: replan at 10 Hz, track at the control rate.

Ties the pieces together. Every replan tick: repair the local goal against
what the map now says, run the projected A* guide, initialise a B-spline from
it, optimise with rebound pairs, and safety-check the result with the
yaw-aware footprint. Between replans the tracker samples the spline a
lookahead ahead of the current time and turns the position error into the
(vx, vy, wyaw) body-frame command the locomotion policy already speaks.

Two deviations from the paper's controller, both about this robot:

* The velocity policy is omnidirectional (it was trained on vy up to
  0.5 m/s), so the tracker emits a lateral component instead of forcing the
  body to rotate onto the tangent first. Heading still chases the tangent --
  that is what keeps the twin-cylinder footprint narrow in a doorway -- but a
  30 cm sidestep does not cost a turn-in-place.
* Speed is capped while the lookahead point sits in never-observed space.
  The paper's LiDAR sees 360 degrees; a forward-facing RealSense does not,
  and creeping into the unseen part of the map is how the robot avoids
  discovering a table leg with its face.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from wojtek_rl.scan.footprint import Footprint
from wojtek_rl.scan.guide import GuideSearch
from wojtek_rl.scan.localmap import SlidingOccupancyMap
from wojtek_rl.scan.traj import (
    BSpline,
    OptConfig,
    TrajectoryOptimizer,
    init_from_path,
    optimize_with_rebound,
)


@dataclass
class PlannerConfig:
    replan_dt: float = 0.1        # 10 Hz, as in the paper
    knot_dt: float = 0.35         # B-spline knot span
    horizon_s: float = 3.0        # trajectory horizon
    lookahead_s: float = 0.25     # tracking carrot
    yaw_lead_s: float = 0.8       # heading reference, further out than the carrot
    turn_first_rad: float = 1.05  # beyond ~60 deg of heading error, slow down
    turn_first_scale: float = 0.35
    v_max: float = 0.40           # forward cap (NAV profile of the room demo)
    vy_max: float = 0.25          # lateral cap
    yaw_max: float = 0.70
    k_pos: float = 2.0
    k_yaw: float = 1.6
    min_speed: float = 0.12       # below this the policy freezes its gait clock
    unknown_speed: float = 0.22   # creep cap while heading into unseen space
    goal_tol: float = 0.14
    stall_s: float = 1.6          # not moving at all for this long -> blocked
    stall_eps: float = 0.06       # m of net displacement that counts as moving
    no_progress_s: float = 8.0    # moving, but never nearer the goal -> blocked
    fail_ticks: int = 5           # consecutive unplannable ticks -> blocked
    goal_repair_m: float = 1.2    # how far to look for a feasible goal cell


@dataclass
class PlanDebug:
    """Last replan's internals, for the map overlay and tests."""

    guide: list = field(default_factory=list)
    traj: list = field(default_factory=list)
    goal: tuple | None = None
    repaired_goal: tuple | None = None
    status: str = "idle"
    rebounds: int = 0


class ScanPlanner:
    """Local planner over a SlidingOccupancyMap. One goal at a time."""

    def __init__(self, omap: SlidingOccupancyMap, cfg: PlannerConfig | None = None,
                 footprint: Footprint | None = None, opt: OptConfig | None = None):
        self.omap = omap
        self.cfg = cfg or PlannerConfig()
        self.fp = footprint or Footprint(radius=omap.cfg.inflate_r)
        opt = opt or OptConfig(v_max=self.cfg.v_max)
        self.optimizer = TrajectoryOptimizer(opt)
        self.goal: tuple[float, float] | None = None
        self.status = "idle"  # idle | tracking | reached | blocked
        self.debug = PlanDebug()
        self._spline: BSpline | None = None
        self._traj_t = 0.0            # time along the current spline
        self._since_replan = math.inf
        self._cmd_w = np.zeros(2)     # last commanded world-frame velocity
        self._fails = 0
        self._best_dist = math.inf
        self._stall_s = 0.0
        self._moved_s = 0.0
        self._last_xy = (math.inf, math.inf)

    # -- goal handling -----------------------------------------------------

    def set_goal(self, x: float, y: float) -> None:
        self.goal = (float(x), float(y))
        self.status = "tracking"
        self._spline = None
        self._since_replan = math.inf
        self._fails = 0
        self._best_dist = math.inf
        self._stall_s = 0.0
        self._moved_s = 0.0
        self._last_xy = (math.inf, math.inf)
        self._cmd_w = np.zeros(2)

    def clear(self) -> None:
        self.goal = None
        self.status = "idle"
        self._spline = None
        self._cmd_w = np.zeros(2)
        self.debug = PlanDebug()

    def distance_to_goal(self, x: float, y: float) -> float:
        if self.goal is None:
            return 0.0
        return math.hypot(self.goal[0] - x, self.goal[1] - y)

    # -- control tick ------------------------------------------------------

    def update(self, x: float, y: float, yaw: float, dt: float) -> tuple[float, float, float]:
        """One control tick -> body-frame (vx, vy, wyaw)."""
        if self.goal is None or self.status in ("reached", "blocked"):
            self._cmd_w = np.zeros(2)
            return (0.0, 0.0, 0.0)

        dist = self.distance_to_goal(x, y)
        if dist < self.cfg.goal_tol:
            self.status = "reached"
            self._cmd_w = np.zeros(2)
            return (0.0, 0.0, 0.0)

        if self._stalled(x, y, dist, dt):
            self.status = "blocked"
            self._cmd_w = np.zeros(2)
            return (0.0, 0.0, 0.0)

        self._since_replan += dt
        self._traj_t += dt
        if self._since_replan >= self.cfg.replan_dt or self._spline is None:
            self._replan(x, y, yaw)
        if self.status == "blocked" or self._spline is None:
            self._cmd_w = np.zeros(2)
            return (0.0, 0.0, 0.0)
        return self._track(x, y, yaw)

    def _stalled(self, x: float, y: float, goal_dist: float, dt: float) -> bool:
        """Two separate give-up conditions, because they mean different things.

        *Not moving* (the body is wedged, or the planner keeps refusing to
        emit) is the fast one. *Moving but never getting closer* is the slow
        one, and it must be slow: a detour around a wall increases the
        distance to the goal for several seconds by design, and judging that
        on the same 1.6 s clock aborts every real avoidance manoeuvre -- which
        is exactly what it did on the multi-room scene before this split.
        """
        cfg = self.cfg
        # Net displacement from an anchor, not per-tick distance: a walking
        # robot covers ~8 mm per 20 ms tick, and one pushing against a sofa
        # still wobbles a few mm per tick with the gait. Only leaving the
        # anchor's neighbourhood counts as moving.
        if math.hypot(x - self._last_xy[0], y - self._last_xy[1]) > cfg.stall_eps:
            self._last_xy = (x, y)
            self._moved_s = 0.0
        else:
            self._moved_s += dt
        if self._moved_s >= cfg.stall_s:
            self.debug.status = "stalled"
            return True

        if goal_dist < self._best_dist - cfg.stall_eps:
            self._best_dist = goal_dist
            self._stall_s = 0.0
        else:
            self._stall_s += dt
            if self._stall_s >= cfg.no_progress_s:
                self.debug.status = "no-progress"
                return True
        return False

    # -- planning ----------------------------------------------------------

    def _replan(self, x: float, y: float, yaw: float) -> None:
        cfg = self.cfg
        self._since_replan = 0.0
        self.omap.recenter(x, y)
        search = GuideSearch(self.omap, self.fp)

        gx, gy = self.goal
        bearing = math.atan2(gy - y, gx - x)
        repaired = None
        if self.fp.collides(self.omap, gx, gy, bearing):
            # The VLM aimed through furniture: keep the direction, move the
            # target to the nearest pose the body actually fits in.
            repaired = search.nearest_free(gx, gy, bearing, cfg.goal_repair_m)
            if repaired is None:
                self._note_failure("goal-unreachable")
                return
            gx, gy = repaired

        path, status = search.search((x, y), (gx, gy))
        if status == "blocked" or len(path) < 2:
            self._note_failure("no-guide")
            return
        self._fails = 0

        init = init_from_path(
            path, cfg.knot_dt, (np.array([x, y]), self._cmd_w, np.zeros(2)),
            v_nom=cfg.v_max, horizon_s=cfg.horizon_s,
        )
        spline, rebounds = optimize_with_rebound(
            init, path, self.omap, self.fp, self.optimizer
        )
        if self._first_unsafe(spline):
            # Penalty methods can hand back an infeasible optimum. The guide
            # path is collision-free by construction, so fall back to its
            # unoptimised (less smooth, still safe) parameterisation.
            if self._first_unsafe(init):
                self._note_failure("unsafe-plan")
                return
            spline, status = init, status + "+init"

        self._spline = spline
        self._traj_t = 0.0
        self.debug = PlanDebug(
            guide=[tuple(p) for p in path],
            traj=[tuple(p) for p in spline.sample(np.linspace(0, spline.duration, 24))],
            goal=self.goal,
            repaired_goal=repaired,
            status=status,
            rebounds=rebounds,
        )
        self.status = "tracking"

    def _note_failure(self, why: str) -> None:
        self._fails += 1
        self.debug.status = why
        self._spline = None
        if self._fails >= self.cfg.fail_ticks:
            self.status = "blocked"

    # -- tracking ----------------------------------------------------------

    def _track(self, x: float, y: float, yaw: float) -> tuple[float, float, float]:
        """Feedforward the trajectory's own velocity, correct the position
        error on top. (Pure position feedback to a carrot 0.35 s out caps the
        robot at k_pos * 0.35 * v -- it crawls behind its own plan.)"""
        cfg = self.cfg
        spline = self._spline
        pos = np.array([x, y])
        t = min(self._traj_t + cfg.lookahead_s, spline.duration)
        ref = spline.sample(t)[0]
        v_ff = spline.velocity(t)[0]

        v_des = v_ff + cfg.k_pos * (ref - pos)
        speed = float(np.linalg.norm(v_des))
        if speed < 1e-6:
            direction = np.array([math.cos(yaw), math.sin(yaw)])
            speed = 0.0
        else:
            direction = v_des / speed

        goal_dist = self.distance_to_goal(x, y)
        speed = min(speed, cfg.v_max, max(cfg.k_pos * goal_dist, 0.0))
        if self.omap.is_unknown(ref[0], ref[1]):
            speed = min(speed, cfg.unknown_speed)

        # Heading chases a point further down the trajectory than the tracking
        # carrot: the instantaneous tangent of a freshly optimised spline
        # jitters, and a jittering heading reference wags the body into the
        # clearance the planner just bought.
        lead = spline.sample(min(self._traj_t + cfg.yaw_lead_s, spline.duration))[0] - pos
        if float(np.linalg.norm(lead)) < 1e-3:
            lead = direction
        yaw_err = _wrap(math.atan2(lead[1], lead[0]) - yaw)
        if abs(yaw_err) > cfg.turn_first_rad:
            speed *= cfg.turn_first_scale  # face it before committing forward
        if 0.0 < speed < cfg.min_speed:
            speed = cfg.min_speed

        vx_w, vy_w = speed * direction
        c, s = math.cos(yaw), math.sin(yaw)
        vx = float(np.clip(c * vx_w + s * vy_w, -cfg.v_max, cfg.v_max))
        vy = float(np.clip(-s * vx_w + c * vy_w, -cfg.vy_max, cfg.vy_max))
        # Remember what was actually commanded (post-clamp) as the next plan's
        # initial velocity, or the spline starts from a state we never held.
        self._cmd_w = np.array([c * vx - s * vy, s * vx + c * vy])

        wyaw = float(np.clip(cfg.k_yaw * yaw_err, -cfg.yaw_max, cfg.yaw_max))
        return (vx, vy, wyaw)

    def _first_unsafe(self, spline: BSpline, check_s: float = 1.2) -> bool:
        """True if the near part of the trajectory puts the body in an obstacle.

        The optimiser is a penalty method: nothing guarantees the result is
        feasible. The paper safety-checks the optimised trajectory with the
        same yaw-aware footprint before handing it to the controller.
        """
        t = np.linspace(0.0, min(check_s, spline.duration), 12)
        pts = spline.sample(t)
        d = np.gradient(pts, axis=0)
        yaw = np.arctan2(d[:, 1], d[:, 0])
        return bool(self.fp.collides(self.omap, pts[:, 0], pts[:, 1], yaw).any())


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi
