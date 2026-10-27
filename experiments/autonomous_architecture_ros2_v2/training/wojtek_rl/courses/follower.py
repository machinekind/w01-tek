"""The frozen pure-pursuit follower that drives every course.

These are the benchmark's own constants, deliberately NOT navigation.py's
NavConfig: tuning the demo's go-to-point gains must never move benchmark
numbers. Do not "improve" them -- a change here silently invalidates every
score ever recorded. The follower is non-holonomic on purpose (vy is always
0): letting it strafe would let the robot crab through the turning courses
without ever testing turning.
"""

import math
from dataclasses import dataclass, field

import jax.numpy as jp
import numpy as np

from wojtek_rl.courses.spec import HEIGHT_CMD, Course
from wojtek_rl.navigation import _wrap

LOOKAHEAD_M = 0.40  # ~1.6 stance half-widths ahead on the path
# Inside the joystick policies' trained command box (stiff_b: wz in [-1, 1]).
# NavConfig's 1.2 is NOT usable here: commanding wz beyond the box froze the
# stiff_b keeper solid on the slalom (measured: wz=-1.2 held for 60 s moved
# the yaw by less than one degree).
YAW_MAX = 1.0  # rad/s
# Spin-in-place branch with hysteresis: enter above SPIN_ENTER_RAD, leave
# only below SPIN_EXIT_RAD. A single threshold chatters when alpha sits on
# it -- the command flips between (vx, wz) and (0, wz_max) every step, the
# policy's action filter averages that to a stand, and the rollout deadlocks
# (the exact stall every slalom seed hit at s = 2.08 m).
SPIN_ENTER_RAD = 1.05  # 60 deg
SPIN_EXIT_RAD = 0.35  # 20 deg
K_YAW_SPIN = 1.5  # proportional yaw gain in the spin-in-place branch
GOAL_RADIUS_M = 0.25  # course counts as finished inside this of the last point
RESAMPLE_DS_M = 0.02  # polyline densification spacing
PROGRESS_WINDOW_M = 1.0  # closest-point search looks only this far ahead
# `reached` needs BOTH the goal radius AND arclength progress past this much
# of the course: a closed course's final waypoint IS its start point, so a
# distance check alone "completes" a circle 1.9 s in, on the lead-in.
GOAL_MIN_PROGRESS_M = 2 * LOOKAHEAD_M


@dataclass
class Pursuit:
    """Frozen pure-pursuit follower over one course, in world coordinates.

    Constructed from the pose measured after the standing settle, so the
    course is laid out ahead of wherever the robot actually ended up.
    Progress is monotone: the closest-point search only ever looks forward
    from the last match (PROGRESS_WINDOW_M), which is what keeps
    figure_eight_r15 from snapping onto the crossing it already passed.
    """

    pts: np.ndarray  # (N, 2) world-frame, ~RESAMPLE_DS_M apart
    tangents: np.ndarray  # (N, 2) unit
    speed: np.ndarray  # (N,) commanded speed at each point
    cum_s: np.ndarray  # (N,) arclength from the start
    i: int = field(default=0)
    spinning: bool = field(default=False)  # hysteresis state, see SPIN_*_RAD

    @classmethod
    def from_course(cls, course: Course, x0: float, y0: float, yaw0: float):
        wp = np.asarray(course.waypoints, dtype=float)
        seg_speed = course.segment_speeds
        pts, spd = [], []
        for k in range(len(wp) - 1):
            a, b = wp[k], wp[k + 1]
            seg_len = float(np.hypot(*(b - a)))
            n = max(1, int(round(seg_len / RESAMPLE_DS_M)))
            t = np.arange(n) / n  # excludes the endpoint; next segment owns it
            pts.append(a + np.outer(t, b - a))
            spd.append(np.full(n, seg_speed[k]))
        pts.append(wp[-1:])
        spd.append(seg_speed[-1:])
        local = np.concatenate(pts, axis=0)
        speed = np.concatenate(spd, axis=0)

        # start frame -> world
        c, s = math.cos(yaw0), math.sin(yaw0)
        world = np.stack(
            [x0 + c * local[:, 0] - s * local[:, 1],
             y0 + s * local[:, 0] + c * local[:, 1]], -1
        )
        d = np.diff(world, axis=0, append=world[-1:] + (world[-1:] - world[-2:-1]))
        norm = np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        seg = np.linalg.norm(np.diff(world, axis=0), axis=1)
        return cls(
            pts=world,
            tangents=d / norm,
            speed=speed,
            cum_s=np.concatenate([[0.0], np.cumsum(seg)]),
        )

    @property
    def total_length(self) -> float:
        return float(self.cum_s[-1])

    def command(self, x: float, y: float, yaw: float):
        """One follower step.

        Returns (cmd, xte, s, reached) where `cmd` is the 4-vector
        [vx, vy, wz, height] the env consumes, `xte` is the signed
        cross-track error (m, + = left of the path), `s` is arclength
        progress (m) and `reached` says the course is done: within
        GOAL_RADIUS_M of the final point AND nearly through the course's
        arclength (see GOAL_MIN_PROGRESS_M).
        """
        p = np.array([x, y])
        n = len(self.pts)
        window = max(1, int(round(PROGRESS_WINDOW_M / RESAMPLE_DS_M)))
        hi = min(n, self.i + window + 1)  # +1: the window is inclusive of the
        # point exactly PROGRESS_WINDOW_M ahead
        seg = self.pts[self.i : hi]
        j = int(np.argmin(np.square(seg - p).sum(axis=1)))
        self.i += j
        i0 = self.i

        # Signed cross-track: + when the robot is left of the path tangent.
        off = p - self.pts[i0]
        tx, ty = self.tangents[i0]
        xte = float(tx * off[1] - ty * off[0])

        look = min(n - 1, i0 + int(round(LOOKAHEAD_M / RESAMPLE_DS_M)))
        to_look = self.pts[look] - p
        alpha = _wrap(math.atan2(to_look[1], to_look[0]) - yaw)

        v_target = float(self.speed[i0])
        # Spin-in-place with hysteresis (the branch square_3m's corners and
        # u_turn's reversal exercise): once entered, keep spinning until the
        # heading error is small, so the command cannot chatter across a
        # single threshold and average out to a stand.
        if self.spinning:
            if abs(alpha) < SPIN_EXIT_RAD:
                self.spinning = False
        elif abs(alpha) > SPIN_ENTER_RAD:
            self.spinning = True
        if self.spinning:
            vx = 0.0
            wz = float(np.clip(K_YAW_SPIN * alpha, -YAW_MAX, YAW_MAX))
        else:
            vx = v_target * math.cos(alpha)
            # Textbook pure pursuit: curvature 2 sin(alpha) / L, times speed.
            wz = float(
                np.clip(2.0 * vx * math.sin(alpha) / LOOKAHEAD_M, -YAW_MAX, YAW_MAX)
            )
        # Goal radius alone is not completion: a closed course ends where it
        # started, so also require the monotone progress index to be nearly
        # through the course.
        near_end = float(self.cum_s[i0]) >= self.total_length - GOAL_MIN_PROGRESS_M
        reached = near_end and bool(np.hypot(*(self.pts[-1] - p)) < GOAL_RADIUS_M)
        return (
            jp.array([vx, 0.0, wz, HEIGHT_CMD]),
            xte,
            float(self.cum_s[i0]),
            reached,
        )
