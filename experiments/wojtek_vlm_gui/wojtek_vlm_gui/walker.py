"""A person as the robot's legs: /cmd_vel in, a pose and a walking guide out.

Pure: no ROS; tested model-free. `walker_node` is the ROS shell.

On the cable bench nobody walks but the operator's helper, who carries the
camera. The whole nav stack runs for real (the brain, the pixel resolver,
goto, the costmap on the camera's depth), and this replaces the one thing
that is missing: the legs. It integrates the drive command into the
odom -> base_link pose *as if the robot had walked it* (the same dead
reckoning a robot without odometry would do), and turns the command into
words and an arrow for the page, so the helper walks what the robot would.

Two gains make the mock move like the gait instead of the command:
  * yaw_gain 0.76 -- the brain commands 0.5 rad/s and times a turn by the
    0.38 rad/s the gait delivers (vlm_brain_node `turn_cmd_rad_s`,
    `turn_real_rad_s`); with the gain, a 45 deg brain turn is 45 deg here.
  * speed_gain 1.0 -- goto's 0.3 m/s is a slow walk, taken as is.

A command older than `stale_s` (policy_node's 0.5 s dead-man) is a stop,
the same rule the robot applies.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

STALE_S = 0.5
STILL = 1e-3            # m/s or rad/s below which an axis counts as still
GUARD_LISTEN_S = 2.0    # how long the walker listens on /tf before it publishes


class TfGuard:
    """Never a second publisher of odom -> base_link.

    The walker broadcasts that transform; so does the robot's leg odometry
    (on in the drop-in since fork PR #14) and the simulator. Two sources of
    one transform make goto and the pixel resolver read a pose that jumps
    between them. The guard tells the walker's own broadcasts from anyone
    else's by their stamps: every stamp the walker sends is remembered, and
    an odom -> base_link seen on /tf with a stamp it never sent is foreign.
    Once foreign, always foreign for this run: the walker stops
    broadcasting and says so, it does not try to take the frame back.

    Pure: the node feeds it stamps in nanoseconds.
    """

    def __init__(self, listen_s: float = GUARD_LISTEN_S, keep: int = 400) -> None:
        self.listen_s = float(listen_s)
        self._mine: deque = deque(maxlen=keep)
        self._mine_set: set = set()
        self._started: Optional[float] = None
        self.foreign = False

    def start(self, now: float) -> None:
        self._started = float(now)

    def may_broadcast(self, now: float) -> bool:
        """False while listening (the first listen_s after start) and for
        good once another publisher was seen."""
        if self.foreign or self._started is None:
            return False
        return now - self._started >= self.listen_s

    def sent(self, stamp_ns: int) -> None:
        if len(self._mine) == self._mine.maxlen:
            self._mine_set.discard(self._mine[0])
        self._mine.append(int(stamp_ns))
        self._mine_set.add(int(stamp_ns))

    def seen(self, stamp_ns: int) -> bool:
        """An odom -> base_link on /tf; True (and latched) when it is not ours."""
        if int(stamp_ns) not in self._mine_set:
            self.foreign = True
        return self.foreign


@dataclass
class Command:
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    at: float = -1e9     # when it arrived (monotonic s)


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class Walker:
    def __init__(self, yaw_gain: float = 0.76, speed_gain: float = 1.0, stale_s: float = STALE_S) -> None:
        self.yaw_gain = float(yaw_gain)
        self.speed_gain = float(speed_gain)
        self.stale_s = float(stale_s)
        self.x = self.y = self.yaw = 0.0
        self.cmd = Command()
        self.segment_deg = 0.0        # this turn so far, signed (left +)
        self.segment_m = 0.0          # this straight walk so far
        self._segment_kind = "still"
        self.goal: Optional[Tuple[float, float]] = None   # odom x, y

    # -- inputs ------------------------------------------------------------

    def command(self, vx: float, vy: float, wz: float, now: float) -> None:
        self.cmd = Command(float(vx), float(vy), float(wz), float(now))

    def set_goal_odom(self, x: float, y: float) -> None:
        self.goal = (float(x), float(y))

    def set_goal_base(self, x: float, y: float) -> None:
        """A goal given in base_link, taken into odom at the current pose."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.goal = (self.x + c * x - s * y, self.y + s * x + c * y)

    def clear_goal(self) -> None:
        self.goal = None

    # -- the effective command ---------------------------------------------

    def active(self, now: float) -> Tuple[float, float, float]:
        """(vx, vy, wz) the legs deliver now: the command with the gait's
        gains, or zero once it went stale."""
        c = self.cmd
        if now - c.at > self.stale_s:
            return 0.0, 0.0, 0.0
        return c.vx * self.speed_gain, c.vy * self.speed_gain, c.wz * self.yaw_gain

    def kind(self, now: float) -> str:
        vx, vy, wz = self.active(now)
        if abs(vx) < STILL and abs(vy) < STILL and abs(wz) < STILL:
            return "still"
        if abs(vx) < STILL and abs(vy) < STILL:
            return "turn"
        return "walk"

    # -- integration ---------------------------------------------------------

    def step(self, now: float, dt: float) -> None:
        """Advance the pose by dt seconds of the active command (midpoint
        yaw, so an arc stays an arc), and the per-segment counters the helper
        follows ("turn left: 30 of 45 deg")."""
        vx, vy, wz = self.active(now)
        kind = self.kind(now)
        if kind != self._segment_kind:
            self.segment_deg, self.segment_m = 0.0, 0.0
            self._segment_kind = kind
        mid = self.yaw + 0.5 * wz * dt
        c, s = math.cos(mid), math.sin(mid)
        self.x += (c * vx - s * vy) * dt
        self.y += (s * vx + c * vy) * dt
        self.yaw = _wrap(self.yaw + wz * dt)
        self.segment_deg += math.degrees(wz * dt)
        self.segment_m += math.hypot(vx, vy) * dt

    # -- the guide -------------------------------------------------------------

    def goal_vector(self) -> Optional[Dict[str, float]]:
        """The goal from where the robot stands: distance and bearing (deg,
        left positive, 0 straight ahead)."""
        if self.goal is None:
            return None
        dx, dy = self.goal[0] - self.x, self.goal[1] - self.y
        bearing = _wrap(math.atan2(dy, dx) - self.yaw)
        return {"dist_m": round(math.hypot(dx, dy), 2), "bearing_deg": round(math.degrees(bearing), 1)}

    def instruction(self, now: float) -> str:
        vx, vy, wz = self.active(now)
        kind = self.kind(now)
        if kind == "still":
            return "STAND STILL"
        if kind == "turn":
            side = "LEFT" if wz > 0 else "RIGHT"
            return f"TURN {side} on the spot, {abs(math.degrees(wz)):.0f} deg/s"
        parts = []
        if abs(vx) >= STILL:
            parts.append(f"WALK {'FORWARD' if vx > 0 else 'BACKWARD'} {abs(vx):.2f} m/s")
        if abs(vy) >= STILL:
            parts.append(f"STEP {'LEFT' if vy > 0 else 'RIGHT'} {abs(vy):.2f} m/s")
        if abs(wz) >= STILL:
            parts.append(f"curving {'left' if wz > 0 else 'right'} {abs(math.degrees(wz)):.0f} deg/s")
        return ", ".join(parts)

    def guide(self, now: float) -> Dict[str, Any]:
        """Everything the page's Walk panel shows, JSON-ready. `heading_deg`
        is the direction to move relative to where the helper faces (0
        ahead, 90 left); for a turn it is +-90 so the arrow points the way
        to turn."""
        vx, vy, wz = self.active(now)
        kind = self.kind(now)
        if kind == "walk":
            heading = math.degrees(math.atan2(vy, vx))
        elif kind == "turn":
            heading = 90.0 if wz > 0 else -90.0
        else:
            heading = 0.0
        return {
            "kind": kind,
            "text": self.instruction(now),
            "heading_deg": round(heading, 1),
            "vx": round(vx, 3), "vy": round(vy, 3), "wz_deg_s": round(math.degrees(wz), 1),
            "segment_deg": round(self.segment_deg, 1),
            "segment_m": round(self.segment_m, 2),
            "pose": {"x": round(self.x, 2), "y": round(self.y, 2), "yaw_deg": round(math.degrees(self.yaw), 1)},
            "goal": self.goal_vector(),
        }
