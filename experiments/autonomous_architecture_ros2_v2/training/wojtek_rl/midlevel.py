"""Mid-level discrete commands on top of the velocity-tracking policy.

The VLM-ready seam: a text command ("turn_left 30", "forward 1.5", "stop")
parses into Turn/Forward/Stop, queues in MidLevelExecutor, and update(x, y,
yaw) called at 50 Hz emits the body-frame [vx, vy, wyaw] the policy tracks.
Today the text comes from a box in the web UI; later a VLM emits the same
commands from the ego camera view.

Pure math on top of wojtek_rl.navigation -- no mujoco import, trivially
unit-testable (see tests/test_midlevel.py).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, replace

from wojtek_rl.navigation import NavConfig, _wrap, command_to_target


@dataclass(frozen=True)
class Turn:
    angle_rad: float  # +left (CCW), relative to the yaw when the command starts


@dataclass(frozen=True)
class Forward:
    meters: float  # along the heading when the command starts


@dataclass(frozen=True)
class Backward:
    """Straight retreat, no yaw correction: the escape move when the robot is
    wedged against an obstacle (the policy is trained down to vx = -0.6)."""

    meters: float


@dataclass(frozen=True)
class Stop:
    pass


Move = Turn | Forward | Backward

_USAGE = "expected: turn_left <deg> | turn_right <deg> | forward <m> | backward <m> | stop"


def parse_command(text: str) -> Move | Stop:
    """Parse a mid-level command string; raises ValueError with usage."""
    parts = text.strip().lower().split()
    if not parts:
        raise ValueError(_USAGE)
    name, args = parts[0], parts[1:]
    if name == "stop":
        if args:
            raise ValueError(_USAGE)
        return Stop()
    if name not in ("turn_left", "turn_right", "forward", "backward") or len(args) != 1:
        raise ValueError(_USAGE)
    try:
        value = float(args[0])
    except ValueError:
        raise ValueError(_USAGE) from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"amount must be positive; {_USAGE}")
    if name == "turn_left":
        return Turn(math.radians(value))
    if name == "turn_right":
        return Turn(-math.radians(value))
    if name == "backward":
        return Backward(value)
    return Forward(value)


def _describe(cmd: Move) -> str:
    if isinstance(cmd, Turn):
        side = "turn_left" if cmd.angle_rad >= 0 else "turn_right"
        return f"{side} {abs(math.degrees(cmd.angle_rad)):.0f}deg"
    if isinstance(cmd, Backward):
        return f"backward {cmd.meters:.2f}m"
    return f"forward {cmd.meters:.2f}m"


class MidLevelExecutor:
    """FIFO queue of Turn/Forward/Backward commands executed as velocities.

    Each command latches the pose at activation: Turn targets an absolute
    yaw, Forward targets the world point ``start + R(yaw0) @ [m, 0]`` and
    delegates steering to navigation.command_to_target (which also corrects
    lateral drift); Backward retreats toward the mirrored point with no yaw
    correction. Stop clears everything.

    A command that stops making progress (robot pushing against furniture)
    is aborted after STALL_TICKS and counted in ``blocked`` -- without this
    the min-speed floors below would grind the robot into the obstacle until
    an external timeout.
    """

    # The policy freezes its gait clock below a command speed of ~0.05
    # (|v_xy| + 0.3|wz|, see fbb_policy STAND_SPEED), so a pure P-controller
    # deadlocks near the goal: tiny command -> robot stands -> error stays.
    # Keep the emitted command above these floors until the goal is reached.
    MIN_YAW = 0.25   # rad/s  (cmd speed 0.3*0.25 = 0.075)
    MIN_SPEED = 0.12  # m/s

    # Blocked detection: at the floors above, an unobstructed robot gains
    # >= 0.0024 m (or 0.005 rad) per tick, so 1.5 s with < 1 cm net progress
    # means it is pushing against something.
    STALL_TICKS = 75   # 1.5 s at 50 Hz
    STALL_EPS = 0.01   # m or rad of progress that re-arms the stall clock

    def __init__(
        self,
        cfg: NavConfig,
        yaw_tol: float = 0.06,
        pos_tol: float = 0.06,
        queue_max: int = 8,
    ):
        self.cfg = cfg
        self._fwd_cfg = replace(cfg, stop_radius=pos_tol)
        self.yaw_tol = yaw_tol
        self.pos_tol = pos_tol
        self.queue_max = queue_max
        self._queue: deque[Move] = deque()
        self._current: Move | None = None
        self._goal: float | tuple[float, float] | None = None
        self._remaining = 0.0
        self._best_remaining = math.inf
        self._stall = 0
        self.blocked = 0  # monotonic count of stall-aborted commands

    def submit(self, cmd: Move | Stop) -> None:
        if isinstance(cmd, Stop):
            self.clear()
            return
        if len(self._queue) >= self.queue_max:
            raise ValueError(f"command queue full (max {self.queue_max})")
        self._queue.append(cmd)

    def clear(self) -> None:
        self._queue.clear()
        self._current = None
        self._goal = None
        self._remaining = 0.0

    @property
    def active(self) -> bool:
        return self._current is not None or bool(self._queue)

    def status(self) -> dict:
        return {
            "active": _describe(self._current) if self._current else None,
            "remaining": round(self._remaining, 3),
            "queued": len(self._queue),
        }

    def _latch(self, x: float, y: float, yaw: float) -> None:
        cmd = self._current
        self._best_remaining = math.inf
        self._stall = 0
        if isinstance(cmd, Turn):
            self._goal = _wrap(yaw + cmd.angle_rad)
        else:
            sign = -1.0 if isinstance(cmd, Backward) else 1.0
            self._goal = (
                x + sign * math.cos(yaw) * cmd.meters,
                y + sign * math.sin(yaw) * cmd.meters,
            )

    def _stalled(self) -> bool:
        """True once the current command has gone STALL_TICKS without progress."""
        if self._remaining < self._best_remaining - self.STALL_EPS:
            self._best_remaining = self._remaining
            self._stall = 0
            return False
        self._stall += 1
        return self._stall >= self.STALL_TICKS

    def update(self, x: float, y: float, yaw: float) -> tuple[float, float, float]:
        """One 50 Hz tick: (vx, vy, wyaw) toward the current command's goal."""
        while True:
            if self._current is None:
                if not self._queue:
                    self._remaining = 0.0
                    return 0.0, 0.0, 0.0
                self._current = self._queue.popleft()
                self._latch(x, y, yaw)

            if isinstance(self._current, Turn):
                err = _wrap(self._goal - yaw)
                self._remaining = abs(err)
                if abs(err) < self.yaw_tol:
                    self._current = None
                    continue
                if self._stalled():
                    break
                mag = min(self.cfg.yaw_max, max(self.MIN_YAW, self.cfg.k_yaw * abs(err)))
                return 0.0, 0.0, math.copysign(mag, err)

            if isinstance(self._current, Backward):
                gx, gy = self._goal
                dist = math.hypot(gx - x, gy - y)
                self._remaining = dist
                if dist < self.pos_tol:
                    self._current = None
                    continue
                if self._stalled():
                    break
                speed = min(self.cfg.vx_max, max(self.MIN_SPEED, self.cfg.k_v * dist))
                return -speed, 0.0, 0.0

            gx, gy = self._goal
            vx, vy, wyaw, dist, reached = command_to_target(
                x, y, yaw, gx, gy, self._fwd_cfg
            )
            self._remaining = dist
            if reached:
                self._current = None
                continue
            if self._stalled():
                break
            speed = math.hypot(vx, vy)
            if 0.0 < speed < self.MIN_SPEED:
                vx, vy = vx * self.MIN_SPEED / speed, vy * self.MIN_SPEED / speed
            return vx, vy, wyaw

        # Stall-abort: drop the blocked command AND the queue behind it (later
        # commands were planned assuming this one completed), stand still.
        self.blocked += 1
        self.clear()
        return 0.0, 0.0, 0.0
