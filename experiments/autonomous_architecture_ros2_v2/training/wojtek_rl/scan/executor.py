"""Mid-level commands executed through the SCAN local planner.

Drop-in for wojtek_rl.midlevel.MidLevelExecutor: same submit / active /
blocked / status / update(x, y, yaw) surface, so VlmNavigator, the room demo
and the Tier-A kinematic sim keep working unchanged.

The difference is what a command *means*. MidLevelExecutor latches
"forward 1.5" as a world point and drives a straight line at it, which is
precisely how the robot ends up nose-first in a chair leg the VLM could not
judge the distance of. Here the same point becomes the local goal of a
route-guided planner: the VLM still decides *where*, the planner decides
*how to get there without hitting anything*, and the deformation is lateral
and small, so the executed motion stays recognisably the one the VLM asked
for.

Turns and the backward escape stay on the original executor: rotating in
place needs no route, and "backward" exists exactly for the case where the
map is wrong and the robot is already wedged.

Feedback to the VLM is preserved and sharpened. ``blocked`` still counts
commands that could not be carried out, now including the case where the
planner had to truncate the move because the requested point is inside
furniture -- the robot stops short instead of pushing, and the navigator gets
told.
"""

from __future__ import annotations

import math

from wojtek_rl.midlevel import Forward, MidLevelExecutor, Move, Stop, Turn, _describe
from wojtek_rl.navigation import NavConfig
from wojtek_rl.scan.localmap import SlidingOccupancyMap
from wojtek_rl.scan.planner import PlannerConfig, ScanPlanner

TRUNCATED_FRACTION = 0.35  # travelled less than this share of the ask -> blocked


class ScanExecutor:
    """FIFO of mid-level commands; forward moves go through ScanPlanner."""

    def __init__(
        self,
        cfg: NavConfig,
        omap: SlidingOccupancyMap,
        dt: float = 0.02,
        planner: ScanPlanner | None = None,
        queue_max: int = 8,
    ):
        self.cfg = cfg
        self.dt = dt
        self.omap = omap
        self.planner = planner or ScanPlanner(
            omap,
            PlannerConfig(v_max=cfg.vx_max, vy_max=cfg.vy_max, yaw_max=cfg.yaw_max),
        )
        # Turns and retreats reuse the proven straight-line executor.
        self._basic = MidLevelExecutor(cfg, queue_max=queue_max)
        self._queue: list[Move] = []
        self.queue_max = queue_max
        self._current: Move | None = None
        self._start: tuple[float, float] | None = None
        self._requested = 0.0
        self.blocked = 0
        self._basic_blocked0 = 0
        self._last_pose: tuple[float, float, float] | None = None

    # -- queue -------------------------------------------------------------

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
        self._start = None
        self._basic.clear()
        self.planner.clear()

    def goto(self, x: float, y: float, from_xy: tuple[float, float]) -> None:
        """Drive to a world point (the demo's click-to-walk), planned.

        Not a mid-level command -- there is no VLM in this path -- but it is
        the same planner, so a click behind the sofa routes around it.
        """
        self.clear()
        dist = math.hypot(x - from_xy[0], y - from_xy[1])
        self._current = Forward(max(dist, 1e-3))
        self._start = tuple(from_xy)
        self._requested = 0.0  # a point goal cannot be "truncated"
        self.planner.set_goal(x, y)

    @property
    def active(self) -> bool:
        return self._current is not None or bool(self._queue)

    def status(self) -> dict:
        return {
            "active": _describe(self._current) if self._current else None,
            "remaining": round(self._remaining(), 3),
            "queued": len(self._queue),
            "plan": self.planner.status,
            "plan_note": self.planner.debug.status,
        }

    def _remaining(self) -> float:
        if isinstance(self._current, Forward) and self._last_pose:
            return self.planner.distance_to_goal(self._last_pose[0], self._last_pose[1])
        return self._basic.status()["remaining"]

    # -- control tick ------------------------------------------------------

    def update(self, x: float, y: float, yaw: float) -> tuple[float, float, float]:
        self._last_pose = (x, y, yaw)
        while True:
            if self._current is None:
                if not self._queue:
                    return (0.0, 0.0, 0.0)
                self._current = self._queue.pop(0)
                self._latch(x, y, yaw)

            if isinstance(self._current, Forward):
                cmd = self.planner.update(x, y, yaw, self.dt)
                state = self.planner.status
                if state == "tracking":
                    return cmd
                if state == "blocked":
                    self._abort()
                    return (0.0, 0.0, 0.0)
                self._finish(x, y)  # reached
                continue

            # Turn / Backward: delegate, and mirror an abort into our counter.
            cmd = self._basic.update(x, y, yaw)
            if self._basic.blocked > self._basic_blocked0:
                self._basic_blocked0 = self._basic.blocked
                self._abort()
                return (0.0, 0.0, 0.0)
            if not self._basic.active:
                self._current = None
                continue
            return cmd

    def _latch(self, x: float, y: float, yaw: float) -> None:
        cmd = self._current
        self._start = (x, y)
        if isinstance(cmd, Forward):
            self._requested = cmd.meters
            self.planner.set_goal(x + math.cos(yaw) * cmd.meters, y + math.sin(yaw) * cmd.meters)
            return
        self._requested = 0.0
        if isinstance(cmd, Turn):
            cmd = self._trim_turn(x, y, yaw, cmd)
            if cmd is None:
                return
        self._basic.clear()
        self._basic.submit(cmd)

    def _trim_turn(self, x: float, y: float, yaw: float, cmd: Turn) -> Turn | None:
        """Clip a turn to the part of the sweep the body actually fits through.

        Measured on the physics tier: turns were 30 % of the time but 41 % of
        all furniture contacts, because they were the one command that never
        touched the planner -- the base does not move, so nothing checked that
        the *swept* body stays clear. A trimmed turn leaves the robot facing
        as far as it safely can; a fully blocked one is reported like any
        other blocked command so the VLM can pick a different move.
        """
        safe = self.planner.fp.sweep_free(self.omap, x, y, yaw, yaw + cmd.angle_rad)
        if abs(safe) < math.radians(3):
            self.planner.debug.status = "turn-blocked"
            self._current = None
            self._start = None
            self.blocked += 1
            return None
        if abs(safe) < abs(cmd.angle_rad) - 1e-3:
            self.planner.debug.status = (
                f"turn-trimmed {math.degrees(cmd.angle_rad):.0f}->{math.degrees(safe):.0f}deg"
            )
            return Turn(safe)
        return cmd

    def _finish(self, x: float, y: float) -> None:
        """Command done. A move that got nowhere is reported as blocked -- the
        planner refusing to ram an obstacle is information the VLM needs."""
        if self._start is not None and self._requested > 0:
            travelled = math.hypot(x - self._start[0], y - self._start[1])
            if travelled < TRUNCATED_FRACTION * self._requested:
                self.blocked += 1
        self._current = None
        self._start = None
        self.planner.clear()

    def _abort(self) -> None:
        """Blocked: drop this command and the queue behind it (later commands
        assumed this one completed), same contract as MidLevelExecutor."""
        self.blocked += 1
        self.clear()
