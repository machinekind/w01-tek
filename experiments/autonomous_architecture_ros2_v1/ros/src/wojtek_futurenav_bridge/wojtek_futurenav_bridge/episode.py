"""Episode state machine between the FutureNav grid and the velocity policy.

rclpy-free on purpose (tested model-free, like every node-logic module in
this experiment).  The node shell owns ROS I/O and threading; this module
owns everything decidable: mapping discrete actions onto mid-level moves,
executing them through `wojtek_rl.midlevel.MidLevelExecutor` (the same
executor the demo app drives), and the guards that end an episode --
step budget, anti-spin, consecutive-blocked, decision failure.

Timeline of one episode::

    start(instruction)
      -> needs_decision -> mark_decision_pending -> apply_action(...)
      -> tick(x, y, yaw) x N until the executor drains
      -> needs_decision -> ...          (loop)
      -> finished (outcome done/aborted/cancelled), take_end() once

`tick` returns a (vx, vy, wyaw) command whenever the episode is active --
zeros while a decision is in flight, so the downstream command stream never
goes stale with a non-zero value latched -- and None when there is nothing
to say (idle or finished; the node then stays silent, text_commander style).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from wojtek_rl.midlevel import Forward, MidLevelExecutor, Turn
from wojtek_rl.navigation import NavConfig

from .futurenav_http import FORWARD_STEP_M, TURN_STEP_DEG

# Same command profile the room demo runs the policy with.
DEFAULT_NAV = NavConfig(vx_max=0.4, vy_max=0.25, yaw_max=0.7, stop_radius=0.12)

# Upstream FutureNav eval budget and anti-spin threshold
# (wojtek_rl.futurenav_nav.FUTURENAV_MAX_STEPS / FUTURENAV_MAX_ROTATION).
DEFAULT_MAX_STEPS = 400
DEFAULT_MAX_ROTATION = 25
# Agent-layer lesson: this many stall-aborted moves in a row means the robot
# is wedged and more decisions will not free it.
DEFAULT_MAX_BLOCKED = 6


@dataclass(frozen=True)
class EpisodeEnd:
    outcome: str      # "done" | "aborted" | "cancelled"
    reason: str
    steps: int


class FutureNavEpisode:
    def __init__(
        self,
        nav_cfg: NavConfig | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_rotation: int = DEFAULT_MAX_ROTATION,
        max_blocked: int = DEFAULT_MAX_BLOCKED,
    ):
        self.nav_cfg = nav_cfg or DEFAULT_NAV
        self.max_steps = max_steps
        self.max_rotation = max_rotation
        self.max_blocked = max_blocked
        self._end: EpisodeEnd | None = None
        # Bumped on every start and finish: a decision requested under one
        # epoch is dropped if it arrives under another (an HTTP reply that
        # outlived its episode must not steer the next one).
        self.epoch = 0
        self._reset()

    def _reset(self) -> None:
        """Fresh per-episode state; the pending end event (if any) survives."""
        self.instruction: str | None = None
        self.executor = MidLevelExecutor(self.nav_cfg)
        self.steps = 0
        self._active = False
        self._decision_pending = False
        self._rotation_streak = 0
        self._blocked_streak = 0
        self._blocked_snapshot = 0
        self._exec_was_active = False

    # -- lifecycle ----------------------------------------------------------

    @property
    def active(self) -> bool:
        return self._active

    def start(self, instruction: str) -> None:
        """Begin a new episode; an in-flight one is cancelled first."""
        if self._active:
            self._finish("cancelled", "superseded by a new instruction")
        instruction = instruction.strip()
        if not instruction:
            raise ValueError("empty instruction")
        self._reset()
        self.instruction = instruction
        self._active = True
        self.epoch += 1

    def cancel(self, reason: str = "cancelled") -> None:
        if self._active:
            self._finish("cancelled", reason)

    def take_end(self) -> EpisodeEnd | None:
        """Consume the finish event (None if the episode has not ended)."""
        end, self._end = self._end, None
        return end

    def _finish(self, outcome: str, reason: str) -> None:
        self.executor.clear()
        self._active = False
        self._decision_pending = False
        self.epoch += 1
        self._end = EpisodeEnd(outcome=outcome, reason=reason, steps=self.steps)

    # -- decisions ----------------------------------------------------------

    @property
    def needs_decision(self) -> bool:
        return self._active and not self._decision_pending and not self.executor.active

    def mark_decision_pending(self) -> int:
        """Returns the epoch token the eventual reply must carry."""
        self._decision_pending = True
        return self.epoch

    def fail_decision(self, error: str, epoch: int | None = None) -> None:
        """The action server call failed; the episode cannot continue blind."""
        if epoch is not None and epoch != self.epoch:
            return  # failure of a request the episode already left behind
        self._decision_pending = False
        if self._active:
            self._finish("aborted", f"decision failed: {error}")

    def apply_action(self, action: str, epoch: int | None = None) -> None:
        """Feed one discrete FutureNav action into the executor."""
        if epoch is not None and epoch != self.epoch:
            return  # stale reply from a superseded or cancelled episode
        self._decision_pending = False
        if not self._active:
            return  # late reply after a cancel; drop it
        self.steps += 1
        if action == "STOP":
            self._finish("done", "model reported the instruction complete")
            return
        if self.steps > self.max_steps:
            self._finish("aborted", f"step budget exhausted ({self.max_steps})")
            return
        if action == "MOVE_FORWARD":
            self._rotation_streak = 0
            move = Forward(FORWARD_STEP_M)
        elif action in ("TURN_LEFT", "TURN_RIGHT"):
            self._rotation_streak += 1
            if self._rotation_streak >= self.max_rotation:
                self._finish(
                    "aborted",
                    f"spinning in place ({self._rotation_streak} consecutive turns)",
                )
                return
            sign = 1.0 if action == "TURN_LEFT" else -1.0
            move = Turn(sign * math.radians(TURN_STEP_DEG))
        else:
            self._finish("aborted", f"unknown action {action!r}")
            return
        self._blocked_snapshot = self.executor.blocked
        self._exec_was_active = True
        self.executor.submit(move)

    # -- execution ----------------------------------------------------------

    def tick(self, x: float, y: float, yaw: float) -> tuple[float, float, float] | None:
        """One command tick.  None = nothing to publish (idle or finished)."""
        if not self._active:
            return None
        vx, vy, wyaw = self.executor.update(x, y, yaw)
        if self._exec_was_active and not self.executor.active:
            # The submitted move just ended: completed, or stall-aborted.
            # executor.blocked is monotonic (never reset), so a snapshot
            # comparison distinguishes the two.
            self._exec_was_active = False
            if self.executor.blocked > self._blocked_snapshot:
                self._blocked_streak += 1
                if self._blocked_streak >= self.max_blocked:
                    self._finish(
                        "aborted",
                        f"wedged ({self._blocked_streak} blocked moves in a row)",
                    )
                    return None
            else:
                self._blocked_streak = 0
        return vx, vy, wyaw
