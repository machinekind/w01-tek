"""The VLM seam, now closed: Claude drives the robot through the room.

The browser submits a goal ("go to the bed"). VlmNavigator renders the ego
camera, sends the JPEG + goal (+ a short textual history) to a Claude model,
receives ONE mid-level command via forced tool use, feeds it through the same
parse_command/MidLevelExecutor path a human uses, waits for the executor to
finish, and repeats -- NaVILA-style hierarchy: VLM at ~0.3 Hz, RL velocity
policy at 50 Hz.

The Anthropic SDK is an optional extra (`uv sync --extra vlm`) and is imported
lazily inside AnthropicVlmClient, so the server and all offline tests run
without it. ANTHROPIC_API_KEY comes from the environment.
"""

from __future__ import annotations

import asyncio
import itertools
import math
import os
import re
from dataclasses import dataclass
from typing import Protocol

from loguru import logger

DEFAULT_MODEL = "claude-haiku-4-5"  # override: --vlm-model / VLM_MODEL env

MAX_STEPS = 20
MAX_FORWARD_M = 2.0
MIN_FORWARD_M = 0.1
MAX_BACKWARD_M = 0.5   # escape move only; long blind reverses are unsafe
MAX_TURN_DEG = 180.0
MIN_TURN_DEG = 5.0
VLM_TIMEOUT_S = 45.0   # per API call
# Per executed command. Worst legal command is forward 2.0 m at the 0.12 m/s
# floor (~17 s); the margin covers brief client disconnects (sim pauses while
# no browser steps it). A timeout is NOT fatal: the robot is likely pushing
# against an obstacle, so the step is reported to the VLM and the loop goes on.
CMD_TIMEOUT_S = 30.0
POLL_S = 0.1
MAX_CONSECUTIVE_FAILURES = 2
HISTORY_MAX = 8        # entries shown to the VLM
# Evidence-based stop, for runs with no step budget (max_steps=None): a robot
# whose every command comes back blocked is wedged, and no number of further
# decisions will unwedge it. This is what a benchmark's step cap stands in for
# outside a benchmark -- a counter cannot tell progress from thrashing.
MAX_CONSECUTIVE_BLOCKED = 6

MOVE_ACTIONS = ("turn_left", "turn_right", "forward", "backward")
ACTIONS = MOVE_ACTIONS + ("stop", "done")

_SECRET_RE = re.compile(r"sk-[A-Za-z0-9_-]{8,}")


def _safe_err(e: object) -> str:
    """Error text for logs/status; redacts anything that looks like an API key."""
    return _SECRET_RE.sub("sk-[redacted]", str(e))


@dataclass(frozen=True)
class VlmDecision:
    """One navigation decision extracted from a model response."""

    action: str
    amount: float | None
    reasoning: str


NAV_TOOL = {
    "name": "navigate",
    "description": "Choose the robot's next single mid-level command.",
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": list(ACTIONS)},
            "amount": {
                "type": "number",
                "description": (
                    f"degrees for turns ({MIN_TURN_DEG:g}-{MAX_TURN_DEG:g}), "
                    f"meters for forward ({MIN_FORWARD_M:g}-{MAX_FORWARD_M:g}) "
                    f"or backward ({MIN_FORWARD_M:g}-{MAX_BACKWARD_M:g}); "
                    "omit for stop/done"
                ),
            },
            "reasoning": {"type": "string"},
        },
        "required": ["action", "reasoning"],
    },
}

# The prompt text lives in agent/prompts/nav_system.txt (editable).
from wojtek_rl.agent.prompts import load as _load_prompt

SYSTEM_PROMPT = _load_prompt(
    "nav_system",
    MIN_TURN_DEG=f"{MIN_TURN_DEG:g}", MAX_TURN_DEG=f"{MAX_TURN_DEG:g}",
    MIN_FORWARD_M=f"{MIN_FORWARD_M:g}", MAX_FORWARD_M=f"{MAX_FORWARD_M:g}",
    MAX_BACKWARD_M=f"{MAX_BACKWARD_M:g}",
)


def decision_to_command(decision: VlmDecision) -> str | None:
    """Map a decision to a mid-level command string.

    `done` is terminal (returns None); amounts are clamped into the safe
    range. The returned string still goes through parse_command() as the
    second validation gate before reaching the executor.
    """
    action = decision.action
    if action == "done":
        return None
    if action == "stop":
        return "stop"
    if action not in MOVE_ACTIONS:
        raise ValueError(f"unknown action {action!r}")
    amount = decision.amount
    if amount is None or not math.isfinite(amount):
        raise ValueError(f"{action} needs a finite amount, got {amount!r}")
    if action == "forward":
        # WOJTEK_NAV_FORWARD_SCALE stretches forward legs (user call from the
        # castle takes: FutureNav's VLN-CE grid of 0.25 m steps crosses a big
        # hall glacially, and the prompt-based backends are timid too). Scale
        # BEFORE the clamp, so the ceiling still holds; turns are untouched
        # because scaling a rotation changes where the robot looks, not just
        # how fast it gets there.
        try:
            scale = float(os.environ.get("WOJTEK_NAV_FORWARD_SCALE", "1"))
        except ValueError:
            scale = 1.0
        if scale > 0:
            amount *= scale
        amount = min(MAX_FORWARD_M, max(MIN_FORWARD_M, amount))
    elif action == "backward":
        amount = min(MAX_BACKWARD_M, max(MIN_FORWARD_M, amount))
    else:
        amount = min(MAX_TURN_DEG, max(MIN_TURN_DEG, amount))
    return f"{action} {amount:g}"


def situation_text(
    goal: str,
    history: list[dict],
    step: int,
    max_steps: int | None,
    pose: tuple[float, float, float],
) -> str:
    """The textual half of a navigation turn (goal, history, odometry).

    max_steps=None means no budget (the interactive demo): say nothing about
    a remaining count rather than inventing one -- telling a model it is on
    "step 40 of 60" pressures it to declare `done` early.
    """
    lines = [f"Goal: {goal}", f"Step {step}." if max_steps is None else f"Step {step} of {max_steps}."]
    if history:
        lines.append("Recent commands:")
        for j, h in enumerate(history[-HISTORY_MAX:], 1):
            lines.append(f"  {j}. {h['cmd']} -> {h['result']}")
    x, y, yaw = pose
    lines.append(f"Odometry: x={x:.2f} m, y={y:.2f} m, yaw={math.degrees(yaw):.0f} deg.")
    lines.append("Look at the camera image and choose the next single command.")
    return "\n".join(lines)


def build_messages(
    goal: str,
    ego_b64: str,
    history: list[dict],
    step: int,
    max_steps: int,
    pose: tuple[float, float, float],
) -> list[dict]:
    """One user message: current ego frame (image first), then the situation."""
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": ego_b64,
                    },
                },
                {"type": "text", "text": situation_text(goal, history, step, max_steps, pose)},
            ],
        }
    ]


_TEXT_CMD_RE = re.compile(
    r"\b(turn_left|turn_right|forward|backward|stop|done)\b\s*([-\d.]+)?", re.IGNORECASE
)


def parse_response(message) -> VlmDecision:
    """Extract a VlmDecision from an Anthropic message (or a stand-in).

    Prefers the forced `navigate` tool_use block; falls back to scanning
    text blocks so a model that ignores tool_choice still gets parsed.
    """
    text_parts = []
    for block in getattr(message, "content", []) or []:
        if getattr(block, "type", None) == "tool_use" and block.name == "navigate":
            inp = block.input
            amount = inp.get("amount")
            return VlmDecision(
                action=str(inp.get("action", "")),
                amount=float(amount) if amount is not None else None,
                reasoning=str(inp.get("reasoning", "")),
            )
        if getattr(block, "type", None) == "text":
            text_parts.append(block.text)
    m = _TEXT_CMD_RE.search(" ".join(text_parts))
    if m:
        amount = float(m.group(2)) if m.group(2) else None
        return VlmDecision(action=m.group(1).lower(), amount=amount, reasoning=" ".join(text_parts).strip())
    raise ValueError("no navigate tool_use or command found in model response")


class VlmClientProto(Protocol):
    async def decide(
        self,
        goal: str,
        ego_b64: str,
        history: list[dict],
        step: int,
        max_steps: int,
        pose: tuple[float, float, float],
    ) -> VlmDecision: ...


class AnthropicVlmClient:
    """Thin Claude wrapper; the only place the anthropic SDK is touched."""

    def __init__(self, model: str = DEFAULT_MODEL):
        from anthropic import AsyncAnthropic  # lazy: `vlm` extra is optional

        self._client = AsyncAnthropic()  # ANTHROPIC_API_KEY from env
        self.model = model

    async def decide(self, goal, ego_b64, history, step, max_steps, pose) -> VlmDecision:
        message = await self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=[NAV_TOOL],
            tool_choice={"type": "tool", "name": "navigate"},
            messages=build_messages(goal, ego_b64, history, step, max_steps, pose),
        )
        return parse_response(message)


class VlmNavigator:
    """Closed-loop goal runner: think (Claude) -> execute (MidLevelExecutor).

    Runs as an asyncio task on the same event loop as the websocket control
    loop; the only synchronous sim calls are ego_jpeg() (~10 ms) and
    submit_command(). It never steps the sim itself, so it simply pauses
    while no browser client is connected.
    """

    def __init__(
        self,
        sim,
        client: VlmClientProto,
        max_steps: int | None = MAX_STEPS,
        vlm_timeout_s: float = VLM_TIMEOUT_S,
        cmd_timeout_s: float = CMD_TIMEOUT_S,
        poll_s: float = POLL_S,
        overlap: bool = False,
        overlap_delay_s: float = 0.0,
        max_rotation: int | None = None,
        vlnce_frame: bool = False,
    ):
        self.sim = sim
        self.client = client
        self.max_steps = max_steps
        self.vlm_timeout_s = vlm_timeout_s
        self.cmd_timeout_s = cmd_timeout_s
        self.poll_s = poll_s
        # overlap = pipeline the next decision while the current command runs, so
        # the robot keeps moving instead of standing still during inference.
        self.overlap = overlap
        self.overlap_delay_s = overlap_delay_s
        self._pending: asyncio.Task | None = None
        # Anti-spin: stop after this many consecutive turns in place (upstream
        # eval/run.py EARLY_STOP_ROTATION). None disables the guard.
        self.max_rotation = max_rotation
        # vlnce_frame: feed the model the clean square VLN-CE frame (FutureNav)
        # instead of the HUD-composited ego view the prompt-based backends use.
        self.vlnce_frame = vlnce_frame

        self.rev = 0
        self._task: asyncio.Task | None = None
        self._state = "idle"
        self._goal: str | None = None
        self._step = 0
        self._last: dict | None = None
        self._reason: str | None = None
        self._error: str | None = None

    # -- control (called from the ws reader; same event loop) ---------------

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self, goal: str) -> dict:
        goal = goal.strip()
        if not goal:
            return {"ok": False, "error": "empty goal"}
        if self.running:
            return {"ok": False, "error": "goal already active"}
        self._goal = goal
        self._step = 0
        self._last = None
        self._reason = None
        self._error = None
        self._set_state("thinking", step=1)
        self._task = asyncio.create_task(self._run(goal))
        return {"ok": True, "goal": goal}

    def cancel(self, reason: str = "user"):
        if self._task is not None and not self._task.done():
            self._task.cancel()
            self.sim.submit_command("stop")
            logger.info(f"vlm goal cancelled ({reason})")
        # Forget the task immediately: a cancelled task is not done() until
        # the event loop runs, and set_goal starts the replacement goal in
        # the same synchronous block (same race as SearchController.cancel).
        self._task = None
        if self._state in ("thinking", "executing"):
            self._set_state("idle", reason=f"cancelled ({reason})")

    # -- status for the ws broadcast -----------------------------------------

    def status(self) -> dict:
        return {
            "state": self._state,
            "step": self._step,
            "max_steps": self.max_steps,
            "goal": self._goal,
            "last": self._last,
            "reason": self._reason,
            "error": self._error,
            # Tail of the executed-command log (cmd + reasoning + outcome),
            # enough for a live "thinking" timeline in the UI.
            "history": list(getattr(self, "history", []))[-12:],
        }

    def _set_state(self, state: str, step: int | None = None, reason: str | None = None,
                   error: str | None = None):
        self._state = state
        if step is not None:
            self._step = step
        if reason is not None:
            self._reason = reason
        if error is not None:
            self._error = error
        self.rev += 1

    # -- the loop --------------------------------------------------------------

    def _decision_to_command(self, decision: VlmDecision) -> str | None:
        """Overridable seam: wojtek_eval's navigator adds eval-only actions
        (e.g. frontier 'explore') without touching this loop."""
        return decision_to_command(decision)

    async def _run(self, goal: str):
        from wojtek_rl import perf

        try:
            with perf.span("nav.goal", goal=goal):
                await self._run_inner(goal)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # never leave a zombie "executing" status
            logger.error(f"vlm navigator crashed: {_safe_err(e)}")
            self._set_state("error", error=_safe_err(e))

    def _grab_frame(self) -> str:
        """The frame handed to the model: clean square VLN-CE frame for FutureNav,
        else the HUD-composited ego view the prompt-based backends expect."""
        from wojtek_rl import perf

        with perf.span("nav.frame", vlnce=self.vlnce_frame):
            if self.vlnce_frame:
                return self.sim.vlm_frame_jpeg()
            return self.sim.ego_jpeg()

    async def _think_ahead(self, goal: str, history: list[dict], step: int) -> VlmDecision:
        """Overlap mode: grab a frame mid-execution and start the next decision.

        The history snapshot excludes the currently executing command's outcome
        (not known yet); the frame is slightly stale (captured during the move),
        which for a 0.25 m step is a fair trade to hide the 1-3 s inference."""
        from wojtek_rl import perf

        if self.overlap_delay_s:
            await asyncio.sleep(self.overlap_delay_s)
        ego = self._grab_frame()
        # Tagged `prefetch`: this inference is deliberately hidden behind the
        # walk, so it costs wall-clock but not felt latency. Ranking it next
        # to a blocking decide() without the tag would overstate the cost.
        with perf.span("nav.decide_prefetch", step=step):
            return await self.client.decide(
                goal, ego, list(history), step, self.max_steps, self.sim.pose()
            )

    async def _run_inner(self, goal: str):
        self._pending = None
        try:
            await self._run_loop(goal)
        finally:
            pending = self._pending
            if pending is not None and not pending.done():
                pending.cancel()
            self._pending = None

    async def _run_loop(self, goal: str):
        from wojtek_rl import perf

        history: list[dict] = []
        self.history = history  # exposed for eval logging (read-only)
        failures = 0
        rotations = 0  # consecutive turns in place (anti-spin, see max_rotation)
        blocked_run = 0  # consecutive commands the executor could not carry out
        # max_steps=None: no budget. A benchmark caps steps so episodes are
        # comparable; interactively that cap just guillotines a route mid-way,
        # so the run ends on evidence instead -- the model says done/stop, the
        # user cancels, the agent spins in place, or it is wedged (below).
        steps = itertools.count(1) if self.max_steps is None else range(1, self.max_steps + 1)
        for step in steps:
            self._set_state("thinking", step=step)
            try:
                if self._pending is not None:
                    pending, self._pending = self._pending, None
                    # What the loop actually waits for a prefetched decision:
                    # zero when the overlap worked, the rest of the inference
                    # when it did not.
                    with perf.span("nav.await_prefetch", step=step):
                        decision = await asyncio.wait_for(pending, self.vlm_timeout_s)
                else:
                    ego = self._grab_frame()
                    with perf.span("nav.decide", step=step):
                        decision = await asyncio.wait_for(
                            self.client.decide(goal, ego, history, step, self.max_steps,
                                               self.sim.pose()),
                            self.vlm_timeout_s,
                        )
                cmd = self._decision_to_command(decision)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # API error, timeout, unparseable, bad amount
                failures += 1
                err = _safe_err(e)
                logger.warning(f"vlm step {step} failed: {err}")
                history.append({"cmd": "(no command)", "result": f"failed ({err})"})
                if failures > MAX_CONSECUTIVE_FAILURES:
                    self._set_state("error", error=err)
                    return
                continue
            failures = 0
            self._last = {
                "action": decision.action,
                "amount": decision.amount,
                "reasoning": decision.reasoning,
            }
            logger.info(f"vlm step {step}: {decision.action} {decision.amount} -- {decision.reasoning}")
            if cmd is None:  # done
                self._set_state("done", reason="vlm_done")
                return
            if decision.action == "stop":
                self.sim.submit_command("stop")
                self._set_state("done", reason="vlm_stop")
                return

            # Anti-spin (upstream EARLY_STOP_ROTATION): a run of turns in place
            # with no forward progress means the agent is stuck -- end the episode.
            rotations = rotations + 1 if decision.action in ("turn_left", "turn_right") else 0
            if self.max_rotation is not None and rotations > self.max_rotation:
                self.sim.submit_command("stop")
                logger.warning(f"vlm step {step}: {rotations} turns in place -- anti-spin stop")
                self._set_state("done", reason="max_rotation")
                return

            why = (decision.reasoning or "").strip().replace("\n", " ")[:70]
            cmd_h = f"{cmd} ({why})" if why else cmd
            ack = self.sim.submit_command(cmd)
            if not ack["ok"]:
                failures += 1
                history.append({"cmd": cmd_h, "result": f"rejected ({ack['error']})"})
                if failures > MAX_CONSECUTIVE_FAILURES:
                    self._set_state("error", error=ack["error"])
                    return
                continue

            self._set_state("executing", step=step)
            resets_before = self.sim.resets
            blocked_before = getattr(self.sim.executor, "blocked", 0)
            if self.overlap and (self.max_steps is None or step < self.max_steps):
                self._pending = asyncio.create_task(self._think_ahead(goal, history, step + 1))
            # Walking time, not thinking time: separated so a slow route is
            # never mistaken for a slow model (and vice versa).
            with perf.span("nav.execute", step=step, cmd=cmd):
                result = await self._wait_executor()
            abnormal = True
            if result == "timeout":
                self.sim.submit_command("stop")
                logger.warning(f"vlm step {step}: '{cmd}' timed out (blocked?)")
                history.append(
                    {"cmd": cmd_h, "result": "interrupted (timed out -- probably blocked by an obstacle)"}
                )
            elif getattr(self.sim.executor, "blocked", 0) != blocked_before:
                logger.warning(f"vlm step {step}: '{cmd}' blocked (no progress)")
                history.append(
                    {"cmd": cmd_h, "result": "blocked (no progress -- robot is pressed against an obstacle)"}
                )
            elif self.sim.resets != resets_before:
                history.append({"cmd": cmd_h, "result": "interrupted (robot fell, auto-reset)"})
            else:
                abnormal = False
                history.append({"cmd": cmd_h, "result": "completed"})
            # Wedged: nothing the executor is handed can be carried out. Ending
            # here is a finding, not a budget running out.
            blocked_run = blocked_run + 1 if abnormal else 0
            if blocked_run >= MAX_CONSECUTIVE_BLOCKED:
                self.sim.submit_command("stop")
                logger.warning(f"vlm step {step}: {blocked_run} commands in a row blocked -- giving up")
                self._set_state("done", reason="stuck")
                return
            # A prefetched decision assumed the command completed cleanly; on an
            # abnormal outcome discard it so the next step re-thinks with the
            # real result (blocked/fell) now in history.
            if abnormal and self._pending is not None:
                self._pending.cancel()
                self._pending = None
        self._set_state("done", reason="max_steps")

    async def _wait_executor(self) -> str:
        waited = 0.0
        while self.sim.executor.active:
            await asyncio.sleep(self.poll_s)
            waited += self.poll_s
            if waited >= self.cmd_timeout_s:
                return "timeout"
        return "completed"
