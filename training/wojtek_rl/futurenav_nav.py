"""FutureNav-4B backend: a discrete-action VLN policy behind an HTTP server.

The primary VlmClientProto backend: the "model" is a FutureNav-4B action
server (wojtek_brain/futurenav_server/) on any GPU host -- a Qwen3-VL VLA
fine-tuned for VLN-CE that outputs one of MOVE_FORWARD / TURN_LEFT /
TURN_RIGHT / STOP per frame.

Differences from the other backends the navigator should not notice:

- The server is stateful per episode (frame history + VGGT KV cache), so
  decide() issues POST /reset on the first step of each goal and then streams
  only the current ego frame with POST /act.
- The action space is FutureNav's training-time discrete VLN-CE grid, mapped
  onto the mid-level commands: MOVE_FORWARD -> forward 0.25 m, TURN_LEFT /
  TURN_RIGHT -> turn 15 deg, STOP -> done. Goal text and command history are
  NOT sent -- the model only ever sees the instruction (at reset) and frames,
  exactly like its VLN-CE evaluation harness.

Uses urllib in a worker thread: no extra dependency, and the 50 Hz websocket
loop keeps running while the request is in flight.
"""

from __future__ import annotations

import asyncio
import json
import urllib.request

from loguru import logger

from wojtek_rl.vlm_nav import VlmDecision

DEFAULT_FUTURENAV_URL = "http://127.0.0.1:8100"  # override: --vlm-url / VLM_URL
# Per-call ceiling: 4B generation on the server is ~1-3 s; margin covers the
# first call after a server restart (CUDA warmup) and Tailscale hiccups.
FUTURENAV_TIMEOUT_S = 60.0
# The discrete grid moves 0.25 m / 15 deg per decision, so a room crossing
# needs far more steps than the free-form backends' default budget of 20.
FUTURENAV_MAX_STEPS = 400  # upstream eval budget (eval/run.py --max-steps)
FUTURENAV_MAX_ROTATION = 25  # upstream EARLY_STOP_ROTATION anti-spin threshold

FORWARD_STEP_M = 0.25
TURN_STEP_DEG = 15.0

ACTION_TO_DECISION = {
    "MOVE_FORWARD": ("forward", FORWARD_STEP_M),
    "TURN_LEFT": ("turn_left", TURN_STEP_DEG),
    "TURN_RIGHT": ("turn_right", TURN_STEP_DEG),
    # FutureNav's STOP means "instruction complete", i.e. the navigator's
    # `done`, not its `stop` (= abort, unreachable).
    "STOP": ("done", None),
}


def decision_from_action(action: str, raw: str) -> VlmDecision:
    """Map a FutureNav discrete action name to a navigator decision."""
    try:
        nav_action, amount = ACTION_TO_DECISION[action]
    except KeyError:
        raise ValueError(f"unknown FutureNav action {action!r} (raw: {raw!r})")
    return VlmDecision(action=nav_action, amount=amount, reasoning=f"futurenav: {raw.strip()!r}")


class FutureNavVlmClient:
    """HTTP client for the FutureNav action server; the only place it is touched."""

    def __init__(self, url: str = DEFAULT_FUTURENAV_URL):
        self.url = url.rstrip("/")
        self._episode_goal: str | None = None

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.url}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=FUTURENAV_TIMEOUT_S) as resp:
            return json.load(resp)

    async def decide(self, goal, ego_b64, history, step, max_steps, pose) -> VlmDecision:
        if step == 1 or goal != self._episode_goal:
            await asyncio.to_thread(self._post, "/reset", {"instruction": goal})
            self._episode_goal = goal
            logger.info(f"futurenav episode reset: {goal!r}")
        out = await asyncio.to_thread(self._post, "/act", {"frame_b64": ego_b64})
        return decision_from_action(str(out.get("action", "")), str(out.get("raw", "")))
