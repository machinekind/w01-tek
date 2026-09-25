"""Durable session trace: what fired, when, with what result.

The demo's live debug panels answer "what is it doing right now"; this answers
"what did it do ten minutes ago". Without it the only record of a session is
a ring buffer in memory and whatever is still rendered in a browser tab --
which is exactly one reload away from being gone.

Every event is one JSON object on one line, written as it happens (line
buffered, so a crashed or killed process still leaves a readable file), and
also kept in a bounded in-memory ring that `GET /api/trace` serves.

Event shape: {"t": <s since session start>, "wall": <iso8601>, "kind": ...}
plus kind-specific fields. Kinds emitted today:

  chat.ask     user text
  chat.llm     one model call: raw output, classification, latency, tokens
  chat.tool    tool call + (truncated) result
  chat.say     final answer
  search.<st>  search FSM event, tagged with the state it happened in
  nav.state    navigator state change (goal, step, last decision)
  goal.set     a goal was set/cancelled through the goal machine

Nothing here imports mujoco or a model; a trace is plain text and the writer
is unit-tested with a tmp_path.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

RING_MAX = 2000
TEXT_MAX = 4000  # per field; raw model output is the big one


def default_trace_path(runs_dir: Path, scene: str) -> Path:
    """runs/agent_traces/<scene>_<utc timestamp>_<pid>.jsonl"""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return runs_dir / "agent_traces" / f"{scene}_{stamp}_{os.getpid()}.jsonl"


def _clip(value):
    if isinstance(value, str) and len(value) > TEXT_MAX:
        return value[:TEXT_MAX] + f"... [{len(value) - TEXT_MAX} more chars]"
    return value


class Trace:
    """Append-only JSONL writer + in-memory ring.

    A failed write must never take the robot down, so every IO error is
    logged once and swallowed; the ring keeps working either way.
    """

    def __init__(self, path: Path | None = None, ring_max: int = RING_MAX):
        self.path = Path(path) if path else None
        self.ring: deque[dict] = deque(maxlen=ring_max)
        self.t0 = time.monotonic()
        self.seq = 0
        self._fh = None
        self._write_failed = False
        # Live subscribers (the websocket loop). The trace is already the
        # single place every decision passes through, so streaming it is
        # cheaper and more complete than instrumenting each subsystem again
        # for the UI.
        self._listeners: list = []
        if self.path is not None:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._fh = self.path.open("a", buffering=1)  # line buffered
            except OSError as e:
                logger.warning(f"trace file unavailable ({e}); keeping memory ring only")
                self._fh = None

    def add(self, kind: str, **fields) -> dict:
        self.seq += 1
        event = {
            "seq": self.seq,
            "t": round(time.monotonic() - self.t0, 3),
            "wall": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "kind": kind,
            **{k: _clip(v) for k, v in fields.items()},
        }
        self.ring.append(event)
        if self._fh is not None:
            try:
                self._fh.write(json.dumps(event, default=str) + "\n")
            except (OSError, ValueError) as e:
                if not self._write_failed:  # once, not once per event
                    self._write_failed = True
                    logger.warning(f"trace write failed ({e}); memory ring only")
        for cb in list(self._listeners):
            try:
                cb(event)
            except Exception as e:
                logger.warning(f"trace listener dropped ({e})")
                self.unsubscribe(cb)
        return event

    def subscribe(self, callback) -> None:
        """Call `callback(event)` for every future event. A listener that
        raises is dropped rather than allowed to break tracing."""
        self._listeners.append(callback)

    def unsubscribe(self, callback) -> None:
        if callback in self._listeners:
            self._listeners.remove(callback)

    def recent(self, limit: int = 200, kind_prefix: str | None = None) -> list[dict]:
        events = list(self.ring)
        if kind_prefix:
            events = [e for e in events if e["kind"].startswith(kind_prefix)]
        return events[-limit:]

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None
