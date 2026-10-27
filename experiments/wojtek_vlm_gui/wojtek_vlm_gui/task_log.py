"""The brain's status stream as the page shows it. Pure: no ROS, no UI.

vlm_brain_node publishes one JSON object per step on wojtek/vlm/status
(`action`, `step`, `instruction`, `t`, and per action `answer`,
`latency_s`, `result`, `deg`, `turned_total`, `metres`,
`distance_to_target_m`, `error`). `TaskLog` keeps the lines of the task
in flight, `format_line` turns one into the text the page prints (the same
fields the web console's brain panel shows), `verdict` colours the end.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from wojtek_vlm_gui.limits import STOP_WORDS

FINISHED = "finished"
IDLE = "idle"
GOOD_RESULTS = ("done",)


def is_stop_word(text: str) -> bool:
    """True when the brain would read `text` as a cancel, not a task."""
    return text.strip().lower() in STOP_WORDS


def parse_status(data: str) -> Optional[Dict[str, Any]]:
    """The status JSON as a dict, or None for anything that is not one."""
    try:
        status = json.loads(data)
    except (TypeError, ValueError):
        return None
    return status if isinstance(status, dict) else None


def format_line(s: Dict[str, Any]) -> str:
    """One status as one line: `#3 ask {"type": "goal", ...} 1.2s -> reached`."""
    parts: List[str] = [f"#{s.get('step', '-')} {s.get('action', '')}".rstrip()]
    if s.get("answer") is not None:
        parts.append(json.dumps(s["answer"], ensure_ascii=False))
    if s.get("latency_s") is not None:
        parts.append(f"{s['latency_s']}s")
    if s.get("result"):
        parts.append(f"-> {s['result']}")
    if s.get("deg") is not None:
        parts.append(f"{s['deg']}deg (total {round(float(s.get('turned_total') or 0))}deg)")
    if s.get("metres") is not None:
        parts.append(f"{s['metres']} m")
    if s.get("distance_to_target_m") is not None:
        parts.append(f"target {s['distance_to_target_m']} m")
    if s.get("error"):
        parts.append(f"ERROR {s['error']}")
    return "  ".join(parts)


def verdict(s: Dict[str, Any]) -> Optional[str]:
    """`ok` for a task that finished `done`, `bad` for any other finish,
    None while it runs."""
    if s.get("action") != FINISHED:
        return None
    return "ok" if s.get("result") in GOOD_RESULTS else "bad"


class TaskLog:
    """The status lines of the task in flight, in arrival order.

    A status whose `instruction` differs from the current one starts a new
    task (the brain replaces a running task on a new instruction, and its
    first status carries the new text). `idle` after a stop keeps the old
    lines on screen and marks the log idle.
    """

    def __init__(self) -> None:
        self.instruction: Optional[str] = None
        self.lines: List[Dict[str, Any]] = []
        self.idle = True

    def push(self, status: Dict[str, Any]) -> None:
        action = status.get("action")
        if action == IDLE:
            self.idle = True
            return
        instruction = status.get("instruction")
        if instruction != self.instruction or (self.finished is not None and action != FINISHED):
            self.instruction = instruction
            self.lines = []
        self.lines.append(status)
        self.idle = action == FINISHED

    @property
    def finished(self) -> Optional[Dict[str, Any]]:
        """The finishing status of the current task, or None while it runs."""
        if self.lines and self.lines[-1].get("action") == FINISHED:
            return self.lines[-1]
        return None
