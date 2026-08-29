"""Pure translation between the #131 browser/scenario websocket protocol and
the agent topics. rclpy-free so the message shapes are unit-tested without a
ROS graph; the bridge node owns sockets and publishers and calls into here.

The client protocol is the one room_app speaks and scenario.py drives:
binary frames are 100 ms of PCM16 mono 24 kHz mic audio, text frames are
small JSON control messages, and the server streams back state dicts, reply
audio as binary, and event dicts (heard / say / chat_reply / barge_in /
trace).
"""

from __future__ import annotations

import json

SAMPLE_RATE = 24000


def parse_client_message(text: str) -> tuple[str, dict]:
    """One incoming JSON control frame -> (kind, payload). Unknown kinds are
    reported, not raised: a newer UI must not kill the bridge."""
    try:
        msg = json.loads(text)
    except ValueError:
        return ("invalid", {})
    kind = msg.get("type")
    if kind == "voice":
        return ("voice", {"on": bool(msg.get("on"))})
    if kind == "command":
        cmd = str(msg.get("text") or "").strip()
        return ("command", {"text": cmd}) if cmd else ("invalid", {})
    if kind == "reset":
        return ("reset", {})
    return ("unknown", {"type": kind})


class SayAccumulator:
    """Reassemble Bielik's per-sentence stream into browser events.

    Every sentence becomes an immediate {"type": "say"} caption; the final
    sentence of an utterance additionally closes a {"type": "chat_reply"}
    with the joined text, which is what scenario.py records as the reply.
    """

    def __init__(self):
        self._parts: dict[str, list[str]] = {}

    def feed(self, utterance_id: str, text: str, final: bool,
             source: str) -> list[dict]:
        events: list[dict] = []
        text = (text or "").strip()
        if text:
            events.append({"type": "say", "text": text, "source": source})
            self._parts.setdefault(utterance_id, []).append(text)
        if final:
            joined = " ".join(self._parts.pop(utterance_id, []))
            if joined:
                events.append({"type": "chat_reply", "ok": True, "say": joined,
                               "steps": [], "spoken": True})
        return events

    def drop(self, utterance_id: str | None = None) -> None:
        """Barge-in: whatever was accumulating is stale."""
        if utterance_id is None:
            self._parts.clear()
        else:
            self._parts.pop(utterance_id, None)


def world_command_result(kind: str, text: str, args: list[float],
                         sim) -> dict:
    """Apply one WorldCommand request to the sim. Runs on the sim's thread."""
    if kind == "midlevel":
        return sim.submit_command(text)
    if kind == "goto":
        goto = getattr(sim.executor, "goto", None)
        if goto is None:
            return {"ok": False, "error": "no planner: goto unavailable"}
        if len(args) < 2:
            return {"ok": False, "error": "goto needs [x, y]"}
        x, y, _yaw = sim.pose()
        goto(float(args[0]), float(args[1]), (x, y))
        return {"ok": True, "command": f"goto {args[0]:g} {args[1]:g}"}
    if kind == "trick":
        submit = getattr(sim, "submit_trick", None)
        if submit is None:
            return {"ok": False, "error": "this world has no trick player"}
        return submit(text)
    if kind == "reset":
        sim.reset()
        return {"ok": True, "command": "reset"}
    return {"ok": False, "error": f"unknown command kind {kind!r}"}
