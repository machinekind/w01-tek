"""What crosses the websocket between the server and the JS page. Pure: no
ROS, no web framework; tested model-free.

Server -> page:
  * text frames, one JSON object each, `t` names it:
      {"t": "log", "instruction", "lines": [{"text", "action", "step", "picture", "s"}], "finished": {"text", "ok"} | null}
                                            (`s`: the brain's status fields, JSON-safe, for the chat's wording;
                                             `picture`: the number of the annotated frame that belongs
                                             under this ask line, or null)
      {"t": "nav", "goto", "pixel"}
      {"t": "sys", ...sysmon.Snapshot fields..., "age_s"}
      {"t": "reply", "ok", "text"}         the answer to a task, STOP or service call
      {"t": "guide", ...walker.guide()...}  what the person carrying the camera should do
  * binary frames, a one-byte tag first:
      TAG_CAMERA    + JPEG                  the camera, as the robot encoded it
      TAG_ANNOTATED + picture number u32 LE + JPEG
                                            the picture the brain answered on; the number
                                            matches its ask line's "picture"
      TAG_DEPTH     + 7-byte header + uint16 LE pixels (mm)
                    header: 1 pad byte, width u16 LE, height u16 LE, 2 pad bytes,
                    so the pixels start at offset 8 and a JS Uint16Array can
                    view them without a copy.

Page -> server: text frames only, validated by `parse_command`:
  the handshake itself is checked first, by `origin_allowed`;
  {"t": "task", "text"} | {"t": "stop"} | {"t": "svc", "name": one of SERVICES}
  "stop" freezes the robot (the brain and goto cancelled, /cmd_vel held at
  zero) and is `is_urgent`: the server answers it before anything queued.
"""

from __future__ import annotations

import json
import struct
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

from wojtek_vlm_gui.task_log import format_line, verdict

TAG_CAMERA = 0
TAG_ANNOTATED = 1
TAG_DEPTH = 2
DEPTH_HEADER = struct.Struct("<BBHHH")   # tag, pad, width, height, pad -> 8 bytes
ANNOTATED_HEADER = struct.Struct("<BI")  # tag, picture number -> 5 bytes

# The page's buttons, by name. Each maps to a function in arm_switch; the
# server holds that mapping, the wire only knows the names.
SERVICES = ("stand_up", "lie_down", "arm", "disarm", "policy_on", "policy_off")
MAX_TASK_CHARS = 500


def pack_jpeg(tag: int, jpeg: bytes) -> bytes:
    return bytes([tag]) + jpeg


def pack_annotated(picture: int, jpeg: bytes) -> bytes:
    """The brain's picture with the number of the ask line it belongs under."""
    return ANNOTATED_HEADER.pack(TAG_ANNOTATED, picture) + jpeg


def unpack_annotated(frame: bytes):
    """(picture number, JPEG); the inverse of pack_annotated, for tests."""
    tag, picture = ANNOTATED_HEADER.unpack_from(frame)
    assert tag == TAG_ANNOTATED
    return picture, frame[ANNOTATED_HEADER.size:]


def pack_depth(width: int, height: int, pixels: bytes) -> bytes:
    if len(pixels) != width * height * 2:
        raise ValueError(f"depth: {len(pixels)} bytes for {width}x{height}")
    return DEPTH_HEADER.pack(TAG_DEPTH, 0, width, height, 0) + pixels


def unpack_depth(frame: bytes):
    """(width, height, pixel bytes); the inverse of pack_depth, for tests."""
    tag, _, width, height, _ = DEPTH_HEADER.unpack_from(frame)
    assert tag == TAG_DEPTH
    return width, height, frame[DEPTH_HEADER.size:]


def log_message(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """BrainClient.latest() as the page's log message. Pictures stay out
    (they travel as binary frames; an ask line carries only its picture's
    number); every line is pre-formatted here so the page and the old
    console print the same words."""
    fin = snapshot.get("finished")
    finished = None
    if fin is not None:
        text = f"finished: {fin.get('result')}" + (f" -- {fin['error']}" if fin.get("error") else "")
        finished = {"text": text, "ok": verdict(fin) == "ok"}
    return {
        "t": "log",
        "instruction": snapshot.get("instruction"),
        "lines": [{"text": format_line(line), "action": line.get("action"), "step": line.get("step"),
                   "picture": line.get("_picture"),
                   "s": {k: v for k, v in line.items() if not k.startswith("_")}}
                  for line in snapshot.get("lines", [])],
        "finished": finished,
    }


def nav_message(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    return {"t": "nav", "goto": snapshot.get("goto"), "pixel": snapshot.get("pixel")}


def sys_message(data: Dict[str, Any], age_s: float) -> Dict[str, Any]:
    return {"t": "sys", **data, "age_s": round(age_s, 1)}


def guide_message(data: Dict[str, Any]) -> Dict[str, Any]:
    return {"t": "guide", **data}


def reply_message(ok: bool, text: str) -> Dict[str, Any]:
    return {"t": "reply", "ok": bool(ok), "text": str(text)}


def origin_allowed(origin: Optional[str], host: Optional[str]) -> bool:
    """May a websocket handshake with this Origin header talk to a server
    reached as `host` (the request's Host header)? The page connects back to
    the address it was loaded from, so a browser's Origin always names that
    same host and port. A different one is another web page open in the
    operator's browser trying to drive the robot, and it is refused. No
    Origin at all is not a browser (a script, a test) and passes: it can
    reach the port anyway, and a browser never leaves the header out."""
    if origin is None:
        return True
    if not host:
        return False
    try:
        parts = urlsplit(origin)
    except ValueError:
        return False
    return (parts.scheme in ("http", "https") and bool(parts.netloc)
            and parts.netloc.lower() == host.strip().lower())


def is_urgent(cmd: Dict[str, Any]) -> bool:
    """A command the server must act on the moment it arrives, ahead of
    anything still running: STOP, and nothing else."""
    return cmd.get("t") == "stop"


def parse_command(text: str) -> Optional[Dict[str, Any]]:
    """A page message as a command dict, or None for anything outside the
    protocol (the server ignores those rather than guessing)."""
    try:
        msg = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(msg, dict):
        return None
    kind = msg.get("t")
    if kind == "stop":
        return {"t": "stop"}
    if kind == "task" and isinstance(msg.get("text"), str):
        return {"t": "task", "text": msg["text"][:MAX_TASK_CHARS]}
    if kind == "svc" and msg.get("name") in SERVICES:
        return {"t": "svc", "name": msg["name"]}
    return None
