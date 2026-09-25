"""Tolerant parsing of the agent's reply contract.

The contract (see chat.SYSTEM_PROMPT) is one JSON object per turn, with the
free-text "thought" field FIRST -- forcing a small VLM to emit a decision as
its opening token makes it stop looking at the image (measured: action-first
collapsed Qwen3-VL-4B to `forward` on every frame; reasoning-first fixed it).
test_agent_parsing.py guards that key order.

Small models also butcher JSON in practice (gpt-realtime demo notes): wrapped
in parens with a stray trailing brace, code fences, prose before/after,
several objects in one reply. So extraction is a string-aware balanced-brace
scanner, not find-first-{/rfind-} and not a lazy regex.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class AgentReply:
    """One parsed agent turn: either a tool call or a final answer.

    `fallback` marks a reply that carried no JSON at all and was accepted as
    plain speech.  Callers must treat those as second-class: measured live
    (2026-08-13), letting one plain-text narration into the rolling history
    teaches the model to drop the contract on every later action turn --
    navigate stopped firing for the rest of the session.
    """

    thought: str
    tool: str | None = None
    args: dict = field(default_factory=dict)
    say: str | None = None
    fallback: bool = False


def strip_think(text: str) -> str:
    """Drop a leading Qwen3 <think>...</think> block (keep after the LAST
    closing tag; an unterminated block means the answer was truncated)."""
    idx = text.rfind("</think>")
    if idx != -1:
        return text[idx + len("</think>") :]
    if "<think>" in text:
        raise ValueError("unterminated <think> block in model output")
    return text


def iter_json_objects(text: str):
    """Yield every balanced top-level {...} substring that parses as a dict.

    String-aware: braces inside JSON strings (and escaped quotes) don't count,
    so a "say" value containing '{' cannot derail the scan.
    """
    depth = 0
    start = -1
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            if depth > 0:
                in_str = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    yield obj


def parse_agent_reply(text: str, tool_names: tuple[str, ...] = ()) -> AgentReply:
    """Extract an AgentReply from raw model output.

    Picks the first JSON object carrying a "tool" or "say" key. A reply with
    no usable JSON is treated as a plain spoken answer (free text beats JSON
    for small VLMs, so don't punish the model for dropping the wrapper) --
    EXCEPT when it is obviously a mangled JSON attempt, which must never be
    read out loud.

    `tool_names` enables the common small-model mangle where the tool name
    becomes the key: {"thought": ..., "navigate": {"instruction": ...}}
    instead of {"tool": "navigate", "args": {...}}. Observed live on
    Qwen3-VL-4B; without this the whole JSON blob was spoken to the user.
    """
    text = strip_think(text).strip()
    for obj in iter_json_objects(text):
        thought = str(obj.get("thought", "") or "")
        tool = obj.get("tool")
        say = obj.get("say")
        if tool:
            args = obj.get("args")
            return AgentReply(
                thought=thought,
                tool=str(tool).strip().lower(),
                args=dict(args) if isinstance(args, dict) else {},
            )
        # tool-name-as-key form
        for name in tool_names:
            if name in obj:
                value = obj[name]
                args = value if isinstance(value, dict) else {}
                if not isinstance(value, dict) and value not in (None, "", True):
                    args = {"instruction": str(value)}
                return AgentReply(thought=thought, tool=name, args=args)
        if say is not None:
            return AgentReply(thought=thought, say=str(say))
    if not text:
        raise ValueError("empty model reply")
    # A blob that is clearly a botched JSON attempt is a parse failure, not an
    # answer: speaking it would read braces and field names to the user.
    if text.lstrip().startswith("{") or '"thought"' in text:
        raise ValueError(f"malformed JSON reply with no usable field: {text[:160]!r}")
    return AgentReply(thought="", say=text, fallback=True)
