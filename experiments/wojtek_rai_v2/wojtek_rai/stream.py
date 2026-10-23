"""Drive one agent turn and surface tokens, tool calls, tool results and the
final message list -- the one piece both front ends (terminal, streamlit) use.

Why this exists: RAI's ReAct graph returns ``None`` from its ``llm`` node and
no ``values`` snapshot after it, so the final assistant message never shows
up in ``updates``/``values``. The only place it exists is the ``messages``
token stream. This helper aggregates those chunks into a proper AIMessage
and rebuilds the conversation from the last ``tools`` update plus that
message, so the next turn keeps its context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ToolMessage,
)


def text_of(content: Any) -> str:
    """Plain text of a message content (str or list of content blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return str(content)


def _chunk_to_message(acc: AIMessageChunk) -> AIMessage:
    return AIMessage(
        content=acc.content,
        tool_calls=list(acc.tool_calls or []),
        additional_kwargs=dict(acc.additional_kwargs or {}),
        response_metadata=dict(acc.response_metadata or {}),
    )


@dataclass
class TurnEvents:
    """Callbacks for a front end. Every one is optional."""

    on_token: Optional[Callable[[str, str], None]] = None      # (piece, text_so_far)
    on_tool_calls: Optional[Callable[[List[Dict[str, Any]]], None]] = None
    on_tool_result: Optional[Callable[[ToolMessage], None]] = None
    on_step_end: Optional[Callable[[], None]] = None            # after a tool result
    partial_calls: Optional[Callable[[Dict[int, Dict[str, str]]], None]] = None


@dataclass
class TurnResult:
    messages: List[BaseMessage]
    final_text: str = ""
    assistant_texts: List[str] = field(default_factory=list)


def run_turn(graph, history: List[BaseMessage], events: TurnEvents | None = None) -> TurnResult:
    ev = events or TurnEvents()
    messages: List[BaseMessage] = list(history)
    acc: Optional[AIMessageChunk] = None
    text = ""
    partial: Dict[int, Dict[str, str]] = {}
    texts: List[str] = []

    def flush_ai() -> None:
        nonlocal acc, text, partial
        if acc is None:
            return
        msg = _chunk_to_message(acc)
        if msg.tool_calls and ev.on_tool_calls:
            ev.on_tool_calls(msg.tool_calls)
        if text_of(msg.content).strip():
            texts.append(text_of(msg.content).strip())
        acc, text, partial = None, "", {}

    for mode, payload in graph.stream({"messages": history}, stream_mode=["messages", "updates"]):
        if mode == "updates":
            for _node, update in (payload or {}).items():
                if isinstance(update, dict) and update.get("messages"):
                    # The tools node returns the whole conversation so far.
                    messages = list(update["messages"])
            continue
        chunk, _meta = payload
        if isinstance(chunk, AIMessageChunk):
            acc = chunk if acc is None else acc + chunk
            piece = text_of(chunk.content)
            if piece:
                text += piece
                if ev.on_token:
                    ev.on_token(piece, text)
            for tc in chunk.tool_call_chunks or []:
                slot = partial.setdefault(tc.get("index") or 0, {"name": "", "args": ""})
                slot["name"] += tc.get("name") or ""
                slot["args"] += tc.get("args") or ""
            if chunk.tool_call_chunks and ev.partial_calls:
                ev.partial_calls(partial)
        elif isinstance(chunk, ToolMessage):
            flush_ai()
            if ev.on_tool_result:
                ev.on_tool_result(chunk)
            if ev.on_step_end:
                ev.on_step_end()

    # The final assistant message lives only in the token stream.
    if acc is not None:
        final = _chunk_to_message(acc)
        flush_ai()
        messages.append(final)
    # RAI injects its system prompt into the state; it must not be fed back
    # as history (the graph adds it again on the next turn).
    messages = [m for m in messages if getattr(m, "type", "") != "system"]
    return TurnResult(messages=messages, final_text=texts[-1] if texts else "", assistant_texts=texts)
