"""Streaming client for OpenAI-compatible servers (vLLM serves Bielik/Qwen).

Deliberately httpx + hand-parsed SSE instead of the openai package: the
node needs exactly one endpoint, cancellation between chunks, and no
surprise dependencies.  Blocking by design — the caller runs it on a worker
thread (same pattern as the ASR node).
"""

from __future__ import annotations

import json
import threading
from typing import Iterator


def iter_sse_tokens(lines: Iterator[str]) -> Iterator[str]:
    """Content tokens out of an OpenAI chat-completions SSE line stream.

    Pure function: feed it any iterable of decoded lines.  Ignores blank
    lines, comments and non-content deltas; stops on [DONE].
    """
    for line in lines:
        line = line.strip()
        if not line or not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            return
        try:
            delta = json.loads(payload)["choices"][0].get("delta", {})
        except (json.JSONDecodeError, LookupError):
            continue
        token = delta.get("content")
        if token:
            yield token


class ChatClient:
    def __init__(self, base_url: str, model: str, timeout_s: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def stream(
        self,
        messages: list[dict],
        cancel: threading.Event | None = None,
        temperature: float = 0.7,
        max_tokens: int = 400,
    ) -> Iterator[str]:
        """Yield content tokens; stops early (cleanly) when cancel is set."""
        import httpx

        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        with httpx.stream(
            "POST",
            f"{self.base_url}/v1/chat/completions",
            json=body,
            timeout=self.timeout_s,
        ) as r:
            r.raise_for_status()
            for token in iter_sse_tokens(r.iter_lines()):
                if cancel is not None and cancel.is_set():
                    return
                yield token
