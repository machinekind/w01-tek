"""OpenAI-compatible chat client for the agent (vLLM-served Qwen3-VL etc.).

Same wire protocol as wojtek_eval.vlm_openai but general-purpose: arbitrary
message lists (multi-turn, multi-image) instead of the single-frame navigate
call, so the chat loop and the search controller share one client and one
connection pool. Plain httpx, no `openai` package -- see vlm_openai's
rationale.
"""

from __future__ import annotations

import asyncio
import time

import httpx
from loguru import logger

from wojtek_eval.vlm_openai import _normalize_base_url
from wojtek_rl.vlm_nav import _safe_err

DEFAULT_AGENT_MODEL = "Qwen/Qwen3-VL-4B-Instruct-FP8"
DEFAULT_AGENT_URL = "http://127.0.0.1:8000"


def text_content(text: str) -> dict:
    return {"type": "text", "text": text}


def image_content(jpeg_b64: str) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{jpeg_b64}"}}


def user_message(text: str, images: tuple[str, ...] = ()) -> dict:
    """One user turn: images first (the VLN/eval convention), then the text."""
    content = [image_content(b64) for b64 in images]
    content.append(text_content(text))
    return {"role": "user", "content": content}


class AgentLLM:
    """Chat-completions client; one lazy httpx.AsyncClient per instance."""

    def __init__(
        self,
        base_url: str = DEFAULT_AGENT_URL,
        model: str = DEFAULT_AGENT_MODEL,
        api_key: str = "EMPTY",
        max_tokens: int = 500,
        temperature: float = 0.2,
        timeout_s: float = 60.0,
        attempts: int = 3,
        retry_backoff_s: float = 2.0,
    ):
        self.base_url = _normalize_base_url(base_url)
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_s = timeout_s
        # Retry budget is deliberately small. The eval client rides out a vLLM
        # restart with 12/24/36 s sleeps, but a person is waiting on this one:
        # a dead endpoint (box asleep, wrong tailnet, server down) must reach
        # the UI in seconds, not after a minute of silent "thinking".
        self.attempts = max(1, attempts)
        self.retry_backoff_s = retry_backoff_s
        self._client: httpx.AsyncClient | None = None
        # Metadata of the most recent successful call (latency, token usage):
        # the chat loop reads it right after chat() to build its debug trace.
        self.last_meta: dict | None = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout_s)
        return self._client

    async def chat(self, messages: list[dict], max_tokens: int | None = None) -> str:
        """POST /chat/completions and return the raw assistant text.

        Connection errors and 5xx ride out server restarts (same retry
        schedule as the eval client); anything else raises to the caller.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": self.temperature,
        }
        client = self._ensure_client()
        headers = {"Authorization": f"Bearer {self.api_key}"}
        t0 = time.monotonic()
        resp = None
        last_err = ""
        for attempt in range(self.attempts):
            if attempt:
                await asyncio.sleep(self.retry_backoff_s * attempt)
            try:
                resp = await client.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=headers
                )
            except httpx.TransportError as e:
                last_err = _safe_err(e) or type(e).__name__
                logger.warning(f"agent llm connect attempt {attempt + 1}: {last_err}")
                continue
            if resp.status_code < 500:
                break
            last_err = _safe_err(resp.text[:120])
            logger.warning(f"agent llm 5xx attempt {attempt + 1}: {last_err}")
        if resp is None:
            raise RuntimeError(
                f"agent LLM unreachable at {self.base_url} after {self.attempts} attempts"
                + (f" ({last_err})" if last_err else "")
            )
        if resp.status_code != 200:
            raise RuntimeError(f"agent LLM {resp.status_code}: {_safe_err(resp.text[:200])}")
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage") or {}
        tokens = usage.get("total_tokens")
        latency_ms = round((time.monotonic() - t0) * 1000)
        self.last_meta = {"latency_ms": latency_ms, "tokens": tokens, "model": self.model}
        suffix = f", tokens={tokens}" if tokens is not None else ""
        logger.debug(f"agent llm call: {latency_ms} ms{suffix}")
        return content

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
