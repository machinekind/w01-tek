"""OpenAI-chat-completions VLM backend: any vLLM-served model, no API needed.

Drop-in replacement for AnthropicVlmClient/LocalVlmClient behind the same
VlmClientProto: the navigator loop, safety clamps and parse_command gate are
untouched. Unlike LocalVlmClient (which loads mlx-vlm in-process), this talks
to an already-running OpenAI-compatible server (e.g. `vllm serve
Qwen/Qwen3-VL-30B-A3B-Instruct`) over HTTP, so many eval episodes can hit one
GPU-backed server concurrently instead of each loading its own weights.

We speak the wire protocol with plain httpx rather than depending on the
`openai` package -- it's a handful of fields (messages, image_url data URL,
max_tokens, temperature) and one AsyncClient per instance gives us control
over connection pooling and lifetime that a fresh SDK client per call would
not.

Local vLLM-served models have no forced tool use, so the prompt asks for ONE
single-line JSON object and parse_decision() extracts it -- with the same
free-text regex fallback the Anthropic and local paths use when a model
ignores the format. Qwen3 "thinking" variants may wrap reasoning in
<think>...</think> before the answer; that is stripped before parsing.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from functools import lru_cache

import httpx
from loguru import logger

from wojtek_eval.navigator import EVAL_ACTIONS, EVAL_SYSTEM_PROMPT
from wojtek_rl.vlm_nav import VlmDecision, _safe_err, situation_text

_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _response_format(actions: tuple[str, ...]) -> str:
    """The JSON-response contract appended to the prompt; mirrors
    wojtek_rl.vlm_local.RESPONSE_FORMAT but parametrized over `actions`."""
    return (
        "Respond with ONLY one JSON object on a single line, no other text:\n"
        '{"action": ' + " | ".join(f'"{a}"' for a in actions) + ", "
        '"amount": <number or null>, "reasoning": "<one short sentence>"}'
    )


@lru_cache(maxsize=8)
def _text_cmd_re(actions: tuple[str, ...]) -> re.Pattern[str]:
    """Free-text command regex parametrized over `actions`, cached per tuple
    (there are only ever a couple of distinct action sets in a process)."""
    alts = "|".join(re.escape(a) for a in actions)
    return re.compile(rf"\b({alts})\b\s*([-\d.]+)?", re.IGNORECASE)


def _strip_think(text: str) -> str:
    """Drop a leading Qwen3 <think>...</think> block, if any.

    Uses the LAST "</think>" in the text (a model can emit more than one),
    keeping only what follows. An opening "<think>" with no closing tag
    anywhere means the answer never arrived (e.g. truncated by max_tokens) --
    that is not recoverable, so we raise rather than parse the reasoning
    itself as a decision.
    """
    idx = text.rfind("</think>")
    if idx != -1:
        return text[idx + len("</think>") :]
    if "<think>" in text:
        raise ValueError("unterminated <think> block in model output")
    return text


def parse_decision(text: str, actions: tuple[str, ...]) -> VlmDecision:
    """Extract a VlmDecision from raw chat-completion text.

    Scans for the first JSON object with a valid action (tolerates code
    fences, leading prose, trailing tokens); falls back to a free-text
    command regex. Raises ValueError if neither matches -- the navigator
    counts that as a failed step, same as an HTTP error.
    """
    text = _strip_think(text)
    for m in _JSON_RE.finditer(text):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or obj.get("action") not in actions:
            continue
        amount = obj.get("amount")
        return VlmDecision(
            action=obj["action"],
            amount=float(amount) if amount is not None else None,
            reasoning=str(obj.get("reasoning", "")),
        )
    m = _text_cmd_re(actions).search(text)
    if m:
        amount = float(m.group(2)) if m.group(2) else None
        return VlmDecision(action=m.group(1).lower(), amount=amount, reasoning=text.strip())
    raise ValueError(f"no JSON decision or command found in model output: {text[:200]!r}")


def _normalize_base_url(base_url: str) -> str:
    """vLLM's OpenAI-compatible routes live under /v1; accept the base_url
    with or without it already appended."""
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    return base_url


class OpenAIVlmClient:
    """Chat-completions client for an OpenAI-compatible server (vLLM etc.).

    One httpx.AsyncClient is created lazily on first use and kept for the
    life of this instance -- share one OpenAIVlmClient across concurrent
    eval episodes rather than building a fresh one per episode, so requests
    reuse the connection pool. Call close() when done with it.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        system_prompt: str = EVAL_SYSTEM_PROMPT,
        actions: tuple[str, ...] = EVAL_ACTIONS,
        max_tokens: int = 300,
        temperature: float = 0.0,
        timeout_s: float = 60.0,
    ):
        self.base_url = _normalize_base_url(base_url)
        self.model = model
        self.api_key = api_key
        self.system_prompt = system_prompt
        self.actions = actions
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_s = timeout_s
        self._format = _response_format(actions)
        self._client: httpx.AsyncClient | None = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout_s)
        return self._client

    async def decide(self, goal, ego_b64, history, step, max_steps, pose) -> VlmDecision:
        text = situation_text(goal, history, step, max_steps, pose) + "\n\n" + self._format
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{ego_b64}"},
                        },
                        {"type": "text", "text": text},
                    ],
                },
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        client = self._ensure_client()
        headers = {"Authorization": f"Bearer {self.api_key}"}
        t0 = time.monotonic()
        # Connection errors and 5xx ride out server restarts (a vLLM engine
        # respawn takes ~1 min); anything else fails fast to the navigator.
        resp = None
        for attempt in range(4):
            if attempt:
                await asyncio.sleep(12.0 * attempt)
            try:
                resp = await client.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=headers
                )
            except httpx.TransportError as e:
                logger.warning(f"vlm connect attempt {attempt + 1}: {_safe_err(e)}")
                continue
            if resp.status_code < 500:
                break
            logger.warning(f"vlm 5xx attempt {attempt + 1}: {_safe_err(resp.text[:120])}")
        if resp is None:
            raise RuntimeError("VLM endpoint unreachable after retries")
        latency_ms = (time.monotonic() - t0) * 1000
        if resp.status_code != 200:
            raise RuntimeError(f"VLM endpoint {resp.status_code}: {_safe_err(resp.text[:200])}")
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage") or {}
        tokens = usage.get("total_tokens")
        suffix = f", tokens={tokens}" if tokens is not None else ""
        logger.debug(f"openai vlm call: {latency_ms:.0f} ms{suffix}")
        return parse_decision(content, self.actions)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
