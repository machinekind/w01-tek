"""The VLM client: Qwen3-VL 30B-A3B on Ollama, over the OpenAI chat API.

The navigator's only model backend, behind VlmClientProto: the navigator
loop, safety clamps and parse_command gate never see the transport. Ollama
serves `qwen3-vl:30b-a3b-instruct` with an OpenAI-compatible API under /v1
(default http://127.0.0.1:11434/v1; VLM_URL / VLM_MODEL or the CLI flags
point it elsewhere). One server can take many concurrent eval episodes.

We speak the wire protocol with plain httpx rather than depending on the
`openai` package -- it's a handful of fields (messages, image_url data URL,
max_tokens, temperature) and one AsyncClient per instance gives us control
over connection pooling and lifetime that a fresh SDK client per call would
not.

The prompt asks for ONE single-line JSON object and parse_decision()
extracts it, with a free-text regex fallback for a model that ignores the
format. Qwen3 "thinking" variants may wrap reasoning in <think>...</think>
before the answer; that is stripped before parsing.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from functools import lru_cache

from loguru import logger

from wojtek_rl.vlm_nav import ACTIONS, SYSTEM_PROMPT, VlmDecision, _safe_err, situation_text

DEFAULT_VLM_URL = "http://127.0.0.1:11434/v1"  # override: --vlm-url / VLM_URL
DEFAULT_VLM_MODEL = "qwen3-vl:30b-a3b-instruct"  # override: --vlm-model / VLM_MODEL
# Per call. The first call after Ollama (re)starts loads the 30B's weights
# (tens of seconds); a warm call answers in about a second.
VLM_TIMEOUT_S = 120.0

_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _response_format(actions: tuple[str, ...]) -> str:
    """The JSON-response contract appended to the prompt, over `actions`."""
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
    """The OpenAI-compatible routes live under /v1; accept the base_url
    with or without it already appended."""
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    return base_url


class OpenAIVlmClient:
    """Chat-completions client for the Ollama server.

    One httpx.AsyncClient is created lazily on first use and kept for the
    life of this instance -- share one OpenAIVlmClient across concurrent
    eval episodes rather than building a fresh one per episode, so requests
    reuse the connection pool. Call close() when done with it.

    By default a failed request goes straight back to the navigator; a long
    batch run passes retries= to wait out a server restart.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_VLM_URL,
        model: str = DEFAULT_VLM_MODEL,
        api_key: str = "ollama",
        system_prompt: str = SYSTEM_PROMPT,
        actions: tuple[str, ...] = ACTIONS,
        max_tokens: int = 300,
        temperature: float = 0.0,
        timeout_s: float = VLM_TIMEOUT_S,
        retries: int = 0,  # extra tries after the first failed request
        retry_backoff_s: float = 12.0,  # before extra try k, wait k times this
    ):
        if retries < 0:
            raise ValueError(f"retries must be 0 or more, got {retries}")
        self.base_url = _normalize_base_url(base_url)
        self.model = model
        self.api_key = api_key
        self.system_prompt = system_prompt
        self.actions = actions
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.retries = retries
        self.retry_backoff_s = retry_backoff_s
        self._format = _response_format(actions)
        self._client = None  # httpx.AsyncClient, created on first use

    def _ensure_client(self):
        import httpx  # lazy: the `demo`/`eval` extras carry it

        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout_s)
        return self._client

    async def decide(self, goal, ego_b64, history, step, max_steps, pose) -> VlmDecision:
        import httpx

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
        # By default a failed request goes straight back to the navigator,
        # which already lets a goal survive a couple of failed steps. A long
        # batch run passes retries= so connection errors and 5xx answers wait
        # out an Ollama restart instead of failing episodes. Any other answer,
        # like a 404 for an unknown model, is never retried.
        resp = None
        last_err = None
        for attempt in range(self.retries + 1):
            if attempt > 0:
                await asyncio.sleep(self.retry_backoff_s * attempt)
            try:
                resp = await client.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=headers
                )
            except httpx.TransportError as e:
                last_err = e
                logger.warning(f"vlm connect attempt {attempt + 1}: {_safe_err(e)}")
                continue
            if resp.status_code < 500:
                break
            logger.warning(f"vlm 5xx attempt {attempt + 1}: {_safe_err(resp.text[:120])}")
        if resp is None:
            raise RuntimeError(f"VLM server unreachable at {self.base_url}: {_safe_err(last_err)}")
        latency_ms = (time.monotonic() - t0) * 1000
        if resp.status_code != 200:
            raise RuntimeError(f"VLM endpoint {resp.status_code}: {_safe_err(resp.text[:200])}")
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage") or {}
        tokens = usage.get("total_tokens")
        suffix = f", tokens={tokens}" if tokens is not None else ""
        logger.debug(f"vlm call: {latency_ms:.0f} ms{suffix}")
        return parse_decision(content, self.actions)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
