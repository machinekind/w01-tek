"""Local VLM backend: Qwen3-VL on Apple Silicon via mlx-vlm, no API needed.

Drop-in replacement for AnthropicVlmClient behind the same VlmClientProto:
the navigator loop, safety clamps and parse_command gate are untouched. The
default model is Qwen3-VL-30B-A3B-Instruct (4-bit MLX quant, ~18 GB): a
mixture-of-experts with ~3 B active parameters per token, so it decodes at
interactive speed on an M-series Mac while keeping 30B-class scene
understanding.

Local models have no forced tool use, so the prompt asks for ONE single-line
JSON object and parse_local_text() extracts it -- with the same free-text
regex fallback the Anthropic path uses when a model ignores tool_choice.

mlx-vlm is an optional extra (`uv sync --extra vlm-local`) and is imported
lazily, so the server and all offline tests run without it. Weights download
from HuggingFace on first load (pre-fetch with
`hf download mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit`).
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import re
import threading

from loguru import logger

from wojtek_rl.vlm_nav import (
    ACTIONS,
    SYSTEM_PROMPT,
    _TEXT_CMD_RE,
    VlmDecision,
    situation_text,
)

DEFAULT_LOCAL_MODEL = "mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit"
# First decide() may still be loading weights (~20 s from disk; much longer if
# the snapshot is still downloading). Generation itself is ~5-15 s.
LOCAL_VLM_TIMEOUT_S = 240.0
MAX_TOKENS = 400

RESPONSE_FORMAT = (
    "Respond with ONLY one JSON object on a single line, no other text:\n"
    '{"action": ' + " | ".join(f'"{a}"' for a in ACTIONS) + ", "
    '"amount": <number or null>, "reasoning": "<one short sentence>"}'
)

_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def local_prompt(
    goal: str,
    history: list[dict],
    step: int,
    max_steps: int,
    pose: tuple[float, float, float],
) -> str:
    """Single text prompt: system rules + situation + JSON format contract.

    Folded into one user turn because chat-template system-message handling
    varies across mlx-vlm versions; Qwen follows it either way.
    """
    return "\n\n".join(
        [SYSTEM_PROMPT, situation_text(goal, history, step, max_steps, pose), RESPONSE_FORMAT]
    )


def parse_local_text(text: str) -> VlmDecision:
    """Extract a VlmDecision from raw model text.

    Scans for the first JSON object with a valid action (tolerates code
    fences, leading prose, trailing tokens); falls back to the shared
    free-text command regex. Raises ValueError if neither matches -- the
    navigator counts that as a failed step, same as an API error.
    """
    for m in _JSON_RE.finditer(text):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or obj.get("action") not in ACTIONS:
            continue
        amount = obj.get("amount")
        return VlmDecision(
            action=obj["action"],
            amount=float(amount) if amount is not None else None,
            reasoning=str(obj.get("reasoning", "")),
        )
    m = _TEXT_CMD_RE.search(text)
    if m:
        amount = float(m.group(2)) if m.group(2) else None
        return VlmDecision(action=m.group(1).lower(), amount=amount, reasoning=text.strip())
    raise ValueError(f"no JSON decision or command found in model output: {text[:200]!r}")


class LocalVlmClient:
    """Qwen3-VL via mlx-vlm; the only place the mlx stack is touched.

    Weights load once (thread-safe, lazily or via preload()) and stay
    resident. Generation is synchronous mlx code, so decide() pushes it to a
    worker thread -- the 50 Hz websocket loop keeps running while the model
    thinks.
    """

    def __init__(self, model: str = DEFAULT_LOCAL_MODEL):
        self.model = model
        self._lock = threading.Lock()
        self._loaded = None  # (model, processor, config)

    def preload(self) -> None:
        """Blocking weight load; call from a background thread at startup."""
        self._ensure_loaded()

    def _ensure_loaded(self):
        with self._lock:
            if self._loaded is None:
                from mlx_vlm import load  # lazy: `vlm-local` extra is optional
                from mlx_vlm.utils import load_config

                logger.info(f"loading local VLM {self.model} (first run downloads weights)...")
                model, processor = load(self.model)
                config = load_config(self.model)
                self._loaded = (model, processor, config)
                logger.info(f"local VLM {self.model} ready")
        return self._loaded

    def _generate(self, prompt: str, image) -> str:
        model, processor, config = self._ensure_loaded()
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        formatted = apply_chat_template(processor, config, prompt, num_images=1)
        out = generate(
            model, processor, formatted, [image],
            max_tokens=MAX_TOKENS, temperature=0.0, verbose=False,
        )
        return getattr(out, "text", out)  # GenerationResult in mlx-vlm>=0.2, str before

    async def decide(self, goal, ego_b64, history, step, max_steps, pose) -> VlmDecision:
        from PIL import Image

        image = Image.open(io.BytesIO(base64.b64decode(ego_b64))).convert("RGB")
        prompt = local_prompt(goal, history, step, max_steps, pose)
        text = await asyncio.to_thread(self._generate, prompt, image)
        return parse_local_text(text)
