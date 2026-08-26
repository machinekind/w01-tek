"""Intent routing: which brain handles a final transcript.

Two implementations behind one call shape:

- RuleRouter — Polish keyword/regex rules, no dependencies.  The launch
  default until the fine-tuned encoder lands, and the permanent fallback.
- EncoderRouter — a fine-tuned sequence classifier (HerBERT/mmBERT/
  ModernBERT, trained by tools/train_router.py on the synthetic set from
  tools/gen_router_dataset.py).

Both return (intent, confidence).  Below `min_confidence` the router node
downgrades to `chat` — a mis-route to Bielik costs words, a mis-route to
navigation costs a walking robot.

Priority order matters: `cancel` must win over everything ("stop" said
mid-route is a cancel, not a new instruction), `system` and `visual` are
narrow patterns, `nav` is broad, `chat` is the rest.
"""

from __future__ import annotations

import re

INTENTS = ("chat", "nav", "visual", "cancel", "system")

_CANCEL_RE = re.compile(
    r"\b(stop|stój|przestań|anuluj|zatrzymaj się|zatrzymaj|dosyć|wystarczy|"
    r"nie idź|zostań)\b",
    re.IGNORECASE,
)
_SYSTEM_RE = re.compile(
    r"\b(zresetuj|resetuj|restart|głośniej|ciszej|zmień głos|"
    r"wyłącz (mikrofon|dźwięk)|włącz (mikrofon|dźwięk))\b",
    re.IGNORECASE,
)
_VISUAL_RE = re.compile(
    # Perception AND self-status questions both go to the VLM agent: it has
    # the camera and the pose, Bielik has neither.  "Gdzie teraz jesteś i co
    # robisz?" routed to chat produced a confident story about a garden on
    # camera (2026-08-26) -- a blind persona answers anything.
    r"(co (teraz )?widzisz|co masz przed (sobą|oczami)|opisz (co widzisz|"
    r"otoczenie|pokój|widok)|rozejrzyj się|co jest (przed tobą|obok|dookoła|"
    r"wokół)|widzisz (jakiś|jakieś|gdzieś|coś|tu|tam)|"
    r"gdzie (teraz )?(jesteś|stoisz|się znajdujesz)|"
    r"co (teraz )?(robisz|porabiasz)|"
    r"(dokąd|gdzie|którędy) (teraz )?(idziesz|zmierzasz|biegniesz))",
    re.IGNORECASE,
)
_NAV_RE = re.compile(
    r"\b(idź|pójdź|podejdź|chodź|zaprowadź|zawieź|znajdź|poszukaj|szukaj|"
    r"odszukaj|obróć się|obróć|skręć|zawróć|cofnij|wróć|omiń|okrąż|obejdź|"
    r"podbiegnij|biegnij|ruszaj|jedź|stań (przy|obok|koło)|przynieś|"
    r"zbliż się|oddal się)\b",
    re.IGNORECASE,
)


class RuleRouter:
    """Deterministic Polish keyword router.  Cheap, explainable, testable."""

    def classify(self, text: str) -> tuple[str, float]:
        t = " ".join((text or "").split())
        if not t:
            return ("chat", 0.0)
        if _CANCEL_RE.search(t):
            return ("cancel", 0.95)
        if _SYSTEM_RE.search(t):
            return ("system", 0.9)
        if _VISUAL_RE.search(t):
            return ("visual", 0.85)
        if _NAV_RE.search(t):
            return ("nav", 0.85)
        return ("chat", 0.6)


class EncoderRouter:
    """Fine-tuned sequence classifier.  Lazy-loads transformers on first use.

    The model directory must carry an id2label mapping using the INTENTS
    names (tools/train_router.py writes it that way).
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            from transformers import pipeline

            self._pipe = pipeline(
                "text-classification", model=self.model_path, device=self.device
            )
        return self._pipe

    def classify(self, text: str) -> tuple[str, float]:
        t = " ".join((text or "").split())
        if not t:
            return ("chat", 0.0)
        out = self._load()(t, truncation=True, max_length=128)[0]
        label = out["label"]
        if label not in INTENTS:
            return ("chat", 0.0)
        return (label, float(out["score"]))


def apply_confidence_floor(
    intent: str, confidence: float, min_confidence: float
) -> tuple[str, float]:
    """Below the floor everything becomes chat — words are the cheap error."""
    if intent != "chat" and confidence < min_confidence:
        return ("chat", confidence)
    return (intent, confidence)
