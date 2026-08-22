"""Spoken-text hygiene shared by the demo app and the ROS brain nodes.

Turning model output into something a TTS engine should read aloud, and
splitting it into flushable sentences.  Dependency-free (`re` only) and free of
any engine, asyncio or ROS import, so `wojtek_agent.tts` (demo app) and
`wojtek_brain.sentences` (ROS nodes) share ONE implementation rather than
hand-synced copies.
"""

from __future__ import annotations

import re

__all__ = ["MIN_CHUNK_CHARS", "speakable", "split_sentences"]

_EMOJI_RE = re.compile(
    "[" "\U0001f300-\U0001faff" "\U00002600-\U000027bf" "\U0001f1e6-\U0001f1ff" "]+"
)
_MARKDOWN_RE = re.compile(r"[*_`#>]+")
_ABBREVIATIONS = (
    "np", "itd", "itp", "tzn", "tj", "ok", "dr", "mgr", "inż", "prof", "ul",
    "godz", "min", "sek", "m.in", "tzw", "ww", "br", "pl", "ang", "por",
    "mr", "mrs", "dr", "st", "no", "vs", "etc", "e.g", "i.e",
)
_SENTENCE_END_RE = re.compile(r"(?<=[.!?…])\s+")
MIN_CHUNK_CHARS = 25  # shorter than this, glue onto the next sentence


def speakable(text: str) -> str:
    """Strip what a voice should not read out loud.

    A chat reply may carry emoji and light markdown; spoken, "🐕" becomes
    "pies" or a pause, and asterisks become audible noise. The transcript in
    the UI keeps the original -- only the audio is cleaned.
    """
    text = _EMOJI_RE.sub("", text)
    text = _MARKDOWN_RE.sub("", text)
    return " ".join(text.split())


def split_sentences(text: str, min_chars: int = MIN_CHUNK_CHARS) -> list[str]:
    """Split a reply into speakable chunks, longest-safe first.

    Chunking exists purely for latency: the first sentence can be synthesised
    and playing while the rest is still being made. A too-eager split costs
    more (choppy delivery) than it saves, so single clauses are merged
    forward and known abbreviations never end a chunk.
    """
    text = " ".join((text or "").split())
    if not text:
        return []
    parts = _SENTENCE_END_RE.split(text)
    chunks: list[str] = []
    for part in parts:
        if not part:
            continue
        if chunks:
            prev = chunks[-1]
            last_word = prev.rstrip(".").rsplit(" ", 1)[-1].lower()
            # "...ok." / "...np." is an abbreviation, not an ending; and a
            # digit before the period is a decimal or an ordinal.
            if last_word in _ABBREVIATIONS or (prev[:-1] or " ")[-1].isdigit():
                chunks[-1] = f"{prev} {part}"
                continue
            if len(prev) < min_chars:
                chunks[-1] = f"{prev} {part}"
                continue
        chunks.append(part)
    return chunks
