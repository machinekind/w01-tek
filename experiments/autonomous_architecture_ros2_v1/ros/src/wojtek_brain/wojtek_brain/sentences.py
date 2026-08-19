"""Text-to-speech text hygiene and sentence assembly for token streams.

speakable()/split_sentences() are lifted from training/wojtek_agent/tts.py
(#131) — keep fixes in sync.  SentenceAssembler is new: it turns an LLM
token stream into flushable sentences so TTS starts while the model is
still generating (the punctuation-flush contract from the design doc).
"""

from __future__ import annotations

import re

_EMOJI_RE = re.compile(
    "[" "\U0001f300-\U0001faff" "\U00002600-\U000027bf" "\U0001f1e6-\U0001f1ff" "]+"
)
_MARKDOWN_RE = re.compile(r"[*_`#>]+")


def speakable(text: str) -> str:
    """Strip what a voice should not read out loud.

    A chat reply may carry emoji and light markdown; spoken, "🐕" becomes
    "pies" or a pause, and asterisks become audible noise.  The transcript in
    the UI keeps the original — only the audio is cleaned.
    """
    text = _EMOJI_RE.sub("", text)
    text = _MARKDOWN_RE.sub("", text)
    return " ".join(text.split())


# Sentence splitting for chunked synthesis.  Abbreviations are the whole
# problem: Polish is full of them and each one ends in a period that is not a
# sentence end.  Splitting mid-sentence is audible (a dropped beat and a
# restarted intonation contour), so err towards keeping text together.
_ABBREVIATIONS = (
    "np", "itd", "itp", "tzn", "tj", "ok", "dr", "mgr", "inż", "prof", "ul",
    "godz", "min", "sek", "m.in", "tzw", "ww", "br", "pl", "ang", "por",
    "mr", "mrs", "dr", "st", "no", "vs", "etc", "e.g", "i.e",
)
_SENTENCE_END_RE = re.compile(r"(?<=[.!?…])\s+")
MIN_CHUNK_CHARS = 25  # shorter than this, glue onto the next sentence


def split_sentences(text: str, min_chars: int = MIN_CHUNK_CHARS) -> list[str]:
    """Split a reply into speakable chunks, longest-safe first.

    Chunking exists purely for latency: the first sentence can be synthesised
    and playing while the rest is still being made.  A too-eager split costs
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


class SentenceAssembler:
    """Accumulates streamed tokens, emits complete sentences.

    feed() returns the sentences that became complete with this token
    (usually zero or one).  A sentence is complete when the buffer contains
    a sentence end that split_sentences() respects — abbreviation periods
    and short fragments keep accumulating.  flush() empties whatever
    remains (end of generation).
    """

    def __init__(self, min_chars: int = MIN_CHUNK_CHARS):
        self.min_chars = min_chars
        self._buf = ""

    def feed(self, token: str) -> list[str]:
        self._buf += token
        # A sentence can only be judged complete once something follows its
        # end mark — split_sentences needs the trailing whitespace to split,
        # and a period at the very tip may still be an abbreviation.
        chunks = split_sentences(self._buf, self.min_chars)
        if len(chunks) < 2:
            return []
        done, self._buf = chunks[:-1], chunks[-1]
        return done

    def flush(self) -> list[str]:
        chunks = split_sentences(self._buf, self.min_chars)
        self._buf = ""
        return chunks
