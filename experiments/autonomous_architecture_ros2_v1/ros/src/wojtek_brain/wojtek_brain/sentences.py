"""Sentence assembly for the ROS brain nodes.

`speakable()` / `split_sentences()` live in `wojtek_agent.speech_text` and are
imported, not copied -- this package and `wojtek_agent` ship together as one
experiment.  `SentenceAssembler` is the ROS-side addition: it turns an LLM
token stream into flushable sentences so TTS starts while the model is still
generating (the punctuation-flush contract from the design doc).

Deployment note: the node venv needs the experiment root on PYTHONPATH so
`wojtek_agent` is importable next to the colcon overlay.
"""

from __future__ import annotations

from wojtek_agent.speech_text import (  # noqa: F401  (re-exported for the nodes)
    MIN_CHUNK_CHARS,
    speakable,
    split_sentences,
)





# Sentence splitting for chunked synthesis.  Abbreviations are the whole
# problem: Polish is full of them and each one ends in a period that is not a
# sentence end.  Splitting mid-sentence is audible (a dropped beat and a
# restarted intonation contour), so err towards keeping text together.




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
