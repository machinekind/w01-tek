"""Utterance streaming for the ROS voice nodes.

The frame/segmentation primitives live in `wojtek_agent.audio_frames` and are
imported, not copied: this package and `wojtek_agent` ship together as one
experiment, so there is exactly one VoiceSegmenter to fix when a segmentation
bug turns up.  `UtteranceStream` is the ROS-side addition -- it drives the
segmenter from an AudioChunk topic and hands finished utterances to the ASR
node.

Deployment note: the node venv needs the experiment root on PYTHONPATH so
`wojtek_agent` is importable next to the colcon overlay.
"""

from __future__ import annotations

import numpy as np

from wojtek_agent.audio_frames import (  # noqa: F401  (re-exported for the nodes)
    FRAME_MS,
    MAX_UTTERANCE_S,
    MIN_UTTERANCE_S,
    PREROLL_S,
    SAMPLE_RATE,
    SILENCE_END_S,
    SPEECH_RMS,
    Utterance,
    VoiceSegmenter,
    frame_rms,
    pcm_frames,
    resample_linear,
)









class UtteranceStream:
    """Event view over a VoiceSegmenter, for a node that must narrate state.

    feed() yields (event, utterance_id, payload) triples:
    ("started", id, None) on the speech trigger, ("frame", id, pcm) for every
    in-speech frame (pre-roll first), and ("ended", id, Utterance) when the
    segment closes — the Utterance carries the full PCM, so a final-only
    consumer needs no reassembly.  Segments that close below the minimum
    length yield ("aborted", id, None).
    """

    def __init__(self, segmenter: VoiceSegmenter | None = None, make_id=None):
        self.seg = segmenter or VoiceSegmenter()
        self._n = 0
        self._make_id = make_id or self._default_id
        self.utterance_id: str | None = None
        self._emitted: int = 0  # voiced frames already yielded as "frame"

    def _default_id(self) -> str:
        self._n += 1
        return f"utt-{self._n:04d}"

    def feed(self, pcm: np.ndarray):
        was_speaking = self.seg.speaking
        utt = self.seg.feed(pcm)

        if not was_speaking and self.seg.speaking:
            self.utterance_id = self._make_id()
            self._emitted = 0
            yield ("started", self.utterance_id, None)

        if self.seg.speaking:
            # Emit whatever the segmenter holds beyond what we already sent —
            # on the trigger frame that includes the pre-roll.
            for frame in self.seg._voiced[self._emitted:]:
                yield ("frame", self.utterance_id, frame)
            self._emitted = len(self.seg._voiced)

        if utt is not None:
            uid, self.utterance_id = self.utterance_id, None
            yield ("ended", uid, utt)
        elif was_speaking and not self.seg.speaking:
            uid, self.utterance_id = self.utterance_id, None
            yield ("aborted", uid, None)

    def flush(self):
        uid, self.utterance_id = self.utterance_id, None
        utt = self.seg.flush()
        if utt is not None:
            yield ("ended", uid, utt)
        elif uid is not None:
            yield ("aborted", uid, None)
