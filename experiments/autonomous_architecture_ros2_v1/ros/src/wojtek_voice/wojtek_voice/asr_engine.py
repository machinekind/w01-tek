"""Whisper recognition with the hallucination guards from the #131 stack.

Whisper hallucinates on silence and noise — a phantom "6V" once reached the
agent as a command and preempted a running goal.  The decoder already knows
when it is guessing: drop segments it marks as probable non-speech or decodes
with low confidence.  Thresholds carried over from
training/wojtek_agent/asr_server.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_NO_SPEECH_PROB = 0.6
MIN_AVG_LOGPROB = -1.0
WHISPER_RATE = 16000


@dataclass
class Recognition:
    text: str
    confidence: float        # mean avg_logprob over kept segments (0.0 if none)
    dropped: list            # (text, no_speech_prob, avg_logprob) of guarded-out segments


def filter_segments(segments) -> Recognition:
    """Apply the no-speech/low-confidence guards to decoded segments.

    Pure function so the guard logic is testable without a model.  Accepts
    any iterable of objects with .text and optionally .no_speech_prob /
    .avg_logprob (faster-whisper's Segment shape).
    """
    kept, dropped, logprobs = [], [], []
    for s in segments:
        no_speech = getattr(s, "no_speech_prob", 0.0) or 0.0
        logprob = getattr(s, "avg_logprob", 0.0) or 0.0
        if no_speech > MAX_NO_SPEECH_PROB or logprob < MIN_AVG_LOGPROB:
            dropped.append((s.text.strip(), round(no_speech, 2), round(logprob, 2)))
            continue
        kept.append(s.text.strip())
        logprobs.append(logprob)
    text = " ".join(t for t in kept if t).strip()
    confidence = float(np.mean(logprobs)) if logprobs else 0.0
    return Recognition(text=text, confidence=confidence, dropped=dropped)


def pcm_to_whisper_input(pcm: np.ndarray, sample_rate: int) -> np.ndarray:
    """int16 at any rate -> float32 mono at 16 kHz, what faster-whisper eats."""
    from .transport import resample_linear

    pcm16 = resample_linear(pcm, sample_rate, WHISPER_RATE)
    return pcm16.astype(np.float32) / 32768.0


class WhisperEngine:
    """Lazy faster-whisper wrapper; transcribe() is blocking by design —
    the node calls it from a worker thread."""

    def __init__(self, model_size: str = "large-v3", device: str = "auto",
                 compute_type: str = "default", language: str = "pl"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_size, device=self.device, compute_type=self.compute_type
            )
            # Warm the decoder so the first real utterance does not pay
            # graph-build time (the #131 server did the same with silence).
            silence = np.zeros(WHISPER_RATE, np.float32)
            list(self._model.transcribe(silence, language=self.language, beam_size=1)[0])
        return self._model

    def transcribe(self, pcm: np.ndarray, sample_rate: int) -> Recognition:
        audio = pcm_to_whisper_input(pcm, sample_rate)
        segments, _info = self._load().transcribe(
            audio, language=self.language, beam_size=2, vad_filter=False
        )
        return filter_segments(segments)
