"""Pluggable frame-level VAD backends for the segmenter's `vad=` hook.

Each factory returns callable(pcm_int16_frame) -> bool, or raises with an
actionable message when its dependency is missing.  `energy` is the default
(no dependency, microseconds); `silero` is the proven neural fallback;
`pyannote` is the team-chosen backend for noisy rooms and turn-taking —
it needs ~1.6 s of context, so the wrapper keeps a ring buffer and scores
the newest frame within it.
"""

from __future__ import annotations

import numpy as np


def make_vad(backend: str, sample_rate: int, threshold: float = 0.5):
    if backend == "energy":
        return None  # segmenter falls back to its RMS gate
    if backend == "silero":
        return _silero(sample_rate, threshold)
    if backend == "pyannote":
        return _pyannote(sample_rate, threshold)
    raise ValueError(f"unknown VAD backend {backend!r} (energy|silero|pyannote)")


def _silero(sample_rate: int, threshold: float):
    try:
        import torch
    except ImportError as e:
        raise RuntimeError("silero VAD needs torch: pip install torch") from e

    model, _utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
    # Silero wants 16 kHz and a fixed 512-sample window; score the frame in
    # 512-sample steps and call it speech if any window clears the threshold.
    from .transport import resample_linear

    def is_speech(pcm: np.ndarray) -> bool:
        x = resample_linear(pcm, sample_rate, 16000).astype(np.float32) / 32768.0
        for i in range(0, len(x) - 511, 512):
            p = model(torch.from_numpy(x[i : i + 512]), 16000).item()
            if p >= threshold:
                return True
        return False

    return is_speech


def _pyannote(sample_rate: int, threshold: float, context_s: float = 1.6):
    try:
        from pyannote.audio import Model
        from pyannote.core import SlidingWindowFeature  # noqa: F401 — presence check
        import torch
    except ImportError as e:
        raise RuntimeError(
            "pyannote VAD needs: pip install pyannote.audio torch "
            "(and a HF token for the first model download)"
        ) from e

    model = Model.from_pretrained("pyannote/segmentation-3.0")
    model.eval()
    from .transport import resample_linear

    ring = np.zeros(int(16000 * context_s), np.float32)

    def is_speech(pcm: np.ndarray) -> bool:
        nonlocal ring
        x = resample_linear(pcm, sample_rate, 16000).astype(np.float32) / 32768.0
        if not len(x):
            return False
        ring = np.roll(ring, -len(x))
        ring[-len(x):] = x[-len(ring):]
        with torch.no_grad():
            scores = model(torch.from_numpy(ring)[None, None, :]).squeeze(0).numpy()
        # segmentation-3.0 emits powerset speaker probabilities per time step;
        # speech probability = 1 - P(no speaker).  Judge the tail that
        # corresponds to the newest frame.
        speech_prob = 1.0 - scores[:, 0]
        tail = max(1, int(len(speech_prob) * len(x) / len(ring)))
        return bool(np.max(speech_prob[-tail:]) >= threshold)

    return is_speech
