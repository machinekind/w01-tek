"""Realtime voice plumbing: streaming mic audio in, spoken audio out.

Model-agnostic on purpose. The browser sends 100 ms frames of mono PCM16 at
24 kHz (the AudioWorklet capture path lifted from the gpt-realtime demo);
this module turns that stream into utterances, and turns reply text back into
PCM frames for playback. Which brain sits between them is somebody else's
problem -- swap the ASR or the TTS without touching the transport.

Why an ASR stage at all, when "omni" models take audio directly: the open
Qwen Omni weights do not support Polish speech input (Qwen3-Omni lists 19
speech languages, Polish is not among them), and the Qwen model that is
excellent at Polish speech is API-only. Whisper large-v3 is the best-measured
open Polish recogniser (FLEURS 4.74), so Polish audio goes through it and the
brain receives text.

Segmentation is energy-based by default rather than a neural VAD: it needs no
extra dependency, runs in microseconds, and the failure mode we care about
(cutting a sentence in half) is governed by the silence timeout, not by
frame-level precision. Pass a `vad` callable to swap in silero if the room is
noisy.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
from loguru import logger

# One implementation of the frame/segmentation primitives, shared with the
# ROS voice nodes (wojtek_voice.transport) -- see wojtek_agent.audio_frames.
from wojtek_agent.audio_frames import (  # noqa: F401  (re-exported)
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


class RemoteTranscriber:
    """Drop-in for wojtek_eval.hearing.Transcriber, backed by asr_server.

    Same one method the listener calls, so nothing else changes: point
    ASR_URL at a GPU box and recognition goes from ~2x realtime on a laptop
    CPU to ~0.13x. Blocking by design -- the listener already calls it in a
    worker thread.
    """

    def __init__(self, url: str, timeout_s: float = 30.0):
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s

    def transcribe(self, wav_path) -> str:
        import httpx

        data = Path(wav_path).read_bytes() if not hasattr(wav_path, "read") else wav_path.read()
        r = httpx.post(f"{self.url}/transcribe", content=data, timeout=self.timeout_s)
        r.raise_for_status()
        payload = r.json()
        if not payload.get("ok"):
            raise RuntimeError(payload.get("error", "remote ASR failed"))
        return payload.get("text", "")

    def health(self) -> dict:
        import httpx

        return httpx.get(f"{self.url}/health", timeout=5.0).json()


class VoiceListener:
    """Segmenter + recogniser, driven from the websocket's binary frames.

    Transcription runs in a worker thread (faster-whisper is blocking), so
    the 50 Hz control loop on the same event loop keeps running while the
    robot is being listened to.
    """

    def __init__(self, transcriber, on_text, segmenter: VoiceSegmenter | None = None):
        self.transcriber = transcriber
        self.on_text = on_text          # async callable(text: str, utterance)
        self.seg = segmenter or VoiceSegmenter()
        self.enabled = False
        self._task: asyncio.Task | None = None

    def set_enabled(self, on: bool) -> None:
        self.enabled = on
        if not on:
            self.seg.reset()

    async def feed_frame(self, raw: bytes) -> None:
        if not self.enabled or not raw:
            return
        utt = self.seg.feed(np.frombuffer(raw, dtype=np.int16))
        if utt is not None:
            await self._recognise(utt)

    async def _recognise(self, utt: Utterance) -> None:
        try:
            text = await asyncio.to_thread(self._transcribe_blocking, utt)
        except Exception as e:  # a bad utterance must not kill the socket
            logger.warning(f"transcription failed: {e}")
            return
        text = (text or "").strip()
        logger.info(f"heard ({utt.seconds:.1f}s, {utt.ended_on}): {text!r}")
        if text:
            await self.on_text(text, utt)

    def _transcribe_blocking(self, utt: Utterance) -> str:
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(utt.to_wav_bytes())
            tmp = Path(f.name)
        try:
            return self.transcriber.transcribe(tmp)
        finally:
            tmp.unlink(missing_ok=True)
