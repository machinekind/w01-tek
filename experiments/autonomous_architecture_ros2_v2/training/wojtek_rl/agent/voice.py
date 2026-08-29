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
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger

from wojtek_rl import perf

SAMPLE_RATE = 24000          # what the browser worklet produces
FRAME_MS = 100               # one websocket binary frame
SPEECH_RMS = 0.012           # normalised RMS above which a frame counts as speech
MIN_UTTERANCE_S = 0.35       # shorter than this is a cough, not a sentence
SILENCE_END_S = 0.7          # trailing silence that closes an utterance
MAX_UTTERANCE_S = 20.0       # hard stop; something is wrong (open mic, TV on)
PREROLL_S = 0.3              # audio kept from before the trigger, so no clipped onsets


def frame_rms(pcm: np.ndarray) -> float:
    """Normalised RMS of an int16 frame, 0.0-1.0."""
    if not len(pcm):
        return 0.0
    x = pcm.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(x * x)))


@dataclass
class Utterance:
    """One closed segment of speech, ready for the recogniser."""

    pcm: np.ndarray              # int16, mono, SAMPLE_RATE
    seconds: float
    ended_on: str                # "silence" | "max_length" | "flush"
    # Trailing silence that had to accumulate before the segmenter believed
    # the sentence was over. It is pure dead time on the critical path -- the
    # human has stopped talking and nothing is happening yet -- and it is the
    # one stage that no faster GPU shortens, so the profiler reports it
    # alongside the model calls rather than hiding it inside them.
    endpoint_wait_s: float = 0.0

    def to_wav_bytes(self) -> bytes:
        """16-bit PCM WAV in memory -- what faster-whisper wants to open."""
        import io
        import wave

        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(self.pcm.tobytes())
        return buf.getvalue()


@dataclass
class VoiceSegmenter:
    """Turns a stream of PCM frames into utterances.

    Stateful and cheap: feed() returns an Utterance the moment speech ends,
    otherwise None. Pre-roll keeps the audio just before the trigger so the
    first syllable is not clipped -- a recogniser that never hears the "Wo"
    of "Wojtek" invents something else.
    """

    sample_rate: int = SAMPLE_RATE
    speech_rms: float = SPEECH_RMS
    silence_end_s: float = SILENCE_END_S
    min_utterance_s: float = MIN_UTTERANCE_S
    max_utterance_s: float = MAX_UTTERANCE_S
    preroll_s: float = PREROLL_S
    vad = None  # optional callable(pcm_int16) -> bool, overriding the RMS test

    _speaking: bool = field(default=False, init=False)
    _voiced: list[np.ndarray] = field(default_factory=list, init=False)
    _preroll: list[np.ndarray] = field(default_factory=list, init=False)
    _silence_s: float = field(default=0.0, init=False)
    _length_s: float = field(default=0.0, init=False)

    @property
    def speaking(self) -> bool:
        return self._speaking

    def _is_speech(self, pcm: np.ndarray) -> bool:
        if self.vad is not None:
            return bool(self.vad(pcm))
        return frame_rms(pcm) >= self.speech_rms

    def feed(self, pcm: np.ndarray) -> Utterance | None:
        if not len(pcm):
            return None
        dur = len(pcm) / self.sample_rate
        speech = self._is_speech(pcm)

        if not self._speaking:
            # Keep a rolling pre-roll window while idle.
            self._preroll.append(pcm)
            kept = 0.0
            for i in range(len(self._preroll) - 1, -1, -1):
                kept += len(self._preroll[i]) / self.sample_rate
                if kept >= self.preroll_s:
                    del self._preroll[:i]
                    break
            if speech:
                self._speaking = True
                self._voiced = list(self._preroll)
                self._length_s = sum(len(f) for f in self._voiced) / self.sample_rate
                self._preroll = []
                self._silence_s = 0.0
            return None

        self._voiced.append(pcm)
        self._length_s += dur
        self._silence_s = 0.0 if speech else self._silence_s + dur

        if self._silence_s >= self.silence_end_s:
            return self._close("silence")
        if self._length_s >= self.max_utterance_s:
            return self._close("max_length")
        return None

    def flush(self) -> Utterance | None:
        """End the utterance now (mic released, socket closing)."""
        return self._close("flush") if self._speaking else None

    def reset(self) -> None:
        self._speaking = False
        self._voiced = []
        self._preroll = []
        self._silence_s = 0.0
        self._length_s = 0.0

    def _close(self, reason: str) -> Utterance | None:
        pcm = np.concatenate(self._voiced) if self._voiced else np.zeros(0, np.int16)
        seconds = len(pcm) / self.sample_rate
        # Judge the length on speech alone, discounting however much trailing
        # silence actually accumulated -- NOT the configured timeout, which a
        # flush never reaches and which would then discard a real sentence.
        silence_s = self._silence_s
        speech_s = seconds - silence_s
        self.reset()
        if speech_s < self.min_utterance_s:
            return None
        return Utterance(
            pcm=pcm, seconds=seconds, ended_on=reason, endpoint_wait_s=silence_s
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
        # The turn's clock starts HERE: the human has finished speaking and
        # everything from now to the first spoken syllable is waiting.
        perf.start_turn("voice", seconds=round(utt.seconds, 2), ended_on=utt.ended_on)
        perf.record("mic.endpoint", utt.endpoint_wait_s * 1000.0,
                    ended_on=utt.ended_on)
        try:
            with perf.span("asr.transcribe", audio_s=round(utt.seconds, 2)) as sp:
                text = await asyncio.to_thread(self._transcribe_blocking, utt)
                sp["chars"] = len(text or "")
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


def pcm_frames(pcm: np.ndarray, frame_ms: int = FRAME_MS, rate: int = SAMPLE_RATE):
    """Slice PCM into playback-sized frames (the browser plays them in order)."""
    n = max(1, int(rate * frame_ms / 1000))
    for i in range(0, len(pcm), n):
        yield pcm[i : i + n]


def resample_linear(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Cheap linear resample -- TTS engines rarely emit at the browser's rate.

    Good enough for speech playback; a windowed-sinc would cost more than the
    artefact is worth at these rates.
    """
    if src_rate == dst_rate or not len(pcm):
        return pcm
    n_out = int(round(len(pcm) * dst_rate / src_rate))
    if n_out <= 0:
        return np.zeros(0, np.int16)
    x = np.linspace(0.0, len(pcm) - 1, n_out, dtype=np.float64)
    out = np.interp(x, np.arange(len(pcm)), pcm.astype(np.float64))
    return np.clip(np.rint(out), -32768, 32767).astype(np.int16)
