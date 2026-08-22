"""Speech output: Polish text in, PCM frames out.

Why this exists as its own stage rather than coming from the "omni" model:
the open Qwen Omni Talker synthesises 10 languages (de en es fr it ja ko pt
ru zh) and Polish is not one of them. Neither is it in Qwen3-TTS or
CosyVoice3. So Polish speech is produced here, next to the brain rather than
inside it.

Default engine is Piper: MIT-licensed, ONNX/VITS, several pl_PL voices
(mc_speech, darkman, gosia), and fast enough to run realtime on CPU -- which keeps
the GPU entirely for the brain. The Engine protocol is deliberately tiny, so
a GPU engine (Fish Speech S2-Pro, XTTS-v2) can replace it without touching
the websocket layer; mind their non-commercial licences.

Everything here yields PCM16 at the browser's rate so the playback worklet
can queue frames as they arrive -- a long sentence starts playing while the
rest is still being synthesised.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from loguru import logger

from wojtek_agent.speech_text import (  # noqa: F401  (re-exported)
    MIN_CHUNK_CHARS,
    speakable,
    split_sentences,
)
from wojtek_agent.voice import SAMPLE_RATE, pcm_frames, resample_linear

# Piper's pl_PL voices are 22.05 kHz; the browser context runs at 24 kHz.
PIPER_RATE = 22050
# Male Polish voice, chosen by round-trip WER (synthesise, then recognise with
# whisper large-v3) over the four pl_PL voices Piper ships, 7 sentences each:
#   mc_speech-medium  8.5%  (5/7 perfect)  RTF 0.163   <- default
#   darkman-medium   11.7%  (4/7)          RTF 0.134
#   gosia-medium     11.1%                 RTF 0.131   (female)
#   bass-high        18.5%                 RTF 0.227
# Note the "high" quality tier scored WORST -- tier is model size, not
# intelligibility. mc_speech also has the best provenance of the set: trained
# on czyzi0/the-mc-speech-dataset, 22 h of one male speaker at 44.1 kHz, CC0.
# Round-trip WER measures intelligibility, NOT naturalness; no public
# benchmark ranks Polish TTS, so a native-speaker listen is still the arbiter.
DEFAULT_VOICE = "pl_PL-mc_speech-medium"
SPEAK_MAX_CHARS = 600  # a runaway reply should not become a 3-minute monologue


# Sentence splitting for chunked synthesis. Abbreviations are the whole
# problem: Polish is full of them and each one ends in a period that is not a
# sentence end. Splitting mid-sentence is audible (a dropped beat and a
# restarted intonation contour), so err towards keeping text together.


class TtsEngine(Protocol):
    """Text -> PCM16 mono at `rate`."""

    rate: int

    def synthesize(self, text: str) -> np.ndarray: ...


@dataclass
class PiperTts:
    """Piper via its CLI, which is how it ships and how it stays optional.

    `voice` is a model name (`pl_PL-mc_speech-medium`) resolved inside
    `voices_dir`, or an explicit path to a .onnx. Nothing is downloaded
    implicitly: a missing voice raises with the exact command to fetch it,
    rather than silently falling back to an English voice.
    """

    voice: str = DEFAULT_VOICE
    voices_dir: Path | None = None
    binary: str = "piper"
    rate: int = PIPER_RATE

    def _model_path(self) -> Path:
        if self.voice.endswith(".onnx"):
            return Path(self.voice).expanduser()
        base = self.voices_dir or Path.home() / ".local/share/piper-voices"
        return Path(base).expanduser() / f"{self.voice}.onnx"

    def available(self) -> tuple[bool, str]:
        if shutil.which(self.binary) is None:
            return False, f"piper binary {self.binary!r} not on PATH (pip install piper-tts)"
        model = self._model_path()
        if not model.exists():
            return False, (
                f"voice model {model} missing -- fetch it with: "
                f"python -m piper.download_voices {self.voice}"
            )
        return True, "ok"

    def synthesize(self, text: str) -> np.ndarray:
        ok, why = self.available()
        if not ok:
            raise RuntimeError(why)
        model = self._model_path()
        proc = subprocess.run(
            [self.binary, "--model", str(model), "--output_file", "-"],
            input=text.encode("utf-8"),
            capture_output=True,
            check=True,
        )
        return wav_bytes_to_pcm(proc.stdout)


def wav_bytes_to_pcm(data: bytes) -> np.ndarray:
    """Read a WAV byte string into mono int16 (Piper writes WAV to stdout)."""
    import io

    with wave.open(io.BytesIO(data)) as w:
        frames = w.readframes(w.getnframes())
        pcm = np.frombuffer(frames, dtype=np.int16)
        if w.getnchannels() > 1:
            pcm = pcm.reshape(-1, w.getnchannels()).mean(axis=1).astype(np.int16)
    return pcm


class SilentTts:
    """No-op engine: the demo still runs (and is testable) with no voice
    installed, it just does not speak."""

    rate = SAMPLE_RATE

    def synthesize(self, text: str) -> np.ndarray:
        return np.zeros(0, np.int16)


class RemoteTts:
    """Speech from tts_server.py on a GPU box (Chatterbox multilingual with a
    cloned voice) -- same one-method shape as PiperTts, so the demo swaps
    voices with TTS_ENGINE=remote TTS_URL=http://127.0.0.1:8120.

    Blocking by design: Speaker already synthesises off the event loop.  The
    server owns the voice reference; the client just sends text.
    """

    def __init__(self, url: str, timeout_s: float = 60.0):
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s
        self.rate = SAMPLE_RATE  # server resamples to the browser rate

    def synthesize(self, text: str) -> np.ndarray:
        import httpx

        r = httpx.post(
            f"{self.url}/synthesize", json={"text": text}, timeout=self.timeout_s
        )
        r.raise_for_status()
        return wav_bytes_to_pcm(r.content)

    def health(self) -> dict:
        import httpx

        return httpx.get(f"{self.url}/health", timeout=5.0).json()


class Speaker:
    """Synthesise a reply and push PCM frames to the browser.

    One utterance at a time: starting a new one cancels whatever is still
    playing, which is what barge-in needs -- the dog must shut up the moment
    the human starts talking.
    """

    def __init__(self, engine: TtsEngine, send_frame, out_rate: int = SAMPLE_RATE,
                 max_chars: int = SPEAK_MAX_CHARS):
        self.engine = engine
        self.send_frame = send_frame     # async callable(pcm_bytes: bytes)
        self.out_rate = out_rate
        self.max_chars = max_chars
        self._task: asyncio.Task | None = None
        self.spoken = 0                  # utterances completed (UI/tests)

    @property
    def speaking(self) -> bool:
        return self._task is not None and not self._task.done()

    def say(self, text: str) -> bool:
        """Queue a reply for speaking. Returns False when there is nothing
        to say; cancels any utterance already in flight."""
        text = speakable(text or "")
        if not text:
            return False
        self.cancel()
        self._task = asyncio.create_task(self._run(text[: self.max_chars]))
        return True

    def cancel(self) -> None:
        if self.speaking:
            self._task.cancel()
        self._task = None

    async def _run(self, text: str) -> None:
        """Synthesise sentence by sentence, streaming each as it is ready.

        Time-to-first-audio is what a conversation feels like, and it is set
        by the FIRST chunk rather than the whole reply: a three-sentence
        answer starts playing after one sentence has been made, while the
        rest is synthesised behind it.
        """
        chunks = split_sentences(text) or [text]
        for chunk in chunks:
            try:
                pcm = await asyncio.to_thread(self.engine.synthesize, chunk)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # missing voice, broken binary
                logger.warning(f"tts failed: {e}")
                return
            if not len(pcm):
                continue
            pcm = resample_linear(pcm, self.engine.rate, self.out_rate)
            for frame in pcm_frames(pcm, rate=self.out_rate):
                await self.send_frame(frame.tobytes())
                # Yield between frames so a cancel (barge-in) lands promptly
                # instead of after the whole reply has been queued.
                await asyncio.sleep(0)
        self.spoken += 1


def build_engine(kind: str = "piper", voice: str = DEFAULT_VOICE) -> TtsEngine:
    """`kind`: piper | remote | none. `remote` reads TTS_URL. Unknown kinds
    fall back to silence with a warning rather than taking the demo down."""
    if kind == "none":
        return SilentTts()
    if kind == "remote":
        url = os.environ.get("TTS_URL", "http://127.0.0.1:8120")
        engine = RemoteTts(url)
        try:
            engine.health()
        except Exception as e:
            logger.warning(f"remote TTS at {url} unreachable ({e}); running silent")
            return SilentTts()
        return engine
    if kind != "piper":
        logger.warning(f"unknown TTS engine {kind!r}; running silent")
        return SilentTts()
    engine = PiperTts(voice=voice)
    ok, why = engine.available()
    if not ok:
        logger.warning(f"piper unavailable ({why}); running silent")
        return SilentTts()
    return engine
