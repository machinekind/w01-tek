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
import re
import shutil
import subprocess
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from loguru import logger

from wojtek_rl import perf
from wojtek_rl.agent.voice import SAMPLE_RATE, pcm_frames, resample_linear

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


_EMOJI_RE = re.compile(
    "[" "\U0001f300-\U0001faff" "\U00002600-\U000027bf" "\U0001f1e6-\U0001f1ff" "]+"
)
_MARKDOWN_RE = re.compile(r"[*_`#>]+")


def speakable(text: str) -> str:
    """Strip what a voice should not read out loud.

    A chat reply may carry emoji and light markdown; spoken, "🐕" becomes
    "pies" or a pause, and asterisks become audible noise. The transcript in
    the UI keeps the original -- only the audio is cleaned.
    """
    text = _EMOJI_RE.sub("", text)
    text = _MARKDOWN_RE.sub("", text)
    return " ".join(text.split())


# Sentence splitting for chunked synthesis. Abbreviations are the whole
# problem: Polish is full of them and each one ends in a period that is not a
# sentence end. Splitting mid-sentence is audible (a dropped beat and a
# restarted intonation contour), so err towards keeping text together.
_ABBREVIATIONS = (
    "np", "itd", "itp", "tzn", "tj", "ok", "dr", "mgr", "inż", "prof", "ul",
    "godz", "min", "sek", "m.in", "tzw", "ww", "br", "pl", "ang", "por",
    "mr", "mrs", "dr", "st", "no", "vs", "etc", "e.g", "i.e",
)
_SENTENCE_END_RE = re.compile(r"(?<=[.!?…])\s+")
MIN_CHUNK_CHARS = 25  # shorter than this, glue onto the next sentence
# The FIRST chunk is the only one anyone waits through, so it is allowed to
# be short: "Widzę kanapę." (13 chars) used to be glued onto the sentence
# after it, which meant the whole reply was synthesised before a single
# sample shipped -- exactly the replies the dog gives most often lost the
# pipelining that chunking exists for (measured 2026-08-17). Later chunks
# keep the higher floor, where a merge costs nothing but smoother delivery.
FIRST_CHUNK_MIN_CHARS = 8
# A long opening sentence is split at a clause boundary instead, so the
# voice starts within a breath rather than after the whole thought.
# Roughly five seconds of speech: past that, waiting for the whole opening
# sentence is the dominant cost on a ~1.0 RTF voice, and a comma is a natural
# place to have already started talking.
FIRST_CHUNK_MAX_CHARS = 60
_CLAUSE_END_RE = re.compile(r"(?<=[,;:–—])\s+")


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


def speech_chunks(text: str, min_chars: int = MIN_CHUNK_CHARS) -> list[str]:
    """Sentence chunks, with the FIRST one cut short enough to start fast.

    `split_sentences` merges anything under min_chars forward, which is right
    for the body of a reply (the seams are audible and nobody is waiting) and
    wrong for its opening: "Widzę kanapę." would be glued to the sentence
    after it, so the whole answer had to be synthesised before a single
    sample shipped -- the pipelining that chunking exists for never happened
    for the short punchy replies the dog gives most often (measured
    2026-08-17).

    So the first chunk is taken at the first real sentence end, and only the
    remainder is merged the conservative way. A long opening sentence is cut
    at its first clause boundary instead, so the voice starts within a breath
    rather than after the whole thought.
    """
    sentences = split_sentences(text, min_chars=FIRST_CHUNK_MIN_CHARS)
    if not sentences:
        return []
    first, rest = sentences[0], sentences[1:]
    chunks = [first, *split_sentences(" ".join(rest), min_chars=min_chars)]
    if len(chunks[0]) > FIRST_CHUNK_MAX_CHARS:
        parts = _CLAUSE_END_RE.split(chunks[0], maxsplit=1)
        if len(parts) == 2 and len(parts[0]) >= FIRST_CHUNK_MIN_CHARS:
            chunks = [parts[0], parts[1], *chunks[1:]]
    return chunks


class TtsEngine(Protocol):
    """Text -> PCM16 mono at `rate`.

    An engine MAY also offer `synth_stream(text) -> Iterator[np.ndarray]`,
    yielding audio as it is produced (same contract as the ROS stack's
    engines). The Speaker prefers it when present: with a whole-utterance
    call nothing is audible until the last sample exists, which on a
    diffusion voice is seconds of silence after the answer is already on
    screen.
    """

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


def _header_ms(response, name: str) -> float | None:
    """A timing header the TTS server attached, in ms. Absent or malformed
    means an older server -- the client still works, it just cannot break the
    round trip down."""
    raw = (getattr(response, "headers", None) or {}).get(name)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


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

        t0 = time.monotonic()
        r = httpx.post(
            f"{self.url}/synthesize", json={"text": text}, timeout=self.timeout_s
        )
        r.raise_for_status()
        round_trip_ms = (time.monotonic() - t0) * 1000.0
        perf.record("tts.request", round_trip_ms, chars=len(text), bytes=len(r.content))
        # The server reports what it spent inside the request, so the round
        # trip splits into GPU time, time spent queued behind another
        # sentence, and what is left -- the wire. Without that split a slow
        # voice and a busy voice look identical from here, and they need
        # opposite fixes.
        gpu = _header_ms(r, "x-synth-ms")
        queued = _header_ms(r, "x-queue-ms") or 0.0
        if gpu is not None:
            perf.record("tts.server_gpu", gpu, chars=len(text))
            if queued:
                perf.record("tts.server_queue", queued, chars=len(text))
            perf.record("tts.network", max(0.0, round_trip_ms - gpu - queued),
                        bytes=len(r.content))
        return wav_bytes_to_pcm(r.content)

    def synth_stream(self, text: str):
        """Raw PCM16 as the server produces it, clause by clause.

        The server splits the text and flushes each piece the moment it
        exists, so time-to-first-sound stops being "the whole sentence" and
        becomes "the first clause". Falls back to the one-shot call against
        an older server (or one whose stream endpoint is missing), because a
        voice that speaks late is still better than one that does not speak.
        """
        import httpx

        try:
            with httpx.stream(
                "POST", f"{self.url}/synthesize_stream",
                json={"text": text}, timeout=self.timeout_s,
            ) as r:
                if r.status_code == 404:
                    r.read()
                    yield self.synthesize(text)
                    return
                r.raise_for_status()
                tail = b""
                for raw in r.iter_bytes():
                    if not raw:
                        continue
                    buf = tail + raw
                    # int16 frames must not be split across a chunk boundary.
                    usable = len(buf) - (len(buf) % 2)
                    tail = buf[usable:]
                    if usable:
                        yield np.frombuffer(buf[:usable], dtype=np.int16)
        except httpx.HTTPError as e:
            logger.warning(f"remote TTS stream failed ({e}); falling back to one shot")
            yield self.synthesize(text)

    def health(self) -> dict:
        import httpx

        return httpx.get(f"{self.url}/health", timeout=5.0).json()


def _rate_fields(span_fields: dict, pcm, rate: int, took_s: float) -> None:
    """Audio produced, and the real-time factor it was produced at.

    RTF above 1.0 means the voice cannot keep up with itself: no amount of
    streaming rescues that, only a faster engine. Below 1.0, streaming turns
    the wait into the first partial's time instead of the whole sentence's.
    """
    audio_s = len(pcm) / max(rate, 1)
    span_fields["audio_s"] = round(audio_s, 2)
    span_fields["rtf"] = round(took_s / audio_s, 2) if audio_s else None


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

    async def _audio_for(self, chunk: str, index: int):
        """Yield (part, pcm) for one text chunk, as early as the engine can.

        A streaming engine hands back audio while it is still generating, so
        the first partial can already be playing; a plain engine yields once,
        after the whole chunk exists. Both paths are timed the same way, so
        the report shows what streaming actually bought.

        Blocking work stays off the event loop: the generator is advanced one
        step at a time in a worker thread, which keeps the 50 Hz control loop
        and barge-in responsive while the voice model runs.
        """
        stream = getattr(self.engine, "synth_stream", None)
        if stream is None:
            t0 = time.monotonic()
            with perf.span("tts.synth", chars=len(chunk), chunk=index,
                           streamed=False) as sp:
                pcm = await asyncio.to_thread(self.engine.synthesize, chunk)
                _rate_fields(sp, pcm, self.engine.rate, time.monotonic() - t0)
            yield 0, pcm
            return
        it = await asyncio.to_thread(stream, chunk)
        part = 0
        while True:
            t0 = time.monotonic()
            with perf.span("tts.synth", chars=len(chunk), chunk=index,
                           part=part, streamed=True) as sp:
                pcm = await asyncio.to_thread(next, it, None)
                if pcm is None:
                    sp["end"] = True
                    return
                _rate_fields(sp, pcm, self.engine.rate, time.monotonic() - t0)
            yield part, pcm
            part += 1

    async def _run(self, text: str) -> None:
        """Synthesise sentence by sentence, streaming each as it is ready.

        Time-to-first-audio is what a conversation feels like, and it is set
        by the FIRST chunk rather than the whole reply: a three-sentence
        answer starts playing after one sentence has been made, while the
        rest is synthesised behind it.
        """
        chunks = speech_chunks(text) or [text]
        # Three clocks, because "why is it slow to speak" has three different
        # answers: `tts.first_audio` is this reply's time-to-sound,
        # `reply.text_to_sound` is the gap a person sees between the answer
        # appearing and being heard, and `voice.reply` is the whole mic-to-
        # voice wait. A cancelled reply (barge-in) never reaches its first
        # frame and reports none of them, rather than inventing a number.
        first_audio = perf.Timer("tts.first_audio", chars=len(text),
                                 chunks=len(chunks), first_chunk_chars=len(chunks[0]))
        spoke = False
        speak = perf.Timer("tts.speak", chars=len(text))
        for i, chunk in enumerate(chunks):
            try:
                async for part, pcm in self._audio_for(chunk, i):
                    if not len(pcm):
                        continue
                    pcm = resample_linear(pcm, self.engine.rate, self.out_rate)
                    # Pushing frames onto the socket is separate from making
                    # them: when the event loop is busy rendering the demo's
                    # video panels THIS is where the delay lands, and it
                    # would otherwise hide inside a synthesis number.
                    with perf.span("tts.stream", chunk=i, part=part,
                                   audio_s=round(len(pcm) / max(self.out_rate, 1), 2)):
                        for frame in pcm_frames(pcm, rate=self.out_rate):
                            await self.send_frame(frame.tobytes())
                            if not spoke:
                                spoke = True
                                first_audio.stop()
                                gap = perf.since_mark_ms("reply.text")
                                if gap is not None:
                                    perf.record("reply.text_to_sound", gap,
                                                chars=len(text),
                                                first_chunk_chars=len(chunks[0]))
                                waited = perf.turn_elapsed_ms()
                                if waited is not None:
                                    perf.record("voice.reply", waited, chars=len(text))
                            # Yield between frames so a cancel (barge-in)
                            # lands promptly instead of after the whole
                            # reply is queued.
                            await asyncio.sleep(0)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # missing voice, broken binary, dead server
                logger.warning(f"tts failed: {e}")
                if not spoke:
                    speak.cancel()
                    first_audio.cancel()
                return
        if spoke:
            speak.stop()
        else:
            speak.cancel()
            first_audio.cancel()
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
