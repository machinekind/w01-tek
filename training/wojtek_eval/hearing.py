"""Proves the dog can HEAR: instruction text -> speech -> transcript -> goal.

Each episode's instruction is synthesized to speech with macOS `say` (a
stand-in mouth), converted to 16 kHz mono WAV, then transcribed back with
faster-whisper (the ears). The TRANSCRIPT -- however garbled by TTS/ASR
round-trip noise -- is what the navigation VLM actually receives as its
goal string, and word error rate against the original text is the metric
we report. This is intentionally not a clean-text eval: robustness to a
mangled goal is the point.

`synthesize()` is the eval-only half of this chain: it exists to manufacture
audio offline, on a Mac, without a human in the loop. The real robot skips it
entirely and feeds Transcriber.transcribe() audio from a live microphone --
Transcriber is exactly what ships.
"""

from __future__ import annotations

import hashlib
import re
import string
import subprocess
import threading
from pathlib import Path

from loguru import logger

VOICES = ["Samantha", "Daniel", "Karen", "Moira", "Rishi", "Alex"]

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def pick_voice(rng) -> str:
    """Pick a macOS voice for variety; rng is a numpy Generator."""
    return rng.choice(VOICES)


def synthesize(text: str, wav_path: Path, voice: str = "Samantha", rate_wpm: int = 190) -> Path:
    """Speak `text` with macOS `say`, then downsample to 16 kHz mono WAV.

    Writes an intermediate .aiff next to wav_path via `say`, converts it with
    `afconvert` (both ship with macOS), and deletes the .aiff. If `voice` is
    missing on this machine `say` exits non-zero -- retried once with the
    system default voice (no -v flag) before giving up.
    """
    wav_path = Path(wav_path)
    aiff_path = wav_path.with_suffix(".aiff")

    cmd = ["say", "-v", voice, "-r", str(rate_wpm), "-o", str(aiff_path), text]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        logger.warning(f"say -v {voice} failed ({exc.stderr.strip()!r}), retrying with default voice")
        fallback_cmd = ["say", "-r", str(rate_wpm), "-o", str(aiff_path), text]
        try:
            subprocess.run(fallback_cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc2:
            raise RuntimeError(f"say failed: {exc2.stderr}") from exc2

    convert_cmd = [
        "afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
        str(aiff_path), str(wav_path),
    ]
    try:
        subprocess.run(convert_cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"afconvert failed: {exc.stderr}") from exc

    aiff_path.unlink(missing_ok=True)
    return wav_path


class Transcriber:
    """Lazy-loading faster-whisper wrapper; the model stays resident once loaded.

    faster-whisper is an optional `eval` extra (see pyproject.toml), so the
    import lives inside _ensure_loaded() -- same lazy-import pattern as
    wojtek_rl/vlm_local.py uses for mlx, keeping every other module usable
    without it installed.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = "en",
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        # Whisper decodes far better when told the language, and the eval
        # battery is English by construction -- hence the default. Pass "pl"
        # (or None to auto-detect) for the live demo; "small" int8 on CPU is
        # noticeably weaker outside English, so a Polish deployment wants
        # medium/large-v3, ideally on the GPU.
        self.language = language
        self._lock = threading.Lock()
        self._model = None

    def _ensure_loaded(self):
        with self._lock:
            if self._model is None:
                from faster_whisper import WhisperModel  # lazy: `eval` extra is optional

                logger.info(f"loading faster-whisper model {self.model_size!r}...")
                self._model = WhisperModel(
                    self.model_size, device=self.device, compute_type=self.compute_type
                )
                logger.info("faster-whisper model ready")
        return self._model

    def transcribe(self, wav_path: Path) -> str:
        model = self._ensure_loaded()
        segments, _info = model.transcribe(
            str(wav_path), beam_size=2, language=self.language, vad_filter=False
        )
        return " ".join(seg.text.strip() for seg in segments).strip()


def _normalize_words(text: str) -> list[str]:
    lowered = text.lower()
    stripped = lowered.translate(str.maketrans("", "", string.punctuation))
    return stripped.split()


def wer(ref: str, hyp: str) -> float:
    """Word error rate: Levenshtein edit distance over normalized words / len(ref).

    Normalization lowercases and strips punctuation before splitting on
    whitespace. An empty reference is defined as 0.0 WER if the hypothesis is
    also empty (nothing to get wrong), else 1.0 (all insertions).
    """
    ref_words = _normalize_words(ref)
    hyp_words = _normalize_words(hyp)

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    n, m = len(ref_words), len(hyp_words)
    # dp[i][j] = edit distance between ref_words[:i] and hyp_words[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],  # deletion
                    dp[i][j - 1],  # insertion
                    dp[i - 1][j - 1],  # substitution
                )
    return dp[n][m] / n


def _slugify(text: str) -> str:
    """First 40 chars of `text` reduced to alnum/dash, plus a short hash for uniqueness."""
    slug = _SLUG_RE.sub("-", text.lower()).strip("-")[:40]
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{digest}" if slug else digest


def hear_instruction(text: str, work_dir: Path, transcriber: Transcriber, rng) -> dict:
    """Run one instruction through the full TTS -> ASR chain.

    Synthesizes `text` with a randomly picked voice into work_dir, transcribes
    it back, and returns the record the eval harness logs per episode.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    voice = pick_voice(rng)
    wav_path = work_dir / f"{_slugify(text)}.wav"
    synthesize(text, wav_path, voice=voice)
    transcript = transcriber.transcribe(wav_path)
    return {
        "instruction": text,
        "voice": voice,
        "wav": str(wav_path),
        "transcript": transcript,
        "wer": wer(text, transcript),
    }
