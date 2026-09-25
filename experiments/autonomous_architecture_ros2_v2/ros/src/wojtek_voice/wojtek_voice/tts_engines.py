"""TTS engines behind one protocol (design Decision 3: both, pick by ear).

synth_stream(text) yields (pcm_int16, sample_rate) chunks — a natively
streaming engine yields many small chunks, a whole-utterance engine yields
one.  All engines lazy-load on first use; construction is free so the node
can start before the GPU is warm.

Voice references (wav + transcript) come from parameters/env, NEVER from
the repository — cloned character voices (Głuś) are private-tinkering
assets and this repo is public.
"""

from __future__ import annotations

import os
from typing import Iterator, Protocol

import numpy as np

Chunk = tuple[np.ndarray, int]


class TtsEngine(Protocol):
    def synth_stream(self, text: str) -> Iterator[Chunk]: ...


class SilentTts:
    """Silence proportional to text length — keeps the pipeline alive in
    tests and when no engine is installed."""

    rate = 24000

    def synth_stream(self, text: str) -> Iterator[Chunk]:
        n = int(self.rate * 0.05 * max(1, len(text.split())))
        yield (np.zeros(n, np.int16), self.rate)


def _to_int16(wav: np.ndarray) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).squeeze()
    return np.clip(wav * 32767.0, -32768, 32767).astype(np.int16)


class ChatterboxEngine:
    """Chatterbox multilingual (MIT).  Uses the streaming fork's API when
    installed (`chatterbox-streaming`, generate_stream), otherwise falls
    back to whole-sentence generate — sentence pipelining still keeps the
    first-audio latency bounded.

    Refs must be denoised: Chatterbox faithfully copies reference room tone
    (measured 2026-08, the "przebitki" trap).
    """

    def __init__(self, ref_wav: str = "", language: str = "pl", device: str = "cuda"):
        self.ref_wav = ref_wav or os.environ.get("WOJTEK_TTS_REF_WAV", "")
        self.language = language
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            self._model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
        return self._model

    def synth_stream(self, text: str) -> Iterator[Chunk]:
        model = self._load()
        kwargs = {"language_id": self.language}
        if self.ref_wav:
            kwargs["audio_prompt_path"] = self.ref_wav
        if hasattr(model, "generate_stream"):
            for chunk, _metrics in model.generate_stream(text, **kwargs):
                yield (_to_int16(chunk.detach().cpu().numpy()), model.sr)
        else:
            wav = model.generate(text, **kwargs)
            yield (_to_int16(wav.detach().cpu().numpy()), model.sr)


class RemoteEngine:
    """Thin client for wojtek_rl's tts_server.py over HTTP.

    The server owns the model AND every optimization the demo shipped --
    voice conditionals encoded once, the token cap, tail-noise trimming,
    clause streaming -- so this engine inherits them instead of porting
    them. It is also the process-isolation move: chatterbox pins an old
    transformers, the ASR node wants a new one, and on the GB10 the proven
    layout is chatterbox alone in its own venv behind this exact server.

    Streaming: /synthesize_stream flushes raw PCM16 clause by clause; a 404
    (older server) falls back to the one-shot WAV endpoint. int16 frames are
    reassembled across chunk boundaries, same as the demo client.
    """

    RATE = 24000  # the server resamples to the pipeline rate

    def __init__(self, url: str = "", **_ignored):
        self.url = (url or os.environ.get("TTS_URL",
                                          "http://127.0.0.1:8120")).rstrip("/")
        self.timeout_s = 60.0

    def _one_shot(self, text: str) -> Chunk:
        import io
        import wave

        import httpx

        r = httpx.post(f"{self.url}/synthesize", json={"text": text},
                       timeout=self.timeout_s)
        r.raise_for_status()
        with wave.open(io.BytesIO(r.content)) as w:
            rate = w.getframerate()
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        return (pcm, rate)

    def synth_stream(self, text: str) -> Iterator[Chunk]:
        import httpx

        try:
            with httpx.stream("POST", f"{self.url}/synthesize_stream",
                              json={"text": text},
                              timeout=self.timeout_s) as r:
                if r.status_code == 404:
                    r.read()
                    yield self._one_shot(text)
                    return
                r.raise_for_status()
                tail = b""
                for raw in r.iter_bytes():
                    if not raw:
                        continue
                    buf = tail + raw
                    usable = len(buf) - (len(buf) % 2)
                    tail = buf[usable:]
                    if usable:
                        yield (np.frombuffer(buf[:usable], dtype=np.int16),
                               self.RATE)
        except httpx.HTTPError:
            yield self._one_shot(text)


class F5ForkEngine:
    """F5-TTS Gregniuki Polish fork (CC-BY-NC — non-commercial contexts
    only).  The pip `f5-tts` package is WRONG for this checkpoint (char
    tokenization garbles Polish); the fork pipeline is espeak-ng IPA →
    BPE tokenizer → DiT sample.  Ported from the validated scratchpad
    script (f5_fork_tts.py, 2026-08).

    Needs the fork checkout + assets, all by env/params:
      F5_SPACE_DIR   fork source dir (adds to sys.path; has model/, infer/)
      F5_CKPT        .pt checkpoint (e.g. f5_multi3_900k.pt)
      F5_TOKENIZER   tokenizer.json (BPE)
      ref_wav/ref_text  cloning reference and its EXACT transcript —
                        a poisoned transcript produces garbage.
    """

    RATE = 24000
    HOP = 256
    TARGET_RMS = 0.1
    NFE = 32

    def __init__(self, ref_wav: str = "", ref_text: str = "",
                 language: str = "pl", device: str = "cuda"):
        self.space_dir = os.environ.get("F5_SPACE_DIR", "")
        self.ckpt = os.environ.get("F5_CKPT", "")
        self.tokenizer_path = os.environ.get("F5_TOKENIZER", "")
        self.ref_wav = ref_wav or os.environ.get("WOJTEK_TTS_REF_WAV", "")
        self.ref_text = ref_text or os.environ.get("WOJTEK_TTS_REF_TEXT", "")
        self.language = language
        self.device = device
        self._loaded = None

    def _load(self):
        if self._loaded is None:
            if not (self.space_dir and self.ckpt and self.tokenizer_path):
                raise RuntimeError(
                    "F5 fork engine needs F5_SPACE_DIR, F5_CKPT and F5_TOKENIZER"
                )
            if not (self.ref_wav and self.ref_text):
                raise RuntimeError(
                    "F5 needs a reference: WOJTEK_TTS_REF_WAV + WOJTEK_TTS_REF_TEXT"
                )
            import sys

            import torch
            import torchaudio
            from tokenizers import Tokenizer

            sys.path.insert(0, self.space_dir)
            from model import DiT  # fork module
            from infer.utils_infer import load_model, load_vocoder  # fork module

            cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2,
                       text_dim=512, conv_layers=4)
            model = load_model(DiT, cfg, self.ckpt, vocab_file="", device=self.device)
            vocoder = load_vocoder()
            model = model.to(torch.float32)
            vocoder = vocoder.to(torch.float32)
            tokenizer = Tokenizer.from_file(self.tokenizer_path)

            audio, sr = torchaudio.load(self.ref_wav)
            if audio.shape[0] > 1:
                audio = audio.mean(0, keepdim=True)
            rms = torch.sqrt(torch.mean(audio**2))
            if rms < self.TARGET_RMS:
                audio = audio * self.TARGET_RMS / rms
            if sr != self.RATE:
                audio = torchaudio.transforms.Resample(sr, self.RATE)(audio)
            audio = audio[:, : 10 * self.RATE]  # zero-shot ignores refs past ~10 s

            ref_text = self.ref_text
            if not ref_text.endswith(". "):
                ref_text += " " if ref_text.endswith(".") else ". "
            self._loaded = (model, vocoder, tokenizer, audio, float(rms), ref_text)
        return self._loaded

    def _ipa(self, text: str) -> str:
        import re

        from phonemizer import phonemize

        out = phonemize(text, language=self.language, backend="espeak",
                        strip=False, preserve_punctuation=True, with_stress=True)
        return re.sub(r"\([a-z]{2,3}\)", "", out)

    def synth_stream(self, text: str) -> Iterator[Chunk]:
        import torch
        from einops import rearrange

        model, vocoder, tokenizer, ref_audio, ref_rms, ref_text = self._load()
        tokens = tokenizer.encode(self._ipa(ref_text) + self._ipa(text)).tokens
        text_list = [" ".join(map(str, tokens))]
        ref_len = ref_audio.shape[-1] // self.HOP
        dur = max(250, ref_len + int(
            ref_len / len(ref_text.encode()) * len(text.encode())
        ))
        with torch.inference_mode():
            gen, _ = model.sample(
                cond=ref_audio.to(self.device), text=text_list, duration=dur,
                steps=self.NFE, cfg_strength=2.0, sway_sampling_coef=-1.0,
            )
        mel = rearrange(gen[:, ref_len:, :].to(torch.float32), "1 n d -> 1 d n")
        wave = vocoder.float().decode(mel).squeeze().cpu().numpy()
        if ref_rms < self.TARGET_RMS:
            wave = wave * ref_rms / self.TARGET_RMS
        yield (_to_int16(wave), self.RATE)


def build_engine(kind: str, **kwargs) -> TtsEngine:
    if kind == "chatterbox":
        return ChatterboxEngine(**kwargs)
    if kind == "f5":
        return F5ForkEngine(**kwargs)
    if kind == "remote":
        return RemoteEngine(**kwargs)
    if kind == "silent":
        return SilentTts()
    raise ValueError(f"unknown TTS engine {kind!r} (chatterbox|f5|remote|silent)")
