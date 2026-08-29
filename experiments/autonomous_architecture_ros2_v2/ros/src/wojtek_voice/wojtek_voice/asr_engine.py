"""Whisper recognition with the hallucination guards from the #131 stack.

Whisper hallucinates on silence and noise — a phantom "6V" once reached the
agent as a command and preempted a running goal.  The decoder already knows
when it is guessing: drop segments it marks as probable non-speech or decodes
with low confidence.  Thresholds carried over from
training/wojtek_rl/agent/asr_server.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_NO_SPEECH_PROB = 0.6
MIN_AVG_LOGPROB = -1.0
# Whisper's own third guard (gzip ratio of the text): a repetition loop is
# CONFIDENT -- measured on silence, the "że to jest w tym," loop scored
# avg_logprob -0.25 while real speech scored -0.42 -- so only redundancy
# gives it away. 2.4 is the openai-whisper default threshold.
MAX_COMPRESSION_RATIO = 2.4
WHISPER_RATE = 16000


def compression_ratio(text: str) -> float:
    """len(utf-8)/len(gzip): >2.4 means the decoder is looping, not hearing."""
    import zlib

    data = text.encode("utf-8")
    if not data:
        return 0.0
    return len(data) / len(zlib.compress(data))


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
        ratio = getattr(s, "compression_ratio", None)
        if ratio is None:
            ratio = compression_ratio(s.text)
        if (no_speech > MAX_NO_SPEECH_PROB or logprob < MIN_AVG_LOGPROB
                or ratio > MAX_COMPRESSION_RATIO):
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


@dataclass
class _Segment:
    """Adapter: one transformers decode presented in faster-whisper's segment
    shape, so BOTH engines pass through the same filter_segments guards --
    the phantom-"6V" incident must stay fixed regardless of backend."""

    text: str
    no_speech_prob: float
    avg_logprob: float


def pick_backend(requested: str = "auto", machine: str | None = None) -> str:
    """Which whisper backend to use.

    faster-whisper (ctranslate2) is the default everywhere it can use the
    GPU -- but its aarch64 wheel is CPU-only (measured on a DGX Spark
    2026-08-21: "not compiled with CUDA support"), which would put large-v3
    on 20 ARM cores in the critical path of every spoken turn. On aarch64
    the transformers+CUDA path measured RTF 0.084 for large-v3, so it wins
    there and is selected automatically.
    """
    if requested != "auto":
        return requested
    import platform

    return "transformers" if (machine or platform.machine()) == "aarch64" else "faster-whisper"


class TransformersWhisperEngine:
    """Whisper via transformers+CUDA: the aarch64 path.

    Same transcribe() contract and the same guards as WhisperEngine. The
    guard inputs are recomputed from the decoder's own scores: avg_logprob
    as the mean log-probability of the tokens it chose, and no_speech_prob
    from the <|nospeech|> token at the first decode position -- the same
    quantities faster-whisper reports, so the thresholds carry over.
    """

    def __init__(self, model_size: str = "large-v3", device: str = "auto",
                 compute_type: str = "default", language: str = "pl"):
        self.model_size = model_size
        self.device = device
        self.language = language
        self._model = None
        self._processor = None
        self._no_speech_id = None

    def _repo(self) -> str:
        if "/" in self.model_size:
            return self.model_size
        return f"openai/whisper-{self.model_size}"

    def _load(self):
        if self._model is None:
            import torch
            from transformers import WhisperForConditionalGeneration, WhisperProcessor

            device = self.device
            if device in ("auto", "default"):
                device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            self._processor = WhisperProcessor.from_pretrained(self._repo())
            self._model = WhisperForConditionalGeneration.from_pretrained(
                self._repo(), torch_dtype=dtype
            ).to(device)
            self._model.eval()
            tok = self._processor.tokenizer
            # The token is named <|nospeech|> in large-v3 and <|nocaptions|>
            # in the earlier checkpoints (tiny/base/...); asking for the
            # wrong one returns the unk id and the probe silently reads a
            # meaningless probability (caught by the silence test).
            unk = tok.unk_token_id
            self._no_speech_id = None
            for name in ("<|nospeech|>", "<|nocaptions|>"):
                tid = tok.convert_tokens_to_ids(name)
                if tid is not None and tid != unk:
                    self._no_speech_id = tid
                    break
            # Warm the decoder so the first real utterance does not pay
            # graph-build time (same policy as the faster-whisper engine).
            self._decode(np.zeros(WHISPER_RATE, np.float32))
        return self._model

    def _decode(self, audio: np.ndarray) -> _Segment:
        import torch

        model, proc = self._model, self._processor
        device, dtype = model.device, model.dtype
        feats = proc(audio, sampling_rate=WHISPER_RATE, return_tensors="pt")
        feats = feats.input_features.to(device, dtype)
        with torch.inference_mode():
            # Encode once, use twice (the no-speech probe and the decode).
            enc = model.get_encoder()(feats)
            # P(<|nospeech|>) at the <|startoftranscript|> position -- the
            # definition whisper and faster-whisper use. Reading it out of
            # generate()'s scores is NOT equivalent: the forced language/task
            # tokens occupy the first score steps there, and measuring at the
            # wrong position let a silence hallucination straight through the
            # guards (caught by the silence test, 2026-08-21).
            no_speech = 0.0
            if self._no_speech_id is not None and self._no_speech_id >= 0:
                sot = proc.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
                dec = torch.tensor([[sot]], device=device)
                logits = model(encoder_outputs=enc, decoder_input_ids=dec).logits[0, -1]
                no_speech = float(torch.softmax(logits.float(), dim=-1)[self._no_speech_id])
            out = model.generate(
                encoder_outputs=enc, language=self.language, task="transcribe",
                output_scores=True, return_dict_in_generate=True,
            )
        text = proc.batch_decode(out.sequences, skip_special_tokens=True)[0]
        # Mean log-probability of the chosen tokens: whisper's avg_logprob.
        logprobs = []
        gen = out.sequences[0][-len(out.scores):]
        for step, tok_id in zip(out.scores, gen):
            logprobs.append(
                torch.log_softmax(step[0].float(), dim=-1)[tok_id].item()
            )
        avg_logprob = float(np.mean(logprobs)) if logprobs else 0.0
        return _Segment(text=text.strip(), no_speech_prob=no_speech,
                        avg_logprob=avg_logprob)

    def transcribe(self, pcm: np.ndarray, sample_rate: int) -> Recognition:
        audio = pcm_to_whisper_input(pcm, sample_rate)
        self._load()
        return filter_segments([self._decode(audio)])


def build_asr_engine(backend: str = "auto", **kwargs):
    """One constructor for the node: backend auto|faster-whisper|transformers."""
    resolved = pick_backend(backend)
    if resolved == "transformers":
        return TransformersWhisperEngine(**kwargs)
    return WhisperEngine(**kwargs)
