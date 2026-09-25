"""Speech recognition as a service, so it can live on the GPU.

Whisper large-v3 is the best open Polish recogniser, but on a laptop CPU it
runs at ~2x realtime -- measured 5.6 s for a 2.8 s utterance, which puts five
seconds of dead air into every exchange. The same model on a rented RTX 5880
Ada does it in **0.36 s (0.13x realtime)**, so recognition stops being the
bottleneck entirely.

On the real robot this service is unnecessary: the Jetson is both the mic
host and the GPU, so `Transcriber` runs in-process against the local card.
It exists for the split setup -- browser and sim on a laptop, models on a
rented box.

    python -m wojtek_rl.agent.asr_server --port 8110 --model large-v3

POST /transcribe with a WAV body -> {"ok": true, "text": ...}. Deliberately
the same shape as the demo's own /api/hear, and the same one-method contract
`VoiceListener` needs, so the client is a drop-in for the local Transcriber.
"""

import argparse
import io
import os
import time

from fastapi import FastAPI, Request
from loguru import logger

# NOTE: no `from __future__ import annotations` here, and fastapi is imported
# at module level on purpose. With postponed annotations the `request:
# Request` hint is a STRING that FastAPI resolves against the module
# namespace; with the import hidden inside the factory it cannot resolve it,
# decides `request` must be a query parameter, and every POST comes back 422
# "Field required". Cost me a recording to find.

DEFAULT_ASR_PORT = 8110
# Hallucination gate. Whisper happily emits confident-looking nonsense for
# silence; these are the decoder's own signals for "this was not speech" and
# "I am guessing". Values are the commonly used faster-whisper defaults.
MAX_NO_SPEECH_PROB = 0.6
MIN_AVG_LOGPROB = -1.0
# Third gate: a silence hallucination LOOPS confidently (measured: the
# "że to jest w tym," loop scored a BETTER logprob than real speech), so only
# its redundancy gives it away. gzip ratio > 2.4 is whisper's own default.
MAX_COMPRESSION_RATIO = 2.4


def compression_ratio(text: str) -> float:
    import zlib

    data = text.encode("utf-8")
    return len(data) / len(zlib.compress(data)) if data else 0.0

app = None  # created in main(); keeps the import light for the client side


class _TransformersWhisper:
    """faster-whisper's transcribe() shape on top of transformers+CUDA.

    Returns (segments, info) where the one segment carries text,
    no_speech_prob (probed at <|startoftranscript|>, trying both token
    names -- large-v3 says <|nospeech|>, older checkpoints <|nocaptions|>)
    and avg_logprob, so the guard code above needs no branches.
    """

    def __init__(self, model_size, device, language):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        repo = model_size if "/" in model_size else f"openai/whisper-{model_size}"
        if device in ("auto", "default"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        logger.info(f"loading transformers whisper {repo} on {device}")
        self.proc = WhisperProcessor.from_pretrained(repo)
        self.m = WhisperForConditionalGeneration.from_pretrained(
            repo, torch_dtype=dtype).to(device)
        self.m.eval()
        tok = self.proc.tokenizer
        self.no_speech_id = None
        for name in ("<|nospeech|>", "<|nocaptions|>"):
            tid = tok.convert_tokens_to_ids(name)
            if tid is not None and tid != tok.unk_token_id:
                self.no_speech_id = tid
                break
        self.language = language

    def transcribe(self, audio, language=None, beam_size=1, vad_filter=False):
        import numpy as np
        import torch

        if hasattr(audio, "read"):  # WAV bytes, like faster-whisper accepts
            import soundfile as sf

            data, rate = sf.read(audio, dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            if rate != 16000:
                x = np.linspace(0.0, len(data) - 1, int(len(data) * 16000 / rate))
                data = np.interp(x, np.arange(len(data)), data).astype("float32")
            audio = data
        duration = len(audio) / 16000.0
        feats = self.proc(audio, sampling_rate=16000, return_tensors="pt")
        feats = feats.input_features.to(self.m.device, self.m.dtype)
        with torch.inference_mode():
            enc = self.m.get_encoder()(feats)
            no_speech = 0.0
            if self.no_speech_id is not None:
                sot = self.proc.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
                dec = torch.tensor([[sot]], device=self.m.device)
                logits = self.m(encoder_outputs=enc, decoder_input_ids=dec).logits[0, -1]
                no_speech = float(torch.softmax(logits.float(), dim=-1)[self.no_speech_id])
            out = self.m.generate(
                encoder_outputs=enc, language=language or self.language,
                task="transcribe", output_scores=True, return_dict_in_generate=True,
            )
        text = self.proc.batch_decode(out.sequences, skip_special_tokens=True)[0]
        logprobs = [
            torch.log_softmax(step[0].float(), dim=-1)[tok_id].item()
            for step, tok_id in zip(out.scores, out.sequences[0][-len(out.scores):])
        ]

        class Seg:
            pass

        seg = Seg()
        seg.text = text.strip()
        seg.no_speech_prob = no_speech
        seg.avg_logprob = float(np.mean(logprobs)) if logprobs else 0.0

        class Info:
            pass

        info = Info()
        info.duration = duration
        return [seg], info


def build_app(model_size, device, compute_type, language, backend="faster-whisper"):
    api = FastAPI(title="Wojtek ASR")
    state: dict = {}

    def model():
        if "m" not in state:
            if backend == "transformers":
                # The aarch64 ctranslate2 wheel has no CUDA (measured on a
                # DGX Spark: large-v3 would run on 20 ARM cores). The
                # transformers+CUDA path does RTF 0.084 there, so ARM boxes
                # use it. Standalone copy of the wojtek_voice engine, since
                # this file is scp'd to boxes on its own.
                state["m"] = _TransformersWhisper(model_size, device, language)
            else:
                from faster_whisper import WhisperModel

                logger.info(f"loading faster-whisper {model_size} on {device} ({compute_type})")
                state["m"] = WhisperModel(model_size, device=device, compute_type=compute_type)
            logger.success("ASR ready")
        return state["m"]

    @api.on_event("startup")
    def _warm():
        # Load at startup, not on the first utterance: a 3 GB lazy load in the
        # middle of a conversation looks exactly like the robot ignoring you.
        import numpy as np

        m = model()
        silence = np.zeros(16000, "float32")
        list(m.transcribe(silence, language=language, beam_size=1)[0])
        logger.success("ASR warmed")

    @api.get("/health")
    def health():
        return {"ok": True, "model": model_size, "device": device, "language": language}

    @api.post("/transcribe")
    async def transcribe(request: Request):
        raw = await request.body()
        if len(raw) < 200:
            return {"ok": False, "error": "no audio received"}
        t0 = time.monotonic()
        segments, info = model().transcribe(
            io.BytesIO(raw), language=language, beam_size=2, vad_filter=False
        )
        # Whisper invents text from silence and room tone -- a live session
        # produced a phantom "6V" that reached the agent as a real command and
        # preempted a running goal. The decoder already knows: drop segments it
        # thinks are non-speech or is unsure about.
        kept, dropped = [], []
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
        if dropped:
            logger.info(f"dropped low-confidence segments: {dropped}")
        text = " ".join(kept).strip()
        dt = time.monotonic() - t0
        logger.info(f"transcribed {info.duration:.1f}s in {dt:.2f}s ({dt / max(info.duration, 1e-3):.2f}x RT): {text!r}")
        return {"ok": True, "text": text, "seconds": info.duration, "latency_s": round(dt, 3)}

    return api


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=int(os.environ.get("ASR_PORT", DEFAULT_ASR_PORT)))
    p.add_argument("--model", default=os.environ.get("ASR_MODEL", "large-v3"))
    p.add_argument("--device", default=os.environ.get("ASR_DEVICE", "cuda"))
    p.add_argument("--compute", default=os.environ.get("ASR_COMPUTE", "float16"))
    p.add_argument("--language", default=os.environ.get("ASR_LANGUAGE", "pl"))
    p.add_argument("--backend", default=os.environ.get("ASR_BACKEND", "auto"),
                   choices=("auto", "faster-whisper", "transformers"),
                   help="auto picks transformers on aarch64, where "
                        "ctranslate2 cannot use the GPU")
    args = p.parse_args(argv)
    if args.backend == "auto":
        import platform

        args.backend = "transformers" if platform.machine() == "aarch64" else "faster-whisper"

    import uvicorn

    global app
    app = build_app(args.model, args.device, args.compute, args.language,
                    backend=args.backend)
    logger.info(f"serving ASR on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
