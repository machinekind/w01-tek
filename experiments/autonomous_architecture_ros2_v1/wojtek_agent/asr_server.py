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

    python -m wojtek_agent.asr_server --port 8110 --model large-v3

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

app = None  # created in main(); keeps the import light for the client side


def build_app(model_size, device, compute_type, language):
    api = FastAPI(title="Wojtek ASR")
    state: dict = {}

    def model():
        if "m" not in state:
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
            if no_speech > MAX_NO_SPEECH_PROB or logprob < MIN_AVG_LOGPROB:
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
    args = p.parse_args(argv)

    import uvicorn

    global app
    app = build_app(args.model, args.device, args.compute, args.language)
    logger.info(f"serving ASR on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
