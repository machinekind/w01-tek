"""Chatterbox multilingual TTS as a tiny HTTP service for the GPU box.

The room demo's voice runs wherever the browser is, but a cloned Polish
voice (Chatterbox multilingual + reference wav) needs a GPU.  This server
is the TTS twin of asr_server.py: POST /synthesize {"text": ...} -> WAV,
GET /health.  The demo points at it with TTS_ENGINE=remote TTS_URL=...

The voice reference comes from the environment, never from the repository
(cloned character voices are private-tinkering assets):

  TTS_REF_WAV=/path/ref.wav TTS_LANGUAGE=pl TTS_PORT=8120 \
    python -m wojtek_rl.agent.tts_server

Output is resampled to 24 kHz (the browser worklet rate), so RemoteTts
never needs to know the model's native rate.
"""

from __future__ import annotations

import argparse
import io
import os
import threading
import wave

import numpy as np

SAMPLE_RATE = 24000


def resample_linear(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Copy of wojtek_rl.agent.voice.resample_linear so this file deploys as
    ONE scp'd script (asr_server.py pattern) with no package on the box."""
    if src_rate == dst_rate or not len(pcm):
        return pcm
    n_out = int(round(len(pcm) * dst_rate / src_rate))
    if n_out <= 0:
        return np.zeros(0, np.int16)
    x = np.linspace(0.0, len(pcm) - 1, n_out, dtype=np.float64)
    out = np.interp(x, np.arange(len(pcm)), pcm.astype(np.float64))
    return np.clip(np.rint(out), -32768, 32767).astype(np.int16)


def pcm_to_wav_bytes(pcm: np.ndarray, rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def build_app(ref_wav: str, language: str, device: str):
    from fastapi import FastAPI, Request
    from fastapi.responses import Response

    app = FastAPI()
    lock = threading.Lock()  # one GPU synth at a time
    state: dict = {"model": None}

    def model():
        if state["model"] is None:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            state["model"] = ChatterboxMultilingualTTS.from_pretrained(device=device)
        return state["model"]

    @app.get("/health")
    def health():
        return {
            "ok": True,
            "engine": "chatterbox-mtl",
            "language": language,
            "ref": bool(ref_wav),
            "loaded": state["model"] is not None,
        }

    @app.post("/synthesize")
    async def synthesize(request: Request):
        payload = await request.json()
        text = (payload.get("text") or "").strip()
        if not text:
            return Response(pcm_to_wav_bytes(np.zeros(0, np.int16)), media_type="audio/wav")
        kwargs = {"language_id": language}
        if ref_wav:
            kwargs["audio_prompt_path"] = ref_wav
        with lock:
            m = model()
            pcm_f = m.generate(text, **kwargs).detach().cpu().numpy().squeeze()
        pcm = np.clip(pcm_f * 32767.0, -32768, 32767).astype(np.int16)
        pcm = resample_linear(pcm, m.sr, SAMPLE_RATE)
        return Response(pcm_to_wav_bytes(pcm), media_type="audio/wav")

    return app


def main():
    import uvicorn

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=int(os.environ.get("TTS_PORT", "8120")))
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--ref", default=os.environ.get("TTS_REF_WAV", ""))
    p.add_argument("--language", default=os.environ.get("TTS_LANGUAGE", "pl"))
    p.add_argument("--device", default=os.environ.get("TTS_DEVICE", "cuda"))
    args = p.parse_args()
    app = build_app(args.ref, args.language, args.device)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
