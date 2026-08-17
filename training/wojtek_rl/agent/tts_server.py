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

import argparse
import io
import os
import threading
import time
import wave
from collections import OrderedDict

import numpy as np

SAMPLE_RATE = 24000
# Distinct lines kept as ready-made audio. Small: the repeated lines are a
# handful of fixed phrases, and everything else is said once.
CACHE_SIZE = 64


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


# Streaming granularity. Small enough that the first piece is quick, big
# enough that the voice keeps a natural contour: a piece cut every few words
# sounds like a robot reading a list.
STREAM_MIN_CHARS = 30
# The first piece may be much shorter: it is the only one the listener waits
# through, and the pieces behind it are generated while it plays.
STREAM_FIRST_MIN_CHARS = 12
_PIECE_END = (". ", "! ", "? ", "… ", ", ", "; ", ": ")


def split_for_stream(text: str, min_chars: int = STREAM_MIN_CHARS,
                     first_min_chars: int = STREAM_FIRST_MIN_CHARS) -> list[str]:
    """Cut a reply at clause boundaries into pieces of at least min_chars.

    Deliberately duplicated rather than imported: this file is scp'd to the
    box on its own (the asr_server.py pattern), so it may not import the
    package.
    """
    text = " ".join((text or "").split())
    if not text:
        return []
    pieces: list[str] = []
    start = 0
    i = 0
    while i < len(text) - 1:
        floor = first_min_chars if not pieces else min_chars
        if text[i : i + 2] in _PIECE_END and (i + 1 - start) >= floor:
            pieces.append(text[start : i + 1].strip())
            start = i + 2
            i += 2
            continue
        i += 1
    rest = text[start:].strip()
    if rest:
        # A short tail joins the previous piece rather than becoming a
        # clipped-sounding fragment of its own.
        if pieces and len(rest) < min_chars:
            pieces[-1] = f"{pieces[-1]} {rest}"
        else:
            pieces.append(rest)
    return pieces


def pcm_to_wav_bytes(pcm: np.ndarray, rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def build_app(ref_wav: str, language: str, device: str, cache_size: int = CACHE_SIZE,
              model_factory=None):
    """`model_factory` overrides how the voice is loaded (tests inject a stub;
    the server itself loads Chatterbox lazily on first use)."""
    from fastapi import FastAPI, Request
    from fastapi.responses import Response

    app = FastAPI()
    lock = threading.Lock()  # one GPU synth at a time
    state: dict = {"model": None}
    # Measured on an A6000: Chatterbox runs at RTF 1.3-1.6, i.e. it generates
    # slower than the speech plays, and even a four-character line costs
    # ~2.8 s. The robot's most common lines are fixed strings (goal
    # acknowledgements, "Już się zatrzymuję!", the found-object phrase), so
    # saying one twice should cost the GPU once.
    cache: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def cached(text: str):
        pcm = cache.get(text)
        if pcm is not None:
            cache.move_to_end(text)
        return pcm

    def remember(text: str, pcm) -> None:
        if cache_size <= 0:
            return
        cache[text] = pcm
        cache.move_to_end(text)
        while len(cache) > cache_size:
            cache.popitem(last=False)

    def model():
        if state["model"] is None and model_factory is not None:
            state["model"] = model_factory()
        if state["model"] is None:
            # resemble-perth's implicit watermarker resolves to None when its
            # optional deps are broken (observed on a fresh vast box) and
            # chatterbox then crashes constructing it.  The demo does not
            # need watermarked audio -- fall back to the dummy.
            import perth

            if getattr(perth, "PerthImplicitWatermarker", None) is None:
                from perth.dummy_watermarker import DummyWatermarker

                perth.PerthImplicitWatermarker = DummyWatermarker
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            state["model"] = ChatterboxMultilingualTTS.from_pretrained(device=device)
        return state["model"]

    def warmup():
        """Load the model and synthesise one short line before serving.

        Without it the first /synthesize of a session also pays the model
        load and CUDA's first kernel compile -- which lands on the first
        thing the robot says to an audience.
        """
        t0 = time.monotonic()
        kwargs = {"language_id": language}
        if ref_wav:
            kwargs["audio_prompt_path"] = ref_wav
        try:
            model().generate("Hau hau.", **kwargs)
            print(f"tts warm in {time.monotonic() - t0:.1f}s", flush=True)
        except Exception as e:  # a failed warmup must not stop the server
            print(f"tts warmup failed ({e}); first reply will be slow", flush=True)

    app.state.warmup = warmup

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
        # Timing headers: the caller cannot otherwise tell a slow voice from
        # a voice queued behind the previous sentence, and the two need
        # opposite fixes (smaller model vs more workers). x-audio-ms lets the
        # client compute the real-time factor without decoding the WAV.
        t_in = time.monotonic()
        hit = cached(text)
        if hit is not None:
            pcm, gpu_ms, queue_ms = hit, 0.0, 0.0
        else:
            with lock:
                t_gpu = time.monotonic()
                m = model()
                pcm_f = m.generate(text, **kwargs).detach().cpu().numpy().squeeze()
                gpu_ms = (time.monotonic() - t_gpu) * 1000.0
            queue_ms = (t_gpu - t_in) * 1000.0
            pcm = np.clip(pcm_f * 32767.0, -32768, 32767).astype(np.int16)
            pcm = resample_linear(pcm, m.sr, SAMPLE_RATE)
            remember(text, pcm)
        headers = {
            "x-synth-ms": f"{gpu_ms:.1f}",
            "x-queue-ms": f"{queue_ms:.1f}",
            "x-audio-ms": f"{len(pcm) / SAMPLE_RATE * 1000.0:.1f}",
            "x-cache": "hit" if hit is not None else "miss",
        }
        return Response(pcm_to_wav_bytes(pcm), media_type="audio/wav", headers=headers)

    @app.post("/synthesize_stream")
    async def synthesize_stream(request: Request):
        """Raw PCM16 @ 24 kHz, flushed clause by clause.

        Chatterbox multilingual has no token-level streaming API for Polish
        (the streaming fork is English-only), so the streaming happens at the
        text level: the reply is cut into clause-sized pieces and each one
        ships the moment it is generated. The caller starts talking after the
        FIRST piece instead of after the whole answer -- on a ~1.0 RTF voice
        that is the difference between half a second and several.
        """
        from fastapi.responses import StreamingResponse

        payload = await request.json()
        text = (payload.get("text") or "").strip()
        pieces = split_for_stream(text)
        kwargs = {"language_id": language}
        if ref_wav:
            kwargs["audio_prompt_path"] = ref_wav

        def gen():
            for piece in pieces:
                hit = cached(piece)
                if hit is not None:
                    yield hit.tobytes()
                    continue
                with lock:
                    m = model()
                    pcm_f = m.generate(piece, **kwargs).detach().cpu().numpy().squeeze()
                pcm = np.clip(pcm_f * 32767.0, -32768, 32767).astype(np.int16)
                pcm = resample_linear(pcm, m.sr, SAMPLE_RATE)
                remember(piece, pcm)
                yield pcm.tobytes()

        return StreamingResponse(gen(), media_type="application/octet-stream")

    return app


def main():
    import uvicorn

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=int(os.environ.get("TTS_PORT", "8120")))
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--ref", default=os.environ.get("TTS_REF_WAV", ""))
    p.add_argument("--language", default=os.environ.get("TTS_LANGUAGE", "pl"))
    p.add_argument("--device", default=os.environ.get("TTS_DEVICE", "cuda"))
    p.add_argument("--cache", type=int, default=int(os.environ.get("TTS_CACHE", CACHE_SIZE)),
                   help="lines kept as ready-made audio (0 disables)")
    p.add_argument(
        "--no-warmup",
        action="store_true",
        help="skip the startup load+synth; the FIRST reply then pays for "
        "loading the model and for CUDA's first kernel compile, which is "
        "seconds of silence in front of a live audience",
    )
    args = p.parse_args()
    app = build_app(args.ref, args.language, args.device, cache_size=args.cache)
    if not args.no_warmup:
        app.router.on_startup.append(app.state.warmup)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
