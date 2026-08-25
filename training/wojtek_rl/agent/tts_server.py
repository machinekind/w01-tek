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


# Tail cleanup. Chatterbox routinely generates junk after the speech ends --
# breath, clicks, a synthetic smear (the alignment analyzer's "long_tail"
# warnings are the same phenomenon seen from inside). Every reply in the
# 2026-08-22 takes ended with audible noise. Trim from the END: find the
# last 20 ms frame that still looks like speech, keep a short natural
# release after it, fade out. Conservative on purpose: never cut more than
# TRIM_MAX_S, never below MIN_KEEP_S, and a quiet-throughout clip is left
# alone (it is probably all "release").
TRIM_FRAME_S = 0.02
TRIM_KEEP_S = 0.15       # natural release kept after the last speech frame
TRIM_FADE_S = 0.03
TRIM_MAX_S = 1.5
MIN_KEEP_S = 0.2
TRIM_REL_THRESHOLD = 0.05  # frame RMS below this fraction of peak = not speech


# The v2 takes showed quiet-tail trimming is not enough: chatterbox's junk is
# often LOUD (a synthetic smear or foreign-language babble), so an RMS floor
# keeps it. Second pass: junk almost always sits AFTER the last long silence
# gap, and real final words do not -- Polish sentence-final words follow
# their sentence within ~0.3 s. So cut everything after a trailing gap of
# GAP_MIN_S when what follows is short; a long segment after a gap is treated
# as speech (a genuine dramatic pause) and kept.
GAP_MIN_S = 0.40
GAP_SEGMENT_MAX_S = 1.2


def _fade_out(pcm: np.ndarray, rate: int) -> np.ndarray:
    out = pcm.copy()
    fade = min(int(rate * TRIM_FADE_S), len(out))
    if fade > 1:
        out[-fade:] = (out[-fade:].astype(np.float32)
                       * np.linspace(1.0, 0.0, fade)).astype(np.int16)
    return out


def trim_tail_noise(pcm: np.ndarray, rate: int = SAMPLE_RATE) -> np.ndarray:
    if os.environ.get("TTS_TAIL_TRIM", "on") == "off":
        return pcm
    n = int(rate * TRIM_FRAME_S)
    if len(pcm) < 4 * n:
        return pcm
    frames = pcm[: len(pcm) - len(pcm) % n].reshape(-1, n).astype(np.float32)
    rms = np.sqrt((frames**2).mean(axis=1))
    peak = float(rms.max())
    if peak < 1.0:
        return pcm
    speech = rms >= peak * TRIM_REL_THRESHOLD
    if not speech.any():
        return pcm
    # Pass 1: drop the quiet tail after the last speech-like frame.
    last = int(np.nonzero(speech)[0][-1])
    end = min(len(pcm), (last + 1) * n + int(rate * TRIM_KEEP_S))
    end = max(end, int(rate * MIN_KEEP_S), len(pcm) - int(rate * TRIM_MAX_S))
    # Pass 2: loud junk after a trailing silence gap. Walk the speech mask up
    # to `end`, find the last gap of >= GAP_MIN_S; if the speech after it is
    # a short blob, cut at the gap instead.
    gap_frames = int(GAP_MIN_S / TRIM_FRAME_S)
    upto = min(len(speech), end // n)
    run = 0
    last_gap_end = None
    for i in range(upto):
        if speech[i]:
            if run >= gap_frames:
                last_gap_end = i          # first speech frame after a long gap
            run = 0
        else:
            run += 1
    if last_gap_end is not None:
        tail_s = (upto - last_gap_end) * TRIM_FRAME_S
        if tail_s <= GAP_SEGMENT_MAX_S:
            cut = max((last_gap_end - run) * n, int(rate * MIN_KEEP_S))
            gap_start = last_gap_end
            while gap_start > 0 and not speech[gap_start - 1]:
                gap_start -= 1
            cut = max(gap_start * n + int(rate * TRIM_KEEP_S), int(rate * MIN_KEEP_S))
            end = min(end, cut)
    return _fade_out(pcm[:end], rate)


def pcm_to_wav_bytes(pcm: np.ndarray, rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def build_app(ref_wav: str, language: str, device: str, cache_size: int = CACHE_SIZE,
              model_factory=None, tuning: dict | None = None,
              compile_backbone: bool = False):
    """`model_factory` overrides how the voice is loaded (tests inject a stub;
    the server itself loads Chatterbox lazily on first use). `tuning` carries
    generation knobs (exaggeration, cfg_weight, temperature) so a quality/
    speed A/B is a server flag rather than an edit."""
    from fastapi import FastAPI, Request
    from fastapi.responses import Response

    app = FastAPI()
    lock = threading.Lock()  # one GPU synth at a time
    state: dict = {"model": None, "conds_ready": False}
    tuning = dict(tuning or {})
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

    def compile_model(m):
        """Opt-in torch.compile of the AR backbone.

        Batch-1 autoregressive decode is dominated by kernel-launch overhead,
        which is exactly what reduce-overhead mode (CUDA graphs) removes. Kept
        opt-in and guarded: a compile failure must degrade to the eager model,
        never take the voice down. Costs a slow first call, so it pairs with
        the startup warmup.
        """
        target = getattr(getattr(m, "t3", None), "patched_model", None)
        if target is None:
            print("compile: no t3.patched_model to compile; skipping", flush=True)
            return m
        try:
            import torch

            m.t3.patched_model = torch.compile(target, mode="reduce-overhead")
            print("compile: t3 backbone compiled (reduce-overhead)", flush=True)
        except Exception as e:  # missing torch, unsupported backend, bad graph
            print(f"compile failed ({e}); running eager", flush=True)
        return m

    def model():
        if state["model"] is None and model_factory is not None:
            state["model"] = model_factory()
            if compile_backbone:
                state["model"] = compile_model(state["model"])
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

    def ensure_conditionals() -> float:
        """Encode the reference voice ONCE, not on every line.

        chatterbox's `generate(audio_prompt_path=...)` calls
        prepare_conditionals internally every single call: librosa load and
        resample of the reference, s3gen embed_ref, the S3 tokenizer and the
        voice encoder -- all of it repeated for a voice that never changes.
        Preparing it once and then omitting audio_prompt_path is
        bit-identical and cuts that work from every request.

        Returns the milliseconds spent (0.0 when already prepared).
        """
        if not ref_wav or state["conds_ready"]:
            return 0.0
        t0 = time.monotonic()
        model().prepare_conditionals(ref_wav)
        state["conds_ready"] = True
        return (time.monotonic() - t0) * 1000.0

    def cap_tokens(m, text: str) -> None:
        """Cap T3's speech-token budget from the TEXT length.

        The tail junk exists because nothing forces the LM to stop when the
        text runs out: it keeps sampling plausible speech tokens until the
        alignment watchdog force-stops it, and the junk is already rendered
        by then. Polish speech runs ~12-14 chars/s and speech tokens are
        25/s, so ~2.1 tokens/char plus a fixed margin bounds every legit
        utterance -- the model CANNOT ramble past it. chatterbox hardcodes
        max_new_tokens=1000 in generate(), so the cap is injected by wrapping
        t3.inference once and reading the per-call budget from state.
        """
        if os.environ.get("TTS_TOKEN_CAP", "on") == "off":
            state["token_budget"] = None
            return
        state["token_budget"] = min(1000, int(len(text) * 2.1) + 25)
        t3 = getattr(m, "t3", None)
        if t3 is None:          # engines without a T3 (tests, future backends)
            return
        if not state.get("t3_wrapped"):
            orig = t3.inference

            def inference(*a, **kw):
                budget = state.get("token_budget")
                if budget:
                    kw["max_new_tokens"] = min(kw.get("max_new_tokens", 1000), budget)
                return orig(*a, **kw)

            t3.inference = inference
            state["t3_wrapped"] = True

    def gen_kwargs() -> dict:
        """Generation arguments. audio_prompt_path is deliberately absent:
        the conditionals are already loaded (see ensure_conditionals)."""
        return dict(language_id=language, **tuning)

    def warmup():
        """Load the model, encode the voice, and synthesise one short line
        before serving.

        Without it the first /synthesize of a session also pays the model
        load and CUDA's first kernel compile -- which lands on the first
        thing the robot says to an audience.
        """
        t0 = time.monotonic()
        try:
            cond_ms = ensure_conditionals()
            model().generate("Hau hau.", **gen_kwargs())
            note = f" (voice encoded in {cond_ms / 1000:.1f}s)" if cond_ms else ""
            print(f"tts warm in {time.monotonic() - t0:.1f}s{note}", flush=True)
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
        # Timing headers: the caller cannot otherwise tell a slow voice from
        # a voice queued behind the previous sentence, and the two need
        # opposite fixes (smaller model vs more workers). x-audio-ms lets the
        # client compute the real-time factor without decoding the WAV.
        t_in = time.monotonic()
        hit = cached(text)
        cond_ms = 0.0
        if hit is not None:
            pcm, gpu_ms, queue_ms = hit, 0.0, 0.0
        else:
            with lock:
                t_gpu = time.monotonic()
                m = model()
                cond_ms = ensure_conditionals()
                cap_tokens(m, text)
                pcm_f = m.generate(text, **gen_kwargs()).detach().cpu().numpy().squeeze()
                gpu_ms = (time.monotonic() - t_gpu) * 1000.0
            queue_ms = (t_gpu - t_in) * 1000.0
            pcm = np.clip(pcm_f * 32767.0, -32768, 32767).astype(np.int16)
            pcm = trim_tail_noise(resample_linear(pcm, m.sr, SAMPLE_RATE))
            remember(text, pcm)
        headers = {
            "x-synth-ms": f"{gpu_ms:.1f}",
            "x-queue-ms": f"{queue_ms:.1f}",
            "x-audio-ms": f"{len(pcm) / SAMPLE_RATE * 1000.0:.1f}",
            "x-cache": "hit" if hit is not None else "miss",
            # Non-zero only on the request that encoded the voice; if this is
            # ever non-zero twice, the conditional cache regressed.
            "x-cond-ms": f"{cond_ms:.1f}",
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
        # TTS_STREAM_SPLIT=off: one piece per reply. Piece-level streaming
        # only sounds continuous when piece N+1 is generated before piece N
        # finishes playing; on a contended GPU it is not, and the seams are
        # worse than one longer initial wait (user verdict on the takes).
        if os.environ.get("TTS_STREAM_SPLIT", "on") == "off":
            pieces = [text] if text else []
        else:
            pieces = split_for_stream(text)

        def gen():
            for piece in pieces:
                hit = cached(piece)
                if hit is not None:
                    yield hit.tobytes()
                    continue
                with lock:
                    m = model()
                    ensure_conditionals()
                    cap_tokens(m, piece)
                    pcm_f = m.generate(piece, **gen_kwargs()).detach().cpu().numpy().squeeze()
                pcm = np.clip(pcm_f * 32767.0, -32768, 32767).astype(np.int16)
                pcm = trim_tail_noise(resample_linear(pcm, m.sr, SAMPLE_RATE))
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
    p.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the AR backbone (reduce-overhead/CUDA graphs). "
        "Safe -- same weights, same sampling -- but the first call pays "
        "compilation, so keep the warmup on. Measure before trusting it.",
    )
    p.add_argument(
        "--progress",
        action="store_true",
        help="keep chatterbox's per-token tqdm bar (off by default: it "
        "prints ~24x/s during generation for no benefit on a server)",
    )
    # Generation knobs, so a speed/quality A/B is a flag rather than an edit.
    # NOTE: cfg_weight only changes how the two logit streams are combined --
    # chatterbox duplicates the batch unconditionally, so setting it to 0
    # does NOT halve the compute (verified in chatterbox 0.1.7 source).
    p.add_argument("--exaggeration", type=float, default=None)
    p.add_argument("--cfg-weight", type=float, default=None)
    p.add_argument("--temperature", type=float, default=None)
    args = p.parse_args()
    if not args.progress:
        # tqdm honours this; set before chatterbox is imported.
        os.environ.setdefault("TQDM_DISABLE", "1")
    tuning = {k: v for k, v in (("exaggeration", args.exaggeration),
                                ("cfg_weight", args.cfg_weight),
                                ("temperature", args.temperature)) if v is not None}
    app = build_app(args.ref, args.language, args.device, cache_size=args.cache,
                    tuning=tuning, compile_backbone=args.compile)
    if not args.no_warmup:
        app.router.on_startup.append(app.state.warmup)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
