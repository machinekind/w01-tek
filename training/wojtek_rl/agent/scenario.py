"""Scripted voice session against the room demo — drive, record and VALIDATE
on ONE websocket connection.

Why one connection is load-bearing: room_app is single-viewer — every new
/ws connection supersedes the previous one's streaming loop.  Running
record.py and a driver as two clients silently freezes whichever connected
first (live finding 2026-08-14: three "walking" clips showed one repeated
frame while the real, moving sim streamed to the connection nobody
filmed).  This module therefore does everything on one socket: sends the
prerecorded questions (or midlevel commands), receives frames/captions/
speech, composes the video, mixes BOTH voices into the audio track, and
checks success criteria before calling the take good.

Script format (wavs mono PCM16 @ 24 kHz):
  [{"wav": "questions/q01.wav", "wait": 15},
   {"cmd": "forward 1.0", "wait": 8},          # midlevel command step
   {"pause": 5}]

Usage:
  python -m wojtek_rl.agent.scenario --script sc.json --video clip.mp4 \
      --require-move 0.8          # exit 2 unless the base moved >= 0.8 m

Validation criteria printed (and enforced with --require-*):
  moved_m       max base displacement from the first pose (from state stream)
  speech_s      seconds of dog speech received
  frame_delta   mean |pixel| difference between early and late video frames —
                a frozen recording scores ~0 and fails loudly
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
import wave
from pathlib import Path

import numpy as np
from loguru import logger

from wojtek_rl.agent.voice import SAMPLE_RATE

FRAME = SAMPLE_RATE // 10  # 100 ms, the browser worklet frame


def read_wav(path) -> np.ndarray:
    with wave.open(str(path)) as w:
        if w.getframerate() != SAMPLE_RATE:
            raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz, got {w.getframerate()}")
        pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16)
        if w.getnchannels() > 1:
            pcm = pcm.reshape(-1, w.getnchannels()).mean(axis=1).astype(np.int16)
    return pcm


def lay_track(events: list[tuple[str, float, np.ndarray]], total_s: float,
              rate: int = SAMPLE_RATE) -> np.ndarray:
    """Both voices on one wall-clock track.

    Questions ("q") paste at their send time — the asker spoke then.
    Reply frames ("a") pack back-to-back from their arrival time, because
    the server pushes a whole utterance faster than realtime and the
    browser would have scheduled the frames sequentially on its audio
    clock (same reasoning as record.build_audio_track).
    """
    audio = np.zeros(int(total_s * rate) + rate, np.int16)
    head = 0.0  # reply playback head
    for kind, at, pcm in events:
        if kind == "a":
            at = max(at, head)
            head = at + len(pcm) / rate
        start = int(at * rate)
        end = min(start + len(pcm), len(audio))
        if start < len(audio):
            audio[start:end] = pcm[: end - start]
    return audio


async def _drive(ws, script: list[dict], events, t0: float) -> None:
    """Send the scripted questions/commands; record question timestamps."""
    await ws.send(json.dumps({"type": "voice", "on": True}))
    await asyncio.sleep(1)
    for step in script:
        if "pause" in step:
            await asyncio.sleep(step["pause"])
            continue
        if "cmd" in step:
            print(f"=== COMMAND: {step['cmd']}", flush=True)
            await ws.send(json.dumps({"type": "command", "text": step["cmd"]}))
            await asyncio.sleep(step.get("wait", 8))
            continue
        pcm = read_wav(step["wav"])
        print(f"=== ASKING: {Path(step['wav']).stem}", flush=True)
        events.append(("q", time.monotonic() - t0, pcm))
        for i in range(0, len(pcm), FRAME):
            await ws.send(np.ascontiguousarray(pcm[i : i + FRAME]).tobytes())
            await asyncio.sleep(0.1)
        for _ in range(10):  # trailing silence closes the utterance
            await ws.send(np.zeros(FRAME, np.int16).tobytes())
            await asyncio.sleep(0.1)
        await asyncio.sleep(step.get("wait", 15))


async def run_video(url: str, script: list[dict], out: Path,
                    tail_s: float = 8.0) -> dict:
    """One-connection drive + record.  Returns the metrics dict."""
    import imageio_ffmpeg
    import websockets

    from wojtek_rl.agent.record import FPS, HEIGHT, WIDTH, Recorder

    rec = Recorder()
    events: list[tuple[str, float, np.ndarray]] = []
    poses: list[tuple[float, float, float]] = []
    frames: list[np.ndarray] = []

    async with websockets.connect(url, max_size=None) as ws:
        t0 = time.monotonic()

        async def pump():
            async for msg in ws:
                if isinstance(msg, bytes):
                    events.append(("a", time.monotonic() - t0, np.frombuffer(msg, np.int16)))
                    continue
                m = json.loads(msg)
                t = m.get("type")
                if t == "state":
                    rec.set_frames(m.get("frame"), m.get("ego"), m.get("map"))
                    ex = m.get("exec") or {}
                    rec.status = (
                        f"mode {m.get('mode')} | {ex.get('active') or 'idle'} | "
                        f"plan {ex.get('plan', '-')}"
                    )
                    if m.get("x") is not None:
                        poses.append((time.monotonic() - t0, m["x"], m["y"]))
                elif t == "heard":
                    rec.add_line("you", m["text"])
                    print(f"[heard] {m['text']}", flush=True)
                elif t == "chat_reply" and m.get("ok"):
                    rec.add_line("dog", m.get("say", ""))
                    print(f"[dog] {m.get('say', '')[:120]}", flush=True)
                elif t == "agent_status" and m.get("goal"):
                    detail = m.get("detail") or {}
                    note = detail.get("note") or detail.get("reason") or ""
                    rec.state = f"{m['kind']} '{m['goal']}' → {m['state']} {note}"
                elif t == "vlm_status" and m.get("last"):
                    rec.state = (
                        f"navigate '{m.get('goal')}' step {m.get('step')} → "
                        f"{m['last'].get('action')} {m['last'].get('amount') or ''}"
                    )

        async def grab(stop: asyncio.Event):
            period = 1.0 / FPS
            while not stop.is_set():
                frames.append(rec.compose())
                await asyncio.sleep(period)

        stop = asyncio.Event()
        pump_task = asyncio.create_task(pump())
        grab_task = asyncio.create_task(grab(stop))
        await _drive(ws, script, events, t0)
        await asyncio.sleep(tail_s)  # let the last reply finish speaking
        stop.set()
        await grab_task
        pump_task.cancel()

    video_s = len(frames) / FPS
    audio = lay_track(events, video_s)
    speech_s = sum(len(p) for k, _, p in events if k == "a") / SAMPLE_RATE

    out.parent.mkdir(parents=True, exist_ok=True)
    silent = out.with_suffix(".video.mp4")
    writer = imageio_ffmpeg.write_frames(
        str(silent), (WIDTH, HEIGHT), fps=FPS, quality=7, macro_block_size=8
    )
    writer.send(None)
    for f in frames:
        writer.send(np.ascontiguousarray(f))
    writer.close()
    wav = out.with_suffix(".mix.wav")
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(audio[: int(video_s * SAMPLE_RATE)].tobytes())
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(silent), "-i", str(wav),
         "-c:v", "copy", "-c:a", "aac", str(out)],
        check=True, capture_output=True,
    )
    silent.unlink(missing_ok=True)
    wav.unlink(missing_ok=True)

    # -- success criteria ---------------------------------------------------
    moved = 0.0
    if poses:
        x0, y0 = poses[0][1], poses[0][2]
        moved = max(((x - x0) ** 2 + (y - y0) ** 2) ** 0.5 for _, x, y in poses)
    frame_delta = 0.0
    if len(frames) > 20:
        a = frames[len(frames) // 10].astype(np.float32)
        b = frames[-len(frames) // 10].astype(np.float32)
        frame_delta = float(np.abs(a - b).mean())
    metrics = {
        "video": str(out),
        "video_s": round(video_s, 1),
        "moved_m": round(moved, 2),
        "speech_s": round(speech_s, 1),
        "frame_delta": round(frame_delta, 2),
        "poses": len(poses),
    }
    print("METRICS " + json.dumps(metrics), flush=True)
    return metrics


async def run_headless(url: str, script: list[dict], audio_out: Path | None) -> None:
    """Original audio-only mode (no video composition)."""
    import websockets

    events: list[tuple[str, float, np.ndarray]] = []
    async with websockets.connect(url, max_size=2**23) as ws:
        t0 = time.monotonic()

        async def pump(stop: asyncio.Event):
            while not stop.is_set():
                try:
                    m = await asyncio.wait_for(ws.recv(), 0.25)
                except (asyncio.TimeoutError, TimeoutError):
                    continue
                except websockets.ConnectionClosed:
                    return
                if isinstance(m, bytes):
                    events.append(("a", time.monotonic() - t0, np.frombuffer(m, np.int16)))
                    continue
                ev = json.loads(m)
                if ev.get("type") not in ("state", "trace"):
                    print(f"[{ev.get('type')}]",
                          json.dumps(ev, ensure_ascii=False)[:220], flush=True)

        stop = asyncio.Event()
        pumper = asyncio.create_task(pump(stop))
        await _drive(ws, script, events, t0)
        stop.set()
        await pumper

    if audio_out:
        total = max((at + len(p) / SAMPLE_RATE for _, at, p in events), default=1.0)
        audio = lay_track(events, total)
        with wave.open(str(audio_out), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(audio.tobytes())
        spoken = sum(len(p) for k, _, p in events if k == "a") / SAMPLE_RATE
        print(f"audio track {total:.1f}s ({spoken:.1f}s of dog speech) -> {audio_out}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="ws://127.0.0.1:8010/ws")
    p.add_argument("--script", type=Path, required=True)
    p.add_argument("--audio", type=Path, default=None,
                   help="headless mode: write the both-voices track here")
    p.add_argument("--video", type=Path, default=None,
                   help="record a validated clip (single-connection drive+film)")
    p.add_argument("--require-move", type=float, default=None,
                   help="fail (exit 2) unless the base moved at least this many meters")
    p.add_argument("--require-speech", type=float, default=None,
                   help="fail (exit 2) unless at least this many seconds were spoken")
    p.add_argument("--tail", type=float, default=8.0)
    args = p.parse_args(argv)
    script = json.loads(args.script.read_text())

    if args.video:
        metrics = asyncio.run(run_video(args.url, script, args.video, tail_s=args.tail))
        failed = []
        if metrics["frame_delta"] < 0.5:
            failed.append(f"frozen video (frame_delta {metrics['frame_delta']})")
        if args.require_move is not None and metrics["moved_m"] < args.require_move:
            failed.append(f"moved {metrics['moved_m']} m < required {args.require_move} m")
        if args.require_speech is not None and metrics["speech_s"] < args.require_speech:
            failed.append(f"spoke {metrics['speech_s']} s < required {args.require_speech} s")
        if failed:
            logger.error("TAKE FAILED: " + "; ".join(failed))
            raise SystemExit(2)
        logger.success("TAKE OK: " + json.dumps(metrics))
    else:
        asyncio.run(run_headless(args.url, script, args.audio))


if __name__ == "__main__":
    main()
