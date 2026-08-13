"""Scripted voice session against the room demo: ask prerecorded Polish
questions over the websocket, capture the dog's replies, and write ONE
audio track carrying BOTH voices — the asker's wavs at the moment they were
sent and the dog's speech as it streamed back.

Why this exists: room_app's Speaker pushes reply PCM only to the websocket
connection that asked, so wojtek_rl.agent.record (a second connection)
films silent video.  Run both together and mux:

  python -m wojtek_rl.agent.record   --out clip.mp4 --seconds 120 &
  python -m wojtek_rl.agent.scenario --script sc.json --audio clip.audio.wav
  ffmpeg -i clip.mp4 -i clip.audio.wav -map 0:v -map 1:a -c:v copy clip_voiced.mp4

Script format (wavs must be mono PCM16 at 24 kHz):
  [{"wav": "questions/q01.wav", "wait": 15}, {"pause": 5}, ...]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import wave
from pathlib import Path

import numpy as np

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


async def run(url: str, script: list[dict], audio_out: Path | None) -> None:
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
        await ws.send(json.dumps({"type": "voice", "on": True}))
        await asyncio.sleep(1)
        for step in script:
            if "pause" in step:
                await asyncio.sleep(step["pause"])
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
                   help="write the combined both-voices track here")
    args = p.parse_args(argv)
    script = json.loads(args.script.read_text())
    asyncio.run(run(args.url, script, args.audio))


if __name__ == "__main__":
    main()
