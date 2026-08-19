"""Record a demo session: the room, what the dog sees, and what it says.

Connects to the room demo as an ordinary websocket viewer -- the same client
the browser is -- and writes an MP4 with the chase camera, the ego view, the
self-built map and a caption strip showing the conversation and the state
machine, with the dog's synthesised speech on the audio track.

Recording from the websocket rather than a screen grab means the result is
reproducible on a headless box, needs no compositor, and can run while the
scenario script drives the robot.

    python -m wojtek_agent.record --out demo.mp4 --seconds 120

Audio is the PCM the server streams for playback; video frames are whatever
arrived most recently, sampled at a fixed rate so the two stay in step.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from loguru import logger

from wojtek_agent.voice import SAMPLE_RATE

FPS = 10
WIDTH, HEIGHT = 1280, 720
CAPTION_H = 200


def _font(size: int):
    from PIL import ImageFont

    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


class Recorder:
    """Composites incoming frames; holds the latest of each panel."""

    def __init__(self, width=WIDTH, height=HEIGHT):
        self.width, self.height = width, height
        self.chase = self.ego = self.map = None
        self.lines: list[tuple[str, str]] = []   # (speaker, text)
        self.status = ""
        self.state = ""
        self._big = _font(20)
        self._small = _font(15)
        self._mono = _font(14)

    def add_line(self, who: str, text: str) -> None:
        self.lines.append((who, text))
        del self.lines[:-4]

    @staticmethod
    def _decode(b64: str):
        from PIL import Image

        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

    def set_frames(self, chase=None, ego=None, map_=None) -> None:
        if chase:
            self.chase = self._decode(chase)
        if ego:
            self.ego = self._decode(ego)
        if map_:
            self.map = self._decode(map_)

    def compose(self) -> np.ndarray:
        from PIL import Image, ImageDraw

        canvas = Image.new("RGB", (self.width, self.height), (20, 22, 28))
        draw = ImageDraw.Draw(canvas)
        video_h = self.height - CAPTION_H
        main_w = int(self.width * 0.62)

        def paste(img, box):
            x, y, w, h = box
            draw.rectangle([x, y, x + w, y + h], fill=(0, 0, 0))
            if img is None:
                return
            scale = min(w / img.width, h / img.height)
            im = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
            canvas.paste(im, (x + (w - im.width) // 2, y + (h - im.height) // 2))

        paste(self.chase, (8, 8, main_w - 16, video_h - 16))
        side_x, side_w = main_w, self.width - main_w - 8
        half = (video_h - 24) // 2
        paste(self.ego, (side_x, 8, side_w, half))
        paste(self.map, (side_x, 16 + half, side_w, half))

        for x, y, label in (
            (14, 12, "CHASE CAM"),
            (side_x + 6, 12, "WHAT THE DOG SEES"),
            (side_x + 6, 20 + half, "SELF-BUILT MAP"),
        ):
            draw.text((x, y), label, font=self._small, fill=(210, 210, 200))

        # caption strip: conversation + live state
        top = video_h
        draw.rectangle([0, top, self.width, self.height], fill=(28, 31, 39))
        draw.line([0, top, self.width, top], fill=(60, 64, 76))
        y = top + 10
        for who, text in self.lines[-4:]:
            colour = (95, 196, 207) if who == "you" else (232, 230, 223)
            prefix = "YOU:  " if who == "you" else ("WOJTEK: " if who == "dog" else "")
            if who == "sys":
                colour = (154, 160, 173)
            draw.text((16, y), _clip(prefix + text, 96), font=self._big, fill=colour)
            y += 30
        draw.text((16, self.height - 46), _clip(self.state, 120), font=self._mono, fill=(232, 163, 61))
        draw.text((16, self.height - 26), _clip(self.status, 120), font=self._mono, fill=(154, 160, 173))
        return np.asarray(canvas)


def _clip(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def build_audio_track(speech: list[tuple[float, np.ndarray]], video_s: float) -> np.ndarray:
    """Lay speech frames onto a silent track of the video's length.

    Arrival time alone is NOT playback time: the server pushes a whole
    reply's frames as fast as the socket takes them, so a three-second
    sentence can arrive inside a few milliseconds. Writing each frame at its
    arrival offset stacks them on top of each other and the result is noise.

    So frames are laid CONSECUTIVELY from where the previous one ended --
    what the browser does by scheduling on the audio clock -- and only a
    genuine gap (a new utterance, arriving after the previous one would have
    finished) moves the write head forward to match the picture.
    """
    audio = np.zeros(int(video_s * SAMPLE_RATE) + SAMPLE_RATE, np.int16)
    write_pos = 0
    for at, pcm in speech:
        start = max(int(at * SAMPLE_RATE), write_pos)
        end = min(start + len(pcm), len(audio))
        if end > start:
            audio[start:end] = pcm[: end - start]
        write_pos = end
    return audio


async def record(url: str, out: Path, seconds: float, on_ready=None) -> Path:
    """Record `seconds` of the session at `url` into `out` (MP4)."""
    import imageio_ffmpeg
    import websockets

    rec = Recorder()
    # (arrival time, samples): the dog is silent most of the session, so the
    # track has to be laid out on the wall clock. Concatenating the speech
    # instead would front-load every reply and desync it from the picture.
    speech: list[tuple[float, np.ndarray]] = []
    frames: list[np.ndarray] = []
    t0 = time.monotonic()

    async def pump(ws):
        async for msg in ws:
            if isinstance(msg, bytes):
                speech.append((time.monotonic() - t0, np.frombuffer(msg, np.int16)))
                continue
            m = json.loads(msg)
            t = m.get("type")
            if t == "state":
                rec.set_frames(m.get("frame"), m.get("ego"), m.get("map"))
                ex = m.get("exec") or {}
                rec.status = f"mode {m.get('mode')} | {ex.get('active') or 'idle'} | plan {ex.get('plan', '-')}"
            elif t == "heard":
                rec.add_line("you", m["text"])
            elif t == "chat_reply" and m.get("ok"):
                rec.add_line("dog", m.get("say", ""))
            elif t == "agent_status" and m.get("goal"):
                detail = m.get("detail") or {}
                note = detail.get("note") or detail.get("reason") or ""
                rec.state = f"{m['kind']} '{m['goal']}' → {m['state']} {note}"
            elif t == "vlm_status" and m.get("last"):
                rec.state = (
                    f"navigate '{m.get('goal')}' step {m.get('step')} → "
                    f"{m['last'].get('action')} {m['last'].get('amount') or ''}"
                )

    async def grab():
        period = 1.0 / FPS
        while time.monotonic() - t0 < seconds:
            frames.append(rec.compose())
            await asyncio.sleep(period)

    async with websockets.connect(url, max_size=None) as ws:
        pump_task = asyncio.create_task(pump(ws))
        if on_ready is not None:
            asyncio.create_task(on_ready(ws))
        await grab()
        pump_task.cancel()

    spoken_s = sum(len(pcm) for _, pcm in speech) / SAMPLE_RATE
    video_s = len(frames) / FPS
    logger.info(f"composing {len(frames)} frames ({video_s:.1f}s), {spoken_s:.1f}s of speech")

    audio = build_audio_track(speech, video_s)
    out.parent.mkdir(parents=True, exist_ok=True)
    silent = out.with_suffix(".video.mp4")
    writer = imageio_ffmpeg.write_frames(
        str(silent), (WIDTH, HEIGHT), fps=FPS, quality=7, macro_block_size=8
    )
    writer.send(None)
    for f in frames:
        writer.send(np.ascontiguousarray(f))
    writer.close()

    if not speech:
        silent.replace(out)
        return out

    import wave

    wav = out.with_suffix(".wav")
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(audio.tobytes())
    # No -shortest: the tracks are the same length by construction now, and
    # -shortest would silently clip the video to whatever the encoder made of
    # the audio.
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(silent), "-i", str(wav),
         "-c:v", "copy", "-c:a", "aac", str(out)],
        check=True, capture_output=True,
    )
    silent.unlink(missing_ok=True)
    wav.unlink(missing_ok=True)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="ws://127.0.0.1:8010/ws")
    p.add_argument("--out", type=Path, default=Path("demo.mp4"))
    p.add_argument("--seconds", type=float, default=60.0)
    args = p.parse_args(argv)
    out = asyncio.run(record(args.url, args.out, args.seconds))
    logger.success(f"wrote {out}")


if __name__ == "__main__":
    main()
