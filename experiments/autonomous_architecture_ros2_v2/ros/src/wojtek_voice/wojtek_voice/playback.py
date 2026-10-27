"""Sentence queue + synthesis worker, rclpy-free so it is testable.

Sentences arrive in order per reply; the worker synthesizes one at a time
and emits browser-rate PCM frames through a callback.  cancel() implements
barge-in: it drops everything queued and stops the current sentence at the
next chunk boundary — measured 90 ms to silence in #131, same contract.
"""

from __future__ import annotations

import queue
import threading
import time

import numpy as np

from .transport import FRAME_MS, SAMPLE_RATE, pcm_frames, resample_linear


class SpeechSynthesizer:
    def __init__(self, engine, emit_frame, out_rate: int = SAMPLE_RATE,
                 frame_ms: int = FRAME_MS, on_error=None, pace: bool = True):
        self.engine = engine
        self.emit_frame = emit_frame            # callable(np.int16 frame)
        self.out_rate = out_rate
        self.frame_ms = frame_ms
        self.on_error = on_error or (lambda text, exc: None)
        self.pace = pace
        self._q: queue.Queue = queue.Queue()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        # Which reply the frames coming out right now belong to.  Written by
        # the worker thread that calls emit_frame, read inside that same
        # callback, so the published audio can carry the utterance id it
        # answers -- without which no one can measure how long the human
        # waited between speaking and hearing.
        self.current_tag = None

    def start(self):
        self._thread = threading.Thread(target=self._work, daemon=True)
        self._thread.start()

    def say(self, text: str, tag=None):
        text = (text or "").strip()
        if text:
            self._q.put((text, tag))

    def cancel(self):
        """Barge-in: drop the queue, stop mid-sentence at next chunk."""
        self._cancel.set()
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass

    @property
    def idle(self) -> bool:
        return self._q.empty()

    # Frames the player may hold ahead of real time. A cached utterance
    # arrives in one burst; unpaced, its frames pile onto the previous
    # reply's and the recorded mix garbles into skipped syllables and
    # mid-word cuts (heard on camera, 2026-08-27). Three frames of lead
    # keep latency low while the rest plays out at wall-clock speed.
    LEAD_FRAMES = 3

    def _work(self):
        while True:
            text, tag = self._q.get()
            self.current_tag = tag
            self._cancel.clear()
            frame_s = self.frame_ms / 1000.0
            t0, sent = time.monotonic(), 0
            try:
                for pcm, rate in self.engine.synth_stream(text):
                    if self._cancel.is_set():
                        break
                    out = resample_linear(np.asarray(pcm, np.int16), rate, self.out_rate)
                    for frame in pcm_frames(out, self.frame_ms, self.out_rate):
                        if self._cancel.is_set():
                            break
                        if self.pace:
                            ahead = (t0 + (sent - self.LEAD_FRAMES) * frame_s
                                     - time.monotonic())
                            if ahead > 0:
                                time.sleep(ahead)
                        self.emit_frame(frame)
                        sent += 1
            except Exception as e:  # an engine hiccup must not kill the voice
                self.on_error(text, e)
