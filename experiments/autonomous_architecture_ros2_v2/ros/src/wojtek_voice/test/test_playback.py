"""SpeechSynthesizer queue/cancel behavior with fake engines (no rclpy)."""

import threading
import time

import numpy as np

from wojtek_voice.playback import SpeechSynthesizer
from wojtek_voice.tts_engines import SilentTts, build_engine


def drain(synth, timeout=2.0):
    t0 = time.time()
    while not synth.idle and time.time() - t0 < timeout:
        time.sleep(0.01)
    time.sleep(0.05)  # let the last frames emit


class TestSpeechSynthesizer:
    def test_sentences_become_frames_in_order(self):
        frames = []
        synth = SpeechSynthesizer(SilentTts(), frames.append)
        synth.pace = False  # tests assert on totals, not on wall-clock pacing
        synth.start()
        synth.say("jedno zdanie")
        synth.say("drugie zdanie")
        drain(synth)
        assert len(frames) >= 2
        assert all(f.dtype == np.int16 for f in frames)

    def test_cancel_stops_current_and_drops_queue(self):
        emitted = []
        gate = threading.Event()

        class SlowEngine:
            def synth_stream(self, text):
                for _ in range(50):
                    gate.set()
                    time.sleep(0.01)
                    yield (np.zeros(2400, np.int16), 24000)

        synth = SpeechSynthesizer(SlowEngine(), emitted.append)
        synth.pace = False  # tests assert on totals, not on wall-clock pacing
        synth.start()
        synth.say("długie zdanie do przerwania")
        synth.say("nigdy nie zagra")
        gate.wait(1.0)
        synth.cancel()
        n = len(emitted)
        time.sleep(0.2)
        assert len(emitted) <= n + 2  # stops within a chunk or two
        assert synth.idle

    def test_engine_error_reported_not_fatal(self):
        errors = []

        class BrokenEngine:
            def synth_stream(self, text):
                raise RuntimeError("no GPU")
                yield  # pragma: no cover

        synth = SpeechSynthesizer(
            BrokenEngine(), lambda f: None, on_error=lambda t, e: errors.append(t)
        )
        synth.pace = False  # tests assert on totals, not on wall-clock pacing
        synth.start()
        synth.say("pierwsze")
        synth.say("drugie")
        drain(synth)
        assert errors == ["pierwsze", "drugie"]

    def test_resamples_engine_rate_to_output(self):
        frames = []

        class Rate16k:
            def synth_stream(self, text):
                yield (np.zeros(16000, np.int16), 16000)  # 1 s

        synth = SpeechSynthesizer(Rate16k(), frames.append, out_rate=24000)
        synth.pace = False  # tests assert on totals, not on wall-clock pacing
        synth.start()
        synth.say("sekunda ciszy")
        drain(synth)
        assert sum(len(f) for f in frames) == 24000


def test_build_engine_silent_and_unknown():
    assert isinstance(build_engine("silent"), SilentTts)
    try:
        build_engine("piper")
        raise AssertionError("should have raised")
    except ValueError:
        pass
