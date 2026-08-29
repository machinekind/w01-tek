"""Segmenter and UtteranceStream behavior, model-free (no rclpy needed)."""

import numpy as np
import pytest

from wojtek_voice.transport import (
    SAMPLE_RATE,
    UtteranceStream,
    VoiceSegmenter,
    pcm_frames,
    resample_linear,
)

FRAME = int(SAMPLE_RATE * 0.1)  # 100 ms


def loud(n=FRAME):
    rng = np.random.default_rng(0)
    return (rng.normal(0, 3000, n)).astype(np.int16)


def quiet(n=FRAME):
    return np.zeros(n, np.int16)


def feed_all(stream, frames):
    events = []
    for f in frames:
        events.extend(stream.feed(f))
    return events


class TestVoiceSegmenter:
    def test_speech_then_silence_closes_utterance(self):
        seg = VoiceSegmenter()
        utt = None
        for _ in range(10):
            utt = seg.feed(loud()) or utt
        for _ in range(8):
            utt = seg.feed(quiet()) or utt
        assert utt is not None
        assert utt.ended_on == "silence"
        assert utt.seconds > 1.0

    def test_short_blip_discarded(self):
        seg = VoiceSegmenter()
        results = [seg.feed(loud())]  # one 100 ms frame < MIN_UTTERANCE_S
        for _ in range(8):
            results.append(seg.feed(quiet()))
        assert all(r is None for r in results)

    def test_preroll_included(self):
        seg = VoiceSegmenter()
        for _ in range(5):
            seg.feed(quiet())
        for _ in range(10):
            seg.feed(loud())
        utt = None
        for _ in range(8):
            utt = seg.feed(quiet()) or utt
        # 10 loud frames (1.0 s) + preroll (0.3 s) + trailing silence (0.7 s)
        assert utt.seconds >= 1.9

    def test_flush_returns_open_utterance(self):
        seg = VoiceSegmenter()
        for _ in range(10):
            seg.feed(loud())
        utt = seg.flush()
        assert utt is not None and utt.ended_on == "flush"
        assert seg.flush() is None  # idempotent once closed


class TestUtteranceStream:
    def test_event_sequence(self):
        stream = UtteranceStream()
        events = feed_all(stream, [quiet()] * 3 + [loud()] * 10 + [quiet()] * 8)
        kinds = [e[0] for e in events]
        assert kinds[0] == "started"
        assert kinds[-1] == "ended"
        assert set(kinds[1:-1]) == {"frame"}

    def test_ended_event_keeps_utterance_id(self):
        stream = UtteranceStream()
        events = feed_all(stream, [loud()] * 10 + [quiet()] * 8)
        started = next(e for e in events if e[0] == "started")
        ended = next(e for e in events if e[0] == "ended")
        assert started[1] == ended[1] != None  # noqa: E711
        assert stream.utterance_id is None  # reset after close

    def test_frames_cover_full_utterance(self):
        stream = UtteranceStream()
        events = feed_all(stream, [quiet()] * 3 + [loud()] * 10 + [quiet()] * 8)
        streamed = np.concatenate([e[2] for e in events if e[0] == "frame"])
        final = next(e[2] for e in events if e[0] == "ended").pcm
        # Live frames re-emit the pre-roll first, so a streaming consumer and
        # a final-only consumer hear the same onset; the recap additionally
        # carries the trailing-silence frames a live consumer never needs.
        assert len(streamed) >= 10 * FRAME
        assert np.array_equal(final[: len(streamed)], streamed)

    def test_short_blip_aborts(self):
        stream = UtteranceStream()
        events = feed_all(stream, [loud()] + [quiet()] * 8)
        kinds = [e[0] for e in events]
        assert "started" in kinds and "aborted" in kinds
        assert "ended" not in kinds

    def test_distinct_ids_per_utterance(self):
        stream = UtteranceStream()
        first = feed_all(stream, [loud()] * 10 + [quiet()] * 8)
        second = feed_all(stream, [loud()] * 10 + [quiet()] * 8)
        id1 = next(e[1] for e in first if e[0] == "ended")
        id2 = next(e[1] for e in second if e[0] == "ended")
        assert id1 != id2

    def test_flush_ends_open_utterance(self):
        stream = UtteranceStream()
        feed_all(stream, [loud()] * 10)
        events = list(stream.flush())
        assert events[0][0] == "ended"
        assert events[0][2].ended_on == "flush"


class TestResample:
    def test_rate_conversion_length(self):
        pcm = loud(2400)
        out = resample_linear(pcm, 24000, 16000)
        assert len(out) == 1600

    def test_identity_when_rates_match(self):
        pcm = loud(100)
        assert resample_linear(pcm, 16000, 16000) is pcm

    @pytest.mark.parametrize("n", [0, 1, 7])
    def test_tiny_inputs_survive(self, n):
        out = resample_linear(loud(max(n, 1))[:n], 24000, 16000)
        assert out.dtype == np.int16


def test_pcm_frames_covers_everything():
    pcm = loud(FRAME * 3 + 17)
    frames = list(pcm_frames(pcm))
    assert sum(len(f) for f in frames) == len(pcm)
    assert all(len(f) <= FRAME for f in frames)
