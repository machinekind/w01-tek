"""Voice segmentation and PCM helpers -- no audio hardware, no model."""

import asyncio

import numpy as np
import pytest

from wojtek_agent.voice import (
    SAMPLE_RATE,
    Utterance,
    VoiceListener,
    VoiceSegmenter,
    frame_rms,
    pcm_frames,
    resample_linear,
)

FRAME = SAMPLE_RATE // 10  # 100 ms


def speech(n_frames=1, amp=0.3):
    rng = np.random.default_rng(0)
    x = rng.normal(0, amp, FRAME * n_frames)
    return np.clip(x * 32767, -32768, 32767).astype(np.int16)


def silence(n_frames=1):
    return np.zeros(FRAME * n_frames, np.int16)


def feed_all(seg, pcm, chunk=FRAME):
    out = []
    for i in range(0, len(pcm), chunk):
        u = seg.feed(pcm[i : i + chunk])
        if u is not None:
            out.append(u)
    return out


# -- RMS / helpers ------------------------------------------------------------


def test_frame_rms_ranges():
    assert frame_rms(np.zeros(10, np.int16)) == 0.0
    assert frame_rms(np.full(10, 32767, np.int16)) == pytest.approx(1.0, abs=1e-3)
    assert frame_rms(np.zeros(0, np.int16)) == 0.0


def test_pcm_frames_slices_evenly():
    frames = list(pcm_frames(np.zeros(SAMPLE_RATE, np.int16), frame_ms=100))
    assert len(frames) == 10
    assert all(len(f) == FRAME for f in frames)


def test_resample_linear_changes_length_and_keeps_dtype():
    pcm = np.linspace(-1000, 1000, 480, dtype=np.int16)
    out = resample_linear(pcm, 24000, 16000)
    assert out.dtype == np.int16
    assert len(out) == 320
    assert resample_linear(pcm, 24000, 24000) is pcm


def test_utterance_wav_roundtrip():
    import io
    import wave

    utt = Utterance(pcm=speech(2), seconds=0.2, ended_on="silence")
    with wave.open(io.BytesIO(utt.to_wav_bytes())) as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == SAMPLE_RATE
        assert w.getnframes() == len(utt.pcm)


# -- segmentation --------------------------------------------------------------


def test_silence_alone_never_yields_an_utterance():
    seg = VoiceSegmenter()
    assert feed_all(seg, silence(30)) == []
    assert not seg.speaking


def test_speech_then_silence_closes_one_utterance():
    seg = VoiceSegmenter()
    got = feed_all(seg, np.concatenate([silence(3), speech(10), silence(12)]))
    assert len(got) == 1
    assert got[0].ended_on == "silence"
    assert got[0].seconds >= 1.0


def test_short_blip_is_discarded():
    """A cough is not a sentence."""
    seg = VoiceSegmenter()
    got = feed_all(seg, np.concatenate([speech(1), silence(12)]))
    assert got == []


def test_preroll_keeps_audio_before_the_trigger():
    """Without pre-roll the recogniser never hears the first syllable."""
    seg = VoiceSegmenter()
    got = feed_all(seg, np.concatenate([silence(5), speech(10), silence(12)]))
    assert len(got) == 1
    # 1.0 s speech + up to 0.3 s pre-roll + 0.7 s trailing silence
    assert got[0].seconds > 1.7


def test_pause_inside_a_sentence_does_not_split_it():
    seg = VoiceSegmenter()
    body = np.concatenate([speech(6), silence(4), speech(6)])  # 0.4 s gap < 0.7 s
    got = feed_all(seg, np.concatenate([body, silence(12)]))
    assert len(got) == 1


def test_max_length_closes_a_runaway_utterance():
    seg = VoiceSegmenter(max_utterance_s=1.0)
    got = feed_all(seg, speech(30))
    assert got and got[0].ended_on == "max_length"


def test_flush_closes_an_open_utterance():
    seg = VoiceSegmenter()
    feed_all(seg, np.concatenate([speech(10)]))
    assert seg.speaking
    utt = seg.flush()
    assert utt is not None and utt.ended_on == "flush"
    assert seg.flush() is None


def test_reset_drops_state():
    seg = VoiceSegmenter()
    feed_all(seg, speech(10))
    seg.reset()
    assert not seg.speaking
    assert seg.flush() is None


def test_custom_vad_overrides_rms():
    """Injecting silero (or anything) must bypass the energy test."""
    seg = VoiceSegmenter()
    seg.vad = lambda pcm: True  # everything is speech
    got = feed_all(seg, silence(30))
    assert got == []  # never any silence -> never closes
    assert seg.speaking


# -- listener ------------------------------------------------------------------


class FakeTranscriber:
    def __init__(self, text="cześć Wojtek"):
        self.text = text
        self.calls = 0

    def transcribe(self, wav_path):
        self.calls += 1
        assert wav_path.exists(), "listener must hand the recogniser a real file"
        return self.text


def test_listener_transcribes_and_reports():
    heard = []

    async def on_text(text, utt):
        heard.append((text, round(utt.seconds, 1)))

    tr = FakeTranscriber()
    listener = VoiceListener(tr, on_text)
    listener.set_enabled(True)

    async def scenario():
        for pcm in (silence(3), speech(10), silence(12)):
            for i in range(0, len(pcm), FRAME):
                await listener.feed_frame(pcm[i : i + FRAME].tobytes())

    asyncio.run(scenario())
    assert tr.calls == 1
    assert heard and heard[0][0] == "cześć Wojtek"


def test_listener_ignores_audio_while_disabled():
    tr = FakeTranscriber()
    listener = VoiceListener(tr, lambda *a: None)

    async def scenario():
        for i in range(0, len(speech(10)), FRAME):
            await listener.feed_frame(speech(1).tobytes())

    asyncio.run(scenario())
    assert tr.calls == 0


def test_listener_survives_a_failing_recogniser():
    class Boom:
        def transcribe(self, path):
            raise RuntimeError("whisper exploded")

    called = []

    async def on_text(text, utt):
        called.append(text)

    listener = VoiceListener(Boom(), on_text)
    listener.set_enabled(True)

    async def scenario():
        for pcm in (speech(10), silence(12)):
            for i in range(0, len(pcm), FRAME):
                await listener.feed_frame(pcm[i : i + FRAME].tobytes())

    asyncio.run(scenario())  # must not raise
    assert called == []


def test_listener_drops_empty_transcripts():
    called = []

    async def on_text(text, utt):
        called.append(text)

    listener = VoiceListener(FakeTranscriber(text="   "), on_text)
    listener.set_enabled(True)

    async def scenario():
        for pcm in (speech(10), silence(12)):
            for i in range(0, len(pcm), FRAME):
                await listener.feed_frame(pcm[i : i + FRAME].tobytes())

    asyncio.run(scenario())
    assert called == []


# -- remote ASR ---------------------------------------------------------------


def test_remote_transcriber_posts_wav_and_returns_text(tmp_path, monkeypatch):
    """Whisper large-v3 is ~2x realtime on a laptop CPU and ~0.13x on a GPU,
    so recognition can move to the box the models are on. Same one method."""
    import wojtek_agent.voice as voice

    wav = tmp_path / "u.wav"
    wav.write_bytes(b"RIFF" + b"\0" * 300)
    seen = {}

    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"ok": True, "text": "cześć Wojtek"}

    def fake_post(url, content=None, timeout=None):
        seen["url"] = url
        seen["bytes"] = len(content)
        return FakeResp()

    monkeypatch.setitem(__import__("sys").modules, "httpx",
                        type("m", (), {"post": staticmethod(fake_post)}))
    t = voice.RemoteTranscriber("http://gpu-box:8110/")
    assert t.transcribe(wav) == "cześć Wojtek"
    assert seen["url"] == "http://gpu-box:8110/transcribe"
    assert seen["bytes"] == 304


def test_remote_transcriber_raises_on_service_error(tmp_path, monkeypatch):
    import wojtek_agent.voice as voice

    wav = tmp_path / "u.wav"
    wav.write_bytes(b"x" * 300)

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"ok": False, "error": "no audio received"}

    monkeypatch.setitem(__import__("sys").modules, "httpx",
                        type("m", (), {"post": staticmethod(lambda *a, **k: FakeResp())}))
    with pytest.raises(RuntimeError, match="no audio"):
        voice.RemoteTranscriber("http://gpu:8110").transcribe(wav)
