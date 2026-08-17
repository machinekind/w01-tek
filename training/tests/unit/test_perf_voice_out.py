"""The text-on-screen -> sound-out gap, split into what it is made of.

"The answer appears instantly but takes ages to hear" is a different
question from "the turn is slow", and it has its own decomposition: waiting
behind another sentence, the voice model, the wire, and pushing frames onto
a socket that may be busy rendering video. Each of those has a different
fix, so each is measured separately.
"""

import asyncio

import numpy as np
import pytest

from wojtek_rl import perf, perf_report
from wojtek_rl.agent import tts as tts_mod
from wojtek_rl.agent.trace import Trace
from wojtek_rl.agent.tts import Speaker
from wojtek_rl.agent.tts_server import pcm_to_wav_bytes
from wojtek_rl.agent.voice import SAMPLE_RATE


@pytest.fixture(autouse=True)
def bound():
    trace = Trace()
    perf.bind(trace)
    perf.set_turn(None)
    yield trace
    perf.bind(None)
    perf.set_turn(None)


class SlowEngine:
    """Synthesises a fixed amount of audio after a measurable pause."""

    rate = SAMPLE_RATE

    def __init__(self, seconds=0.6, pause_s=0.02):
        self.pause_s = pause_s
        self._pcm = np.zeros(int(SAMPLE_RATE * seconds), np.int16)

    def synthesize(self, text):
        import time

        time.sleep(self.pause_s)
        return self._pcm


class FakeResp:
    def __init__(self, content, headers):
        self.content = content
        self.headers = headers

    def raise_for_status(self):
        pass


def spans(trace, stage):
    return [e for e in trace.recent(limit=500)
            if e["kind"] == "perf.span" and e["stage"] == stage]


def speak(text, engine=None, mark_text=True):
    frames = []

    async def send(pcm_bytes):
        frames.append(pcm_bytes)

    async def scenario():
        perf.start_turn("voice")
        if mark_text:
            perf.mark("reply.text")
        speaker = Speaker(engine or SlowEngine(), send)
        speaker.say(text)
        await speaker._task

    asyncio.run(scenario())
    return frames


def test_the_read_to_heard_gap_is_measured(bound):
    speak("Widzę kanapę.")
    (gap,) = spans(bound, "reply.text_to_sound")
    (first,) = spans(bound, "tts.first_audio")
    # The gap starts when the text was shown, so it can only be shorter than
    # the reply's own clock, never longer.
    assert gap["ms"] <= first["ms"] + 1


def test_no_text_mark_means_no_invented_gap(bound):
    """A spoken line with no on-screen text (a goal announcement) has no
    read-to-heard gap to report."""
    speak("Znalazłem kanapę!", mark_text=False)
    assert not spans(bound, "reply.text_to_sound")
    assert spans(bound, "tts.first_audio")


TWO_CHUNKS = (
    "Widzę dużą kanapę stojącą przy oknie po lewej stronie. "
    "Zaraz sprawdzę co tam jeszcze widać w kuchni."
)


def test_synthesis_and_streaming_are_separate_stages(bound):
    """Making the audio and shipping it are different problems: a busy event
    loop delays the second without touching the first."""
    speak(TWO_CHUNKS)
    assert len(spans(bound, "tts.synth")) == 2
    assert len(spans(bound, "tts.stream")) == 2
    assert spans(bound, "tts.synth")[0]["chunk"] == 0


def test_first_chunk_size_is_recorded(bound):
    """Nothing is audible until the FIRST chunk is fully synthesised, so its
    length is a latency knob and the report needs to see it."""
    speak(TWO_CHUNKS)
    first = spans(bound, "tts.first_audio")[0]
    assert first["chunks"] == 2
    assert first["first_chunk_chars"] == len(TWO_CHUNKS.split(". ")[0]) + 1


def test_a_punchy_reply_starts_speaking_after_its_first_words(bound):
    """MIN_CHUNK_CHARS used to merge a short opening sentence into the next
    one, so "Widzę kanapę. Hau hau!" was synthesised as ONE chunk and the
    pipelining that chunking exists for never happened for exactly the
    replies the dog gives most often (measured 2026-08-17). The first chunk
    now stands alone; later chunks keep the higher floor."""
    speak("Widzę kanapę. Hau hau!")
    first = spans(bound, "tts.first_audio")[0]
    assert first["chunks"] == 2
    assert first["first_chunk_chars"] == len("Widzę kanapę.")


def test_real_time_factor_is_reported(bound):
    synth = spans(bound, "tts.synth") if speak("Hau hau.") else []
    assert synth[0]["audio_s"] == pytest.approx(0.6, abs=0.05)
    assert synth[0]["rtf"] is not None


def test_remote_engine_splits_gpu_queue_and_wire(bound, monkeypatch):
    """A slow voice and a voice queued behind another sentence look
    identical from the client unless the server says which it was."""
    import httpx

    wav = pcm_to_wav_bytes(np.zeros(2400, np.int16))
    headers = {"x-synth-ms": "800.0", "x-queue-ms": "150.0", "x-audio-ms": "100.0"}
    monkeypatch.setattr(
        httpx, "post", lambda url, json=None, timeout=None: FakeResp(wav, headers)
    )
    tts_mod.RemoteTts("http://box:8120").synthesize("cześć")
    assert spans(bound, "tts.server_gpu")[0]["ms"] == 800.0
    assert spans(bound, "tts.server_queue")[0]["ms"] == 150.0
    # Whatever the round trip cost beyond the server's own accounting is the
    # wire, and it is never negative.
    assert spans(bound, "tts.network")[0]["ms"] >= 0
    assert spans(bound, "tts.request")


def test_an_old_server_without_headers_still_works(bound, monkeypatch):
    import httpx

    wav = pcm_to_wav_bytes(np.zeros(2400, np.int16))
    monkeypatch.setattr(
        httpx, "post", lambda url, json=None, timeout=None: FakeResp(wav, {})
    )
    out = tts_mod.RemoteTts("http://box:8120").synthesize("cześć")
    assert len(out) == 2400
    assert spans(bound, "tts.request")      # round trip still timed
    assert not spans(bound, "tts.server_gpu")


def test_report_ranks_the_pieces_of_the_gap():
    events = [
        {"kind": "perf.span", "stage": "reply.text_to_sound", "ms": 3200,
         "turn": "voice1", "first_chunk_chars": 44},
        {"kind": "perf.span", "stage": "tts.first_audio", "ms": 3260,
         "turn": "voice1", "chunks": 2},
        {"kind": "perf.span", "stage": "tts.synth", "ms": 3100, "turn": "voice1",
         "chunk": 0, "rtf": 1.4},
        {"kind": "perf.span", "stage": "tts.request", "ms": 3080, "turn": "voice1",
         "chunk": 0},
        {"kind": "perf.span", "stage": "tts.server_gpu", "ms": 2600, "turn": "voice1",
         "chunk": 0},
        {"kind": "perf.span", "stage": "tts.server_queue", "ms": 300, "turn": "voice1",
         "chunk": 0},
        {"kind": "perf.span", "stage": "tts.network", "ms": 180, "turn": "voice1",
         "chunk": 0},
        {"kind": "perf.span", "stage": "tts.stream", "ms": 60, "turn": "voice1",
         "chunk": 0},
        # The second sentence plays behind the first: not part of the wait.
        {"kind": "perf.span", "stage": "tts.synth", "ms": 9000, "turn": "voice1",
         "chunk": 1},
    ]
    vo = perf_report.voice_out(events)
    assert vo["text_to_sound_ms"] == 3200
    assert vo["first_chunk_chars"] == 44
    assert vo["rtf"] == 1.4
    ranked = [r["stage"] for r in vo["stages"]]
    assert ranked[0] == "tts.synth"            # the whole call, then its parts
    assert "tts.server_gpu" in ranked and "tts.stream" in ranked
    # Chunk 1's 9 s must not pollute the first-chunk numbers.
    assert all(r["median_ms"] < 9000 for r in vo["stages"])
    text = perf_report.render(events, [__import__("pathlib").Path("t.jsonl")])
    assert "TEXT ON SCREEN -> SOUND" in text
    assert "READ -> HEARD" in text


def test_remote_detail_stops_the_call_being_counted_twice():
    """With a remote engine, tts.synth CONTAINS the server and wire rows, so
    it moves out of the ranked table; with a local engine it is the work."""
    remote = [{"kind": "perf.span", "stage": "tts.synth", "ms": 3100, "chunk": 0},
              {"kind": "perf.span", "stage": "tts.request", "ms": 3080, "chunk": 0},
              {"kind": "perf.span", "stage": "tts.server_gpu", "ms": 2600, "chunk": 0}]
    roles = {r["stage"]: r["role"] for r in perf_report.span_stats(remote)}
    assert roles["tts.synth"] == "umbrella" and roles["tts.request"] == "umbrella"
    assert roles["tts.server_gpu"] == "work"

    local = [{"kind": "perf.span", "stage": "tts.synth", "ms": 760, "chunk": 0}]
    assert perf_report.span_stats(local)[0]["role"] == "work"


# ---- the fixes: start speaking before the whole reply exists ----------------


def test_short_first_sentence_now_ships_on_its_own():
    """The fix for the merge above: the FIRST chunk may be short, because it
    is the only one anyone waits through. Later chunks keep the higher floor
    so delivery does not turn choppy."""
    chunks = tts_mod.speech_chunks("Widzę kanapę. Hau hau! Idę tam teraz.")
    assert chunks[0] == "Widzę kanapę."
    assert len(chunks) > 1


def test_a_long_opening_sentence_is_cut_at_a_clause():
    long_open = (
        "Szukam kanapy, ale nie widzę jej teraz i idę w stronę, gdzie stoją "
        "półki oraz reszta mebli w tym pokoju. Zaraz sprawdzę."
    )
    chunks = tts_mod.speech_chunks(long_open)
    assert chunks[0] == "Szukam kanapy,"
    assert "ale nie widzę jej teraz" in chunks[1]


def test_a_tiny_reply_is_still_one_chunk():
    assert tts_mod.speech_chunks("Hau!") == ["Hau!"]


def test_streaming_engine_speaks_before_synthesis_finishes(bound):
    """The point of the whole change: first audio must land after the first
    PARTIAL, not after the last one."""
    class StreamEngine:
        rate = SAMPLE_RATE

        def synth_stream(self, text):
            for _ in range(3):
                yield np.zeros(int(SAMPLE_RATE * 0.4), np.int16)

        def synthesize(self, text):  # never used when streaming works
            raise AssertionError("streaming engine must not fall back")

    frames = speak("Widzę kanapę.", engine=StreamEngine())
    assert frames
    synth = spans(bound, "tts.synth")
    assert [s["part"] for s in synth[:3]] == [0, 1, 2]
    assert all(s["streamed"] for s in synth)
    first_audio = spans(bound, "tts.first_audio")[0]["ms"]
    speak_total = spans(bound, "tts.speak")[0]["ms"]
    assert first_audio <= speak_total


def test_remote_stream_reassembles_split_int16_frames(monkeypatch):
    """Chunk boundaries fall anywhere in the byte stream; a sample split
    across two chunks must not become noise (or shift every later sample)."""
    import httpx

    pcm = np.arange(-4, 6, dtype=np.int16)
    raw = pcm.tobytes()

    class FakeStream:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def raise_for_status(self):
            pass

        def read(self):
            pass

        def iter_bytes(self):
            yield raw[:3]        # 1.5 samples
            yield raw[3:11]
            yield raw[11:]

    monkeypatch.setattr(httpx, "stream", lambda *a, **k: FakeStream())
    out = np.concatenate(list(tts_mod.RemoteTts("http://box:8120").synth_stream("hej")))
    assert np.array_equal(out, pcm)


def test_remote_stream_falls_back_on_an_old_server(monkeypatch):
    import httpx

    wav = pcm_to_wav_bytes(np.full(1200, 7, np.int16))

    class Missing:
        status_code = 404

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            pass

        def raise_for_status(self):
            raise AssertionError("404 must be handled before raise_for_status")

        def iter_bytes(self):
            return iter(())

    monkeypatch.setattr(httpx, "stream", lambda *a, **k: Missing())
    monkeypatch.setattr(
        httpx, "post", lambda url, json=None, timeout=None: FakeResp(wav, {})
    )
    out = np.concatenate(list(tts_mod.RemoteTts("http://box:8120").synth_stream("hej")))
    assert len(out) == 1200


def test_server_splits_a_reply_into_streamable_pieces():
    from wojtek_rl.agent.tts_server import split_for_stream

    pieces = split_for_stream(
        "Szukam kanapy, ale nie widzę jej teraz, idę w stronę półek. Zaraz wrócę."
    )
    assert pieces[0] == "Szukam kanapy,"
    assert len(pieces) >= 2
    # A short tail is glued on rather than shipped as a clipped fragment.
    assert not any(len(p) < 10 for p in pieces)
    assert " ".join(pieces).replace(" ,", ",") .count("Zaraz wrócę.") == 1


def test_server_stream_split_keeps_short_replies_whole():
    from wojtek_rl.agent.tts_server import split_for_stream

    assert split_for_stream("Już się zatrzymuję!") == ["Już się zatrzymuję!"]
    assert split_for_stream("") == []


# ---- the server's line cache ------------------------------------------------


class StubModel:
    """Stands in for Chatterbox: counts generations, returns fixed audio."""

    sr = SAMPLE_RATE

    def __init__(self):
        self.calls = []

    def generate(self, text, **kwargs):
        self.calls.append(text)
        return _StubTensor()


class _StubTensor:
    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self

    def squeeze(self):
        return np.zeros(2400, np.float32)


def stub_server(**kwargs):
    from fastapi.testclient import TestClient

    from wojtek_rl.agent import tts_server

    model = StubModel()
    app = tts_server.build_app("", "pl", "cpu", model_factory=lambda: model, **kwargs)
    return TestClient(app), model


def test_a_repeated_line_costs_the_gpu_once():
    """Goal acknowledgements and "Już się zatrzymuję!" are fixed strings said
    over and over, and Chatterbox needs 1.3-1.6 wall seconds per audio second
    (measured on an A6000). Saying one twice must not cost twice."""
    client, model = stub_server()
    first = client.post("/synthesize", json={"text": "Już się zatrzymuję!"})
    second = client.post("/synthesize", json={"text": "Już się zatrzymuję!"})
    assert first.headers["x-cache"] == "miss"
    assert second.headers["x-cache"] == "hit"
    assert float(second.headers["x-synth-ms"]) == 0.0
    assert model.calls == ["Już się zatrzymuję!"]     # generated exactly once
    assert first.content == second.content


def test_the_cache_is_bounded():
    client, model = stub_server(cache_size=2)
    for line in ("jeden", "dwa", "trzy"):
        client.post("/synthesize", json={"text": line})
    # "jeden" fell out; "trzy" is still there.
    assert client.post("/synthesize", json={"text": "jeden"}).headers["x-cache"] == "miss"
    assert client.post("/synthesize", json={"text": "trzy"}).headers["x-cache"] == "hit"


def test_cache_can_be_switched_off():
    client, model = stub_server(cache_size=0)
    client.post("/synthesize", json={"text": "Hau hau!"})
    again = client.post("/synthesize", json={"text": "Hau hau!"})
    assert again.headers["x-cache"] == "miss"
    assert model.calls == ["Hau hau!", "Hau hau!"]


def test_the_stream_endpoint_ships_pieces_and_caches_them():
    client, model = stub_server()
    text = "Szukam kanapy, ale nie widzę jej teraz, idę w stronę półek. Zaraz wrócę."
    body = client.post("/synthesize_stream", json={"text": text}).content
    assert len(model.calls) >= 2                      # split into pieces
    assert len(body) == 2400 * 2 * len(model.calls)   # raw PCM16, no WAV header
    before = len(model.calls)
    client.post("/synthesize_stream", json={"text": text})
    assert model.calls[before:] == []                 # second time, all cached
