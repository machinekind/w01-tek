"""Stage timing: spans, turn correlation, hot-loop rollups.

Model-free by construction -- perf is stdlib plus the trace writer, so the
whole latency vocabulary is testable without a GPU, a sim or a socket.
"""

import asyncio

import pytest

from wojtek_rl import perf
from wojtek_rl.agent.trace import Trace


@pytest.fixture(autouse=True)
def unbound():
    """Each test binds its own trace and leaves nothing behind."""
    perf.bind(None)
    perf.set_turn(None)
    yield
    perf.bind(None)
    perf.set_turn(None)


def spans(trace, stage=None):
    out = [e for e in trace.recent(limit=1000) if e["kind"] == "perf.span"]
    return [e for e in out if stage is None or e["stage"] == stage]


def test_span_records_stage_and_duration():
    t = Trace()
    perf.bind(t)
    with perf.span("asr.transcribe", audio_s=1.5):
        pass
    (event,) = spans(t)
    assert event["stage"] == "asr.transcribe"
    assert event["audio_s"] == 1.5
    assert event["ok"] is True
    assert event["ms"] >= 0


def test_span_is_a_noop_without_a_trace():
    with perf.span("asr.transcribe"):
        pass  # nothing bound: must not raise


def test_span_fields_can_be_filled_in_at_the_end():
    t = Trace()
    perf.bind(t)
    with perf.span("llm.chat") as sp:
        sp["tokens"] = 412
    assert spans(t)[0]["tokens"] == 412


def test_a_raising_stage_is_still_timed():
    """A stage that spends four seconds and then throws is exactly the kind
    of latency worth seeing, so it is recorded and tagged."""
    t = Trace()
    perf.bind(t)
    with pytest.raises(RuntimeError):
        with perf.span("tool.look"):
            raise RuntimeError("camera gone")
    assert spans(t)[0]["ok"] is False


def test_turn_id_lands_on_every_span():
    t = Trace()
    perf.bind(t)
    turn = perf.start_turn("voice", seconds=1.2)
    with perf.span("asr.transcribe"):
        pass
    with perf.span("chat.turn"):
        pass
    assert {e["turn"] for e in spans(t)} == {turn}
    assert [e["kind"] for e in t.recent()][0] == "perf.turn"


def test_turn_id_is_inherited_by_tasks():
    """The chat turn and its TTS reply run as tasks started from the handler;
    contextvars copy into them, which is what joins a turn's spans."""
    t = Trace()
    perf.bind(t)

    async def stage():
        with perf.span("tts.first_audio"):
            await asyncio.sleep(0)

    async def scenario():
        turn = perf.start_turn("voice")
        await asyncio.create_task(stage())
        return turn

    turn = asyncio.run(scenario())
    assert spans(t, "tts.first_audio")[0]["turn"] == turn


def test_turn_elapsed_tracks_the_whole_wait():
    perf.set_turn(None)
    assert perf.turn_elapsed_ms() is None
    perf.start_turn("voice")
    assert perf.turn_elapsed_ms() >= 0


def test_record_reports_a_timing_measured_elsewhere():
    t = Trace()
    perf.bind(t)
    perf.record("mic.endpoint", 700.0, ended_on="silence")
    event = spans(t)[0]
    assert event["stage"] == "mic.endpoint" and event["ms"] == 700.0
    assert event["ended_on"] == "silence"


def test_timer_reports_once_and_cancel_reports_never():
    t = Trace()
    perf.bind(t)
    timer = perf.Timer("tts.first_audio", chars=12)
    timer.stop()
    timer.stop()  # idempotent: a second frame is not a second first-audio
    assert len(spans(t, "tts.first_audio")) == 1
    cancelled = perf.Timer("tts.speak")
    cancelled.cancel()
    cancelled.stop()
    assert not spans(t, "tts.speak")


def test_meter_rolls_up_instead_of_one_event_per_tick():
    t = Trace()
    meter = perf.Meter(rollup_s=0.0, trace=t)
    for _ in range(100):
        meter.add("sim.step", 2.0)
    meter.add("sim.step", 9.0)
    meter.flush(force=True)
    rollups = [e for e in t.recent() if e["kind"] == "perf.rollup"]
    assert len(rollups) == 1
    assert rollups[0]["n"] == 101
    assert rollups[0]["ms"] == pytest.approx(209.0)
    assert rollups[0]["ms_max"] == 9.0


def test_meter_flush_starts_a_new_window():
    t = Trace()
    meter = perf.Meter(rollup_s=0.0, trace=t)
    meter.add("ws.send", 1.0)
    meter.flush(force=True)
    meter.flush(force=True)  # nothing accumulated since: no empty rollup
    assert len([e for e in t.recent() if e["kind"] == "perf.rollup"]) == 1


def test_meter_time_is_a_context_manager():
    t = Trace()
    meter = perf.Meter(rollup_s=0.0, trace=t)
    with meter.time("sim.render_pair"):
        pass
    meter.flush(force=True)
    assert [e["stage"] for e in t.recent() if e["kind"] == "perf.rollup"] == [
        "sim.render_pair"
    ]
