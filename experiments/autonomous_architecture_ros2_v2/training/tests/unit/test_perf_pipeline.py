"""The instrumentation itself: a scripted turn must leave a readable trace.

The unit tests for perf.py prove the primitives work; these prove the stages
are actually WIRED into the pipeline -- the failure mode of a profiler is not
a wrong number, it is a missing one, and a stage nobody instrumented looks
exactly like a stage that costs nothing.

Scripted models and a fake engine throughout: no network, no sim, no audio
device.
"""

import asyncio
import json

import numpy as np
import pytest

from wojtek_rl import perf, perf_report
from wojtek_rl.agent.chat import WojtekAgent
from wojtek_rl.agent.tools import Tool, ToolResult
from wojtek_rl.agent.trace import Trace
from wojtek_rl.agent.voice import SAMPLE_RATE, Utterance, VoiceListener, VoiceSegmenter


@pytest.fixture(autouse=True)
def bound_trace(tmp_path):
    trace = Trace(tmp_path / "trace.jsonl")
    perf.bind(trace)
    perf.set_turn(None)
    yield trace
    perf.bind(None)
    perf.set_turn(None)
    trace.close()


class ScriptedLLM:
    async def chat(self, messages, max_tokens=None):
        # First call takes a look, second answers.
        if any("TOOL RESULT" in c.get("text", "")
               for m in messages for c in m.get("content", [])
               if isinstance(c, dict)):
            return json.dumps({"thought": "seen", "say": "Widzę kanapę!"})
        return json.dumps({"thought": "need a look", "look": {}})


class SlowTranscriber:
    def transcribe(self, wav_path):
        return "co widzisz"


def stages_of(trace):
    return [e["stage"] for e in trace.recent(limit=500) if e["kind"] == "perf.span"]


def test_a_chat_turn_times_its_model_calls_and_tools(bound_trace):
    async def look(args):
        return ToolResult(text="a sofa", images=("b64",))

    agent = WojtekAgent(ScriptedLLM(), {"look": Tool("look", "look {}", "cam", look)})
    result = asyncio.run(agent.ask("co widzisz?", voice=True))
    assert result["say"] == "Widzę kanapę!"
    stages = stages_of(bound_trace)
    assert stages.count("llm.chat") == 2
    assert "tool.look" in stages
    assert stages[-1] == "chat.turn"          # the container closes last


def test_a_failing_tool_is_still_timed(bound_trace):
    async def boom(args):
        raise RuntimeError("renderer exploded")

    agent = WojtekAgent(ScriptedLLM(), {"look": Tool("look", "look {}", "cam", boom)})
    asyncio.run(agent.ask("co widzisz?"))
    failed = [e for e in bound_trace.recent(limit=500)
              if e["kind"] == "perf.span" and e["stage"] == "tool.look"]
    assert failed and failed[0]["ok"] is False


def test_hearing_an_utterance_opens_a_turn_and_times_the_asr(bound_trace):
    heard = []

    async def on_text(text, utt):
        heard.append(text)

    listener = VoiceListener(SlowTranscriber(), on_text)
    listener.set_enabled(True)
    utt = Utterance(pcm=np.zeros(SAMPLE_RATE, np.int16), seconds=1.0,
                    ended_on="silence", endpoint_wait_s=0.7)
    asyncio.run(listener._recognise(utt))
    assert heard == ["co widzisz"]
    events = {e["stage"]: e for e in bound_trace.recent(limit=500)
              if e["kind"] == "perf.span"}
    assert events["mic.endpoint"]["ms"] == 700.0
    assert events["asr.transcribe"]["audio_s"] == 1.0
    assert events["asr.transcribe"]["turn"] == events["mic.endpoint"]["turn"]


def test_the_endpoint_wait_is_measured_not_assumed(bound_trace):
    """The segmenter reports the silence it actually accumulated, so a VAD
    that closes early or late is visible instead of hidden behind a config
    value."""
    seg = VoiceSegmenter(silence_end_s=0.3, min_utterance_s=0.1)
    frame = int(SAMPLE_RATE * 0.1)
    loud = (np.ones(frame) * 8000).astype(np.int16)
    quiet = np.zeros(frame, np.int16)
    for _ in range(5):
        assert seg.feed(loud) is None
    utt = None
    for _ in range(5):
        utt = seg.feed(quiet) or utt
    assert utt is not None
    assert utt.endpoint_wait_s == pytest.approx(0.3, abs=0.05)


def test_the_written_trace_feeds_the_report(bound_trace, tmp_path):
    """End to end: instrument, write JSONL, rank it. This is the contract
    between the two halves of the profiler."""
    async def look(args):
        return ToolResult(text="a sofa")

    perf.start_turn("voice", seconds=1.0)
    perf.record("mic.endpoint", 700.0)
    agent = WojtekAgent(ScriptedLLM(), {"look": Tool("look", "look {}", "cam", look)})
    asyncio.run(agent.ask("co widzisz?", voice=True))
    bound_trace.close()

    events = perf_report.load_events(tmp_path / "trace.jsonl")
    rows = {r["stage"]: r for r in perf_report.span_stats(events)}
    assert rows["llm.chat"]["n"] == 2
    (turn,) = perf_report.turns(events)
    assert turn["llm_calls"] == 2 and turn["tools"] == ["look"]
    text = perf_report.render(events, [tmp_path / "trace.jsonl"])
    assert "llm.chat" in text and "chat.turn" in text
