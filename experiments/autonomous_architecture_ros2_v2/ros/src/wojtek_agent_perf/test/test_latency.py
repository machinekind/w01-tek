"""Pipeline stamps -> stage timings. No ROS graph, no messages, no models."""

import json

from wojtek_agent_perf.latency import STAGES, LatencyProbe, SpanWriter, stamp_to_s


class FakeStamp:
    def __init__(self, sec, nanosec):
        self.sec, self.nanosec = sec, nanosec


def run_turn(probe, uid, t0=100.0, asr=0.6, route=0.05, agent=1.4, first=1.9,
             final=2.6, audio=0.9):
    """One complete VLM-agent turn, stamps in pipeline order. Bielik-direct
    turns simply never emit say_en_first, so agent.turn/brain.translate stay
    silent for them (a missing stage means the boundary was not crossed)."""
    records = []
    records += probe.observe("speech_end", uid, t0)
    records += probe.observe("asr_final", uid, t0 + asr, text="gdzie jest kanapa")
    records += probe.observe("intent", uid, t0 + asr + route)
    records += probe.observe("say_en_first", uid, t0 + asr + route + agent)
    records += probe.observe("say_first", uid, t0 + asr + route + first)
    records += probe.observe("say_final", uid, t0 + asr + route + final)
    records += probe.observe("audio_first", uid, t0 + asr + route + first + audio)
    return records


def stages(records):
    return {r["stage"]: r["ms"] for r in records}


def test_a_turn_decomposes_into_stages():
    got = stages(run_turn(LatencyProbe(), "u1"))
    assert got["asr.transcribe"] == 600.0
    assert got["brain.route"] == 50.0
    assert got["llm.first_sentence"] == 1900.0
    assert got["llm.reply"] == 2600.0
    assert got["tts.first_audio"] == 900.0
    assert got["voice.reply"] == 3450.0    # mic to sound, the felt number


def test_endpoint_wait_is_reported_and_marked_assumed():
    """The VAD's silence timeout is real waiting no one can observe from
    outside the node; dropping it would flatter every total."""
    probe = LatencyProbe(endpoint_wait_s=0.7)
    (record,) = probe.observe("speech_end", "u1", 10.0)
    assert record["stage"] == "mic.endpoint"
    assert record["ms"] == 700.0
    assert record["assumed"] is True


def test_stages_are_emitted_once_each():
    probe = LatencyProbe()
    first = run_turn(probe, "u1")
    again = probe.observe("audio_first", "u1", 999.0)   # later frames of the reply
    assert "voice.reply" in stages(first)
    assert again == []


def test_first_sentence_wins_but_final_moves():
    """The reply streams sentence by sentence: the first one is the latency
    that matters, the last one closes llm.reply."""
    probe = LatencyProbe()
    probe.observe("speech_end", "u1", 0.0)
    probe.observe("asr_final", "u1", 0.5)
    probe.observe("intent", "u1", 0.6)
    probe.observe("say_first", "u1", 1.6)
    probe.observe("say_first", "u1", 2.4)      # second sentence, not the first
    records = probe.observe("say_final", "u1", 3.1)
    assert stages(records)["llm.reply"] == 2500.0
    assert probe.turns["u1"].at["say_first"] == 1.6


def test_a_broken_turn_reports_only_what_happened():
    """The model died after routing: the stages before it stand, the ones
    after it are absent -- absence means broken, not instant."""
    probe = LatencyProbe()
    probe.observe("speech_end", "u1", 0.0)
    got = stages(probe.observe("asr_final", "u1", 0.7) + probe.observe("intent", "u1", 0.8))
    assert set(got) == {"asr.transcribe", "brain.route"}


def test_turns_do_not_bleed_into_each_other():
    probe = LatencyProbe()
    probe.observe("speech_end", "u1", 0.0)
    probe.observe("speech_end", "u2", 5.0)
    got = stages(probe.observe("asr_final", "u1", 1.0))
    assert got["asr.transcribe"] == 1000.0
    assert "asr.transcribe" not in stages(probe.observe("intent", "u2", 6.0))


def test_untagged_messages_are_ignored():
    """Mic frames and system chatter carry no utterance id; correlating them
    to whatever turn was last would invent latency."""
    assert LatencyProbe().observe("audio_first", "", 1.0) == []


def test_old_turns_are_dropped():
    probe = LatencyProbe(max_turns=3)
    for i in range(10):
        probe.observe("speech_end", f"u{i}", float(i))
    assert len(probe.turns) == 3
    assert "u9" in probe.turns


def test_records_are_the_span_schema_the_report_reads():
    record = run_turn(LatencyProbe(), "u1")[-1]
    assert record["kind"] == "perf.span"
    assert set(record) >= {"seq", "t", "wall", "kind", "stage", "ms", "turn", "ok"}
    assert record["turn"] == "u1"
    assert record["t"] >= 0


def test_every_declared_stage_can_be_produced():
    got = stages(run_turn(LatencyProbe(), "u1"))
    assert {s for s, _a, _b in STAGES} <= set(got)


def test_stamp_conversion():
    assert stamp_to_s(FakeStamp(12, 500_000_000)) == 12.5


def test_writer_appends_jsonl(tmp_path):
    path = tmp_path / "deep" / "perf.jsonl"
    writer = SpanWriter(path)
    for record in run_turn(LatencyProbe(), "u1"):
        writer.write(record)
    writer.close()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["stage"] == "mic.endpoint"
    assert rows[-1]["stage"] == "voice.reply"


def test_writer_survives_an_unusable_path(tmp_path):
    """A probe that takes the robot down to report a number is worse than no
    number."""
    errors = []
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")
    writer = SpanWriter(blocked / "perf.jsonl", on_error=errors.append)
    writer.write({"kind": "perf.span"})
    writer.close()
    assert errors
