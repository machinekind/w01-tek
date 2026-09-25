"""The latency report: ranked stages, hot-loop rollups, critical path."""

import json

from wojtek_rl import perf_report


def write_trace(tmp_path, events):
    path = tmp_path / "trace.jsonl"
    with path.open("w") as f:
        for i, e in enumerate(events, 1):
            f.write(json.dumps({"seq": i, "t": i * 0.1, **e}) + "\n")
    return path


def span(stage, ms, turn=None, ok=True, **fields):
    return {"kind": "perf.span", "stage": stage, "ms": ms, "turn": turn,
            "ok": ok, **fields}


SESSION = [
    {"kind": "session.start", "scene": "flat"},
    {"kind": "perf.turn", "turn": "voice1", "turn_kind": "voice"},
    span("mic.endpoint", 700, "voice1"),
    span("asr.transcribe", 600, "voice1", audio_s=2.0),
    span("llm.chat", 2000, "voice1"),
    span("tool.look", 300, "voice1"),
    span("llm.chat", 1800, "voice1"),
    span("chat.turn", 4200, "voice1"),
    span("tts.synth", 900, "voice1"),
    span("tts.first_audio", 950, "voice1"),
    span("voice.reply", 6500, "voice1"),
    {"kind": "perf.turn", "turn": "voice2", "turn_kind": "voice"},
    span("mic.endpoint", 700, "voice2"),
    span("asr.transcribe", 400, "voice2", audio_s=1.2),
    span("llm.chat", 2400, "voice2"),
    span("chat.turn", 2500, "voice2"),
    span("tts.first_audio", 1050, "voice2"),
    span("voice.reply", 4700, "voice2"),
    # Background work, correlated to no turn: the robot walking.
    span("nav.execute", 3000, None, step=1),
    span("nav.decide", 1500, None, step=2, prefetch=True),
    {"kind": "perf.rollup", "stage": "sim.step", "n": 250, "ms": 400.0,
     "ms_max": 9.0, "ms_mean": 1.6, "window_s": 5.0},
    {"kind": "perf.rollup", "stage": "sim.render_pair", "n": 125, "ms": 1500.0,
     "ms_max": 30.0, "ms_mean": 12.0, "window_s": 5.0},
    {"kind": "perf.rollup", "stage": "sim.step", "n": 250, "ms": 500.0,
     "ms_max": 21.0, "ms_mean": 2.0, "window_s": 5.0},
]


def test_stages_ranked_by_total_time():
    rows = [r for r in perf_report.span_stats(SESSION) if r["role"] != "umbrella"]
    assert rows[0]["stage"] == "llm.chat"            # 2.0 + 1.8 + 2.4 s
    assert rows[0]["total_ms"] == 6200
    assert rows[0]["n"] == 3
    assert rows[0]["max_ms"] == 2400
    assert rows[0]["p50_ms"] == 2000


def test_containers_are_separated_from_the_work_they_contain():
    """chat.turn holds its own model calls and voice.reply holds everything;
    ranked together they would double count every second in the session."""
    roles = {r["stage"]: r["role"] for r in perf_report.span_stats(SESSION)}
    assert roles["chat.turn"] == "umbrella"
    assert roles["voice.reply"] == "umbrella"
    assert roles["llm.chat"] == "work"


def test_walking_time_is_marked_background():
    """nav.execute is wall-clock nobody waits through; ranking it beside a
    model call without saying so points optimisation at the wrong stage."""
    rows = {r["stage"]: r for r in perf_report.span_stats(SESSION)}
    assert rows["nav.execute"]["background"] is True
    assert rows["llm.chat"]["background"] is False


def test_failed_stages_are_counted_not_dropped():
    events = [span("tool.look", 4000, "voice1", ok=False),
              span("tool.look", 100, "voice1")]
    (row,) = perf_report.span_stats(events)
    assert row["n"] == 2 and row["failed"] == 1


def test_rollups_accumulate_across_windows():
    rows = {r["stage"]: r for r in perf_report.rollup_stats(SESSION)}
    assert rows["sim.step"]["n"] == 500
    assert rows["sim.step"]["total_ms"] == 900.0
    assert rows["sim.step"]["max_ms"] == 21.0    # worst tick of either window
    assert rows["sim.step"]["window_s"] == 10.0


def test_turns_are_rebuilt_from_span_ids():
    rows = {r["turn"]: r for r in perf_report.turns(SESSION)}
    assert set(rows) == {"voice1", "voice2"}
    first = rows["voice1"]
    assert first["llm_calls"] == 2
    assert first["tools"] == ["look"]
    assert first["llm_ms"] == 3800
    assert first["tool_ms"] == 300
    assert first["measured_ms"] == 6500
    # The path sums only the four critical-path stages -- nested llm/tool
    # spans live inside chat.turn and must not be double counted.
    assert first["path_ms"] == 700 + 600 + 4200 + 950
    assert first["spoke"] is True


def test_critical_path_is_the_median_turn():
    rows = {r["stage"]: r for r in perf_report.critical_path(perf_report.turns(SESSION))}
    assert rows["mic.endpoint"]["median_ms"] == 700
    assert rows["asr.transcribe"]["median_ms"] == 500      # median of 600, 400
    assert rows["chat.turn"]["median_ms"] == 3350
    assert rows["tts.first_audio"]["median_ms"] == 1000
    assert rows["mic.endpoint"]["n"] == 2


def test_render_names_the_biggest_stages(tmp_path):
    path = write_trace(tmp_path, SESSION)
    text = perf_report.render(SESSION, [path])
    assert "WHERE THE TIME GOES" in text
    assert "CRITICAL PATH" in text
    assert "HOT LOOP" in text
    assert "chat.turn" in text and "sim.render_pair" in text
    # Background stages are flagged where they are ranked.
    assert "background" in text


def test_cli_reads_a_trace_file(tmp_path, capsys):
    path = write_trace(tmp_path, SESSION)
    assert perf_report.main([str(path)]) == 0
    assert "WHERE THE TIME GOES" in capsys.readouterr().out


def test_cli_json_is_machine_readable(tmp_path, capsys):
    path = write_trace(tmp_path, SESSION)
    assert perf_report.main([str(path), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    work = [r for r in data["stages"] if r["role"] == "work"]
    assert work[0]["stage"] == "llm.chat"
    assert len(data["turns"]) == 2


def test_cli_reports_a_missing_file(tmp_path, capsys):
    assert perf_report.main([str(tmp_path / "nope.jsonl")]) == 2
    assert "no such trace" in capsys.readouterr().err


def test_a_half_written_last_line_costs_only_that_line(tmp_path):
    """The trace is written live; a killed process leaves a partial line and
    that must not cost the whole session's timings."""
    path = write_trace(tmp_path, SESSION)
    with path.open("a") as f:
        f.write('{"kind": "perf.span", "stage": "tts.syn')
    events = perf_report.load_events(path)
    assert len(events) == len(SESSION)


def test_a_ros_probe_trace_reads_the_same_way():
    """The ROS latency probe writes the same schema with the ROS stack's
    stage names; one report tool must rank both, or the two stacks' numbers
    are never comparable."""
    ros = [
        span("mic.endpoint", 700, "utt-7", assumed=True),
        span("asr.transcribe", 620, "utt-7"),
        span("brain.route", 40, "utt-7"),
        span("llm.first_sentence", 1400, "utt-7"),
        span("llm.reply", 2600, "utt-7"),
        span("tts.first_audio", 800, "utt-7"),
        span("voice.reply", 3560, "utt-7"),
    ]
    path = [r["stage"] for r in perf_report.critical_path(perf_report.turns(ros))]
    assert path == ["mic.endpoint", "asr.transcribe", "brain.route",
                    "llm.first_sentence", "tts.first_audio"]
    roles = {r["stage"]: r["role"] for r in perf_report.span_stats(ros)}
    assert roles["llm.reply"] == "umbrella"       # contains llm.first_sentence
    assert roles["llm.first_sentence"] == "work"
    (turn,) = perf_report.turns(ros)
    assert turn["llm_ms"] == 1400                 # the container is not added in
    assert turn["measured_ms"] == 3560


def test_a_trace_without_timings_says_so(tmp_path):
    path = write_trace(tmp_path, [{"kind": "chat.ask", "text": "hej"}])
    text = perf_report.render(perf_report.load_events(path), [path])
    assert "No timing events" in text
