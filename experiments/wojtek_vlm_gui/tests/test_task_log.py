"""The brain's status stream as the page shows it. Pure, no ROS."""

import json

from wojtek_vlm_gui.task_log import (
    TaskLog,
    format_line,
    is_stop_word,
    parse_status,
    verdict,
)


def _s(action, step=1, instruction="go to the purple pillar", **kw):
    return {"action": action, "step": step, "instruction": instruction, "t": 1.0, **kw}


def test_stop_words_are_the_brains():
    for word in ("", "  ", "stop", "STOP", "Stop.", "stój"):
        assert is_stop_word(word), word
    assert not is_stop_word("go to the purple pillar")
    assert not is_stop_word("stop at the door")


def test_parse_status_takes_json_objects_only():
    assert parse_status(json.dumps({"action": "idle"})) == {"action": "idle"}
    assert parse_status("not json") is None
    assert parse_status("[1, 2]") is None


def test_format_line_shows_the_same_fields_as_the_web_console():
    line = format_line(_s("ask", step=3, answer={"type": "goal", "point_2d": [500, 600], "label": "pillar"},
                          latency_s=1.23))
    assert line.startswith("#3 ask")
    assert '"type": "goal"' in line and "1.23s" in line

    assert format_line(_s("turn", deg=45.0, turned_total=90.0)) == "#1 turn  45.0deg (total 90deg)"
    assert format_line(_s("approach", metres=1.0, result="reached")) == "#1 approach  -> reached  1.0 m"
    assert "target 0.85 m" in format_line(_s("pixel_goal", result="reached", distance_to_target_m=0.85))
    assert "ERROR RuntimeError: no fresh colour frame" in format_line(
        _s("finished", result="error", error="RuntimeError: no fresh colour frame"))


def test_verdict_colours_only_a_finished_task():
    assert verdict(_s("ask")) is None
    assert verdict(_s("finished", result="done")) == "ok"
    for bad in ("gave_up", "cancelled", "replaced", "error", "timeout"):
        assert verdict(_s("finished", result=bad)) == "bad", bad


def test_lines_accumulate_in_order_until_finished():
    log = TaskLog()
    log.push(_s("ask", step=1))
    log.push(_s("verify", step=1))
    log.push(_s("pixel_goal", step=1, result="reached"))
    log.push(_s("finished", step=1, result="done"))
    assert [s["action"] for s in log.lines] == ["ask", "verify", "pixel_goal", "finished"]
    assert log.finished["result"] == "done"
    assert log.idle


def test_a_new_instruction_starts_a_new_task():
    log = TaskLog()
    log.push(_s("ask", step=1, instruction="go to the pillar"))
    log.push(_s("finished", step=1, instruction="go to the pillar", result="replaced"))
    log.push(_s("ask", step=1, instruction="go to the box"))
    assert log.instruction == "go to the box"
    assert [s["action"] for s in log.lines] == ["ask"]
    assert log.finished is None and not log.idle


def test_the_same_instruction_again_after_a_finish_is_a_new_task():
    log = TaskLog()
    log.push(_s("ask"))
    log.push(_s("finished", result="gave_up"))
    log.push(_s("ask"))
    assert [s["action"] for s in log.lines] == ["ask"]


def test_idle_keeps_the_lines_and_marks_idle():
    log = TaskLog()
    log.push(_s("ask"))
    log.push(_s("finished", result="cancelled"))
    log.push({"action": "idle", "step": 1, "instruction": "", "t": 2.0})
    assert log.idle
    assert [s["action"] for s in log.lines] == ["ask", "finished"]
