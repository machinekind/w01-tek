"""Unit tests for the local (mlx-vlm) backend: prompt + response parsing.

Pure-text logic only -- mlx is never imported, so these run everywhere
(including CI boxes without Apple Silicon).
"""

import pytest

from wojtek_rl.midlevel import parse_command
from wojtek_rl.vlm_local import local_prompt, parse_local_text
from wojtek_rl.vlm_nav import SYSTEM_PROMPT, decision_to_command


# -- parse_local_text ---------------------------------------------------------


def test_parses_clean_json():
    d = parse_local_text('{"action": "turn_left", "amount": 30, "reasoning": "bed is left"}')
    assert d.action == "turn_left"
    assert d.amount == 30.0
    assert d.reasoning == "bed is left"


def test_parses_fenced_json_with_prose():
    text = 'Sure! Here is my decision:\n```json\n{"action": "forward", "amount": 0.5, "reasoning": "clear path"}\n```'
    d = parse_local_text(text)
    assert d.action == "forward"
    assert d.amount == 0.5


def test_parses_null_amount():
    d = parse_local_text('{"action": "done", "amount": null, "reasoning": "goal reached"}')
    assert d.action == "done"
    assert d.amount is None


def test_skips_invalid_json_objects_before_valid_one():
    text = '{"note": "thinking"} then {"action": "turn_right", "amount": 45, "reasoning": "scan"}'
    d = parse_local_text(text)
    assert d.action == "turn_right"
    assert d.amount == 45.0


def test_falls_back_to_free_text_command():
    d = parse_local_text("I will go forward 0.4 to approach the chair.")
    assert d.action == "forward"
    assert d.amount == 0.4


def test_garbage_raises():
    with pytest.raises(ValueError):
        parse_local_text("The room contains a bed and a window.")


def test_unknown_action_in_json_raises_via_fallback_miss():
    with pytest.raises(ValueError):
        parse_local_text('{"action": "jump", "amount": 1, "reasoning": "wheee"}')


def test_parsed_decision_survives_the_full_gate():
    """Local output -> decision -> clamp -> parse_command: the whole path."""
    d = parse_local_text('{"action": "forward", "amount": 99, "reasoning": "far away"}')
    cmd = decision_to_command(d)  # clamps 99 m -> 2 m
    assert cmd == "forward 2"
    parse_command(cmd)  # must not raise


# -- local_prompt -------------------------------------------------------------


def test_prompt_contains_rules_situation_and_format():
    p = local_prompt("go to the bed", [{"cmd": "forward 0.5", "result": "completed"}],
                     step=2, max_steps=20, pose=(1.0, -0.5, 1.57))
    assert SYSTEM_PROMPT.strip() in p
    assert "Goal: go to the bed" in p
    assert "forward 0.5 -> completed" in p
    assert "yaw=90 deg" in p
    assert "JSON" in p
