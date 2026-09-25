"""The VLM brain's policy on a desk: answers in, actions out, no model."""

import pytest

from wojtek_nav.vlm_brain import (
    Explorer,
    SCHEMA,
    chat_url,
    normalised_point,
    parse_answer,
    task_prompt,
    verify_prompt,
)


@pytest.mark.parametrize("base", [
    "http://dgx:8000", "http://dgx:8000/", "http://dgx:8000/v1", "http://dgx:8000/v1/",
])
def test_chat_url_takes_a_base_url_with_or_without_v1(base):
    assert chat_url(base) == "http://dgx:8000/v1/chat/completions"


def test_schema_is_the_whole_vocabulary():
    assert SCHEMA["properties"]["type"]["enum"] == ["goal", "turn", "not_visible", "done"]
    assert SCHEMA["additionalProperties"] is False


def test_prompts_carry_the_instruction_verbatim():
    assert "podejdź do fioletowego słupa" in task_prompt("podejdź do fioletowego słupa")
    v = verify_prompt("podejdź do niskiej skrzynki", "skrzynka")
    assert "podejdź do niskiej skrzynki" in v and "skrzynka" in v
    assert "size" in v  # the property that a big crate failed on


def test_parse_tolerates_wrapping_and_rejects_prose():
    assert parse_answer('```json\n{"type":"done"}\n```')["type"] == "done"
    assert parse_answer("I see a crate at the left")["type"] == "unparsable"


def test_normalised_point_bounds():
    assert normalised_point({"point_2d": [500, 250]}) == (0.5, 0.25)
    assert normalised_point({"point_2d": [1001, 0]}) is None
    assert normalised_point({"point_2d": [1]}) is None
    assert normalised_point({}) is None


def test_goal_is_verified_before_it_moves():
    e = Explorer()
    assert e.on_answer({"type": "goal", "point_2d": [600, 250], "label": "słup"}) == ("verify", "słup")
    assert e.on_verified(True) == ("goal", (0.6, 0.25))


def test_unverified_goal_becomes_a_search():
    e = Explorer(turn_deg=45)
    e.on_answer({"type": "goal", "point_2d": [600, 250], "label": "x"})
    assert e.on_verified(False) == ("search", 45.0)


def test_not_visible_turns_and_a_full_circle_explores():
    e = Explorer(turn_deg=90)
    for _ in range(4):
        assert e.on_answer({"type": "not_visible"}) == ("search", 90.0)
    assert e.on_answer({"type": "not_visible"}) == ("explore", 1.0)
    assert e.on_answer({"type": "not_visible"}) == ("search", 90.0)  # the circle restarts


def test_models_own_turn_is_honoured():
    e = Explorer()
    assert e.on_answer({"type": "turn", "deg": -60}) == ("search", -60.0)


def test_done_is_the_executives_call_not_the_models():
    e = Explorer(done_within_m=1.1)
    assert e.on_answer({"type": "done"}) == ("search", 45.0)  # not trusted
    e.on_answer({"type": "goal", "point_2d": [500, 500]})
    e.on_verified(True)
    assert e.on_goal_result("reached", distance_m=2.4) == ("look",)
    assert e.on_goal_result("reached", distance_m=0.85) == ("done",)


def test_far_target_is_approached_and_a_blocked_approach_turns():
    e = Explorer(approach_m=1.0)
    e.on_answer({"type": "goal", "point_2d": [500, 500]})
    e.on_verified(True)
    assert e.on_goal_result("no_depth") == ("approach", 1.0)
    assert e.on_move_result("blocked") == ("search", 45.0)
    assert e.on_move_result("reached") == ("look",)


def test_blocked_goal_turns_instead_of_pushing_again():
    e = Explorer()
    e.on_answer({"type": "goal", "point_2d": [500, 500]})
    e.on_verified(True)
    assert e.on_goal_result("blocked") == ("search", 45.0)


def test_gives_up_after_max_steps():
    e = Explorer(max_steps=2)
    e.on_answer({"type": "not_visible"})
    e.on_answer({"type": "not_visible"})
    assert e.on_answer({"type": "not_visible"}) == ("gave_up",)


@pytest.mark.parametrize("bad", [{"type": "unparsable"}, {"type": "goal"}, {"type": "goal", "point_2d": [2000, 0]}])
def test_garbage_answers_search(bad):
    assert Explorer().on_answer(bad) == ("search", 45.0)
