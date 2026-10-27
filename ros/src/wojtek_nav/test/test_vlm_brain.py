"""The VLM brain's policy on a desk: answers in, actions out, no model."""

import math

import pytest

from wojtek_nav.vlm_brain import (
    Explorer,
    SCHEMA,
    chat_url,
    normalised_point,
    parse_answer,
    pixel_ray,
    ray_heading,
    rotate,
    step_along,
    task_prompt,
    verify_prompt,
)


@pytest.mark.parametrize("base", [
    "http://box:11434", "http://box:11434/", "http://box:11434/v1", "http://box:11434/v1/",
])
def test_chat_url_takes_a_base_url_with_or_without_v1(base):
    assert chat_url(base) == "http://box:11434/v1/chat/completions"


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
    # Digit strings are still a usable point.
    assert normalised_point({"point_2d": ["500", "250"]}) == (0.5, 0.25)


@pytest.mark.parametrize("bad", [
    [500], [], [1, 2, 3], "500,250", 500, {"x": 500, "y": 250}, [None, 5], [[500], [250]],
    float("nan"), [float("nan"), 0],
])
def test_normalised_point_rejects_malformed_shapes(bad):
    # The node draws and acts on this value only; the raw point_2d never
    # reaches arithmetic.
    assert normalised_point({"type": "goal", "point_2d": bad}) is None


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


def _pointed(e, point=(500, 500)):
    """The model points, the check confirms: the explorer holds a pixel goal."""
    e.on_answer({"type": "goal", "point_2d": list(point)})
    return e.on_verified(True)


def test_far_target_is_approached_along_the_pixels_bearing():
    e = Explorer(approach_m=1.0)
    _pointed(e, (468, 244))
    # The pixel travels with the approach: the node walks its bearing.
    assert e.on_goal_result("no_depth") == ("approach", 1.0, (0.468, 0.244), 0.0)
    assert e.on_move_result("reached", moved_m=0.95) == ("look",)


def test_blocked_approach_looks_again_and_detours_alternating_sides():
    e = Explorer(detour_deg=40)
    detours, after = [], []
    for _ in range(3):
        _pointed(e)
        detours.append(e.on_goal_result("no_depth")[3])
        after.append(e.on_move_result("blocked", moved_m=0.6))
    assert detours == [0.0, 40.0, -40.0]
    # The straight walk looks again; a detoured one turns back onto the
    # bearing first.
    assert after == [("look",), ("search", -40.0), ("search", 40.0)]
    assert e.turn_done() == ("look",)
    _pointed(e)
    e.on_goal_result("no_depth")
    e.on_move_result("reached", moved_m=1.0)
    _pointed(e)
    assert e.on_goal_result("no_depth")[3] == 0.0  # a clear walk restores the bearing


def _detoured(e):
    """A blocked straight approach, then the next approach goes 40 deg off
    the bearing."""
    _pointed(e)
    e.on_goal_result("no_depth")
    assert e.on_move_result("blocked", moved_m=0.6) == ("look",)
    _pointed(e)
    assert e.on_goal_result("no_depth")[3] == 40.0


def test_reached_detour_turns_back_onto_the_bearing_before_looking():
    """After the detour the body points 40 deg off the target, out of the
    picture: turn back first, or the look says not_visible."""
    e = Explorer(detour_deg=40)
    _detoured(e)
    assert e.on_move_result("reached", moved_m=1.0) == ("search", -40.0)
    assert e.turn_done() == ("look",)
    _pointed(e)
    assert e.on_goal_result("no_depth")[3] == 0.0


def test_turning_back_after_a_detour_does_not_count_towards_the_circle():
    e = Explorer(turn_deg=90, detour_deg=40, max_steps=100)
    _detoured(e)
    e.on_move_result("reached", moved_m=1.0)
    assert e.turned_deg == 0.0
    for _ in range(4):
        assert e.on_answer({"type": "not_visible"}) == ("search", 90.0)
    assert e.on_answer({"type": "not_visible"}) == ("explore", 1.0)


def test_explore_never_turns_back():
    e = Explorer(turn_deg=90, detour_deg=40, max_steps=100)
    _detoured(e)
    e.on_move_result("blocked", moved_m=0.3)  # the detour flips to -40 and waits
    for _ in range(5):
        e.on_answer({"type": "not_visible"})  # four turns, then the explore
    assert e.on_move_result("reached", moved_m=1.0) == ("look",)


def test_stalled_detour_still_gives_up():
    e = Explorer(detour_deg=40, min_progress_m=0.25, max_stalls=1, max_steps=100)
    _detoured(e)
    assert e.on_move_result("reached", moved_m=0.02)[0] == "gave_up"


def test_blocked_explore_still_turns():
    e = Explorer(turn_deg=90)
    for _ in range(5):
        e.on_answer({"type": "not_visible"})  # four turns, then the explore
    assert e.on_move_result("blocked", moved_m=0.5) == ("search", 90.0)


def test_blocked_near_the_target_is_done():
    e = Explorer(done_within_m=1.1)
    _pointed(e)
    assert e.on_goal_result("blocked", distance_m=0.95) == ("done",)
    _pointed(e)
    assert e.on_goal_result("blocked", distance_m=1.8) == ("look",)


def test_target_that_never_enters_depth_range_gives_up_with_a_reason():
    e = Explorer(max_blind=3, max_steps=100)
    for _ in range(3):
        _pointed(e)
        assert e.on_goal_result("no_depth")[0] == "approach"
        assert e.on_move_result("reached", moved_m=1.0) == ("look",)
    _pointed(e)
    kind, why = e.on_goal_result("no_depth")
    assert kind == "gave_up" and "depth range" in why


def test_resolved_depth_resets_the_blind_count():
    e = Explorer(max_blind=2, max_steps=100)
    for _ in range(2):
        _pointed(e)
        e.on_goal_result("no_depth")
        e.on_move_result("reached", moved_m=1.0)
    _pointed(e)
    assert e.on_goal_result("reached", distance_m=2.5) == ("look",)
    _pointed(e)
    assert e.on_goal_result("no_depth")[0] == "approach"


def test_moves_that_go_nowhere_give_up_with_a_reason():
    e = Explorer(min_progress_m=0.25, max_stalls=3, max_steps=100)
    for _ in range(2):
        _pointed(e)
        e.on_goal_result("no_depth")
        assert e.on_move_result("reached", moved_m=0.02) == ("look",)
    _pointed(e)
    e.on_goal_result("no_depth")
    kind, why = e.on_move_result("reached", moved_m=0.02)
    assert kind == "gave_up" and "no progress" in why


def test_a_real_move_clears_the_stall_count():
    e = Explorer(min_progress_m=0.25, max_stalls=2, max_steps=100)
    for moved in (0.1, 0.8, 0.1):
        _pointed(e)
        e.on_goal_result("no_depth")
        assert e.on_move_result("reached", moved_m=moved) == ("look",)


def test_unknown_progress_is_not_a_stall():
    e = Explorer(max_stalls=1)
    _pointed(e)
    e.on_goal_result("no_depth")
    assert e.on_move_result("reached") == ("look",)


def test_pixel_ray_through_the_principal_point_is_the_optical_axis():
    k = [600.0, 0, 424.0, 0, 600.0, 240.0, 0, 0, 1]
    assert pixel_ray(424.0, 240.0, k) == (0.0, 0.0, 1.0)
    assert pixel_ray(1024.0, 240.0, k) == (1.0, 0.0, 1.0)


# Optical frame (x right, y down, z forward) in a body frame (x forward,
# y left, z up) looking straight ahead: the robot's camera, 0 deg pitch.
OPTICAL_IN_BODY = (-0.5, 0.5, -0.5, 0.5)


def _yaw_q(yaw, q):
    """q premultiplied by a yaw about z: the body turned by `yaw` in odom."""
    c, s = math.cos(yaw / 2), math.sin(yaw / 2)
    x, y, z, w = q
    return (c * x - s * y, c * y + s * x, c * z + s * w, c * w - s * z)


def test_rotate_is_a_rotation():
    q = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))  # 90 deg about z
    assert rotate(q, (1.0, 0.0, 0.0)) == pytest.approx((0.0, 1.0, 0.0))


@pytest.mark.parametrize("robot_yaw", [0.0, 0.7, -2.0])
def test_ray_heading_follows_the_pixel_and_the_body(robot_yaw):
    q = _yaw_q(robot_yaw, OPTICAL_IN_BODY)
    k = [600.0, 0, 424.0, 0, 600.0, 240.0, 0, 0, 1]
    assert ray_heading(q, pixel_ray(424.0, 240.0, k)) == pytest.approx(robot_yaw)
    # A pixel right of centre is a bearing to the right (negative yaw).
    right = ray_heading(q, pixel_ray(1024.0, 240.0, k))
    assert right == pytest.approx(robot_yaw - math.pi / 4)


def test_ray_heading_ignores_the_cameras_pitch():
    # base_link -> camera_color_optical_frame as the sim's TF has it: the
    # camera looks 15 deg down.
    q = (0.561, -0.561, 0.430, -0.430)
    k = [600.0, 0, 424.0, 0, 600.0, 240.0, 0, 0, 1]
    for v in (40.0, 240.0, 460.0):  # above, on and below the optical axis
        assert ray_heading(q, pixel_ray(424.0, v, k)) == pytest.approx(0.0, abs=1e-3)
    assert ray_heading(q, pixel_ray(1024.0, 240.0, k)) < -0.5


def test_step_along():
    assert step_along((1.0, 2.0), math.pi / 2, 1.5) == pytest.approx((1.0, 3.5))


def test_blocked_goal_looks_again_then_turns_instead_of_pushing_again():
    e = Explorer()
    _pointed(e)
    assert e.on_goal_result("blocked", distance_m=1.3) == ("look",)  # re-resolve from here
    _pointed(e)
    assert e.on_goal_result("blocked", distance_m=1.3) == ("search", 45.0)
    _pointed(e)
    assert e.on_goal_result("blocked", distance_m=1.3) == ("look",)  # the count restarted


def test_reached_goal_clears_the_block_count():
    e = Explorer()
    _pointed(e)
    e.on_goal_result("blocked", distance_m=2.0)
    _pointed(e)
    e.on_goal_result("reached", distance_m=1.5)
    _pointed(e)
    assert e.on_goal_result("blocked", distance_m=1.3) == ("look",)


def _turn_between(e):
    assert e.on_answer({"type": "not_visible"}) == ("search", 45.0)
    e.turn_done()


def _model_turn_between(e):
    assert e.on_answer({"type": "turn", "deg": -30}) == ("search", -30.0)
    e.turn_done()


def _approach_between(e):
    _pointed(e)
    assert e.on_goal_result("no_depth")[0] == "approach"
    assert e.on_move_result("reached", moved_m=1.0) == ("look",)


def _timeout_between(e):
    _pointed(e)
    assert e.on_goal_result("timeout") == ("search", 45.0)
    e.turn_done()


@pytest.mark.parametrize("between", [_turn_between, _model_turn_between, _approach_between, _timeout_between])
def test_only_a_block_straight_after_a_look_counts_twice(between):
    """A turn, a walk or another result between two blocks: the robot has
    moved or the picture changed, so the second block gets its own fresh
    look too."""
    e = Explorer()
    _pointed(e)
    assert e.on_goal_result("blocked", distance_m=1.8) == ("look",)
    between(e)
    _pointed(e)
    assert e.on_goal_result("blocked", distance_m=1.8) == ("look",)  # a fresh look, not a turn


def test_time_budget_gives_up_with_a_reason():
    e = Explorer(max_s=10.0)
    assert e.out_of_time(("look",), 9.9) == ("look",)
    kind, why = e.out_of_time(("search", 45.0), 10.0)
    assert kind == "gave_up" and "time budget" in why and "10 s" in why


def test_a_step_that_ends_the_task_late_keeps_its_result():
    e = Explorer(max_s=10.0)
    assert e.out_of_time(("done",), 50.0) == ("done",)
    late = ("gave_up", "no progress: 3 moves in a row covered less than 0.25 m each")
    assert e.out_of_time(late, 50.0) == late


def test_time_budget_default_matches_the_node():
    assert Explorer().max_s == 600.0


def test_gives_up_after_max_steps():
    e = Explorer(max_steps=2)
    e.on_answer({"type": "not_visible"})
    e.on_answer({"type": "not_visible"})
    kind, why = e.on_answer({"type": "not_visible"})
    assert kind == "gave_up" and "step budget" in why


@pytest.mark.parametrize("bad", [{"type": "unparsable"}, {"type": "goal"}, {"type": "goal", "point_2d": [2000, 0]}])
def test_garbage_answers_search(bad):
    assert Explorer().on_answer(bad) == ("search", 45.0)
