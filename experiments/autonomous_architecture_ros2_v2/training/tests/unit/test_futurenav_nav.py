"""Unit tests for the FutureNav backend: action mapping + episode protocol.

Pure logic on a fake HTTP layer -- no server, no torch, so these run
everywhere.
"""

import asyncio

import pytest

from wojtek_rl.futurenav_nav import (
    FORWARD_STEP_M,
    TURN_STEP_DEG,
    FutureNavVlmClient,
    decision_from_action,
)
from wojtek_rl.midlevel import parse_command
from wojtek_rl.vlm_nav import decision_to_command


# -- decision_from_action -----------------------------------------------------


def test_move_forward_maps_to_forward_step():
    d = decision_from_action("MOVE_FORWARD", "MOVE_FORWARD")
    assert d.action == "forward"
    assert d.amount == FORWARD_STEP_M


def test_turns_map_to_turn_step():
    left = decision_from_action("TURN_LEFT", "TURN_LEFT")
    right = decision_from_action("TURN_RIGHT", "TURN_RIGHT")
    assert (left.action, left.amount) == ("turn_left", TURN_STEP_DEG)
    assert (right.action, right.amount) == ("turn_right", TURN_STEP_DEG)


def test_stop_means_done_not_abort():
    d = decision_from_action("STOP", "STOP")
    assert d.action == "done"
    assert d.amount is None


def test_unknown_action_raises():
    with pytest.raises(ValueError):
        decision_from_action("JUMP", "JUMP over the couch")


def test_raw_text_lands_in_reasoning():
    d = decision_from_action("MOVE_FORWARD", " MOVE_FORWARD because hallway ")
    assert "MOVE_FORWARD because hallway" in d.reasoning


def test_mapped_decision_survives_the_full_gate():
    """FutureNav action -> decision -> clamp -> parse_command: whole path."""
    d = decision_from_action("MOVE_FORWARD", "MOVE_FORWARD")
    cmd = decision_to_command(d)
    assert cmd == "forward 0.25"
    parse_command(cmd)  # must not raise


# -- FutureNavVlmClient episode protocol --------------------------------------


class FakeHttp:
    """Records posts; scripted /act responses."""

    def __init__(self, actions):
        self.actions = list(actions)
        self.posts = []

    def __call__(self, path, payload):
        self.posts.append((path, payload))
        if path == "/reset":
            return {"ok": True}
        action = self.actions.pop(0)
        return {"action": action, "raw": action, "step": len(self.posts)}


def _decide(client, goal, step):
    return asyncio.run(client.decide(goal, "b64frame", [], step, 80, (0.0, 0.0, 0.0)))


def test_resets_once_then_streams_frames():
    fake = FakeHttp(["MOVE_FORWARD", "TURN_LEFT"])
    client = FutureNavVlmClient("http://fake:1")
    client._post = fake

    d1 = _decide(client, "go to the bed", step=1)
    d2 = _decide(client, "go to the bed", step=2)

    assert [p[0] for p in fake.posts] == ["/reset", "/act", "/act"]
    assert fake.posts[0][1] == {"instruction": "go to the bed"}
    assert fake.posts[1][1] == {"frame_b64": "b64frame"}
    assert (d1.action, d2.action) == ("forward", "turn_left")


def test_new_goal_triggers_new_reset():
    fake = FakeHttp(["MOVE_FORWARD", "MOVE_FORWARD"])
    client = FutureNavVlmClient("http://fake:1")
    client._post = fake

    _decide(client, "go to the bed", step=1)
    _decide(client, "go to the window", step=1)

    resets = [p for p in fake.posts if p[0] == "/reset"]
    assert [p[1]["instruction"] for p in resets] == ["go to the bed", "go to the window"]


def test_trailing_slash_stripped():
    client = FutureNavVlmClient("http://fake:1/")
    assert client.url == "http://fake:1"
