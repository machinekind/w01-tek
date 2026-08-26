"""Websocket protocol translation and the world-command dispatch, sim faked."""

import pytest

from wojtek_sim_bridge.protocol import (
    SayAccumulator,
    parse_client_message,
    world_command_result,
)


def test_parse_client_message_shapes():
    assert parse_client_message('{"type":"voice","on":true}') == ("voice", {"on": True})
    assert parse_client_message('{"type":"command","text":"forward 1"}') == (
        "command", {"text": "forward 1"})
    assert parse_client_message('{"type":"command","text":"  "}')[0] == "invalid"
    assert parse_client_message('{"type":"reset"}') == ("reset", {})
    assert parse_client_message("not json")[0] == "invalid"
    assert parse_client_message('{"type":"overlay"}')[0] == "unknown"


def test_say_accumulator_streams_and_closes():
    acc = SayAccumulator()
    events = acc.feed("u1", "Dobrze.", False, "qwen")
    assert events == [{"type": "say", "text": "Dobrze.", "source": "qwen"}]
    events = acc.feed("u1", "Idę tam!", True, "qwen")
    assert events[0]["type"] == "say"
    assert events[1] == {"type": "chat_reply", "ok": True,
                         "say": "Dobrze. Idę tam!", "steps": [], "spoken": True}
    # Closed utterances leave no residue.
    assert acc.feed("u1", "x", True, "qwen")[1]["say"] == "x"


def test_say_accumulator_empty_final_closes_silently():
    acc = SayAccumulator()
    assert acc.feed("u2", "", True, "qwen") == []


def test_say_accumulator_barge_in_drops_partial():
    acc = SayAccumulator()
    acc.feed("u3", "Pierwsza część,", False, "qwen")
    acc.drop()
    assert acc.feed("u3", "koniec.", True, "qwen")[1]["say"] == "koniec."


class FakeExecutor:
    def __init__(self, with_goto=True):
        self.goto_calls = []
        if not with_goto:
            self.goto = None

    def goto(self, x, y, from_xy):
        self.goto_calls.append((x, y, from_xy))


class FakeSim:
    def __init__(self, with_goto=True):
        self.executor = FakeExecutor(with_goto)
        self.resets_called = 0
        self.commands = []

    def pose(self):
        return (1.0, 2.0, 0.0)

    def submit_command(self, text):
        self.commands.append(text)
        return {"ok": True, "command": text}

    def reset(self):
        self.resets_called += 1


def test_world_command_midlevel_passthrough():
    sim = FakeSim()
    out = world_command_result("midlevel", "turn_left 45", [], sim)
    assert out["ok"] and sim.commands == ["turn_left 45"]


def test_world_command_goto_uses_current_pose():
    sim = FakeSim()
    out = world_command_result("goto", "", [3.0, -1.0], sim)
    assert out["ok"]
    assert sim.executor.goto_calls == [(3.0, -1.0, (1.0, 2.0))]


def test_world_command_goto_without_planner_is_refused():
    sim = FakeSim(with_goto=False)
    out = world_command_result("goto", "", [1.0, 1.0], sim)
    assert not out["ok"] and "planner" in out["error"]


def test_world_command_reset_and_unknown():
    sim = FakeSim()
    assert world_command_result("reset", "", [], sim)["ok"]
    assert sim.resets_called == 1
    assert not world_command_result("teleport", "", [], sim)["ok"]
