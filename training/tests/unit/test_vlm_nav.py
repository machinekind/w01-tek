"""Unit tests for the VLM navigation bridge.

Pure offline: no anthropic import, no room assets. The navigator loop runs
against a FakeSim/FakeClient pair via asyncio.run().
"""

import asyncio
import math
from types import SimpleNamespace

import pytest

from wojtek_rl.midlevel import parse_command
from wojtek_rl.vlm_nav import (
    MAX_FORWARD_M,
    MAX_STEPS,
    MAX_TURN_DEG,
    VlmDecision,
    VlmNavigator,
    build_messages,
    decision_to_command,
    parse_response,
    situation_text,
)

POSE = (0.0, 0.0, 0.0)


def dec(action, amount=None, reasoning="because"):
    return VlmDecision(action=action, amount=amount, reasoning=reasoning)


# --- decision_to_command --------------------------------------------------


def test_decision_to_command_maps_actions():
    assert decision_to_command(dec("turn_left", 30)) == "turn_left 30"
    assert decision_to_command(dec("turn_right", 45)) == "turn_right 45"
    assert decision_to_command(dec("forward", 0.5)) == "forward 0.5"
    assert decision_to_command(dec("stop")) == "stop"


def test_decision_to_command_done_is_terminal():
    assert decision_to_command(dec("done")) is None


def test_decision_to_command_clamps_forward():
    assert decision_to_command(dec("forward", 10)) == f"forward {MAX_FORWARD_M:g}"
    assert decision_to_command(dec("forward", 0.001)) == "forward 0.1"


def test_decision_to_command_backward_maps_and_clamps():
    from wojtek_rl.vlm_nav import MAX_BACKWARD_M

    assert decision_to_command(dec("backward", 0.3)) == "backward 0.3"
    assert decision_to_command(dec("backward", 5)) == f"backward {MAX_BACKWARD_M:g}"
    assert decision_to_command(dec("backward", 0.001)) == "backward 0.1"


def test_decision_to_command_clamps_turn():
    assert decision_to_command(dec("turn_left", 720)) == f"turn_left {MAX_TURN_DEG:g}"
    assert decision_to_command(dec("turn_right", 1)) == "turn_right 5"


@pytest.mark.parametrize("amount", [None, math.nan, math.inf])
def test_decision_to_command_rejects_bad_amount(amount):
    with pytest.raises(ValueError):
        decision_to_command(dec("forward", amount))


def test_decision_to_command_rejects_unknown_action():
    with pytest.raises(ValueError):
        decision_to_command(dec("fly", 3))


@pytest.mark.parametrize(
    "decision",
    [dec("turn_left", 30), dec("turn_right", 720), dec("forward", 10), dec("stop")],
)
def test_mapped_commands_pass_parse_command(decision):
    """Two-gate invariant: every mapped string must survive parse_command."""
    parse_command(decision_to_command(decision))


# --- build_messages ---------------------------------------------------------


def test_build_messages_contains_image_and_goal():
    msgs = build_messages("go to the bed", "abc123", [], 1, 20, POSE)
    assert len(msgs) == 1
    content = msgs[0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["data"] == "abc123"
    assert "go to the bed" in content[1]["text"]
    assert "Step 1 of 20" in content[1]["text"]


def test_build_messages_truncates_history():
    history = [{"cmd": f"forward {i}", "result": "completed"} for i in range(20)]
    msgs = build_messages("goal", "x", history, 21, 30, POSE)
    text = msgs[0]["content"][1]["text"]
    assert "forward 19" in text
    assert "forward 11" not in text  # only last 8 entries


# --- parse_response -----------------------------------------------------------


def tool_block(**inp):
    return SimpleNamespace(type="tool_use", name="navigate", input=inp)


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def test_parse_response_reads_tool_use():
    msg = SimpleNamespace(content=[tool_block(action="forward", amount=0.5, reasoning="bed ahead")])
    assert parse_response(msg) == dec("forward", 0.5, "bed ahead")


def test_parse_response_falls_back_to_text():
    msg = SimpleNamespace(content=[text_block("I will turn_left 30 to scan the room.")])
    d = parse_response(msg)
    assert d.action == "turn_left"
    assert d.amount == 30.0


def test_parse_response_raises_on_garbage():
    with pytest.raises(ValueError):
        parse_response(SimpleNamespace(content=[text_block("no command here")]))


# --- navigator loop -----------------------------------------------------------


class FakeExecutor:
    """Executor stub: `active` flips False after `ticks_per_cmd` polls."""

    def __init__(self, ticks_per_cmd=2):
        self.ticks_per_cmd = ticks_per_cmd
        self._left = 0

    def begin(self):
        self._left = self.ticks_per_cmd

    @property
    def active(self):
        if self._left > 0:
            self._left -= 1
            return True
        return False


class FakeSim:
    def __init__(self, ticks_per_cmd=2, stuck=False):
        self.executor = FakeExecutor(0 if stuck else ticks_per_cmd)
        self.stuck = stuck
        self.resets = 0
        self.submitted = []

    def ego_jpeg(self):
        return "ZmFrZQ=="

    def pose(self):
        return POSE

    def submit_command(self, text):
        self.submitted.append(text)
        try:
            parse_command(text)
        except ValueError as e:
            return {"ok": False, "error": str(e)}
        if not self.stuck:
            self.executor.begin()
        return {"ok": True, "command": text}


class StuckExecutor:
    active = True


class FakeClient:
    """Scripted decisions; records what it saw."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    async def decide(self, goal, ego_b64, history, step, max_steps, pose):
        self.calls.append({"goal": goal, "ego": ego_b64, "history": list(history)})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_nav(sim, client, **kw):
    kw.setdefault("poll_s", 0.001)
    kw.setdefault("cmd_timeout_s", 1.0)
    return VlmNavigator(sim, client, **kw)


def run(nav, goal="find the bed"):
    asyncio.run(nav._run(goal))


def test_loop_submits_one_command_per_step_and_terminates_on_done():
    sim = FakeSim()
    client = FakeClient([dec("turn_left", 30), dec("forward", 0.5), dec("done")])
    nav = make_nav(sim, client)
    run(nav)
    assert sim.submitted == ["turn_left 30", "forward 0.5"]
    assert nav.status()["state"] == "done"
    assert nav.status()["reason"] == "vlm_done"


def test_loop_stop_submits_stop_and_finishes():
    sim = FakeSim()
    client = FakeClient([dec("stop")])
    nav = make_nav(sim, client)
    run(nav)
    assert sim.submitted == ["stop"]
    assert nav.status()["reason"] == "vlm_stop"


def test_loop_stops_at_max_steps():
    sim = FakeSim()
    client = FakeClient([dec("forward", 0.3)] * 3)
    nav = make_nav(sim, client, max_steps=3)
    run(nav)
    assert len(sim.submitted) == 3
    assert nav.status()["reason"] == "max_steps"


def test_loop_reports_error_after_repeated_failures():
    sim = FakeSim()
    client = FakeClient([RuntimeError("api down")] * 5)
    nav = make_nav(sim, client)
    run(nav)
    assert nav.status()["state"] == "error"
    assert "api down" in nav.status()["error"]
    assert sim.submitted == []


def test_loop_recovers_from_single_failure():
    sim = FakeSim()
    client = FakeClient([RuntimeError("blip"), dec("done")])
    nav = make_nav(sim, client)
    run(nav)
    assert nav.status()["state"] == "done"
    # The failure is visible in the history the client saw on the next call.
    assert any("failed" in h["result"] for h in client.calls[1]["history"])


def test_stuck_command_times_out_stops_and_loop_continues():
    """A blocked command must not kill the episode: stop the robot, tell the
    VLM in history, keep going."""
    sim = FakeSim(stuck=True)
    sim.executor = StuckExecutor()
    client = FakeClient([dec("forward", 0.5), dec("done")])
    nav = make_nav(sim, client, cmd_timeout_s=0.01)
    run(nav)
    assert sim.submitted == ["forward 0.5", "stop"]
    assert nav.status()["state"] == "done"
    assert any("blocked" in h["result"] for h in client.calls[1]["history"])


def test_fall_reset_recorded_in_history():
    sim = FakeSim()

    class FallingClient(FakeClient):
        async def decide(self, goal, ego_b64, history, step, max_steps, pose):
            if step == 2:
                assert any("interrupted" in h["result"] for h in history)
            return await super().decide(goal, ego_b64, history, step, max_steps, pose)

    client = FallingClient([dec("forward", 0.5), dec("done")])
    orig = sim.submit_command

    def falling_submit(text):
        sim.resets += 1  # robot fell while executing
        return orig(text)

    sim.submit_command = falling_submit
    nav = make_nav(sim, client)
    run(nav)
    assert nav.status()["state"] == "done"


def test_start_rejects_second_goal_and_empty_goal():
    async def scenario():
        sim = FakeSim()
        client = FakeClient([dec("forward", 0.5), dec("done")])
        nav = make_nav(sim, client)
        assert nav.start("")["ok"] is False
        ack = nav.start("find the bed")
        assert ack["ok"] is True
        assert nav.start("another")["ok"] is False
        await nav._task
        return nav

    nav = asyncio.run(scenario())
    assert nav.status()["state"] == "done"


def test_cancel_returns_to_idle():
    async def scenario():
        sim = FakeSim()

        class SlowClient:
            async def decide(self, *a, **kw):
                await asyncio.sleep(10)

        nav = make_nav(sim, SlowClient())
        nav.start("find the bed")
        await asyncio.sleep(0.01)
        nav.cancel("user")
        await asyncio.sleep(0.01)
        return nav, sim

    nav, sim = asyncio.run(scenario())
    assert nav.status()["state"] == "idle"
    assert not nav.running
    assert sim.submitted[-1] == "stop"


# -- no step budget (interactive demo) -------------------------------------


class BlockingExecutor:
    """Every command comes back blocked: the wedged robot.

    `blocked` rises when the command FINISHES, like the real executors --
    the navigator samples the counter after submitting, so a fake that
    increments on submit would look like no change at all.
    """

    def __init__(self):
        self.blocked = 0
        self._left = 0
        self._running = False

    def begin(self):
        self._left = 1
        self._running = True

    @property
    def active(self):
        if self._left > 0:
            self._left -= 1
            return True
        if self._running:
            self._running = False
            self.blocked += 1
        return False


def test_max_steps_none_runs_past_the_default_budget():
    """A benchmark caps steps for comparable episodes; interactively the cap
    just guillotines a route mid-way, so None means 'run until it resolves'."""
    sim = FakeSim()
    script = [dec("forward", 0.3) for _ in range(MAX_STEPS + 5)] + [dec("done")]
    nav = make_nav(sim, FakeClient(script), max_steps=None)
    run(nav)
    assert len(sim.submitted) == MAX_STEPS + 5  # would have stopped at MAX_STEPS
    assert nav.status()["reason"] == "vlm_done"
    assert nav.status()["max_steps"] is None


def test_uncapped_run_still_stops_when_wedged():
    sim = FakeSim()
    sim.executor = BlockingExecutor()
    nav = make_nav(sim, FakeClient([dec("forward", 0.3) for _ in range(50)]), max_steps=None)
    run(nav)
    assert nav.status()["state"] == "done"
    assert nav.status()["reason"] == "stuck"
    assert sim.submitted[-1] == "stop"
    assert len(sim.submitted) < 20  # gave up on evidence, not on a counter


def test_uncapped_run_still_stops_when_spinning():
    sim = FakeSim()
    nav = make_nav(
        sim, FakeClient([dec("turn_left", 30) for _ in range(60)]),
        max_steps=None, max_rotation=14,
    )
    run(nav)
    assert nav.status()["reason"] == "max_rotation"


def test_situation_text_omits_the_budget_when_uncapped():
    capped = situation_text("go", [], 3, 20, POSE)
    uncapped = situation_text("go", [], 3, None, POSE)
    assert "Step 3 of 20." in capped
    assert "Step 3." in uncapped and " of " not in uncapped.splitlines()[1]
