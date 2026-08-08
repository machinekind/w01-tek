"""SearchController FSM + ValueMap, driven end-to-end with a fake sim.

Mirrors test_vlm_nav.py's approach: the fake executor completes every command
instantly (teleport kinematics), so the whole FSM runs in milliseconds under
asyncio.run() with a scripted observer standing in for the VLM.
"""

import asyncio
import math

import numpy as np

from wojtek_eval.mapping import FREE, OnlineMap
from wojtek_rl.agent.search import (
    SearchController,
    ValueMap,
    ViewScore,
    parse_view_score,
)

# -- ValueMap ---------------------------------------------------------------


def make_value_map():
    return ValueMap(res=0.1, origin=(-2.0, -2.0), shape=(40, 40), hfov_deg=90.0, max_range=3.0)


def test_splat_scores_ahead_not_behind():
    vm = make_value_map()
    vm.splat(0.0, 0.0, 0.0, score=8.0)  # facing +x
    assert vm.value_near(1.0, 0.0, 0.2) > 6.0   # straight ahead
    assert vm.value_near(-1.0, 0.0, 0.2) == 0.0  # behind the cone
    assert vm.confidence_near(-1.0, 0.0, 0.2) == 0.0


def test_splat_axis_beats_cone_edge():
    vm = make_value_map()
    vm.splat(0.0, 0.0, 0.0, score=8.0)
    on_axis = vm.confidence_near(1.5, 0.0, 0.1)
    near_edge = vm.confidence_near(1.5, 1.4, 0.1)  # ~43 deg off-axis, hfov/2=45
    assert on_axis > near_edge > 0.0


def test_fusion_weighted_average():
    vm = make_value_map()
    vm.splat(0.0, 0.0, 0.0, score=8.0)
    vm.splat(0.0, 0.0, 0.0, score=2.0)  # same viewpoint, contradicting score
    v = vm.value_near(1.0, 0.0, 0.1)
    assert 2.0 < v < 8.0


def test_decay_reduces_confidence():
    vm = make_value_map()
    vm.splat(0.0, 0.0, 0.0, score=5.0)
    c0 = vm.confidence_near(1.0, 0.0, 0.2)
    vm.decay(0.5)
    assert vm.confidence_near(1.0, 0.0, 0.2) < c0


def test_parse_view_score_tolerates_garbage():
    vs = parse_view_score({"description": "a sofa", "target_visible": True,
                           "score": "7", "bbox_2d": [100, 100, 400, 500]})
    assert vs.visible and vs.score == 7.0 and vs.bbox == (100.0, 100.0, 400.0, 500.0)
    vs = parse_view_score({"score": "high?", "bbox_2d": "nope"})
    assert not vs.visible and vs.score == 0.0 and vs.bbox is None


# -- fake sim ---------------------------------------------------------------


class FakeExecutor:
    """Commands complete instantly; goto teleports next to the goal."""

    def __init__(self, sim):
        self.sim = sim
        self.blocked = 0
        self.active = False

    def goto(self, x, y, from_xy):
        self.sim.x, self.sim.y = x, y

    def status(self):
        return {"active": None, "remaining": 0.0, "queued": 0}


class FakeSim:
    def __init__(self, omap):
        self.omap = omap
        self.x = self.y = self.yaw = 0.0
        self.resets = 0
        self.executor = FakeExecutor(self)
        self.commands: list[str] = []

    def pose(self):
        return (self.x, self.y, self.yaw)

    def submit_command(self, text):
        self.commands.append(text)
        parts = text.split()
        if parts[0] == "turn_left":
            self.yaw += math.radians(float(parts[1]))
        elif parts[0] == "turn_right":
            self.yaw -= math.radians(float(parts[1]))
        elif parts[0] == "forward":
            d = float(parts[1])
            self.x += d * math.cos(self.yaw)
            self.y += d * math.sin(self.yaw)
        return {"ok": True, "command": text}

    def ego_jpeg(self):
        return "fakejpegb64"


def closed_map():
    """Fully-explored map: no frontiers anywhere."""
    m = OnlineMap(res=0.1, origin=(-2.0, -2.0), shape=(40, 40))
    m.state[:] = FREE
    return m


def open_map():
    """Free block away from the robot with unknown beyond: has frontiers,
    and their midpoint (~(-1, -1)) is a real drive from the origin."""
    m = OnlineMap(res=0.1, origin=(-2.0, -2.0), shape=(40, 40))
    m.state[5:15, 5:15] = FREE
    return m


def make_controller(sim, score_view):
    return SearchController(
        sim, score_view,
        poll_s=0.001, verify_pause_s=0.0, detect_period_s=0.001,
        cmd_timeout_s=1.0, max_wall_s=5.0,
    )


def run_search(ctrl, target="ball"):
    async def scenario():
        ack = ctrl.start(target)
        assert ack["ok"], ack
        await ctrl._task
        return ctrl

    return asyncio.run(scenario())


# -- FSM scenarios --------------------------------------------------------------


def test_found_on_first_look():
    sim = FakeSim(closed_map())
    big_centered = ViewScore("a red ball", True, 9.0, bbox=(300, 300, 700, 700))

    async def score_view(target, frame):
        return big_centered

    ctrl = run_search(make_controller(sim, score_view))
    st = ctrl.status()
    assert st["state"] == "found"
    assert st["found_xy"] is not None
    assert sim.commands[-1] == "stop"


def test_not_found_when_map_closed_and_nothing_seen():
    sim = FakeSim(closed_map())

    async def score_view(target, frame):
        return ViewScore("empty floor", False, 0.0)

    ctrl = run_search(make_controller(sim, score_view))
    assert ctrl.status()["state"] == "not_found"
    # Full initial scan happened: 7 turns of 45 deg.
    assert sum(1 for c in sim.commands if c.startswith("turn_left 45")) == 7


def test_failed_verify_blacklists_and_continues():
    sim = FakeSim(closed_map())
    calls = {"n": 0}

    async def score_view(target, frame):
        calls["n"] += 1
        # First observation: convincing detection; everything after: nothing
        # (the phantom-detection case verify exists for).
        if calls["n"] == 1:
            return ViewScore("maybe a ball", True, 8.0, bbox=(350, 350, 750, 750))
        return ViewScore("nothing here", False, 0.0)

    ctrl = run_search(make_controller(sim, score_view))
    st = ctrl.status()
    assert st["state"] == "not_found"
    assert st["blacklisted"] == 1


def test_frontier_drive_then_found():
    sim = FakeSim(open_map())
    calls = {"n": 0}

    async def score_view(target, frame):
        calls["n"] += 1
        # Invisible during the initial 8-stop scan; found after the robot
        # commits to a frontier and drives there.
        if calls["n"] <= 8:
            return ViewScore("walls", False, 1.0)
        return ViewScore("the ball!", True, 9.0, bbox=(300, 300, 700, 700))

    ctrl = run_search(make_controller(sim, score_view))
    assert ctrl.status()["state"] == "found"
    moved = math.hypot(sim.x, sim.y)
    assert moved > 0.3  # actually committed to a frontier


def test_cancel_mid_search():
    sim = FakeSim(open_map())
    started = asyncio.Event()

    async def score_view(target, frame):
        started.set()
        await asyncio.sleep(10)  # park the search inside an observation
        return ViewScore("", False, 0.0)

    async def scenario():
        ctrl = make_controller(sim, score_view)
        ctrl.start("ball")
        await started.wait()
        ctrl.cancel("test")
        try:
            await ctrl._task
        except asyncio.CancelledError:
            pass
        return ctrl

    ctrl = asyncio.run(scenario())
    assert ctrl.status()["state"] == "idle"
    assert not ctrl.running


def test_event_log_traces_the_state_machine():
    """Debug events carry commands, per-observation scores, and frontier
    selection -- the UI's state-machine panel renders exactly these."""
    sim = FakeSim(open_map())
    calls = {"n": 0}

    async def score_view(target, frame):
        calls["n"] += 1
        if calls["n"] <= 8:
            return ViewScore("walls", False, 1.0)
        return ViewScore("the ball!", True, 9.0, bbox=(300, 300, 700, 700))

    ctrl = run_search(make_controller(sim, score_view))
    texts = [e["text"] for e in ctrl._events]
    assert any(t.startswith("cmd turn_left 45") for t in texts)
    assert any(t.startswith("obs #1 ") and "score 1" in t for t in texts)
    assert any("VISIBLE" in t and "bbox=" in t for t in texts)
    assert any(t.startswith("frontiers:") for t in texts)
    assert any(t.startswith("goto (") for t in texts)
    assert any(t.startswith("verify:") for t in texts)
    # Every event is tagged with the FSM state it happened in.
    assert {e["state"] for e in ctrl._events} >= {"scanning", "verifying"}
    # status() exposes the tail for the UI.
    st = ctrl.status()
    assert st["events"] == ctrl._events[-20:]
    assert st["attempted"] == 1


def test_observer_uses_injected_frame_fn():
    """The observer must score the BARE camera: with the HUD composited in,
    a model that was never told about the inset detects the target inside its
    own minimap (measured in a live session -- bbox landed exactly on the
    paste rectangle, and every approach then 'lost' the candidate)."""
    sim = FakeSim(closed_map())
    sim.ego_jpeg = lambda: "HUD_FRAME"
    seen = []

    async def score_view(target, frame):
        seen.append(frame)
        return ViewScore("nothing", False, 0.0)

    ctrl = make_controller(sim, score_view)
    ctrl.frame_fn = lambda: "CLEAN_FRAME"
    run_search(ctrl)
    assert seen and set(seen) == {"CLEAN_FRAME"}


def test_frame_fn_defaults_to_sim_ego_jpeg():
    sim = FakeSim(closed_map())
    ctrl = SearchController(sim, score_view=None)
    assert ctrl.frame_fn == sim.ego_jpeg


def test_events_go_to_trace_with_pose_and_state():
    from wojtek_rl.agent.trace import Trace

    sim = FakeSim(closed_map())
    trace = Trace(None)

    async def score_view(target, frame):
        return ViewScore("empty", False, 0.0)

    ctrl = make_controller(sim, score_view)
    ctrl.trace = trace
    run_search(ctrl, target="ball")
    kinds = {e["kind"] for e in trace.recent(limit=500)}
    assert "search.scanning" in kinds
    events = trace.recent(limit=500)
    assert all(e["target"] == "ball" for e in events)
    assert all(len(e["pose"]) == 3 for e in events)
    assert any(e["text"].startswith("obs #1 ") for e in events)


def test_start_rejects_while_running():
    sim = FakeSim(open_map())

    async def score_view(target, frame):
        await asyncio.sleep(10)
        return ViewScore("", False, 0.0)

    async def scenario():
        ctrl = make_controller(sim, score_view)
        assert ctrl.start("ball")["ok"]
        await asyncio.sleep(0.01)
        second = ctrl.start("mug")
        ctrl.cancel("test")
        try:
            await ctrl._task
        except asyncio.CancelledError:
            pass
        return second

    assert asyncio.run(scenario())["ok"] is False
