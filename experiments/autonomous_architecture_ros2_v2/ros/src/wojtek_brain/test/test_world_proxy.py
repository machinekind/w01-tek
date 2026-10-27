"""WorldProxy protocol conformance, no ROS graph and no wojtek_rl import.

The proxy's whole job is to look exactly like RoomSim to the controllers, so
these tests exercise the nine-member surface with fakes on both ends: a
recorded command function below, an injected map factory instead of
wojtek_eval's OnlineMap, and a list-backed pose history."""

import base64

import pytest

from wojtek_brain import world_proxy
from wojtek_brain.world_proxy import WorldProxy


class FakeHistory:
    def __init__(self):
        self.samples = []

    def add(self, t, x, y, yaw):
        self.samples.append((t, x, y, yaw))


class FakeMap:
    def __init__(self, res, origin, shape):
        self.res, self.origin, self.shape = res, origin, shape
        self.state = None
        self.trail = []


def make_proxy(calls):
    def command_fn(kind, text, args):
        calls.append((kind, text, args))
        return {"ok": True, "command": text or kind,
                "cmd_seq": len(calls)}

    return WorldProxy(command_fn, pose_history=FakeHistory(),
                      map_factory=FakeMap)


def test_status_feeds_pose_executor_and_history():
    proxy = make_proxy([])
    proxy.on_status(1.0, 2.0, 0.5, True, 3, 1, 12.5, '{"queued": 2}')
    assert proxy.pose() == (1.0, 2.0, 0.5)
    assert proxy.executor.active is True
    assert proxy.executor.blocked == 3
    assert proxy.resets == 1
    assert proxy.sim_time == 12.5
    assert proxy.executor.status() == {"queued": 2}
    assert proxy.pose_history.samples == [(12.5, 1.0, 2.0, 0.5)]


def test_garbage_exec_json_degrades_to_empty_status():
    proxy = make_proxy([])
    proxy.on_status(0, 0, 0, False, 0, 0, 0.0, "not json")
    assert proxy.executor.status() == {}


def test_commands_go_through_the_one_door():
    calls = []
    proxy = make_proxy(calls)
    ack = proxy.submit_command("forward 0.6")
    assert ack["ok"] and calls[-1] == ("midlevel", "forward 0.6", [])
    proxy.executor.goto(1.5, -2.0, (0.0, 0.0))
    assert calls[-1] == ("goto", "", [1.5, -2.0])


def test_map_message_rebuilds_a_real_grid():
    proxy = make_proxy([])
    state = bytes([0, 1, 2, 1, 0, 2])  # 2x3
    proxy.on_map(0.05, (-1.0, -2.0), (2, 3), state,
                 [0.1, 0.2, 0.3, 0.4], 90.0)
    assert proxy._ego_fovy == 90.0
    assert proxy.omap.shape == (2, 3)
    assert proxy.omap.state.tolist() == [[0, 1, 2], [1, 0, 2]]
    assert proxy.omap.trail == [(pytest.approx(0.1), pytest.approx(0.2)),
                                (pytest.approx(0.3), pytest.approx(0.4))]
    # A second message with the same shape reuses the map object.
    first = proxy.omap
    proxy.on_map(0.05, (-1.0, -2.0), (2, 3), bytes(6), [], 90.0)
    assert proxy.omap is first
    assert proxy.omap.state.sum() == 0


def test_frames_wait_then_serve_base64(monkeypatch):
    monkeypatch.setattr(world_proxy, "FRAME_WAIT_S", 0.01)
    proxy = make_proxy([])
    with pytest.raises(RuntimeError):
        proxy.ego_jpeg(hud=False)
    proxy.on_ego_jpeg(b"\xff\xd8jpegbytes")
    assert base64.b64decode(proxy.ego_jpeg()) == b"\xff\xd8jpegbytes"
    with pytest.raises(RuntimeError):
        proxy.vlm_frame_jpeg()
    proxy.on_vln_jpeg(b"vln")
    assert base64.b64decode(proxy.vlm_frame_jpeg()) == b"vln"


def test_active_is_sequence_fenced():
    """An accepted command counts as running until a status snapshot that
    already reflects it arrives -- the race that machine-gunned commands."""
    proxy = make_proxy([])
    proxy.on_status(0, 0, 0, False, 0, 0, 1.0, "{}", cmd_seq=0)
    assert proxy.executor.active is False
    proxy.submit_command("forward 0.6")            # ack seq 1
    assert proxy.executor.active is True           # despite raw False
    proxy.on_status(0, 0, 0, True, 0, 0, 1.1, "{}", cmd_seq=1)
    assert proxy.executor.active is True           # genuinely running
    proxy.on_status(0.6, 0, 0, False, 0, 0, 2.5, "{}", cmd_seq=1)
    assert proxy.executor.active is False          # done, snapshot is current


def test_rejected_command_does_not_fence():
    calls = []

    def command_fn(kind, text, args):
        calls.append(kind)
        return {"ok": False, "error": "blocked", "cmd_seq": 0}

    from wojtek_brain.world_proxy import WorldProxy
    proxy = WorldProxy(command_fn, pose_history=FakeHistory(),
                       map_factory=FakeMap)
    proxy.on_status(0, 0, 0, False, 0, 0, 1.0, "{}", cmd_seq=0)
    proxy.submit_command("forward 9")
    assert proxy.executor.active is False
