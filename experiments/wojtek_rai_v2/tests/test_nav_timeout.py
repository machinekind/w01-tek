"""The Nav2 goal timeout and cancel path of `_TimedNavMixin`. Model-free:
the action client, goal handle and futures are fakes; no node, no DDS."""

import pytest

from wojtek_rai import limits

rai = pytest.importorskip("rai")

from wojtek_rai import nav_tools  # noqa: E402
from wojtek_rai.nav_tools import _TimedNavMixin, cancel_active_nav_goal  # noqa: E402


class _Future:
    """rclpy.task.Future stand-in: `add_done_callback` fires only if `done`."""

    def __init__(self, result=None, done=True):
        self._result, self._done = result, done

    def add_done_callback(self, cb):
        if self._done:
            cb(self)

    def result(self):
        return self._result


class _GoalHandle:
    accepted = True

    def __init__(self, result_future):
        self._result_future = result_future
        self.cancelled = 0

    def get_result_async(self):
        return self._result_future

    def cancel_goal_async(self):
        self.cancelled += 1
        return _Future()


class _ActionClient:
    def __init__(self, goal_handle):
        self.goal_handle = goal_handle

    def wait_for_server(self, timeout_sec=None):
        return True

    def send_goal_async(self, goal):
        return _Future(result=self.goal_handle)


class _Stamp:
    def to_msg(self):
        from builtin_interfaces.msg import Time

        return Time()


class _Node:
    def get_clock(self):
        return self

    def now(self):
        return _Stamp()


class _Connector:
    node = _Node()


class _Tool(_TimedNavMixin):
    connector = _Connector()
    frame_id = limits.MAP_FRAME
    action_name = limits.NAV_ACTION
    writable = True

    def is_writable(self, name: str) -> bool:  # BaseROS2Tool's permission check
        return self.writable


def test_non_writable_action_is_refused(monkeypatch):
    monkeypatch.setattr(nav_tools, "ActionClient", lambda *a, **k: pytest.fail("no client"))
    tool = _Tool()
    tool.writable = False
    assert "not writable" in tool._navigate(0.0, 0.0, 0.0)


@pytest.fixture
def fast_timeouts(monkeypatch):
    monkeypatch.setattr(limits, "NAV_GOAL_TIMEOUT_S", 0.05)
    monkeypatch.setattr(nav_tools, "NAV_SERVER_TIMEOUT_S", 0.05)


def test_never_completing_result_times_out_and_cancels_the_goal(monkeypatch, fast_timeouts):
    handle = _GoalHandle(_Future(done=False))
    monkeypatch.setattr(nav_tools, "ActionClient", lambda *a, **k: _ActionClient(handle))

    out = _Tool()._navigate(1.0, 0.0, 0.0)

    assert "timed out" in out and "cancelled" in out
    assert handle.cancelled == 1
    assert nav_tools._ACTIVE["handle"] is None  # cleared in the finally


def test_cancel_active_nav_goal_is_a_noop_when_nothing_is_active():
    nav_tools._ACTIVE["handle"] = None
    assert cancel_active_nav_goal() is False


def test_cancel_active_nav_goal_cancels_the_handle_in_flight():
    handle = _GoalHandle(_Future(done=False))
    nav_tools._ACTIVE["handle"] = handle
    try:
        assert cancel_active_nav_goal() is True
        assert handle.cancelled == 1
    finally:
        nav_tools._ACTIVE["handle"] = None


def test_goal_outside_the_workspace_is_refused_before_any_client_is_made(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("ActionClient must not be created")

    monkeypatch.setattr(nav_tools, "ActionClient", boom)
    out = _Tool()._navigate(limits.WORKSPACE_MAX[0] + 1.0, 0.0, 0.0)
    assert "outside the workspace" in out


def test_missing_action_server_returns_a_hint(monkeypatch, fast_timeouts):
    class _NoServer(_ActionClient):
        def wait_for_server(self, timeout_sec=None):
            return False

    monkeypatch.setattr(nav_tools, "ActionClient", lambda *a, **k: _NoServer(None))
    assert "wojtek_nav" in _Tool()._navigate(0.0, 0.0, 0.0)
