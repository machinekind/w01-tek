"""cmd_vel watchdog stop paths, driven with a stub node. Model-free: no rclpy
context, the methods are called unbound on a SimpleNamespace."""

from types import SimpleNamespace

import pytest

rclpy = pytest.importorskip("rclpy")

from geometry_msgs.msg import Twist  # noqa: E402
from rclpy.time import Time  # noqa: E402

from wojtek_rai.nav import cmd_vel_watchdog as wd  # noqa: E402

W = wd.CmdVelWatchdog


class _Pub:
    def __init__(self):
        self.msgs = []

    def publish(self, msg):
        self.msgs.append(msg)


class _Clock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return Time(seconds=self.t)


class _Log:
    def __init__(self):
        self.lines = []

    def warn(self, s):
        self.lines.append(s)


def _stub(burst=True, timeout=0.5, hb_timeout=1.0):
    clock, pub, log = _Clock(), _Pub(), _Log()
    node = SimpleNamespace(
        _pub=pub,
        _timeout=timeout,
        _hb_timeout=hb_timeout,
        _burst_enabled=burst,
        _last=None,
        _stopped=True,
        _hb_last=None,
        _hb_lost=False,
        _burst_left=0,
        get_clock=lambda: clock,
        get_logger=lambda: log,
    )
    node._check_heartbeat = lambda: W._check_heartbeat(node)
    return node, clock, pub, log


def _is_zero(msg):
    return (msg.linear.x, msg.linear.y, msg.angular.z) == (0.0, 0.0, 0.0)


def _forward(vx=0.3):
    t = Twist()
    t.linear.x = vx
    return t


def test_send_stop_publishes_stop_repeats_zero_twists(monkeypatch):
    monkeypatch.setattr(wd.time, "sleep", lambda _s: None)
    node, _, pub, _ = _stub()

    W.send_stop(node)

    assert len(pub.msgs) == wd.STOP_REPEATS
    assert all(_is_zero(m) for m in pub.msgs)


def test_nav_silence_sends_one_stop_then_goes_quiet():
    node, clock, pub, log = _stub()
    W._on_cmd(node, _forward())
    assert not node._stopped

    clock.t = 0.6
    W._tick(node)
    W._tick(node)

    assert len(pub.msgs) == 2 and _is_zero(pub.msgs[-1])
    assert node._stopped and len(log.lines) == 1


def test_heartbeat_loss_enters_stopped_state_and_logs():
    node, clock, pub, log = _stub()
    W._on_heartbeat(node, None)
    W._on_cmd(node, _forward())
    clock.t = 0.2
    W._on_cmd(node, _forward())

    clock.t = 1.3  # heartbeat 1.3 s old, nav command 1.1 s old
    W._tick(node)

    assert node._hb_lost and node._stopped
    assert _is_zero(pub.msgs[-1])
    assert any("heartbeat" in s for s in log.lines)
    # Only the heartbeat stop went out: the stopped state suppresses a second one.
    assert len(pub.msgs) == 3


def test_heartbeat_return_bursts_zero_twists_at_the_tick_rate():
    node, clock, pub, log = _stub(burst=True)
    W._on_heartbeat(node, None)
    clock.t = 1.5
    W._tick(node)
    assert node._hb_lost
    before = len(pub.msgs)

    W._on_heartbeat(node, None)
    assert not node._hb_lost and node._burst_left == wd.STOP_BURST_TICKS
    for _ in range(wd.STOP_BURST_TICKS + 3):
        clock.t += wd.TICK_S
        W._on_heartbeat(node, None)
        W._tick(node)

    burst = pub.msgs[before:]
    assert len(burst) == wd.STOP_BURST_TICKS
    assert all(_is_zero(m) for m in burst)
    assert node._burst_left == 0
    assert any("bursting stop" in s for s in log.lines)


def test_heartbeat_return_without_burst_sends_nothing():
    node, clock, pub, log = _stub(burst=False)
    W._on_heartbeat(node, None)
    clock.t = 1.5
    W._tick(node)
    before = len(pub.msgs)

    W._on_heartbeat(node, None)
    for _ in range(3):
        clock.t += wd.TICK_S
        W._tick(node)

    assert len(pub.msgs) == before and node._burst_left == 0
    assert any("disabled" in s for s in log.lines)


def test_stop_burst_covers_one_second():
    assert wd.STOP_BURST_TICKS * wd.TICK_S == pytest.approx(wd.STOP_BURST_S)
