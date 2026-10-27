"""The page server's command path against a mock client. No ROS runtime.

What it guards: Disarm and Lie down stop the brain before their service
call, whatever the call answers, and even when the page closes while the
call runs; no other button stops it. Needs rclpy importable (the
wojtek_vlm_gui container); skipped elsewhere.
"""

import asyncio
import threading
from unittest.mock import MagicMock

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("starlette")

from wojtek_vlm_gui import server, wire  # noqa: E402

STOPS_THE_BRAIN = {"disarm", "lie_down"}


@pytest.mark.parametrize("name", sorted(STOPS_THE_BRAIN))
def test_disarm_and_lie_down_stop_the_brain_before_the_call(monkeypatch, name):
    order = []
    client = MagicMock()
    client.stop.side_effect = lambda: order.append("stop")

    def fake(_node):
        order.append("call")
        return True, "done"

    monkeypatch.setitem(server.SERVICE_CALLS, name, (fake, True))
    reply = asyncio.run(server.run_command(client, {"t": "svc", "name": name}))

    assert order == ["stop", "call"]
    assert reply == {"t": "reply", "ok": True, "text": f"{name}: done (brain STOP sent)"}


def test_a_failed_disarm_still_stops_the_brain(monkeypatch):
    client = MagicMock()
    monkeypatch.setitem(server.SERVICE_CALLS, "disarm",
                        (lambda _n: (False, "/wojtek/arm not available (is the robot stack up?)"), True))
    reply = asyncio.run(server.run_command(client, {"t": "svc", "name": "disarm"}))
    assert client.stop.call_count == 1
    assert reply["ok"] is False


def test_a_page_closed_mid_call_still_stops_the_brain(monkeypatch):
    client = MagicMock()
    started, release = threading.Event(), threading.Event()

    def slow_disarm(_node):
        started.set()
        release.wait(timeout=5)
        return True, "disarmed"

    monkeypatch.setitem(server.SERVICE_CALLS, "disarm", (slow_disarm, True))

    async def close_the_page_mid_call():
        task = asyncio.create_task(server.run_command(client, {"t": "svc", "name": "disarm"}))
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()   # what ws_endpoint does to running commands when the socket closes
        with pytest.raises(asyncio.CancelledError):
            await task
        stops = client.stop.call_count
        # Let the worker thread finish now: asyncio.run waits for it on the
        # way out.
        release.set()
        return stops

    try:
        assert asyncio.run(close_the_page_mid_call()) == 1
    finally:
        release.set()


def test_only_disarm_and_lie_down_stop_the_brain(monkeypatch):
    for name in wire.SERVICES:
        client = MagicMock()
        _call, stops_brain = server.SERVICE_CALLS[name]
        monkeypatch.setitem(server.SERVICE_CALLS, name, (lambda _n: (True, "ok"), stops_brain))
        asyncio.run(server.run_command(client, {"t": "svc", "name": name}))
        assert client.stop.called == (name in STOPS_THE_BRAIN), name
