"""Client protocol tests plus the constants cross-check against wojtek_rl."""

import pytest

from wojtek_futurenav_bridge import futurenav_http
from wojtek_futurenav_bridge.futurenav_http import (
    FutureNavHttpClient,
    FutureNavHttpError,
)


def test_grid_constants_match_the_training_client():
    """The restated grid must equal wojtek_rl.futurenav_nav's original."""
    fn = pytest.importorskip("wojtek_rl.futurenav_nav")
    assert futurenav_http.FORWARD_STEP_M == fn.FORWARD_STEP_M
    assert futurenav_http.TURN_STEP_DEG == fn.TURN_STEP_DEG
    assert set(futurenav_http.KNOWN_ACTIONS) == set(fn.ACTION_TO_DECISION)
    assert futurenav_http.DEFAULT_FUTURENAV_URL == fn.DEFAULT_FUTURENAV_URL


def test_act_returns_known_action(monkeypatch):
    client = FutureNavHttpClient("http://example.invalid")
    monkeypatch.setattr(
        client, "_post", lambda path, payload: {"action": "TURN_LEFT", "raw": "tl"}
    )
    assert client.act("frame") == "TURN_LEFT"


def test_act_rejects_unknown_action(monkeypatch):
    client = FutureNavHttpClient("http://example.invalid")
    monkeypatch.setattr(
        client, "_post", lambda path, payload: {"action": "JUMP", "raw": "?"}
    )
    with pytest.raises(FutureNavHttpError):
        client.act("frame")


def test_transport_failure_is_wrapped():
    client = FutureNavHttpClient("http://127.0.0.1:1", timeout_s=0.2)
    with pytest.raises(FutureNavHttpError):
        client.reset("go")


def test_reset_and_act_hit_the_documented_paths(monkeypatch):
    calls = []

    def fake_post(path, payload):
        calls.append((path, payload))
        return {"action": "STOP", "raw": "stop"}

    client = FutureNavHttpClient("http://example.invalid/")
    monkeypatch.setattr(client, "_post", fake_post)
    client.reset("znajdź kanapę")
    client.act("abc123")
    assert calls == [
        ("/reset", {"instruction": "znajdź kanapę"}),
        ("/act", {"frame_b64": "abc123"}),
    ]
    assert client.url == "http://example.invalid"  # trailing slash stripped
