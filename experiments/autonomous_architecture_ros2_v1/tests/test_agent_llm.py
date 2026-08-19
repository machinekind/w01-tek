"""AgentLLM wire behaviour: retry budget, error text, metadata.

Pure offline -- httpx.AsyncClient.post is monkeypatched, no network touched
(same approach as tests/unit/test_vlm_openai.py).
"""

import asyncio

import pytest

httpx = pytest.importorskip("httpx")

from wojtek_agent.llm import AgentLLM, user_message  # noqa: E402


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


def ok_payload(content="hi", tokens=42):
    return {"choices": [{"message": {"content": content}}], "usage": {"total_tokens": tokens}}


def test_successful_call_returns_content_and_meta(monkeypatch):
    async def post(self, url, **kw):
        assert url.endswith("/v1/chat/completions")
        return FakeResponse(payload=ok_payload("woof"))

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    llm = AgentLLM(base_url="http://box:8090", model="m")
    assert asyncio.run(llm.chat([user_message("hi")])) == "woof"
    assert llm.last_meta["tokens"] == 42
    assert llm.last_meta["model"] == "m"
    assert isinstance(llm.last_meta["latency_ms"], int)


def test_unreachable_endpoint_fails_fast_with_url_in_message(monkeypatch):
    """A person is waiting on this call: a dead endpoint must surface in
    seconds with the address, not after a minute of silent retries."""
    attempts = []

    async def post(self, url, **kw):
        attempts.append(url)
        raise httpx.ConnectError("No route to host")

    slept = []

    async def fake_sleep(s):
        slept.append(s)

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    llm = AgentLLM(base_url="http://box:8090", model="m", attempts=3, retry_backoff_s=2.0)
    with pytest.raises(RuntimeError) as e:
        asyncio.run(llm.chat([user_message("hi")]))
    assert "unreachable" in str(e.value)
    assert "http://box:8090/v1" in str(e.value)
    assert "No route to host" in str(e.value)
    assert len(attempts) == 3
    assert sum(slept) <= 6.0  # total backoff stays in seconds, not a minute


def test_retries_recover_from_a_transient_5xx(monkeypatch):
    calls = {"n": 0}

    async def post(self, url, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return FakeResponse(status_code=503, text="starting up")
        return FakeResponse(payload=ok_payload("back"))

    async def no_sleep(_s):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    llm = AgentLLM(base_url="http://box:8090", model="m")
    assert asyncio.run(llm.chat([user_message("hi")])) == "back"
    assert calls["n"] == 2


def test_non_5xx_error_raises_with_status(monkeypatch):
    async def post(self, url, **kw):
        return FakeResponse(status_code=404, text="no such model")

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    llm = AgentLLM(base_url="http://box:8090", model="m")
    with pytest.raises(RuntimeError, match="404"):
        asyncio.run(llm.chat([user_message("hi")]))


def test_single_attempt_configuration(monkeypatch):
    attempts = []

    async def post(self, url, **kw):
        attempts.append(1)
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    llm = AgentLLM(base_url="http://box:8090", model="m", attempts=1)
    with pytest.raises(RuntimeError):
        asyncio.run(llm.chat([user_message("hi")]))
    assert len(attempts) == 1


def test_user_message_puts_images_before_text():
    msg = user_message("what is this?", ("b64frame",))
    assert [c["type"] for c in msg["content"]] == ["image_url", "text"]
