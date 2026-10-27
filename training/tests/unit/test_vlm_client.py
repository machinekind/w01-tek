"""Unit tests for the VLM client (Ollama, OpenAI chat-completions API).

Pure offline: httpx.AsyncClient.post is monkeypatched, no network touched.
Async code runs via asyncio.run(), matching tests/test_vlm_nav.py.
"""

import asyncio

import pytest

# httpx ships with the `demo`/`eval` extras, not the default install. Skip rather
# than error out on collection, so a missing optional dep cannot break the
# fast unit suite for everyone.
httpx = pytest.importorskip("httpx")

from wojtek_eval.navigator import EVAL_ACTIONS  # noqa: E402
from wojtek_rl.vlm_client import (  # noqa: E402
    DEFAULT_VLM_MODEL,
    DEFAULT_VLM_URL,
    OpenAIVlmClient,
    parse_decision,
)
from wojtek_rl.vlm_nav import ACTIONS, SYSTEM_PROMPT  # noqa: E402

POSE = (0.0, 0.0, 0.0)


# --- parse_decision -----------------------------------------------------------


def test_parses_clean_json():
    d = parse_decision(
        '{"action": "turn_left", "amount": 30, "reasoning": "bed is left"}', EVAL_ACTIONS
    )
    assert d.action == "turn_left"
    assert d.amount == 30.0
    assert d.reasoning == "bed is left"


def test_parses_fenced_json_with_prose():
    text = (
        "Sure! Here is my decision:\n```json\n"
        '{"action": "forward", "amount": 0.5, "reasoning": "clear path"}\n```'
    )
    d = parse_decision(text, EVAL_ACTIONS)
    assert d.action == "forward"
    assert d.amount == 0.5


def test_parses_json_with_surrounding_prose():
    text = 'The robot should proceed. {"action": "backward", "amount": 0.2, "reasoning": "too close"} done.'
    d = parse_decision(text, EVAL_ACTIONS)
    assert d.action == "backward"
    assert d.amount == 0.2


def test_strips_think_block_before_parsing():
    text = (
        "<think>The bed is not visible, I should scan left.</think>"
        '{"action": "turn_left", "amount": 45, "reasoning": "scanning"}'
    )
    d = parse_decision(text, EVAL_ACTIONS)
    assert d.action == "turn_left"
    assert d.amount == 45.0


def test_strips_multiple_think_blocks_uses_text_after_last():
    text = (
        "<think>first pass</think>garbage<think>second pass</think>"
        '{"action": "done", "amount": null, "reasoning": "arrived"}'
    )
    d = parse_decision(text, EVAL_ACTIONS)
    assert d.action == "done"


def test_unterminated_think_block_raises():
    with pytest.raises(ValueError):
        parse_decision("<think>still thinking, never emitted an answer", EVAL_ACTIONS)


def test_falls_back_to_free_text_command():
    d = parse_decision("turn_left 30 because the bed is off to the side.", EVAL_ACTIONS)
    assert d.action == "turn_left"
    assert d.amount == 30.0


def test_invalid_action_in_json_tries_next_object():
    text = '{"action": "jump", "reasoning": "nope"} then {"action": "forward", "amount": 1.0, "reasoning": "ok"}'
    d = parse_decision(text, EVAL_ACTIONS)
    assert d.action == "forward"
    assert d.amount == 1.0


def test_nothing_parseable_raises():
    with pytest.raises(ValueError):
        parse_decision("The room contains a bed and a window.", EVAL_ACTIONS)


def test_explore_action_parses_with_amount_none():
    d = parse_decision('{"action": "explore", "amount": null, "reasoning": "search other rooms"}', EVAL_ACTIONS)
    assert d.action == "explore"
    assert d.amount is None


# --- base_url normalization ----------------------------------------------------


def test_base_url_normalization_appends_v1():
    c = OpenAIVlmClient(base_url="http://localhost:11434", model="qwen3-vl:30b-a3b-instruct")
    assert c.base_url == "http://localhost:11434/v1"


def test_base_url_normalization_leaves_v1_alone():
    c = OpenAIVlmClient(base_url="http://localhost:11434/v1", model="qwen3-vl:30b-a3b-instruct")
    assert c.base_url == "http://localhost:11434/v1"


def test_base_url_normalization_strips_trailing_slash():
    c = OpenAIVlmClient(base_url="http://localhost:11434/v1/", model="qwen3-vl:30b-a3b-instruct")
    assert c.base_url == "http://localhost:11434/v1"


# --- decide() -------------------------------------------------------------------


def _install_fake_post(monkeypatch, response: httpx.Response):
    """Monkeypatch httpx.AsyncClient.post; records call kwargs on `calls`."""
    calls = []

    async def fake_post(self, url, *, json=None, headers=None, **kw):
        calls.append({"url": url, "json": json, "headers": headers})
        return response

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    return calls


def _install_fake_post_sequence(monkeypatch, outcomes):
    """Like _install_fake_post, but each call takes the next outcome: an
    Exception instance is raised, anything else is returned. The last
    outcome repeats once the list runs out."""
    calls = []

    async def fake_post(self, url, *, json=None, headers=None, **kw):
        calls.append({"url": url, "json": json, "headers": headers})
        outcome = outcomes[min(len(calls), len(outcomes)) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    return calls


def _record_sleeps(monkeypatch):
    """Swap asyncio.sleep for a stub that only writes down each wait, so the
    retry tests finish at once. The client uses asyncio only to sleep, and
    asyncio.run does not call sleep itself."""
    sleeps = []

    async def fake_sleep(delay, *a, **k):
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return sleeps


_FORWARD_BODY = {
    "choices": [{"message": {"content": '{"action":"forward","amount":0.5,"reasoning":"clear"}'}}]
}


def test_decide_success_returns_vlm_decision(monkeypatch):
    body = {
        "choices": [
            {"message": {"content": '{"action":"forward","amount":0.5,"reasoning":"clear"}'}}
        ],
        "usage": {"total_tokens": 123},
    }
    response = httpx.Response(200, json=body)
    _install_fake_post(monkeypatch, response)

    client = OpenAIVlmClient(base_url="http://localhost:11434", model="qwen3-vl:30b-a3b-instruct")
    decision = asyncio.run(client.decide("go to the bed", "abc123", [], 1, 20, POSE))
    assert decision.action == "forward"
    assert decision.amount == 0.5
    assert decision.reasoning == "clear"


def test_decide_raises_on_non_200(monkeypatch):
    response = httpx.Response(500, text="internal server error, something broke badly")
    calls = _install_fake_post(monkeypatch, response)
    sleeps = _record_sleeps(monkeypatch)

    client = OpenAIVlmClient(base_url="http://localhost:11434", model="qwen3-vl:30b-a3b-instruct")
    with pytest.raises(RuntimeError, match="500"):
        asyncio.run(client.decide("go to the bed", "abc123", [], 1, 20, POSE))
    # The default client fails fast: one request, no waiting.
    assert len(calls) == 1
    assert sleeps == []


def test_default_client_does_not_retry_connect_errors(monkeypatch):
    calls = _install_fake_post_sequence(
        monkeypatch, [httpx.ConnectError("All connection attempts failed")]
    )
    sleeps = _record_sleeps(monkeypatch)

    client = OpenAIVlmClient(base_url="http://localhost:11434", model="qwen3-vl:30b-a3b-instruct")
    with pytest.raises(RuntimeError, match="unreachable") as exc:
        asyncio.run(client.decide("go to the bed", "abc123", [], 1, 20, POSE))
    assert client.base_url in str(exc.value)
    assert len(calls) == 1
    assert sleeps == []


def test_retries_ride_out_connect_errors_then_succeed(monkeypatch):
    calls = _install_fake_post_sequence(
        monkeypatch,
        [
            httpx.ConnectError("All connection attempts failed"),
            httpx.ConnectError("All connection attempts failed"),
            httpx.Response(200, json=_FORWARD_BODY),
        ],
    )
    sleeps = _record_sleeps(monkeypatch)

    client = OpenAIVlmClient(
        base_url="http://localhost:11434", model="qwen3-vl:30b-a3b-instruct", retries=3
    )
    decision = asyncio.run(client.decide("go to the bed", "abc123", [], 1, 20, POSE))
    assert decision.action == "forward"
    assert len(calls) == 3
    assert sleeps == [12.0, 24.0]


def test_retries_give_up_after_the_last_5xx(monkeypatch):
    calls = _install_fake_post(monkeypatch, httpx.Response(503, text="model is loading"))
    sleeps = _record_sleeps(monkeypatch)

    client = OpenAIVlmClient(
        base_url="http://localhost:11434", model="qwen3-vl:30b-a3b-instruct", retries=3
    )
    with pytest.raises(RuntimeError, match="503"):
        asyncio.run(client.decide("go to the bed", "abc123", [], 1, 20, POSE))
    assert len(calls) == 4
    assert sleeps == [12.0, 24.0, 36.0]


def test_4xx_is_never_retried(monkeypatch):
    calls = _install_fake_post(monkeypatch, httpx.Response(404, text="model not found"))
    sleeps = _record_sleeps(monkeypatch)

    client = OpenAIVlmClient(
        base_url="http://localhost:11434", model="qwen3-vl:30b-a3b-instruct", retries=3
    )
    with pytest.raises(RuntimeError, match="404"):
        asyncio.run(client.decide("go to the bed", "abc123", [], 1, 20, POSE))
    assert len(calls) == 1
    assert sleeps == []


def test_negative_retries_is_rejected():
    with pytest.raises(ValueError):
        OpenAIVlmClient(retries=-1)


def test_decide_request_contains_image_and_system_prompt(monkeypatch):
    body = {"choices": [{"message": {"content": '{"action":"done","amount":null,"reasoning":"x"}'}}]}
    response = httpx.Response(200, json=body)
    calls = _install_fake_post(monkeypatch, response)

    client = OpenAIVlmClient(base_url="http://localhost:11434", model="qwen3-vl:30b-a3b-instruct")
    asyncio.run(client.decide("go to the bed", "abc123", [], 1, 20, POSE))

    assert len(calls) == 1
    payload = calls[0]["json"]
    assert payload["model"] == "qwen3-vl:30b-a3b-instruct"
    messages = payload["messages"]
    assert messages[0] == {"role": "system", "content": client.system_prompt}
    content = messages[1]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"] == "data:image/jpeg;base64,abc123"
    assert content[1]["type"] == "text"
    assert "go to the bed" in content[1]["text"]
    assert calls[0]["url"] == "http://localhost:11434/v1/chat/completions"


def test_decide_reuses_single_client_across_calls(monkeypatch):
    body = {"choices": [{"message": {"content": '{"action":"done","amount":null,"reasoning":"x"}'}}]}
    response = httpx.Response(200, json=body)
    _install_fake_post(monkeypatch, response)

    client = OpenAIVlmClient(base_url="http://localhost:11434", model="qwen3-vl:30b-a3b-instruct")

    async def scenario():
        await client.decide("goal", "abc", [], 1, 20, POSE)
        first = client._client
        await client.decide("goal", "abc", [], 2, 20, POSE)
        second = client._client
        return first, second

    first, second = asyncio.run(scenario())
    assert first is second
    assert first is not None
    asyncio.run(client.close())
    assert client._client is None


def test_close_without_ever_deciding_is_a_noop():
    client = OpenAIVlmClient(base_url="http://localhost:11434", model="qwen3-vl:30b-a3b-instruct")
    asyncio.run(client.close())


def test_defaults_are_the_30b_on_ollama_with_the_room_prompt():
    client = OpenAIVlmClient()
    assert client.base_url == DEFAULT_VLM_URL == "http://127.0.0.1:11434/v1"
    assert client.model == DEFAULT_VLM_MODEL == "qwen3-vl:30b-a3b-instruct"
    assert client.system_prompt == SYSTEM_PROMPT
    assert client.actions == ACTIONS
    assert client.retries == 0
