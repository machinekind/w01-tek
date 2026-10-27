"""Unit tests for the OpenAI-chat-completions VLM backend.

Pure offline: httpx.AsyncClient.post is monkeypatched, no network touched.
Async code runs via asyncio.run(), matching tests/test_vlm_nav.py.
"""

import asyncio

import pytest

# httpx ships with the `eval` extra, not the default install. Skip rather
# than error out on collection, so a missing optional dep cannot break the
# fast unit suite for everyone.
httpx = pytest.importorskip("httpx")

from wojtek_eval.navigator import EVAL_ACTIONS  # noqa: E402
from wojtek_eval.vlm_openai import OpenAIVlmClient, parse_decision  # noqa: E402

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
    c = OpenAIVlmClient(base_url="http://localhost:8000", model="qwen")
    assert c.base_url == "http://localhost:8000/v1"


def test_base_url_normalization_leaves_v1_alone():
    c = OpenAIVlmClient(base_url="http://localhost:8000/v1", model="qwen")
    assert c.base_url == "http://localhost:8000/v1"


def test_base_url_normalization_strips_trailing_slash():
    c = OpenAIVlmClient(base_url="http://localhost:8000/v1/", model="qwen")
    assert c.base_url == "http://localhost:8000/v1"


# --- decide() -------------------------------------------------------------------


def _install_fake_post(monkeypatch, response: httpx.Response):
    """Monkeypatch httpx.AsyncClient.post; records call kwargs on `calls`."""
    calls = []

    async def fake_post(self, url, *, json=None, headers=None, **kw):
        calls.append({"url": url, "json": json, "headers": headers})
        return response

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    return calls


def test_decide_success_returns_vlm_decision(monkeypatch):
    body = {
        "choices": [
            {"message": {"content": '{"action":"forward","amount":0.5,"reasoning":"clear"}'}}
        ],
        "usage": {"total_tokens": 123},
    }
    response = httpx.Response(200, json=body)
    _install_fake_post(monkeypatch, response)

    client = OpenAIVlmClient(base_url="http://localhost:8000", model="qwen-vl")
    decision = asyncio.run(client.decide("go to the bed", "abc123", [], 1, 20, POSE))
    assert decision.action == "forward"
    assert decision.amount == 0.5
    assert decision.reasoning == "clear"


def test_decide_raises_on_non_200(monkeypatch):
    response = httpx.Response(500, text="internal server error, something broke badly")
    _install_fake_post(monkeypatch, response)

    client = OpenAIVlmClient(base_url="http://localhost:8000", model="qwen-vl")
    with pytest.raises(RuntimeError, match="500"):
        asyncio.run(client.decide("go to the bed", "abc123", [], 1, 20, POSE))


def test_decide_request_contains_image_and_system_prompt(monkeypatch):
    body = {"choices": [{"message": {"content": '{"action":"done","amount":null,"reasoning":"x"}'}}]}
    response = httpx.Response(200, json=body)
    calls = _install_fake_post(monkeypatch, response)

    client = OpenAIVlmClient(base_url="http://localhost:8000", model="qwen-vl")
    asyncio.run(client.decide("go to the bed", "abc123", [], 1, 20, POSE))

    assert len(calls) == 1
    payload = calls[0]["json"]
    assert payload["model"] == "qwen-vl"
    messages = payload["messages"]
    assert messages[0] == {"role": "system", "content": client.system_prompt}
    content = messages[1]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"] == "data:image/jpeg;base64,abc123"
    assert content[1]["type"] == "text"
    assert "go to the bed" in content[1]["text"]
    assert calls[0]["url"] == "http://localhost:8000/v1/chat/completions"


def test_decide_reuses_single_client_across_calls(monkeypatch):
    body = {"choices": [{"message": {"content": '{"action":"done","amount":null,"reasoning":"x"}'}}]}
    response = httpx.Response(200, json=body)
    _install_fake_post(monkeypatch, response)

    client = OpenAIVlmClient(base_url="http://localhost:8000", model="qwen-vl")

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
    client = OpenAIVlmClient(base_url="http://localhost:8000", model="qwen-vl")
    asyncio.run(client.close())
