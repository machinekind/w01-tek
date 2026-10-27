"""SSE parsing, model-free."""

import json

from wojtek_brain.llm_client import iter_sse_tokens


def sse(content):
    return "data: " + json.dumps({"choices": [{"delta": {"content": content}}]})


def test_yields_content_tokens_in_order():
    lines = [sse("Cz"), sse("eść"), "", sse("!"), "data: [DONE]", sse("po-done")]
    assert list(iter_sse_tokens(iter(lines))) == ["Cz", "eść", "!"]


def test_skips_noise_and_bad_json():
    lines = [
        ": comment",
        "event: ping",
        "data: {broken",
        "data: " + json.dumps({"choices": [{"delta": {"role": "assistant"}}]}),
        sse("ok"),
        "data: [DONE]",
    ]
    assert list(iter_sse_tokens(iter(lines))) == ["ok"]


def test_empty_stream():
    assert list(iter_sse_tokens(iter([]))) == []
