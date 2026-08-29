"""Agent reply contract parsing: tolerant JSON extraction + the key-order guard.

The thought-first ordering is load-bearing (a decision-first contract made
Qwen3-VL-4B ignore the image entirely -- see vlm_local's RESPONSE_FORMAT and
its guard test); the tests here keep the agent contract from regressing the
same way, and pin the messy-output repairs the gpt-realtime demo needed.
"""

import pytest

from wojtek_rl.agent.chat import CONTRACT
from wojtek_rl.agent.parsing import iter_json_objects, parse_agent_reply, strip_think
from wojtek_rl.agent.search import score_view_prompt


def test_plain_say():
    r = parse_agent_reply('{"thought": "easy", "say": "Woof! I am great!"}')
    assert r.tool is None
    assert r.say == "Woof! I am great!"
    assert r.thought == "easy"


def test_tool_call_with_args():
    r = parse_agent_reply('{"thought": "need eyes", "tool": "look", "args": {}}')
    assert r.tool == "look"
    assert r.args == {}
    assert r.say is None


def test_tool_args_passthrough():
    r = parse_agent_reply('{"thought": "", "tool": "route", "args": {"seconds": 5}}')
    assert r.args == {"seconds": 5}


def test_code_fence_and_prose():
    text = 'Sure! Here you go:\n```json\n{"thought": "t", "say": "hi"}\n```\nHope that helps.'
    assert parse_agent_reply(text).say == "hi"


def test_parens_and_stray_trailing_brace():
    # gpt-realtime-2.1-mini's signature mangle: ({...})}
    assert parse_agent_reply('({"thought": "t", "say": "ok"})}').say == "ok"


def test_braces_inside_string_values():
    r = parse_agent_reply('{"thought": "a {weird} one", "say": "map has {cells}"}')
    assert r.say == "map has {cells}"


def test_skips_json_without_tool_or_say():
    text = '{"irrelevant": 1} {"thought": "t", "tool": "map", "args": {}}'
    assert parse_agent_reply(text).tool == "map"


def test_free_text_fallback_is_say():
    r = parse_agent_reply("Woof woof, doing great!")
    assert r.say == "Woof woof, doing great!"
    assert r.tool is None


def test_think_block_stripped():
    r = parse_agent_reply('<think>hmm</think>{"thought": "t", "say": "yes"}')
    assert r.say == "yes"


def test_unterminated_think_raises():
    with pytest.raises(ValueError):
        parse_agent_reply('<think>never ends {"say": "no"}')


def test_empty_reply_raises():
    with pytest.raises(ValueError):
        parse_agent_reply("   ")


def test_tool_name_normalized():
    r = parse_agent_reply('{"thought": "", "tool": " Look ", "args": {}}')
    assert r.tool == "look"


def test_iter_json_objects_multiple():
    objs = list(iter_json_objects('{"a": 1} noise {"b": {"c": 2}}'))
    assert objs == [{"a": 1}, {"b": {"c": 2}}]


def test_strip_think_keeps_after_last_close():
    assert strip_think("<think>a</think>x</think>tail") == "tail"


def test_contract_puts_thought_first():
    """Free-text field FIRST in every contract shown to the model: a 4B VLM
    forced to open with a decision token stops looking at the image."""
    for form in CONTRACT.split("{")[1:]:
        if "thought" in form:
            assert form.index("thought") < len(form.split(",")[0])
    assert CONTRACT.index('"thought"') < CONTRACT.index('"say"')
    assert CONTRACT.index('"thought"') < CONTRACT.index('"tool"')


def test_observer_prompt_puts_description_first():
    p = score_view_prompt("red ball")
    assert p.index('"description"') < p.index('"target_visible"')
    assert p.index('"description"') < p.index('"score"')


def test_tool_name_as_key_is_a_tool_call():
    """Observed live on Qwen3-VL-4B: it emitted
    {"thought": ..., "navigate": {"instruction": ...}} instead of
    {"tool": "navigate", "args": {...}} and the whole blob was SPOKEN."""
    raw = '{"thought": "I am moving toward the table.", "navigate": {"instruction": "podejdź do stołu"}}'
    r = parse_agent_reply(raw, ("look", "navigate", "search"))
    assert r.tool == "navigate"
    assert r.args == {"instruction": "podejdź do stołu"}
    assert r.say is None


def test_tool_name_as_key_with_a_bare_string_value():
    r = parse_agent_reply('{"thought": "", "navigate": "idź do łóżka"}', ("navigate",))
    assert r.tool == "navigate" and r.args == {"instruction": "idź do łóżka"}


def test_proper_tool_form_still_wins():
    r = parse_agent_reply('{"thought": "", "tool": "look", "args": {}}', ("look", "navigate"))
    assert r.tool == "look" and r.args == {}


def test_malformed_json_is_never_spoken():
    """Better a repair turn than reading braces and field names aloud."""
    with pytest.raises(ValueError, match="malformed"):
        parse_agent_reply('{"thought": "I am moving toward the table", "nav": ', ())
    with pytest.raises(ValueError):
        parse_agent_reply('{"thought": "hm", "unknown_field": 1}', ("look",))


def test_plain_prose_is_still_accepted():
    assert parse_agent_reply("Woof, doing great!", ("look",)).say == "Woof, doing great!"
