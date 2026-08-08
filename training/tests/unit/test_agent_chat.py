"""WojtekAgent chat loop against a scripted LLM: tool dispatch, repair
turns, context trimming. No network, no model."""

import asyncio

import pytest

from wojtek_rl.agent.chat import WojtekAgent, system_prompt
from wojtek_rl.agent.tools import Tool, ToolResult


class FakeLLM:
    """Returns scripted replies in order; records every message list."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    async def chat(self, messages, max_tokens=None):
        self.calls.append(messages)
        if not self.replies:
            raise AssertionError("FakeLLM ran out of scripted replies")
        return self.replies.pop(0)


def make_tools(record):
    async def look(args):
        record.append(("look", args))
        return ToolResult(text="you see a sofa", images=("imgb64",))

    async def boom(args):
        raise RuntimeError("renderer exploded")

    return {
        "look": Tool("look", "look {}", "camera", look),
        "boom": Tool("boom", "boom {}", "always fails", boom),
    }


def make_agent(replies, record=None):
    record = record if record is not None else []
    llm = FakeLLM(replies)
    agent = WojtekAgent(llm, make_tools(record), max_tool_steps=3, keep_exchanges=2)
    return agent, llm, record


def ask(agent, text, voice=False):
    return asyncio.run(agent.ask(text, voice=voice))


def test_direct_answer_no_tools():
    agent, llm, record = make_agent(['{"thought": "chitchat", "say": "Woof! Great!"}'])
    out = ask(agent, "how are you?")
    assert out["ok"] and out["say"] == "Woof! Great!"
    assert out["steps"] == [] and record == []
    # System prompt went out with the call.
    assert llm.calls[0][0]["role"] == "system"


def test_tool_then_answer():
    agent, llm, record = make_agent([
        '{"thought": "need eyes", "tool": "look", "args": {}}',
        '{"thought": "sofa it is", "say": "I see a sofa!"}',
    ])
    out = ask(agent, "what do you see?")
    assert out["say"] == "I see a sofa!"
    assert out["steps"] == [{"tool": "look", "args": {}, "result": "you see a sofa"}]
    assert record == [("look", {})]
    # Second call carries the tool result, image included.
    last_user = llm.calls[1][-1]
    kinds = [c["type"] for c in last_user["content"]]
    assert "image_url" in kinds
    assert any("you see a sofa" in c.get("text", "") for c in last_user["content"])


def test_tool_error_surfaced_not_fatal():
    agent, _, _ = make_agent([
        '{"thought": "", "tool": "boom", "args": {}}',
        '{"thought": "", "say": "oops, my camera hiccuped"}',
    ])
    out = ask(agent, "look around")
    assert out["ok"]
    assert "tool error" in out["steps"][0]["result"]


def test_unknown_tool_gets_corrective_turn():
    agent, llm, _ = make_agent([
        '{"thought": "", "tool": "teleport", "args": {}}',
        '{"thought": "", "say": "no teleporting, sorry"}',
    ])
    out = ask(agent, "teleport home")
    assert out["say"] == "no teleporting, sorry"
    correction = llm.calls[1][-1]["content"][-1]["text"]
    assert "No tool named" in correction


def test_invalid_json_gets_repair_turn():
    agent, llm, _ = make_agent([
        "<think>never closed and no json",
        '{"thought": "", "say": "second try"}',
    ])
    out = ask(agent, "hi")
    assert out["say"] == "second try"
    assert "not valid JSON" in llm.calls[1][-1]["content"][-1]["text"]


def test_repeated_identical_call_is_nudged():
    agent, llm, record = make_agent([
        '{"thought": "", "tool": "look", "args": {}}',
        '{"thought": "", "tool": "look", "args": {}}',
        '{"thought": "", "say": "still a sofa"}',
    ])
    out = ask(agent, "what do you see?")
    assert out["say"] == "still a sofa"
    assert record == [("look", {})]  # executed once, not twice
    assert "already made that exact tool call" in llm.calls[2][-1]["content"][-1]["text"]


def test_budget_exhaustion_falls_back_to_last_result():
    agent, _, _ = make_agent(
        ['{"thought": "", "tool": "look", "args": {}}'] * 4
    )
    out = ask(agent, "what do you see?")
    assert out["ok"]
    assert "you see a sofa" in out["say"]


def test_history_trims_and_is_text_only():
    replies = []
    for k in range(4):
        replies.append('{"thought": "", "tool": "look", "args": {"k": %d}}' % k)
        replies.append('{"thought": "", "say": "answer %d"}' % k)
    agent, llm, _ = make_agent(replies)
    for k in range(4):
        ask(agent, f"question {k}")
    # keep_exchanges=2 -> 4 messages of history.
    assert len(agent._history) == 4
    texts = [c["text"] for m in agent._history for c in m["content"]]
    assert texts == ["question 2", "answer 2", "question 3", "answer 3"]
    # No images survive into history.
    assert all(c["type"] == "text" for m in agent._history for c in m["content"])


def test_free_text_reply_is_accepted_as_say():
    agent, _, _ = make_agent(["Woof, doing great, thanks!"])
    assert ask(agent, "hi")["say"] == "Woof, doing great, thanks!"


def test_empty_message_rejected():
    agent, _, _ = make_agent([])
    assert ask(agent, "   ")["ok"] is False


def test_debug_trace_records_every_llm_call():
    agent, _, _ = make_agent([
        '{"thought": "need eyes", "tool": "look", "args": {}}',
        '{"thought": "sofa", "say": "I see a sofa!"}',
    ])
    out = ask(agent, "what do you see?")
    calls = out["debug"]["llm_calls"]
    assert [c["kind"] for c in calls] == ["tool", "say"]
    assert calls[0]["tool"] == "look" and calls[0]["args"] == {}
    assert calls[0]["tool_result"] == "you see a sofa"
    assert calls[0]["tool_images"] == 1
    assert calls[0]["thought"] == "need eyes"
    assert "need eyes" in calls[0]["raw"]
    assert calls[1]["say"] == "I see a sofa!"
    assert out["debug"]["history_len"] == 2


def test_debug_trace_classifies_repairs():
    agent, _, _ = make_agent([
        "<think>never closed and no json",
        '{"thought": "", "tool": "teleport", "args": {}}',
        '{"thought": "", "say": "done"}',
    ])
    out = ask(agent, "hi")
    kinds = [c["kind"] for c in out["debug"]["llm_calls"]]
    assert kinds == ["parse_error", "unknown_tool", "say"]


def test_debug_trace_carries_llm_meta_when_available():
    agent, llm, _ = make_agent(['{"thought": "", "say": "hi"}'])
    llm.last_meta = None

    async def chat(messages, max_tokens=None):
        llm.calls.append(messages)
        llm.last_meta = {"latency_ms": 123, "tokens": 456}
        return llm.replies.pop(0)

    llm.chat = chat
    out = ask(agent, "hello")
    call = out["debug"]["llm_calls"][0]
    assert call["latency_ms"] == 123 and call["tokens"] == 456


def test_trace_records_the_whole_turn():
    from wojtek_rl.agent.trace import Trace

    trace = Trace(None)
    agent, _, _ = make_agent([
        '{"thought": "need eyes", "tool": "look", "args": {}}',
        '{"thought": "sofa", "say": "I see a sofa!"}',
    ])
    agent.trace = trace
    ask(agent, "what do you see?")
    kinds = [e["kind"] for e in trace.recent()]
    assert kinds == ["chat.ask", "chat.llm", "chat.tool", "chat.llm", "chat.say"]
    events = {e["kind"]: e for e in trace.recent()}
    assert events["chat.ask"]["text"] == "what do you see?"
    assert events["chat.tool"]["tool"] == "look"
    assert events["chat.tool"]["result"] == "you see a sofa"
    assert events["chat.tool"]["images"] == 1
    assert events["chat.say"]["say"] == "I see a sofa!"
    assert events["chat.say"]["tools"] == ["look"]
    assert events["chat.llm"]["raw"]


def test_trace_optional_and_reset_recorded():
    from wojtek_rl.agent.trace import Trace

    agent, _, _ = make_agent(['{"thought": "", "say": "hi"}'])
    ask(agent, "hello")  # no trace attached: must not raise
    trace = Trace(None)
    agent.trace = trace
    agent.reset()
    assert [e["kind"] for e in trace.recent()] == ["chat.reset"]


def test_look_tool_prefers_hud_free_frame():
    """Regression guard for the minimap self-detection bug: the camera tool
    must ask for the bare frame when the sim can give one."""
    from wojtek_rl.agent.tools import build_tools

    calls = []

    class Sim:
        omap = None
        executor = None

        def ego_jpeg(self, hud=True):
            calls.append(hud)
            return "frame"

        def pose(self):
            return (0.0, 0.0, 0.0)

    tools = build_tools(Sim(), goals=None, pose_history=None)
    asyncio.run(tools["look"].fn({}))
    assert calls == [False]


def test_look_tool_falls_back_for_old_sims():
    from wojtek_rl.agent.tools import build_tools

    class OldSim:
        omap = None
        executor = None

        def ego_jpeg(self):  # no hud kwarg
            return "frame"

        def pose(self):
            return (0.0, 0.0, 0.0)

    tools = build_tools(OldSim(), goals=None, pose_history=None)
    assert asyncio.run(tools["look"].fn({})).images == ("frame",)


class NavSim:
    """Minimal sim for the navigate/search tools."""

    omap = None
    executor = None

    def ego_jpeg(self, hud=True):
        return "frame"

    def pose(self):
        return (0.0, 0.0, 0.0)

    def submit_command(self, text):
        return {"ok": True}


class RecordingGoals:
    def __init__(self):
        self.set = []

    def set_goal(self, text, kind="navigate"):
        self.set.append((kind, text))
        return {"ok": True, "goal": text}

    def cancel(self, reason="user"):
        pass


def nav_tools(user_text):
    from wojtek_rl.agent.tools import build_tools

    goals = RecordingGoals()
    tools = build_tools(NavSim(), goals, None, turn_context={"user_text": user_text})
    return tools, goals


def test_navigate_keeps_multi_step_instruction_verbatim():
    """The navigator is an instruction follower: collapsing 'walk around the
    table, then stop by the door' to 'the table' throws the route away."""
    spoken = "walk around the table, then stop by the door"
    tools, goals = nav_tools(spoken)
    asyncio.run(tools["navigate"].fn({"instruction": "the table"}))
    assert goals.set == [("navigate", spoken)]


def test_navigate_accepts_a_faithful_instruction():
    spoken = "please walk around the table and stop by the door"
    faithful = "walk around the table and stop by the door"
    tools, goals = nav_tools(spoken)
    asyncio.run(tools["navigate"].fn({"instruction": faithful}))
    assert goals.set == [("navigate", faithful)]


def test_navigate_accepts_legacy_goal_arg():
    tools, goals = nav_tools("go to the bed")
    asyncio.run(tools["navigate"].fn({"goal": "the bed"}))
    assert goals.set == [("navigate", "go to the bed")]


def test_navigate_short_command_passes_through():
    """A genuinely short instruction is not a collapsed one."""
    tools, goals = nav_tools("go to the bed")
    asyncio.run(tools["navigate"].fn({"instruction": "go to the bed"}))
    assert goals.set == [("navigate", "go to the bed")]


def test_navigate_falls_back_to_spoken_when_arg_missing():
    tools, goals = nav_tools("walk past the sofa and turn left")
    asyncio.run(tools["navigate"].fn({}))
    assert goals.set == [("navigate", "walk past the sofa and turn left")]


def test_chat_turn_publishes_user_text_to_tools():
    ctx = {}
    llm = FakeLLM(['{"thought": "", "say": "ok"}'])
    agent = WojtekAgent(llm, make_tools([]), turn_context=ctx)
    ask(agent, "walk around the table twice")
    assert ctx["user_text"] == "walk around the table twice"


def test_instruction_prompt_covers_ordered_routes():
    from wojtek_rl.agent.nav import INSTRUCTION_PROMPT, NAV_MAX_ROTATION, NAV_MAX_STEPS
    from wojtek_rl.vlm_nav import SYSTEM_PROMPT

    assert SYSTEM_PROMPT in INSTRUCTION_PROMPT  # base contract intact
    assert "IN ORDER" in INSTRUCTION_PROMPT
    assert "LAST step" in INSTRUCTION_PROMPT
    # Interactive runs are not benchmark episodes: no step budget, and the
    # anti-spin guard (not a counter) is what ends a lost run.
    assert NAV_MAX_STEPS is None
    assert NAV_MAX_ROTATION >= 12


def test_voice_turn_uses_the_spoken_style_prompt():
    """A spoken reply must not contain lists or emoji -- the synthesiser
    reads them out loud."""
    llm = FakeLLM(['{"thought": "", "say": "Woof!"}', '{"thought": "", "say": "Woof!"}'])
    agent = WojtekAgent(llm, make_tools([]))
    asyncio.run(agent.ask("jak się masz?", voice=True))
    spoken_system = llm.calls[0][0]["content"]
    assert "SPOKEN OUT LOUD" in spoken_system
    asyncio.run(agent.ask("jak się masz?"))
    assert "SPOKEN OUT LOUD" not in llm.calls[1][0]["content"]


def test_typed_replies_stay_english():
    """Everything but the spoken sentence runs in English."""
    llm = FakeLLM(['{"thought": "", "say": "Woof, doing great!"}'])
    agent = WojtekAgent(llm, make_tools([]))
    ask(agent, "how are you?")
    system = llm.calls[0][0]["content"]
    assert "Write everything in English" in system
    assert "SPOKEN ALOUD" not in system


def test_spoken_replies_switch_only_the_say_field():
    llm = FakeLLM(['{"thought": "greeting", "say": "Cześć!"}'])
    agent = WojtekAgent(llm, make_tools([]), reply_language="pl")
    out = ask(agent, "jak się masz?", voice=True)
    assert out["say"] == "Cześć!"
    system = llm.calls[0][0]["content"]
    assert "SPOKEN ALOUD to a Polish speaker" in system
    assert "Everything else stays English" in system
    assert len(llm.calls) == 1  # direct mode: no extra hop


def test_translate_mode_only_affects_spoken_turns():
    llm = FakeLLM([
        '{"thought": "greeting", "say": "Woof! I am great!"}',
        "Hau! Świetnie się mam!",
        '{"thought": "greeting", "say": "Woof! I am great!"}',
    ])
    agent = WojtekAgent(llm, make_tools([]), lang_mode="translate", reply_language="pl")
    spoken = ask(agent, "jak się masz?", voice=True)
    assert spoken["say"] == "Hau! Świetnie się mam!"
    assert "Polish" in llm.calls[1][-1]["content"][-1]["text"]
    typed = ask(agent, "how are you?")
    assert typed["say"] == "Woof! I am great!"  # typed stays English, no hop


def test_translation_failure_falls_back_to_the_source_line():
    """Saying something in the wrong language beats saying nothing."""

    class HalfBrokenLLM(FakeLLM):
        async def chat(self, messages, max_tokens=None):
            self.calls.append(messages)
            if len(self.calls) == 1:
                return '{"thought": "", "say": "Woof! On my way!"}'
            raise RuntimeError("translator down")

    agent = WojtekAgent(HalfBrokenLLM([]), make_tools([]), lang_mode="translate")
    assert ask(agent, "chodz", voice=True)["say"] == "Woof! On my way!"


def test_translation_strips_quotes_and_think_blocks():
    llm = FakeLLM([
        '{"thought": "", "say": "Woof!"}',
        '<think>hmm</think> "Hau!" ',
    ])
    agent = WojtekAgent(llm, make_tools([]), lang_mode="translate")
    assert ask(agent, "hej", voice=True)["say"] == "Hau!"


def test_translate_mode_is_traced_with_both_versions():
    from wojtek_rl.agent.trace import Trace

    trace = Trace(None)
    llm = FakeLLM(['{"thought": "", "say": "Woof! Hello!"}', "Hau! Cześć!"])
    agent = WojtekAgent(llm, make_tools([]), lang_mode="translate", trace=trace)
    ask(agent, "cześć", voice=True)
    say = [e for e in trace.recent() if e["kind"] == "chat.say"][0]
    assert say["say"] == "Hau! Cześć!"
    assert say["source"] == "Woof! Hello!"  # English kept for debugging


def test_unknown_lang_mode_is_rejected_at_construction():
    with pytest.raises(ValueError, match="lang_mode"):
        WojtekAgent(FakeLLM([]), make_tools([]), lang_mode="pig-latin")


def test_persona_carries_no_language_policy():
    """Language belongs to the per-turn prompt, not the persona: the same dog
    types English and speaks Polish."""
    from wojtek_rl.agent.chat import PERSONA

    assert "Polish" not in PERSONA
    assert "English" not in PERSONA


def test_system_prompt_mentions_every_tool():
    tools = make_tools([])
    prompt = system_prompt(tools)
    for name in tools:
        assert name in prompt
    assert "Wojtek" in prompt and "dog" in prompt
