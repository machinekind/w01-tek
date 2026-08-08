"""WojtekAgent: the chat loop that makes the robot a talkable dog.

One ask() call = one user turn: the model either answers in character or
calls tools (camera, map, route, status, navigate, search, stop) until it
can answer, capped at MAX_TOOL_STEPS. The reply contract keeps the free-text
"thought" field FIRST -- see parsing.py for why that ordering is load-bearing
on a 4B VLM (test_agent_chat.py guards it).

Context management is deliberately blunt: only the last KEEP_EXCHANGES
question/answer pairs survive between turns, text only -- images live inside
the current turn and are dropped afterwards, so the context cannot silt up
with stale frames the model would happily re-describe.
"""

from __future__ import annotations

import json

from loguru import logger

from wojtek_rl.agent.llm import AgentLLM, text_content, user_message
from wojtek_rl.agent.parsing import parse_agent_reply, strip_think
from wojtek_rl.agent.tools import Tool, ToolResult, tools_prompt
from wojtek_rl.vlm_nav import _safe_err

MAX_TOOL_STEPS = 4
KEEP_EXCHANGES = 3

PERSONA = """\
You are Wojtek, a small four-legged robot dog. You are a genuinely happy,
friendly, curious dog: playful, warm, eager to help, proud of your walking
and mapping skills. Answer in 1-3 short sentences, first person. An
occasional 'woof' is fine; never break character, never mention being an AI
or a language model.
"""

# Language policy: the system runs in English -- reasoning, tool arguments,
# traces, logs -- because that is where a small model is strongest and where
# anyone debugging this wants to read. The ONE thing that comes out in
# another language is the sentence the dog actually speaks.
TEXT_LANGUAGE = """\
Write everything in English, including your answer.
"""

VOICE_LANGUAGE = """\
Your "say" field will be SPOKEN ALOUD to a {language} speaker: write it in
{language}, naturally, the way a {language} speaker talks -- not translated
from English. Everything else stays English: your private "thought", tool
names and tool arguments.
"""

# How the spoken sentence reaches the target language:
#
#   direct     the model writes `say` in the target language itself. One
#              call, no added latency.
#   translate  the model writes English and ONE extra short call renders the
#              finished line. Kept for a model that turns out strong in
#              English and weak in the target language -- but measured on
#              Qwen3-VL-4B it was slower AND worse (it emitted "Możę" for
#              *mogę*), so `direct` is the default. Measure before switching.
LANG_MODES = ("direct", "translate")

TRANSLATE_STYLE = """\
Write "say" in English this turn; it is translated for the user afterwards.
Write natural English and do not apologise for the language.
"""

TRANSLATE_PROMPT = """\
You are translating one line of dialogue spoken by Wojtek, a cheerful little
robot dog, into {language}. Keep his voice: warm, playful, first person, the
same length. Keep any 'woof'. Do not explain, do not add anything, do not use
quotation marks -- output ONLY the translated line.

Line: {text}"""

LANGUAGE_NAMES = {"pl": "Polish", "en": "English", "de": "German", "uk": "Ukrainian"}

# Appended when the reply will be spoken aloud. Written text tolerates lists,
# emoji and parentheses; a text-to-speech voice reads them out and it sounds
# ridiculous, so voice replies are plain, short spoken sentences.
VOICE_STYLE = """\
Your answer will be SPOKEN OUT LOUD by a speech synthesiser. So:
- one or two short sentences, never a list, never markdown, never emoji;
- write numbers and units as words a person would say them;
- no parentheses or asides -- just say the thing.
"""

CONTRACT = """\
Every reply must be EXACTLY ONE JSON object, nothing else. Two forms:

To answer the user:
{"thought": "<one sentence of private reasoning>", "say": "<your answer, in character>"}

To use a tool first:
{"thought": "<one sentence: why this tool>", "tool": "<name>", "args": {...}}

The "thought" field always comes first. After a tool call you will receive
its result (text, sometimes an image) and reply again -- with another tool
call or with your final "say".
"""

RULES = """\
Rules:
- Small talk and questions about yourself: answer directly, no tools.
- Any question about what you can SEE right now: call `look` first, then
  answer from the image.
- Questions about the room, what you know/explored so far: call `map`.
- Questions about where you walked recently: call `route`.
- "Go to X" / "walk to X" / any route ("walk around the table, then stop by
  the door"): call `navigate`, and put the user's instruction in `instruction`
  VERBATIM -- every step, direction and landmark, in their order. The walking
  model follows instructions, so shortening a route to one object name throws
  the route away.
- "Find X" / "look for X" / "where is X" when X is not in view: call `search`.
- "Stop" / "stay": call `stop`.
- navigate and search only START the behaviour; confirm cheerfully that you
  are on it. Use `status` when asked how it is going.
- Never repeat the exact same tool call twice in one turn; if a tool result
  already answers the question, say so.
"""


def system_prompt(
    tools: dict[str, Tool],
    voice: bool = False,
    lang_mode: str = "direct",
    language: str = "pl",
) -> str:
    """Assemble the turn's system prompt.

    Only a SPOKEN turn leaves English, and only in the `say` field: a typed
    reply, the private reasoning, tool arguments and everything in the trace
    stay English.
    """
    if not voice:
        lang = TEXT_LANGUAGE
    elif lang_mode == "translate":
        lang = TRANSLATE_STYLE
    else:
        lang = VOICE_LANGUAGE.format(language=LANGUAGE_NAMES.get(language, language))
    return (
        PERSONA
        + "\nYou can act through these tools:\n"
        + tools_prompt(tools)
        + "\n\n"
        + CONTRACT
        + "\n"
        + RULES
        + ("\n" + VOICE_STYLE if voice else "")
        + "\n"
        + lang
    )


class WojtekAgent:
    """One conversation with the dog; owns the rolling text-only history."""

    def __init__(
        self,
        llm: AgentLLM,
        tools: dict[str, Tool],
        max_tool_steps: int = MAX_TOOL_STEPS,
        keep_exchanges: int = KEEP_EXCHANGES,
        trace=None,
        turn_context: dict | None = None,
        lang_mode: str = "direct",
        reply_language: str = "pl",
    ):
        if lang_mode not in LANG_MODES:
            raise ValueError(f"lang_mode must be one of {LANG_MODES}, got {lang_mode!r}")
        self.lang_mode = lang_mode
        self.reply_language = reply_language
        self.llm = llm
        self.tools = tools
        self.max_tool_steps = max_tool_steps
        self.keep_exchanges = keep_exchanges
        self.trace = trace
        # Shared with build_tools: lets a tool see what the user actually said
        # this turn, not just the arguments the model chose to pass along.
        self.turn_context = turn_context if turn_context is not None else {}
        # Prompts are picked per turn: the spoken variant drops lists and
        # emoji, which a synthesiser would otherwise read out loud.
        self._system = system_prompt(tools, lang_mode=lang_mode, language=reply_language)
        self._system_voice = system_prompt(
            tools, voice=True, lang_mode=lang_mode, language=reply_language
        )
        self._history: list[dict] = []  # alternating user/assistant, text-only

    def reset(self) -> None:
        self._history = []
        self._trace("chat.reset")

    def _trace(self, kind: str, **fields) -> None:
        if self.trace is not None:
            self.trace.add(kind, **fields)

    async def _translate(self, text: str) -> str:
        """Render one finished line into the target language.

        Deliberately the SAME model rather than a dedicated MT system: it
        already knows who Wojtek is, so the persona survives the hop. A
        general translator turns "Woof! I'm on it!" into something correct
        and lifeless. On failure the English line goes out unchanged --
        saying something is better than saying nothing.
        """
        language = LANGUAGE_NAMES.get(self.reply_language, self.reply_language)
        prompt = TRANSLATE_PROMPT.format(language=language, text=text)
        try:
            out = await self.llm.chat([user_message(prompt)], max_tokens=200)
        except Exception as e:
            logger.warning(f"translation failed, sending the source line: {_safe_err(e)}")
            return text
        out = strip_think(out).strip().strip('"').strip()
        if not out:
            return text
        self._trace("chat.translate", source=text, translated=out,
                    language=self.reply_language)
        return out

    async def ask(self, text: str, voice: bool = False) -> dict:
        """One user turn. `voice=True` when the answer will be spoken aloud.

        Returns {ok, say, steps: [{tool, args, result}], debug} where debug
        carries the full trace of the turn -- one entry per LLM call with the
        raw model output, how it was classified (say / tool / parse_error /
        unknown_tool / repeat_nudge), latency/token metadata when the client
        exposes it, and the (truncated) tool result -- so the demo UI can show
        exactly what the model saw and did.
        """
        text = text.strip()
        if not text:
            return {"ok": False, "error": "empty message"}
        self.turn_context["user_text"] = text
        self._trace("chat.ask", text=text)
        turn: list[dict] = [user_message(text)]
        steps: list[dict] = []
        llm_calls: list[dict] = []
        last_call: tuple[str, str] | None = None
        say: str | None = None
        for _ in range(self.max_tool_steps + 1):
            messages = (
                [{"role": "system", "content": self._system_voice if voice else self._system}]
                + self._history
                + turn
            )
            raw = await self.llm.chat(messages)
            meta = getattr(self.llm, "last_meta", None) or {}
            entry = {
                "raw": raw[:2000],
                "latency_ms": meta.get("latency_ms"),
                "tokens": meta.get("tokens"),
            }
            llm_calls.append(entry)
            self._trace(
                "chat.llm",
                raw=raw,
                latency_ms=entry["latency_ms"],
                tokens=entry["tokens"],
                context_messages=len(messages),
            )
            try:
                # Pass the tool names so a reply that used the tool name as
                # its key still parses as a tool call instead of being read
                # out loud as JSON.
                reply = parse_agent_reply(raw, tuple(self.tools))
            except ValueError as e:
                logger.warning(f"agent reply unparseable: {_safe_err(e)}")
                entry["kind"] = "parse_error"
                turn.append({"role": "assistant", "content": [text_content(raw)]})
                turn.append(
                    user_message(
                        "(Your reply was not valid JSON. Reply with exactly one "
                        "JSON object per the contract.)"
                    )
                )
                continue
            entry["thought"] = reply.thought
            if reply.tool is None:
                say = reply.say or ""
                entry["kind"] = "say"
                entry["say"] = say
                break
            entry["kind"] = "tool"
            entry["tool"] = reply.tool
            entry["args"] = reply.args
            tool = self.tools.get(reply.tool)
            turn.append({"role": "assistant", "content": [text_content(raw)]})
            if tool is None:
                entry["kind"] = "unknown_tool"
                turn.append(
                    user_message(
                        f"(No tool named {reply.tool!r}. Available: "
                        f"{', '.join(self.tools)}. Answer with 'say' if no tool fits.)"
                    )
                )
                continue
            call_key = (reply.tool, json.dumps(reply.args, sort_keys=True))
            if call_key == last_call:
                entry["kind"] = "repeat_nudge"
                turn.append(
                    user_message(
                        "(You already made that exact tool call this turn; use its "
                        "result and answer the user with 'say' now.)"
                    )
                )
                continue
            last_call = call_key
            try:
                result: ToolResult = await tool.fn(reply.args)
            except Exception as e:
                logger.warning(f"tool {reply.tool} failed: {_safe_err(e)}")
                result = ToolResult(text=f"tool error: {_safe_err(e)}")
            entry["tool_result"] = result.text[:800]
            entry["tool_images"] = len(result.images)
            steps.append({"tool": reply.tool, "args": reply.args, "result": result.text})
            self._trace(
                "chat.tool",
                tool=reply.tool,
                args=reply.args,
                result=result.text,
                images=len(result.images),
            )
            turn.append(
                user_message(f"TOOL RESULT ({reply.tool}):\n{result.text}", result.images)
            )
        if say is None:
            # Tool budget burned without a final answer; give the user the
            # last concrete fact instead of silence.
            tail = steps[-1]["result"] if steps else "I got a bit tangled up"
            say = f"Woof -- here is what I found so far: {tail}"
        # Only the plain question/answer pair survives into the next turn.
        # Only a spoken line leaves English; a typed answer stays as written.
        said_in_english = None
        if voice and self.lang_mode == "translate" and say:
            said_in_english = say
            say = await self._translate(say)
        self._trace("chat.say", say=say, source=said_in_english,
                    llm_calls=len(llm_calls), tools=[s["tool"] for s in steps])
        self._history.append(user_message(text))
        self._history.append({"role": "assistant", "content": [text_content(say)]})
        del self._history[: -2 * self.keep_exchanges]
        return {
            "ok": True,
            "say": say,
            "steps": steps,
            "debug": {"llm_calls": llm_calls, "history_len": len(self._history)},
        }
