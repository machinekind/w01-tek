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
import re

from loguru import logger

from wojtek_agent.llm import AgentLLM, text_content, user_message
from wojtek_agent.parsing import parse_agent_reply, strip_think
from wojtek_agent.prompts import load
from wojtek_agent.tools import Tool, ToolResult, tools_prompt
from wojtek_rl.vlm_nav import _safe_err

MAX_TOOL_STEPS = 4
KEEP_EXCHANGES = 3

# Movement imperatives (PL + EN) that must end in a navigate call.  Search
# verbs (znajdź/poszukaj) are deliberately absent -- they belong to `search`
# and the model calls that one reliably.  STOP_HINT_RE is the same guard for
# stop: a stop heard during a live goal that only produced words left the
# robot walking (live 2026-08-14).
NAV_HINT_RE = re.compile(
    r"\b(idź|pójdź|podejdź|chodź|zaprowadź|obejdź|okrąż|omiń|skręć|zawróć|"
    r"cofnij|wróć|biegnij|ruszaj|jedź|stań (przy|obok|koło)|zbliż się|"
    r"oddal się|go to|walk (to|around)|come (here|back)|head to)\b",
    re.IGNORECASE,
)
STOP_HINT_RE = re.compile(
    r"\b(stop|stój|przestań|zatrzymaj|dosyć|wystarczy|nie idź|zostań|"
    r"stand still|halt)\b",
    re.IGNORECASE,
)

PERSONA = load("persona")

# Language policy: the system runs in English -- reasoning, tool arguments,
# traces, logs -- because that is where a small model is strongest and where
# anyone debugging this wants to read. The ONE thing that comes out in
# another language is the sentence the dog actually speaks.
TEXT_LANGUAGE = load("text_language")

VOICE_LANGUAGE = load("voice_language")

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

TRANSLATE_STYLE = load("translate_style")

TRANSLATE_PROMPT = load("translate")

LANGUAGE_NAMES = {"pl": "Polish", "en": "English", "de": "German", "uk": "Ukrainian"}

# Appended when the reply will be spoken aloud. Written text tolerates lists,
# emoji and parentheses; a text-to-speech voice reads them out and it sounds
# ridiculous, so voice replies are plain, short spoken sentences.
VOICE_STYLE = load("voice_style")

CONTRACT = load("contract")

RULES = load("rules")


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
        prompt = TRANSLATE_PROMPT.format(
            language=language, text=text,
            question=str(self.turn_context.get("user_text", "") or "-"),
        )
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
        said_fallback = False
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
                entry["kind"] = "fallback_say" if reply.fallback else "say"
                entry["say"] = say
                said_fallback = reply.fallback
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
        # Nav guard: a movement instruction that produced words but no action
        # is a dropped command, not an answer.  Measured live: after a
        # contract collapse the model narrates ("Wchodzi w kierunku...")
        # instead of calling navigate.  The instruction goes to the navigator
        # VERBATIM, same rule as tools.keep_full_instruction.
        if not steps and "stop" in self.tools and STOP_HINT_RE.search(text):
            try:
                result = await self.tools["stop"].fn({})
                steps.append({"tool": "stop", "args": {}, "result": result.text})
                llm_calls.append({"kind": "stop_guard"})
                self._trace("chat.stop_guard", result=result.text)
                if said_fallback:
                    say = "Już się zatrzymuję!"
            except Exception as e:
                logger.warning(f"stop guard failed: {_safe_err(e)}")
        if not steps and "navigate" in self.tools and NAV_HINT_RE.search(text):
            try:
                result = await self.tools["navigate"].fn({"instruction": text})
                steps.append(
                    {"tool": "navigate", "args": {"instruction": text}, "result": result.text}
                )
                llm_calls.append({"kind": "nav_guard", "args": {"instruction": text}})
                self._trace("chat.nav_guard", instruction=text, result=result.text)
                if said_fallback:
                    say = "Jasne, już się ruszam!"
            except Exception as e:
                logger.warning(f"nav guard navigate failed: {_safe_err(e)}")
        # Only the plain question/answer pair survives into the next turn.
        # Only a spoken line leaves English; a typed answer stays as written.
        said_in_english = None
        if voice and self.lang_mode == "translate" and say:
            said_in_english = say
            say = await self._translate(say)
        self._trace("chat.say", say=say, source=said_in_english,
                    llm_calls=len(llm_calls), tools=[s["tool"] for s in steps])
        # Quarantine: a fallback (non-JSON) line never enters the rolling
        # history -- one narration in history teaches the model to drop the
        # JSON contract on every later action turn (observed 2026-08-13:
        # navigate stopped firing for the rest of the session).
        if not said_fallback:
            self._history.append(user_message(text))
            self._history.append({"role": "assistant", "content": [text_content(say)]})
            del self._history[: -2 * self.keep_exchanges]
        return {
            "ok": True,
            "say": say,
            "steps": steps,
            "debug": {"llm_calls": llm_calls, "history_len": len(self._history)},
        }
