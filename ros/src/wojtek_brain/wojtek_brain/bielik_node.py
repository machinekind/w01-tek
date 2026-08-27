"""Bielik node: the Polish mouth of the stack.

Three jobs (design doc, Decision 2):
- `chat` intents → generate the reply, streaming sentences to TTS the
  moment punctuation closes them.
- `nav` intents → speak an immediate canned acknowledgement (zero latency;
  the Qwen agent does the actual walking and reports outcomes itself).
- /wojtek/say_en (English sentences from the VLM agent) → translate to
  Polish, streamed the same way.

Barge-in: /wojtek/audio/speech_started sets the cancel event — in-flight
generation stops between tokens and queued work for the interrupted reply
is dropped.  One worker thread owns all LLM traffic, so replies never
interleave.
"""

from __future__ import annotations

import queue
import re
import threading
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty

from wojtek_agent_msgs.msg import RoutedIntent, Sentence, Transcript

from . import phrases
from .llm_client import ChatClient
from .sentences import SentenceAssembler, speakable

# All prompt text is editable without touching code: prompts/bielik/*.txt
# (persona, translate; canned acks live in the phrases bank).
# Everything in there lands on Bielik and stays POLISH; the English Qwen
# prompts live in training/wojtek_rl/agent/prompts/qwen/.
_PROMPTS = Path(__file__).resolve().parent / "prompts" / "bielik"


def _prompt(name: str) -> str:
    return (_PROMPTS / f"{name}.txt").read_text(encoding="utf-8").strip()



PERSONA = _prompt("persona")
TRANSLATE_PROMPT = _prompt("translate")

# The router lumps "go to X" and "find X" into one nav intent; the ack
# should not promise walking when the ask was a search.
_SEARCH_ASK_RE = re.compile(r"\b(znajdź|poszukaj|szukaj|odszukaj|wyszukaj)\b",
                            re.IGNORECASE)
HISTORY_TURNS = 6  # rolling text-only chat memory


class BielikNode(Node):
    def __init__(self):
        super().__init__("wojtek_bielik")
        self.declare_parameter("author_chat", True)  # false when vlm_agent runs
        self.declare_parameter("url", "http://127.0.0.1:8091")
        self.declare_parameter("model", "speakleash/Bielik-4.5B-v3.0-Instruct-FP8-Dynamic")
        self.declare_parameter("temperature", 0.7)
        self.declare_parameter("max_tokens", 200)

        self.client = ChatClient(
            self.get_parameter("url").value, self.get_parameter("model").value
        )
        self.temperature = self.get_parameter("temperature").value
        self.max_tokens = self.get_parameter("max_tokens").value

        self.pub_say = self.create_publisher(Sentence, "/wojtek/say", 10)
        self.create_subscription(RoutedIntent, "/wojtek/intent", self.on_intent, 10)
        self.create_subscription(Sentence, "/wojtek/say_en", self.on_english, 10)
        self.create_subscription(
            Empty, "/wojtek/audio/speech_started", self.on_barge_in, 10
        )
        # Last heard question: context for translations, so the Polish
        # rendering actually answers what was asked.
        self._last_heard = ""
        self.create_subscription(Transcript, "/wojtek/asr/final", self.on_heard, 10)

        self.history: list[dict] = []
        self._cancel = threading.Event()
        self._author_chat = bool(self.get_parameter("author_chat").value)
        self._q: queue.Queue = queue.Queue()
        threading.Thread(target=self._work, daemon=True).start()

    # -- subscriptions ------------------------------------------------------
    def on_intent(self, msg: RoutedIntent):
        if msg.intent == "chat":
            # With the VLM agent in the graph, Bielik never AUTHORS -- it
            # only translates /wojtek/say_en. A 4.5B authoring blind produced
            # garbled echoes and stage-direction prose on camera (2026-08-26,
            # "rzucam radosne spojrzenie Elo! Masz na imię? Wojtek!").
            if self._author_chat:
                self._q.put(("chat", msg.utterance_id, msg.text))
        elif msg.intent == "nav":
            kind = ("search_ack" if _SEARCH_ASK_RE.search(msg.text or "")
                    else "nav_ack")
            self._speak_now(msg.utterance_id, phrases.sample(kind))
        elif msg.intent == "cancel":
            self._cancel.set()
            self._speak_now(msg.utterance_id, phrases.sample('cancel_ack'))
        # visual: the VLM agent answers; its sentences arrive on /wojtek/say_en.
        # system: v1 ignores (volume/reset live elsewhere).

    def on_heard(self, msg: Transcript):
        self._last_heard = msg.text

    def on_english(self, msg: Sentence):
        self._q.put(("translate", msg.utterance_id, msg.text))

    def on_barge_in(self, _msg: Empty):
        self._cancel.set()
        # Drop queued work too — the human is saying something new.
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass

    # -- worker -------------------------------------------------------------
    def _work(self):
        while True:
            kind, uid, text = self._q.get()
            self._cancel.clear()
            try:
                if kind == "chat":
                    self._generate_chat(uid, text)
                else:
                    self._generate_translation(uid, text)
            except Exception as e:  # LLM down must not kill the mouth
                self.get_logger().warning(f"generation failed for {uid}: {e}")

    def _generate_chat(self, uid: str, user_text: str):
        messages = (
            [{"role": "system", "content": PERSONA}]
            + self.history[-2 * HISTORY_TURNS :]
            + [{"role": "user", "content": user_text}]
        )
        reply = self._stream_out(uid, messages, source="bielik")
        if reply:
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": reply})

    def _generate_translation(self, uid: str, english: str):
        user = english
        if self._last_heard:
            user = f"Pytanie użytkownika: {self._last_heard}\n\nTekst: {english}"
        messages = [
            {"role": "system", "content": TRANSLATE_PROMPT},
            {"role": "user", "content": user},
        ]
        self._stream_out(uid, messages, source="qwen", temperature=0.3)

    def _stream_out(self, uid, messages, source, temperature=None) -> str:
        assembler = SentenceAssembler()
        seq = 0
        spoken: list[str] = []
        for token in self.client.stream(
            messages,
            cancel=self._cancel,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens,
        ):
            for sentence in assembler.feed(token):
                seq = self._publish(uid, seq, sentence, source, final=False)
                spoken.append(sentence)
        if self._cancel.is_set():
            return ""  # interrupted reply is not history — it was not heard
        tail = assembler.flush()
        for i, sentence in enumerate(tail):
            seq = self._publish(uid, seq, sentence, source, final=(i == len(tail) - 1))
            spoken.append(sentence)
        if tail == [] and spoken:
            # everything went out as non-final; close the reply
            self._publish(uid, seq, "", source, final=True)
        return " ".join(spoken)

    def _speak_now(self, uid: str, text: str):
        self._publish(uid, 0, text, source="system", final=True)

    def _publish(self, uid, seq, text, source, final) -> int:
        msg = Sentence()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.utterance_id = uid
        msg.seq = seq
        msg.text = speakable(text)
        msg.final = final
        msg.source = source
        self.pub_say.publish(msg)
        return seq + 1


def main(args=None):
    rclpy.init(args=args)
    node = BielikNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
