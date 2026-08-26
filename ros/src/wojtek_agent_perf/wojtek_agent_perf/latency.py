"""Turn every message stamp into a stage timing. rclpy-free, so it is tested
without a ROS graph.

The agent pipeline is a chain of topics, each message carrying the header
stamp of the moment it was produced and the `utterance_id` it belongs to.
That is enough to reconstruct where a turn's seconds went without touching a
single node's hot path: subtract consecutive stamps of one utterance.

    /wojtek/audio/speech (end_of_utterance)   the human stopped talking
    /wojtek/asr/final                         text exists
    /wojtek/intent                            routed
    /wojtek/say_en (first sentence)           the VLM agent answered (nav/visual)
    /wojtek/say (first sentence)              the brain has something to say
    /wojtek/tts/audio (first frame)           the human hears it

Stage names are deliberately the SAME vocabulary the demo stack emits
(`wojtek_rl.perf`), and the output is the same `perf.span` JSONL, so one
report tool reads both stacks:  ./training/run.sh perf <file.jsonl>

Two honesty rules:

  * `mic.endpoint` is not observable from outside the VAD (the closing chunk
    is the first evidence anything happened), so it is reported from the
    configured silence timeout and tagged `assumed`.  It is on the critical
    path of every turn and dropping it would flatter the total.
  * A stage is emitted only when BOTH of its endpoints were seen.  A turn
    that died halfway reports what happened up to that point and nothing
    after it -- a missing stage means the turn broke, not that it was fast.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

# stage -> (from event, to event). Order is the pipeline order.
STAGES: tuple[tuple[str, str, str], ...] = (
    ("asr.transcribe", "speech_end", "asr_final"),
    ("brain.route", "asr_final", "intent"),
    # Only present on turns the VLM agent answered (nav/visual): the agent's
    # whole tool loop, then the Bielik hop that renders it in Polish.
    ("agent.turn", "intent", "say_en_first"),
    ("brain.translate", "say_en_first", "say_first"),
    ("llm.first_sentence", "intent", "say_first"),
    ("llm.reply", "intent", "say_final"),
    ("tts.first_audio", "say_first", "audio_first"),
    # The whole felt wait, mic to sound: also the sum of the chain above plus
    # the handoffs between nodes, which is exactly the gap worth seeing.
    ("voice.reply", "speech_end", "audio_first"),
)

EVENTS = ("speech_end", "asr_final", "intent", "say_en_first", "say_first",
          "say_final", "audio_first")


def stamp_to_s(stamp) -> float:
    """ROS Time message (or anything with sec/nanosec) -> float seconds."""
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class Turn:
    """One utterance's stamps, and the spans they imply."""

    def __init__(self, utterance_id: str, endpoint_wait_s: float = 0.0):
        self.utterance_id = utterance_id
        self.endpoint_wait_s = endpoint_wait_s
        self.at: dict[str, float] = {}
        self.emitted: set[str] = set()
        self.text = ""

    def mark(self, event: str, t: float) -> None:
        """Record when an event happened. First stamp wins: a reply's later
        sentences and audio frames must not overwrite the first one, which is
        the one the latency question is about."""
        if event not in EVENTS:
            raise ValueError(f"unknown pipeline event {event!r}")
        if event == "say_final" or event not in self.at:
            self.at[event] = t

    def ready(self) -> list[tuple[str, float]]:
        """Stages whose both ends are known and that were not reported yet."""
        out = []
        for stage, a, b in STAGES:
            if stage in self.emitted or a not in self.at or b not in self.at:
                continue
            self.emitted.add(stage)
            out.append((stage, (self.at[b] - self.at[a]) * 1000.0))
        return out

    def complete(self) -> bool:
        return "audio_first" in self.at


class LatencyProbe:
    """Accumulates turns and emits `perf.span` records as stages complete.

    Bounded by design: a live stage session runs for hours and a probe that
    remembers every utterance is a slow leak, so finished turns are dropped
    once `max_turns` is exceeded (oldest first).
    """

    def __init__(self, endpoint_wait_s: float = 0.7, max_turns: int = 256,
                 t0: float | None = None):
        self.endpoint_wait_s = endpoint_wait_s
        self.max_turns = max_turns
        self.t0 = t0
        self.turns: dict[str, Turn] = {}
        self.seq = 0

    def _turn(self, utterance_id: str) -> Turn:
        turn = self.turns.get(utterance_id)
        if turn is None:
            turn = Turn(utterance_id, self.endpoint_wait_s)
            self.turns[utterance_id] = turn
            while len(self.turns) > self.max_turns:
                self.turns.pop(next(iter(self.turns)))
        return turn

    def observe(self, event: str, utterance_id: str, t: float,
                text: str = "") -> list[dict]:
        """Feed one message. Returns the span records it completed."""
        if not utterance_id:
            return []
        if self.t0 is None:
            self.t0 = t
        turn = self._turn(utterance_id)
        if text:
            turn.text = text
        records = []
        if event == "speech_end":
            # Dead time the human already spent waiting before this stamp.
            records.append(self._record(turn, "mic.endpoint",
                                        self.endpoint_wait_s * 1000.0, t,
                                        assumed=True))
        turn.mark(event, t)
        for stage, ms in turn.ready():
            records.append(self._record(turn, stage, ms, t))
        return records

    def _record(self, turn: Turn, stage: str, ms: float, t: float,
                **fields) -> dict:
        self.seq += 1
        return {
            "seq": self.seq,
            "t": round(t - (self.t0 or t), 3),
            "wall": datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds"),
            "kind": "perf.span",
            "stage": stage,
            "ms": round(ms, 1),
            "turn": turn.utterance_id,
            "ok": True,
            **fields,
        }


class SpanWriter:
    """Append-only JSONL, line buffered, never fatal.

    Same file format and failure policy as the demo stack's trace writer: a
    probe that takes the robot down to report a number is worse than no
    number at all.
    """

    def __init__(self, path, on_error=None):
        self.path = path
        self.on_error = on_error or (lambda e: None)
        self._fh = None
        self._failed = False
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("a", buffering=1)
        except OSError as e:
            self.on_error(e)

    def write(self, record: dict) -> None:
        if self._fh is None:
            return
        try:
            self._fh.write(json.dumps(record, default=str) + "\n")
        except (OSError, ValueError) as e:
            if not self._failed:
                self._failed = True
                self.on_error(e)

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None
