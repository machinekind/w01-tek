"""Stage timing: where a turn's seconds actually go.

The demo already records WHAT fired (`trace.py`); this records HOW LONG each
part took, in one vocabulary, so the answer to "why does the dog feel slow"
is a sorted table instead of a guess.  Three primitives, deliberately small:

  span()    one timed stage, emitted as a `perf.span` trace event
  Meter     aggregate for stages that fire at control rate (50 Hz), rolled
            up every ROLLUP_S seconds instead of one event per tick
  turn ids  a contextvar stamped on every span, so the spans of one spoken
            turn (endpointing -> ASR -> chat -> TTS) join into a critical
            path without threading an id through six call sites

Timings ride the existing trace: same JSONL file, same `/api/trace` feed,
same ring.  A trace file is therefore a complete latency recording, and
`perf_report.py` reads exactly what a live session already writes -- no
separate profiling mode to remember to switch on, because the mode you
forget to switch on is the one you need when the stage demo drags.

Correlation covers async naturally: `contextvars` are copied into a task at
creation, so a chat task started from the utterance handler inherits its
turn id, and the TTS task started inside that chat turn inherits it again.

Nothing here imports a model, a sim or ROS; the whole module is stdlib plus
the trace writer, and it degrades to a no-op when no trace is bound.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar

from loguru import logger

# Aggregated stages report every this many seconds. Long enough that a 50 Hz
# loop costs ~0.2 events/s of trace, short enough that a stall shows up in
# the window it happened in.
ROLLUP_S = 5.0

# The critical path of a spoken turn, in pipeline order. The report walks
# this to show where the mic-to-voice latency went; anything else is
# background work. It spans both stacks -- the demo's single agent turn
# (`chat.turn`) and the ROS pipeline's router + brain hops -- because a trace
# only ever contains one of them and the report skips what is absent.
SPOKEN_PATH = (
    "mic.endpoint",        # silence we wait out before deciding the human stopped
    "asr.transcribe",      # speech -> text
    "brain.route",         # ROS: transcript -> routed intent
    "chat.turn",           # demo: the agent turn (llm calls + tools nested inside)
    "llm.first_sentence",  # ROS: intent -> the brain's first spoken sentence
    "tts.first_audio",     # reply text -> first PCM frame on the wire
)

_trace = None                                   # bound once by the app
_turn: ContextVar[str | None] = ContextVar("wojtek_perf_turn", default=None)
_turn_t0: ContextVar[float | None] = ContextVar("wojtek_perf_turn_t0", default=None)
_marks: ContextVar[dict | None] = ContextVar("wojtek_perf_marks", default=None)
_turn_seq = 0


def bind(trace) -> None:
    """Point spans at the session trace. Called once at startup; before it
    (and in unit tests) every span is a no-op with real timing still returned
    to the caller."""
    global _trace
    _trace = trace


def bound_trace():
    return _trace


def start_turn(kind: str = "turn", **fields) -> str:
    """Open a correlation scope and return its id.

    Called where a turn is BORN -- the moment an utterance closes, or a typed
    message arrives -- so every later stage carries the same id and the
    report can add them up.
    """
    global _turn_seq
    _turn_seq += 1
    turn_id = f"{kind}{_turn_seq}"
    _turn.set(turn_id)
    _turn_t0.set(time.monotonic())
    _marks.set({})
    # `turn_kind`, not `kind`: every trace event already has a `kind` field.
    emit("perf.turn", turn=turn_id, turn_kind=kind, **fields)
    return turn_id


def current_turn() -> str | None:
    return _turn.get()


def set_turn(turn_id: str | None) -> None:
    """Adopt an existing id (a background goal continuing a turn's work);
    None ends the scope, so nothing later is attributed to a finished turn."""
    _turn.set(turn_id)
    if turn_id is None:
        _turn_t0.set(None)


def mark(name: str) -> None:
    """Stamp a moment inside the current turn, to be measured from later.

    For gaps between things that happen in different tasks: the instant the
    reply text reached the browser is one task, the instant the first audio
    frame left is another, and the wait between them is what a person
    actually complains about.
    """
    marks = dict(_marks.get() or {})
    marks[name] = time.monotonic()
    _marks.set(marks)


def since_mark_ms(name: str) -> float | None:
    """Milliseconds since `mark(name)`, or None if it was never stamped."""
    t = (_marks.get() or {}).get(name)
    return None if t is None else (time.monotonic() - t) * 1000.0


def turn_elapsed_ms() -> float | None:
    """Milliseconds since this turn began, or None outside a turn.

    This is what the human actually waits: the clock starts when the mic
    decided the sentence was over, not when some later stage got around to
    starting.
    """
    t0 = _turn_t0.get()
    return None if t0 is None else (time.monotonic() - t0) * 1000.0


def emit(kind: str, **fields) -> None:
    if _trace is not None:
        _trace.add(kind, **fields)


def record(stage: str, ms: float, *, trace=None, ok: bool = True, **fields) -> None:
    """Emit one already-measured stage (for timings we get from elsewhere:
    a server's own latency number, a wall clock difference)."""
    sink = trace if trace is not None else _trace
    if sink is None:
        return
    sink.add(
        "perf.span",
        stage=stage,
        ms=round(float(ms), 1),
        turn=_turn.get(),
        ok=ok,
        **fields,
    )


@contextmanager
def span(stage: str, *, trace=None, **fields):
    """Time a block and emit it as `perf.span`.

    Works for sync and async code alike (`with span("x"): await f()`), so
    there is one spelling to remember.  A raising block is still recorded,
    tagged `ok=false` -- a stage that takes four seconds and then throws is
    exactly the kind of latency worth seeing.

    The yielded dict is writable: put facts known only at the end (token
    counts, audio seconds, whether the model called a tool) into it and they
    land on the event.
    """
    extra: dict = {}
    t0 = time.monotonic()
    ok = True
    try:
        yield extra
    except BaseException:
        ok = False
        raise
    finally:
        record(
            stage,
            (time.monotonic() - t0) * 1000.0,
            trace=trace,
            ok=ok,
            **{**fields, **extra},
        )


class Timer:
    """Open-ended stopwatch for stages whose start and end are in different
    callbacks (say() -> first audio frame, utterance start -> utterance end).

    `stop()` is idempotent: a cancelled reply (barge-in) that never reaches
    its first frame simply never reports, instead of reporting a lie.
    """

    def __init__(self, stage: str, *, trace=None, **fields):
        self.stage = stage
        self.fields = fields
        self.trace = trace
        self.t0 = time.monotonic()
        self.turn = _turn.get()
        self._done = False

    @property
    def ms(self) -> float:
        return (time.monotonic() - self.t0) * 1000.0

    def stop(self, **fields) -> float:
        ms = self.ms
        if self._done:
            return ms
        self._done = True
        sink = self.trace if self.trace is not None else _trace
        if sink is not None:
            sink.add(
                "perf.span",
                stage=self.stage,
                ms=round(ms, 1),
                turn=self.turn,
                ok=True,
                **{**self.fields, **fields},
            )
        return ms

    def cancel(self) -> None:
        self._done = True


class Meter:
    """Aggregate a hot stage instead of tracing every occurrence.

    The control loop runs at 50 Hz: one span per tick per stage would be
    ~200 trace events a second, which drowns the decision events the trace
    exists for.  A meter keeps count/total/max in memory and emits one
    `perf.rollup` per stage every ROLLUP_S -- enough to rank the loop's parts
    and to catch a stall (max), without the flood.
    """

    def __init__(self, *, rollup_s: float = ROLLUP_S, trace=None):
        self.rollup_s = rollup_s
        self.trace = trace
        self._stats: dict[str, list[float]] = {}   # stage -> [n, total_ms, max_ms]
        self._t0 = time.monotonic()

    @contextmanager
    def time(self, stage: str):
        t0 = time.monotonic()
        try:
            yield
        finally:
            self.add(stage, (time.monotonic() - t0) * 1000.0)

    def add(self, stage: str, ms: float) -> None:
        s = self._stats.get(stage)
        if s is None:
            self._stats[stage] = [1.0, ms, ms]
        else:
            s[0] += 1
            s[1] += ms
            s[2] = max(s[2], ms)

    def due(self) -> bool:
        return bool(self._stats) and (time.monotonic() - self._t0) >= self.rollup_s

    def flush(self, force: bool = False) -> None:
        """Emit one rollup per stage and start a new window."""
        if not self._stats or (not force and not self.due()):
            return
        window_s = time.monotonic() - self._t0
        sink = self.trace if self.trace is not None else _trace
        stats, self._stats, self._t0 = self._stats, {}, time.monotonic()
        if sink is None:
            return
        for stage, (n, total, peak) in stats.items():
            sink.add(
                "perf.rollup",
                stage=stage,
                n=int(n),
                ms=round(total, 1),
                ms_max=round(peak, 1),
                ms_mean=round(total / max(n, 1.0), 2),
                window_s=round(window_s, 2),
            )

    def tick(self) -> None:
        """Call once per loop iteration; flushes when the window is up."""
        if self.due():
            try:
                self.flush()
            except Exception as e:  # metering must never take the loop down
                logger.warning(f"perf rollup dropped ({e})")
