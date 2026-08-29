"""Read a session trace and say where the robot's time went.

Input is the JSONL a live session already writes (`runs/agent_traces/*.jsonl`
or `AGENT_TRACE=`), so profiling is a question you ask afterwards rather than
a mode you had to remember to switch on.  Three views, because "slow" means
three different things in this stack:

  WHERE THE TIME GOES   every timed stage, sorted by total -- the ranked list
                        to work down when accelerating the system
  HOT LOOP              the 50 Hz control loop's parts, from the rollups; a
                        loop.tick over 20 ms is the robot walking badly, not
                        the robot answering slowly
  SPOKEN TURNS          the critical path a human actually experiences,
                        endpoint -> ASR -> chat (llm + tools) -> first sound

The distinction that matters: total time and felt latency are different
rankings.  `nav.execute` (the robot walking) and prefetched `nav.decide`
dominate wall-clock while costing nobody any waiting, whereas 0.7 s of VAD
endpointing is invisible in a totals table and is felt in every single turn.
Both tables are printed; neither alone is the answer.

Usage:
    ./training/run.sh perf                      # newest trace in runs/agent_traces
    ./training/run.sh perf <file.jsonl> [...]   # one or more explicit traces
    ./training/run.sh perf --json               # machine-readable, for CI/diffs
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from wojtek_rl import paths
from wojtek_rl.perf import SPOKEN_PATH

# Umbrella spans CONTAIN other spans (a chat turn contains its model calls, a
# search goal contains everything it did). Ranking them beside their own parts
# would double count every second and put the total at the top of the list, so
# they get their own table: useful as headline numbers, useless as targets.
UMBRELLA_STAGES = (
    "voice.reply", "chat.turn", "tts.speak", "nav.goal", "search.goal",
    "search.scan_carousel", "search.leg", "search.approach_verify",
    "llm.reply",   # ROS: the whole reply, which starts with llm.first_sentence
)

# Stages that cost wall-clock but no waiting: the robot physically walking,
# and the inference deliberately overlapped with it. Real seconds, so they are
# ranked -- but tagged, so nobody optimises the number a human never felt.
BACKGROUND_STAGES = ("nav.execute", "search.turn", "nav.decide_prefetch")


# Stages that contain others ONLY when those others were recorded. A remote
# TTS call breaks down into server GPU time plus the wire, and then the call
# itself is a container; a local engine has no such detail and the call IS
# the work. Deciding per trace keeps both readable.
CONDITIONAL_UMBRELLA = {
    "tts.synth": ("tts.request",),
    "tts.request": ("tts.server_gpu", "tts.network", "tts.server_queue"),
}


def stage_role(stage: str, present: set[str] | None = None) -> str:
    """work | background | umbrella -- which table a stage belongs in."""
    if stage in UMBRELLA_STAGES:
        return "umbrella"
    children = CONDITIONAL_UMBRELLA.get(stage)
    if children and present and any(c in present for c in children):
        return "umbrella"
    if stage in BACKGROUND_STAGES:
        return "background"
    return "work"


def load_events(path: Path) -> list[dict]:
    """Parse one trace file, skipping malformed lines.

    A trace is written live and line-buffered: a killed process can leave a
    half-written last line, and that must not cost the whole session's data.
    """
    events = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    return events


def latest_trace(runs_dir: Path | None = None) -> Path | None:
    d = (runs_dir or paths.PROJECT_DIR / "runs") / "agent_traces"
    files = sorted(d.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def span_stats(events: list[dict]) -> list[dict]:
    """Per-stage aggregate of `perf.span` events, biggest total first."""
    by_stage: dict[str, list[float]] = {}
    fails: dict[str, int] = {}
    for e in events:
        if e.get("kind") != "perf.span":
            continue
        stage = e.get("stage")
        ms = e.get("ms")
        if stage is None or ms is None:
            continue
        by_stage.setdefault(stage, []).append(float(ms))
        if e.get("ok") is False:
            fails[stage] = fails.get(stage, 0) + 1
    rows = []
    present = set(by_stage)
    for stage, values in by_stage.items():
        rows.append(
            {
                "stage": stage,
                "n": len(values),
                "total_ms": sum(values),
                "mean_ms": statistics.fmean(values),
                "p50_ms": _pct(values, 0.5),
                "p95_ms": _pct(values, 0.95),
                "max_ms": max(values),
                "failed": fails.get(stage, 0),
                "role": stage_role(stage, present),
                "background": stage_role(stage, present) == "background",
            }
        )
    rows.sort(key=lambda r: r["total_ms"], reverse=True)
    return rows


def rollup_stats(events: list[dict]) -> list[dict]:
    """Per-stage aggregate of the hot-loop `perf.rollup` windows."""
    acc: dict[str, dict] = {}
    for e in events:
        if e.get("kind") != "perf.rollup":
            continue
        stage = e.get("stage")
        if stage is None:
            continue
        row = acc.setdefault(
            stage, {"stage": stage, "n": 0, "total_ms": 0.0, "max_ms": 0.0, "window_s": 0.0}
        )
        row["n"] += int(e.get("n") or 0)
        row["total_ms"] += float(e.get("ms") or 0.0)
        row["max_ms"] = max(row["max_ms"], float(e.get("ms_max") or 0.0))
        row["window_s"] += float(e.get("window_s") or 0.0)
    rows = list(acc.values())
    for r in rows:
        r["mean_ms"] = r["total_ms"] / max(r["n"], 1)
    rows.sort(key=lambda r: r["total_ms"], reverse=True)
    return rows


def turns(events: list[dict]) -> list[dict]:
    """Rebuild each turn from the spans that carry its id.

    A turn is whatever shares a `turn` field: the mic's endpoint wait, the
    recognition, the chat turn with its nested model calls and tools, and the
    reply's time-to-first-sound.  `voice.reply` is the measured end-to-end
    wait; the per-stage sum is what it decomposes into (they differ by the
    handoffs between stages, which is itself worth seeing).
    """
    out: dict[str, dict] = {}
    for e in events:
        kind = e.get("kind")
        turn = e.get("turn")
        if not turn:
            continue
        row = out.setdefault(
            turn,
            {"turn": turn, "kind": turn.rstrip("0123456789"), "t": e.get("t"),
             "stages": {}, "tools": [], "llm_calls": 0},
        )
        if kind == "perf.turn":
            row["t"] = e.get("t")
            continue
        if kind != "perf.span":
            continue
        stage, ms = e.get("stage"), float(e.get("ms") or 0.0)
        row["stages"][stage] = row["stages"].get(stage, 0.0) + ms
        if stage == "llm.chat":
            row["llm_calls"] += 1
        elif stage.startswith("tool."):
            row["tools"].append(stage[len("tool."):])
    rows = []
    for row in out.values():
        stages = row["stages"]
        row["path_ms"] = sum(stages.get(s, 0.0) for s in SPOKEN_PATH)
        row["measured_ms"] = stages.get("voice.reply")
        row["tool_ms"] = sum(ms for s, ms in stages.items() if s.startswith("tool."))
        row["llm_ms"] = sum(
            ms for s, ms in stages.items()
            if s.startswith("llm.") and stage_role(s) != "umbrella"
        )
        row["spoke"] = "tts.first_audio" in stages
        rows.append(row)
    rows.sort(key=lambda r: (r["t"] is None, r["t"]))
    return rows


def critical_path(turn_rows: list[dict]) -> list[dict]:
    """Median of each critical-path stage over the turns that completed one.

    Median, not mean: one 30 s turn where the endpoint was reachable but the
    model was reloading should not redefine what a turn costs.
    """
    done = [r for r in turn_rows if r["spoke"]]
    if not done:
        done = [r for r in turn_rows if r["stages"].get("chat.turn")]
    rows = []
    for stage in SPOKEN_PATH:
        values = [r["stages"][stage] for r in done if stage in r["stages"]]
        if not values:
            continue
        rows.append({"stage": stage, "n": len(values),
                     "median_ms": statistics.median(values)})
    return rows


# The gap between the answer appearing and the answer being heard, split
# into what it is actually made of. Only the FIRST chunk matters: nothing is
# audible until it is finished, and everything after it plays behind speech
# already coming out of the speaker.
VOICE_OUT_STAGES = (
    "tts.server_queue",   # waiting behind another sentence on the TTS server
    "tts.server_gpu",     # the voice model itself
    "tts.network",        # shipping the audio back
    "tts.synth",          # the whole synthesis call (local engines: all of it)
    "tts.stream",         # pushing frames onto the socket
)


def voice_out(events: list[dict]) -> dict:
    """How long after the text does sound start, and what made up the wait.

    Reported for the first chunk of each reply, because that is the only one
    a person waits through.
    """
    first: dict[str, list[float]] = {}
    gap: list[float] = []
    total: list[float] = []
    chunks: list[int] = []
    first_chars: list[int] = []
    rtf: list[float] = []
    for e in events:
        if e.get("kind") != "perf.span":
            continue
        stage, ms = e.get("stage"), float(e.get("ms") or 0.0)
        if stage == "reply.text_to_sound":
            gap.append(ms)
            if e.get("first_chunk_chars"):
                first_chars.append(int(e["first_chunk_chars"]))
        elif stage == "tts.first_audio":
            total.append(ms)
            if e.get("chunks"):
                chunks.append(int(e["chunks"]))
        # First chunk, and with a streaming engine its first PART: the later
        # parts arrive already-buffered in milliseconds and would drag the
        # median down to a number nobody waited for.
        elif (stage in VOICE_OUT_STAGES
              and e.get("chunk", 0) in (0, None)
              and e.get("part", 0) in (0, None)):
            first.setdefault(stage, []).append(ms)
            # RTF only from whole-utterance synthesis: for a streamed piece
            # the span covers the server's generation of that piece, so
            # dividing by the piece's own audio would overstate it wildly.
            if stage == "tts.synth" and e.get("rtf") and not e.get("streamed"):
                rtf.append(float(e["rtf"]))
    return {
        "text_to_sound_ms": statistics.median(gap) if gap else None,
        "first_audio_ms": statistics.median(total) if total else None,
        "worst_ms": max(gap or total) if (gap or total) else None,
        "n": len(gap or total),
        "chunks": statistics.median(chunks) if chunks else None,
        "first_chunk_chars": statistics.median(first_chars) if first_chars else None,
        "rtf": statistics.median(rtf) if rtf else None,
        "stages": [
            {"stage": s, "median_ms": statistics.median(v), "n": len(v)}
            for s, v in sorted(first.items(), key=lambda kv: -statistics.median(kv[1]))
        ],
    }


# -- rendering ---------------------------------------------------------------


def _ms(ms: float) -> str:
    if ms >= 10_000:
        return f"{ms / 1000:.0f}s"
    if ms >= 1000:
        return f"{ms / 1000:.2f}s"
    return f"{ms:.0f}ms"


def _bar(frac: float, width: int = 22) -> str:
    return "#" * max(0, min(width, int(round(frac * width))))


def render(events: list[dict], sources: list[Path], top: int = 0) -> str:
    spans = span_stats(events)
    rollups = rollup_stats(events)
    turn_rows = turns(events)
    path = critical_path(turn_rows)
    wall_s = max((float(e.get("t") or 0.0) for e in events), default=0.0)
    lines: list[str] = []
    names = ", ".join(p.name for p in sources)
    lines.append(f"trace: {names}")
    n_spans = sum(r["n"] for r in spans)
    lines.append(
        f"session {wall_s:.0f}s, {n_spans} timed stages, {len(turn_rows)} turns"
    )
    if not spans and not rollups:
        lines.append("")
        lines.append("No timing events in this trace: it predates the profiler, or")
        lines.append("nothing ran. Start a session and ask the dog something.")
        return "\n".join(lines)

    header = (
        f"  {'stage':<24}{'n':>5}{'total':>9}{'mean':>9}{'p50':>9}{'p95':>9}{'max':>9}"
    )

    def table(rows):
        for r in rows:
            mark = " ~" if r["background"] else ""
            fail = f"  ({r['failed']} failed)" if r["failed"] else ""
            lines.append(
                f"  {r['stage']:<24}{r['n']:>5}{_ms(r['total_ms']):>9}"
                f"{_ms(r['mean_ms']):>9}{_ms(r['p50_ms']):>9}{_ms(r['p95_ms']):>9}"
                f"{_ms(r['max_ms']):>9}{mark}{fail}"
            )

    leaves = [r for r in spans if r["role"] != "umbrella"]
    shown = leaves[:top] if top else leaves
    lines.append("")
    lines.append("WHERE THE TIME GOES  (one row per stage, by total time)")
    lines.append(header)
    table(shown)
    if any(r["background"] for r in shown):
        lines.append("  ~ background: wall-clock the human does not wait through")

    umbrellas = [r for r in spans if r["role"] == "umbrella"]
    if umbrellas:
        lines.append("")
        lines.append("WHOLE UNITS OF WORK  (these CONTAIN the stages above)")
        lines.append(header)
        table(umbrellas)

    if rollups:
        lines.append("")
        lines.append("HOT LOOP  (50 Hz control loop, rolled up per window)")
        lines.append(f"  {'stage':<24}{'n':>8}{'total':>9}{'mean':>9}{'max':>9}{'duty':>8}")
        for r in rollups:
            duty = r["total_ms"] / (r["window_s"] * 1000.0) if r["window_s"] else 0.0
            lines.append(
                f"  {r['stage']:<24}{r['n']:>8}{_ms(r['total_ms']):>9}"
                f"{_ms(r['mean_ms']):>9}{_ms(r['max_ms']):>9}{duty * 100:>7.1f}%"
            )

    if path:
        total = sum(r["median_ms"] for r in path)
        lines.append("")
        lines.append(f"CRITICAL PATH  (median over {path[0]['n']} turns, heard -> first sound)")
        for r in path:
            frac = r["median_ms"] / total if total else 0.0
            lines.append(
                f"  {r['stage']:<20}{_ms(r['median_ms']):>9}  {_bar(frac):<22}{frac * 100:>5.0f}%"
            )
        lines.append(f"  {'TOTAL':<20}{_ms(total):>9}")
        measured = [r["measured_ms"] for r in turn_rows if r["measured_ms"]]
        if measured:
            lines.append(
                f"  measured end to end (voice.reply): median "
                f"{_ms(statistics.median(measured))}, worst {_ms(max(measured))}"
            )

    vo = voice_out(events)
    if vo["n"]:
        lines.append("")
        lines.append("TEXT ON SCREEN -> SOUND  (first chunk only; the rest plays behind it)")
        headline = vo["text_to_sound_ms"] or vo["first_audio_ms"]
        for row in vo["stages"]:
            frac = row["median_ms"] / headline if headline else 0.0
            lines.append(
                f"  {row['stage']:<20}{_ms(row['median_ms']):>9}  "
                f"{_bar(frac):<22}{frac * 100:>5.0f}%"
            )
        if vo["text_to_sound_ms"] is not None:
            lines.append(f"  {'READ -> HEARD':<20}{_ms(vo['text_to_sound_ms']):>9}"
                         f"   median of {vo['n']}, worst {_ms(vo['worst_ms'])}")
        shape = []
        if vo["first_chunk_chars"]:
            shape.append(f"first sentence {vo['first_chunk_chars']:.0f} chars")
        if vo["chunks"]:
            shape.append(f"{vo['chunks']:.0f} chunks per reply")
        if vo["rtf"]:
            shape.append(f"RTF {vo['rtf']:.2f} (wall seconds per audio second)")
        if shape:
            lines.append("  " + ", ".join(shape))

    spoken = [r for r in turn_rows if r["stages"]]
    if spoken:
        lines.append("")
        lines.append("PER TURN")
        lines.append(
            f"  {'turn':<10}{'endpoint':>9}{'asr':>9}{'chat':>9}{'llm':>9}"
            f"{'tools':>9}{'tts':>9}{'total':>9}  calls"
        )
        for r in spoken:
            s = r["stages"]
            lines.append(
                f"  {r['turn']:<10}{_ms(s.get('mic.endpoint', 0)):>9}"
                f"{_ms(s.get('asr.transcribe', 0)):>9}{_ms(s.get('chat.turn', 0)):>9}"
                f"{_ms(r['llm_ms']):>9}{_ms(r['tool_ms']):>9}"
                f"{_ms(s.get('tts.first_audio', 0)):>9}"
                f"{_ms(r['measured_ms'] or r['path_ms']):>9}"
                f"  {r['llm_calls']}x llm"
                + (f", {', '.join(r['tools'])}" if r["tools"] else "")
            )
    return "\n".join(lines)


def report(events: list[dict]) -> dict:
    """The same numbers as `render`, as data (for --json, tests, CI gates)."""
    turn_rows = turns(events)
    return {
        "stages": span_stats(events),
        "loop": rollup_stats(events),
        "critical_path": critical_path(turn_rows),
        "voice_out": voice_out(events),
        "turns": turn_rows,
        "wall_s": max((float(e.get("t") or 0.0) for e in events), default=0.0),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("traces", nargs="*", type=Path,
                   help="trace JSONL files (default: newest in runs/agent_traces)")
    p.add_argument("--top", type=int, default=0, help="show only the N slowest stages")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    args = p.parse_args(argv)

    sources = list(args.traces)
    if not sources:
        newest = latest_trace()
        if newest is None:
            print("no traces in runs/agent_traces (run a session first)", file=sys.stderr)
            return 2
        sources = [newest]
    missing = [s for s in sources if not s.exists()]
    if missing:
        print(f"no such trace: {', '.join(str(m) for m in missing)}", file=sys.stderr)
        return 2
    events: list[dict] = []
    for src in sources:
        events.extend(load_events(src))
    if args.json:
        print(json.dumps(report(events), indent=2, default=str))
    else:
        print(render(events, sources, top=args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
