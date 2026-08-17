# Latency: measuring the robot before accelerating it

Every session already writes a trace.  With `wojtek_rl.perf` that trace also
carries a timing for every stage, and `./training/run.sh perf` ranks them.
Nothing has to be switched on beforehand — the profiling mode you forget to
enable is the one you needed when the demo dragged on stage.

```bash
# after (or during) a session
./training/run.sh perf                     # newest trace in runs/agent_traces
./training/run.sh perf runs/agent_traces/flat_20260814_1830_9021.jsonl
./training/run.sh perf --json              # for diffs and CI gates
```

## What is timed

One vocabulary across both stacks, so numbers are comparable.

| stage | what it measures | where |
|---|---|---|
| `mic.endpoint` | trailing silence waited out before the utterance closes | `agent/voice.py` (measured), ROS probe (from `silence_end_s`) |
| `asr.transcribe` | speech → text, whisper local or remote | `agent/voice.py`, ROS `/wojtek/asr/final` |
| `brain.route` | transcript → routed intent (ROS only) | ROS `/wojtek/intent` |
| `chat.turn` | the whole agent turn — **contains** the rows below | `agent/chat.py` |
| `llm.chat` | one model call in the tool loop | `agent/chat.py` |
| `llm.translate` | the extra EN→PL call in `lang_mode=translate` | `agent/chat.py` |
| `llm.first_sentence` / `llm.reply` | intent → first / last spoken sentence (ROS) | ROS `/wojtek/say` |
| `tool.<name>` | one tool call (`look`, `map`, `navigate`, …) | `agent/chat.py` |
| `tts.synth` | synthesising one sentence chunk | `agent/tts.py` |
| `tts.first_audio` | reply text → first PCM frame on the wire | `agent/tts.py`, ROS `/wojtek/tts/audio` |
| `voice.reply` | **the felt wait**: utterance closed → first sound | `agent/tts.py`, ROS probe |
| `nav.frame` / `nav.decide` / `nav.execute` | navigator: render, VLM inference, walking | `vlm_nav.py` |
| `nav.decide_prefetch` / `nav.await_prefetch` | overlapped inference, and what the loop still waited for it | `vlm_nav.py` |
| `search.score_view` / `search.scan_carousel` / `search.leg` | observer call, the 360° scan, one frontier drive | `agent/search.py` |
| `sim.step` / `sim.render_pair` / `sim.map_jpeg` / `ws.send` / `loop.tick` | the 50 Hz control loop, rolled up per 5 s window | `room_app.py` |

Three kinds of row, and the difference decides what is worth optimising:

- **work** — someone is waiting for it.
- **background** (`~`) — real seconds nobody waits through: the robot
  physically walking (`nav.execute`), and inference deliberately hidden
  behind that walk (`nav.decide_prefetch`).
- **umbrella** — spans that *contain* other spans (`chat.turn`,
  `voice.reply`, `search.goal`).  Reported in their own table; ranking them
  next to their own parts would double count every second.

## Reading the report

```
WHERE THE TIME GOES  (one row per stage, by total time)
  stage                       n    total     mean      p50      p95      max
  llm.chat                   26   71.80s    2.76s    2.61s    4.90s    5.40s
  nav.execute                38  120.40s    3.17s    3.10s    4.80s    6.10s ~
  ...
CRITICAL PATH  (median over 9 turns, heard -> first sound)
  mic.endpoint            700ms  ###                      13%
  asr.transcribe          620ms  ##                       11%
  chat.turn               3.10s  ############             57%
  tts.first_audio         1.05s  ####                     19%
  TOTAL                   5.47s
```

The totals table and the critical path are different rankings on purpose.
A stage can dominate wall-clock and cost nobody a second of waiting; 0.7 s of
VAD endpointing is invisible in the totals and is paid in **every** turn.
Fix what the critical path shows, in the order it shows it.

`PER TURN` is where an outlier is identified: which turn, how many model
calls it took, which tools ran.  A turn with `3x llm` is a turn where the
model needed two tool round trips before answering — the fix there is the
prompt or the tool set, not a faster GPU.

## Measured: why the answer appears long before it is heard

The demo releases the reply text and the reply audio in the same instant, so
that gap is TTS and nothing else. Measured 2026-08-17, four typical replies:

| engine | read → heard | RTF (wall s per audio s) |
|---|---|---|
| piper `pl_PL-mc_speech-medium`, local CPU | **0.48 s** | 0.20–0.39 |
| Chatterbox multilingual, RTX A6000 (Ada) | **5.7 s** (worst 6.4 s) | **1.33–1.61** |
| Chatterbox multilingual, RTX 5080 (Blackwell) | first piece **1.9–3.0 s** | **0.86–0.90** |

The deployment target is a DGX Spark (GB10, Blackwell), and on that
generation the same engine already runs faster than real time — so
clause-level streaming pays there and cannot underrun. The Ada numbers below
are kept because they are what the demo ran on in August.

Chatterbox's cost curve (A6000, median of 3, via the server's `x-synth-ms`):

| text | GPU time | audio produced | RTF |
|---|---|---|---|
| `Hau.` (4 chars) | 2.77 s | 1.72 s | 1.61 |
| `Widzę kanapę.` (13) | 4.70 s | 3.44 s | 1.36 |
| `Szukam kanapy,` (14) | 2.85 s | 1.76 s | 1.62 |
| 55 chars | 5.84 s | 4.40 s | 1.33 |
| 78 chars | 5.67 s | 4.20 s | 1.35 |

**RTF above 1.0 is the whole story**: the voice generates slower than it
plays, so it can never catch up with itself, and no amount of client-side
pipelining fixes a four-word answer costing three seconds. What chunking
*does* fix is long replies — cutting the first chunk at a clause took the
78-char reply from 6.42 s to 3.32 s to first sound, because the first piece
is short. Short replies were unchanged (5.74 s → 5.98 s, inside the noise):
their floor is the engine.

The option space for making the cloned voice fast enough — safe levers,
lossy levers, and what to try first — is in
[tts-optimization.md](tts-optimization.md).

So, for a stage voice, in order:

1. **Pick an engine with RTF < 1** for anything spoken live; keep the
   expensive cloned voice for pre-rendered lines. Piper is 10× faster here
   and sounds like a different (uncloned) dog — that is the trade.
2. **The line cache** (`tts_server`, 64 entries, `x-cache` header) makes a
   repeated fixed line free the second time: acknowledgements, "Już się
   zatrzymuję!", the found-object phrase. Pre-warm it by POSTing the stage
   script's fixed lines before the show.
3. **Warm up the server** (`tts_server` does this at startup now, ~57 s on an
   A6000). Without it the FIRST thing the robot says also pays model load.
4. `speech_chunks` keeps the first chunk short and `synth_stream` ships
   pieces as they are made — worth it on long replies, neutral on short ones.

## Fixed costs worth knowing before you tune

- `SILENCE_END_S = 0.7` (`agent/voice.py`, ROS `silence_end_s`) is a floor on
  every spoken turn, and it trades directly against cutting people off
  mid-sentence.
- `max_tool_steps` multiplies the model call cost: each tool round trip is a
  full `llm.chat` on a growing context.
- The search's scan carousel is ~8 observer calls before exploration starts;
  a castle search needs a ~140 s window because of it (see the status doc).
- `RENDER_EVERY`/`MAP_EVERY` set how much of each 20 ms control tick goes to
  JPEG encoding rather than physics — watch `loop.tick` p95 against 20 ms.

## The ROS stack

The probe is a passive subscriber; it changes no node and adds no work to
any callback.

```bash
ros2 launch wojtek_agent_bringup voice_stack.launch.py perf:=true perf_out:=/tmp/wojtek_perf.jsonl
```

```bash
./training/run.sh perf /tmp/wojtek_perf.jsonl
```

See `ros/src/wojtek_agent_perf/README.md`.  Keep `vad_silence_s` in sync with
the VAD node, and remember the ROS `mic.endpoint` is `assumed` (taken from
config) while the demo stack measures the silence it actually accumulated.

## Adding a stage

```python
from wojtek_rl import perf

with perf.span("tool.look") as sp:
    result = await do_the_thing()
    sp["images"] = len(result.images)
```

For anything firing at control rate use a `perf.Meter` instead — one rollup
per window, not one event per tick.  For a stage whose start and end live in
different callbacks use `perf.Timer`, which reports once and stays silent if
the work was cancelled (a barge-in must not report an invented
time-to-first-sound).  New umbrella or background stages need a line in
`perf_report.UMBRELLA_STAGES` / `BACKGROUND_STAGES`, or they will be ranked
as if a human waited for them.
