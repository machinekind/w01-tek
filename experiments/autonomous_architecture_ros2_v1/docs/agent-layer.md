# Wojtek chat agent (chat + search + goal state machine)

The layer above the navigation stack: a small instruct VLM
(`Qwen/Qwen3-VL-4B-Instruct-FP8` by default) is the robot's front door. You
talk to it in the room demo's chat panel (or `POST /api/chat`); it answers in
character as a happy dog, and acts through a fixed tool set. Code lives in
`wojtek_agent/`; the design notes below explain the pieces that are not
obvious from the module docstrings.

## Architecture

```
user text ──> WojtekAgent (chat.py)          one JSON reply per turn
                 │  tools (tools.py)
                 ├─ look      ego frame (also fuses depth into the map)
                 ├─ map       OnlineMap image + text summary (spatial.py)
                 ├─ route     PoseHistory over the last N seconds
                 ├─ status    pose + goal state + executor
                 ├─ navigate ─┐
                 ├─ search  ──┤
                 └─ stop      │
                              ▼
              GoalManager (goals.py)          one goal at a time
                 ├─ navigate: VlmNavigator (FutureNav or any backend)
                 └─ search:   SearchController (search.py)
                                 │ mid-level commands / executor.goto
                                 ▼
                    ScanExecutor -> ScanPlanner -> 50 Hz policy
```

The VLM never emits velocities and never picks map cells. It picks *tools*;
the FSMs underneath are classical code (CogNav's lesson: use the model inside
states — scoring views, verifying detections — not as the transition
authority).

## Reply contract: "thought" first, on purpose

Every agent reply is one JSON object whose FIRST key is free text
(`thought`, and the search observer's `description`). A 4B VLM forced to
open with a decision token stops looking at the image — measured on this
robot: an action-first contract collapsed Qwen3-VL-4B to `forward` on every
frame. `test_agent_parsing.py` guards the ordering. Parsing is a
string-aware balanced-brace scanner with repair turns for the classic small
model mangles (code fences, `({...})}`, prose instead of JSON).

## Context management

Blunt by design: only the last 3 question/answer pairs survive between
turns, text only. Images exist only inside the current turn (tool results),
so the context cannot accumulate stale frames the model would re-describe.
`chat_reset` (the UI's "forget" button) drops everything.

## Search: VLFM-lite with verification

`SearchController` is the VLFM recipe (Yokoyama et al., ICRA 2024 — deployed
on Spot) sized down for this stack:

1. **Initial 360° scan** — 8 stops × 45°; each stop grabs the ego frame
   (which fuses depth into the occupancy map) and asks the VLM for
   `{description, target_visible, score 0-10, bbox_2d}`.
2. **Value map** — scores are splatted over the camera FOV cone (cos² axis
   weighting, confidence-weighted fusion — VLFM's update rule) into a grid
   aligned with the OnlineMap.
3. **Frontier selection** — connected frontier clusters from the OnlineMap,
   scored `value − 0.5 × distance`; when the value map is uninformative it
   falls back to nearest-frontier FBE (CoW showed plain FBE is a strong
   baseline). The search *commits* to a frontier until the leg ends —
   per-step re-selection oscillates.
4. **Drive + watch** — `executor.goto` (planned, via SCAN) with detection at
   ~0.7 Hz en route; arrival triggers a ±60° sweep.
5. **Approach + verify** — a detection is a *candidate*, never a success:
   center the bbox, close to ~bbox-fills-30%-of-frame, then require 2-of-3
   stationary re-detections (TriHelper's check; VLM false positives are the
   dominant real-world search failure). A failed verify blacklists the spot
   (r = 0.7 m) so the search cannot orbit a phantom.

Everything degrades gracefully: an unparseable observer reply scores 0, a
blocked leg just ends, a crash lands in the `error` state instead of a
zombie "searching".

## Talking to Wojtek in Polish

The mic stays open, the server decides where an utterance ends, and the reply
comes back as streamed audio. Click the mic once to start a conversation —
there is no push-to-talk.

```
browser mic ──100 ms PCM16 @24 kHz──▶ VoiceSegmenter ──▶ faster-whisper (pl)
                                                              │
                                                         Polish text
                                                              ▼
                                                        WojtekAgent (tools)
                                                              │
                                                         Polish reply
                                                              ▼
   speaker ◀──100 ms PCM16 frames── Speaker ◀── Piper pl_PL
```

**Why speech is not done by an "omni" model.** The obvious design — one
multimodal model that hears and speaks — does not work for Polish today:

- **Qwen3.5-Omni** does speak Polish, but it is **API-only**; there are no
  open weights to self-host.
- **Qwen3-Omni-30B-A3B** (Apache-2.0, the open one) supports **19 speech-input
  languages and 10 Talker output languages — Polish is in neither list**. Its
  *text* side covers 119 languages, so Polish text in and out is fine.
- Same gap across the family: Qwen3-TTS, CosyVoice3 and Voxtral-TTS are all
  ~10 languages, none with Polish.

So hearing is Whisper (`large-v3` is the best-measured open Polish
recogniser — FLEURS 4.74) and speaking is a separate Polish TTS. The brain
only ever sees text, which also means it can be any capable model.

Measured on this laptop, no GPU: Piper synthesises at **RTF 0.18** (5.4 s of
Polish speech in 1.0 s); a full spoken exchange took **2.2 s** from end of
speech to recognised text and **2.7 s** to the first audio frame back;
**barge-in cuts playback 90 ms** after the human starts talking.

### English inside, Polish only out loud

The system runs in English — reasoning, tool arguments, traces, logs, the UI
— because that is where a small model is strongest and where anyone debugging
this wants to read. The single exception is the sentence the dog speaks
aloud. A typed answer stays English; the same question asked by voice comes
back in Polish, with the private `thought` still in English:

```
[typed ] What do you see?  → "Woof! I see the bed with the red blanket and a chair…"
[spoken] Co widzisz?       → "Woof! Widzę łóżko z czerwonym kocem i krzesło…"
                             thought: "I see the bed with the red blanket and a chair…"
```

Model and option survey behind these choices: [the Polish-voice research report](polish-voice-research-report.md).

### You do not give up the brain to speak Polish

The brain only ever sees text, and text is where these models are
multilingual — so nothing is traded away by speaking Polish. What *is* real
is that a small model reasons and obeys a JSON contract more reliably in
English, which is why everything except the spoken line stays English.

How the spoken line gets there is `AGENT_LANG_MODE`:

| mode | how | cost |
|---|---|---|
| **`direct`** (default) | the model writes `say` in Polish itself | none |
| `translate` | the model writes English; one extra short call renders the line | +1 call (~0.5 s) |

Measured on Qwen3-VL-4B-FP8, three Polish questions each:

```
direct     0.6-1.7 s   "Widzę drewnianą podłogę, łóżko z czerwonym kocem i krzesło."
translate  1.6-2.1 s   "Zauważam drewnianą podłogę, łóżko z czerwonym kocem i krzesło."
```

**`translate` was both the slowest and the worst** — it produced `"Możę"`
for *mogę* and stilted word choices (`zauważam` where a dog would say
`widzę`). Translating a finished line through the same small model adds a
place to go wrong without adding knowledge. Keep it for the case where a
model turns out genuinely strong in English and genuinely poor in the target
language — and measure before believing that.

The translation hop uses the same model rather than a dedicated MT system on
purpose: it knows who Wojtek is, so the persona survives. A general
translator renders "Woof! I'm on it!" correctly and lifelessly. Both versions
are kept in the trace (`chat.say` carries `source`), and a failed translation
sends the English line rather than nothing.

### Pieces

- `agent/voice.py` — `VoiceSegmenter` turns the frame stream into utterances
  (energy + silence timeout, with a 0.3 s pre-roll so the first syllable is
  never clipped; inject `vad=` to swap in silero). `VoiceListener` runs
  recognition in a worker thread so the 50 Hz control loop keeps running.
- `agent/tts.py` — `PiperTts` (MIT voices, CPU, `pl_PL-mc_speech-medium` by
  default) behind a two-method `TtsEngine` protocol; `Speaker` streams frames
  and cancels cleanly for barge-in; `speakable()` strips emoji and markdown,
  which a synthesiser would otherwise read out loud.
- **Chunked synthesis**: a reply is split into sentences and each is
  synthesised while the previous one plays, so time-to-first-audio is set by
  the first sentence rather than the whole answer. Measured on a 10.2 s
  Polish reply: **0.68 s → 0.49 s**, 27 % earlier. Splitting is deliberately
  conservative — `split_sentences()` merges anything under 25 characters and
  never breaks after a Polish abbreviation (`np.`, `itd.`, `ok.`, `godz.`…)
  or a decimal, because a mid-sentence seam is audible and costs more than
  the latency it saves. Short replies stay whole.
- Prompting: the persona insists on answering in the user's language, and
  spoken turns get an extra style block (no lists, no emoji, numbers written
  as words) — `ask(text, voice=True)` picks it.

### Where recognition runs

Whisper `large-v3` is the accuracy pick, and on a laptop CPU it is also the
latency problem — measured on the same 2.8 s Polish utterance:

| host | compute | time | vs realtime |
|---|---|---|---|
| laptop CPU | int8 | **5.58 s** | 1.98× |
| RTX 5880 Ada | fp16 | **0.36 s** | **0.13×** |
| RTX 5880 Ada | int8 | 0.41 s | 0.15× |

~15× faster on the GPU, and `large-v3` is the variant that gets *krzesła*
right where `small` and `large-v3-turbo` both return *przesła*.

So recognition can move to the box the models are on:
`wojtek_agent/asr_server.py` serves `POST /transcribe` (WAV in, text out),
and `ASR_URL` switches the demo to `RemoteTranscriber` — the same one-method
contract `VoiceListener` already used, so nothing else changes. Unset, it
falls back to in-process CPU whisper.

On the robot this service is unnecessary: the Jetson is both the mic host and
the GPU. Expect large-v3 int8 in the 1–1.5 s range on an AGX Orin (~⅕ of the
card above), and `whisper_trt` claims roughly another 3× on Orin.

### Knobs

`ASR_MODEL` (default `large-v3`), `ASR_URL` (GPU ASR service; unset = local CPU),
`ASR_LANGUAGE` (`pl`), `ASR_DEVICE`, `ASR_COMPUTE` (`int8`, ~2.5 GB),
`TTS_ENGINE` (`piper` | `none`), `TTS_VOICE`.

```bash
uv sync --extra voice
python -m piper.download_voices pl_PL-mc_speech-medium
```

A missing Piper binary or voice degrades to `SilentTts` with a warning naming
the exact download command — the demo still runs, it just does not speak.

Upgrades worth benchmarking on your own Polish audio, since **no public
benchmark ranks TTS on Polish**: **MOSS-TTS-Realtime** (Apache-2.0, Polish
listed, 180 ms TTFB, native streaming) and **Supertonic 3** (99M params, ONNX,
CPU-only — zero VRAM next to the brain). Note Piper's runtime moved to
`OHF-Voice/piper1-gpl` (GPL-3.0 runtime, MIT/CC0 voices); XTTS-v2 and
MMS-TTS-pol are non-commercial licences.

## Serving: the split stack

The intended production shape is two models, each doing what it is good at:

| model | port | serves |
|---|---|---|
| FutureNav-4B | 8100 | `navigate` — the VLN-CE instruction follower |
| Qwen3-VL-4B-FP8 (vLLM) | 8090 | chat, tools, and the search observer |

```bash
./training/run.sh room --vlm-backend futurenav \
    --vlm-url http://127.0.0.1:8100 --agent-url http://127.0.0.1:8090
```

Both fit on one rented GPU box; the provisioning scripts for concrete
machines live in the private operations repository (root `CLAUDE.md`).
Sizing in short: FutureNav wants ~10 GB (its own deploy notes ask for 24 GB to
be comfortable) and vLLM ~8 GB at `--gpu-memory-utilization 0.45`, so one
≥40 GB card is the easy path and two 24 GB cards (`FUTURENAV_GPU=0
VLLM_GPU=1`) the cheap one. The traps that cost time, whatever the host: pin
FutureNav's torch to cu124 (an unpinned wheel silently falls back to CPU on
older drivers), put `ninja` on `PATH`, and set
`VLLM_USE_FLASHINFER_SAMPLER=0`.

## Serving one model for everything

One vLLM server can serve both the chat agent and the search observer:

```bash
vllm serve Qwen/Qwen3-VL-4B-Instruct-FP8 --port 8000
```

Any Ampere-or-newer card runs the FP8 checkpoint natively; a 16 GB card is
enough for the 4B model alone (~8 GB resident). Provisioning scripts for
concrete GPU hosts — shared or rented — live in the private operations
repository; this repository stays free of private-machine identity.

Two vLLM traps worth knowing regardless of the host:

- **`ninja` is required.** vLLM's compile path shells out to it, but it is not
  a vLLM dependency, and it must be on `PATH` — not merely installed in the
  venv. Without it the server dies with `FileNotFoundError: 'ninja'` seconds
  after loading weights.
- **FlashInfer sampler off** (`VLLM_USE_FLASHINFER_SAMPLER=0`). FlashInfer
  JIT-builds sampling kernels against the system nvcc, whose CUB no longer
  has `BlockAdjacentDifference::FlagHeads`; the build fails and engine init
  dies. vLLM's native sampler is fine at our request rate.

Measured with `Qwen3-VL-4B-Instruct-FP8` on a 16 GB RTX 4090 Mobile (Ada,
~8 GB resident): 0.9 s for a text-only chat turn, 1.4-1.7 s when a camera
frame is attached, so a look-then-answer exchange lands around 2 s.

Point the demo at it (both knobs also exist as `--agent-url/--agent-model`):

```bash
AGENT_URL=http://127.0.0.1:8000 ./training/run.sh room --vlm-backend futurenav --vlm-url http://127.0.0.1:8100
```

`AGENT_URL` defaults to `VLM_URL`, then `http://127.0.0.1:8000`. The chat
endpoint is independent of the navigation backend: FutureNav keeps serving
`navigate` goals on its own server.

## Routes are instructions, not point goals

The navigator underneath is an instruction follower (VLN-CE style, and
FutureNav is trained on exactly this), so `navigate` forwards what the user
actually said — "walk past the chair, then turn left at the doorway and stop
next to the bed" — not a distilled destination. Left to itself the chat model
collapses routes to a noun phrase ("the table"), which throws away both the
manner ("around", "past") and every step after the first.

Two guards, because prompt instructions alone are not reliable on a 4B:

- the tool's signature and the chat rules demand the instruction VERBATIM;
- `tools.keep_full_instruction()` compares the model's argument against the
  raw user text (shared through `turn_context`) and substitutes the user's
  own words when the argument is much shorter. A genuinely short command
  ("go to the bed") passes through untouched.

`INSTRUCTION_PROMPT` extends the base navigation prompt with ordered-step
execution and "answer `done` only when the LAST step is complete".

### No step budget interactively

A step cap is a benchmark device: it keeps episodes comparable and bounds SPL.
Interactively it is just a guillotine — one live goal ended on `max_steps` at
step 20 with the route half-walked — and a counter cannot tell "still working"
from "thrashing". So `max_steps=None` on the `openai` backend (`VlmNavigator`
then iterates with `itertools.count`), and runs end on evidence:

- the model answers `done` or `stop`;
- the user cancels, or sends any manual command;
- **anti-spin**: `NAV_MAX_ROTATION = 14` consecutive turns in place, generous
  enough for a full look-around plus a correction;
- **wedged**: `MAX_CONSECUTIVE_BLOCKED = 6` commands in a row that the
  executor could not carry out, which ends the goal with reason `stuck`.

`situation_text` also stops telling the model "step 40 of 60" when there is no
budget — a visible countdown pressures it into declaring `done` early. Pass
`--vlm-max-steps` / `VLM_MAX_STEPS` when you *want* benchmark-shaped episodes;
`wojtek_eval`'s runner and `nav_episode` always set it explicitly, so
benchmarks are unaffected.

## Never show the observer the HUD

`RoomSim.ego_jpeg()` composites the self-built minimap into the bottom-right
corner. That is deliberate for the navigator, whose prompt explains the
minimap legend — but any model that is *not* told about the inset reads it as
part of the room. Measured in a live session: the search observer scored the
inset itself as the target ("a map showing a toy's location", bounding box
landing exactly on the paste rectangle), approached it, lost it at close
range, blacklisted the spot, picked a new frontier and did it again. The
robot wandered for four minutes and found nothing.

So the search observer and the chat `look` tool both take `ego_jpeg(hud=False)`.
Clean separation: `look` is the camera, `map` is the map. If you add a
consumer whose prompt does not describe the minimap, it wants `hud=False` too.

## Session trace

Live debug panels answer "what is it doing now". The trace
(`wojtek_agent/trace.py`) answers "what did it do ten minutes ago", and
survives a browser reload:

- one JSON object per line in `runs/agent_traces/<scene>_<stamp>_<pid>.jsonl`,
  written line-buffered so a killed process still leaves a readable file
  (override the location with `AGENT_TRACE`);
- `GET /api/trace?limit=200&kind=search` serves the same events from a
  bounded memory ring; `kind` is a prefix filter (`chat`, `search`, `nav`,
  `goal`). The UI's debug toggle reveals a **trace ↗** link to it.

Event kinds: `chat.ask` / `chat.llm` (raw model output, latency, tokens,
context size) / `chat.tool` / `chat.say` / `chat.reset`, `search.<state>`
(every FSM event tagged with the state and pose it happened in), `nav.state`
(navigator transitions with goal, step, last decision, blocked count),
`goal.set` / `goal.cancel`, `session.start`.

The trace pays for itself immediately: the first run after adding it showed
`cmd turn_left 45 -> timed out` with the pose unchanged, which is the demo's
documented "the sim only steps while a browser is attached" behaviour biting
a search started over `/api/chat` with no viewer open.

## Debugging

The demo UI ships a full trace, toggled by the **debug** button in the chat
header (hidden by default):

- every dog reply carries a per-turn trace — one block per LLM call with the
  raw model output, its classification (`say` / `tool` / `parse_error` /
  `unknown_tool` / `repeat_nudge`), latency + token usage, the tool call and
  its (truncated) result;
- a pinned **goal state machine** panel shows the live GoalManager status:
  FSM state, note, observation/blacklist/attempted counters, and the search
  event log (every submitted command and its outcome, every observation with
  pose/score/bbox/description, frontier selection with candidate counts,
  verify votes). For `navigate` goals it shows the VlmNavigator step/history
  instead.

The same data is available headlessly: `POST /api/chat` returns the `debug`
object verbatim, and `agent_status` websocket messages carry the event tail
(`SearchController.status()["events"]`, last 20 of a 200-event ring).

## Testing

All agent logic is model-free and unit-tested (`tests/unit/test_agent_*.py`):
the whole search FSM runs under `asyncio.run()` against a teleporting fake
sim and a scripted observer, same pattern as `test_vlm_nav.py`.
