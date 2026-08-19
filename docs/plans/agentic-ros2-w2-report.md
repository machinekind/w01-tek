# Agentic stack — implementation report, W1–W2 (2026-08-16)

Companion to the [target architecture](agentic-ros2.md), whose decisions are
settled. This report records what has been **built and verified**, the
measurements and failure modes behind those claims, and the work that remains.
Every behavioral claim below was checked on a rented cloud GPU box with
recorded, gate-checked footage.

## 1. Verified: interactive demo

`SCENE=castle|flat|room|apartment ./training/run.sh room`, with FutureNav,
Qwen3-VL, whisper and Chatterbox served from one GPU host:

- **Polish voice loop, end to end**: mic → whisper large-v3 → Qwen3-VL agent →
  Chatterbox over HTTP (`tts_server.py`, `TTS_ENGINE=remote TTS_URL=...`).
- **Visibility-gated navigation**: `navigate` checks the current camera view
  first; a target that is not visible turns the goal into a search and says so
  honestly ("Szukam okna, nie widzę go teraz"). Question-shaped input bounces
  out of `navigate`/`search` with a corrective turn. Routes stay verbatim.
- **Goal FSM**: switching mid-goal is spoken ("Ok, przerywam trasę…"), `status`
  reports pose plus goal history, and `stop`/`navigate` both have deterministic
  guards for the turns where the model answers in words instead of a tool call.
- **VLFM-lite object search**: scan carousel → frontier exploration → 2-of-3
  verification; a found object is announced ("Znalazłem: kanapa! Hau hau!").
- **Fallback-reply quarantine**: one bad LLM turn can no longer collapse the
  JSON contract for the rest of a session (see §4).

## 2. Verified: ROS 2 stack (W1 + W2 of the architecture)

`ros/src/wojtek_agent_msgs`, `wojtek_voice` (audio bridge speaking the
browser protocol, silero VAD, whisper ASR, dual-engine TTS), `wojtek_brain`
(rule router plus a trainable encoder option, Bielik node with
sentence-streamed replies, canned navigation acks, question-aware EN→PL
translation, barge-in) and `wojtek_agent_bringup` (talk-only launch).

Verified end to end on a GPU host: spoken Polish in, Bielik answer in the
selected voice out. `tts_engine:=silent asr_model:=tiny` is the CPU smoke
configuration. Node logic modules are rclpy-free and covered by model-free
unit tests.

## 3. Verified: recording and validation harness

`wojtek_rl.agent.scenario` drives scripted voice sessions (wav questions plus
mid-level `cmd` steps), films and mixes both voices on **one** websocket
connection, and enforces success gates (`--require-move`, `--require-speech`,
frozen-frame detector). A take that fails a gate exits non-zero.

Deliverable clips additionally get frames extracted and inspected by eye:
during this work, metrics alone reported success on two takes that were in
fact frozen video.

## 4. Findings that changed the code

These are the non-obvious failures found by running the stack live; each one
is now guarded in code.

- **`room_app` is single-viewer.** A second websocket connection supersedes the
  first one's stream loop. Filming with a client separate from the driver made
  the recorder repeat a single frame for the full clip while the moving sim
  streamed unfilmed. Fix: `scenario.py` drives, films and validates on one
  socket.
- **Narrated tool calls collapse the JSON contract.** `parse_agent_reply`
  accepted plain text as `say`, so one narrated line in the rolling history
  spread until the model stopped emitting tools at all — observed as
  `navigate` never firing across a whole session. Fix: fallback replies are
  quarantined from history, and a hint regex forces a verbatim `navigate`.
- **A heard stop must halt the goal even when the model replies in prose.**
  A narrated stop left the goal running; the stop guard is now deterministic.
- **Whisper hallucinates on silence** — a phantom token reached the agent as a
  command and preempted a running goal. The ASR service now drops segments
  with `no_speech_prob > 0.6` or `avg_logprob < -1.0`.

## 5. Scene and staging findings

**Scenes** (all CC, stage- and commercial-safe with attribution): `castle`
(Skokloster hall) and `flat` (ReplicaCAD apartment), built via
`room-assets`/`build-room`/`wojtek_eval.gridmap` with hand-annotated
`objects.json` carrying Polish aliases. Spawn auto-relocates off occupied
cells; `WOJTEK_SPAWN="x,y"` overrides it (castle takes use `2.5,-3.0`).

- Auto-relocation alone can land ~21 cm off open floor — visually still under
  furniture. Pin the spawn for filmed takes.
- A search target visible from spawn is found in seconds, which produces a
  zero-distance take. Pick unseen targets, or script `{"cmd": "turn_left
  180"}` first.
- Castle searches need a ~140 s roam window; the scan carousel alone takes
  ~60 s. Anything under 60 s is a guaranteed zero-distance take.
- FutureNav walks well in the flat and poorly in the large castle hall.

**Prompts** are plain text, split by model and language:
`training/wojtek_rl/agent/prompts/qwen/` (English) and
`ros/src/wojtek_brain/wojtek_brain/prompts/bielik/` (Polish). Edit, restart,
done.

## 6. Deployment findings

Provisioning scripts for concrete rented machines live in the private
operations repository; this repository keeps only cluster-agnostic payloads
(root `CLAUDE.md`). The transferable configuration knowledge:

- Image must be `nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04` — ROS 2 Jazzy
  exists only for Noble, and faster-whisper needs the cuDNN variant.
- The node venv uses `--system-site-packages` so it sees apt `rclpy`, and
  colcon must build with that venv active so installed node scripts get venv
  shebangs. Otherwise put venv site-packages on `PYTHONPATH`.
- FutureNav venv: uv, Python 3.11, requirements plus a cu130 torchvision.
  vLLM needs `python3.12-dev` for its Triton JIT path.
- Chatterbox must be the official multilingual `chatterbox-tts` package — the
  streaming fork is **English-only**. Keep the perth-watermarker fallback in
  `tts_server.py` for hosts where the optional dependency is broken.
- Two LLMs plus FutureNav fit a 48 GB card only with tight memory
  utilisation (Bielik vLLM ~0.18, Qwen ~0.45; peak measured 47.3/49 GB).
- `pkill -f <pattern>` that also matches the launcher's own command line kills
  the launcher; drive remote work through `bash -s` heredocs instead.

## 7. Licensing status

MP3D/HM3D rejected (signed non-commercial ToS). The F5 Polish checkpoint is
CC-BY-NC and its cloned voices are development-only. The FutureNav weight
license remains an open question for a commercial fork. The full audit table
is in the architecture document.

## 8. Remaining work, in order

1. **W3** — wrap the Qwen agent, the room sim and the demo UI as ROS nodes
   (`vlm_agent`, sim bridge, ui bridge) so the ROS stack walks, not just talks.
2. **Router fine-tune** — `ros/src/wojtek_brain/tools/gen_router_dataset.py` →
   `train_router.py` (HerBERT default; compare mmBERT and ModernBERT on the
   same Polish set), then set the router node's `model_path`.
3. **Whisper large-v2 vs large-v3 A/B** on real microphone audio; v3
   hallucinates more on silence, and the guards apply either way.
4. **Stage voice decision and clone** — the development voice must be replaced
   by a neutral, rights-clear voice before any public demo.
5. **W4 hardening** — failure drills, latency pass, stage runbook.
