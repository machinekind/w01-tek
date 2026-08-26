# Agentic stack — state of play (2026-08-16)

Read this first, then the [architecture](agentic-ros2.md) (decisions there
are settled).  This is what EXISTS and WORKS, verified live on rented GPUs
with recorded, gate-checked footage — not a plan.

## What works today

**Interactive demo (the #131 stack, upgraded)** — `SCENE=castle|flat|room|apartment`
`./training/run.sh room` with FutureNav + Qwen3-VL + whisper + Chatterbox on
one vast box:

- Polish voice loop end to end: mic → whisper large-v3 → Qwen3-VL agent →
  Chatterbox with a cloned voice over HTTP (`tts_server.py`,
  `TTS_ENGINE=remote TTS_URL=...`).
- **Visibility-gated navigation**: `navigate` checks the current camera view
  first; target not visible → the goal becomes a search, announced honestly
  ("Szukam okna, nie widzę go teraz").  Questions bounce out of
  navigate/search with a corrective turn.  Routes stay verbatim.
- Goal FSM: switching mid-goal is spoken ("Ok, przerywam trasę…"), `status`
  reports pose + goal history, stop and navigate both have deterministic
  guards for turns where the model answers with words instead of a tool.
- VLFM-lite object search: scan carousel → frontier exploration → 2-of-3
  verify; found objects announced ("Znalazłem: kanapa! Hau hau!").
- Fallback-reply quarantine keeps one bad LLM turn from collapsing the JSON
  contract for the rest of a session.

**ROS 2 stack (target architecture, W1+W2 done)** — `ros/src/wojtek_agent_msgs`,
`wojtek_voice` (audio bridge speaking the #131 browser protocol, silero VAD,
whisper ASR, dual-engine TTS), `wojtek_brain` (rule router + trainable
encoder option, Bielik node with sentence-streamed replies, canned nav acks,
question-aware EN→PL translation, barge-in), `wojtek_agent_bringup`
(talk-only launch).  Verified E2E on a vast box: spoken Polish in, Bielik
answer in the cloned voice out.  Deploy: `scripts/vast_voice_stack.sh`.

**Recording/validation harness** — `wojtek_rl.agent.scenario`: drives
scripted voice sessions (wav questions + midlevel `cmd` steps), films and
mixes BOTH voices on ONE websocket connection, and enforces success gates
(`--require-move`, `--require-speech`, frozen-frame detector).  A take that
fails a gate exits non-zero.  Deliverable clips additionally get frames
extracted and looked at — metrics alone have lied before.

**Scenes** (all CC, stage- and commercial-safe with attribution):
`castle` (Skokloster hall) and `flat` (ReplicaCAD apartment) built via
`room-assets`/`build-room`/`wojtek_eval.gridmap`, with hand-annotated
`objects.json` (Polish aliases).  Spawn auto-relocates off occupied cells;
`WOJTEK_SPAWN="x,y"` overrides (castle demos use `2.5,-3.0`).

**Prompts** are plain text, split by model and language:
`training/wojtek_rl/agent/prompts/qwen/` (English) and
`ros/src/wojtek_brain/wojtek_brain/prompts/bielik/` (Polish).  Edit,
restart, done.

## Operating knowledge that saves a session

- **room_app is single-viewer**: a second websocket supersedes the first.
  Never film with a separate client — scenario.py does drive+film together.
- Scenario stagecraft: a search target visible from spawn gets found in
  seconds (a 0 m take); pick unseen targets or script `{"cmd": "turn_left
  180"}` first.  Castle searches need a ~140 s roam window (the scan
  carousel alone takes ~60 s).  FutureNav walks well in the flat, poorly in
  the huge hall.
- vast recipe (proven 4×, ~$0.45/session): host **134600**, image
  `nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04`, arm
  `scripts/vast_destroy_at.sh` BEFORE deploying.  FutureNav venv = uv
  python 3.11 + requirements + cu130 torchvision; vLLM needs
  `python3.12-dev`; Chatterbox needs the official `chatterbox-tts`
  (multilingual) — the streaming fork is English-only — plus the
  perth-watermarker fallback already in `tts_server.py`.
- Licensing: MP3D/HM3D rejected (signed non-commercial ToS); F5 PL
  checkpoint CC-BY-NC and Głuś voice are dev-only; FutureNav weight license
  is still an open question for the commercial fork.  Full audit table in
  the architecture doc.

## Settled 2026-08-25: ROS is the boundary

Everything that will live on the robot runs and communicates via ROS 2 (user
decision). The demo/training stack keeps only what never ships: the MuJoCo
sim, the browser UI, the recording harness. Consequences:

- W3 (wrap the Qwen agent + sim bridge + ui bridge as ROS nodes) is the
  convergence path, not an option among several.
- Until W3 lands, any behaviour added to the demo must mirror the ROS
  topology so it ports 1:1 -- e.g. the demo's Polish mouth is Bielik
  rendering the agent's English via a second vLLM (:8091), exactly the ROS
  bielik node's /wojtek/say_en contract, wired in-process.

## Measured 2026-08-26: the two-box split works, and it is FASTER

Two GB10 instances at the same vast site (0.6 ms TCP between them): box A =
the robot brain (Qwen, Bielik, whisper, Chatterbox), box B = the world
(MuJoCo sim, renderer, recorder), wired over ssh tunnels at the service
boundary. Median mic-to-first-sound **3.94 s** (worst 4.04) against 4.39 s
(worst 4.74) with everything on one box — hardware isolation beats
colocation because the sim's 25 fps renderer stops competing with the
models' GPU. At 0.6 ms the 50 Hz policy loop could also cross this wire.
This is the W3 deployment shape, validated before W3 is written: the
boundary becomes ROS topics instead of tunneled HTTP, the split stays.

## Next (in order)

1. **W3**: wrap the Qwen agent + room sim + demo UI as ROS nodes
   (`vlm_agent`, sim bridge, ui bridge) so the ROS stack walks, not just
   talks.
2. Router fine-tune: `ros/src/wojtek_brain/tools/gen_router_dataset.py` →
   `train_router.py` (HerBERT default; compare mmBERT/ModernBERT on the same
   Polish set), then set the router node's `model_path`.
3. Whisper large-v2 vs large-v3 A/B on real mic audio (v3 hallucinates more
   on silence; guards exist either way).
4. Stage voice decision + clone (Głuś must be replaced before the demo).
5. W4 hardening: failure drills, latency pass, stage runbook.  The
   measurement half of the latency pass exists — every stage of both stacks
   is timed into the session trace and ranked by `./training/run.sh perf`
   ([training/docs/latency.md](../../training/docs/latency.md)); what is
   missing is a live GPU session to record, after which the acceleration
   work is ordered by the critical path instead of by guess.

Branch: `agentic-ros2` (checked out at the repo root for convenient prompt
editing; `claude/quadruped-agentic-ros2-0a713b` is the same line of work).
Demo clips from the validated takes: `~/Desktop/machinekind/2026-08-14/`.
