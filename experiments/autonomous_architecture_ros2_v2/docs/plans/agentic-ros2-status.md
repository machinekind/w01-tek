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

## W3 landed 2026-08-26 — and verified LIVE the same day

The agent is a ROS node and the sim is another. The cut is a **proxy, not a
rewrite**: the whole demo agent stack (WojtekAgent, GoalManager,
VlmNavigator, SearchController) imports from `wojtek_rl` unmodified and
talks to a `WorldProxy` that reconstructs RoomSim's nine-member surface from
topics -- one source of truth for agent logic in both stacks.

- `wojtek_brain/vlm_agent_node.py` -- subscribes `/wojtek/intent` (nav and
  visual are its lanes; cancel aborts; chat stays Bielik's), camera and
  `/wojtek/exec/status` + `/wojtek/map`; publishes English to
  `/wojtek/say_en` (first publisher of that topic -- Bielik renders the
  Polish) and `/wojtek/trace`. Nav intents get no reply text (Bielik's
  canned ack already played); the goal watcher announces terminal outcomes
  in English via the same mouth.
- `wojtek_agent_msgs`: `ExecStatus` (pose + executor + resets heartbeat,
  one snapshot so the pollers never read torn state), `WorldMap` (the
  OnlineMap grid on the wire; the proxy rebuilds a real OnlineMap so
  frontier math and `map_image()` run agent-side unchanged), and the
  `WorldCommand` service (midlevel / goto / reset -- a service because the
  controllers consume the `{ok, error}` ack synchronously).
- `wojtek_sim_bridge/bridge_node.py` -- self-stepping 50 Hz headless sim
  node (fixes the #131 freeze-without-browser defect), HUD-free VLM frames
  (fixes the second), SCAN + walking policy world-side as the body
  emulator, and the browser/scenario websocket on :8010 speaking the
  room_app protocol so `scenario.py` records takes against it unchanged.
  It replaces `audio_bridge` in a sim world (mic and reply audio ride the
  same socket).
- Probe extended: `say_en_first` event, `agent.turn` and `brain.translate`
  stages -- the Bielik hop is now measured per turn in the ROS stack too.
- Launch: `agent_stack.launch.py` (brain box: voice stack + vlm_agent, no
  audio_bridge) and `world.launch.py` (world box: sim bridge).

**Live E2E on a rented GB10 (2026-08-26, `spark_ros_rig.sh`)**: the whole
graph ran -- spoken Polish -> vad -> asr -> router -> vlm_agent (look /
navigate / search tools) -> say_en -> bielik -> tts -> audio out through
the bridge. `ros_castle_tour` walked **5.76 m** on a spoken route
(gate-checked take), planner `goto` and midlevel forward/turn verified to
completion over the service, and the probe measured **voice.reply median
4.69 s** -- the demo rig's band (4.39 one-box / 3.94 split), so the ROS hop
tax is ~zero on healthy hardware.

Two traps the live run found, both now baked into code:

- **The stale-active race**: controllers polled `executor.active` from a
  25 Hz snapshot predating their own command and machine-gunned commands
  3 ms apart (the robot "searched" a flat while moving 6 cm). WorldCommand
  acks carry `cmd_seq`, ExecStatus reports the seq its snapshot reflects,
  and the proxy holds `active` high until status catches up.
- **The GB10 clock lottery**: vast specific marketplace machines is platform power-capped
  at a third of its clocks (TTS RTF 3.5 vs 0.44) and nothing unlocks it;
  deploy now asserts clocks under load and names the machine a lemon
  before any model downloads. Healthy machines boost to ~2.5 GHz.

Not done yet, in order: two-box split of the ROS stack (DDS across boxes
needs unicast peer config on vast -- multicast does not cross their NAT;
zenoh-bridge-ros2dds over the measured 0.6 ms TCP is the likely shape),
flat_guest stagecraft (targets visible from spawn -> a 0 m take; pick
unseen targets or script a turn first, the #131 lesson resurfacing),
browser UI static serving on the bridge (scenario/recorder path works; a
live human viewer still uses room_app for now).

## Added 2026-08-26 evening: authorship, canned voice, show tricks

- **Bielik never authors** when the agent runs: chat routes to the VLM agent
  (Qwen writes English, Bielik renders Polish -- restoring the demo's ear-
  approved pipeline after Bielik-authored replies produced garbled echoes
  and an invented garden on camera). Pose/status questions route visual.
- **Canned Polish phrase bank** (wojtek_brain/prompts/bielik/phrases/):
  acks, progress heartbeats, outcomes, switch and trick lines, sampled
  without repeats, pre-synthesized into the TTS cache at stack start --
  a canned line costs ~0.1 s instead of a 2-4 s synthesis. Onomatopoeia
  banned across the voice path (TTS renders hau/woof as fault-like noises).
- **Voice-triggered show tricks**: main's wojtek_policy.tricks (bow, sit,
  paw_wave, shake) plus a new MuJoCo-validated "pee" clip (rear-left leg
  9.8 cm up, peak torque 7.7 of 9 N*m). Router trick lane (siad / daj łapę /
  ukłoń się / otrząśnij się / siku / sztuczka), WorldCommand kind=trick,
  RoomSim plays the clip with executor and policy benched. Validated take:
  flat_intro_pee -- introduce yourself, find the bike, pee on it (chat +
  search + walk + trick in one recording).

## Two-box ROS split VERIFIED 2026-08-27 (the W3 endgame shape)

Two same-site GB10s, brain (models + voice + vlm_agent) and world (sim
bridge), the ROS graph bridged by zenoh-bridge-ros2dds over one ssh -L
tunnel. Camera streams cross-box at full rate; the ONE thing zenoh did not
carry was service REPLIES (double-router hop), so the world command channel
became the WorldCmd/WorldAck topic pair correlated by req_seq -- the
service remains for one-box debugging. World zenoh listens on 7448 (its
default bind collided with the tunnel). Validated take: ros_flat_grand --
prerecorded intro, sit, paw_wave, bike search with approach, and the pee
trick, every lane crossing the wire (gates green, 0.59 m walked).

## Next (in order)
2. Router fine-tune: `ros/src/wojtek_brain/tools/gen_router_dataset.py` →
   `train_router.py` (HerBERT default; compare mmBERT/ModernBERT on the same
   Polish set), then set the router node's `model_path`.
3. Whisper large-v2 vs large-v3 A/B on real mic audio (v3 hallucinates more
   on silence; guards exist either way).
4. ~~Stage voice decision~~ SETTLED 2026-08-27 (user call): the stock
   Chatterbox multilingual voice, no clone. Code and weights are MIT
   (ResembleAI/chatterbox), so it is commercial- and stage-safe with no
   personality-rights exposure; Głuś retires. The disk-cached prerecorded
   lines are keyed "stock" accordingly.
5. W4 hardening: failure drills, latency pass, stage runbook.  The
   measurement half of the latency pass exists — every stage of both stacks
   is timed into the session trace and ranked by `./training/run.sh perf`
   ([training/docs/latency.md](../../training/docs/latency.md)); what is
   missing is a live GPU session to record, after which the acceleration
   work is ordered by the critical path instead of by guess.

Branch: `agentic-ros2` (checked out at the repo root for convenient prompt
editing; `claude/quadruped-agentic-ros2-0a713b` is the same line of work).
Demo clips from the validated takes: `~/Desktop/machinekind/2026-08-14/`.
