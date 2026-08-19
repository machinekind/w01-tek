# Wojtek agentic stack on ROS 2 — target architecture

Status: agreed design, 2026-08-13. Owner: Maciej. Deadline: on-stage demo in
one month, running on a rented cloud GPU box with a soft path to the robot
(Jetson).

Successor to the machinekind/wojtek#131 agent layer (imported on this branch:
`wojtek_agent/`, `docs/agent-layer.md`,
`docs/polish-voice-research-report.md`, `skills/wojtek-voice-agent-demo/`). #131 was
a single-process asyncio app; this design re-plumbs the pipeline as native
ROS 2 nodes so the rented-GPU research setup migrates 1:1 onto the robot.

## Decisions (settled, do not re-litigate without new evidence)

1. **Hybrid port.** The AI pipeline (VAD, ASR, router, LLM, VLM, TTS) becomes
   ROS 2 nodes. The #131 demo UI and trace stay as-is, bridged by one
   websocket bridge node. The #131 agent modules (chat/tools/goals/search)
   are libraries wrapped by nodes, not rewritten.
2. **Brain = router + Bielik + Qwen VLM.**
   - A small encoder classifier routes each final utterance.
   - Simple conversation, general knowledge, robot bio → **Bielik** answers
     directly (Polish-native LLM, streams sentences to TTS).
   - Navigation instruction → **both**: Bielik speaks a quick acknowledgement
     while the **Qwen3-VL agent** (#131 chat loop + tools) owns search / goal
     FSM / FutureNav execution.
   - Visual question ("co widzisz?") → Qwen3-VL with the `look` tool, answer
     generated in English, streamed through Bielik for Polish translation.
3. **TTS: both engines behind the existing `TtsEngine` protocol**
   (`wojtek_agent/tts.py`), config-switched: Chatterbox multilingual (MIT)
   and the F5-TTS Gregniuki PL fork (CC-BY-NC). Pick by ear later; anything
   public/commercial must run Chatterbox (license) and a non-cloned voice
   (personality rights — see Voice below).
4. **Everything runs on the GPU box; the laptop is a dumb mic/speaker.**
   Browser AudioWorklet → websocket → `audio_bridge` node → topics, and TTS
   frames back the same way. This is the robot topology (mic/speaker local to
   the compute) so the migration stays 1:1.
5. **Simulator: MuJoCo, single engine.** No Gazebo. The model source, the
   training stack, and `wojtek_mujoco_hardware_interface` are all MuJoCo.

## Colleague-plan review (what we keep, what we corrected)

Kept: node-per-model process isolation; streaming pipeline with
sentence-boundary token flush; QoS split; walking policy decoupled from AI
latency, consuming only high-level intent.

Corrected:

- **Zero-copy IPC: deferred.** True zero-copy (loaned messages) requires
  rclcpp + fixed-size types; an rclpy subscriber deserializes (copies)
  regardless. A frame copy on localhost costs ~1 ms against 300–1700 ms of
  VLM inference. `SensorDataQoS` + JPEG-compressed image topics now; NITROS /
  Isaac ROS on the Jetson later if profiling ever demands it.
- **C++ walking-policy rewrite: not on the critical path.** The numpy rclpy
  `wojtek_policy` node already deploys to the real robot. Process separation
  already gives the isolation the C++ argument wants. A C++ port is a
  hardware-track task, independent of this demo.
- **GIL framing.** Process-per-node is right, but the payoff is crash
  isolation, per-node dependency sets, and independent lifecycles; GPU-bound
  tensor ops release the GIL anyway.
- **The plan was silent on audio/video transport during the rented-GPU phase** —
  that is Decision 4 and the `audio_bridge` node.
- It's a quadruped, not a humanoid.

## Node graph

```
browser (mic worklet + speaker + #131 UI)
   │ websocket (PCM16 mono 16 kHz, 100 ms frames; JSON control)
   ▼
audio_bridge ──/wojtek/audio/mic──▶ vad ──/wojtek/audio/speech──▶ asr
   ▲                                 │                             │
   │                          speech_started              /wojtek/asr/partial
   │                          (barge-in)                  /wojtek/asr/final
   │                                 │                             ▼
   │                                 ▼                          router
   │◀──/wojtek/tts/audio── tts ◀──/wojtek/say── bielik ◀── /wojtek/intent
   │                                              ▲                │
   │                                       translate stream        │ nav / visual
   │                                              │                ▼
ui_bridge ◀──/wojtek/trace/*────────────── vlm_agent (chat loop, tools,
                                             goal FSM, VLFM-lite search)
                                                  │
                                    /wojtek/nav/instruction → FutureNav client
                                                  │
                                                  ▼
                             sim node (MuJoCo room scene, self-stepping)
                             camera → /wojtek/camera/image_raw/compressed
                             mid-level cmd → SCAN planner → walking policy
```

vLLM serves Qwen3-VL and Bielik as two models (or two vLLM instances);
`bielik` and `vlm_agent` nodes are OpenAI-API clients, so the LLM processes
never share the nodes' Python runtime.

### Topics and QoS

| topic | type | QoS |
|---|---|---|
| `/wojtek/audio/mic` | `AudioChunk` (PCM16, seq, stamp) | SensorData (BestEffort, small depth) |
| `/wojtek/audio/speech` | `AudioChunk` + utterance_id, end flag | Reliable |
| `/wojtek/audio/speech_started` | `std_msgs/Empty`-like event | Reliable — barge-in trigger |
| `/wojtek/asr/partial`, `/asr/final` | `Transcript` (text, conf, utterance_id) | Reliable |
| `/wojtek/intent` | `RoutedIntent` (class, conf, text) | Reliable |
| `/wojtek/say` | `Sentence` (text, utterance_id, seq, final) | Reliable |
| `/wojtek/tts/audio` | `AudioChunk` | Reliable |
| `/wojtek/camera/image_raw/compressed` | `sensor_msgs/CompressedImage` | SensorData |
| `/wojtek/nav/*`, `/wojtek/goal/*` | goal cmd / status | Reliable |
| `/wojtek/trace/*` | JSON string events (existing trace schema) | Reliable |

Barge-in contract: `speech_started` makes `tts` drop its queue and
`audio_bridge` flush the browser playback (#131 measured 90 ms); `bielik`
cancels in-flight generation for the interrupted utterance_id.

## Router

- Model: small multilingual/Polish encoder fine-tuned on a **synthetic
  classification set** (generated by a big LLM, reviewed by hand). ModernBERT
  is English-centric — evaluate **HerBERT-base** and **mmBERT/EuroBERT**
  against it on the same set before committing; the router must read Polish.
- Classes (v1): `chat` (small talk, bio, general knowledge), `nav`
  (movement/route/search instruction), `visual` (asks about the current
  view), `cancel` (stop/abort), `system` (volume, reset, mode).
- Latency budget ≤ 30 ms on GPU; it sits on every utterance.
- Fallback: below confidence threshold → `chat` (Bielik), which can still
  hand off — wrong-side errors then cost words, not runaway navigation.

## Speech stack

- **ASR**: faster-whisper, model configurable (`model` parameter).
  **Benchmark large-v2 against large-v3 on our own Polish recordings in W1**
  — v3 scores better on FLEURS (PL 4.74; 0.36 s measured on an RTX 5880)
  but is known to hallucinate more on silence/noise, and v2 sometimes wins
  in practice. The #131 guards (`no_speech_prob > 0.6`,
  `avg_logprob < -1.0` drop rules) stay either way.
- **VAD: silero by default** (decided 2026-08-13 on license/latency/memory/
  accuracy: MIT, ~2 MB model, <1 ms per frame on CPU, streaming-native
  512-sample windows, VAD accuracy on par with heavier models). pyannote
  segmentation-3.0 stays a config-selectable backend (MIT weights but
  HF-gated, needs ~1.6 s context ring + torch/pyannote.audio stack, tens of
  ms per window) — worth revisiting only if the stage room defeats silero
  or we want model-based turn-taking; turn-end today is the segmenter's
  silence timeout. The energy gate remains the zero-dependency fallback.
- **TTS**: dual engine (Decision 3). Neither streams natively → two-level
  streaming: sentence pipelining always (`bielik` flushes on punctuation,
  `tts` synthesizes per sentence; F5 RTF ~0.12 GPU → ~0.4 s for a short
  sentence), plus intra-sentence streaming for Chatterbox.  **Measured
  2026-08-13: the `chatterbox-streaming` fork is English-only (no
  `mtl_tts`), so Polish intra-sentence streaming means porting the fork's
  chunked decode onto the multilingual model ourselves.**  Sentence
  pipelining alone measured fine on an RTX 6000 Ada (~2 s per short
  sentence at ~50 tok/s T3 sampling); revisit only if stage latency
  demands it.
- **Voice**: cloning prep pipeline exists (`f5_prep.py` scratchpad: whisper
  timestamps → best 3–8 s window → loudnorm; denoise refs for Chatterbox).
  **Głuś is the development default for now; it will be replaced before the
  stage** — a cloned character voice in public is a personality-rights
  exposure and F5's PL checkpoint is CC-BY-NC. The stage voice must be
  chosen and cloned (or stock) by end of W3 so rehearsals use the real one.

## Simulator strategy

Phase 1 (demo path): wrap the existing room sim (`wojtek_rl` MuJoCo scene +
walking policy + SCAN planner) as a **self-stepping headless ROS node**
publishing the camera and consuming mid-level commands. Two #131 defects to
fix in the wrap: the sim only stepped while a browser was attached, and any
VLM-facing frame must be HUD-free (`ego_jpeg(hud=False)`).

Phase 2 (hardware parity, post-demo): the locomotion layer moves to
`ros2_control` + `wojtek_mujoco_hardware_interface`, which is the exact
robot topology; the AI graph above does not change — that is the point of
the intent-vector boundary.

## Scenes

The demo scenes were dark and sparse ("room" = Van Gogh's Arles bedroom,
"apartment").  **"castle"** (Skokloster castle hall from the same Habitat
test-scenes zip — bright 18×21 m baroque gallery: checkered floor, table
with chairs, fireplace, horse paintings) is the pretty default now; no new
licensing, rebuilt with:

```bash
./training/run.sh room-assets --zip-member skokloster-castle.glb \
    --name castle --max-extent 40 --skip-collision --up z
./training/run.sh build-room --name castle
python -m wojtek_eval.gridmap --name castle      # occupancy for SCAN/search
SCENE=castle ./experiments/autonomous_architecture_ros2_v1/run.sh room ...
```

`objects.json` is hand-annotated (mesh-band scatter, see the file comment).
**"flat"** is the regular-home scene: ReplicaCAD baked stage sc0_00 (CC-BY
4.0, public HF download, no agreement) — modern apartment with sofa, TV,
kitchen strip, bookshelves, stairs.  Use the `stages_uncompressed` GLBs;
the compressed ones carry KTX2 textures trimesh cannot decode.  More
CC-BY interiors (Hallwyl museum rooms, Sketchfab apartment scans) need a
logged-in download — see the scene LICENSE.md files for the pattern.
**R2R/RxR scenes are Matterport3D houses** — meshes require the signed MP3D
Terms of Use (non-commercial research; email form, human approval), and
HM3D likewise.  Once granted, MP3D ships per-house GLBs that go through the
same `room-assets` pipeline (single floor, or z-cropped), and running in
them matches FutureNav's training distribution — better nav, not just
better looks.  Someone with signing authority has to request access; the
meshes stay out of this public repository either way (only `objects.json`
annotations are committed, same as "apartment").

## Commercial-fork audit (endgame: fork + monetize)

The CODE is Apache-2.0 throughout — forking and monetizing it is exactly
what the license permits.  Every constraint lives in assets and weights:

| component | license | commercial fork |
|---|---|---|
| whisper, chatterbox-tts, silero VAD, pyannote weights | MIT | fine |
| Bielik v3, Qwen3-VL | Apache-2.0 | fine |
| Piper `pl_PL-mc_speech` voice | CC0 (dataset) | fine |
| castle / van-gogh / apartment scenes | CC-BY (Sketchfab scans) | fine, keep attribution |
| MP3D / HM3D scenes (R2R/RxR houses) | signed ToS, non-commercial research | **dev/eval only — never in a product, never on a promo stage, never trained into shipped weights** |
| F5 Gregniuki PL checkpoint | CC-BY-NC | **out of any commercial build** |
| Głuś / Walaszek cloned voices | personality rights | **dev only; stage voice replaced (decided)** |
| FutureNav weights | **unverified** | check before it becomes load-bearing commercially; it is itself trained on MP3D-family data |

Working rules that keep the fork clean: research-restricted scenes are
runtime assets on dev machines only (this repo never carries meshes —
existing rule); public-facing footage (stage, marketing clips) uses only
the CC-BY scenes; none of OUR trained artifacts (router, fine-tunes,
cloned voices we ship) may be trained on ToS-restricted data.  A real
counsel pass is due before actual monetization — this table is an
engineering audit, not legal advice.

## Deployment (rented cloud GPU)

- One ≥48 GB card (or 2×24): Qwen3-VL-4B-FP8 ~8 GB + KV, FutureNav ~10 GB
  (grows with VGGT cache), whisper large-v3 ~4 GB, Bielik-4.5B-v3 ~5–6 GB
  (FP8) — Bielik-11B wants the second card. pyannote + TTS ~4 GB.
- Image must be **`nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04`** — ROS 2
  Jazzy exists only for Noble, and faster-whisper needs the cudnn variant.
  The node venv uses `--system-site-packages` so it sees apt rclpy, and
  colcon builds with that venv active so installed node scripts get venv
  shebangs.  Bielik default:
  `speakleash/Bielik-4.5B-v3.0-Instruct-FP8-Dynamic` (~5 GB, Ada+; bf16 repo
  on Ampere).  The talk-only stack fits a 24 GB card; the full W3 stack
  (+ Qwen + FutureNav) returns to the ≥48 GB sizing above.
- Provisioning scripts, host selection and scheduled teardown live in the
  private operations repository — this repository never names private hosts
  or a specific provider account (see root `CLAUDE.md`).  Only
  cluster-agnostic payloads belong here.

## Package layout

```
experiments/autonomous_architecture_ros2_v1/ros/src/
  wojtek_agent_msgs/         # AudioChunk, Transcript, RoutedIntent, Sentence, goal msgs
  wojtek_voice/              # audio_bridge, vad_node, asr_node, tts_node
  wojtek_brain/              # router_node, bielik_node, vlm_agent_node (wraps wojtek_agent)
  wojtek_sim_bridge/         # room-sim node + ui/trace websocket bridge
experiments/autonomous_architecture_ros2_v1/
  wojtek_agent/              # chat, tools, goal FSM, search, TTS/ASR services (libraries)
                             # audio_frames.py + speech_text.py are shared with the nodes
  wojtek_demo/               # the interactive demo app (fork of wojtek_rl.room_app)
  tests/                     # model-free unit tests for the above
training/jobs/agent_stack.sh # parameterized GPU-box payload (env-declared, no secrets)
```

Python nodes are `ament_python` packages; their heavy model deps
(faster-whisper, pyannote, TTS engines) are declared in a per-package
`requirements.txt` installed into the deployment venv by the provisioning
payload — `package.xml` carries only ROS deps. Apache-2.0 everywhere.

## Milestones (4 weeks)

1. **W1 — hear yourself think.** `wojtek_agent_msgs`; `audio_bridge` (reuse
   the #131 mic worklet); `vad` + `asr` nodes; router dataset + first
   fine-tune; E2E on the GPU box: speak Polish in browser →
   `/wojtek/asr/final`.
2. **W2 — talk.** `bielik` + `tts` nodes (both engines), barge-in path,
   router in the loop; talk-only stage demo works end to end.
3. **W3 — walk and look.** `vlm_agent` node wrapping the #131 modules; sim
   node + ui_bridge; FutureNav + SCAN wired; visual-question translate path;
   full loop: "znajdź krzesło" → search → spoken outcome.
4. **W4 — harden.** Latency budget pass, failure drills (model down, GPU
   OOM, network blip), demo recording (`agent.record` port), stage runbook,
   scheduled teardown of the rented box (operations repository).

Every node ships with model-free unit tests in the #131 style (fakes, no GPU)
so `./training/run.sh test` / `colcon test` stay fast and honest.

## Risks

- **pyannote streaming latency** — mitigated by the silero fallback hook.
- **Bielik translation quality/latency for the visual path** — measure early
  in W2; fallback is Qwen answering in Polish directly (`think_en` mode was
  measured fine in #131).
- **Two LLMs + FutureNav on one card** — VRAM plan above is tight on 48 GB;
  budget a 2×24 or 80 GB offer as plan B.
- **Voice rights on stage** — decided above; needs an actual chosen stage
  voice by W2.
