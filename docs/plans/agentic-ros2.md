# Wojtek agentic stack on ROS 2 — target architecture

Status: agreed design, 2026-08-13. Owner: Maciej. Deadline: on-stage demo in
one month, running on a rented vast.ai GPU box with a soft path to the robot
(Jetson).

Successor to the machinekind/wojtek#131 agent layer (imported on this branch:
`training/wojtek_rl/agent/`, `training/docs/agent.md`,
`training/docs/polish-voice.md`, `skills/wojtek-voice-agent-demo/`). #131 was
a single-process asyncio app; this design re-plumbs the pipeline as native
ROS 2 nodes so the vast.ai research setup migrates 1:1 onto the robot.

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
   (`wojtek_rl/agent/tts.py`), config-switched: Chatterbox multilingual (MIT)
   and the F5-TTS Gregniuki PL fork (CC-BY-NC). Pick by ear later; anything
   public/commercial must run Chatterbox (license) and a non-cloned voice
   (personality rights — see Voice below).
4. **Everything runs on the vast box; the laptop is a dumb mic/speaker.**
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
- **The plan was silent on audio/video transport during the vast phase** —
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

- **ASR**: faster-whisper **large-v3** (FLEURS PL 4.74; 0.36 s measured on an
  RTX 5880). Keep #131's hallucination guards (`no_speech_prob > 0.6`,
  `avg_logprob < -1.0` drop rules).
- **VAD + turn detection**: **pyannote** (segmentation model streamed) per
  the team decision; it also gives turn-taking signals the energy gate
  can't. Risk: heavier than silero — the #131 `VoiceSegmenter` keeps its
  `vad=` hook, so silero remains a drop-in fallback if pyannote adds
  latency.
- **TTS**: dual engine (Decision 3). Neither streams natively → sentence
  pipelining: `bielik` flushes on punctuation, `tts` synthesizes per
  sentence (F5 RTF ~0.12 GPU → ~0.4 s for a short sentence; Chatterbox has a
  streaming fork if first-chunk latency disappoints).
- **Voice**: cloning prep pipeline exists (`f5_prep.py` scratchpad: whisper
  timestamps → best 3–8 s window → loudnorm; denoise refs for Chatterbox).
  **Głuś/Walaszek voices are private-tinkering only** — a cloned character
  voice on a public stage is a personality-rights exposure and F5's PL
  checkpoint is CC-BY-NC. Ship a neutral cloned-from-licensed or stock
  Polish voice as the stage default; character voices stay a config flag.

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

## Deployment (vast.ai)

- One ≥48 GB card (or 2×24): Qwen3-VL-4B-FP8 ~8 GB + KV, FutureNav ~10 GB
  (grows with VGGT cache), whisper large-v3 ~4 GB, Bielik-4.5B-v3 ~5–6 GB
  (FP8) — Bielik-11B wants the second card. pyannote + TTS ~4 GB.
- Provisioning extends `scripts/vast_stack.sh` (cluster-agnostic, env-driven;
  keep the cu124 pin, `ninja`, `VLLM_USE_FLASHINFER_SAMPLER=0` lessons).
  ROS 2 Jazzy on the box via the official binary or a `ros:jazzy` base image;
  all nodes on localhost DDS (`ROS_LOCALHOST_ONLY=1`).
- Machine-specific scripts stay in the private operations repository — this
  repo never names private hosts (see root `CLAUDE.md`).

## Package layout

```
ros/src/wojtek_agent_msgs/   # AudioChunk, Transcript, RoutedIntent, Sentence, goal msgs
ros/src/wojtek_voice/        # audio_bridge, vad_node, asr_node, tts_node
ros/src/wojtek_brain/        # router_node, bielik_node, vlm_agent_node (wraps wojtek_rl.agent)
ros/src/wojtek_sim_bridge/   # room-sim node + ui/trace websocket bridge
training/jobs/agent_stack.sh # parameterized vast payload (env-declared, no secrets)
```

Python nodes are `ament_python` packages; their heavy model deps
(faster-whisper, pyannote, TTS engines) are declared in a per-package
`requirements.txt` installed into the deployment venv by the provisioning
payload — `package.xml` carries only ROS deps. Apache-2.0 everywhere.

## Milestones (4 weeks)

1. **W1 — hear yourself think.** `wojtek_agent_msgs`; `audio_bridge` (reuse
   the #131 mic worklet); `vad` + `asr` nodes; router dataset + first
   fine-tune; E2E on vast: speak Polish in browser → `/wojtek/asr/final`.
2. **W2 — talk.** `bielik` + `tts` nodes (both engines), barge-in path,
   router in the loop; talk-only stage demo works end to end.
3. **W3 — walk and look.** `vlm_agent` node wrapping the #131 modules; sim
   node + ui_bridge; FutureNav + SCAN wired; visual-question translate path;
   full loop: "znajdź krzesło" → search → spoken outcome.
4. **W4 — harden.** Latency budget pass, failure drills (model down, GPU
   OOM, network blip), demo recording (`agent.record` port), stage runbook,
   teardown (`vast_destroy_at.sh`).

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
