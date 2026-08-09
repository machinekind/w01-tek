# Wojtek instruction-following: sim, model stack, eval, deploy — roadmap

> Research synthesis, 2026-07-13. Goal: Wojtek follows verbal/text instructions
> ("go to the chair", "find the ball", "turn around"). No speech output, no
> manipulation. Hardware target: RealSense + Jetson + mic. Training compute:
> vast.ai RTX; locomotion training stays on the 4×H100 node.

## 0. Where we already are

The `vlm-local` branch already implements the architecture the field converged
on (NaVILA, RSS 2025): a slow VLM (~0.3 Hz) emits mid-level commands
(`turn_left/right deg`, `forward/backward m`, `stop`, `done`) that a fast
50 Hz RL velocity policy executes, with stall/blocked detection and safety
clamps (`wojtek_brain/vlm_nav.py`, `midlevel.py`). One scanned room
(van-gogh-room) with CoACD collision in MuJoCo, ego camera, closed loop with
Claude API or local Qwen3-VL-30B (mlx-vlm). What's missing is exactly what
this doc plans: **scale (scenes, episodes), a real evaluation harness, a
deployable-size model, exploration ("find X" when X is not visible), speech
input, and the Jetson port.**

Key validation from the literature: NaVILA's physics-in-the-loop benchmark
showed legged execution changes VLN results materially (~14 % SR gap vs blind
low-level); our two-level split matches their design one-to-one. LocalNav
(2026) distilled a frontier-VLM system into Qwen-4B running **onboard a Jetson
AGX Orin** at ~39 tok/s (IQ4-XS quant) — proof our deployment envelope works.

## 1. Target deployment stack (the dog)

```
mic ──► openWakeWord ──► faster-whisper / whisper_trt (small/distil) ──► goal text
                                                                            │
RealSense D435i ──► depth ──► occupancy/frontier map ─────────┐             ▼
                └─► RGB ego frame ────────────────────────────┴──► VLM (Qwen3-VL 4B, INT4)
                                                                    ~0.3–0.5 Hz, terse JSON command
                                                                            │
                                                    mid-level executor (existing, 50 Hz)
                                                                            │
                                                    velocity policy locomotion_v8 (existing)
```

- **Jetson**: **AGX Orin 64 GB** recommended (~$2k devkit, 15–60 W nMode caps;
  comfortable for 4–8B VLM + ASR + RealSense + ROS2 concurrently).
  **Orin NX 16 GB** is the aggressive minimum (≤4B VLM, terse outputs,
  ~0.15–0.3 Hz). **Thor** ($3.5k, ~100 W+) is wrong for a small dog's battery.
- **Latency reality** (AGX Orin, 4B INT4 via llama.cpp/MLC): ~30–40 tok/s
  decode + 1–2 s image prefill. A 100-token output ⇒ ~4–5 s/decision; a
  **15–30-token structured command ⇒ 2–3 s ⇒ 0.3–0.5 Hz**. Our JSON one-liner
  format is already right; keep `reasoning` to one short clause or drop it on
  device.
- **ASR**: whisper_trt (TensorRT, ~3× faster than PyTorch on Orin) or
  faster-whisper; NVIDIA Riva/Parakeet as the all-NVIDIA alternative. Gate
  with openWakeWord (negligible CPU).
- **RealSense**: D435i (lighter, cheaper, IMU) unless we need range → D455.
  Known JetPack 6 USB pain: build librealsense with the RSUSB backend (no
  kernel patches), pin librealsense+firmware versions, quality USB3 cable,
  udev rules. Budget a day of fighting it.
- **Dev fallback**: stream JPEG ego frames over WiFi to a workstation/4090 for
  the VLM (NaVILA did exactly this; ~0.3–1.4 s round-trip), keep locomotion +
  safety onboard. Ship onboard once the distilled 4B works.

## 2. Simulation plan

Two-tier, keeping MuJoCo for what it's uniquely good at:

- **Tier A — fast VLM-iteration loop (new):** abstracted motion, no legged
  physics. A kinematic velocity integrator (or teleport-with-collision-check)
  executes mid-level commands in the scene mesh; ego RGB rendered offscreen.
  100–1000× faster than stepping the walker; this is where episodes, prompts,
  and models get iterated. Sits behind the *same* `VlmClientProto` /
  command interface, so results transfer.
- **Tier B — physics gate (exists):** the current `room_app` loop —
  full MuJoCo + locomotion_v8 + MidLevelExecutor. Run the same episode suite
  here before any hardware test; it catches falls, blocked-robot, stall-abort
  behavior that Tier A hides. This mirrors NaVILA's VLN-CE-Isaac finding.
- **Scenes:** do **not** hand-convert HM3D/MP3D photogrammetry into MuJoCo
  (monolithic meshes → thousands of CoACD hulls, no semantics, non-commercial
  licenses). Scale with **object-composed scenes** where per-asset CoACD is
  tractable and semantics are known by construction:
  - **HSSD** (211 scenes, 18k objects, CC-BY) — composable via the existing
    `room_assets` pipeline generalized to per-object conversion (`obj2mjcf`).
  - **Infinigen Indoors** (BSD, procedural, unbounded) — heavier Blender
    setup, but license-clean and infinite; semantics + collision baked at
    generation time.
  - Keep the scanned room as the "golden" held-out scene; add 1–3 more phone
    scans of real rooms we can also test in physically.
- **Camera realism:** VLMs are trained on real images — render fidelity is
  not the bottleneck. Add cheap RGB domain randomization (lighting, exposure,
  white balance, camera intrinsics/height jitter) to ego frames.
- **Watch:** GaussGym (arXiv 2510.15352) — Gaussian-splat renderer inside
  vectorized sims, iPhone scans in, legged policies out. Closest published
  system to our splat-room premise; could merge the tiers later.

## 3. Model stack (pre-trained first, train later)

Phase order: frozen models → measure → fine-tune only what the eval says is
weak.

1. **Nav VLM (frozen v0):** swap the local backend to a deployable size —
   **Qwen3-VL-4B / 8B (instruct)** behind the existing `LocalVlmClient`
   interface. The 30B-A3B stays as the Mac dev model / strong baseline;
   Claude API stays as the teacher/upper-bound baseline.
2. **Exploration for "find X" (new component):** frozen-VLM frontier
   navigation, VLFM-style — build a 2D occupancy map from depth (sim: ego
   depth camera; real: RealSense), extract frontiers, score them with a
   frozen VLM/ITM ("photo of a ⟨target⟩" similarity), navigate to the best
   frontier until the target is visible, then hand off to the existing
   visible-goal loop. Zero-training, proven on Spot (VLFM, arXiv 2312.03275)
   and it cleanly separates *explore* from *servo*. A lighter fallback is
   LOVON-style: open-vocab detector (YOLO-World/OWLv2) + rotate-to-search
   state.
3. **ASR:** whisper-family, unchanged interface: text goal in, same pipeline.
   Trivially testable in sim by typing instead of speaking.
4. **Locomotion:** locomotion_v8 unchanged. Only revisit if the eval shows
   command-execution failures (e.g. VLM-commanded distributions the policy
   wasn't trained for).

Released systems to borrow from / benchmark against: NaVILA checkpoints
(`a8cheng/*`, 8B, ~1 FPS on 4090), StreamVLN (code+weights, SR 56.9 R2R-CE),
Uni-NaVid, Qwen-RobotNav (Qwen3-VL 2B/4B/8B, waypoint output, reported
onboard Jetson — verify arXiv 2606.18112 before relying on numbers).

## 4. Evaluation (build this before training anything)

Extend the "battery" culture from locomotion to navigation:

- **Episode generator:** sample (scene, start pose, goal object/instruction)
  with a guaranteed-reachable target: geodesic distance from a navmesh /
  occupancy grid gives shortest-path length ℓ*, required for SPL.
- **Success check (automatic):** stop within 1.0 m of target instance AND
  target visible from stop pose (render + check, or ground-truth visibility).
- **Metrics (standard, comparable to literature):** SR, SPL, SoftSPL,
  distance-to-goal; add OSR to diagnose early/late stopping. Later, nDTW if
  we do route-following instructions.
- **Task suite v1:** (a) visible-goal go-to, (b) hidden-goal find (requires
  exploration), (c) relational/turn commands ("turn around", "go to the door
  behind you"), (d) multi-step ("go to the bed then to the plant"). ~50–200
  episodes per scene, fixed seed set, one JSON scoreboard per model —
  same discipline as `./run.sh battery`.
- **Two rungs:** every model change runs Tier A (fast); promotion to hardware
  requires Tier B (physics) on the golden scenes.
- **Cost:** a few thousand Tier-A episodes is a single-GPU, few-hours, <$10
  job — dominated by VLM forward passes, not rendering (Madrona-MJX does
  batched ego-RGB at 10⁴–10⁵ steps/s if we ever need scale).

## 5. Training plan (vast.ai), only after eval exists

Evidence says fine-tuning beats frozen on ObjectNav-style tasks
(FiLM-Nav +14 SR / +10 SPL over frozen VLFM), and RL-after-SFT is very
data-efficient (VLN-R1: 2B + GRPO ≈ 7B SFT, 20K samples).

1. **Oracle SFT data:** privileged planner in sim (shortest-path or greedy
   frontier oracle — the latter avoids overfitting to omniscient paths) emits
   optimal mid-level commands; render ego RGB at decision points. Target
   **50–120K decision points** across scenes/tasks.
2. **Teacher distillation set:** a few hundred–few thousand *successful*
   Claude-driven trajectories (the existing anthropic backend is the data
   collector — log everything now) to teach instruction reasoning.
   LocalNav claims ~500 traces sufficed at 4B (verify before betting on it).
3. **Recipe:** LoRA-SFT Qwen3-VL-4B (freeze ViT, small visual-token budget) →
   **GRPO** on success + progress reward in the Tier-A loop.
   Frameworks: **ms-swift** (SFT+GRPO in one tool) or Unsloth (fastest
   single-GPU SFT) + TRL GRPOTrainer.
4. **Budget:** SFT ~$30–120 (1×5090 ~1 day or 4×4090 ~half day); GRPO
   experimentation ~$50–200; data-gen/eval <$50. **Whole training program is
   a few hundred dollars** — the scarce resource is the oracle + reward
   engineering, not GPUs.
5. **Deploy:** quantize the tuned 4B to INT4 (IQ4-XS / AWQ), serve via
   llama.cpp or MLC on the Jetson.

## 6. Suggested milestone order

1. **M1 — Eval harness + Tier-A fast loop** (episode gen, metrics, scoreboard;
   abstracted motion). Everything else is measured against this.
2. **M2 — Deployable frozen baseline:** Qwen3-VL-4B/8B backend + prompt work;
   scoreboard vs 30B and vs Claude. Start logging Claude trajectories as
   future training data.
3. **M3 — Scene scale-up:** HSSD (or Infinigen) → per-object MuJoCo pipeline;
   3–10 scenes with semantics; regenerate episode suite.
4. **M4 — Exploration:** depth→occupancy frontier module + VLM scoring;
   "find X" tasks pass without the goal initially visible.
5. **M5 — Fine-tune:** oracle SFT → GRPO on vast.ai; beat the frozen baseline
   on the scoreboard, then quantize.
6. **M6 — Hardware bring-up (parallel track):** buy AGX Orin 64 GB + D435i +
   mic; port ASR + VLM serving; WiFi-offboard VLM first, onboard after M5.
   Tier-B physics gate before every hardware session.

## 7. Risks / open decisions

- **Jetson model** — AGX Orin 64 GB vs Orin NX 16 GB is a payload/power/budget
  call for a small custom dog; NX halves throughput and caps at ~4B.
- **Latency vs verbosity** — on-device budget forces terse outputs; drop or
  truncate `reasoning` on the Jetson path.
- **RealSense × JetPack 6** — known USB/enumeration issues; pin versions.
- **Licensing** — HM3D/MP3D are non-commercial; HSSD (CC-BY) and Infinigen
  (BSD) are the clean scaling paths.
- **Unverified 2026 papers** — Qwen-RobotNav (2606.18112) and LocalNav
  (2606.27871) numbers came from search snippets; read the PDFs before
  copying their design choices. ("NaVILA 2", "CrossVLN", "TagNav" do not
  exist.)
- **Odometry on real hardware** — sim gives perfect pose to MidLevelExecutor;
  the dog needs onboard odometry (leg kinematics + IMU, or RealSense
  tracking/VIO). This is the biggest sim-vs-real gap in the current code and
  deserves its own spike during M6.

## Key references

- NaVILA: https://arxiv.org/abs/2412.04453 · https://github.com/AnjieCheng/NaVILA · bench: https://github.com/yang-zj1026/VLN-CE-Isaac
- VLFM (frozen frontier nav): https://arxiv.org/abs/2312.03275
- FiLM-Nav (fine-tuned frontier selection): https://arxiv.org/abs/2509.16445
- VLN-R1 (GRPO for nav VLM): https://arxiv.org/abs/2506.17221
- StreamVLN: https://github.com/OpenRobotLab/StreamVLN · LOVON: https://arxiv.org/abs/2507.06747
- GaussGym (splat-rendered legged sim): https://arxiv.org/abs/2510.15352
- HSSD: https://huggingface.co/datasets/hssd · Infinigen Indoors (CVPR 2024)
- obj2mjcf: https://pypi.org/project/obj2mjcf/ · Madrona-MJX: https://github.com/shacklettbp/madrona_mjx
- ms-swift: https://github.com/modelscope/ms-swift · Unsloth Qwen3-VL: https://unsloth.ai/docs
- whisper_trt: https://github.com/NVIDIA-AI-IOT/whisper_trt · openWakeWord: https://github.com/dscripka/openWakeWord
- Jetson VLM numbers: https://multimodalflow.net/en/blog/jetson-orin-llm-benchmark/ · https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics/
- LocalNav (onboard 4B distillation, verify): https://arxiv.org/abs/2606.27871 · Qwen-RobotNav (verify): https://github.com/QwenLM/Qwen-RobotNav
