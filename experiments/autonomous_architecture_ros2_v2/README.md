# Autonomous architecture v2 — the W3 split, live-verified

The second iteration of the agentic ROS 2 stack: the demo's Qwen agent
(chat loop, tools, goal FSM, VLFM-lite search) runs UNMODIFIED as a ROS
node against a `WorldProxy`, the MuJoCo room sim is another node, and the
two halves meet only at topics — verified on rented GB10s both one-box and
as a true two-box brain/world split (zenoh over an ssh tunnel).

Layout (self-contained; the robot model and walking policy are shared from
the repo root):

- `ros/src/` — the six agentic packages (`wojtek_agent_msgs`,
  `wojtek_voice`, `wojtek_brain` with the `vlm_agent` node, bringup, the
  passive latency probe, `wojtek_sim_bridge`).
- `training/` — the vendored demo/agent tree this stack wraps (`wojtek_rl`,
  `wojtek_eval`, scenario scripts + recorder, unit tests). `assets` is a
  symlink to the repo root's.
- `spark/` — session rigs: deploy/serve/record on rented GB10s, one-box and
  split, with the GPU-clock gate and the rental kill-timer.
- `docs/` — architecture, state-of-play, latency/TTS/canned-voice guides.

Highlights over v1: sequence-fenced world commands, the canned Polish
phrase bank (prerecorded via the TTS server's disk cache), a rehearsed
intro, voice-triggered show tricks incl. the COM-analysed pee clip, ASR/
LLM/TTS warm gates, and drift-free take recording.

Start with `docs/plans/agentic-ros2-status.md`.
