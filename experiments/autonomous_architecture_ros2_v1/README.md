# Experiment: autonomous architecture on ROS 2, v1

> **Status: EXPERIMENTAL. Not production, not on the robot.**
> Nothing here is deployed by `ros/deploy.sh`, and no package here is a
> dependency of `wojtek_bringup`. Treat every interface in this directory as
> unstable: names, topics and message fields change without a migration path.

A conversational, voice-driven agent layer for Wojtek, ported to ROS 2 nodes so
that a research setup running on a rented cloud GPU box migrates 1:1 onto the
robot later. W1 and W2 of the architecture are built and verified live; W3
(walking through the ROS stack rather than the demo app) is not.

| document | what it is |
|---|---|
| [docs/architecture.md](docs/architecture.md) | target architecture and the settled decisions |
| [docs/w1-w2-implementation-report.md](docs/w1-w2-implementation-report.md) | what is built and verified, with the findings behind it |
| [docs/agent-layer.md](docs/agent-layer.md) | agent design reference: tools, prompting contracts, measured latencies |
| [docs/polish-voice-research-report.md](docs/polish-voice-research-report.md) | Polish voice research report and decision record |

## Layout

```
ros/src/wojtek_agent_msgs/     AudioChunk, Transcript, RoutedIntent, Sentence
ros/src/wojtek_voice/          audio_bridge, VAD, ASR, TTS nodes + web/mic.html
ros/src/wojtek_brain/          router, Bielik node, router training tools
ros/src/wojtek_agent_bringup/  voice_stack.launch.py
wojtek_agent/                  chat, tools, goal FSM, search, spatial map, trace,
                               TTS/ASR services, scenario driver.  Two modules are
                               shared with the ROS nodes on purpose:
                                 audio_frames.py  segmentation/framing primitives
                                 speech_text.py   spoken-text hygiene, sentence split
wojtek_demo/                   the interactive demo app (fork of wojtek_rl.room_app)
tests/                         model-free unit tests for wojtek_agent
run.sh                         test | room | build
```

## The isolation rules this experiment follows

The point of `experiments/` is that an experiment can be deleted in one
`rm -rf` and nothing else notices. Concretely:

1. **Nothing outside this directory imports anything inside it.** The
   dependency arrow points one way: this experiment imports `wojtek_rl` and
   `wojtek_eval`, never the reverse. `training/`'s own test suite passes with
   this directory deleted, and that is worth re-checking whenever the boundary
   moves.
2. **No shared build or test config knows about the experiment.** `ros/sim.sh`,
   `ros/docker/compose.yaml`, `training/run.sh` and `training/pyproject.toml`
   are untouched; `run.sh` here sets up its own `PYTHONPATH` and runs its own
   `colcon build`.
3. **The ROS packages live outside `ros/src`**, so `ros/deploy.sh` — which
   rsyncs `ros/src/` and builds `--packages-up-to wojtek_bringup` — cannot ship
   them to the robot even by accident.
4. **Behaviour the experiment needs from existing code is added by extension,
   not edit**: `wojtek_agent/nav.py` subclasses `VlmNavigator`,
   `wojtek_agent/futurenav_client.py` subclasses `FutureNavVlmClient`, and the
   demo app is a fork rather than a set of patches to the training project's.
5. **One implementation of shared logic.** The ROS voice/brain nodes import
   `wojtek_agent.audio_frames` and `wojtek_agent.speech_text` instead of
   carrying hand-synced copies of the segmenter and the sentence splitter.

What this experiment *does* change outside itself, deliberately and in full:

| change | why it is not experiment plumbing |
|---|---|
| `CLAUDE.md` | declares the `experiments/` boundary these rules describe |
| `training/wojtek_eval/hearing.py` (+ its test) | `Transcriber(language=...)` instead of a hardcoded `"en"` — a strict generalisation, default unchanged |
| `training/wojtek_rl/vlm_nav.py` (+ its test) | evidence-based stop for uncapped runs; a navigator improvement that stands on its own |
| `training/assets/scenes/{castle,flat}/`, `ros/src/wojtek_description/mujoco/scene_{castle,flat}.xml` | new scenes, in the paths the training scene pipeline owns |
| `training/.gitignore` | unrelated fix: `dir/` patterns do not match a worktree's symlinked asset dirs |
| `skills/wojtek-voice-agent-demo/` | the demo runbook, where all skills live |

## Running the tests

```bash
./experiments/autonomous_architecture_ros2_v1/run.sh test
```

263 model-free tests: no GPU, no ROS runtime, no env instantiation. The suite
uses `training/.venv/bin/python` when it exists; in a worktree pass an
interpreter explicitly with `EXP_PY=/path/to/python`.

## Running the stack

The interactive demo (agent + sim + UI):

```bash
SCENE=flat ./experiments/autonomous_architecture_ros2_v1/run.sh room \
    --vlm-backend futurenav \
    --vlm-url http://127.0.0.1:8100 --agent-url http://127.0.0.1:8090
```

That runs `wojtek_demo/room_app.py`, this experiment's fork of the training
project's room demo. `./training/run.sh room` still starts the original,
agent-free demo — the fork exists so the experiment can evolve the app (and be
replaced by ROS nodes at W3) without the training project carrying agent code.

The talk-only ROS stack, after `run.sh build` in an environment with ROS 2
sourced:

```bash
ros2 launch wojtek_agent_bringup voice_stack.launch.py \
    tts_engine:=silent asr_model:=tiny        # CPU smoke configuration
```

The node venv needs this directory on `PYTHONPATH` so `wojtek_agent` is
importable next to the colcon overlay.

Model serving, host provisioning and teardown for a rented box are **not** in
this repository — see the root `CLAUDE.md`. `skills/wojtek-voice-agent-demo/`
carries the demo runbook.
