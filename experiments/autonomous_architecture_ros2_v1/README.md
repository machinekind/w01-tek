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
wojtek_agent/                  chat, tools, goal FSM, search, spatial map,
                               trace, TTS/ASR services, scenario driver
tests/                         model-free unit tests for wojtek_agent
```

## Why it lives outside `ros/` and `training/`

- **`ros/src` is the robot workspace.** `ros/deploy.sh` rsyncs `ros/src/` to the
  RPi and builds `--packages-up-to wojtek_bringup`. Keeping these packages out
  of that tree means an experiment can never reach the robot by accident.
- **`training/` is the MJX/Brax training project.** The agent layer is an
  application on top of it, not part of it.
- The PC dev container still builds these packages: `ros/docker/compose.yaml`
  mounts this `ros/src` as a second colcon base path (`/ros2_ws/experimental_src`)
  and `ros/sim.sh --build` passes both base paths.

## Running the tests

```bash
./experiments/autonomous_architecture_ros2_v1/run.sh test
```

791 model-free tests: no GPU, no ROS runtime, no env instantiation. The Python
layer's tests need an interpreter with `numpy` — `run.sh` uses
`training/.venv/bin/python` when it exists, otherwise `python3`.

## Running the stack

The interactive demo (agent + sim + UI, driven from `training/`):

```bash
SCENE=flat ./training/run.sh room --vlm-backend futurenav \
    --vlm-url http://127.0.0.1:8100 --agent-url http://127.0.0.1:8090
```

`training/run.sh` puts this directory on `PYTHONPATH`, because
`wojtek_rl.room_app` and `wojtek_rl.vlm_nav` import `wojtek_agent`. That
dependency direction — training code importing an experiment — is a known wart
of the current split, and it resolves when W3 moves the demo behind the ROS
nodes.

The talk-only ROS stack (built in the dev container, or in a venv on a GPU box):

```bash
ros2 launch wojtek_agent_bringup voice_stack.launch.py \
    tts_engine:=silent asr_model:=tiny        # CPU smoke configuration
```

Model serving, host provisioning and teardown for a rented box are **not** in
this repository — see the root `CLAUDE.md`. `skills/wojtek-voice-agent-demo/`
carries the demo runbook.
