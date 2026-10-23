# RAI on Wojtek — plan (attempt 2)

Status: plan, 2026-09-12. Branch `worktree-rai-on-wojtek`.
Goal: a typed instruction ("walk forward two seconds, then tell me what you
see") moves the simulated Wojtek through RobotecAI's RAI, using the ROS 2
graph `ros/sim.sh` already exposes. Sim first, text only. Physical robot is
a later, human-authorized step and is not planned here.

Attempt 1 (remote branch `origin/gsd/phase-01-isolated-rai-environment`)
died of process weight: five planned phases, one executed, no robot moved.
This plan inverts that: a walking demo by the end of step 2, hardening
after. Reused from attempt 1 only where it saves a day (listed at the end).

## 1. Facts the plan rests on

### RAI (checked against upstream `main`, 2026-09-08; PyPI: rai-core 2.12.1, rai-whoami 0.0.5)

- Python `>=3.10,<3.13`, ROS 2 Humble or Jazzy. `pip install rai-core`
  (PyPI lags git: 2.12.1 vs 2.12.4 on `main`; pin PyPI, bump on purpose).
  `rai-whoami` is a separate package. `rai_interfaces` is a ROS package
  (`RobotecAI/rai_interfaces`); the docs mention an apt build but
  packages.ros.org does not carry it today, so colcon-build it at a pinned
  SHA inside the image.
- Vendor config is `config.toml` in the process cwd; `[vendor]` picks
  `openai | ollama | aws | google`; secrets come from env
  (`OPENAI_API_KEY`). No secret field exists in the file.
- Agent: `rai.agents.langchain.ReActAgent(target_connectors, llm, tools,
  system_prompt)`. Two front ends:
  - `rai.frontend.streamlit.run_streamlit_app(agent.agent, title, greeting)`
    — browser chat, "direct mode", `target_connectors={}`. No
    `rai_interfaces` needed. This is the demo path.
  - `ROS2HRIConnector` on `/from_human` -> `/to_human`
    (`rai_interfaces/msg/HRIMessage`). Later, if the deck panel wants it.
- Tools (`rai.tools.ros2`): every tool takes `connector=ROS2Connector()`
  and has `readable` / `writable` / `forbidden` name lists. Ready-made:
  `GetROS2ImageConfiguredTool(topic=...)`,
  `GetROS2TransformConfiguredTool(source_frame, target_frame)`,
  `PublishROS2MessageTool`, `CallROS2ServiceTool`,
  `GetROS2TopicsNamesAndTypesTool`, `rai.tools.time.WaitForSecondsTool`.
  `rai/tools/ros2/generic/topics.py` imports `cv_bridge` at module scope, so
  the runtime needs `ros-jazzy-cv-bridge`.
- Embodiment: `EmbodimentInfo.from_file("x.json")` with keys `rules`,
  `capabilities`, `behaviors`, `description`, `images`; `.to_langchain()`
  gives the system prompt. Hand-written JSON is enough (that is what the
  upstream ROSbot demo ships). The `build-whoami` doc pipeline and vector DB
  are optional and skipped here.
- Upstream Docker image is `osrf/ros:jazzy-desktop-full` + `uv sync` +
  colcon; defaults to FastDDS. We will not use it: Wojtek runs CycloneDDS,
  and we need only the core package.

### Wojtek sim (this repo, `main` at c1b315d)

- `ros/sim.sh` -> container `wojtek_robot` (`ros:jazzy-ros-core` base,
  `network_mode: host`, `ROS_DOMAIN_ID=42`, `rmw_cyclonedds_cpp`,
  `CYCLONEDDS_URI=file:///config/cyclonedds.xml`). GPU overlay
  `compose.gpu.yaml` when the nvidia runtime exists. Default `hw:=mujoco`,
  `camera:=true`.
- Control input is one topic: `/cmd_vel` (`geometry_msgs/Twist`), clamped in
  `policy_node._on_cmd` to the policy's trained command box
  (`policy_meta.json`, typically vx ±0.3–0.5 m/s, wz ±0.5–1 rad/s). The
  policy holds the last command; there is no dead-man in `policy_node`.
- `wojtek_teleop/text_commander`: `/wojtek/nav_command` (`std_msgs/String`:
  `forward | left | right | stop`) -> `/cmd_vel` at 20 Hz, `v_forward=0.3`,
  `w_turn=0.5`, dead-man `command_timeout=2.0 s`, then one zero Twist. It is
  a separate node; step 1 verifies whether `sim.sh` starts it.
- Sim-only ground truth: TF `odom -> base_link`, `/odom_vel`, `/sim/rtf`.
  Camera: `/camera/camera/color/image_raw` (rgb8),
  `/camera/camera/depth/image_rect_raw` (16UC1).
- Services: `/wojtek/stand_up`, `/wojtek/lie_down`, `/wojtek/zero`,
  `/wojtek/reset` (Trigger); `/wojtek/arm`, `/wojtek/enable` (SetBool);
  `/sim/reset`. Session bring-up (`robot.py`) already does zero -> stand_up
  -> arm, so the agent only ever needs stand_up / lie_down.
- Rules: everything lives under `experiments/<name>/`, nothing outside may
  import it, `ros/src/` stays untouched (that is what `deploy.sh` rsyncs),
  no credentials or private host identity anywhere in the tree,
  Apache-2.0.
- Remote aarch64 GPU dev box: CycloneDDS multicast loopback is broken
  there; every ROS process needs `ROS_LOCALHOST_ONLY=1`. Remote runs are
  human-authorized.

## 2. Architecture (what gets built)

```
experiments/wojtek_rai_v2/
  README.md                 status + how to run (experiment contract)
  run.sh                    build | up | agent | topics | test | shell | down
  docker/Dockerfile         ros:jazzy-ros-base + cv_bridge + rai-core/whoami + rai_interfaces
  docker/compose.yaml       service wojtek_rai: host net, ROS_DOMAIN_ID=42, cyclonedds, mounts
  config.toml               RAI vendor config (names only; openai by default)
  wojtek_rai/
    __init__.py
    embodiment.json         hand-written Wojtek EmbodimentInfo
    limits.py               MOVE_MAX_SECONDS, MOVE_DIRECTIONS, topic/service allowlists (pure)
    tools.py                WalkTool, StopTool, StandUpTool, LieDownTool, GetPoseTool, GetCameraTool
    agent.py                build_agent(connector) -> ReActAgent
    app.py                  streamlit entry (run_streamlit_app)
    topics.py               `python -m wojtek_rai.topics`: print live graph (smoke)
  tests/                    model-free: limits, tool arg validation, embodiment schema, no-secrets
```

Runtime shape: a second container, `wojtek_rai`, next to `wojtek_robot`.
Both on the host network, same domain, same CycloneDDS profile, so DDS
discovery is identical to two nodes on one host. Nothing is installed into
`wojtek_robot` (attempt 1 mutated it at runtime: apt inside, root-owned
`.venv` under the repo). Image build is reproducible from the Dockerfile
and works on x86_64 and aarch64 (`ros:jazzy-ros-base` is multi-arch; the
RAI deps are pure Python + `opencv-python-headless`, which ships aarch64
wheels; `numpy<2` matches Jazzy's system numpy 1.26).

Motion path, step 2: the agent's `WalkTool(direction, seconds)` publishes
`std_msgs/String` to `/wojtek/nav_command` every 0.5 s for `seconds`
(capped at `MOVE_MAX_SECONDS = 5`), then `stop`. `text_commander` turns
that into `/cmd_vel` and its 2 s dead-man covers an agent crash mid-walk.
The LLM never writes `/cmd_vel` directly and never sees `/wojtek/arm`,
`/wojtek/enable`, `/wojtek/zero`, `/wojtek/reset`, `/wojtek/joint_targets`
(all in `forbidden`). Only `stand_up` and `lie_down` are exposed as
services. Camera and odom TF are read-only tools.

If step 1 shows `sim.sh` does not launch `text_commander`, `run.sh up`
starts it in the `wojtek_robot` container (`ros2 run wojtek_teleop
text_commander`) rather than porting it; the experiment must not
duplicate `ros/` code.

## 3. Steps

Each step ends in something observable. Do not start the next before the
acceptance line passes.

### Step 0 — preconditions (30 min, laptop)

- `./ros/sim.sh` runs and Foxglove shows a standing robot. Note whether
  `ros2 node list` (inside `wojtek_robot`) contains `text_commander` and
  whether `/clock` is published (decides `use_sim_time` for the connector).
- Root `.env` has `OPENAI_API_KEY`; `.env.example` gains the placeholder
  name (name only). Nothing else in the tree ever carries it.

Acceptance: `ros2 topic echo --once /camera/camera/color/image_raw` returns
a frame; `ros2 topic pub -1 /wojtek/nav_command std_msgs/String "data:
forward"` visibly moves the robot for 2 s.

### Step 1 — RAI sees Wojtek (half a day)

- `docker/Dockerfile`: `FROM ros:jazzy-ros-base`; apt `ros-jazzy-cv-bridge
  ros-jazzy-rmw-cyclonedds-cpp python3-pip ros-dev-tools`; `vcs import`
  of `rai_interfaces` from `ros/rai_interfaces.repos` (SHA pin) + `rosdep
  install` + `colcon build` into `/rai_ws`; `pip install
  --break-system-packages rai-core==2.12.1 rai-whoami==0.0.5`;
  `WORKDIR /exp`.
- `docker/compose.yaml`: service `wojtek_rai`, `network_mode: host`,
  `ipc: host`, env `ROS_DOMAIN_ID=42`, `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`,
  `CYCLONEDDS_URI=file:///config/cyclonedds.xml`,
  `ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY:-0}`; mounts
  `../../../ros/docker/config/cyclonedds.xml:/config/cyclonedds.xml:ro` and
  `..:/exp` (the experiment dir); `env_file` not used — `run.sh` forwards
  `OPENAI_API_KEY` by name (`docker compose run -e OPENAI_API_KEY`), the
  attempt 1 pattern, so the value never lands on a command line or in a
  compose file. Port 8501 (streamlit) is on the host network already.
- `run.sh`: `build` (compose build), `up` (`docker compose up -d`),
  `topics` (`python -m wojtek_rai.topics`: `ROS2Connector` +
  `GetROS2TopicsNamesAndTypesTool`, prints graph), `shell`, `down`.

Acceptance: with `ros/sim.sh` running, `./experiments/wojtek_rai_v2/run.sh
topics` lists `/cmd_vel`, `/wojtek/nav_command`,
`/camera/camera/color/image_raw`, `/wojtek/stand_up`. No LLM call yet.

### Step 2 — chat that walks (one day)

- `embodiment.json`: description (quadruped, sim, camera forward, moves by
  discrete walk commands, max 5 s per command, must stop before answering),
  capabilities, rules ("never call a tool you were not asked for", "ask
  before walking more than 3 m total"), behaviors.
- `limits.py`: constants + `validate_walk(direction, seconds)`; pure, tested.
- `tools.py`: `WalkTool`, `StopTool` (nav_command), `StandUpTool`,
  `LieDownTool` (`CallROS2ServiceTool`-style via connector, `writable`
  restricted), `GetPoseTool` (`GetROS2TransformConfiguredTool`,
  `odom -> base_link`), `GetCameraTool`
  (`GetROS2ImageConfiguredTool` on the color topic), `WaitForSecondsTool`.
  All share one `ROS2Connector(executor_type="multi_threaded",
  use_sim_time=<step 0 answer>)`.
- `agent.py`: `ReActAgent(target_connectors={},
  llm=get_llm_model("complex_model", streaming=True),
  system_prompt=EmbodimentInfo.from_file(...).to_langchain(), tools=...)`.
- `app.py`: streamlit; `run.sh agent` = `streamlit run wojtek_rai/app.py
  --server.headless=true` inside the container; open `http://localhost:8501`.

Acceptance (recorded as a short screen capture or Foxglove odom plot):
"stand up, walk forward for two seconds, turn left for one second, stop,
report your position" — the robot does it, `GetPoseTool` shows the
displacement, no tool outside the allowlist was called (agent log).

### Step 3 — chat that looks (half a day)

- Prompt tuning so the agent takes a camera frame before answering
  "what do you see" and before choosing a direction.
- Scenario: place a coloured box in the MuJoCo scene (existing sim scene
  options; if none, note it and use whatever the scene has), "walk towards
  the red box and stop in front of it" with the 5 s cap forcing a
  look–walk–look loop.

Acceptance: three consecutive runs reach within ~0.5 m of the target
(odom), none walk into it. Save the transcripts under
`experiments/wojtek_rai_v2/docs/runs/` (no secrets in them).

### Step 4 — harden and record (half a day)

- `tests/`: `validate_walk` bounds; tool `writable`/`forbidden` lists
  contain exactly the intended names; `embodiment.json` loads through
  `EmbodimentInfo`; no-secrets scan over the experiment (regex on
  `sk-`, `AKIA`, tokens, private hostnames); isolation grep (nothing in
  `ros/`, `training/`, root scripts references `wojtek_rai_v2`). All
  model-free, run by `run.sh test` (pytest inside the container, or
  `EXP_PY=<host python>` for CI-style runs).
- README: status EXPERIMENTAL, how to run, what it can/cannot do, known
  issues. Add the experiment to the `experiments/` note in `CLAUDE.md`
  validation list (one line).
- Second-machine check on the aarch64 dev box with
  `ROS_LOCALHOST_ONLY=1` (human-authorized; identity stays out of the
  repo). Record findings in `docs/VERIFICATION.md` of the experiment.

Acceptance: `run.sh test` green on laptop; steps 2–3 reproduced on the
dev box; PR opened from `worktree-rai-on-wojtek`.

### Later, not in this plan

- `/from_human`–`/to_human` HRI wiring so the Deck cockpit can chat.
- `build-whoami` over `docs/` + `wojtek_description` for a doc-grounded
  identity and the documentation vector DB (needs an embeddings vendor).
- Hand the agent the SCAN planner (`training/wojtek_rl/scan/`) as a
  navigation tool instead of raw walk commands.
- Jetson deployment and the physical robot: separate authorization,
  separate plan. Requires a real dead-man in `policy_node` first.

## 4. Risks and how each is handled

| risk | handling |
|---|---|
| `rai_interfaces` rosdep pulls `vision_msgs`, `nav2_msgs`, `portaudio` | accepted image size cost; `rosdep install -r` so a missing key does not fail the build |
| LLM decision latency (2–10 s) vs 2 s dead-man | the tool pulses `nav_command` itself; the LLM only chooses direction and duration |
| Agent spams walk commands | `MOVE_MAX_SECONDS`, one walk per tool call, rules in embodiment; `forbidden` list closes every other actuator path |
| CycloneDDS discovery between two containers | both host-network, same domain and profile; on the dev box `ROS_LOCALHOST_ONLY=1` for both |
| `cv_bridge` / `rclpy` ABI vs pip numpy | `numpy<2` in rai-core matches Jazzy; install with `--break-system-packages` into the system interpreter, no venv |
| Secrets | env by name only; `.env` gitignored; no-secrets test; never `set -x` in `run.sh` |
| Upstream RAI API drift (LangChain 1.x churn) | pin `rai-core==2.12.1`, re-pin deliberately |

## 5. Reused from attempt 1 (remote branch), by reference only

- `run.sh` credential forwarding (`docker ... -e NAME` without value) and
  the `EXP_PY` test escape hatch.
- `tests/test_no_secrets_in_config.py`, `test_isolation_boundary.py`
  ideas (rewrite small; do not cherry-pick the GSD-era commits).
- Findings: cv_bridge is required at import; rai_interfaces pulls
  `vision_msgs`, `nav2_msgs` via rosdep (why the image has its own
  rosdep step); root-owned artefacts appear if a venv is created under a
  bind mount from inside the container (why this plan installs into the
  image instead).

## 6. Commands (once the steps exist)

```bash
./ros/sim.sh                                   # terminal 1: MuJoCo sim + Foxglove
./experiments/wojtek_rai_v2/run.sh build       # once per machine
./experiments/wojtek_rai_v2/run.sh topics      # step 1 acceptance
./experiments/wojtek_rai_v2/run.sh agent       # step 2: http://localhost:8501
./experiments/wojtek_rai_v2/run.sh test        # model-free suite
```
