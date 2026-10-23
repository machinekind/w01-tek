# Experiment: RAI (RobotecAI) on the simulated Wojtek, v2

> **Status: EXPERIMENTAL. Not production; nothing here is deployed to the robot.**
> Nothing here is deployed by `ros/deploy.sh`, and no package here is a
> dependency of `wojtek_bringup`. It runs on the laptop; pointing it at the
> physical robot is a separate, human-authorized session (see
> [Physical robot](#physical-robot-human-authorized)). Interfaces are
> unstable by definition.

A typed instruction ("stand up, walk forward for three seconds, tell me what
you see") drives the MuJoCo-simulated Wojtek through
[RAI](https://github.com/RobotecAI/rai): a LangGraph ReAct agent with a small
set of robot tools, talking to the simulation's existing ROS 2 graph. Plan and
rationale: [docs/rai-on-wojtek.md](docs/rai-on-wojtek.md).

## How it is wired

```
 laptop                                   inference box (remote GPU, aarch64)
 ┌──────────────────────────────────┐     ┌──────────────────────────┐
 │ wojtek_robot  (ros/sim.sh)       │     │ ~/ollama/bin/ollama serve │
 │   MuJoCo plant, policy_node,     │     │ qwen3-vl:30b-a3b-instruct  │
 │   text_commander, camera         │     └────────────▲─────────────┘
 │        ▲ /wojtek/nav_command     │                  │ ssh -L 11435:11434
 │        │ /camera/.../image_raw   │                  │ (run.sh tunnel)
 │ wojtek_rai  (this experiment)    │                  │
 │   ReActAgent + tools ── ollama ──┼──────────────────┘
 │   streamlit :8501 / run.sh chat  │
 └──────────────────────────────────┘
   both containers: host network, ROS_DOMAIN_ID=42, CycloneDDS
```

- The agent never writes `/cmd_vel`. Its `walk` tool publishes
  `forward|left|right` to `/wojtek/nav_command` every 0.5 s for at most 5 s,
  then `stop`; the simulation's own `text_commander` turns that into
  `/cmd_vel` with a 2 s dead-man. `/wojtek/arm`, `enable`, `zero`, `reset`,
  `joint_targets` and `/cmd_vel` are on RAI's `forbidden` list
  (`wojtek_rai/limits.py`).
- Tools: `walk`, `turn` (closed-loop on odometry yaw, so only where
  odometry exists), `stop`, `stand_up`, `lie_down`, `get_robot_position`
  (TF `odom -> base_link`), `get_camera_image`, `wait_for_seconds`. The
  Nav2 and perception sections below add theirs.
- The robot's identity is a hand-written `wojtek_rai/embodiment.json`
  (RAI `EmbodimentInfo`); the `build-whoami` doc pipeline is not used yet.
- LLM vendor is `config.toml`: Ollama over the tunnel by default; `[vendor]`
  can be switched to `openai` (needs `OPENAI_API_KEY` in the root `.env`).

## Run it

```bash
# 0. once per machine
./experiments/wojtek_rai_v2/run.sh build           # wojtek_rai image (ros:jazzy-ros-base + rai-core)
./experiments/wojtek_rai_v2/run.sh inference       # Ollama on the inference box (needs WOJTEK_RAI_INFERENCE_SSH in .env)
./experiments/wojtek_rai_v2/run.sh pull qwen3-vl:30b-a3b-instruct   # NOT the bare qwen3-vl:30b: that is the thinking build

# 1. every session, three terminals
./ros/sim.sh --foxglove boot_pose:=folded          # MuJoCo sim; then zero -> stand_up -> arm from the console/Foxglove
./experiments/wojtek_rai_v2/run.sh tunnel          # localhost:11435 -> Ollama
./experiments/wojtek_rai_v2/run.sh agent           # http://localhost:8501

# or, scripted
./experiments/wojtek_rai_v2/run.sh topics          # smoke: RAI sees the sim graph
./experiments/wojtek_rai_v2/run.sh chat "report your position, walk forward for 3 seconds, report again"
```

Arming stays a human action, in the simulation too: the agent has no arm
tool. In the sim the standing pose sags a little under the soft PD servo, so
if `/wojtek/arm` refuses with a joint displacement just over the 0.15 rad
limit, loosen it for the session:
`ros2 param set /wojtek_real_io max_arm_jump_rad 0.3`.

## Navigation (Nav2) — `run.sh nav`

A third container, `wojtek_nav`, runs Nav2 + slam_toolbox fed by the depth
camera, and the agent gets RAI's Nav2 tools on top (`navigate_to_pose`,
`cancel_navigation`, `go_to_place`, `get_map_pose`, `get_map_image`). Plan
and rationale: [docs/rai-nav2-on-wojtek.md](docs/rai-nav2-on-wojtek.md).

```
depth (best-effort) ─► depth_relay ─► point_cloud_xyz_node ─► /nav/points
     ─► pointcloud_to_laserscan (height slice in base_link) ─► /scan ─► slam_toolbox ─► /map, map→odom
TF odom→base_link + /odom_vel (sim) ─► odom_relay ─► /odom
Nav2: planner (NavFn) · controller (Regulated Pure Pursuit) · behaviors · bt_navigator · velocity_smoother
      ─► /cmd_vel_nav ─► cmd_vel_watchdog (0.5 s dead-man) ─► /cmd_vel ─► policy_node
```

```bash
./experiments/wojtek_rai_v2/run.sh nav build      # once
./experiments/wojtek_rai_v2/run.sh nav launch     # with the sim running and the robot armed
./experiments/wojtek_rai_v2/run.sh chat "go to the hydrant and tell me what you see"
```

Named places for the sim scene live in `wojtek_rai/nav/config/places_sim.yaml`
(prop positions from `scene_sim.xml`, pulled back 0.6 m). Lessons that shaped
the parameters (`wojtek_rai/nav/config/nav2.yaml`):

- the camera publishes best-effort, the `depth_image_proc` point-cloud node
  subscribes reliable: hence `depth_relay`;
- the camera is pitched 15° down, so a row-based `depthimage_to_laserscan`
  read the floor as a wall 0.8 m ahead; the scan is a height slice of the
  point cloud in `base_link` instead (`wojtek_rai/nav/config/cloud_to_scan.yaml`).
  The slice is only as good as `base_link -> camera_link`: on the real robot
  measure the mount and fill `extrinsics.yaml` first;
- slam_toolbox is a lifecycle node in Jazzy: it needs its own lifecycle manager;
- the SLAM map starts tiny, so the global costmap is a 12 m rolling window
  (otherwise ComputePathToPose fails with 203, start outside map);
- the gait has a dead band (below ~0.2 rad/s the robot hardly turns), and
  RPP clamps its command to measured speed ± accel·dt: with a normal accel
  limit the command never leaves the dead band. Accel limits are therefore
  effectively off, `min_rotational_vel` 0.4.

## Object-grounded goals — `run.sh perception`

A fourth container, `wojtek_perception` (laptop GPU via the nvidia runtime),
runs RAI's GroundingDINO detection service on `/detection`
(`rai_interfaces/srv/RAIGroundingDino`). Detection only: SAM2-large next to
the simulator's renderer overflows a 4 GB GPU. Weights (~700 MB) download on
first start into the `wojtek_rai_weights` volume.

Agent tools (`wojtek_rai/perception_tools.py`):

- `find_objects(["ball", "fire hydrant"])` → per object: score, distance,
  bearing, position in `map`. Colour and depth have different intrinsics, so
  the colour box goes through the viewing ray (colour K) into the depth image
  (depth K); the depth is the median of the box centre; the point is placed
  in `base_link` and transformed to `map`. Measured error in the sim: 8 cm.
- `go_to_object("fire hydrant")` → the same detection, then a Nav2 goal 0.6 m
  in front of the object, facing it. Not visible → the agent is told to turn
  and retry (rule in the embodiment).

```bash
./experiments/wojtek_rai_v2/run.sh perception build   # once (~8 GB image: torch cu121)
./experiments/wojtek_rai_v2/run.sh perception up
./experiments/wojtek_rai_v2/run.sh chat "find the ball and walk to it"
```

## Physical robot (human-authorized)

Nothing here reaches the robot through `ros/deploy.sh`: the agent stays in
its container on the laptop and talks to the robot's ROS 2 graph over the
WiFi AP, exactly as it talks to the sim. Launching, arming and disarming the
robot remain human actions outside this experiment. A real-robot session
sets these switches in the root `.env`; `run.sh` forwards every
`WOJTEK_RAI_*` name into the `wojtek_rai` container (`agent`, `chat`,
`shell`, `topics`, `test`; `nav launch` gets none of them):

| switch | effect as implemented |
|---|---|
| `WOJTEK_RAI_READONLY=1` | first-contact mode: no `walk`, `turn`, `stop`, `stand_up`, `lie_down`, no Nav2 or perception tools; the agent can look, read its position and wait, and its prompt says so (`agent.py`, `tools.py`) |
| `WOJTEK_RAI_ODOMETRY=0` | the robot's `odom -> base_link` is a static identity, so `turn` and every Nav2 tool (`navigate_to_pose`, `go_to_place`, `go_to_object`, map tools) are dropped and `get_robot_position` describes itself as static; `find_objects` stays, bearing and distance need no map (`tools.py`) |
| `WOJTEK_RAI_COLOR_TRANSPORT=compressed` | every colour reader (camera tool, sidebar feed, `find_objects`) subscribes to `.../image_raw/compressed`: raw 640x480 rgb8 starved to 4 frames in 10 s over WiFi, the driver's JPEG arrives at full rate (`limits.py`) |
| `WOJTEK_RAI_PLACES=none` | no place registry, so `go_to_place` is not offered (`nav_tools.py`) |
| `WOJTEK_RAI_WORKSPACE_M=3` | goal box of ±3 m in `map` (one room) instead of the ±8 m sim arena; goals outside it are refused before Nav2 sees them (`limits.py`) |

Two things the real bringup does not provide:

- `text_commander`: the sim launch starts it, `wojtek_bringup` does not, so
  `walk`/`stop` have no consumer until `ros2 run wojtek_teleop
  text_commander` runs on the PC. The tools refuse to publish while nobody
  subscribes (an error, never a silent no-op), and over the AP that
  subscription shows up ~2 s after the publisher is created, hence
  `SUBSCRIBER_WAIT_S` = 5 s in `limits.py`.
- odometry: `run.sh nav launch target:=real` exists (no odom relay, no static
  `map -> odom`, slam_toolbox with scan matching on) but needs an external
  odometry node (plan phase N4) and gates Nav2 on a preflight that shuts the
  launch down without one. Nav2 therefore remains sim-validated only.

## Layout

| path | purpose |
|---|---|
| `run.sh` | `build \| up \| down \| shell \| topics \| agent \| chat \| bench \| tunnel \| inference \| pull \| test \| nav ... \| perception ...` |
| `docker/` | the `wojtek_rai` image and compose service (host net, same DDS settings as `wojtek_robot`) |
| `ros/rai_interfaces.repos` | SHA pin of `rai_interfaces`, colcon-built in the image |
| `config.toml` | RAI vendor/model names; no credentials |
| `wojtek_rai/limits.py` | the safety envelope: topic/service allowlists, walk bounds |
| `wojtek_rai/tools.py` | the tools above, on RAI's `BaseROS2Tool` |
| `wojtek_rai/agent.py`, `app.py`, `chat.py`, `stream.py` | agent assembly, streamlit UI, terminal one-shot, turn streaming |
| `wojtek_rai/nav_tools.py` | RAI Nav2 tools + `go_to_place` |
| `wojtek_rai/perception_tools.py`, `perception_services.py` | `find_objects` / `go_to_object` on RAI's `/detection`; detection-only service launcher |
| `wojtek_rai/camera_feed.py` | live colour/depth preview for the streamlit sidebar |
| `wojtek_rai/nav/` | nav container glue: launch, Nav2/SLAM params, depth relay, odom relay, cmd_vel watchdog |
| `wojtek_rai/embodiment.json` | who the robot is, for the system prompt |
| `tests/` | model-free tests: limits, tools against a fake connector, hygiene |

## Isolation rules

1. Nothing outside this directory imports anything inside it.
2. `ros/sim.sh`, `ros/dev.sh`, `ros/docker/compose.yaml` are untouched; this
   experiment runs its own compose project (`wojtek_rai`) next to them.
3. Its ROS packages (`rai_interfaces`) are built inside its own image, never
   under `ros/src/`, so `ros/deploy.sh` cannot ship them.
4. Secrets and private host identities enter only through the gitignored
   root `.env` (`OPENAI_API_KEY`, `WOJTEK_RAI_INFERENCE_SSH`, ...), forwarded
   by name; the `WOJTEK_RAI_*` session switches above travel the same way.

## Known gaps

- `qwen3-vl:30b-a3b-instruct` is the model in use; tool-call reliability on long
  instructions is unmeasured. The bare `qwen3-vl:30b` tag is the thinking build
  and Ollama 0.34 ignores `think=false` for it (8-70 s of hidden thinking per
  step, measured with `run.sh bench`).
- Navigation is sim-validated only: `map = odom` (static identity, scan
  matching off); the physical robot has no odometry source for Nav2 (plan
  phase N4).
- Detection thresholds (0.35/0.45) miss small or distant objects (the sim
  person at 5 m); no object memory: an object out of view must be searched
  for by turning.
- `get_map_image` shows only what the depth camera has swept; the sim scene
  has no walls, so the map is mostly "unknown" with a few props.
- Deck cockpit / `/from_human` HRI wiring, `build-whoami`, and the SCAN
  planner as a navigation tool are listed as later steps in the plan.
