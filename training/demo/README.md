# Training simulation demos

For the deployed ROS graph, use the [ROS simulation guide](../../docs/simulation.md).
These demos run training or exported-policy code directly; they do not exercise
ROS bringup, arm gates or robot networking.

| Path | Status | What it exercises |
|---|---|---|
| `run.sh room` | Current stock-robot demo | Plain MuJoCo, exported NumPy policy, rendered scan and optional VLM navigation. |
| `run.sh nav-episode` | Evaluation tool | Headless room simulation with contact dynamics, a VLM goal and recorded results. |
| `run.sh nav-eval` | Evaluation tool | Kinematic navigation and perception decisions; not gait/contact dynamics. |
| `run.sh app` | Legacy prototype | MJX click-to-walk with a default stock environment; does not reconstruct arbitrary saved run settings. |
| [Isaac demo](isaac/README.md) | Historical prototype | RTX rendering and PhysX import experiments; requires repair before a supported launch. |

## Run the room demo

From the repository root, install the optional dependencies and build the scene:

```bash
uv sync --project training --locked --extra dev --extra demo --extra room
./training/run.sh room-assets
./training/run.sh build --robot wojtek
./training/run.sh build-room
./training/run.sh room --policy runs/wojtek_candidate/deploy
```

Use an exported stock policy with a passing walking evaluation, or pass a
Hugging Face policy reference instead of the local path. The local path above
is relative to `training/`. See [policy artifacts](../docs/policy-lifecycle.md).
A matching actuator name list alone does not establish new-leg compatibility:
this scene, camera and planner footprint are sized for the stock robot.

`room-assets` downloads the Habitat test-scene archive, extracts the Van Gogh
room, creates visual meshes/textures and convex collision pieces, and writes
`training/assets/room/manifest.json`. Processing can take time; keep the cached
assets for later runs. The source attribution is in
[the asset license file](../assets/room/LICENSE.md). Downloads and processed
assets are ignored by Git. For a local scan, use
`room-assets --glb /absolute/path/to/scan.glb --up z --name my_room`; check its
license and select `--up y` if that scan uses Y-up coordinates.

`build-room` writes `scene_room.xml` beside the robot model and checks that it
loads. It places a flat walking plane at z=0; the scan's floor is flattened
for this demo. Inspect scale, orientation, spawn clearance and collisions
before using a new scan as an evaluation scene. For named scenes, pass the same
name to `build-room --name my_room` and `room --scene-name my_room`.

Open `http://127.0.0.1:8010`. The page shows chase and onboard camera views,
a minimap and command fields. Click a clear nearby point, observe movement and
arrival, then send `stop` in the command field. Ctrl-C stops the server;
restart it for a fresh session. A loaded page without a working camera/control
connection is not a successful rollout.

Linux rendering uses EGL; macOS uses CGL. The room physics loop uses CPU
MuJoCo, but rendering still needs a working graphics context. For a remote
host, keep the default loopback bind and forward its port through SSH.

## Add the VLM

Configure the Ollama server described in the [VLM guide](../../ros/src/wojtek_nav/README.md)
and set `VLM_URL`/`VLM_MODEL` in the repo-root `.env`, or pass:

```bash
./training/run.sh room --policy runs/wojtek_candidate/deploy \
  --vlm-url http://127.0.0.1:11434 --vlm-model qwen3-vl:30b-a3b-instruct
```

Use the page's goal field for a target visible in the room. The room demo uses
its SCAN local planner and its own mid-level command executor; the ROS VLM
path uses the ROS pixel resolver/goto/costmap stack. Their results validate
different integrations. Keep the planner enabled for normal room tests;
`--no-local-planner` is an explicit A/B baseline.

For a recorded episode with simulated dynamics:

```bash
./training/run.sh nav-episode --goal 'go to the bed' \
  --policy runs/wojtek_candidate/deploy --out runs/room_episode
```

Use `nav-episode --help` for step budgets and camera choices. `nav-eval`
requires the separate `eval` extra and evaluates kinematic navigation. Report
which evaluator, scene, camera, policy and model/server versions produced a
result; a kinematic success does not establish walking or collision fidelity.

## Legacy click-to-walk app

`run.sh app` defaults to `policies/fbb_v3`, an artifact not shipped in the
repository. It creates the default stock environment instead of restoring
all settings from `run.json`, so an arbitrary current checkpoint can have an
incompatible observation layout. Its wrapper also defaults EGL on macOS.
Use the room demo for exported policies or ROS simulation for integration
work. Retain this app as a prototype until its configuration loading and host
setup are updated and tested.

## Troubleshooting

| Symptom | Check |
|---|---|
| Missing manifest or scene XML | Run `room-assets`, `build` and `build-room` in that order. Keep the same name for a named scene. |
| Missing optional Python module | Install the corresponding extras together; later `uv sync` calls must retain extras you need. |
| Policy/model mismatch or bad standing pose | Verify stock geometry, exported contract, gains and policy evaluation before debugging navigation. |
| Black image or renderer failure | Check the host GL backend and its ability to create an offscreen context. |
| Model server cannot be reached | Check its loopback bind/tunnel and URL from the machine running the demo. |
| Scan appears on its side or at the wrong scale | Check source up-axis and physical dimensions; do not tune the policy around a bad scene transform. |
