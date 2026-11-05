# Visual navigation

`wojtek_nav` turns a language instruction or metric setpoint into velocity
commands. A VLM selects and verifies a visual target; the pixel resolver uses
depth and TF to locate it; the setpoint driver turns and walks toward it.
A rolling local costmap can veto forward motion into known obstacles.

The core route has recorded **stock-robot simulation** results. The direct
RPi camera/costmap/PC-brain route still needs hardware validation, including
camera mounting and CPU budget. New-leg navigation geometry is not validated.
Experimental remote deployments have separate capabilities and evidence; their
results do not validate this launch. Start with [compatibility](../../../docs/compatibility.md).

## Run the stock simulation

Prerequisites: a built ROS image, a compatible policy, and an Ollama server
with `qwen3-vl:30b-a3b-instruct`. See [ROS setup](../../README.md) and
[simulation](../../../docs/simulation.md). There is no committed minimum GPU/RAM
benchmark or pinned Ollama binary version; record those for your deployment.

On the model host, start Ollama if no service is already running:

```bash
ollama serve
```

In another shell on that host:

```bash
ollama pull qwen3-vl:30b-a3b-instruct
ollama list
```

For a remote model host, tunnel its loopback endpoint from the PC. Use an SSH
alias from your own configuration:

```bash
ssh -N -L 11435:127.0.0.1:11434 <model-host-alias>
```

Keep the tunnel running. In the repo-root `.env`, set `VLM_URL` to
`http://127.0.0.1:11435` on Linux, or `http://host.docker.internal:11435` on macOS.
These differ because the Linux container shares host networking, while Docker
Desktop uses a bridge. If the model runs locally, use its actual local port
instead. Check the tunnel from the PC:

```bash
curl --fail http://127.0.0.1:11435/api/tags
```

The requested tag should appear. Then, from the repository root:

```bash
./ros/sim.sh --foxglove model_xml:=scene_nav.xml \
  leg_odom:=true nav:=true vlm:=true telemetry:=true
```

Open `http://localhost:8080`. Complete zero → stand → arm using the
[operator runbook](../../../docs/operations.md), then enter “walk to the purple
pillar” in the brain panel. Leave other drive sources idle.

In a second `./ros/dev.sh` shell, inspect or submit a task:

```bash
ros2 topic echo /wojtek/vlm/status
# In another shell while watching status:
ros2 topic pub --once /wojtek/vlm/instruction std_msgs/msg/String \
  '{data: "walk to the purple pillar"}'
# Stop the task:
ros2 topic pub --once /wojtek/nav/cancel std_msgs/msg/Empty '{}'
```

A completed task reports JSON with `action: "finished"` and a result such as
`done`, `gave_up`, `cancelled`, `replaced` or `error`. It does not auto-arm.
An unarmed robot can accept a task but make no progress. Status includes the
instruction, step and timestamp, not a unique request ID; use a single task
owner. See the [integration guide](../../../docs/integration.md) for client
lifecycle and the bounded metric-goal example.

## Components and contracts

```text
raw depth → crop/decimate → point cloud → rolling costmap ─────┐
colour JPEG → VLM → pixel + image stamp → pixel resolver ──┐  │
                                                         ▼  ▼
metric goal (odom) ─────────────────────────────────────→ goto → /cmd_vel
```

`costmap.launch.py` starts the local perception chain, goto and pixel resolver.
`brain.launch.py` starts the VLM client. To supply your own metric goals, omit
the brain; to run against an existing camera and odometry, launch the costmap
standalone. `nav_costmap:=false` starts goto without the depth chain or resolver.

### Metric goals

| Interface | Contract |
|---|---|
| `/wojtek/nav/goal` | `geometry_msgs/PoseStamped`; position is the setpoint, orientation is ignored |
| Frame/time | `odom` is used directly; another frame is transformed at the message timestamp. A zero timestamp requests latest TF. |
| Refresh | Goal expires 3 s after receipt by default; resend while the task owns control. |
| Output | `geometry_msgs/Twist` at 20 Hz while moving; one zero on transition to stopped, then silence |
| `/wojtek/nav/status` | `std_msgs/String`: `idle`, `turning`, `driving`, `blocked`, `reached`; reliable/transient-local, depth 1 |
| `/wojtek/nav/cancel` | `std_msgs/Empty`; clears the setpoint and rejects incoming goals for 0.5 s |

Goals and cancel use reliable/volatile QoS with depth 10. A current goal in
`base_link` should carry the timestamp of the observation that produced it;
otherwise inference delay moves the meaning of “ahead.” Goal expiry uses the
receiver's clock, not the source timestamp.

Goto defaults to 0.3 m/s forward speed, 0.5 rad/s yaw and 0.15 m reach tolerance.
It turns first when the target bearing is large and never strafes. It does not
plan a route around an obstacle. A blocked forward path can still allow turning
toward the goal; this is not a full swept-body collision check.

### Pixel goals

`/wojtek/nav/pixel_goal` uses `geometry_msgs/PointStamped` with a project-specific
meaning:

- `header.stamp`: source colour-image timestamp.
- `point.x/y`: normalized image coordinates; values above 1 are interpreted as
  absolute pixels. Prefer normalized coordinates consistently.
- `point.z`: standoff in metres; 0 selects the default 0.7 m.

The resolver uses colour/depth CameraInfo and optical-frame TF, then takes the
median of a depth patch from the nearest timestamp (default maximum skew 0.1 s).
It transforms the target once into `odom` and resends the standoff goal, avoiding
repeated transforms through an expired camera-frame timestamp.

`/wojtek/nav/pixel_status` is a latched string: `idle`, `resolving`, `no_frame`,
`no_depth`, `no_tf`, `sent`, `reached`, `blocked`, `timeout`, `replaced` or
`cancelled`. `/wojtek/nav/pixel_target` is the resolved object point in `odom`.
Both use reliable/transient-local QoS. Images/CameraInfo use sensor-data QoS.
The resolver waits through brief blocked states, and ends on a terminal result
or the configured time limit. A new pixel replaces the previous goal.

### VLM task loop

`/wojtek/vlm/instruction` accepts a `std_msgs/String`; a new instruction replaces
the running task. The node reads colour JPEG, asks for `goal`, `turn`,
`not_visible` or `done`, and verifies a proposed target with a second question
before sending its pixel. Arrival is decided from execution/target distance,
not solely the model's `done` answer.

The loop explores by turning when the target is absent, can approach along the
viewing ray when depth is unavailable, and tries a detour after blockage. Default
budgets are 30 steps, 600 s, 8 blind approaches and 3 stalled moves. These bound
an attempt; they do not establish navigation reliability.

`/wojtek/vlm/status` carries latched JSON in `std_msgs/String`. Common fields are
`step`, `instruction`, `t` and `action`; action-specific fields include `result`,
`error`, `answer`, `latency_s` and target distance. `/wojtek/vlm/annotated` shows
the frame and selected point. The JSON is a project interface, not a ROS action
with request/result correlation.

`VLM_MODEL` or `brain.launch.py model:=` selects another Ollama tag. A replacement
must accept image input and the structured response schemas in `vlm_brain.py`;
an OpenAI-compatible URL alone does not prove compatibility. Run the same
present/absent-target and cancellation cases before treating another model as
supported. The current repository provides only this Ollama backend.

## Stop and hand over

An empty instruction, `stop`, the panel's STOP button or `/wojtek/nav/cancel`
ends the task. Goto clears its goal; on its next control tick with a valid pose
it emits the stop transition. The resolver stops resending goals, and the brain
halts its own turn command. Late goals within 0.5 s of cancel are ignored.
If pose is unavailable, the policy's command timeout remains the fallback.

The default direct `/cmd_vel` path has no ownership arbiter. Cancel navigation
before taking over with a pad/Deck, and stop other sources before starting a
new task. Disarm closes the policy-target gate but does not cancel the brain.
Neither cancel nor disarm removes motor torque.

Shared launches support `command_mux:=true` for explicit ownership; in that
mode manual takeover and disarm cancel navigation and inhibit autonomy.
Enable autonomy explicitly before a new task. For standalone launches, set
`cmd_vel_topic:=/wojtek/cmd_vel/nav` on `costmap.launch.py` and
`cmd_vel_topic:=/wojtek/cmd_vel/brain` on `brain.launch.py`.
See [command ownership](../../../docs/command-ownership.md) for the complete
setup and current validation limits.

## Costmap behavior and limits

The costmap is a 6×6 m rolling window in `odom`, with 5 cm cells and a retained
history of observed obstacles. Depth is decimated before projection. Separate
marking and clearing sources distinguish obstacles above 6 cm from floor rays;
IMU/odometry levels the cloud. Settings and rationale are in `config/costmap.yaml`.

The current controller treats **missing grids, unknown cells and positions
outside the grid as free**, and does not check grid freshness. Therefore,
`nav:=true` alone does not guarantee a working obstacle veto. Verify the map
and its update rate before a test; `nav_costmap:=false` provides no local
costmap protection. Sensor loss is not a certified obstacle-stop mechanism.

Other limits:

- No persistent SLAM map or general route planner is in the current launch.
- Negative obstacles can look unknown; step-height decisions are unfinished.
- Camera mounting is still a design value awaiting physical measurement.
- The fixed camera has blind regions; close targets can leave its image.
- Stock camera, scene and footprint assumptions do not validate new-leg use.

## Physical deployment

The documented robot-side camera path is `deck_camera:=true nav:=true` in the
effective RT service configuration. The installed service already enables the
Deck camera, but navigation is opt-in. Keep `deck_camera_depth:=true`; depth is
required by both resolver and costmap. Do not also set `perception:=true`.

On the PC, `ros2 run wojtek_bringup robot --web-console --vlm` adds the brain to
that configured robot session. Its `nav:=` tokens do not reconfigure the RT
service. The brain and web console read camera JPEG to limit wireless traffic;
`compressed:=false` is a raw-image fallback with substantially higher bandwidth.
Use the [camera guide](../wojtek_perception_bringup/README.md) and
[operator runbook](../../../docs/operations.md) to prepare the hardware route.

## Evaluation and recorded evidence

For a repeatable simulated evaluation, record commit, profile, policy pin,
scene/spawn, model digest/tag, Ollama version, host/GPU and launch arguments.
Restart the scene between cases and collect status, `/cmd_vel`, TF and timing:

| Case | Check |
|---|---|
| Visible target | Correct object selected; final distance and elapsed time recorded |
| Target initially behind | Exploration finds it without inventing a visible target |
| Look-alike objects | Verification rejects wrong colour/size/identity |
| Absent target | Attempt ends within budgets; fabricated goals counted |
| Obstacle on direct line | Costmap is present; blocked behavior and any detour recorded |
| Cancel while moving | Motion command stops; late goal cannot immediately restart it |
| Model/camera failure | Task reports error or timeout; no stale task resumes on reconnect |
| Unarmed robot | Task reports lack of progress instead of reporting success |

Historical stock-sim results from September 2026 include 25 s and 55 s target
approaches, and a 66 s approach from beyond the depth window ending 0.77 m from
the pillar. These are individual observations, not a success-rate benchmark.
A nine-frame pointing test reported median metric error around 0.15–0.18 m,
with false “chair” detections in 4 of 9 absent-object queries.

### Build a labelled pointing fixture

Use a stationary stock simulation with the camera running. Inside the ROS
container, capture one matched pair at a time:

```bash
python3 /ros2_ws/src/wojtek_nav/scripts/capture_frames.py --out /tmp/wojtek-point-frames
```

Each call appends a JPEG, depth array in millimetres and intrinsics/timestamps to
`index.json`. It refuses timestamp skew above 0.1 s by default and writes no
object labels. Use one capture process at a time. Move to another test view,
stop motion, and capture again. Export the directory from the PC host before
recreating the container:

```bash
docker cp wojtek_robot:/tmp/wojtek-point-frames "$HOME/wojtek-point-frames"
```

Review each JPEG and edit its entry's `truth`. For example, only when the pillar
is visible at the image centre and a chair is absent:

```json
"annotation_status": "reviewed",
"truth": {
  "pillar": {"visible": true, "u_norm": 0.5, "v_norm": 0.5},
  "chair": {"visible": false}
}
```

Normalize a marked pixel as `x / colour_width`, `y / colour_height`. The current
prompt vocabulary is `crate`, `pillar`, `low_box`, `chair`, `person`; add a name
to the script's prompt table before using another object. Label only known
visible/absent cases. Capture rejects no objects itself, and the scorer refuses
unreviewed/empty annotations instead of guessing an absent target.

From an environment with NumPy and access to the model endpoint:

```bash
python3 ros/src/wojtek_nav/scripts/point_bench.py \
  --frames "$HOME/wojtek-point-frames" --url http://127.0.0.1:11435/v1 \
  --model qwen3-vl:30b-a3b-instruct --json-schema --tag baseline
```

The report records pointing error, explicit absent-object errors and latency.
Colour dimensions come from the captured frame. Metric scoring uses the stored
intrinsics and a shared viewing-ray approximation; it does not apply a measured
physical colour-to-depth extrinsic. Use this fixture for the stock simulation,
and validate physical-camera projection separately before interpreting its
metric errors. A human point is a reference annotation, not exact geometric
truth. Archive labels, images, model digest and versions with the report. The
original nine-frame fixture is not committed, so this creates a new baseline.

### Automated checks

From the repository root with pytest and NumPy, run the model-free checks:

```bash
PYTHONPATH=ros/src/wojtek_nav:$PYTHONPATH python3 -m pytest \
  ros/src/wojtek_nav/test/test_goto.py \
  ros/src/wojtek_nav/test/test_pixel_goal.py \
  ros/src/wojtek_nav/test/test_vlm_brain.py \
  ros/src/wojtek_nav/test/test_point_capture.py -q
```

For launch-composition tests, use the ROS container with its workspace sourced:

```bash
cd /ros2_ws/src/wojtek_nav
PYTHONPATH=$PWD:$PYTHONPATH python3 -m pytest test/ -q
```

These checks do not exercise a live model, ROS transport, physical sensors or a
powered robot. Capture synchronization still needs a live ROS camera test.
