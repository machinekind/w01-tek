# Nav2 on Wojtek, driven by RAI — plan

Status: N0–N3 done in sim. N0–N2 on 2026-09-12 (first Nav2 goal reached;
agent `go_to_place(hydrant)` → arrival → camera report); N3 detection-only:
GroundingDINO on `/detection` in a separate `wojtek_perception` container,
SAM2 dropped because it does not fit beside the renderer on a 4 GB GPU. N4
(physical robot) open. Lessons learned are in
[../README.md](../README.md) (Navigation section). Builds on
[rai-on-wojtek.md](rai-on-wojtek.md)
(the working sim demo: RAI ReAct agent → `walk` tool → `text_commander`).
Goal: the agent gets what RAI's ROSbot demo has — `navigate_to_pose`,
`get_current_pose` in a `map` frame, an occupancy grid it can look at — so
"go to the hydrant" becomes one Nav2 goal instead of blind 5-second walks.
Sim first; the physical robot is a separate, human-authorized phase.

## 1. What RAI needs from the robot (checked in rai-core 2.12 sources)

| RAI tool | needs on the ROS side |
|---|---|
| `Nav2Toolkit` / `NavigateToPoseBlockingTool` | action `navigate_to_pose` (`nav2_msgs/action/NavigateToPose`), goals in `frame_id` (default `map`); optional `workspace_bounds_min/max` reject out-of-area goals |
| `GetCurrentPoseTool(frame_id="map", robot_frame_id="base_link")` | TF `map → base_link` |
| `GetOccupancyGridTool` | `nav_msgs/OccupancyGrid` on `/map` + TF, rendered as an image for the VLM |
| `GetROS2ImageConfiguredTool` | camera topic (already used) |
| `rai_perception.GetDistanceToObjectsTool` / `GetObjectPositionsTool` | a `/detection` service (GroundingDINO, GPU) + depth image + camera_info; gives object pose → a Nav2 goal |

Nothing of that exists in `ros/` today: no Nav2, no SLAM, no `map` frame,
no `nav_msgs/Odometry`. All Jazzy packages are in apt (verified):
`navigation2 1.3.13`, `nav2-bringup`, `slam-toolbox 2.8.5`, `rtabmap-ros
0.23.7`, `depthimage-to-laserscan`, `robot-localization`, `depth-image-proc`,
`spatio-temporal-voxel-layer`, `nav2-mppi-controller`.

## 2. What Wojtek has (this repo)

- Frames (`wojtek_description/urdf/body.urdf.xacro`): `base_link`,
  `imu_link` (z 0.10), `camera_link` at xyz (0.32, 0, 0.07) pitched 15°
  down, `camera_depth_optical_frame`, `camera_color_optical_frame`.
- Body: 0.59 × 0.225 × 0.10 m box, legs ±0.13 m, standing height
  0.084–0.182 m (policy height command). A conservative circular footprint
  is r ≈ 0.35 m; SCAN-Planner's tuned twin-cylinder is d_off 0.25 m, r 0.20 m.
- Odometry: **sim** publishes ground-truth TF `odom → base_link` and
  `/odom_vel` (`wojtek_sim_ground_truth`); no `nav_msgs/Odometry`.
  **Real** has a static `odom → base_link` only. That is the one hard gap
  for the physical robot.
- Depth: D435 (real, `wojtek_perception_bringup`) and the virtual camera
  (sim) both publish `/camera/camera/depth/image_rect_raw` (424×240 real /
  sim render, 15 Hz) + `camera_info`; colour 5–6 Hz. `cloud_reduce` makes a
  coarse point cloud for the deck panel. Horizontal FOV ≈ 87°.
- Velocity: policy box vx −0.8..1.2, vy ±0.5, wz ±1.0 (rad/s), height in
  `Twist.linear.z` (0 = keep). `policy_node` clamps and **holds the last
  command; no dead-man**. `text_commander` adds a 2 s dead-man for text
  commands only.
- Compute: RPi runs control (400 Hz) + policy (50 Hz) + camera driver;
  everything else runs on the PC container. The PC image has no nav
  packages.

## 3. Target architecture (sim and real share it)

```
                          RAI agent (wojtek_rai container)
   Nav2Toolkit ── navigate_to_pose ──►  bt_navigator ─► planner (Smac2D/NavFn)
   GetCurrentPose ◄── TF map→base_link   │              controller (RPP, then MPPI)
   GetOccupancyGrid ◄── /map             │              behaviors (spin, backup, wait)
   camera / detection tools              ▼
                                   /cmd_vel_nav ─► cmd_vel_watchdog (0.5 s) ─► /cmd_vel ─► policy_node (RPi / sim)
                                                                                  ▲
   text_commander (/wojtek/nav_command) ──── still there for the `walk` tool ─────┘

   sensing:  depth ─► depth_relay ─► point cloud ─► height slice (base_link) ─► /scan ─► slam_toolbox ─► /map, TF map→odom
                   └► (later) STVL / voxel layer for the local costmap
   odometry: sim  = wojtek_sim_ground_truth TF + small odom_relay (TF → nav_msgs/Odometry)
             real = rtabmap rgbd_odometry (D435, on the PC)  → TF odom→base_link + /odom
                    later: leg kinematics + IMU through robot_localization EKF
```

All navigation nodes live in a new compose service `wojtek_nav` inside
`experiments/wojtek_rai_v2/` (own image: `ros:jazzy-ros-base` + the apt
packages above), host network, domain 42, CycloneDDS — exactly like
`wojtek_rai`. `ros/` stays untouched until this graduates.

Design decisions:

- **LaserScan from depth, not a 3D voxel layer, first.** slam_toolbox and
  the costmap obstacle layer are proven with `/scan`; the D435's 87° FOV
  is narrow, so behaviors get `spin` recovery and the local costmap keeps
  obstacles for a while (`observation_persistence`). *Done as* depth →
  point cloud → height slice in `base_link` → scan: the row-based
  `depthimage_to_laserscan` reads the floor as a wall with the 15° tilt.
- **SLAM scan matching off in sim.** The narrow, sparse scan produced a
  bogus `map→odom` within a minute; the sim's odometry is ground truth, so
  the map is accumulated on it. The real robot needs matching or a better
  odometry source (N4).
- **Controller: Regulated Pure Pursuit first**, holonomic off, `vx` 0.15–
  0.5 m/s, `wz` ≤ 0.8 rad/s, `min_vel_x 0.15` (below that the gait does
  not step cleanly). MPPI later if RPP hunts on a legged base. *Done as*
  RPP with `desired_linear_vel` 0.3 (the real-robot cap; the same
  `nav2.yaml` serves sim and hardware), no `min_vel_x` (velocity_smoother
  `min_velocity` x 0.0, `min_approach_linear_velocity` 0.15),
  `rotate_to_heading_angular_vel` 0.6, `max_angular_accel` 10.0 /
  `rotational_acc_lim` 10.0 (effectively off: RPP clamps its command to
  measured speed ± accel·dt, and the gait's ~0.2 rad/s dead band keeps the
  measured `wz` at zero, so an accel-limited command never left the dead
  band) and `min_rotational_vel` 0.4; see the README, Navigation.
- **A `cmd_vel_watchdog` node** sits between Nav2 and the robot: republishes
  `/cmd_vel_nav` as `/cmd_vel`, zeroes it 0.5 s after the last message.
  `policy_node` has no dead-man of its own; this is the safety piece that
  makes Nav2 acceptable on the real robot.
- **Height stays untouched**: Nav2 sends `linear.z = 0`, which the policy
  reads as "keep the contract height".
- **Goals come from three sources**, in this order of arrival: explicit
  `(x, y, yaw)` from the user; named places from a small YAML registry
  (sim props: ball, hydrant, stop sign...; later real rooms); object
  detections (rai_perception GroundingDINO + depth → pose in `map`).

## 4. Phases

### N0 — nav container and the TF/odom/scan basics in sim (½ day)

- `docker/nav.Dockerfile` + `wojtek_nav` compose service; `run.sh nav
  {up|down|shell}`.
- `odom_relay` (tiny rclpy node): TF `odom→base_link` + `/odom_vel` →
  `nav_msgs/Odometry` on `/odom`. `depthimage_to_laserscan` on the depth
  topic → `/scan` (frame `camera_depth_optical_frame`; check the
  `output_frame` and range limits 0.3–4 m).
- Acceptance: `ros2 run tf2_tools view_frames` shows
  `odom→base_link→camera_*`; `/scan` at ~15 Hz; the scan in Foxglove
  outlines the hydrant and the ball.

### N1 — SLAM + Nav2 in sim, goals by hand (1–2 days)

- slam_toolbox `online_async` (map→odom, `/map`).
- Nav2 params (`config/nav2_sim.yaml`): circular footprint r 0.35,
  inflation 0.45, global costmap from `/map` + `/scan`, local costmap
  rolling 4×4 m from `/scan`, RPP as above, behaviors, bt_navigator with
  the default `navigate_to_pose` tree. `cmd_vel_watchdog`.
- Acceptance: `ros2 action send_goal /navigate_to_pose ... {x: 1.6, y:
  -0.3}` walks the robot next to the hydrant without touching it, three
  runs in a row; goal into a prop is refused or replanned; Ctrl-C on Nav2
  stops the robot within 0.5 s (watchdog).

### N2 — RAI drives Nav2 (1 day)

- `wojtek_rai/tools.py`: add `Nav2Toolkit(connector, action_name=
  "navigate_to_pose", workspace_bounds ±6 m)`, `GetCurrentPoseTool("map",
  "base_link")`, `GetOccupancyGridTool`, plus a `GoToNamedPlaceTool` that
  resolves the YAML registry to a pose and calls the same action. Keep
  `walk`/`stop` (text_commander) for short adjustments; `stand_up`,
  `lie_down`, camera unchanged.
- Embodiment: "prefer navigate_to_pose for anything farther than 1 m;
  look at the map when unsure".
- Acceptance: in streamlit, "go to the stop sign and tell me what you
  see" → one `navigate_to_pose` call, arrival, camera, answer. Logged
  transcript under `runs/`.

### N3 — object-grounded goals (2 days, GPU on the laptop)

- `rai_perception` services (GroundingDINO + SAM) in the `wojtek_rai`
  image with the laptop GPU (nvidia runtime is present); `/detection`.
- `GetDistanceToObjectsTool` / `GetObjectPositionsTool` with the depth
  topic and `camera_depth_optical_frame → map` TF → goal 0.5 m in front
  of the object.
- Acceptance: "go to the ball" with no registry entry: detection → pose →
  Nav2 → stops 0.4–0.6 m from the ball (sim ground truth), 3/3.
- *Done as* detection only (GroundingDINO, no SAM) in its own container,
  `docker/perception.Dockerfile` / `run.sh perception`, because SAM2-large
  next to the simulator's renderer overflows a 4 GB GPU; own
  `find_objects` / `go_to_object` tools instead of RAI's, with the colour
  box projected through the depth intrinsics; goal 0.6 m in front of the
  object; measured position error 8 cm in the sim.

### N4 — the physical robot (1 week, human-authorized, separate go)

- Odometry: `rtabmap_odom rgbd_odometry` on the PC from the D435 colour +
  depth over WiFi (colour must go compressed and ≤ 640×360; raw 1280×720
  RGB at 15 Hz is ~40 MB/s and does not fit). Fallback if the floor is too
  feature-poor: leg-kinematic odometry from `/joint_states` + IMU in a
  `robot_localization` EKF.
- Same slam_toolbox + Nav2 params, footprint re-measured, `vx ≤ 0.3`.
- Sequence: (1) mapping-only session, robot carried/driven by pad, agent
  read-only; (2) goals with a person on the disarm button, short hops;
  (3) agent-issued goals.
- Acceptance: map of one room from a pad-driven lap; a 3 m Nav2 goal
  reached without contact, twice; watchdog stop verified.

### Later

- Jetson: move `wojtek_nav` + perception off the laptop.
- SCAN-Planner as a Nav2 controller plugin, or its local map feeding the
  local costmap, once RPP's limits on a legged base are measured.
- `rai_whoami` from `docs/` and the deck cockpit as the chat surface.

## 5. Risks

| risk | handling |
|---|---|
| Narrow depth FOV: costmap blind to the sides while turning | `spin` before long plans, observation persistence 5 s, conservative inflation; later STVL from the point cloud |
| Gait vs. controller: policy does not walk cleanly below ~0.15 m/s or with rapid sign flips | RPP with `use_rotate_to_heading` on; `max_angular_accel` ended up effectively off (see the controller note above: with the gait's dead band an accel-limited turn command never starts); measure tracking error in sim before touching MPPI |
| No dead-man in `policy_node` | `cmd_vel_watchdog` in the loop from N1 on; never bypass it on the real robot |
| Real odometry (visual) drifts or loses track on a plain floor | EKF fallback with leg odometry; SLAM loop closure; accept mapping-only for the first session |
| WiFi bandwidth for depth + colour | depth 424×240 raw (3 MB/s) fine; colour compressed and downscaled |
| Nav2 and the sim both on the laptop CPU (MuJoCo + rendering) | RTF is 1.0 today; if it drops, lower camera rates first |
| Interference with the operator: Nav2 and pad both publish `/cmd_vel` | `twist_mux` with pad priority; disarm always wins |

## 6. Effort

| phase | days |
|---|---|
| N0 basics | 0.5 |
| N1 SLAM + Nav2 in sim | 1–2 |
| N2 RAI tools | 1 |
| N3 object grounding | 2 |
| N4 real robot | 5+ |

Sim end-to-end ("go to the hydrant" via Nav2 from the chat): about 3 days.
