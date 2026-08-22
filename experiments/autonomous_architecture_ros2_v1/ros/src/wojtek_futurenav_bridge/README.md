# wojtek_futurenav_bridge

The missing edge of the sim E2E loop
([machinekind/w01-tek#13](https://github.com/machinekind/w01-tek/issues/13)):
drive the deployment ROS graph with the FutureNav action server, with nothing
else from the agent stack in between — no voice pipeline, no router, no chat
agent.

```
/wojtek/nav_instruction (String)             "walk to the sofa" | "stop"
/camera/camera/color/image_raw (Image)       simulated D435 color feed
TF odom -> base_link                         sim ground-truth pose
        │
        ▼
POST /reset, POST /act  ──►  discrete action (MOVE_FORWARD 0.25 m /
        │                    TURN_LEFT|RIGHT 15° / STOP)
        ▼
MidLevelExecutor (wojtek_rl.midlevel)  ──►  /cmd_vel (Twist) @ 20 Hz
```

Command-stream etiquette matches `text_commander`: continuous publishing
while an episode runs (zeros while a decision is in flight, so a non-zero
command is never left latched), exactly one zero Twist when the episode ends,
then silence — the console or a pad can take `/cmd_vel` without being
shouted over.

Guards: 400-step budget and 25-turn anti-spin (FutureNav's own eval limits),
6 consecutive stall-aborted moves = wedged, any HTTP failure = abort + zero
Twist. All of it lives in the rclpy-free `episode.py`/`frames.py`/
`futurenav_http.py`, covered by the experiment's model-free suite
(`./experiments/autonomous_architecture_ros2_v1/run.sh test`).

This package lives outside `ros/src` on purpose: `ros/deploy.sh` can never
ship it to the robot, and it may import `wojtek_rl` (the training project)
where `ros/src` may not.

## Run sequence

1. **FutureNav server** on a GPU host (~10 GB VRAM alone):
   see `training/wojtek_rl/futurenav_server/README.md`; serves on `:8100`.
2. **Scene meshes** (once): `./training/run.sh room-assets` / `build-room`
   for the scene you want; `scene_flat.xml` / `scene_castle.xml` live in
   `ros/src/wojtek_description/mujoco/`.
3. **The ROS sim session** from the host:

   ```bash
   ./ros/sim.sh model_xml:=/ros2_ws/src/wojtek_description/mujoco/scene_flat.xml \
       policy:=<walking-policy-ref>
   ```

   (The published keeper's `policy_meta.json` carries a wrong `action_scale`
   — use a locally corrected artifact until a fixed keeper lands; see
   `training/docs/scan-planner.md`.)
4. **Build and start the bridge** in a second container shell (`./ros/dev.sh`):

   ```bash
   cd /ros2_ws/../w01-tek/experiments/autonomous_architecture_ros2_v1
   ./run.sh build            # colcon build of the experiment packages
   source install/setup.bash
   # wojtek_rl must be importable next to the overlay:
   export PYTHONPATH=$PWD:$PWD/../../training:$PYTHONPATH
   ros2 run wojtek_futurenav_bridge futurenav_bridge \
       --ros-args -p vlm_url:=http://<gpu-host>:8100
   ```
5. **Arm the robot** (web console on `http://localhost:8080`, or services):
   `/wojtek/zero` → `/wojtek/stand_up` → `/wojtek/arm`.
6. **Give it an instruction**:

   ```bash
   ros2 topic pub -1 /wojtek/nav_instruction std_msgs/String "data: walk to the sofa"
   ros2 topic pub -1 /wojtek/nav_instruction std_msgs/String "data: stop"   # cancel
   ```

Watch it in the web console (camera + telemetry), RViz/Foxglove (TF + robot
walking), and the bridge's own log (one line per decision, one per episode
end).

## Parameters

| param | default | meaning |
|---|---|---|
| `vlm_url` | `http://127.0.0.1:8100` | FutureNav action server |
| `cmd_rate_hz` | 20.0 | `/cmd_vel` publish rate |
| `frame_px` | 224 | square frame size sent to the server |
| `odom_frame` / `base_frame` | `odom` / `base_link` | pose lookup |
| `vx_max` / `vy_max` / `yaw_max` | 0.4 / 0.25 / 0.7 | command caps (demo profile) |
| `max_steps` / `max_rotation` / `max_blocked` | 400 / 25 / 6 | episode guards |

## Known v1 compromises (accepted in the issue)

- No SCAN planner: straight-march execution; brushing furniture is the
  VLM's raw behavior and part of what this stack observes.
- Camera FOV: the simulated D435 color is ~69° HFOV; FutureNav trained on
  90° square Habitat frames. Measured degradation is a follow-up.
- Pose is sim ground truth; the robot has no odometry yet — this bridge is
  a sim-testing tool, not a deployment artifact.
- The single-zero guarantee holds only while this node is alive:
  `policy_node` latches the last `/cmd_vel`, so if the bridge is killed
  mid-walk the robot keeps executing the last command (the same property
  `text_commander` documents). In sim that is a walking-into-the-wall
  annoyance; it is one more reason this node must never ship to the robot
  without a downstream command watchdog.
