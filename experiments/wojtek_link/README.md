# Experiment: the DGX <-> Wojtek link

> **Status: EXPERIMENTAL. Not production; nothing here is deployed to the robot.**
> Nothing here is deployed by `ros/deploy.sh`, and no package here is a
> dependency of `wojtek_bringup`. It runs on the DGX. Tested against the
> simulation standing in for the robot; not yet run against the robot.

The robot's brain runs on the DGX (the model, the pixel resolver, the
operator page); the robot's computer runs the control loop, the policy,
the leg odometry and goto alone (`nav:=true nav_costmap:=false`, fork PR
#14). This relay is the one door between the two ROS graphs, and it needs
nothing new on the robot: it is a client of the `foxglove_bridge` the
robot's service already runs (`foxglove:=true`, port 8765).

```
 DGX, its own ROS domain, localhost only                         the robot (Pi), untouched
 ┌──────────────────────────────────────────────┐   websocket   ┌───────────────────────────┐
 │ brain · pixel resolver · page · (costmap)     │◄────────────►│ foxglove_bridge :8765      │
 │            ▲ IN            │ OUT              │   :8765      │ goto · policy · odometry   │
 │        relay_node (this)  ─┘                  │              └───────────────────────────┘
 └──────────────────────────────────────────────┘
  camera Pi ──► DGX directly (not through the robot's bridge)
```

## Why a relay and not the robots joining one DDS graph

One graph would make every node on the DGX a node on the robot. A sim
started on the DGX without `ROS_LOCALHOST_ONLY`, a test publisher, a second
brain: each would put `/wojtek/joint_targets`, `/cmd_vel` or `/tf` straight
into the real robot. Here the DGX's graph stays on its own domain,
localhost only, and only the topics in `wojtek_link/topics.py` cross:

| direction | topics | why |
|---|---|---|
| IN (robot -> DGX) | `/wojtek/odom` (also published as the DGX's `odom -> base_link` TF), `/tf_static` (merged, latched), `/wojtek/nav/status` | the pose the resolver and the brain need, goto's word; one 25 Hz Odometry costs the robot's bridge far less than its whole `/tf` |
| OUT (DGX -> robot) | `/wojtek/nav/goal`, `/wojtek/nav/cancel`, `/cmd_vel` | the resolver's setpoints, the brain's approach steps and search turns, STOP |
| never | `/wojtek/joint_targets`, `/wojtek/arm`, `/wojtek/enable`, `/tf`, `/tf_static`, `/joint_states`, `/wojtek/odom`, `/robot_description` | nothing that sets a joint, arms, or re-parents the robot's pose; `topics.check()` refuses a list that would |

- **Receive-only by default.** OUT is off unless `--send`. The first contact
  with the robot reads its pose and goto's word and sends nothing.
- **OUT is dropped, never queued, while the link is down**: a `/cmd_vel`
  from before a drop-out must not arrive after it.
- **A goal's stamp goes out as zero** ("now" on the robot). goto takes an
  odom goal as it is and transforms a base_link one at the latest pose; a
  DGX stamp would be read against the robot's clock. Order is kept on the
  one websocket, so a cancel still lands after the goal it cancels.
- **If the link drops, the robot's own dead-men stop it**: goto drops its
  setpoint 3 s after the last one, policy_node zeroes `/cmd_vel` after
  0.5 s of silence (the brain's turns).
- The page's Robot buttons (Stand up, Arm, Disarm, Policy) are ROS services
  and do not cross: arming stays with the pad and the Deck on the robot.
  STOP crosses (a cancel and zero `/cmd_vel`).

## Run it

In the robot's Docker image on the DGX (it has rclpy and websockets):

```bash
docker run -d --name wojtek_link --network host --ipc host \
  -e ROS_LOCALHOST_ONLY=1 -e ROS_DOMAIN_ID=44 -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -v "$PWD/experiments/wojtek_link:/link" docker-wojtek_robot:latest sleep infinity
docker exec -it wojtek_link bash -lc 'source /opt/ros/jazzy/setup.bash; cd /link; \
  PYTHONPATH=$PWD:$PYTHONPATH python3 -m wojtek_link.relay_node --url ws://<robot>:8765'          # receive only
# ... --send                  the DGX's goals, cancels and /cmd_vel reach the robot
# ... --in-camera             ALSO the camera, for the sim test only (on the robot it comes from the camera Pi)
```

`<robot>` comes from `WOJTEK_BRIDGE_URL` or `--url` (default the robot's AP
anchor, `ws://10.42.0.2:8765`). The DGX must be on the robot's network.
Everything else on the DGX side (the brain, `pixel_goal_node`, the page)
runs on the same domain (44 here), localhost only.

`/wojtek/link/status` (JSON, latched, once a second): connected, messages
in and out per topic, drops, and `clock_offset_ms` -- the robot's clock seen
from the DGX, read off the Odometry stamps (`clock.py`).

## Clocks

The resolver looks the robot's pose up at the stamp of the camera's
picture: the camera Pi, the robot Pi and the DGX must agree on the time.
`clock_offset_ms` says how far the robot is (plus a few ms of delivery);
do the same for the camera Pi's stream. Over ~50 ms, fix the clocks before
trusting the resolver: at 0.3 m/s, 0.1 s is 3 cm of error, a clock set to
boot time is nowhere.

## Measured in the simulation (2026-09-27, DGX)

The sim as the robot: its own domain (42), `nav:=true nav_costmap:=false`
(goto alone, as the robot's drop-in), its foxglove_bridge 3.5.0 (the
robot's version) on 8765. The DGX side on domain 44: this relay, the
resolver, the brain (Qwen3-VL 30B-A3B on Ollama) and the page.

| check | result |
|---|---|
| receive only | odom 25.0 Hz on the DGX, `odom -> base_link` and the camera mount (0.32 / 0 / 0.07 m, 15 deg) resolved there, goto's word latched; a goal published on the DGX did NOT reach the robot |
| a goal from the DGX (`--send`) | goto `driving` 0.02 s after it, `reached` after 2.3 s, the robot 0.43 m further (ground truth); goto's words back on the DGX in the same order |
| the allow-list | a JointState on `/wojtek/joint_targets` and a `/tf` published on the DGX: 0 reached the robot |
| the brain across the link | "podejdź do skrzyni": the robot walked at 0.17 m/s |
| STOP on the page, walking | on the robot: 23 zero Twists in 1.2 s, no non-zero one after, goto `idle` 0.04 s after; 0.04 m/s in the first second, 0.004 after |
| STOP on the page, turning (19 deg/s) | 0.0 deg/s in the first second, no fall |
| the link killed mid-walk (SIGKILL) | goto kept its last setpoint for 2.7 s, then `idle` (its 3 s dead-man) |
| SIGTERM / SIGINT | the relay exits, link closed (a first version did not: rclpy's handlers stopped the ROS side and left the websocket up) |
| clock offset | 1.0 ms (one machine: the delivery) |

**Seen on the way, and the reason for the next step:** with no costmap on
the robot, goto drives straight. In the link-loss run the robot walked into
an obstacle and kept pushing at 0.3 m/s commanded while standing still. The
veto has to come back before long tasks on the robot: the DGX builds the
costmap from the camera's depth and sends it as `/wojtek/nav/costmap` (goto
reads it from anyone). Not in OUT yet: goto subscribes to it transient-local,
and whether the bridge's client publisher can match that is untested.

## Layout and tests

| path | purpose |
|---|---|
| `wojtek_link/topics.py` | what crosses, what never does, the invariants (pure) |
| `wojtek_link/protocol.py` | the Foxglove WebSocket frames the relay reads and writes (pure) |
| `wojtek_link/clock.py` | the robot's clock offset read off the traffic (pure) |
| `wojtek_link/relay_node.py` | the relay: rclpy on one thread, the websocket on asyncio |
| `tests/` | model-free: the frames, the offset, the lists |

```bash
cd experiments/wojtek_link && python3 -m pytest -q     # 15 tests, no ROS needed
```
