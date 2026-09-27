# Experiment: the DGX <-> Wojtek link

> **Status: EXPERIMENTAL. Not production; nothing here is deployed to the robot.**
> Nothing here is deployed by `ros/deploy.sh`, and no package here is a
> dependency of `wojtek_bringup`. It runs on the DGX. Tested against the
> simulation standing in for the robot, then on the real robot and camera
> Pi: receive only first, then the brain's tasks from the page (below).

The robot's brain runs on the DGX (the model, the pixel resolver, the
operator page); the robot's computer runs the control loop, the policy,
the leg odometry and goto alone (`nav:=true nav_costmap:=false`, fork PR
#14). This relay is the one door between the two ROS graphs, and it needs
nothing new on the robot: it is a client of the `foxglove_bridge` the
robot's service already runs (`foxglove:=true`, port 8765).

```
 DGX, its own ROS domain, localhost only                        the robot (Pi), untouched
 ┌──────────────────────────────────────────────┐   websocket   ┌─────────────────────────────┐
 │ brain · pixel resolver · page · (costmap)    │◄─────────────►│ foxglove_bridge :8765       │
 │            ▲ IN            │ OUT             │     :8765     │ goto · policy · odometry    │
 │        relay_node (this)  ─┘                 │               └─────────────────────────────┘
 │            ▲ IN only                         │   websocket   ┌─────────────────────────────┐
 │            └─────────────────────────────────│◄──────────────│ camera Pi (the second Pi)   │
 └──────────────────────────────────────────────┘     :8765     │ D435 driver + read-only     │
                                                                │ bridge (camera_pi/)         │
                                                                └─────────────────────────────┘
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

## The camera Pi

The pictures come from the second Pi on Wojtek, not through the robot's
bridge (that would put them on the robot's cores 0,1). The camera Pi runs
two nodes from its own `/opt/ros/jazzy`, started by `camera_pi/run.sh`,
and the relay reads them as a second link (`--camera-url`):

| | |
|---|---|
| the D435 driver (`realsense.yaml`) | the set measured safe on a Pi (no cloud, sync or align), colour 640x480 JPEG at 15 fps, depth 424x240 at 6 fps |
| a foxglove_bridge (`bridge.yaml`) | read only: no `clientPublish`, services, parameters or assets; serves exactly `topics.CAMERA_PI` |
| its ROS graph | localhost only, its own domain (43): nothing joins the robot's graph or the DGX's over DDS |

- **The rates are the camera Pi's wifi**, not the sensor's. Its uplink
  showed a 65 Mbit/s link rate (5 GHz, -57 dBm). Depth at 15 fps is 3.2 MB/s
  (measured through the link in the sim); at 6 fps with the colour JPEG
  it is about 1.8 MB/s. 6 Hz depth keeps every colour frame within 83 ms of
  a depth frame, under the resolver's 0.1 s pairing tolerance.
- **The camera link is receive-only by construction**: only the robot's
  link can send (`Link` refuses otherwise), and the camera Pi's bridge
  accepts no client publishing anyway.
- **Static frames**: `base_link -> camera_link` is the robot's URDF (the
  mount, fork PR #9); the driver adds only `camera_link -> *_optical`. The
  relay refuses anything else from the camera Pi (`topics.merge_static`),
  and a frame the robot's URDF names always wins.
- **The two links are independent**: the camera Pi dropping out leaves
  the robot's link up (and the other way round); each reconnects alone.
- **The camera needs `/dev/video*`**: the camera Pi's login user is not in
  the video group, so `run.sh` stops with the two ways out. `sudo bash
  run.sh` changes nothing on the Pi; for good, the groups and
  `ros/deploy/rpi/99-wojtek-realsense.rules` as the robot's `install.sh`
  gives `rpi` (without the rule, a non-root driver cannot reset the camera).
- `run.sh` pins the driver and the bridge to an isolated core each when
  the image isolates some (the Wojtek image's `isolcpus` is for a control
  loop the camera Pi does not run); `WOJTEK_CAMERA_PIN=0` turns that off.
  Ctrl-C, SIGTERM or the SSH session ending stops both nodes.

```bash
# from a machine with this checkout, to the camera Pi (installs nothing):
scp -r experiments/wojtek_link/camera_pi <user>@<camera Pi>:wojtek_camera
ssh -t <user>@<camera Pi> 'bash wojtek_camera/run.sh'
```

## Run it

On the DGX, one command brings up the whole DGX side: the relay, the pixel
resolver, the brain and the operator page, on the DGX's own domain (44),
localhost only. The page listens on 127.0.0.1 only; reach it through SSH.

```bash
# the addresses, once, in the gitignored .env (this directory's or the repo root's):
#   WOJTEK_BRIDGE_URL=ws://<robot>:8765
#   WOJTEK_CAMERA_URL=ws://<camera Pi>:8765
experiments/wojtek_link/run.sh up            # receive only: nothing reaches the robot
experiments/wojtek_link/run.sh up --send     # the page's tasks and STOP drive the robot
experiments/wojtek_link/run.sh status        # both links, the page, what runs
experiments/wojtek_link/run.sh down          # stops only what `up` started
experiments/wojtek_link/run.sh test          # the model-free tests, in the image
ssh -L 8501:127.0.0.1:8501 <DGX>             # then http://localhost:8501
```

`run.sh` runs the relay as below in its own container (`wojtek_link`), and
the resolver and the brain in the `wojtek_robot` container (its built
workspace: create it once with `./ros/dev.sh`). `WOJTEK_REPO` points it at
a checkout with `experiments/wojtek_vlm_gui` when this directory is a copy.
It refuses to start without `WOJTEK_BRIDGE_URL`, or a second time. Tested
on the DGX against the sim: up, a second up refused, `down` leaving the sim
running, `up --send` with goals reaching the sim.

The relay by hand, in the robot's Docker image (it has rclpy and websockets):

```bash
python3 -m wojtek_link.relay_node --url ws://<robot>:8765 --camera-url ws://<camera Pi>:8765   # receive only
# ... --send                  the DGX's goals, cancels and /cmd_vel reach the robot
# ... --in-camera             the camera through the ROBOT's bridge instead: the sim test only
```

`<robot>` comes from `WOJTEK_BRIDGE_URL` or `--url` (default the robot's AP
anchor, `ws://10.42.0.2:8765`), `<camera Pi>` from `WOJTEK_CAMERA_URL` or
`--camera-url` (no default: without it there is no camera link). The DGX
must be on the robot's network.
Everything else on the DGX side (the brain, `pixel_goal_node`, the page)
runs on the same domain (44 here), localhost only.

`/wojtek/link/status` (JSON, latched, once a second): connected, messages
in and out per topic, drops, `kB_s_in`, and `clock_offset_ms` -- the
robot's clock seen from the DGX, read off the Odometry stamps (`clock.py`);
`camera_mount` (whether the robot's URDF gave `camera_link`) and
`static_refused`; the same fields for the camera Pi under `camera`, its
offset read off the camera_info stamps.

## Clocks

The resolver looks the robot's pose up at the stamp of the camera's
picture: the camera Pi, the robot Pi and the DGX must agree on the time.
`clock_offset_ms` says how far the robot is (plus a few ms of delivery);
`camera.clock_offset_ms` the same for the camera Pi (plus the driver's
pipeline, since a picture's stamp is its exposure). Over ~50 ms beyond
that pipeline, fix the clocks before
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

The camera link (2026-09-27, DGX): a second bridge with `camera_pi/bridge.yaml`
(port 8766) on the sim's graph as the camera Pi, the relay with
`--camera-url`, receive only, the resolver on 44.

| check | result |
|---|---|
| the camera Pi's bridge | capabilities `['connectionGraph']` only; an empty list aborts 3.5.0 |
| pictures on the DGX | depth 15.0 Hz, colour 5.0 Hz (the sim's rates), 3182 kB/s on the camera link, 18 kB/s on the robot's |
| who reads the depth on the robot's graph | the camera Pi's bridge and the sim; the robot's bridge not |
| the resolver off the camera link | 4 pixels -> `sent`, object points consistent (0.64 / 0.89 m ahead, +-0.21 m to the sides, on the floor), standoff goals 0.7 m short; none reached the robot |
| static frames | the robot's URDF gave the mount and the optical chain; all 11 from the camera Pi's side refused (in the sim it is the same graph); `base_link -> camera_color_optical_frame` 0.32 / 0 / 0.07 m on the DGX |
| the camera Pi dropped out | camera link `ConnectionRefused`, the robot's link up (odom 25.8 Hz); back after the restart, depth 15 Hz again |
| `camera_pi/run.sh` (the DGX, no D435) | both params files parse, both nodes up, SIGTERM and SIGHUP stop both |
| unit tests | 25 passed |

The real camera Pi (2026-09-27): `sudo bash run.sh` on it (its login user
is not in the video group, see below), the relay on the DGX with
`--camera-url` only; its robot URL pointed at a closed local port, so the
robot was not contacted.

| check | result |
|---|---|
| the driver | `RealSense Node Is Up!`, depth 424x240 at 6 fps, colour 640x480 at 15 fps, the initial reset done; pinned to the isolated cores 2 and 3 |
| on the DGX | depth 6.1 Hz, 204 KB a frame; colour JPEG 15.1 Hz, 26 KB a frame (quality 80); 1672 kB/s on the camera link |
| a picture's age on landing | median 150-170 ms, max 1.2 s over 12 s (the wifi); the resolver works on stamps, so this is freshness, not error |
| clocks | camera Pi vs DGX `clock_offset_ms` -9.4 (both NTP-synced) |
| the depth | 68 % valid pixels; the camera was 0.13 m from a wall, which the JPEG shows |
| static frames | the driver's `camera_link -> camera_{color,depth}_frame -> *_optical_frame`, all accepted |

The first contact with the real robot (2026-09-27, receive only: no
`--send`, nothing advertised to the robot), with the camera Pi as above:

| check | result |
|---|---|
| the robot's bridge | foxglove_bridge 3.5.0, 38 topics, every IN and OUT topic present; `clientPublish` on (and services, parameters, assets: see below) |
| receive only | `out {}`, nothing advertised; both links up |
| odometry on the DGX | 28.9 Hz, 18 kB/s from the robot |
| the mount | `base_link -> camera_link` 0.32 / 0 / 0.07 m, pitch 0.262 rad, from the robot's URDF: fork PR #9 is on the robot |
| the whole chain | `base_link -> camera_color_optical_frame` 0.32 / 0.015 / 0.07 m (the robot's mount + the camera Pi's driver), nothing refused |
| clocks vs the DGX | robot 21.6 ms, camera Pi -4.5 ms (plus delivery): within 30 ms of each other |
| the resolver on the DGX | 1 of 4 pixels resolved (an object 0.74 m ahead-left on the floor); 3 `no_depth` (the camera stood 0.13 m from a wall, inside the D435's minimum range); the goal stayed on the DGX |
| stopping the relay | clean, 16 of 16 SIGINT/SIGTERM stops after the spinner fix (before it, 2 of 8 aborted at teardown) |

The first tasks on the real robot (2026-09-27, `--send`, a person at the
pad). The robot was stood up with the pad: the page's arm, stand, policy
and lie-down buttons are ROS services, which the relay does not carry, so
the page answers them with "not available". Tasks and STOP cross. Each
task ran under a 2 m fence around its start that would have frozen the
robot; it never tripped.

| task | what happened |
|---|---|
| "podejdź do drzwi" | the door not in view: search turns, about a full circle; one "door" answer failed its own verification. At 26 s the door was seen, but its pixel had no depth, so the brain stepped towards it (1.0 m, goto `reached`). At 44 s it looked again and resolved the door 0.62 m away, inside the 0.7 m standoff: done at 45 s. Out: 181 `/cmd_vel`, 12 goals, nothing dropped |
| "podejdź do banana" | seen at once, 2.08 m away; goto walked 1.3 m in 5 s, then stood 0.16 m short of the setpoint for 15 s, turning in place (-65 deg), until STOP from the page at 26 s |
| "znajdź banana i podejdź do niego" | two search turns (about 200 deg), seen at 12 s, 2.03 m away; walked 1.4 m, then 15 s creeping and turning at the setpoint; `reached` at 46 s, 0.77 m from the banana: done |

**The slow finish is goto against the real gait.** goto slows into a goal at
0.6 x the distance (`goto.py`), so 0.16 m out it commands 0.1 m/s, which the
real gait does not walk; it reports `reached` only inside 0.15 m
(`reach_tolerance`). The robot stands just outside, correcting its heading.
The sim's gait walks small commands, which is why the sim never showed it.
Not changed here: navigation stays as it is on the robot.

The robot's link dropped once between tasks (a keepalive timeout on the
wifi, back after 1.5 s); a drop mid-walk is caught by goto's 3 s dead-man.

**The robot's bridge is open to its whole network.** It offers
`clientPublish`, `services`, `parameters` and `assets` to any client, and
the robot now sits on a shared wifi: anyone there with Foxglove could
publish `/cmd_vel` or joint targets, or call the arm services. The relay
only ever uses `clientPublish` for the OUT list. Narrowing the robot's
bridge (`capabilities`, `client_topic_whitelist`, `service_whitelist`) is a
change on the robot, left to its owners.

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
| `run.sh` | the DGX side in one command: up [--send] / down / status / test |
| `wojtek_link/relay_node.py` | the relay: rclpy on one thread, one websocket per link on asyncio |
| `camera_pi/` | what runs on the camera Pi: the driver's and the bridge's parameters, `run.sh` |
| `tests/` | model-free: the frames, the offset, the lists, the static-frame rule, the camera Pi's files |

```bash
cd experiments/wojtek_link && python3 -m pytest -q     # 25 tests, no ROS needed
```
