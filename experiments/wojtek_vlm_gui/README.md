# Experiment: the operator GUI for Wojtek's VLM brain

> **Status: EXPERIMENTAL. Not production.** Nothing here is deployed by
> `ros/deploy.sh`, and no package here is a dependency of `wojtek_bringup`.
> All of it runs on the PC, next to the simulation or next to the robot's
> ROS 2 graph, with one exception: for the camera bench, `sysmon_node`, a
> small read-only reader of the RPi's load, is copied to the RPi and started
> there by hand (see "The camera bench"). Launching, arming and disarming
> the robot remain human actions. Interfaces are unstable by definition.

A JavaScript page on http://localhost:8501 (served by `wojtek_vlm_gui.server`) where the operator types a task
("go to the purple pillar"), watches Wojtek's VLM brain work through it
step by step, and keeps the robot's Stand up / Arm / Disarm buttons under
their hand. The brain is `wojtek_nav`'s `vlm_brain_node`
([ros/src/wojtek_nav/README.md](../../ros/src/wojtek_nav/README.md)): it
reads the camera, asks the model, verifies, hands pixels to the resolver
and setpoints to `goto`, and turns to search. This page does none of that.
It is a thin ROS 2 client: two topics out, four topics in, four operator
services.

## How it is wired

```
 operator ──► JS page :8501 (server.py, this experiment, its own container)
                │ task ─────────► /wojtek/vlm/instruction (String) ──► vlm_brain_node ── /v1/chat/completions ──► Ollama on the DGX
                │ STOP / Esc ───► /wojtek/nav/cancel + "" instruction + /cmd_vel 0 (1 s) │ pixel
                │ step log ◄──── /wojtek/vlm/status (JSON, latched)                     ▼
                │ picture ◄───── /wojtek/vlm/annotated (one frame per model call)  pixel_goal_node ──► /wojtek/nav/goal ──► goto_node ──► /cmd_vel
                │ words ◄─────── /wojtek/nav/status, /wojtek/nav/pixel_status           costmap (odom, 6 x 6 m) ──────────┘
                └ Robot buttons ► /wojtek/stand_up, lie_down, arm, enable (services); Disarm and Lie down also send STOP

 pad, Steam Deck, web console, text_commander ──► /cmd_vel too: policy_node keeps the last message, 0.5 s dead-man
```

- The page never drives the robot. The one thing it puts on `/cmd_vel` is
  a zero Twist, on STOP (`wojtek_vlm_gui/limits.py` lists everything it
  touches; the tests assert the publisher and subscription sets and that
  every Twist it sends is zero). It shows two pictures: the frame the brain
  annotated, one per model call, and (since 2026-09-26) the camera's own
  JPEG live, the same `/compressed` stream the brain reads. The page runs
  on the PC, never on the robot, so it is a second reader of the JPEG next
  to the brain (tens of KB a frame at 640x480), over the cable on the
  bench or the robot's AP otherwise.
- **STOP freezes Wojtek where it stands** (the button, Esc anywhere on the
  page, or a stop word typed as a task). The server acts on it the moment
  it arrives, ahead of any service call or task still running. It cancels
  goto's goal and the brain's task, then holds `/cmd_vel` at zero for 1 s,
  one zero Twist every 50 ms, which outlasts a brain turn or goto tick in
  flight; after that `policy_node`'s 0.5 s dead-man keeps the robot at zero
  until a drive source speaks again. Zero velocity is the freeze, on
  purpose: with the policy off or the robot disarmed the drives hold their
  last joint targets, and a trot stopped mid-step stands on two diagonal
  feet and tips over. At zero velocity the gait keeps balancing and the
  robot stands. A pad or Deck pushed during the hold fights it.
- The model route is the brain's, not the page's: `VLM_URL` and
  `VLM_MODEL` in the repo-root `.env`, read by `ros/sim.sh` into the
  `wojtek_robot` container and by `brain.launch.py`. The only inference
  route is Ollama on the DGX, reached through an ssh tunnel from the PC.
  The tunnel listens on the PC's own 127.0.0.1, so the address in
  `VLM_URL` depends on the OS: 127.0.0.1 on Linux, host.docker.internal on
  macOS (step 1 below).
- Arming stays a human button. Disarm and Lie down also STOP the brain,
  because disarming stops the motors, not a task that would otherwise
  keep publishing goals into a disarmed robot.

## Run it (simulation)

```bash
# 0. once per machine
./experiments/wojtek_vlm_gui/run.sh build

# 1. every session: the tunnel to the DGX's Ollama (alias in your ssh config, never in the tree)
ssh -f -N -L 11435:127.0.0.1:11434 <dgx alias>
curl -s http://127.0.0.1:11435/api/tags | head -c 200      # the model tag must be listed
#    root .env: VLM_MODEL=qwen3-vl:30b-a3b-instruct, and VLM_URL as the sim's container sees the tunnel:
#      Linux: VLM_URL=http://127.0.0.1:11435             (the container shares the PC's network)
#      macOS: VLM_URL=http://host.docker.internal:11435  (Docker runs in a VM; that name is the Mac)

# 2. the sim with Pawel's nav stack and the brain
./ros/sim.sh model_xml:=scene_nav.xml leg_odom:=true nav:=true vlm:=true

# 3. the page
./experiments/wojtek_vlm_gui/run.sh gui        # http://localhost:8501 (this computer only)
```

The page's server listens on this computer only, and it refuses a
websocket whose Origin is not the page's own address, so another web page
open in the operator's browser cannot drive the robot. To use the page
from another device, pass `--host 0.0.0.0` (`run.sh gui --host 0.0.0.0`,
Linux) or change the ports line in `docker/compose.mac.yaml` (Mac). Then
anything on that network can arm the robot.

In the page: `Stand up`, then `Arm`, then type the task. Use the
`-instruct` model tags only: Ollama ignores `think=false` on the thinking
tags and every step pays a hidden preamble.

## The camera bench: a person as the robot's legs

For testing the brain on the real camera without the robot walking. The
robot's computer is cabled to the PC and runs only the camera, the
`sysmon_node` and a zenoh bridge (zenoh-plugin-ros2dds) that passes an allow
list of topics over the cable: the colour JPEG, the raw depth, their
camera_info, the load, the drive topics and the four operator services.
Everything else runs on the PC, on its own ROS domain so it never mixes
with a sim session: the other end of the bridge, the real nav stack
(`costmap.launch.py`: costmap, goto, the pixel resolver) on the robot's
depth, the brain, the camera mount transform (from
`wojtek_perception_bringup/config/extrinsics.yaml`), and this page.

The one missing piece is the legs. `./run.sh walker` supplies them: it
integrates `/cmd_vel` into odom -> base_link as if the robot had walked it
(yaw gain 0.76, the gait's 0.38 of the commanded 0.5 rad/s the brain times
its turns by) and publishes a guide the page shows in its **Walk** panel:
an arrow and words for the command (`WALK FORWARD 0.30 m/s`, `TURN LEFT on
the spot, 22 deg/s`, `STAND STILL`), the turn or distance so far, and a
compass to goto's goal from where the robot stands. The person carrying the
camera does what the panel says. Never run the walker next to a real robot:
it publishes a second odom -> base_link.

```bash
ROS_DOMAIN_ID=43 ./experiments/wojtek_vlm_gui/run.sh walker
ROS_DOMAIN_ID=43 ./experiments/wojtek_vlm_gui/run.sh gui     # http://localhost:8501
```

`sysmon_node` is the one piece that has to run on the RPi, because the
Computer panel's numbers live in the RPi's `/proc`. `ros/deploy.sh` never
copies it, so it goes over by hand, into its own folder outside
`~/wojtek_ws`: deploy's `--delete` copy of the workspace must never pick it
up, and it must not become part of the robot's workspace.

```bash
# on the PC, from the repository root; <robot> is the RPi's ssh address (RPI_HOST, the one ros/deploy.sh uses)
rsync -a --delete --exclude __pycache__ --exclude web \
  experiments/wojtek_vlm_gui/wojtek_vlm_gui <robot>:wojtek_vlm_gui_bench/
# then in an ssh shell on the RPi (its ~/.bashrc sources ROS 2 on the robot's own domain)
cd ~/wojtek_vlm_gui_bench && taskset -c 1 python3 -m wojtek_vlm_gui.sysmon_node
```

It only reads `/proc` and `/sys` and publishes one message a second.
Ctrl-C stops it, and `rm -rf ~/wojtek_vlm_gui_bench` removes it. Start it
from an ssh shell, not as a one-shot `ssh <robot> 'python3 ...'`: a shell
that is not interactive skips the ROS block in `~/.bashrc`. `python3 -m`
finds the package in the current folder, and it needs only the standard
library, rclpy and std_msgs, which the RPi's ROS install already has.
`taskset -c 1` puts it on the core it shares with the bridge in the
measurements below.

The page is laid out as a chat: the operator's tasks on the right, the
brain's steps on the left in plain words ("I see the chair, heading
there", "Blocked on the way, target 1.6 m away"), each look with the
picture the model answered on, newest at the bottom; the log scrolls
inside the screen and the page itself never grows. The history lives in
the browser (a reload starts with the brain's last task only).
`GET /health` on the server reports each stream's counter and age, the
first place to look when a panel stops moving.

Two bridge traps, both seen on the bench. A latched (transient-local)
topic comes out of the bridge as old messages out of order, so the bench
topics are volatile. And a topic whose only PC-side reader restarts can
lose its route for good: the RPi end drops its reader and never rebuilds
it. Keep one permanent reader on the PC side of any such topic (the bench
does, for `/wojtek/sys/stat`).

Measured on the bench (2026-09-26, RPi 4, D435 on a USB 2 port): colour
640x480 and raw depth 480x270, both at 30 fps from the camera, arrive on
the PC at ~20 fps (the bridge caps them). The camera driver, pinned to
core 0 with the depth filters, point cloud, alignment and RGBD off, fills
that core: ~65 % converting and JPEG-encoding colour at 30 fps (a third
of those frames never leave, the bridge sends 20), ~19 % taking USB
frames, ~13 % kernel interrupt work for USB and for the Ethernet, whose
interrupts also land on core 0. The bridge and `sysmon_node` share core 1
at ~35 %: ~21 % the bridge reading the frames and sending ~5.5 MB/s, the
rest cross-core wake-ups and kernel workers.

## The rule with the pad and the Deck

`/cmd_vel` is shared by every drive source, and `policy_node` keeps
whichever message came last. While a brain task drives, a stick or the
Deck's gate publishes into the same topic and the robot alternates between
the two. **STOP the brain before driving by hand.** Disarm from anywhere
(pad A, the Deck, this page) stops the motors; only this page's Disarm
also ends the task.

## What the page shows

| element | source |
|---|---|
| one line per step: `#3 ask {"type":"goal",...} 1.2s`, `#3 verify ...`, `#3 pixel_goal -> reached target 0.85 m`, `#4 turn 45.0deg (total 90deg)` | `/wojtek/vlm/status` |
| the picture under each `ask` line, with the model's point drawn by the brain | `/wojtek/vlm/annotated` |
| `finished: done` in green, any other result (`gave_up`, `cancelled`, `replaced`, `error ...`, `timeout`) in red | the finishing status |
| `goto: driving`, `pixel: sent` in the sidebar | `/wojtek/nav/status`, `/wojtek/nav/pixel_status` |
| the robot's own answer after every Robot button | the services' responses |
| right column: one bar per core of the robot's computer (the RT cores marked), load, memory, SoC temperature | `sysmon_node` on the RPi (`/proc/stat`, `/proc/loadavg`, `/proc/meminfo`, the thermal zone) via `/wojtek/sys/stat` |
| right column, below: the camera live, with the frame's size and age; a warning when the driver goes quiet | `/camera/camera/color/image_raw/compressed` |

A task typed while nobody subscribes to the instruction topic (no session
with `vlm:=true`) is refused in the page with a message, never dropped
silently. A task still waiting for the brain when STOP is pressed is
dropped with a message too, so a stop is never followed by the task it
stopped. A stop word (`stop`, empty, the brain's own `stój`) typed as a
task is a STOP, with the same freeze.

## Layout

| path | purpose |
|---|---|
| `run.sh` | `build \| up \| down \| shell \| gui \| test` |
| `docker/` | the `wojtek_vlm_gui` image (ros:jazzy-ros-base + a Starlette/uvicorn venv) and compose service (host net, the sim's DDS settings) |
| `wojtek_vlm_gui/limits.py` | every topic and service the page touches; what it never publishes |
| `wojtek_vlm_gui/task_log.py` | the brain's status stream as the page prints it (pure) |
| `wojtek_vlm_gui/brain_client.py` | the ROS 2 node: instruction and cancel out, status, picture and camera JPEG in |
| `wojtek_vlm_gui/sysmon.py` | the computer panel's readers of /proc and /sys (pure) |
| `wojtek_vlm_gui/arm_switch.py` | the operator's service calls |
| `wojtek_vlm_gui/server.py` | the page's server: one ROS node, the page on `/`, the websocket on `/ws`; listens on this computer only and refuses a websocket from any other page's Origin |
| `wojtek_vlm_gui/web/` | the page itself: `index.html`, `app.js`, `style.css` (no build step) |
| `wojtek_vlm_gui/wire.py` | the websocket protocol: JSON text frames, tagged binary frames for the camera, the annotated picture (numbered for its `ask` line) and the depth (pure) |
| `wojtek_vlm_gui/walker.py`, `walker_node.py` | the bench's legs: /cmd_vel integrated into odom->base_link and the page's Walk guide |
| `wojtek_vlm_gui/sysmon_node.py` | the robot's load on `/wojtek/sys/stat`, the only piece that runs on the robot, copied there and started by hand (see The camera bench) |
| `tests/` | model-free: the log, the client against a mock node, the switch, the server's command path, the machine parsers, what `run.sh` hands the container, the hygiene guard |

```bash
./experiments/wojtek_vlm_gui/run.sh test                      # in the container
EXP_PY=<python with pytest> ./experiments/wojtek_vlm_gui/run.sh test   # on the host: the ROS-free tests, the rest skipped
```

## Isolation rules

1. Nothing outside this directory imports anything inside it, and nothing
   in the root markdown, `ros/`, `training/` or `docs/` names this
   directory (`tests/test_repo_hygiene.py`).
2. `ros/sim.sh`, `ros/dev.sh`, `ros/docker/compose.yaml` are untouched; this
   experiment runs its own compose project next to them.
3. No secrets and no private host identities in the tree: the DGX alias is
   the operator's ssh config, the model endpoint is the gitignored `.env`.

## Physical robot (human-authorized)

The page talks to the robot's graph over the WiFi AP exactly as it talks
to the sim, with the brain running on the PC (`ros2 run wojtek_bringup
robot --web-console --vlm`) and the RPi stack with
`deck_camera:=true nav:=true` (the service already has deck_camera).
Launching, arming and disarming stay human actions outside this
experiment. Not run yet.
