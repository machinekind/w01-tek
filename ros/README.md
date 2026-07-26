# wojtek_ws

ROS 2 Jazzy workspace for the four-bar-bot ("wojtek"). The system is split
across **two machines** talking over DDS:

- **PC** — development + visualization (RViz, PlotJuggler, MuJoCo sim, teleop).
  Runs in Docker; no ROS install on the host.
- **RPi 5** — the robot: the hard real-time control loop (`ros2_control` 400 Hz,
  MD80/candle, IMU, policy). Native Ubuntu + RT kernel, no Docker.

They share `ROS_DOMAIN_ID=42` over CycloneDDS. The RPi is a fixed anchor at
`10.42.0.2` on whichever link is up — ethernet cable when docked, its own wifi
AP (`wojtek-link`) when mobile. See [`deploy/rpi/README.md`](deploy/rpi/README.md).

---

## PC (dev / viz) — quickstart

**Just want the simulator?** One command, Linux or macOS (only Docker needed;
first run builds the image):

```bash
./sim.sh        # MuJoCo sim + policy + viz + drive UI, torn down on Ctrl-C
```

The operator console (drive pad, arm/pose buttons, jog, telemetry) is the
**web console** by default — open <http://localhost:8080> in any browser (a
phone on the robot's AP works too). Drive commands are dead-man guarded:
if the page goes silent mid-drive, `/cmd_vel` is zeroed.

3D visualization: on Linux `sim.sh` opens RViz (X11). On macOS GL-heavy RViz
over X11 is slow, so it starts `foxglove_bridge` instead — open the native
[Foxglove](https://foxglove.dev/download) app and connect to
`ws://localhost:8765` (add a 3D panel). `--rviz` / `--foxglove` override the
platform default.

**Bluetooth Xbox pad**: left stick = vx/yaw, right stick left-right = strafe,
**A** toggles arm, D-pad up/down steps the standing height; on the `joy`
paths **Y**/**B** additionally trigger the stand-up / lie-down ramps (browser
pads keep those on the console buttons). Driving is always live (no
drive-enable gate); a dead-man zeroes `/cmd_vel` if the pad drops off. Two
paths, same drive mapping:

- The **web console reads a pad in the browser** (Gamepad API): pair the pad
  with whatever machine the browser runs on (macOS included), open the
  console page and press any pad button — the sticks take over the drive pad.
- `./sim.sh --gamepad` runs the in-container `joy` driver + `gamepad_teleop`
  (from `wojtek_teleop`) instead of the web console. Linux only (the
  container sees the pad through the `/dev` mount; Docker on macOS can't
  pass input devices, so there this flag falls back to the web console path
  with a hint).
- On the real robot the pad can pair with the **RPi itself** — no PC in the
  loop: `robot.launch.py gamepad:=true` (the `wojtek_teleop` package is part
  of the RPi build; bluez/ERTM groundwork comes from `deploy/rpi/install.sh`,
  pairing itself is a one-time `bluetoothctl` scan/pair/trust/connect).

`--qt-console` switches to the original Qt operator console (an X11 app).
On macOS that needs a one-time XQuartz setup — `sim.sh` then starts XQuartz
itself when needed:

```bash
brew install --cask xquartz
defaults write org.xquartz.X11 nolisten_tcp -bool false   # allow TCP :6000
```

For everything else (real robot, deploys, hand-run launches):

```bash
git clone <this repo> && cd wojtek_ws
./build.sh                 # build the ROS 2 Jazzy Docker image (once)
./deploy/pc/setup-net.sh   # create the wojtek-eth link profile (once)
./dev.sh                   # drop into a shell in the container (ROS already sourced)
```
Inside the container, e.g.:
```bash
ros2 launch wojtek_viz viz.launch.py            # RViz/PlotJuggler for the live robot
ros2 launch wojtek_viz sim.launch.py            # MuJoCo sim
```
`setup-net.sh` makes a normal internet cable "just work"; dock the robot by
cable with `nmcli con up wojtek-eth`. (Details: the "PC side" table in
[`deploy/rpi/README.md`](deploy/rpi/README.md).)

## RPi (robot) — quickstart

**Fresh RPi**, from nothing:

1. Flash the card (downloads + verifies the image, writes cloud-init with your
   SSH key auto-injected):
   ```bash
   ./deploy/rpi/flash-card.sh /dev/sdX      # check `lsblk` for the right device!
   ```
   Put the card in the Pi, dock it by cable (`nmcli con up wojtek-eth`), power on
   — it comes up reachable at `10.42.0.2`. (Image identity: [`deploy/rpi/IMAGE.md`](deploy/rpi/IMAGE.md).)
2. Put your Ubuntu Pro token in `.env` (copy `.env.example`) — needed once for
   the RT kernel.
3. From the PC:
   ```bash
   ./deploy.sh --provision      # ROS + RT kernel + RT tuning + network + build
   ```
   Idempotent — safe to re-run. It reboots the Pi once (RT kernel) and continues.

**Routine redeploy** (after editing code):
```bash
./deploy.sh                    # rsync src + colcon build on the Pi (fast)
```

## Run the robot

You work from **one container shell** — enter it once, drive everything from
there. The RPi stack is started **for the session** (not on boot — that keeps
the RPi free for other launches during dev), and always over its **RT systemd
service** (there's no non-RT way to run it):

```bash
./dev.sh                          # enter the container (once)
ros2 run wojtek_bringup robot     # start the RPi RT stack (over SSH) + RViz
```
Flags:
```
--dry-run       BENCH: launch on the RPi WITHOUT RT and no torque (testing only)
--sim           MuJoCo sim instead of the RPi (no hardware)
--no-viz        skip RViz;   --plotjuggler  also open PlotJuggler
--foxglove      foxglove_bridge instead of RViz (native Foxglove app on
                ws://localhost:8765 -- the fast path on macOS)
--web-console   browser operator console on http://localhost:8080 instead
                of the Qt window (no X11; works from a phone on the AP)
--gamepad       bluetooth Xbox pad teleop (left stick vx/yaw, right stick
                strafe, A arms, Y/B stand up / lie down, D-pad height);
                runs ON the RPi against the real robot (pair the pad with
                the robot), locally with --sim; add --no-console to
                replace the console entirely
```
The stack comes up DISARMED. **Arming is manual** (that's when torque reaches the
motors) — from another shell in the same container:
```bash
ros2 service call /wojtek/zero      std_srvs/srv/Trigger
ros2 service call /wojtek/stand_up  std_srvs/srv/Trigger
ros2 service call /wojtek/arm       std_srvs/srv/SetBool "{data: true}"
ros2 run teleop_twist_keyboard teleop_twist_keyboard      # drive
```
Ctrl-C in the `robot` shell tears down both sides (stops the RT service + viz).
Wind down gently first: disarm, then `ros2 service call /wojtek/lie_down
std_srvs/srv/Trigger`.

## Layout

| Path | What |
|---|---|
| `build.sh` / `dev.sh`      | PC: build image / enter container |
| `sim.sh`                  | PC: one-command full sim (web console + RViz on Linux / Foxglove on macOS) |
| `deploy/pc/setup-net.sh`  | PC: create the `wojtek-eth` link profile |
| `deploy.sh`               | host orchestrator: provision + build the RPi |
| `.env.example`            | template for secrets (Ubuntu Pro token) |
| `deploy/rpi/flash-card.sh`| flash a fresh card: image + cloud-init + SSH key |
| `deploy/rpi/`             | RPi provisioning: `install.sh`, network + service configs, `IMAGE.md`, `cloud-init/` |
| `deploy/wojtek-robot.service` | RPi systemd unit (RT control stack) |
| `src/`                    | ROS 2 packages (`wojtek_bringup`, `wojtek_policy`, `wojtek_viz`, hardware ifaces) |
| `docker/`                 | PC image + compose |
