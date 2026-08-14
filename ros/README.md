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

Everything ROS-side on the PC happens **inside the container**. Enter it once,
work in the shell — the workspace is a lab you walk into, not a set of wrapper
scripts:

```bash
./build.sh      # build the image (once)
./dev.sh        # shell inside the container; ROS + the workspace already sourced
```

Then, from that shell:

```bash
ros2 launch wojtek_pc sim.launch.py      # the whole simulation session
```

That one launch brings up the plant (MuJoCo), the robot's own control stack
(ros2_control at 400 Hz, `real_io_node`, `policy_node`), the virtual D435,
RViz and the operator console. Ctrl-C tears the session down; the container
stays for the next one. Open a second `./dev.sh` shell for service calls,
`ros2 topic echo` and teleop while it runs.

Useful arguments (`ros2 launch wojtek_pc sim.launch.py --show-args` lists all):

```bash
hw:=mock            # no dynamics, just the node graph (fast, CI-friendly)
boot_pose:=folded   # start from the real boot/zeroing pose
console:=qt|none    # Qt window instead of the browser, or nothing
gamepad:=true       # bluetooth Xbox pad paired with THIS machine
camera:=false       # no virtual D435 (weak machines)
rviz:=false         # headless
```

The startup procedure is the robot's, on purpose — see
[`../docs/sim-test-contract.md`](../docs/sim-test-contract.md) for what a
simulation run does and does not prove.

`./sim.sh` from the host collapses the above into one command: it starts the
container, sorts out X11/XQuartz, picks RViz (Linux/X11) or Foxglove (macOS,
headless) for the platform, and runs the session. Every `name:=value` argument
passes straight through to `sim.launch.py`, so the launch vocabulary above is
the only one:

```bash
./sim.sh                          # sim + viz + web console, Ctrl-C tears down
./sim.sh hw:=mock console:=none   # any launch argument passes through
./sim.sh --rviz | --foxglove      # override the platform viz default
```

The operator console (drive pad, arm/pose buttons, jog, telemetry) is the
**web console** by default — open <http://localhost:8080> in any browser (a
phone on the robot's AP works too). Drive commands are dead-man guarded:
if the page goes silent mid-drive, `/cmd_vel` is zeroed.

**Text commands (the VLM contract, #92)**: the web console also shows the
robot's colour camera and a `forward / left / right / stop` command panel —
the browser is a human dry-run of the future VLM, which will watch
`/camera/camera/color/image_raw` and publish `/wojtek/nav_command` and
nothing else. The panel drives only through that contract, via the
`text_commander` bridge node (`wojtek_teleop`), which `sim.launch.py`
starts automatically — safe as a resident, it publishes nothing until
commanded. The CLI works too:

```bash
ros2 topic pub -1 /wojtek/nav_command std_msgs/String "data: forward"
```

Speeds are parameters (`v_forward` 0.3 m/s, `w_turn` 0.5 rad/s) and a 2 s
dead-man (`command_timeout`) stops the robot when commands stop coming —
keep clicking (or talking) to keep walking.

3D visualization: `sim.launch.py` opens RViz over X11 (`./dev.sh` sets up the
access). On macOS GL-heavy RViz over X11 is slow, so run the launch with
`rviz:=false` and start `foxglove_bridge` from `viz.launch.py foxglove:=true`
in a second shell — then open the native
[Foxglove](https://foxglove.dev/download) app and connect to
`ws://localhost:8765` (add a 3D panel). The web console can live INSIDE
Foxglove as a panel — see
[`foxglove/wojtek-console-panel`](foxglove/wojtek-console-panel/README.md)
(`npm run local-install`, restart Foxglove, add the "Wojtek console" panel).

There is a ready-made view to import once, in
[`foxglove/layouts`](foxglove/layouts/README.md): temperature, throttling, CPU
per core, policy tick time, drive command, joints, IMU and the console in one
window.

**Bluetooth Xbox pad**: left stick = vx/yaw, right stick left-right = strafe,
**A** toggles arm, D-pad up/down steps the standing height; on the `joy`
paths **Y**/**B** additionally trigger the stand-up / lie-down ramps (browser
pads keep those on the console buttons). Driving is always live (no
drive-enable gate); a dead-man zeroes `/cmd_vel` if the pad drops off. Two
paths, same drive mapping:

- The **web console reads a pad in the browser** (Gamepad API): pair the pad
  with whatever machine the browser runs on (macOS included), open the
  console page and press any pad button — the sticks take over the drive pad.
- `sim.launch.py gamepad:=true` runs the in-container `joy` driver +
  `gamepad_teleop` (from `wojtek_teleop`) alongside the console. Linux only:
  the container sees the pad through the `/dev` mount, Docker on macOS cannot
  pass input devices (use the browser pad path above there).
- On the real robot the pad can pair with the **RPi itself** — no PC in the
  loop: `robot.launch.py gamepad:=true` (the `wojtek_teleop` package is part
  of the RPi build; bluez/ERTM groundwork comes from `deploy/rpi/install.sh`,
  pairing itself is a one-time `bluetoothctl` scan/pair/trust/connect).

`console:=qt` switches to the original Qt operator console (an X11 app).
On macOS that needs a one-time XQuartz setup, and XQuartz has to be running
before you enter the container:

```bash
brew install --cask xquartz
defaults write org.xquartz.X11 nolisten_tcp -bool false   # allow TCP :6000
```

For the real robot you also need the link profile, once, on the host:

```bash
./deploy/pc/setup-net.sh   # create the wojtek-eth profile
```

and then, from the container shell, `ros2 launch wojtek_pc viz.launch.py`
(RViz/PlotJuggler against the live robot) or `ros2 run wojtek_bringup robot`
(see "Run the robot" below).

`wojtek_pc` was called `wojtek_viz` until 2026-08-06 (it carries the
simulator, not just visualization). A workspace built before the rename keeps
the old package in its overlay and shadows the new one; clear it once with
`rm -rf build/wojtek_viz install/wojtek_viz` before rebuilding.

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
./deploy.sh --policy <ref>     # ... and run this policy instead of the pin
```
`--policy` takes a Hugging Face reference (`org/name[@revision]`). It fetches
that policy, syncs it, and leaves the robot running it. A plain `./deploy.sh`
puts the pinned default back.

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

### Watching a run

The robot brings up its own Foxglove bridge on port 8765, so the native
Foxglove app can connect to the robot with nothing running on the PC. Turn it
off with `foxglove:=false`.

Two topics say how the run is going. `/wojtek/sysinfo` is the state of the
computer: CPU per core, memory, SoC temperature, the Raspberry Pi throttle
flags, free space where the bag is written, and wifi traffic.
`/wojtek/policy_timing` is per control tick: how long the policy step took and
how far apart the ticks actually landed. Both are recorded with everything
else, so a stutter can be matched against a hot or throttled Pi afterwards.
Import [`foxglove/layouts/robot-dashboard.json`](foxglove/layouts/README.md)
to see them plotted.

## Layout

| Path | What |
|---|---|
| `build.sh` / `dev.sh`      | PC: build image / enter container |
| `sim.sh`                  | PC: one-command sim session from the host (container + X11 + platform viz around `sim.launch.py`) |
| `deploy/pc/setup-net.sh`  | PC: create the `wojtek-eth` link profile |
| `deploy.sh`               | host orchestrator: provision + build the RPi |
| `.env.example`            | template for secrets (Ubuntu Pro token) |
| `deploy/rpi/flash-card.sh`| flash a fresh card: image + cloud-init + SSH key |
| `deploy/rpi/`             | RPi provisioning: `install.sh`, network + service configs, `IMAGE.md`, `cloud-init/` |
| `deploy/wojtek-robot.service` | RPi systemd unit (RT control stack) |
| `src/`                    | ROS 2 packages (`wojtek_bringup`, `wojtek_policy`, `wojtek_pc`, hardware ifaces) |
| `foxglove/`               | Foxglove extras: the console panel extension and the dashboard layout |
| `docker/`                 | PC image + compose |
