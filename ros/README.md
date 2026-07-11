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
| `deploy/pc/setup-net.sh`  | PC: create the `wojtek-eth` link profile |
| `deploy.sh`               | host orchestrator: provision + build the RPi |
| `.env.example`            | template for secrets (Ubuntu Pro token) |
| `deploy/rpi/flash-card.sh`| flash a fresh card: image + cloud-init + SSH key |
| `deploy/rpi/`             | RPi provisioning: `install.sh`, network + service configs, `IMAGE.md`, `cloud-init/` |
| `deploy/wojtek-robot.service` | RPi systemd unit (RT control stack) |
| `src/`                    | ROS 2 packages (`wojtek_bringup`, `wojtek_policy`, `wojtek_viz`, hardware ifaces) |
| `docker/`                 | PC image + compose |
