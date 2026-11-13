# Steam Deck setup

The Deck runs the [browser panel](../../src/wojtek_deck/README.md) served by the
robot or PC simulation. It needs no ROS installation. Use the
[operator runbook](../../../docs/operations.md) for zeroing, stand/arm, controls
and shutdown; this page covers the Deck itself.

Commands prefixed `./ros/` run on the PC from the repository root.

## One-time prerequisites

On the Deck, switch to Desktop. The helpers expect its normal graphical user
session, Flatpak Chrome (`com.google.Chrome`), OpenSSH, `systemd-run --user` and
`xdotool`. Install Chrome through Discover if absent. A USB keyboard is useful;
Steam's on-screen keyboard disappears when Steam is stopped.

The supplied helpers use a user-level SSH server on port 2222. Set the PC's
public key in the Deck's `~/.ssh/authorized_keys` before installing them:

```bash
# PC: read the public key to copy, never the private key.
ssh-add -L
```

On the Deck in Konsole, create the directory and host key if needed, then append
the chosen public key to `authorized_keys`:

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
[ -f ~/.ssh/host_key ] || ssh-keygen -t ed25519 -f ~/.ssh/host_key -N ''
# Paste the PC public key as one line into ~/.ssh/authorized_keys.
chmod 600 ~/.ssh/authorized_keys
/usr/bin/sshd -D -e -p 2222 -h ~/.ssh/host_key \
  -o PidFile=/tmp/sshd.pid -o AuthorizedKeysFile=.ssh/authorized_keys &
```

Keep PC and Deck on the same network, then run on the PC:

```bash
./ros/deck.sh where
./ros/deck.sh install
```

This installs the desktop icons and scripts. `DECK_HOST` and `DECK_SSH_PORT` in
`ros/.env` override discovery; see `ros/.env.example`. The user SSH server exits
on reboot, so tap **Deck SSH on** after returning to Desktop.

## Start a session

1. Join the robot's `wojtek-link` AP, or the same LAN as the simulation PC.
2. Tap **Deck SSH on** if the PC helpers will be used.
3. Tap **Steam off (pad)**. The documented path releases Steam's controller
   capture so the browser receives the raw Deck pad.
4. Start the robot through the operator runbook, or start PC simulation:

   ```bash
   ./ros/sim.sh --foxglove telemetry:=true
   ```

5. Tap **Wojtek Panel**. **Wojtek GO** combines stopping Steam and opening the
   panel when the server is already running.

The icon prefers the robot, then the PC URL saved by the helper, then another
panel found on the current networks. The notification shows what it selected.
To select a particular target from the PC:

```bash
./ros/deck.sh panel                         # this PC's simulation
./ros/deck.sh panel 'http://<host>:8090/'    # explicit target
./ros/deck.sh reload
./ros/deck.sh stop
```

The launcher closes existing Chrome processes on the Deck and uses a separate
panel browser profile. Its HTTP-origin/WebGPU flags enable the controller and
GPU detector on the chosen panel origin; use it for the operator network.

## Check the panel before driving

| Indicator | Meaning |
|---|---|
| `LINK` | Command WebSocket connected |
| `CAM n` | Camera frames arriving at the displayed rate |
| `PAD` | Browser sees the controller; press a button after focusing the page |
| `BRIDGE` | Telemetry connected, when `?telemetry=on` is selected |
| `DET` | Browser detector is running, with backend and timing |

The camera and command connection work without Foxglove. Charts require a
bridge on the same host at port 8765, unless `?bridge=ws://host:port` overrides
it. System charts also require `telemetry:=true`. The installed robot drop-in
enables telemetry and the Foxglove bridge. A manual launch requires
`telemetry:=true foxglove:=true`.

Controls differ from the ROS Xbox driver: Deck LB/RB change height, the right
trigger increases speed, and the D-pad plays tricks. See the full
[control comparison](../../../docs/operations.md#pad-and-steam-deck). The gateway
zeros velocity after 0.5 s without fresh drive input; this does not remove motor
torque. Use one driver at a time and cancel autonomy before manual takeover.

## Robot-side installation

`./ros/deploy.sh` builds the Deck package through `wojtek_bringup` dependencies,
installs its ROS/Python dependencies through rosdep, and fetches/syncs detector
assets. The package includes the `compressed_image_transport` dependency; a
manual wheel/deb copying procedure is not the normal deployment path.

The installed service starts `deck:=true deck_camera:=true`. The camera uses
640×480 colour at 30 fps and raw depth at 424×240 at 15 fps by default. The gateway
reads the camera's JPEG stream. Do not start `perception:=true` beside it: both
would open the same camera. Camera/gateway tasks run on cores 0–1, away from the
control and policy cores.

For an older installation, redeploy while following the robot's maintenance
procedure. Check the effective service and package on the robot:

```bash
systemctl cat wojtek-robot.service
source /opt/ros/jazzy/setup.bash
source ~/wojtek_ws/install/setup.bash
ros2 pkg prefix wojtek_deck
ros2 topic info /camera/camera/color/image_raw/compressed
ss -ltn
```

## Recovery

| Symptom | Action |
|---|---|
| Helper cannot find Deck | Tap **Deck SSH on**, confirm same network, then `./ros/deck.sh where`. |
| Panel opens the wrong machine | Pass the intended URL to `./ros/deck.sh panel`. |
| `PAD` grey | Stop Steam, focus page and press a pad button. |
| `CAM` grey | Inspect camera/gateway logs and compressed image topic on the server. |
| Charts empty | Check query flag, Foxglove bridge and telemetry sources. |
| Page frozen | Tap **RELOAD** or run `./ros/deck.sh reload`. |
| Robot stutters | Stop other drive sources; inspect `/cmd_vel` publishers. |
| Drive link lost | Follow [lost-link recovery](../../../docs/troubleshooting.md#recover-after-a-lost-drive-link). |

**FULL** enters/leaves fullscreen. The panel's restart button requires a
1.5-second hold and a folded-pose check; restarting assumes the physical robot
is folded. It is described in the recovery guide. Do not restart a standing
robot to clear an unexplained fault.

Implementation: `deck.sh` runs on the PC; `deck_link.sh`, `deck_panel.sh`,
`panel_ctl.sh` and desktop icons run on the Deck. The page and command gateway
are the `wojtek_deck` ROS package.
