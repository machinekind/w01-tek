# The Steam Deck as the robot's controller

The Deck shows the robot's camera and sends it commands. It runs the
`wojtek_deck` panel in a browser. The page is served by the robot, or by the
simulation on the PC.

This file is the runbook. The background is at the bottom.

## First contact: the keys

Do this once per machine. Everything below assumes ssh already works.

The Deck is the awkward one. Its sshd only accepts keys, and it cannot be
started remotely, so the first key has to be put there by hand. That needs a
keyboard, and the Deck's on-screen keyboard belongs to Steam. So do this
while Steam is still running, or plug in a USB keyboard.

**1. Read the PC's public key.**

```bash
ssh-add -L | head -1                 # or: cat ~/.ssh/id_ed25519.pub
```

**2. On the Deck, open Konsole and type this.** Press STEAM then X for the
on-screen keyboard. Paste the line from step 1 in place of `<key>`.

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
ssh-keygen -t ed25519 -f ~/.ssh/host_key -N ''    # the sshd's own host key
echo '<key>' >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

**3. Install the helpers from the PC.** This is what puts the six icons on
the Deck's desktop, `Deck SSH on` among them.

```bash
./ros/deck.sh install
```

Until this runs, start the sshd from Konsole by hand:

```bash
/usr/bin/sshd -D -e -p 2222 -h ~/.ssh/host_key \
  -o PidFile=/tmp/sshd.pid -o AuthorizedKeysFile=.ssh/authorized_keys &
```

**4. The robot takes a key the ordinary way.** It answers on port 22 and
accepts a password, so one command does it.

```bash
ssh-copy-id rpi@10.42.0.2
```

## Bring it up from cold

Do these in order. Steps 1 and 2 need a finger on the Deck. The rest is
typed on the PC.

**1. Put the Deck on the desktop.** Press STEAM, then Power, then Switch to
Desktop. From the login screen that entry is missing, so log in to Steam
first.

**2. Tap `Deck SSH on`.** The icon is on the desktop. A notification gives
the Deck's address. Nothing on the PC can reach the Deck before this,
including the command that would start it. The sshd dies on every reboot, so
this step comes back every time.

**3. Tap `Steam off (pad)` if the operator will drive.** Steam keeps the
controller for itself and hands the browser a pad that sends nothing. The
notification names the pad the browser will get. Skip this step if the demo
is camera only. Closing Steam also takes away the Deck's on-screen keyboard.
`Wojtek GO` is this step and step 6 in one tap, for when the robot is
already up.

**4. Put the Deck and the PC on one network.** The robot's access point is
`wojtek-link`. Check from the PC:

```bash
./ros/deck.sh where          # prints the Deck's address
```

**5. Start what serves the page.**

On the robot nothing: the service starts the camera and the gateway with
the control stack (`deck:=true deck_camera:=true` in
`wojtek-robot.service`), so the page is up whenever the robot is, and
comes back after a restart. Check with `ss -ltn | grep 8090` on the
robot. The commands below are the same two nodes by hand, for a robot
whose service does not carry them, or for the panel without the control
stack:

```bash
ssh rpi@10.42.0.2
source /opt/ros/jazzy/setup.bash && source ~/wojtek_ws/install/setup.bash

setsid nohup taskset -c 0,1 ros2 run realsense2_camera realsense2_camera_node \
  --ros-args -p enable_depth:=false -p enable_color:=true \
  -p rgb_camera.color_profile:="640x480x15" -p pointcloud.enable:=false \
  -p align_depth.enable:=false -p enable_rgbd:=false -p enable_sync:=false \
  > ~/cam.log 2>&1 < /dev/null &

PYTHONPATH=$HOME/py_deps:$PYTHONPATH \
CYCLONEDDS_URI="file:///etc/cyclonedds-rpi.xml,<CycloneDDS><Domain><Internal><SocketReceiveBufferSize max=\"8MB\"/></Internal></Domain></CycloneDDS>" \
setsid nohup taskset -c 0,1 \
  ros2 run wojtek_deck deck_gateway \
  --ros-args -p port:=8090 \
  -p policy:=/home/rpi/policy \
  > ~/gateway.log 2>&1 < /dev/null &
```

Cores 0 and 1 are the only ones these may use. The control loop owns 2 and 3.
`py_deps` goes in front of the existing `PYTHONPATH`, never in place of it.
setup.bash put ROS's own python path there, and without it `ros2` dies
before the gateway starts (`No package metadata was found for ros2cli` in
`~/gateway.log`). `policy:=` is the reference the service runs with, so
the gateway drives inside the contract's command box.

The camera runs at 640x480, not the sensor's full 1280x720, and the
gateway asks for an 8 MB DDS receive buffer. Both are about the Pi's
budget, and the day that taught it (2026-09-12): at 1280x720 the camera
node and the gateway together left the Pi 2% idle, the control loop's
command stream to the MD80 drives got gaps, and the drives dropped to
idle with nothing in any log. The legs went soft, the controller stayed
"active". At full size a frame is also 2.7 MB, about 1900 UDP fragments,
and with the default buffer one lost fragment discards the frame; under
load the gateway saw no frames at all. The panel's detector runs on the
Deck from the stream it gets, so it loses nothing at 640x480.

If the legs go soft with a live controller, check `top` on the robot
before blaming the drives: under 30% idle is the warning sign. After a
motor power cycle with the controller running, restart the service
(`sudo systemctl restart wojtek-robot.service`): the drives come back
idle and nothing re-enables them.

In the simulation, the gateway starts by itself:

```bash
./ros/sim.sh --foxglove telemetry:=true policy:=<policy reference>
```

**6. Open the panel.** Tap `Wojtek Panel` on the Deck. The icon finds
the machine serving the page by itself: the robot first, then the address
the PC last gave it, then any host on the Deck's networks with port 8090
open. A notification names what it picked. From the PC the same, or a
particular page:

```bash
./ros/deck.sh panel                                          # this PC's simulation
./ros/deck.sh panel http://10.42.0.2:8090/                   # the robot
./ros/deck.sh panel 'http://<pc>:8090/?telemetry=on'         # the simulation, with the bridge
```

## Put the panel on the robot, once

The robot needs this only the first time, or after it is reflashed. Check
whether it is already there:

```bash
ssh rpi@10.42.0.2 'ls -d ~/wojtek_ws/install/wojtek_deck ~/py_deps ~/wojtek_ws/deck_assets'
```

`./ros/deploy.sh` does not carry the package. It builds
`--packages-up-to wojtek_bringup`, and `wojtek_deck` is not among those.
Nor should the whole workspace be rebuilt while the control stack is
running, which is why this is done by hand.

**1. Send the package and build it.** Cores 0 and 1 only. The control loop
owns 2 and 3.

```bash
rsync -az ros/src/wojtek_deck/ rpi@10.42.0.2:wojtek_ws/src/wojtek_deck/
ssh rpi@10.42.0.2 'source /opt/ros/jazzy/setup.bash && cd ~/wojtek_ws &&
  nice -n 19 taskset -c 0,1 colcon build --packages-select wojtek_deck'
```

**2. Give it aiohttp.** The gateway needs it. The robot has no route to
pypi, so the wheels are carried over from the PC. They go in a directory of
their own, which leaves the system Python alone and makes the whole thing
undoable with one `rm -rf`.

```bash
pip3 download aiohttp typing_extensions --dest /tmp/whl \
  --platform manylinux2014_aarch64 --python-version 312 --only-binary=:all:
scp /tmp/whl/*.whl rpi@10.42.0.2:/tmp/
ssh rpi@10.42.0.2 'python3 -m pip install --no-index --find-links=/tmp --target ~/py_deps aiohttp'
```

`typing_extensions` is asked for by name because pip drops it when the
downloading machine runs Python 3.13 or newer.

**3. Send the detector's assets.**

```bash
./ros/src/wojtek_deck/fetch_assets.sh     # only if ros/deck_assets is empty
rsync -az ros/deck_assets/ rpi@10.42.0.2:wojtek_ws/deck_assets/
```

The store sits next to the workspace's `src/` and `install/`, which is
where the gateway looks when it is started by the service without an
`assets_dir` of its own. (The robot from before this step has it in
`~/deck_assets` with a symlink at `~/wojtek_ws/deck_assets`; either
works.) Now step 5 above works, and a reboot keeps all three.

**4. Give the camera node its JPEG plugin.** The gateway streams the
camera node's own compressed frames (`compressed_image_transport`,
encoding in C++, only while the gateway subscribes), so it receives 40 KB
a frame instead of a 0.9 MB raw image and encodes nothing. That is what
makes 30 fps fit the Pi. The robot has no route to the package server, so
the `.deb` comes over from the PC; its dependencies are already on the
robot.

```bash
# On the PC. The pool keeps only the current build: list it, take the
# arm64 file it shows, do not trust an older version string.
curl -s http://packages.ros.org/ros2/ubuntu/pool/main/r/ros-jazzy-compressed-image-transport/ \
  | grep -oE 'ros-jazzy-compressed-image-transport_[^"]+_arm64\.deb' | sort -u | tail -1
curl -s -o cit.deb "http://packages.ros.org/ros2/ubuntu/pool/main/r/ros-jazzy-compressed-image-transport/<that file>"
scp cit.deb rpi@10.42.0.2:/tmp/
ssh rpi@10.42.0.2 'sudo dpkg -i /tmp/cit.deb'
```

Without the plugin the gateway still works from the raw image: start it
with `compressed:=false` (and expect the camera at 15 fps to be the
limit; the raw path costs a third of a core in the gateway alone).

## Read the top band

| lamp | lit when |
|---|---|
| `LINK` | the command socket to the gateway is open |
| `BRIDGE` | telemetry is flowing, and only with `?telemetry=on` |
| `CAM n` | camera frames are arriving, n is the rate |
| `PAD` | the browser can see the controller |
| `DET` | the detector is running, with its backend and frame time |

`FULL` fills the screen and gives it back. `RELOAD` loads the page again.
The Deck has no keyboard, so these two are the only way to do either.

## Drive it

| input | action |
|---|---|
| left stick | forward, back, turn, at half the trained range |
| right trigger | the other half, in proportion to the pull (the `SPD` readout shows the current range) |
| right stick | strafe |
| A | arm and disarm |
| Y | stand up |
| B | lie down |
| LB, RB | stance height by 5 mm |
| D-pad up, left, right, down | paw wave, bow, sit, shake |

The buttons along the bottom of the page do the same things with a finger.

The robot stops when the sticks go quiet for half a second. The gateway
holds that timer, so a dropped wifi link stops the robot rather than
latching the last command.

The `restart` button on the page restarts the robot's control stack
(`wojtek-robot.service`). Hold it for a second and a half; a tap does
nothing. The gateway refuses unless the robot is lying, because the stack
assumes the folded pose when it starts. It is the button for the motor
power cycle: switch the motors off and on under a running controller and
the drives come back idle, keep answering, and nothing re-enables them,
so the legs go soft with everything reporting fine. Lie, then hold
`restart`, then wait for `LINK` to settle and the stack to come back, about
30 s. On the robot the gateway is a node of that same launch, so the
restart takes it down with the stack: `LINK` drops and the camera image
freezes. The page reconnects on its own (it retries every second, and
closes a socket that has gone silent for 6 s) once the service is back,
and points the camera stream at the gateway afresh. The log line
`restart_stack: restart requested ...` is the last thing the gateway sends
before it goes; a later `restart_stack refused` means `sudo`/`systemctl`
turned the request down before anything stopped.

The robot's own Xbox pad can stay plugged in. Its teleop publishes only
while its sticks are deflected and goes quiet two seconds after they
return to centre, so an idle pad does not talk over the Deck. Two people
driving at once still fight; nothing arbitrates that.

Arming refuses while any joint sits more than 0.15 rad from the home pose.
The panel prints the refusal in its log.

## When it goes wrong

| what you see | what it is | what to do |
|---|---|---|
| `./ros/deck.sh` finds nothing | the Deck's sshd died with its session | tap `Deck SSH on` |
| the icon says "nothing serves the panel" | no gateway on port 8090 on any network the Deck is on | start the robot or the simulation, then tap `Panel RELOAD` |
| the panel opens on the wrong machine | the robot's gateway is up and wins over the PC | `./ros/deck.sh panel` from the PC, which names the machine |
| `ERR_CONNECTION_REFUSED` | the gateway is not running | check `~/gateway.log` on the robot |
| `PAD` stays grey | Steam is running | tap `Steam off (pad)`, then press a pad button on the page |
| `PAD` stays grey with Steam closed | the browser reveals a pad only after a press | press A |
| `CAM` stays grey | the camera node died | check `~/cam.log` on the robot |
| a dash in every instrument | the bridge is not running | leave `?telemetry=on` off |
| the panel covers the whole screen | it is in full screen | tap `FULL`, which now reads `WINDOW` |
| nothing on screen responds | the page is stuck | run `./ros/deck.sh reload` |
| the robot stutters while driving | two sources on `/cmd_vel` | `ros2 topic info -v /cmd_vel` on the robot; only one node may drive |

A reboot of the robot wipes `/tmp` and stops both processes. The installed
files live in `$HOME` and survive it. Start again from step 5.

## What this machine makes awkward

**No sudo.** The `deck` account's password was set once in 2022 and nobody
remembers it. Everything runs as the user: a user-level sshd, flatpak
Chrome, and `systemd-run --user` for anything that must outlive the shell
that started it.

**No fixed address.** The Deck takes what DHCP gives it, and the PC is
often on two networks at once. No address is written down in this tree.
`./ros/deck.sh` looks for the one host on any of the PC's networks answering
on the ssh port, and gives the Deck the PC address on the Deck's own
network. `DECK_HOST` in `ros/.env` skips that search while it answers. The
Deck, in turn, looks for the machine serving the panel each time the icon
is tapped; see the top of `deck_panel.sh` for the order.

**No keyboard.** A full-screen browser window cannot be left without one,
which is why the panel window has a frame and the page carries `FULL` and
`RELOAD`. The Deck's only on-screen keyboard belongs to Steam.

**Steam owns the controller.** With Steam running the browser gets a virtual
pad that sends no events. With Steam closed the kernel exposes the Deck
itself as `js0`, with the buttons in the driver's order. `deck.js` carries
both orders and picks one when the pad connects.

## The pieces

| file | runs on | what it is |
|---|---|---|
| `../../deck.sh` | PC | login, install, panel, stop, reload, steam, shot, run |
| `deck_link.sh` | Deck | ssh up, Steam off and on, screenshot |
| `deck_panel.sh` | Deck | finds the machine serving the panel, then opens Chrome on it with the flags it needs |
| `panel_ctl.sh` | Deck | start, stop, reload, from an icon or over ssh |
| `*.desktop` | Deck | the six icons |

`./ros/deck.sh install` puts all of it on the Deck. It also writes
`~/.config/wojtek/panel-url`, the PC's address as a hint for the panel
icon; the icon uses it only when it answers and the robot does not.

## Chrome's flags

`--enable-features=Vulkan,WebGPU --ignore-gpu-blocklist
--enable-unsafe-webgpu` all three together. The last one alone puts the
detector in SwiftShader, which burns six cores. None of them drops the
detector to the CPU at about 117 ms a frame against 36 on the GPU.

`--unsafely-treat-insecure-origin-as-secure` lets the page use the APIs it
needs over plain http. The robot has no certificate.

`--test-type` silences Chrome's warning bar about the flag above.
