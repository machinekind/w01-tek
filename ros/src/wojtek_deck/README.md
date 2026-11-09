# Deck panel

A browser camera/control panel served by the robot or ROS simulation. The Steam
Deck is the main handheld target; a laptop or phone can use the same page.

For setup use the [Steam Deck guide](../../deploy/deck/README.md). For startup,
control mappings and stopping use the [operator runbook](../../../docs/operations.md).

## Components

```text
browser                              robot or simulation
  controls -------- ws /ws --------> deck_gateway ----> /cmd_vel and services
  camera <--------- /stream.mjpg --- deck_gateway <---- colour JPEG
  charts <--------- ws :8765 ------ foxglove_bridge <-- ROS topics
  detector <------- /det/ ---------- local model/runtime assets
```

The command watchdog lives in the gateway. After 0.5 s without fresh drive
input it publishes zero velocity for two seconds, then goes silent. The policy
also has a 0.5 s command timeout. Height is retained. These stop commanded
locomotion; they do not remove drive torque.

The gateway normally forwards camera JPEGs as MJPEG. Detection reads the same
browser image, so it does not request another camera stream. Charts connect to
Foxglove directly. Fonts, scripts and detector assets are served locally for
use on an AP without internet.

The graph above shows the default direct command path. Shared launches with
`command_mux:=true` route Deck drive commands to `/wojtek/cmd_vel/deck`; manual
takeover cancels navigation. See [command ownership](../../../docs/command-ownership.md).

## Run

From the repository root:

```bash
./ros/sim.sh --foxglove telemetry:=true
```

Open `http://localhost:8090`, or `http://<PC-address>:8090` from the handheld.
The sim enables `deck:=true` by default. For a direct container launch, the
Foxglove bridge can be started separately with:

```bash
ros2 launch wojtek_pc viz.launch.py foxglove:=true rviz:=false
```

Run only one bridge per port. Charts use `?telemetry=on`; the bridge defaults to
the page host on port 8765, with `?bridge=ws://host:port` as an override.

The physical robot's installed service drop-in enables Deck and camera, with
`deck_cpus:=0,1`. It also enables telemetry and the Foxglove bridge for the
charts. `deploy.sh` builds the package,
resolves dependencies and syncs detector assets.

## Controls and state

| Input | Action |
|---|---|
| Left stick | Forward/back and yaw |
| Right stick horizontal | Strafe |
| Right trigger | Increase drive range from half to the full policy command box |
| A / Y / B | Arm toggle / stand / lie |
| LB / RB | Height −/+ 5 mm |
| D-pad up / left / right / down | Paw / bow / sit / shake |
| W S A D Q E, arrows | Keyboard drive |
| Shift | Keyboard speed increase |
| Space | Stop keyboard drive |

The page buttons call the same services. Stand, lie and tricks are refused
while armed. Arm indicators reflect service responses; they are not a separate
hardware-state stream. The ROS Xbox driver uses different height/speed buttons;
see the operator guide before switching devices.

Only one source may drive through the current direct command path. An idle
robot-paired pad becomes silent after its stop burst; two active drivers still
compete. Cancel the VLM task before manual takeover. Disarm alone does not cancel
a task or an active pose/trick ramp.

The restart button is a 1.5 s hold. The gateway checks fresh activation-relative
joints near zero before restarting the configured systemd unit. This assumes
folded boot calibration; verify the physical folded pose first. The page retries
its connection after the service and gateway restart.

## Detection

YOLOX-nano runs through onnxruntime-web in a worker on the handheld, using WebGPU
when available and CPU otherwise. On a measured laptop at 416×416 it took about
10 ms/frame on WebGPU and 44 ms/frame on CPU; those figures are not a Deck
performance guarantee. The page requests at most one frame per 66 ms.

Fetch the pinned model/runtime assets on an internet-connected PC:

```bash
./ros/src/wojtek_deck/fetch_assets.sh
```

They go to `ros/deck_assets/`, mounted into the PC container and synced by
`deploy.sh`. Download hashes are checked. Missing assets disable detection but
leave the panel usable. `LICENSES.txt` in the store identifies YOLOX and runtime
licenses. The first GPU frame has an 8 s deadline before CPU fallback.

| Query | Behavior |
|---|---|
| `?det=off` | Disable detection |
| `?det=cpu` / `?det=gpu` | Select backend |
| `?detsrc=/det/<image>` | Use a same-origin still image for detector testing |
| `?det=ws://host:port` | Receive boxes from an external detector |

An external detector reads `http://<server>:8090/stream.mjpg` and sends one JSON
frame per result. Coordinates are pixels in the image it processed:

```json
{"t":"det","w":640,"h":480,"boxes":[{"x":10,"y":20,"w":100,"h":200,"label":"person","p":0.91}]}
```

The gateway sends cross-origin isolation headers. Keep detector assets/images
same-origin; arbitrary external images may not load under those headers.

## Tests and implementation

```bash
python3 -m pytest ros/src/wojtek_deck/test
node --test ros/src/wojtek_deck/web/test/*.test.js
```

The gateway handles services, camera forwarding and the drive watchdog.
`deck.js` maps input and renders the UI; `bridge.js`/`cdr.js` read ROS telemetry;
`attitude.js` applies the IMU mount; `yolox.js` decodes and filters detections.
The tests exercise these calculations and command gates without a robot.
