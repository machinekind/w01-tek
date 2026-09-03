# wojtek_deck — the deck panel

A browser cockpit for a handheld on the robot's wifi (a Steam Deck is the
target, any laptop or phone works), and the one robot-side process it needs.

```
handheld (browser)                          robot (RPi)
  page + charts + pad  --ws /ws-->            deck_gateway   --> /cmd_vel, services
                       <--mjpg /stream.mjpg-- deck_gateway   <-- camera colour
                       <--ws :8765----------- foxglove_bridge <-- every topic
  detector (in the page)                      deck_gateway   <-- /det/ assets
```

Three links, three jobs:

- **Commands** go through `deck_gateway` (`/ws`, JSON). The dead-man lives
  in the gateway, on the robot: sticks arrive as normalized frames, and when
  they stop for 0.5 s the gateway zeroes `/cmd_vel` for two seconds and then
  goes silent. This is the point of having a robot-side process at all.
  `policy_node` latches the last command it saw, so a dead-man on the far
  side of a wifi link would protect nothing.
- **Camera** is the gateway's MJPEG stream (`/stream.mjpg`), shown in a plain
  `<img>`. The detector reads its frames out of that same image rather than
  opening a second stream, so detection costs the wifi nothing.
- **Charts** read `foxglove_bridge` directly (`bridge.js` + `cdr.js`, a small
  ros2msg/CDR decoder, no library). Nothing on the robot changes to add a
  chart: subscribe to the topic in `deck.js`.

Everything is served from the robot; the page loads no fonts or scripts
from the internet, because the robot's access point has none.

## Run

Simulation (the gateway is on by default there):

```bash
./ros/sim.sh                                   # or inside ./ros/dev.sh:
ros2 launch wojtek_pc sim.launch.py            # deck:=true is the default;
                                               # telemetry:=true fills the systems panel
ros2 launch wojtek_pc viz.launch.py foxglove:=true rviz:=false   # the charts' source
```

Open <http://localhost:8090>. A handheld on the same LAN uses the machine's
address instead of `localhost`; the page finds the bridge on the same host,
port 8765 (`?bridge=ws://host:port` overrides).

Robot: `robot.launch.py deck:=true deck_cpus:=0,1`, plus `foxglove:=true
telemetry:=true` for the charts (the RPi service already passes those two;
the deck itself stays opt-in there). The handheld joins the robot's access
point and opens `http://10.42.0.2:8090`.

Docker note: the dev image needs a rebuild once for the new package's
dependencies (`python3-aiohttp`): `docker compose build` in `ros/docker`.

## Controls

| input | action |
|---|---|
| left stick | forward/back, turn |
| right stick | strafe |
| A / Y / B | arm toggle / stand up / lie down |
| LB / RB | stance height −/+ 5 mm |
| D-pad up / left / right / down | paw wave / bow / sit / shake |
| W S A D Q E, arrows | drive from a keyboard (desk testing) |
| space | stop |

The pad is read in the browser (Gamepad API), the same mapping as
`wojtek_teleop/gamepad_teleop.py`. Buttons on the page cover the same
services for a touchscreen.

## Detection

The panel finds objects in the camera picture and draws a box around each
one: magenta for a person, cyan for everything else. It is on by default and
needs nothing running anywhere else.

It runs **in the page**, on the handheld. YOLOX-nano goes through
onnxruntime-web in a worker (`det_worker.js`), on the GPU through WebGPU
where the browser has it and on the CPU where it does not. The robot is not
asked for anything, which is the whole point: the RPi is already spending
its cores on the control loop, and the handheld is the machine with a GPU
sitting idle. The wifi does not notice either, because the frames come off
the camera image the page is already showing — there is no second stream.

Measured on a laptop at 416x416: about 10 ms a frame on WebGPU and 44 ms on
the CPU. The page asks for a frame at most every 66 ms, so both keep up.

`yolox.js` is the arithmetic on its own — reading the network's numbers into
boxes, throwing away the duplicates, scaling back to the camera's pixels —
so it can be tested from node. `yolox.json` holds the settings and the class
names.

### The assets

The network and the runtime are other people's binaries, tens of megabytes,
and they are not committed. Fetch them:

```bash
ros/src/wojtek_deck/fetch_assets.sh          # into ros/deck_assets/
```

That store works like the policy store: it sits next to the workspace's
`src/`, the script pins the hash of both downloads, and `deploy.sh` runs it
and rsyncs the result to the robot, which has no internet. The gateway
serves the store at `/det/`. Without it the panel is the panel it always
was, minus the boxes, and says so once in the log.

`yolox_nano.onnx` is YOLOX (Megvii, Apache-2.0); the rest is
onnxruntime-web (Microsoft, MIT). `LICENSES.txt` in the store says so too.

The gateway sends `Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp` on every response, which is
what a browser wants before it hands a page shared memory. Everything the
panel loads is same-origin, so nothing else had to change.

### Switches

| query | what it does |
|---|---|
| `?det=off` | no detection |
| `?det=cpu` / `?det=gpu` | pin the backend (default: try the GPU, fall back to the CPU) |
| `?det=ws://host:port` | boxes from a detector in another process, below |
| `?detsrc=<url>` | a still image in the shard instead of the camera |

`?detsrc=` is for testing the detector without pointing a robot at
something interesting: drop a picture in the asset store and open
`?detsrc=/det/<name>`. It has to be same-origin — the page is cross-origin
isolated, so a picture from elsewhere will not load at all.

The GPU path gets a deadline: if the first frame has not come back within
8 s the worker is thrown away and started again on the CPU, and the log
says so. That covers a browser whose WebGPU takes the session and then
never answers. The first WebGPU run in a fresh browser profile can also be
slow while the shaders compile once; after that it is quick.

### A detector somewhere else

`?det=ws://...` puts the page back on an outside detector, kept as an
escape hatch for anything the in-page one cannot do (a bigger network on a
laptop, a tracker with its own state). It sends one JSON frame per pass,
coordinates in pixels of the frame it looked at, and reads those frames from
`http://<robot>:8090/stream.mjpg` itself:

```json
{"t": "det", "w": 640, "h": 360,
 "boxes": [{"x": 10, "y": 20, "w": 100, "h": 200, "label": "person", "p": 0.91}]}
```

## Tests

```bash
pytest ros/src/wojtek_deck/test          # the drive gate (dead-man), no ROS
node --test ros/src/wojtek_deck/web/test # the CDR decoder and the YOLOX maths
```

## The look

The panel follows the Machinekind design system, on paper, the way the
system's own pages do: white ground, ink for text at three strengths, and
regions told apart by 1px hairlines rather than boxes. No frames around
blocks, no shadows, no glow, no gradients. There is one accent, brand red
`#bd3e3e`, and it marks the thing worth looking at: the joint working
hardest, a refused command, the dead-man word, the armed button. The
camera picture is the one dark thing on the page, so the text drawn over
it takes the system's on-dark whites and the soft red `#d86a6a`, the red
the system allows on ink.

Three faces with three jobs, served from the robot (`web/fonts/`,
fontsource 5.3.0 builds, SIL Open Font License; latin-ext is in, it
carries the Polish letters): Big Shoulders Display carries the WOJTEK
title and the mode word, IBM Plex Sans carries words a person reads, IBM
Plex Mono carries measurement: rates, angles, counts, the clock, the log.

`web/mark.svg` and `web/favicon.svg` are the Machinekind mark from the
brand kit: the mark in its red variant, the primary one on paper, and the
favicon on its red field because at 16 px the knot needs it.
