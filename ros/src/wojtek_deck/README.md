# wojtek_deck — the deck panel

A browser cockpit for a handheld on the robot's wifi (a Steam Deck is the
target, any laptop or phone works), and the one robot-side process it needs.

```
handheld (browser)                          robot (RPi)
  page + charts + pad  --ws /ws-->            deck_gateway   --> /cmd_vel, services
                       <--mjpg /stream.mjpg-- deck_gateway   <-- camera colour
                       <--ws :8765----------- foxglove_bridge <-- every topic
  detector (optional)  --ws :8091--> page
```

Three links, three jobs:

- **Commands** go through `deck_gateway` (`/ws`, JSON). The dead-man lives
  in the gateway, on the robot: sticks arrive as normalized frames, and when
  they stop for 0.5 s the gateway zeroes `/cmd_vel` for two seconds and then
  goes silent. This is the point of having a robot-side process at all.
  `policy_node` latches the last command it saw, so a dead-man on the far
  side of a wifi link would protect nothing.
- **Camera** is the gateway's MJPEG stream (`/stream.mjpg`). The page shows it
  in a plain `<img>`; a detector on the handheld opens the same URL.
- **Charts** read `foxglove_bridge` directly (`bridge.js` + `cdr.js`, a small
  ros2msg/CDR decoder, no library). Nothing on the robot changes to add a
  chart: subscribe to the topic in `deck.js`.

Everything is served from the robot; the page loads no fonts or scripts
from the internet, because the robot's access point has none.

## Run

Simulation (the gateway is on by default there):

```bash
./ros/sim.sh                                   # or inside ./ros/dev.sh:
ros2 launch wojtek_pc sim.launch.py            # deck:=true is the default
ros2 launch wojtek_pc viz.launch.py foxglove:=true rviz:=false   # the charts' source
```

Open <http://localhost:8090>. A handheld on the same LAN uses the machine's
address instead of `localhost`; the page finds the bridge on the same host,
port 8765 (`?bridge=ws://host:port` overrides).

Robot: `robot.launch.py deck:=true deck_cpus:=0,1` (plus `foxglove:=true
telemetry:=true` from the telemetry bringup for the charts). The handheld
joins the robot's access point and opens `http://10.42.0.2:8090`.

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

## Detector protocol

A detector running on the handheld sends the page, over a local websocket
(`ws://localhost:8091`, `?det=` overrides), one JSON frame per detection
pass:

```json
{"t": "det", "w": 640, "h": 360,
 "boxes": [{"x": 10, "y": 20, "w": 100, "h": 200, "label": "person", "p": 0.91}]}
```

Coordinates are pixels of the frame the detector looked at; the page maps
them onto the displayed image. The detector reads frames from
`http://<robot>:8090/stream.mjpg`.

## Tests

```bash
pytest ros/src/wojtek_deck/test          # the drive gate (dead-man), no ROS
node --test ros/src/wojtek_deck/web/test # the CDR decoder
```

The panel is a head-up display: the camera fills the screen and the
symbology is drawn on it in one colour at one line weight, the way a
cockpit HUD works. A heading tape across the top, a pitch ladder with an
artificial horizon in the middle, a speed tape (commanded vx) on the left
and a stance-height tape on the right, three strip charts bottom-left
(series told apart by dash pattern, direct-labeled), readouts bottom-right,
the service buttons along the bottom edge. Amber marks a warning that is
on; the dead-man is hazard stripes around the whole screen and the one
thing allowed to shout.
