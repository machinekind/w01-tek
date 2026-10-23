# Foxglove layouts

`robot-dashboard.json` is the view for watching a normal run. It shows the
SoC temperature, the Raspberry Pi throttle flags, CPU use per core, how long
each policy tick took, the drive command, the measured joint angles, the IMU,
and the Wojtek console panel.

## Import it

In the Foxglove app open the layout menu in the top left, pick **Import from
file** and choose this file. It is saved as a personal layout from then on, so
you import it once per machine. Re-import after pulling a newer version.

Install the console extension first, otherwise the console tile comes up
empty. See [`../wojtek-console-panel`](../wojtek-console-panel/README.md).
The extension's panel is addressed by name in the layout, so if the tile still
says the panel is unknown, delete it and add **Wojtek console** by hand.
This is a Foxglove 2.x layout: in 3.x desktop the console tile cannot
render at all (see the end of this page), so use the `e2e` layouts there
and open the deck panel in a browser instead.

The console tile points at `http://localhost:8080`, which is right for a
simulation. Watching the real robot from a PC, open the tile's settings and
change the URL to `http://<robot>:8080`.

## What the run has to publish

The system panels read `/wojtek/sysinfo` and `/wojtek/policy_timing`, both
from `wojtek_telemetry`. A run publishes them only with `telemetry:=true`,
and it opens the bridge on port 8765 only with `foxglove:=true`. The base
unit (`ros/deploy/wojtek-robot.service`) passes both; the provisioning
drop-in (`ros/deploy/rpi/wojtek-robot-local.conf`) overrides `ExecStart`
without them, so a provisioned robot needs the two added there, or a
`perf.sh run full` session. A manual `robot.launch.py` run needs them on
the command line.

Connect Foxglove to the robot directly. A simulation session gets its bridge
from `viz.launch.py` in the dev container instead, on `ws://localhost:8765`.

A recording covers all topics, so the same layout works on a bag afterwards.
The service and `real.launch.py` record by default. A manual
`robot.launch.py` run needs `bag:=true`.

## `e2e.json`: the whole pipeline on one screen

Robot model (3D), the colour and depth streams, the joint targets the
policy sends next to the joint angles the drives report, the policy tick
timing, CPU per core, the velocity command and the controller-manager
diagnostics. Made for checking a run end to end (simulation:
`./ros/sim.sh telemetry:=true`, then `ws://localhost:8765`; without
`telemetry:=true` the timing and CPU tiles stay empty). The image tiles
read the raw topics the simulation publishes.

`e2e-robot.json` is the same view for the robot (`ws://10.42.0.2:8765`): the
colour tile reads the JPEG (`.../image_raw/compressed`). The robot's bridge
does not offer the raw colour image at all: one panel on it pulled ~19 MB/s
out of the Pi, saturated cores 0 and 1, and stretched the policy's output
gaps from 23 to 111 ms.

The depth tile maps millimetres 0..3000 through the turbo colour map:
Foxglove's default range for a 16-bit image is 0..10000, under which a
3 m scene (the camera clips at `clip_distance` 3.0) uses under a third of
the map and looks dim. Black pixels in it are holes (0 = no depth), normal
for a D435 at edges, shadows and shiny or dark surfaces.

No console tile in the e2e layouts: the Wojtek console panel is an iframe,
and Foxglove 3.x desktop serves every page under a Content-Security-Policy
without `frame-src`, so `default-src 'self'` blocks any `http://` iframe and
the tile stays empty. The diagnostics summary (controller manager, controllers,
hardware components) takes its place; open the deck panel in a browser at
`http://<robot>:8090`. (3.x also names an extension panel
`<displayName>.<panel name>`, so the tile in `robot-dashboard.json`, written
for 2.x as `machinekind.wojtek-console-panel.Wojtek console`, reports "Unknown
panel type" there.)

## `nav-sim.json`: the navigation session

The view for `sim.launch.py model_xml:=scene_nav.xml leg_odom:=true
nav:=true` (see `wojtek_nav/README.md`). The world itself is not a topic:
the simulator publishes only what the camera sees, so the 3D tile shows the
environment through `/wojtek/nav/points` (the decimated depth cloud) and
`/wojtek/nav/costmap` (the rolling 6 x 6 m window) around the robot model,
with the goto setpoint (`/wojtek/nav/goal`) and the resolved object point.
The TF axes are scaled down so they do not hide the cloud. Beside it: the
colour image the VLM sees, `/wojtek/vlm/annotated` with the model's point
drawn on it, the depth image, `/cmd_vel`, and the goto and brain status
strings. Import it the same way as the others; it needs no extension.
