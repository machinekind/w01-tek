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

The console tile points at `http://localhost:8080`, which is right for a
simulation. Watching the real robot from a PC, open the tile's settings and
change the URL to `http://<robot>:8080`.

## What the run has to publish

The system panels read `/wojtek/sysinfo` and `/wojtek/policy_timing`, both
from `wojtek_telemetry`. A run publishes them only with `telemetry:=true`,
and it opens the bridge on port 8765 only with `foxglove:=true`. The RPi
service passes both, so a service-driven run is ready to watch. A manual
`robot.launch.py` run needs them on the command line.

Connect Foxglove to the robot directly. A simulation session gets its bridge
from `viz.launch.py` in the dev container instead, on `ws://localhost:8765`.

A recording covers all topics, so the same layout works on a bag afterwards.
The service and `real.launch.py` record by default. A manual
`robot.launch.py` run needs `bag:=true`.
