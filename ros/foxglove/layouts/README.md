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

The robot runs its own bridge on port 8765 while the stack is up, so connect
Foxglove to the robot directly. In a simulation session the bridge runs in the
dev container instead, on `ws://localhost:8765`.

The system panels read `/wojtek/sysinfo` and `/wojtek/policy_timing`, both
from `wojtek_telemetry`. Every run records all topics, so the same layout
works on a recorded bag.
