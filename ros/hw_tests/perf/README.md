# Robot computer profiling (`perf.sh`)

Measures what the RPi does while running Wojtek's stack: control-loop and
policy rates and their worst gaps, policy inference time, per-core and
per-thread CPU, interrupts on the isolated cores, temperature and
throttling, and the camera stream as a viewer on the PC receives it.
Every run is one JSON file under `results/` (gitignored), so a change can
be measured before and after with the same command.

The stack under test is the robot's own: `perf.sh up` reads the launch
arguments of the RPi's `wojtek-robot.service` and starts the same launch
as a transient unit, with three changes: `mock_hw:=true` (the MD80 drives
and the IMU replaced by ros2_control's GenericSystem, so the motors can
stay unpowered), `telemetry:=true` (for `/wojtek/policy_timing`) and
`bag:=false` (the service's recorder would write ~200 MB/min to the card
during every window; pass `bag:=true` to measure it). The boot service is
stopped while a perf stack runs and started again by `down` if it had
been running. The perf unit conflicts with the boot service in systemd,
so starting one stops the other: the real drivers can never come up next
to a running mock stack, and the mock stack's panel has no restart button.
Nothing arms the robot.

What the mock leaves out: the CANdle's SPI traffic and the blocking I2C
IMU read. Numbers measured with it are the floor of the real load. `MOCK=0`
runs the real drivers (motors powered, a person at the robot).

## Use

From the repository root, with the RPi reachable (`RPI_HOST`, default
`rpi@10.42.0.2`, over the cable; set it, `REMOTE_WS` and `DECK_PORT` in
`ros/.env`, the file `deploy.sh` reads, see `ros/.env.example`):

```bash
./ros/hw_tests/perf/perf.sh run stack 30      # control stack only, 30 s window
./ros/hw_tests/perf/perf.sh run camera 30     # + colour camera + panel, PC as the viewer
./ros/hw_tests/perf/perf.sh down              # stop it, restore the boot service

# Finer steps: bring a scenario up once (this syncs probe.py), probe it
# several times.
./ros/hw_tests/perf/perf.sh up camera deck_camera_profile:=424x240x15
./ros/hw_tests/perf/perf.sh probe small-camera 30 --viewer
./ros/hw_tests/perf/perf.sh probe depth 30 --topic /camera/camera/depth/image_rect_raw=sensor_msgs/msg/Image
./ros/hw_tests/perf/perf.sh logs

# Before / after. The column names carry the stamp and the label.
python3 ros/hw_tests/perf/compare.py ros/hw_tests/perf/results/A.json ros/hw_tests/perf/results/B.json
```

Extra arguments to `up`/`run` are launch arguments and override the
service's own (`deck_stream_hz:=10`, `policy_cpus:=1`, ...). Each scenario
names the extras it turns off as well as on (`stack` is
`deck:=false deck_camera:=false deck_camera_depth:=false foxglove:=false`),
so the ladder reads the same whatever the service itself runs.

A changed package reaches the RPi with `perf.sh down` followed by
`ros/deploy.sh` (deploy restarts the boot service, which stops a running
perf stack), or, while iterating on one package, an rsync of it into
`~/wojtek_ws/src` plus `colcon build --packages-select <pkg>` there; Python
and launch files are symlink-installed and need no rebuild.

## Files

| File | Runs on | Does |
|---|---|---|
| `perf.sh` | PC | scenarios, the transient unit, collecting results |
| `probe.py` | RPi | one measurement window -> JSON (subscribes only, pinned to core 0) |
| `viewer.py` | PC | pulls the panel's MJPEG stream like the Deck would, times it from its first frame |
| `compare.py` | PC | the summary table, one or several results side by side |
