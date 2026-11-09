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

## Latency report

`latency_report.py` measures delay at each stage of the feedback and command
path from one robot bag. It also reports the age of drive readings.
Diagnostics supply hardware read and write execution times. Measurements
use header stamps and timing fields from the robot's nodes. These values
use the Pi's clock. Recorder receive times include delivery and scheduling
delays at the recorder. The report excludes those times.

### Record

This run powers the robot with the real drivers. An operator must stay at
the robot. Follow the [operator runbook](../../../docs/operations.md).
The launch arguments below enable recording.
[ros/README.md](../../README.md) describes them under "Recording and live
diagnostics".

```text
telemetry:=true bag:=true bag_cpus:=0,1 foxglove:=false
```

`telemetry:=true` enables the policy's `/wojtek/policy_timing` publisher.
It also enables the `/wojtek/io_timing` publisher in `real_io`.
The bag records every topic. It includes `/dynamic_joint_states` and
`/diagnostics`. The Foxglove bridge competes for the same cores.
Keep it disabled during measurement.

Record one run of about 60 s while the robot stands armed.
Record another run of about 60 s while it walks. Each report then covers
one condition. Each launch starts a new bag under
`~/wojtek_bags/run_<timestamp>` on the Pi.

### Copy the bag to the PC

Set `RPI_HOST` in `ros/.env` as required by `deploy.sh`.
Run these commands on the PC from the repository root.

```bash
set -a; . ros/.env; set +a
ssh "$RPI_HOST" 'ls -t ~/wojtek_bags | head -3'
rsync -a "$RPI_HOST:wojtek_bags/run_<timestamp>" ./bags/
```

### Run

The script needs Python 3.10 or later and the `rosbags` Python package.
ROS is optional. Run these commands on the PC from the repository root.

```bash
python3 -m venv /tmp/rosbags-venv && /tmp/rosbags-venv/bin/pip install rosbags
/tmp/rosbags-venv/bin/python ros/hw_tests/perf/latency_report.py bags/run_<timestamp> \
    --json bags/run_<timestamp>.json
```

`--stale-s` sets the `response_age` threshold. The default is 0.02 s.
That spans two periods of the 100 Hz controller. A bag recorded by Jazzy
carries its message definitions. The script uses those definitions as
recorded. Bags without definitions use telemetry types from
`ros/src/wojtek_telemetry/msg` in the checkout. `--msg-dir` selects another
directory for those files.

### Read the table

Each stage row gives a sample count, p50, p95, p99 and max. Latencies are in
milliseconds. Stage statistics include only finite values.
The notes count NaN ages excluded from these statistics.

| Row | From | Meaning |
|---|---|---|
| `joints_age` | policy_timing | The age spans the joint state's header stamp to the start of the policy tick that uses it. |
| `joints_wait` | policy_timing | The wait spans the sample's arrival at the policy to the tick that uses it. |
| `joints_delivery` | policy_timing | The delivery delay equals `joints_age - joints_wait`. It spans the header stamp to arrival in the policy's callback. |
| `imu_age` | policy_timing | The age spans the IMU sample's stamp to tick start. It stays empty for a policy that omits the IMU. |
| `inference` | policy_timing | The duration covers the policy step. |
| `period` | policy_timing | The period measures time between ticks. It is 20 ms at 50 Hz. |
| `target_age` | io_timing | The age spans the joint_targets stamp to publication of the motor command by `real_io`. |
| `state_relay` | io_timing | The delay spans the joint state's stamp to its republication by `real_io`. |
| `pipeline` | both | The duration equals `joints_age + inference + target_age` for the same tick. |

The IO rows and `pipeline` use only samples with `command_sent` set to true.
The line under the table counts disarmed samples. It also counts armed
samples that sent no command. This can happen during dry_run. Targets with
non-finite values or missing joints also cause it. `pipeline` matches a
policy tick to its IO sample by exact stamp. A difference of 1 ns leaves
the sample unmatched. The join line gives the counts of matched and
unmatched samples.

`response_age` lists each joint's largest age. It also counts samples above
the threshold. `hardware components` shows hardware read and write
execution times from controller_manager. Each statistic has an average,
minimum, maximum and standard deviation. `controller manager` shows the
update loop's periodicity. The last table gives each topic's message count
and mean rate from header stamps.

### Measurement limits

- `pipeline` covers the software path from the joint state's header stamp
  to publication of the command by `real_io`. The controller manager picks
  up the command on a later cycle. That can take up to 10 ms at 100 Hz.
  CAN transmission, drive processing and mechanical motion follow.
  These delays lie outside `pipeline`. The drive data may already be old
  at the joint state's stamp. That earlier age also lies outside `pipeline`.
- `response_age` measures time since a drive's position, velocity or effort
  last changed. It approximates time since the drive last answered.
  A stationary joint can repeat the same reading over a healthy link.
  Its reported age can then grow. Interpret this value alongside motion.
- controller_manager accumulates statistics from startup. A capture of
  version 4.48.1 showed that minimum and maximum values persisted across
  messages. The report shows each hardware metric's last value in the bag.
  It also shows the worst average seen in the bag. A component at the manager's rate reports
  execution time only. The `controller manager` periodicity line covers
  its timing. The manager reports NaN before its first measurement.
  Unknown values appear as `n/a` in the table. They appear as `null` in
  JSON. The worst average includes only finite values.
- An older bag may lack timing fields, `/wojtek/io_timing` or `response_age`.
  The script reports the available data. Its notes identify missing data.
  An IoTiming message without `command_sent` leaves command delivery
  unknown. Targets received during dry_run can have `armed` set to true.
  The `target_age`, `state_relay` and `pipeline` stages then stay empty.
  The sample counts and armed counts remain available.

The tests use small bags with known stamps. They need `rosbags` and
`pytest`. Run them on the PC from the repository root.

```bash
python3 -m pytest ros/hw_tests/perf/test_latency_report.py
```

## Files

| File | Runs on | Does |
|---|---|---|
| `perf.sh` | PC | scenarios, the transient unit, collecting results |
| `probe.py` | RPi | one measurement window -> JSON (subscribes only, pinned to core 0) |
| `viewer.py` | PC | pulls the panel's MJPEG stream like the Deck would, times it from its first frame |
| `compare.py` | PC | the summary table, one or several results side by side |
| `latency_report.py` | PC | The script reports latency at each stage from a robot bag. |
| `test_latency_report.py` | PC | The suite tests the report with synthetic bags. |
