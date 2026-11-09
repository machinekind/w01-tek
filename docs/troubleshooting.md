# Troubleshooting

Start with the symptom below. Use the [operator runbook](operations.md) for normal
startup/shutdown and the [architecture](ros-architecture.md) for interfaces.
Commands marked **robot** run over SSH on the RPi; other ROS commands run inside
`./ros/dev.sh`.

## Find the failing layer

| Symptom | Check | Next action |
|---|---|---|
| Empty policy reference | Repo-root `.env`, explicit `policy:=`, selected robot profile | Set keeper organization/access or use a compatible local export. Container paths differ from host paths. |
| Policy download denied | Credentials and access to the requested repo/revision | Resolve on the PC; the robot uses the synced offline store. |
| Policy rejected for wrong legs | Launch `robot:=` versus metadata `robot` | Select the matching profile/policy; do not bypass the check. |
| No ROS topics | Domain/RMW, link, `ROS_LOCALHOST_ONLY`, effective DDS configuration | Use domain 42/CycloneDDS and a reachable interface; localhost-only intentionally prevents robot discovery. |
| Arm refused | Service response, fresh joints, correct zero, completed stand | Fix the failed precondition. Hardware arm tolerance is 0.15 rad. |
| Robot moves the wrong joint/direction | Profile, joint map, physical zero and drive IDs | Stop powered testing and repeat the manual sign-check commissioning procedure. |
| Stuttering or conflicting motion | `ros2 topic info -v /cmd_vel` | Stop competing sources; cancel autonomy before manual control. With the opt-in [mux](command-ownership.md), verify it is the only `/cmd_vel` publisher and inspect its status. |
| Policy stops producing targets | Required sensor ages, policy log, CPU load | Restore the sensor/compute path; zero velocity and stale-sensor hold are different conditions. |
| Legs go soft while service says active | Controller process/component state and `Drive link lost` log | Follow lost-link recovery below; active systemd status alone does not prove active hardware. |
| Deck `PAD` stays grey | Steam controller capture, browser focus and pad press | Stop Steam with the helper, focus the page and press a button. |
| Deck camera absent | Camera node, compressed topic and gateway log | Use one camera driver; check JPEG plugin and USB/power before changing resolution. |
| Deck charts empty | `?telemetry=on`, bridge reachability, telemetry topics | The installed service drop-in enables telemetry and the bridge. A manual launch requires `telemetry:=true foxglove:=true`. |
| No bag after a session | Effective `bag:=` and recording process | Recording is opt-in; confirm a bag directory while the run is active. |
| VLM connection refused | Ollama readiness and tunnel; container's host address | Use the [VLM setup](../ros/src/wojtek_nav/README.md); remote loopback is not the PC's loopback. |
| Navigation task stalls | Camera/depth/TF, arm state, costmap and status reason | Inspect the named failure; a missing costmap is not evidence of clear space. |

## Inspect health

Inside the container, with a running stack:

```bash
ros2 node list
ros2 control list_controllers
ros2 control list_hardware_components
ros2 topic info -v /cmd_vel
ros2 topic hz /wojtek/joint_states_abs
ros2 topic hz /imu_sensor_broadcaster/imu
ros2 run tf2_ros tf2_echo odom base_link
```

The absolute-joint topic carries both actuated and inferred passive-joint
messages; its aggregate message rate is not the actuator sample rate. Inspect
`/joint_states` when measuring the latter. Current configured joint/IMU rates
are 50 Hz, controller manager 100 Hz, policy normally 50 Hz. On a loaded Pi,
Python CLI startup and subscriptions can themselves be expensive.

On the robot:

```bash
systemctl cat wojtek-robot.service
systemctl status wojtek-robot.service wojtek-cm-watchdog.service --no-pager
journalctl -u wojtek-robot.service -n 100 --no-pager
journalctl -t wojtek-cm-watchdog -n 50 --no-pager
top
```

For a planned diagnostic run, enable `telemetry:=true` and, if needed,
`bag:=true bag_cpus:=0,1` in the effective launch. Check CPU, temperature,
throttling, policy tick gaps and free space. Camera and logging load must fit the
actual board; do not assume results measured on another Pi model apply.

## Recover after a lost drive link

The provisioned CM watchdog stops the control service when the controller dies
or its hardware leaves the active state. It waits for a deliberate motor-power
cycle: drives silent for at least **10 seconds**, then answering again. It starts
the robot service only if that service is enabled. A brief supply dip should not
restart the stack with an unknown zero pose.

For a commissioned unit with the documented power control:

1. Remove motor power and put the robot in its profile's folded boot pose.
2. Keep motor power off for at least 10 seconds, then restore it.
3. Watch `journalctl -t wojtek-cm-watchdog -f` and the robot-service log.
4. If the robot service is disabled, restart it through the normal operator
   session. Check joints, zero and stand before arming again.

A manual service restart also assumes the configured boot pose. The Deck
**restart** button requires a long press and fresh joint states near folded;
its check uses activation-relative positions and assumes a folded boot. It is
not a substitute for verifying the physical pose. The browser reconnects when
the gateway returns.

If a drive, cable, zero or sensor has changed, return to
[commissioning](commissioning.md) instead of resuming the previous session.

## Read a recording

Robot recordings default to `~/wojtek_bags/run_<timestamp>`. PC-container
recordings live in the persistent `wojtek_bags` Docker volume mounted at
`/root/wojtek_bags`. List and inspect them where they were recorded:

```bash
ls -lt ~/wojtek_bags
ros2 bag info <bag-directory>
```

Playback republishes recorded topics, which can include commands. Use an
isolated ROS domain with no running robot control stack:

```bash
ROS_DOMAIN_ID=43 ros2 bag play <bag-directory>
```

Run analysis/visualization in that same isolated domain. This is message replay,
not a simulator driven by the recorded physical state.

## Record enough evidence to reproduce the problem

Capture the code commit, robot/profile, policy reference and metadata, effective
service unit, launch arguments, relevant journal excerpt, topic endpoint/rate
observations and bag path. Include whether the camera, detector, recording and
VLM were active. Keep private addresses, credentials and personal identifiers
out of a public issue or shared bundle.
