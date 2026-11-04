# Simulation test contract

A simulation run exercises ROS behavior and modeled dynamics. It does not certify
physical hardware. Use the [simulation guide](simulation.md) for setup and the
[operator runbook](operations.md) for the startup sequence.

The following checks are acceptance criteria, not a claim that every row has
passed on the current revision. Record commit, robot profile, policy reference,
scene, host, result and log/bag path for each validation run.

## Choose a plant

From a configured PC container:

```bash
ros2 launch wojtek_pc sim.launch.py hw:=mock rviz:=false
# Or, in a separate session after stopping the first:
ros2 launch wojtek_pc sim.launch.py hw:=mujoco
```

Both need a compatible policy. `mock` echoes commands as states; it cannot walk
or fall. `mujoco` supplies physics and the virtual camera. Both use the shared
controller graph, including zeroing, pose ramps and the policy-target gate.

## Acceptance checks

| Check | Expected result | Plant |
|---|---|---|
| Bring-up | Hardware component and position controller active; joint/IMU/magnetometer broadcasters active with IMU enabled; no unexplained errors | mock |
| Initial state | I/O adapter reports disarmed; no forwarded policy targets | mock |
| Zero | Succeeds only while disarmed, after joint states arrive and with no ramp; offset matches the declared boot pose | mock |
| Stand | Service reports ramp started; wait for `stand_up: ramp complete`; final pose approaches home | mock, then MuJoCo |
| Arm | Refused during a ramp and when too far from home; accepted after a valid stand | mock, then MuJoCo |
| Pose/trick exclusion | Stand, lie and tricks refuse while armed; zero refuses during a ramp | mock |
| Disarm | Policy targets stop being forwarded; this is not a torque-removal or ramp-cancel test | mock |
| Lie | After disarm, the ramp completes toward the profile's folded target | MuJoCo; see new-leg limit below |
| Command loss | Stop the sole `/cmd_vel` source; policy velocity command becomes zero after 0.5 s, with height retained | mock |
| Required-sensor loss | Remove a required observation; policy output stops after the 0.2 s sensor watchdog | mock |
| Pad/console | One selected source drives; release/disconnect produces a stop | MuJoCo |
| Navigation cancel | While moving, cancel drops the goal; late goals inside the 0.5 s hold-off do not resume it | MuJoCo |
| Walking | Selected policy stands and follows commands without falling for the declared test interval | MuJoCo |
| Camera | Depth/colour topics and CameraInfo arrive; image and TF timestamps permit projection | MuJoCo |
| TF | Look up every required frame from `odom`; merely listing frame names is insufficient | MuJoCo |
| Recording | `bag:=true` creates a readable bag containing the requested topics | either |
| Timing | Compare measured rates with config; record `/sim/rtf`, which should be near 1 for wall-time operation | MuJoCo |

Useful inspection commands:

```bash
ros2 control list_controllers
ros2 control list_hardware_components
ros2 topic hz /joint_states
ros2 topic hz /imu_sensor_broadcaster/imu
ros2 topic echo /sim/rtf
ros2 run tf2_ros tf2_echo odom base_link
```

Current configuration is controller manager 100 Hz, joint and IMU broadcasters
50 Hz, and magnetometer broadcaster 100 Hz. Policy frequency comes from the
loaded contract, normally 50 Hz. Subscriber timing measurements add overhead;
record the host and load rather than treating one reading as a guarantee.

## Known differences and limits

- Hardware arm tolerance is 0.15 rad; sim uses 0.35 rad to allow post-ramp sag.
  Validate the stricter gate separately before claiming hardware readiness.
- The new legs' folded hip target exceeds the exported model range. Primitive
  colliders also differ from the physical lying geometry. See
  [robot variants](../training/docs/robots.md); do not claim folded-pose fidelity
  from this model until those mismatches are resolved.
- `leg_odom:=false` uses simulator truth. Test navigation with `leg_odom:=true`
  to expose the odometry drift that hardware consumers will see.
- The current launch has no `/sim/reset` service. Stop and restart it. The old
  standalone simulator's reset interface is not the current contract.
- The camera mirrors `/sim/qpos` into its own model. Its timestamp is receive
  time, so transport/render latency is not an exact physical camera model.
- `dry_run:=true` tests command suppression. It does not reproduce electrical
  drive-disable behavior; inspect the selected plant's dynamics separately.

## Hardware-only validation

Simulation does not establish CAN reliability or termination, drive firmware
behavior, encoder zero accuracy, battery or power integrity, real-time jitter,
thermal limits, wireless discovery/recovery, sensor calibration, or measured
friction/backlash/compliance. Camera mount and floor-noise checks are also
physical measurements. The [hardware guide](hardware.md) tracks those checks.

A passing report must identify its tested scope. Keep older measurements as
historical evidence with their original profile, scene and commit; rerun the
relevant acceptance checks after changing a model, policy or control contract.
