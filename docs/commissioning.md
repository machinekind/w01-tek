# Commissioning a physical robot

Use this procedure after assembly, a leg change, drive replacement or a change
to encoder zero, IMU mount or camera mount. Routine driving starts in
[Operations](operations.md). Record results against one physical unit and its
revision; a successful simulation does not confirm a motor's physical direction.

Commands below are for the operator to run on the named machine. Bench work
begins with the robot supported or lying, motor power removed, and the local
physical power-isolation method identified. Disarm only closes the policy gate;
it does not remove motor torque.

## 1. Record the unit and stop automatic recovery

Record the leg revision, ROS profile, model, policy reference, computer, drive
firmware/configuration and calibration files. Use the
[hardware reference](hardware.md) and [evidence register](hardware-evidence.md).

On the robot, stop both services before bench tools take the bus:

```bash
sudo systemctl stop wojtek-cm-watchdog.service wojtek-robot.service
systemctl is-active wojtek-cm-watchdog.service wojtek-robot.service
```

Both should report `inactive`. A nonzero status from `is-active` is expected.
The watchdog can otherwise restart the stack after a motor power cycle if
`wojtek-robot.service` is enabled.
Check for separately launched ROS/controller processes before continuing.

Read the installed computer and software identity on the robot:

```bash
tr -d '\0' < /proc/device-tree/model
cat /etc/os-release
uname -r
systemctl cat wojtek-robot.service
```

The effective service includes drop-ins. Verify its `robot:=` and `boot_pose:=`
arguments and whether recording/telemetry are enabled.

## 2. Inspect wiring and test each bus

With power removed, inspect the harness, motor labels, connector retention,
isolation switch and termination locations against the unit's wiring record.
Unconfirmed power ratings or an unidentified isolation switch leave this step
open; software cannot establish them.

Restore the power required for readout and follow the
[CAN and IMU bench tests](../ros/hw_tests/README.md). Acceptance:

- CAN ping returns exactly IDs 10–12, 20–22, 30–32 and 40–42.
- The full 300-second soak completes with `PASS` and no missing drive.
- IMU chip IDs and stationary checks pass. Save the measured sample rate, gyro
  bias and acceleration norm, rather than only the word `PASS`.

These tools never enable the motors, but they do not prove another process has
disabled them. Keep exclusive ownership of the bus throughout the check.

## 3. Confirm encoder zeros and physical directions

Follow the [manual sign-check runbook](../ros/hw_tests/signcheck/README.md) in
`dry_run:=true`. Select the actual profile and a compatible policy. The bench
launch keeps drives disabled while reading encoders; it still configures drive
registers. This is different from disarming a normal launch.

Place the assembly in its defined folded pose before zeroing. Move each joint
by hand toward the standing pose and compare its sign with the profile's expected
sign. Record all twelve joints, not only the ones that look wrong. Resolve any
direction or zero discrepancy before a powered ramp.

For new legs, every lever must reach its folding stop and the body must rest on
the intended standoffs. The folded hip/model mismatch is documented in
[Hardware](hardware.md); do not infer the physical stop from the simulator.

## 4. Check IMU orientation and calibrate the magnetometer

Under the dry-run stack, inspect `/imu_sensor_broadcaster/imu`. Confirm that the
sensor's +y points forward and +x points right, as specified by the mount.
Check the body attitude in the operator display while gently changing body
orientation; raw sensor roll/pitch are not the body's roll/pitch.

The shipped [magnetometer calibration](../ros/src/wojtek_bringup/config/mag_calib.yaml)
is an identity placeholder. It does not represent a calibrated unit.
The calibration tool requires ROS, NumPy and the magnetometer topic. Run from
the repository root in an environment that can see the dry-run stack:

```bash
python3 ros/hw_tests/imu_i2c/mag_calib.py \
  --duration 60 --out /tmp/wojtek-mag-calib.yaml
```

Rotate the supported, torque-free assembly slowly through all orientations.
The tool requires at least 200 samples, normally at least seven octants, and a
plausible fitted field norm. It reports residual error; save that report and
verify heading on a separate capture. These fit checks are not a substitute for
an agreed heading-error limit.

An existing raw-field bag can be fitted without moving hardware:

```bash
python3 ros/hw_tests/imu_i2c/mag_calib.py \
  --bag /path/to/recorded-bag --out /tmp/wojtek-mag-calib.yaml
```

Associate the result with the unit, mount and capture conditions. Install the
reviewed values in `ros/src/wojtek_bringup/config/mag_calib.yaml`, rebuild/deploy,
then restart the stack to load them. The fit corrects static distortion;
current-dependent magnetic interference still needs verification under load.

## 5. Measure camera placement

The current camera transform is a design value: translation `(0.32, 0, 0.07)` m
from `base_link`, pitched 15° down. Measure the actual mount and check a level
floor in the depth cloud before relying on obstacle height.

The mount is represented in the
[URDF](../ros/src/wojtek_description/urdf/body.urdf.xacro),
[sim camera contract](../ros/src/wojtek_pc/wojtek_pc/camera_spec.py) and
[standalone perception extrinsics](../ros/src/wojtek_perception_bringup/config/extrinsics.yaml).
Keep them consistent and run the existing camera-mount checks after a change.
Use one RealSense driver; `deck_camera:=true` and `perception:=true` are mutually
exclusive on the robot.

## 6. Record the first powered movement

Agree the test fixture, physical stop method and allowed torque for this unit
before the first powered test. Choose limits deliberately; a generic low torque
can also fail to support the assembly. Do not change the policy's trained gains
without recording the difference.

Follow [Operations](operations.md): zero in the physical boot pose, stand,
inspect, arm, command a short movement, stop, disarm and lie down. Pose ramps
move the motors while the policy is disarmed. Record joint traces, controller
state, policy timing and observed behavior; repeat after any calibration change.

## Commissioning record

Keep one result per unit with: date, operator, physical revision, code/model
revision, policy pin, drive configuration, calibration files, test commands,
logs/bags, pass criteria and unresolved deviations. A unit is ready for routine
operation only when each relevant check has a result and the hardware owner has
set the operating limits. Re-enable automatic recovery only after returning the
robot to the documented boot pose and completing the handover to normal operation.
