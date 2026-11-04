# Maintenance and recovery

Maintain the physical unit against its recorded assembly and calibration.
[Commissioning](commissioning.md) covers checks after component replacement;
[Operations](operations.md) covers normal startup and shutdown.

## Before each session

- Inspect connectors, strain relief, the CAN harness, foot pads and exposed
  wiring. Stop using a unit with visible damage until it is repaired.
- Check fastener retention, linkage freedom and interference throughout the
  intended motion. Use the mechanical design's tightening values; none are
  currently specified in this repository.
- Confirm the leg profile, policy and physical zero pose match the unit.
- Confirm the battery and isolation method against the unit's power record.
  Battery chemistry, charge settings, voltage thresholds and charger are not
  established by the current repository. Obtain them before charging or setting
  software thresholds.
- Check available storage before enabling recording, and verify that the
  desired topics are actually recorded.

Record operating hours and faults. Fixed replacement intervals for bearings,
foot pads and harnesses have not been established; an inspection log is the
starting evidence for choosing them.

## After a change

| Change | Repeat before operation |
|---|---|
| Leg set, drive mounting or encoder zero | Profile/policy check, manual signs, zero pose and powered commissioning |
| Drive/controller replacement | Identity, firmware/configuration, CAN ID/baud, signs and zero |
| CAN connector or wiring | Harness/termination inspection, exact-ID ping and full soak |
| IMU or nearby magnetic material | Mount/axis check, magnetometer fit and heading validation |
| Camera or bracket | Extrinsics measurement, floor-cloud check and navigation validation |
| Computer, OS/kernel or added camera load | Effective service review, timing/load check and recording verification |
| Model or policy | Compatibility check, fixed evaluation and export/runtime validation |

## Drive-link and motor-power recovery

The installed controller-manager watchdog stops the robot service after a
detected drive-link failure. If `wojtek-robot.service` is enabled, the watchdog can
restart it after drives have been silent for at least ten seconds and then answer
again. A restart assumes the configured boot pose and recomputes encoder offsets.

Before bench work or recovery that involves handling the robot, support or lay
it down and stop **both** `wojtek-cm-watchdog.service` and
`wojtek-robot.service`. Identify the power fault, inspect the harness and put the
robot in its correct folded pose before restarting. Do not use a power cycle as
an unobserved reset while the assembly is in an arbitrary pose.

Read the event history on the robot:

```bash
journalctl -u wojtek-robot.service -b --no-pager
journalctl -t wojtek-cm-watchdog -b --no-pager
systemctl cat wojtek-robot.service
```

The [watchdog implementation](../ros/deploy/rpi/wojtek-cm-watchdog.sh) defines
the current recovery sequence. An `active` systemd service alone does not prove
the controller manager, drive hardware and policy are healthy.

## Operating limits still to establish

The project has simulation command boxes, policy torque caps and measured
individual experiments. It does not yet have a complete, measured physical
envelope for battery voltage, motor/controller temperature, payload, runtime,
slope, ground surface or sustained speed.

Record these limits for each physical revision in the
[hardware evidence register](hardware-evidence.md). Distinguish a drive rating,
a configured limit, an observed peak and an accepted continuous operating limit.
Use the unit's confirmed component instructions for charging/storage; do not
derive those settings from the simulation or an older robot's battery.
