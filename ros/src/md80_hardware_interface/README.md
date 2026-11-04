# MD80 hardware interface

This ros2_control plugin connects Wojtek's twelve MD80 joint controllers through
CANdle. The normal robot launch uses the CANdle HAT over SPI and impedance-mode
position commands. Start with [Hardware](../../../docs/hardware.md) for wiring
and CAN IDs, then [Commissioning](../../../docs/commissioning.md).

## Position and torque semantics

The plugin saves raw encoder positions when activated. Its position state and
position command interfaces are relative to that activation pose. `real_io_node`
converts between those values and absolute URDF joint angles.

In the normal position path, the drive applies the configured impedance gains
and torque cap. These come from the loaded policy's deployment contract; the
launch can explicitly override `kp`, `kd` and `max_torque`. A policy with a
feed-forward torque head adds an effort interface. The launch adjusts the drive
cap to cover the trained PD and feed-forward terms.

Activation enables the drives and starts the communication loop. Deactivation
asks the drives to disable. `dry_run:=true` skips enabling the motors, but still
reads encoders and configures drive registers. The operator's arm/disarm service
belongs to `real_io_node`; disarm does not disable these drives or remove torque.
A drive can continue holding its last target. See [Operations](../../../docs/operations.md).

## Configuration

The normal values are supplied by
[`wojtek_ros2_control.urdf.xacro`](../wojtek_bringup/urdf/wojtek_ros2_control.urdf.xacro)
and the shared launch. Low-level plugin defaults differ where noted.

| Hardware parameter | Meaning |
|---|---|
| `bus` | `spi` for the HAT or `usb` for the dongle; normal launch is `spi`, plugin fallback is `usb` |
| `can_baud` | `1`, `2`, `5` or `8` Mbit/s; normal launch/default is `1`; must match all drives |
| `spi_device` | Optional SPI device; CANdle's default is `/dev/spidev0.0` |
| `usb_port` | Device path for the USB branch |
| `dry_run` | Keep drives disabled while reading and configuring them |
| `link_stall_cycles` | Identical raw state across every drive for this many cycles causes a read error; default 200, zero disables |
| `enable_stagger_ms` | Delay between enabling drives; default zero |

At the current 100 Hz controller rate, 200 stalled cycles are about two seconds.
The detector is skipped in dry-run and detects a complete silent stream, not
every possible individual-drive fault. `wojtek-cm-watchdog.service` handles the
subsequent service recovery; see [Maintenance](../../../docs/maintenance.md).

## Inspect and diagnose

Use the [standalone bus tests](../../hw_tests/README.md) with the ROS stack and
watchdog stopped. Check all IDs, their stored baud, harness termination and
power before changing software parameters. A controller reporting active is
not proof that the motors are producing torque.

For a supervised diagnostic launch, `MD80_SNAPSHOT=1` enables periodic logs of
raw position, target, torque and quick status. Remove it after diagnosis; retain
relevant logs with the unit record.

## Motor setup and replacement

The historical `scripts/flash_motors.py` is **not a supported provisioning
recipe**. It references an absent `config/AK80-9.cfg`, assumes one configuration
for all drives and does not validate each shell command. Do not use it to
commission replacement motors.

Before a replacement is powered under control, obtain the installed motor and
controller identities, firmware and known-good configuration from the hardware
owner. Preserve a backup, restore the matching configuration with the supported
drive tool, set the correct CAN ID/baud, then repeat the manual sign and zero
checks. The repository does not currently contain that complete configuration
pack. Track the missing inputs in [Hardware evidence](../../../docs/hardware-evidence.md).
