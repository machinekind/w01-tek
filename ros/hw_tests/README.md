# Hardware bench tests

Run these on the Raspberry Pi, outside Docker. They check the CANdle HAT and
Adafruit 5543 links before ROS commissioning. Wiring and joint IDs are in
[Hardware](../../docs/hardware.md); the full sequence is in
[Commissioning](../../docs/commissioning.md).

## Prepare the bench

Support or lay down the robot before stopping its control stack. Stop both
`wojtek-robot.service` and `wojtek-cm-watchdog.service`, and check that no manual
controller process owns the bus. The tests never enable motors; they do not
force a drive disabled if another process has already enabled it.

With power removed, fit the HAT and IMU. The IMU wiring is VIN→Pi pin 1 (3.3 V),
GND→pin 9, SDA→pin 3 (GPIO2), SCL→pin 5 (GPIO3). Confirm the HAT exposes these
pins. Enable the buses in `/boot/firmware/config.txt`, then reboot:

```ini
dtparam=spi=on
dtparam=i2c_arm=on,i2c_arm_baudrate=400000
```

On the Pi:

```bash
sudo apt install i2c-tools python3-smbus2
sudo usermod -aG spi,i2c "$USER"
```

Log in again after changing groups. Check devices before starting a driver:

```bash
ls /dev/spidev0.0 /dev/i2c-1
i2cdetect -y 1
```

Expect `0x6A` (LSM6DS3TR-C) and `0x1C` (LIS3MDL). SDO-high alternatives are
`0x6B` and `0x1E`. Address and chip identity are separate checks.

## CAN ping and soak

The current robot launch uses SPI at **1 Mbit/s**. Always pass `--baud 1`:
the standalone test binary retains an older 8M default. Both SPI and USB support
an explicit baud; it must match the drives' stored configuration.

`deploy.sh` does not copy `hw_tests/`. Copy this directory to the Pi or use a
checkout there. From `ros/hw_tests/candle_hat`:

```bash
cmake -B build -DCANDLE_DIR="$HOME/wojtek_ws/src/md80_hardware_interface/3rd_party/candle"
cmake --build build -j2
./build/candle_bus_test ping --bus spi --baud 1
./build/candle_bus_test soak --bus spi --baud 1 --seconds 300 --expect 12
```

`CANDLE_DIR` must name the CANdle checkout used by the deployed workspace.
`ping` should list exactly 10–12, 20–22, 30–32 and 40–42. Check that list manually:
`--expect 12` checks the count, not the expected set of IDs.

The soak passes if all twelve discovered drives can be added and no one-second
window reports a communication rate below 50 Hz. Keep the minimum rate and full
output in the unit's test record. A soak confirms this unloaded link test; it
does not establish performance under camera, ROS or motor load.

## IMU stationary check

From `ros/hw_tests/imu_i2c`, with the assembly stationary:

```bash
python3 imu_i2c_test.py --seconds 10
```

The script configures accel/gyro at 104 Hz (±4 g, ±500 deg/s) and magnetometer at
80 Hz (±4 gauss). Its acceptance checks are:

| Check | Required result |
|---|---|
| Accel/gyro chip identity | `0x6A` |
| Magnetometer chip identity | `0x3D` |
| Effective polling rate | Above 90 Hz |
| Mean acceleration norm | Within 0.5 m/s² of 9.81 m/s² |
| Mean gyro bias | Each axis below 3 deg/s in magnitude |
| Mean magnetic-field norm | Between 15 and 100 µT |

The printed polling rate is not the rate of independent sensor measurements.
A final `OVERALL: PASS` checks communication and these stationary bounds. Mount
orientation and calibrated heading need the separate commissioning checks.

The ROS I2C interface is already implemented in
[`imu_i2c_hardware_interface`](../src/imu_i2c_hardware_interface): it reads both
chips directly and performs orientation fusion on the Pi. The serial BMI/BMX
firmware is a legacy path.

## Changing a drive's CAN baud

Do this only as a planned bench maintenance operation with a recorded current
configuration. The tool changes one drive, preserves its watchdog and termination
settings, and attempts to save to flash. For example, restoring a drive known to
be at 8M to the current 1M configuration:

```bash
./build/candle_bus_test baud --bus spi --baud 8 --id 10 --to 1
```

`--baud` is the current rate; `--to` is the new one. Check `PASS`, then ping at
the new rate and verify again after a power cycle with automatic recovery stopped.
If saving fails, the new setting may exist only in RAM; the tool reports this.
Repeat deliberately for the affected IDs, not blindly for every drive.

The July 2026 8M migration was reversed on 2026-10-05 after harness/termination
faults. A higher baud requires a corrected, measured harness and a new validation
record. It is not required for the HAT.
