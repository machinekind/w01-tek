# Hardware bench tests: CANdle HAT (SPI) + Gravity BMX160 (I2C)

Why: the USB paths have been our least stable layer -- the CANdle USB dongle
can flake at bring-up ("Cannot find device on ID", 2 ms USB timeout in the
vendored lib) and the serial-bridge IMU had its own problems (baud/termios).
This setup replaces both USB links with buses native to the Pi header:

  * **CANdle HAT** -- FDCAN over **SPI** (`/dev/spidev0.0`), no USB dongle.
  * **Gravity BMX160 (DFRobot SEN0373)** -- 9-DoF IMU over **I2C**
    (`/dev/i2c-1`), no MCU + serial bridge in between.

These tests run natively on the RPi (no ROS, no container) and answer one
question each: is the link alive and stable enough to build the stack on it.

## 1. Hardware prep

1. Power off, mount the CANdle HAT on the Pi header, reconnect the FDCAN
   lines to the drives.
2. Wire the Gravity BMX160 to the Pi's I2C1 (check the HAT passes these pins
   through -- if not, wire to the HAT's stacking header):
   * SDA -> GPIO2 (pin 3), SCL -> GPIO3 (pin 5), VCC -> 3V3 (pin 1),
     GND -> pin 9. The Gravity board has its own regulator/level shifting.
3. Enable both buses in `/boot/firmware/config.txt` and reboot:

       dtparam=spi=on
       dtparam=i2c_arm=on
       # optional, faster I2C: dtparam=i2c_arm_baudrate=400000

4. Tools + permissions:

       sudo apt install i2c-tools python3-smbus2
       sudo usermod -aG spi,i2c rpi   # then re-login

5. Sanity: `ls /dev/spidev0.0 /dev/i2c-1` and `i2cdetect -y 1` -- the BMX160
   should show up at **0x68** (0x69 if SDO is pulled high).

## 2. Drive CAN baud: 1M -> 8M (required for SPI)

The SPI path runs FDCAN at **8M** (candle upstream: "before running be sure
all actuators are switched to 8M speed"). Our drives are configured for 1M
(what the USB stack uses). Switch each drive, **over the USB dongle first**,
with the `baud` mode of the test binary; it preserves the drive's CAN
watchdog + termination settings (read back before writing) and saves to
flash:

    ./build/candle_bus_test baud --bus usb --baud 1 --id 10 --to 8
    # ...repeat for every drive ID: 10-12, 20-22, 30-32, 40-42

**WARNING:** after this the ROS stack on the USB dongle (hardcoded 1M) will
NOT see the drives until you either roll back (`--to 1` the same way, over
`--bus usb --baud 8`) or the stack itself moves to the HAT/8M. Do this on the
bench, not before a demo.

## 3. Tests

### CANdle HAT (SPI)

    cd ros/hw_tests/candle_hat
    cmake -B build && cmake --build build -j
    ./build/candle_bus_test ping --bus spi --baud 8
    ./build/candle_bus_test soak --bus spi --baud 8 --seconds 300 --expect 12

`ping` lists the drive IDs found. `soak` adds every found drive, starts the
auto-update loop and streams for N seconds, printing the achieved update
frequency + positions once a second; it exits non-zero if any second's
frequency collapses or a drive count/ID mismatch shows up. This is the flake
detector: the USB path died at bring-up, so a long clean soak on SPI is the
pass criterion. Motors are **never enabled** -- read-only traffic, safe with
torque off.

### Gravity BMX160 (I2C)

    cd ros/hw_tests/bmx160_i2c
    python3 bmx160_i2c_test.py --seconds 10

Checks the chip ID (0xD8 = BMX160; a plain BMI160 reports 0xD1 -- the test
prints which one it found), runs the accel/gyro/mag power-up + magnetometer
indirect-interface init, then streams and reports: achieved sample rate,
|accel| mean/std (expect ~9.81 m/s^2 at rest), gyro bias (expect < ~3 dps),
mag field norm (plausible: ~25-65 uT). PASS/FAIL per criterion.

## 4. What comes after (integration, separate work)

* `md80_hardware_interface` hardcodes `BusType_E::USB`
  (`md80_hardware_interface.cpp`, `add_candle_instances`) -- the xacro `bus`
  param is parsed but ignored. Honor it (`usb`/`spi`) and make the stack baud
  configurable (1M vs 8M). Candle-side knobs belong to the planned candle
  rewrite (backlog #9).
* New `bmx160_i2c_hardware_interface` (or an I2C mode in the existing pkg):
  read BMX160 over `/dev/i2c-1` directly; orientation fusion moves from the
  MCU firmware (Mahony) to the Pi -- or is skipped entirely while the
  springy policy stays IMU-blind.
