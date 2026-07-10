# BMI160 firmware

`main/main.ino` streams accelerometer + gyroscope + fused orientation
(6-DOF Mahony filter, no magnetometer) over serial, in the format expected
by `bmi160_serial.cpp`'s parser:

```
G X: <rad/s> Y: <rad/s> Z: <rad/s> rad/s
A X: <m/s^2> Y: <m/s^2> Z: <m/s^2> m/s^2
R X: <qx> Y: <qy> Z: <qz> W: <qw>
```

## Board

Classic ESP32 (Xtensa, CP210x USB-serial bridge). arduino-cli FQBN:
`esp32:esp32:esp32`. Serial baud: **115200** (must match `baud_rate` param
in the `wojtek_bmi160_imu_ros2_control` xacro macro).

## Dependencies

- **Adafruit Unified Sensor** — on the Arduino Library Manager index,
  install with `arduino-cli lib install "Adafruit Unified Sensor"`.
- **`BMX160/`** in this directory — vendored (not a git submodule) from
  <https://github.com/drcpattison/BMX160> (MIT, `LICENSE` included), which
  is the only place `DPEng_BMX160`/`Mahony_BMX160` (used by
  `bmx160_serial_hardware_interface`'s original firmware too) could be
  found; it isn't on the Library Manager index and the original upstream
  git-submodule reference to it (in `delipl/quadruped_ros2`) is orphaned/
  unrecoverable. Only `DPEng_BMX160.{h,cpp}` and `Mahony_BMX160.{h,cpp}`
  are vendored — `Madgwick_BMX160` isn't used by this firmware.
- `DPEng_BMX160.h` has two deliberate local patches already applied:
  - the enum value `STATUS` (register 0x1B) was renamed to `BMX160_STATUS`
    because it collides with a deprecated `STATUS` typedef pulled in by
    `ets_sys.h` on current arduino-esp32 (3.x) cores, which otherwise fails
    to compile. Not referenced anywhere else in the library.
  - `BMX160_ADDRESS` was changed from the datasheet default `0x68` to
    `0x69` — this specific board pulls the ADO/SDO pin high. Confirmed by
    flashing a plain `Wire` I2C bus scanner: it only ever found a device at
    `0x69`. If you swap to a different physical board, rescan before
    assuming the address.
  - `BMX160_ID` was changed from `0xD8` to `0xD1` — a bare BMI160's
    CHIP_ID register (0x00) reads `0xD1`, not `0xD8` (that's BMX160's ID;
    despite sharing the accel/gyro register map, they are different chip
    IDs). Confirmed by reading register 0x00 directly on the actual
    hardware. Without this fix `dpEng.begin()` always fails with "Ooops,
    no BMX160 detected" even though the sensor is present and wired
    correctly.

To compile/flash (from a checkout with arduino-cli + the `esp32:esp32`
core installed):

```sh
arduino-cli lib install "Adafruit Unified Sensor"
arduino-cli compile --fqbn esp32:esp32:esp32 \
  --library firmware/BMX160 firmware/main
arduino-cli upload -p /dev/ttyUSB0 --fqbn esp32:esp32:esp32 \
  --library firmware/BMX160 firmware/main
```

## Why no magnetometer

This firmware targets a bare BMI160 chip, which has no integrated
magnetometer (unlike BMX160 = BMI160 + BMM150 in one package — see
`bmx160_serial_hardware_interface` for the real-sensor version). The
BMI160 and BMX160 accel/gyro cores share the same register map and chip ID
(`0xD8`), so `DPEng_BMX160`'s chip detection and accel/gyro register I/O
work unmodified on a bare BMI160; only the magnetometer read/print was
removed. Consequence: `R` (orientation quaternion) has accurate roll/pitch
but **no absolute heading** — yaw is free to drift, since nothing anchors
it to magnetic north.
