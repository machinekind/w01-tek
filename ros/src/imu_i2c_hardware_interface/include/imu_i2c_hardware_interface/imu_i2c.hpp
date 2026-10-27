// Copyright (c) 2026, Machinekind
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IMU_I2C_HARDWARE_INTERFACE__IMU_I2C_HPP_
#define IMU_I2C_HARDWARE_INTERFACE__IMU_I2C_HPP_

#include <cstdint>
#include <memory>
#include <string>

// Register-level driver for the Adafruit 5543 board: LSM6DS3TR-C (accel +
// gyro) and LIS3MDL (magnetometer), two independent ST chips on the same
// I2C bus, no MCU/firmware in between. Register map, init sequence and
// scale factors are lifted straight from the bench test that already
// passed on hardware (ros/hw_tests/imu_i2c/imu_i2c_test.py) -- keep the two
// in sync if either changes.
class ImuI2C
{
public:
  using SharedPtr = std::shared_ptr<ImuI2C>;

  struct SensorData
  {
    double accel_x, accel_y, accel_z;  // m/s^2
    double gyro_x, gyro_y, gyro_z;     // rad/s
    double mag_x, mag_y, mag_z;        // uT
    // Data-ready flags: the caller polls faster (400 Hz controller loop)
    // than the sensors sample (LSM6 104 Hz, LIS3 80 Hz), so each read
    // reports which blocks are actually new since the previous one.  The
    // orientation ESKF steps only on fresh samples.
    bool accel_gyro_fresh;
    bool mag_fresh;
  };

  ImuI2C(std::string bus_path, int addr_ag, int addr_mag);
  ~ImuI2C();

  ImuI2C(const ImuI2C &) = delete;
  ImuI2C & operator=(const ImuI2C &) = delete;

  // Opens the bus, checks WHO_AM_I on both chips and writes the ODR/FS
  // config. Never throws: any failure (missing bus device, wrong chip id,
  // a failed register write) is reported via the return value and
  // last_error(), so a missing sensor degrades to a plugin configure error
  // instead of aborting the whole ros2_control_node.
  bool initialize();

  // Reads one accel+gyro+mag sample, skipping blocks whose data-ready
  // status bit is clear (the *_fresh flags say what was updated). Returns
  // false on an I2C transaction error and leaves data untouched -- the
  // caller decides whether to hold the last known-good values.
  bool read_sample(SensorData & data);

  const std::string & last_error() const { return last_error_; }

  // Times the magnetometer was re-initialised from read_sample() because
  // it stopped raising data-ready (see mag_stale_cycles_ below).
  unsigned mag_reinits() const { return mag_reinits_; }

private:
  bool write_reg(int addr, uint8_t reg, uint8_t val);
  bool init_mag();
  bool read_regs(int addr, uint8_t reg, uint8_t * buf, std::size_t len);

  std::string bus_path_;
  int addr_ag_;
  int addr_mag_;
  int fd_ = -1;
  std::string last_error_;
  // 2026-09-27, on the robot: the LIS3MDL lost its configuration mid-run
  // (CTRL_REG3 read 0x03 = power-down, CTRL_REG1 back to 0x18) with the
  // chip still answering WHO_AM_I. Its output froze, ZYXDA stayed clear,
  // and the ESKF, left to the gyro bias alone, spun the yaw at up to
  // 40 deg/s while the robot stood still. Rewriting the control registers
  // brought it straight back, so read_sample() does that itself after
  // this many consecutive cycles without a fresh mag sample.
  int mag_stale_cycles_ = 0;
  static constexpr int kMagStaleReinit = 200;  // ~1 s at the 200 Hz CM rate
  unsigned mag_reinits_ = 0;
};

#endif  // IMU_I2C_HARDWARE_INTERFACE__IMU_I2C_HPP_
