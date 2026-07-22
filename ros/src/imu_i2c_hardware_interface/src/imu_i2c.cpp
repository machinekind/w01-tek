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

#include "imu_i2c_hardware_interface/imu_i2c.hpp"

#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <linux/i2c.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>

namespace
{
// --- LSM6DS3TR-C (accel + gyro) register map ---
constexpr uint8_t LSM6_WHO_AM_I = 0x0F;
constexpr uint8_t LSM6_CTRL1_XL = 0x10;  // accel ODR/FS
constexpr uint8_t LSM6_CTRL2_G = 0x11;   // gyro ODR/FS
constexpr uint8_t LSM6_CTRL3_C = 0x12;   // BDU, IF_INC, SW_RESET
constexpr uint8_t LSM6_OUTX_L_G = 0x22;  // gyro xyz (6B) then accel xyz (6B)

constexpr uint8_t LSM6_ID = 0x6A;
constexpr uint8_t LSM6_SW_RESET = 0x01;
constexpr uint8_t LSM6_BDU_IFINC = 0x44;      // block data update + auto-increment
constexpr uint8_t LSM6_XL_104HZ_4G = 0x48;    // ODR=104 Hz, FS=+/-4 g
constexpr uint8_t LSM6_G_104HZ_500DPS = 0x44;  // ODR=104 Hz, FS=+/-500 dps

constexpr double ACC_MS2_PER_LSB = 0.122e-3 * 9.80665;  // +/-4 g -> 0.122 mg/LSB
constexpr double GYR_RAD_PER_LSB = 17.5e-3 * M_PI / 180.0;  // +/-500 dps -> 17.5 mdps/LSB

// --- LIS3MDL (magnetometer) register map ---
constexpr uint8_t LIS3_WHO_AM_I = 0x0F;
constexpr uint8_t LIS3_CTRL_REG1 = 0x20;
constexpr uint8_t LIS3_CTRL_REG2 = 0x21;
constexpr uint8_t LIS3_CTRL_REG3 = 0x22;
constexpr uint8_t LIS3_CTRL_REG4 = 0x23;
constexpr uint8_t LIS3_OUT_X_L = 0x28;
constexpr uint8_t LIS3_AUTOINC = 0x80;  // ST mag: sub-address MSb enables multi-byte reads

constexpr uint8_t LIS3_ID = 0x3D;
constexpr uint8_t LIS3_UHP_80HZ = 0x7C;  // OM=ultra-high (xy), DO=80 Hz
constexpr uint8_t LIS3_FS_4GAUSS = 0x00;
constexpr uint8_t LIS3_CONTINUOUS = 0x00;
constexpr uint8_t LIS3_UHP_Z = 0x0C;  // OMZ=ultra-high (z)

constexpr double MAG_UT_PER_LSB = 100.0 / 6842.0;  // +/-4 gauss -> 6842 LSB/gauss, 1 G = 100 uT

int16_t s16(uint8_t lo, uint8_t hi)
{
  return static_cast<int16_t>((static_cast<uint16_t>(hi) << 8) | lo);
}
}  // namespace

ImuI2C::ImuI2C(std::string bus_path, int addr_ag, int addr_mag)
: bus_path_(std::move(bus_path)), addr_ag_(addr_ag), addr_mag_(addr_mag)
{
}

ImuI2C::~ImuI2C()
{
  if (fd_ >= 0) {
    close(fd_);
  }
}

bool ImuI2C::write_reg(int addr, uint8_t reg, uint8_t val)
{
  uint8_t payload[2] = {reg, val};
  i2c_msg msg{};
  msg.addr = static_cast<uint16_t>(addr);
  msg.flags = 0;
  msg.len = sizeof(payload);
  msg.buf = payload;

  i2c_rdwr_ioctl_data xfer{};
  xfer.msgs = &msg;
  xfer.nmsgs = 1;

  if (ioctl(fd_, I2C_RDWR, &xfer) < 0) {
    char buf[128];
    std::snprintf(
      buf, sizeof(buf), "write 0x%02X=0x%02X @0x%02X: %s", reg, val, addr, strerror(errno));
    last_error_ = buf;
    return false;
  }
  return true;
}

bool ImuI2C::read_regs(int addr, uint8_t reg, uint8_t * out, std::size_t len)
{
  uint8_t reg_buf[1] = {reg};
  i2c_msg msgs[2]{};
  msgs[0].addr = static_cast<uint16_t>(addr);
  msgs[0].flags = 0;
  msgs[0].len = 1;
  msgs[0].buf = reg_buf;
  msgs[1].addr = static_cast<uint16_t>(addr);
  msgs[1].flags = I2C_M_RD;
  msgs[1].len = static_cast<uint16_t>(len);
  msgs[1].buf = out;

  i2c_rdwr_ioctl_data xfer{};
  xfer.msgs = msgs;
  xfer.nmsgs = 2;

  if (ioctl(fd_, I2C_RDWR, &xfer) < 0) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "read 0x%02X @0x%02X: %s", reg, addr, strerror(errno));
    last_error_ = buf;
    return false;
  }
  return true;
}

bool ImuI2C::initialize()
{
  fd_ = open(bus_path_.c_str(), O_RDWR);
  if (fd_ < 0) {
    char buf[192];
    std::snprintf(buf, sizeof(buf), "open %s: %s", bus_path_.c_str(), strerror(errno));
    last_error_ = buf;
    return false;
  }

  uint8_t who = 0;
  if (!read_regs(addr_ag_, LSM6_WHO_AM_I, &who, 1)) {
    return false;
  }
  if (who != LSM6_ID) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "LSM6DS3TR-C WHO_AM_I 0x%02X (expect 0x6A)", who);
    last_error_ = buf;
    return false;
  }
  if (!read_regs(addr_mag_, LIS3_WHO_AM_I, &who, 1)) {
    return false;
  }
  if (who != LIS3_ID) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "LIS3MDL WHO_AM_I 0x%02X (expect 0x3D)", who);
    last_error_ = buf;
    return false;
  }

  if (!write_reg(addr_ag_, LSM6_CTRL3_C, LSM6_SW_RESET)) {
    return false;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(20));

  const bool ok =
    write_reg(addr_ag_, LSM6_CTRL3_C, LSM6_BDU_IFINC) &&
    write_reg(addr_ag_, LSM6_CTRL1_XL, LSM6_XL_104HZ_4G) &&
    write_reg(addr_ag_, LSM6_CTRL2_G, LSM6_G_104HZ_500DPS) &&
    write_reg(addr_mag_, LIS3_CTRL_REG1, LIS3_UHP_80HZ) &&
    write_reg(addr_mag_, LIS3_CTRL_REG2, LIS3_FS_4GAUSS) &&
    write_reg(addr_mag_, LIS3_CTRL_REG4, LIS3_UHP_Z) &&
    write_reg(addr_mag_, LIS3_CTRL_REG3, LIS3_CONTINUOUS);
  if (!ok) {
    return false;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(20));

  return true;
}

bool ImuI2C::read_sample(SensorData & data)
{
  uint8_t ag[12];
  if (!read_regs(addr_ag_, LSM6_OUTX_L_G, ag, sizeof(ag))) {
    return false;
  }
  data.gyro_x = s16(ag[0], ag[1]) * GYR_RAD_PER_LSB;
  data.gyro_y = s16(ag[2], ag[3]) * GYR_RAD_PER_LSB;
  data.gyro_z = s16(ag[4], ag[5]) * GYR_RAD_PER_LSB;
  data.accel_x = s16(ag[6], ag[7]) * ACC_MS2_PER_LSB;
  data.accel_y = s16(ag[8], ag[9]) * ACC_MS2_PER_LSB;
  data.accel_z = s16(ag[10], ag[11]) * ACC_MS2_PER_LSB;

  // A dropped mag read is not fatal on its own -- accel/gyro (the fields
  // wojtek_policy actually consumes) already succeeded above, so surface
  // that frame rather than discard it; magnetometer.* just holds its last
  // value for this cycle.
  uint8_t mag[6];
  if (read_regs(addr_mag_, LIS3_OUT_X_L | LIS3_AUTOINC, mag, sizeof(mag))) {
    data.mag_x = s16(mag[0], mag[1]) * MAG_UT_PER_LSB;
    data.mag_y = s16(mag[2], mag[3]) * MAG_UT_PER_LSB;
    data.mag_z = s16(mag[4], mag[5]) * MAG_UT_PER_LSB;
  }

  return true;
}
