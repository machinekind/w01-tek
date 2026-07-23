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

#ifndef IMU_I2C_HARDWARE_INTERFACE__IMU_I2C_HARDWARE_INTERFACE_HPP_
#define IMU_I2C_HARDWARE_INTERFACE__IMU_I2C_HARDWARE_INTERFACE_HPP_

#include <memory>
#include <string>
#include <vector>

#include "hardware_interface/handle.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "hardware_interface/sensor_interface.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "imu_i2c_hardware_interface/imu_i2c.hpp"
#include "imu_i2c_hardware_interface/orientation_eskf.hpp"
#include "imu_i2c_hardware_interface/visibility_control.h"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/state.hpp"

namespace imu_i2c_hardware_interface
{
class ImuI2CHardwareInterface : public hardware_interface::SensorInterface
{
public:
  TEMPLATES__ROS2_CONTROL__VISIBILITY_PUBLIC
  hardware_interface::CallbackReturn on_init(
    const hardware_interface::HardwareInfo & info) override;

  TEMPLATES__ROS2_CONTROL__VISIBILITY_PUBLIC
  hardware_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;

  TEMPLATES__ROS2_CONTROL__VISIBILITY_PUBLIC
  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;

  TEMPLATES__ROS2_CONTROL__VISIBILITY_PUBLIC
  hardware_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;

  TEMPLATES__ROS2_CONTROL__VISIBILITY_PUBLIC
  hardware_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  TEMPLATES__ROS2_CONTROL__VISIBILITY_PUBLIC
  hardware_interface::return_type read(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  ImuI2C::SharedPtr imu_;
  rclcpp::Logger logger_{rclcpp::get_logger("ImuI2CHardwareInterface")};
  rclcpp::Clock steady_clock_{RCL_STEADY_TIME};

  // Sensor fusion: the LSM6DS3TR-C/LIS3MDL pair has no onboard fusion, so
  // orientation.* is computed here by an error-state Kalman filter (gyro
  // prediction + accel roll/pitch + mag yaw, see orientation_eskf.hpp).
  // Until the filter has initialized (first fresh accel+mag pair) the
  // quaternion stays (0,0,0,0), which wojtek_policy's IMU callback treats
  // as "no orientation available" and falls back to its own accel+gyro
  // complementary filter (policy_node.py _on_imu) -- a graceful start and
  // the rollback path (gravity_from_accel param) in one.
  std::unique_ptr<OrientationEskf> eskf_;
  // Time accumulated between fresh gyro samples (read() polls at the
  // 400 Hz controller rate, the LSM6 samples at 104 Hz).
  double gyro_dt_ = 0.0;
  // A mag sample can land on a cycle without a fresh gyro sample (80 vs
  // 104 Hz); latch it and hand it to the filter on the next gyro step.
  bool mag_pending_ = false;

  // Layout: magnetometer.{x,y,z} (0-2), angular_velocity.{x,y,z} (3-5),
  // linear_acceleration.{x,y,z} (6-8), orientation.{x,y,z,w} (9-12).
  std::vector<double> hw_states_;
};

}  // namespace imu_i2c_hardware_interface

#endif  // IMU_I2C_HARDWARE_INTERFACE__IMU_I2C_HARDWARE_INTERFACE_HPP_
