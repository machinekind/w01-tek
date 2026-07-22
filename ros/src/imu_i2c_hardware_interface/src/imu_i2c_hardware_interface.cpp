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

#include "imu_i2c_hardware_interface/imu_i2c_hardware_interface.hpp"

#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <limits>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <vector>

namespace imu_i2c_hardware_interface
{

hardware_interface::CallbackReturn ImuI2CHardwareInterface::on_init(
  const hardware_interface::HardwareInfo & info)
{
  if (
    hardware_interface::SensorInterface::on_init(info) !=
    hardware_interface::CallbackReturn::SUCCESS) {
    return hardware_interface::CallbackReturn::ERROR;
  }

  const std::string bus = info_.hardware_parameters.at("bus");
  const int addr_ag = std::stoi(info_.hardware_parameters.at("addr_ag"), nullptr, 0);
  const int addr_mag = std::stoi(info_.hardware_parameters.at("addr_mag"), nullptr, 0);
  imu_ = std::make_shared<ImuI2C>(bus, addr_ag, addr_mag);

  hw_states_.resize(13, std::numeric_limits<double>::quiet_NaN());
  // orientation.{x,y,z,w}: fixed "no fusion available" marker, see header.
  hw_states_[9] = hw_states_[10] = hw_states_[11] = hw_states_[12] = 0.0;

  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn ImuI2CHardwareInterface::on_configure(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  if (!imu_->initialize()) {
    RCLCPP_ERROR(
      logger_, "Failed to initialize IMU over I2C: %s", imu_->last_error().c_str());
    return hardware_interface::CallbackReturn::ERROR;
  }
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn ImuI2CHardwareInterface::on_activate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn ImuI2CHardwareInterface::on_deactivate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::return_type ImuI2CHardwareInterface::read(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  ImuI2C::SensorData d;
  if (!imu_->read_sample(d)) {
    // Transient I2C error: hold the last known-good values instead of
    // writing NaN, same "stale over NaN" degrade the two serial IMU
    // drivers already use -- a single dropped transaction should not trip
    // wojtek_policy's non-finite-input watchdog.
    RCLCPP_WARN_THROTTLE(
      logger_, steady_clock_, 1000, "I2C read failed: %s", imu_->last_error().c_str());
    return hardware_interface::return_type::OK;
  }

  hw_states_[0] = d.mag_x;
  hw_states_[1] = d.mag_y;
  hw_states_[2] = d.mag_z;
  hw_states_[3] = d.gyro_x;
  hw_states_[4] = d.gyro_y;
  hw_states_[5] = d.gyro_z;
  hw_states_[6] = d.accel_x;
  hw_states_[7] = d.accel_y;
  hw_states_[8] = d.accel_z;
  // orientation (9-12) stays (0,0,0,0) for the plugin's whole lifetime.

  return hardware_interface::return_type::OK;
}

std::vector<hardware_interface::StateInterface>
ImuI2CHardwareInterface::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> state_interfaces;

  const auto & sensor_name = info_.sensors.at(0).name;

  state_interfaces.emplace_back(
    hardware_interface::StateInterface(sensor_name, "magnetometer.x", &hw_states_[0]));
  state_interfaces.emplace_back(
    hardware_interface::StateInterface(sensor_name, "magnetometer.y", &hw_states_[1]));
  state_interfaces.emplace_back(
    hardware_interface::StateInterface(sensor_name, "magnetometer.z", &hw_states_[2]));

  state_interfaces.emplace_back(
    hardware_interface::StateInterface(sensor_name, "angular_velocity.x", &hw_states_[3]));
  state_interfaces.emplace_back(
    hardware_interface::StateInterface(sensor_name, "angular_velocity.y", &hw_states_[4]));
  state_interfaces.emplace_back(
    hardware_interface::StateInterface(sensor_name, "angular_velocity.z", &hw_states_[5]));

  state_interfaces.emplace_back(
    hardware_interface::StateInterface(sensor_name, "linear_acceleration.x", &hw_states_[6]));
  state_interfaces.emplace_back(
    hardware_interface::StateInterface(sensor_name, "linear_acceleration.y", &hw_states_[7]));
  state_interfaces.emplace_back(
    hardware_interface::StateInterface(sensor_name, "linear_acceleration.z", &hw_states_[8]));

  state_interfaces.emplace_back(
    hardware_interface::StateInterface(sensor_name, "orientation.x", &hw_states_[9]));
  state_interfaces.emplace_back(
    hardware_interface::StateInterface(sensor_name, "orientation.y", &hw_states_[10]));
  state_interfaces.emplace_back(
    hardware_interface::StateInterface(sensor_name, "orientation.z", &hw_states_[11]));
  state_interfaces.emplace_back(
    hardware_interface::StateInterface(sensor_name, "orientation.w", &hw_states_[12]));

  return state_interfaces;
}

}  // namespace imu_i2c_hardware_interface

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(
  imu_i2c_hardware_interface::ImuI2CHardwareInterface, hardware_interface::SensorInterface)
