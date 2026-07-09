// Copyright (c) 2024, Jakub Delicat
// Copyright (c) 2024, Stogl Robotics Consulting UG (haftungsbeschränkt)
// (template)
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

#include "bmx160_serial_hardware_interface/bmx160_serial_hardware_interface.hpp"
#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <limits>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <vector>

namespace bmx160_serial_hardware_interface
{

hardware_interface::CallbackReturn BMX160SerialHardwareInterface::on_init(
  const hardware_interface::HardwareInfo & info)
{
  if (
    hardware_interface::SensorInterface::on_init(info) !=
    hardware_interface::CallbackReturn::SUCCESS) {
    return hardware_interface::CallbackReturn::ERROR;
  }

  // Initialize BMX160 sensor
  std::string port_name = info_.hardware_parameters.at("port");
  int baud_rate = std::stoi(info_.hardware_parameters.at("baud_rate"));

  bmx160_ = std::make_shared<BMX160Serial>(port_name, baud_rate);

  // Resize hardware states (3 sensors * 3 axes = 9 total state variables + 4
  // orientation )
  hw_states_.resize(13, std::numeric_limits<double>::quiet_NaN());

  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn BMX160SerialHardwareInterface::on_configure(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  if (!bmx160_->initialize()) {
    RCLCPP_ERROR(
      rclcpp::get_logger("BMX160SerialHardwareInterface"), "Failed to initialize BMX160 sensor");
    return hardware_interface::CallbackReturn::ERROR;
  }

  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn BMX160SerialHardwareInterface::on_activate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn BMX160SerialHardwareInterface::on_deactivate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::return_type BMX160SerialHardwareInterface::read(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & period)
{
  sensor_data_ = bmx160_->read_sensor_data();

  // Map sensor data to the hardware state vector
  hw_states_[0] = std::isnan(sensor_data_.mag_x) ? hw_states_[0]
                                                 : sensor_data_.mag_x * 1e-3 - mag_bias_x_;
  hw_states_[1] = std::isnan(sensor_data_.mag_y) ? hw_states_[1]
                                                 : sensor_data_.mag_y * 1e-3 - mag_bias_y_;
  hw_states_[2] = std::isnan(sensor_data_.mag_z) ? hw_states_[2]
                                                 : sensor_data_.mag_z * 1e-3 - mag_bias_z_;

  // TODO @delipl what is an unit of gyration?
  hw_states_[3] = std::isnan(sensor_data_.gyro_x) ? hw_states_[3]
                                                  : sensor_data_.gyro_x * M_PI / 180.0;
  hw_states_[4] = std::isnan(sensor_data_.gyro_y) ? hw_states_[4]
                                                  : sensor_data_.gyro_y * M_PI / 180.0;
  hw_states_[5] = std::isnan(sensor_data_.gyro_z) ? hw_states_[5]
                                                  : sensor_data_.gyro_z * M_PI / 180.0;

  const double accel_x = sensor_data_.accel_x;
  const double accel_y = sensor_data_.accel_y;
  const double accel_z = sensor_data_.accel_z;

  const auto dt = period.seconds();
  const auto nan = std::numeric_limits<double>::quiet_NaN();

  if (std::isnan(hw_states_[6]) || std::isnan(hw_states_[7]) || std::isnan(hw_states_[8])) {
    hw_states_[6] = accel_x;
    hw_states_[7] = accel_y;
    hw_states_[8] = accel_z;
    if (std::isnan(accel_z)) {
      return hardware_interface::return_type::OK;
    }
  }

  hw_states_[6] = std::isnan(sensor_data_.accel_x) ? hw_states_[6] : sensor_data_.accel_x;  // - gx;
  hw_states_[7] = std::isnan(sensor_data_.accel_y) ? hw_states_[7] : sensor_data_.accel_y;  // - gy;
  hw_states_[8] = std::isnan(sensor_data_.accel_z) ? hw_states_[8] : sensor_data_.accel_z;

  hw_states_[9] = sensor_data_.quat_x;
  hw_states_[10] = sensor_data_.quat_y;
  hw_states_[11] = sensor_data_.quat_z;
  hw_states_[12] = sensor_data_.quat_w;

  return hardware_interface::return_type::OK;
}

std::vector<hardware_interface::StateInterface>
BMX160SerialHardwareInterface::export_state_interfaces()
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

}  // namespace bmx160_serial_hardware_interface

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(
  bmx160_serial_hardware_interface::BMX160SerialHardwareInterface,
  hardware_interface::SensorInterface)
