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

#include "bmi160_serial_hardware_interface/bmi160_serial_hardware_interface.hpp"
#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <limits>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <vector>

namespace bmi160_serial_hardware_interface
{

hardware_interface::CallbackReturn BMI160SerialHardwareInterface::on_init(
  const hardware_interface::HardwareInfo & info)
{
  if (
    hardware_interface::SensorInterface::on_init(info) !=
    hardware_interface::CallbackReturn::SUCCESS) {
    return hardware_interface::CallbackReturn::ERROR;
  }

  // Initialize BMI160 sensor
  std::string port_name = info_.hardware_parameters.at("port");
  int baud_rate = std::stoi(info_.hardware_parameters.at("baud_rate"));

  bmi160_ = std::make_shared<BMI160Serial>(port_name, baud_rate);

  // Resize hardware states (gyro x/y/z, accel x/y/z, quaternion x/y/z/w)
  hw_states_.resize(10, std::numeric_limits<double>::quiet_NaN());

  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn BMI160SerialHardwareInterface::on_configure(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  if (!bmi160_->initialize()) {
    RCLCPP_ERROR(
      rclcpp::get_logger("BMI160SerialHardwareInterface"), "Failed to initialize BMI160 sensor");
    return hardware_interface::CallbackReturn::ERROR;
  }

  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn BMI160SerialHardwareInterface::on_activate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn BMI160SerialHardwareInterface::on_deactivate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::return_type BMI160SerialHardwareInterface::read(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  sensor_data_ = bmi160_->read_sensor_data();

  // read_sensor_data() is deadline-bounded and returns NaN for fields that
  // did not arrive in time; latch the previous value for those so a slow or
  // silent sensor degrades to stale data instead of NaN on /imu/data.
  if (
    std::isnan(sensor_data_.gyro_x) || std::isnan(sensor_data_.accel_x) ||
    std::isnan(sensor_data_.quat_x)) {
    RCLCPP_WARN_THROTTLE(
      logger_, steady_clock_, 1000,
      "incomplete IMU read (sensor silent or slow) -- holding previous values");
  }

  // TODO @delipl what is an unit of gyration?
  hw_states_[0] = std::isnan(sensor_data_.gyro_x) ? hw_states_[0]
                                                  : sensor_data_.gyro_x * M_PI / 180.0;
  hw_states_[1] = std::isnan(sensor_data_.gyro_y) ? hw_states_[1]
                                                  : sensor_data_.gyro_y * M_PI / 180.0;
  hw_states_[2] = std::isnan(sensor_data_.gyro_z) ? hw_states_[2]
                                                  : sensor_data_.gyro_z * M_PI / 180.0;

  hw_states_[3] = std::isnan(sensor_data_.accel_x) ? hw_states_[3] : sensor_data_.accel_x;
  hw_states_[4] = std::isnan(sensor_data_.accel_y) ? hw_states_[4] : sensor_data_.accel_y;
  hw_states_[5] = std::isnan(sensor_data_.accel_z) ? hw_states_[5] : sensor_data_.accel_z;

  hw_states_[6] = std::isnan(sensor_data_.quat_x) ? hw_states_[6] : sensor_data_.quat_x;
  hw_states_[7] = std::isnan(sensor_data_.quat_y) ? hw_states_[7] : sensor_data_.quat_y;
  hw_states_[8] = std::isnan(sensor_data_.quat_z) ? hw_states_[8] : sensor_data_.quat_z;
  hw_states_[9] = std::isnan(sensor_data_.quat_w) ? hw_states_[9] : sensor_data_.quat_w;

  return hardware_interface::return_type::OK;
}

std::vector<hardware_interface::StateInterface>
BMI160SerialHardwareInterface::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> state_interfaces;

  const auto & sensor_name = info_.sensors.at(0).name;

  state_interfaces.emplace_back(
    hardware_interface::StateInterface(sensor_name, "angular_velocity.x", &hw_states_[0]));
  state_interfaces.emplace_back(
    hardware_interface::StateInterface(sensor_name, "angular_velocity.y", &hw_states_[1]));
  state_interfaces.emplace_back(
    hardware_interface::StateInterface(sensor_name, "angular_velocity.z", &hw_states_[2]));

  state_interfaces.emplace_back(
    hardware_interface::StateInterface(sensor_name, "linear_acceleration.x", &hw_states_[3]));
  state_interfaces.emplace_back(
    hardware_interface::StateInterface(sensor_name, "linear_acceleration.y", &hw_states_[4]));
  state_interfaces.emplace_back(
    hardware_interface::StateInterface(sensor_name, "linear_acceleration.z", &hw_states_[5]));

  state_interfaces.emplace_back(
    hardware_interface::StateInterface(sensor_name, "orientation.x", &hw_states_[6]));
  state_interfaces.emplace_back(
    hardware_interface::StateInterface(sensor_name, "orientation.y", &hw_states_[7]));
  state_interfaces.emplace_back(
    hardware_interface::StateInterface(sensor_name, "orientation.z", &hw_states_[8]));
  state_interfaces.emplace_back(
    hardware_interface::StateInterface(sensor_name, "orientation.w", &hw_states_[9]));

  return state_interfaces;
}

}  // namespace bmi160_serial_hardware_interface

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(
  bmi160_serial_hardware_interface::BMI160SerialHardwareInterface,
  hardware_interface::SensorInterface)
