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

#ifndef BMI160_SERIAL_HARDWARE_INTERFACE__BMI160_SERIAL_HARDWARE_INTERFACE_HPP_
#define BMI160_SERIAL_HARDWARE_INTERFACE__BMI160_SERIAL_HARDWARE_INTERFACE_HPP_

#include <string>
#include <vector>

#include "bmi160_serial_hardware_interface/bmi160_serial.hpp"
#include "bmi160_serial_hardware_interface/visibility_control.h"
#include "hardware_interface/handle.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "hardware_interface/lexical_casts.hpp"
#include "hardware_interface/sensor_interface.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "rclcpp/macros.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/state.hpp"

namespace bmi160_serial_hardware_interface
{
class BMI160SerialHardwareInterface : public hardware_interface::SensorInterface
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

  hardware_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;

  TEMPLATES__ROS2_CONTROL__VISIBILITY_PUBLIC
  hardware_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  TEMPLATES__ROS2_CONTROL__VISIBILITY_PUBLIC
  hardware_interface::return_type read(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  BMI160Serial::SensorData sensor_data_;
  BMI160Serial::SharedPtr bmi160_;  // Instance of the BMI160Serial class

  rclcpp::Logger logger_{rclcpp::get_logger("BMI160SerialHardwareInterface")};
  rclcpp::Clock steady_clock_{RCL_STEADY_TIME};

  std::vector<double> hw_states_;
};

}  // namespace bmi160_serial_hardware_interface

#endif  // BMI160_SERIAL_HARDWARE_INTERFACE__BMI160_SERIAL_HARDWARE_INTERFACE_HPP_
