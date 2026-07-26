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

#include "magnetometer_broadcaster/magnetometer_broadcaster.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace magnetometer_broadcaster
{

controller_interface::CallbackReturn MagnetometerBroadcaster::on_init()
{
  try {
    auto_declare<std::string>("sensor_name", "imu");
    auto_declare<std::string>("frame_id", "imu_link");
    // Row-major 3x3 covariance for the message; all zeros = "unknown" per
    // the sensor_msgs/MagneticField spec.
    auto_declare<std::vector<double>>("static_covariance", std::vector<double>(9, 0.0));
  } catch (const std::exception & e) {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration
MagnetometerBroadcaster::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::NONE;
  return config;
}

controller_interface::InterfaceConfiguration
MagnetometerBroadcaster::state_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  config.names = {
    sensor_name_ + "/magnetometer.x",
    sensor_name_ + "/magnetometer.y",
    sensor_name_ + "/magnetometer.z",
  };
  return config;
}

controller_interface::CallbackReturn MagnetometerBroadcaster::on_configure(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  sensor_name_ = get_node()->get_parameter("sensor_name").as_string();
  if (sensor_name_.empty()) {
    RCLCPP_ERROR(get_node()->get_logger(), "'sensor_name' parameter must not be empty");
    return controller_interface::CallbackReturn::ERROR;
  }

  const auto covariance = get_node()->get_parameter("static_covariance").as_double_array();
  if (covariance.size() != 9) {
    RCLCPP_ERROR(
      get_node()->get_logger(), "'static_covariance' must have 9 entries, got %zu",
      covariance.size());
    return controller_interface::CallbackReturn::ERROR;
  }

  try {
    publisher_ = get_node()->create_publisher<sensor_msgs::msg::MagneticField>(
      "~/magnetic_field", rclcpp::SystemDefaultsQoS());
    realtime_publisher_ =
      std::make_unique<realtime_tools::RealtimePublisher<sensor_msgs::msg::MagneticField>>(
        publisher_);
  } catch (const std::exception & e) {
    fprintf(stderr, "Exception thrown during publisher creation with message: %s \n", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  realtime_publisher_->lock();
  realtime_publisher_->msg_.header.frame_id = get_node()->get_parameter("frame_id").as_string();
  std::copy(
    covariance.begin(), covariance.end(),
    realtime_publisher_->msg_.magnetic_field_covariance.begin());
  realtime_publisher_->unlock();

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn MagnetometerBroadcaster::on_activate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn MagnetometerBroadcaster::on_deactivate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::return_type MagnetometerBroadcaster::update(
  const rclcpp::Time & time, const rclcpp::Duration & /*period*/)
{
  if (realtime_publisher_ && realtime_publisher_->trylock()) {
    auto & msg = realtime_publisher_->msg_;
    msg.header.stamp = time;
    // The hardware interface reports microtesla; the message wants tesla.
    constexpr double kUTeslaToTesla = 1e-6;
    msg.magnetic_field.x =
      state_interfaces_[0].get_optional().value_or(std::numeric_limits<double>::quiet_NaN()) *
      kUTeslaToTesla;
    msg.magnetic_field.y =
      state_interfaces_[1].get_optional().value_or(std::numeric_limits<double>::quiet_NaN()) *
      kUTeslaToTesla;
    msg.magnetic_field.z =
      state_interfaces_[2].get_optional().value_or(std::numeric_limits<double>::quiet_NaN()) *
      kUTeslaToTesla;
    realtime_publisher_->unlockAndPublish();
  }
  return controller_interface::return_type::OK;
}

}  // namespace magnetometer_broadcaster

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(
  magnetometer_broadcaster::MagnetometerBroadcaster, controller_interface::ControllerInterface)
