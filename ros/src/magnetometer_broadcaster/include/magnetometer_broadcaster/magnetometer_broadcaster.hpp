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

#ifndef MAGNETOMETER_BROADCASTER__MAGNETOMETER_BROADCASTER_HPP_
#define MAGNETOMETER_BROADCASTER__MAGNETOMETER_BROADCASTER_HPP_

#include <memory>
#include <string>
#include <vector>

#include "controller_interface/controller_interface.hpp"
#include "rclcpp_lifecycle/state.hpp"
#include "realtime_tools/realtime_publisher.hpp"
#include "sensor_msgs/msg/magnetic_field.hpp"

namespace magnetometer_broadcaster
{

// Broadcaster counterpart of imu_sensor_broadcaster for the magnetometer:
// upstream ros2_controllers (Jazzy) has no controller that consumes
// magnetometer.{x,y,z} state interfaces, so the field our
// imu_i2c_hardware_interface exports would never reach a topic. Publishes
// sensor_msgs/MagneticField on ~/magnetic_field. Publish rate is governed
// by the controller_manager per-controller update_rate parameter.
class MagnetometerBroadcaster : public controller_interface::ControllerInterface
{
public:
  controller_interface::CallbackReturn on_init() override;

  controller_interface::InterfaceConfiguration command_interface_configuration() const override;

  controller_interface::InterfaceConfiguration state_interface_configuration() const override;

  controller_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;

  controller_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;

  controller_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  controller_interface::return_type update(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  std::string sensor_name_;

  rclcpp::Publisher<sensor_msgs::msg::MagneticField>::SharedPtr publisher_;
  std::unique_ptr<realtime_tools::RealtimePublisher<sensor_msgs::msg::MagneticField>>
    realtime_publisher_;
};

}  // namespace magnetometer_broadcaster

#endif  // MAGNETOMETER_BROADCASTER__MAGNETOMETER_BROADCASTER_HPP_
