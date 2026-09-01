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

#ifndef WOJTEK_MUJOCO_HARDWARE_INTERFACE__MUJOCO_HARDWARE_INTERFACE_HPP_
#define WOJTEK_MUJOCO_HARDWARE_INTERFACE__MUJOCO_HARDWARE_INTERFACE_HPP_

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "geometry_msgs/msg/twist.hpp"
#include "hardware_interface/handle.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/state.hpp"
#include "std_msgs/msg/float32.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "wojtek_mujoco_hardware_interface/joint_map.hpp"
#include "wojtek_mujoco_hardware_interface/mujoco_plant.hpp"

namespace wojtek_mujoco_hardware_interface
{

/// ros2_control system component backed by MuJoCo.
///
/// Drops into the slot md80_hardware_interface + imu_i2c_hardware_interface
/// occupy on the robot, exporting the same interfaces under the same names,
/// so the rest of the stack -- controller_manager, the broadcasters,
/// real_io_node, policy_node -- runs unchanged over simulated physics.
///
/// One component carries both the joints and the IMU, unlike the robot's two
/// drivers on two buses: there is a single physics state here and splitting
/// it across components would mean sharing it between them.
class MujocoHardwareInterface : public hardware_interface::SystemInterface
{
public:
  hardware_interface::CallbackReturn on_init(
    const hardware_interface::HardwareInfo & info) override;

  hardware_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;

  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

  hardware_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;
  hardware_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  hardware_interface::return_type read(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;
  hardware_interface::return_type write(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  void publishGroundTruth(const rclcpp::Duration & period);

  rclcpp::Logger logger_{rclcpp::get_logger("MujocoHardwareInterface")};
  MujocoPlant plant_;
  std::unique_ptr<JointMap> joint_map_;

  std::vector<std::string> joint_names_;
  /// Actuator index in the model per exported joint.
  std::vector<int> actuator_of_joint_;
  std::vector<double> position_, velocity_, effort_, command_, effort_command_;
  /// Which joints declared the effort command interface (tau_ff policies);
  /// the rest run the servo alone and never touch effort_command_.
  std::vector<bool> joint_has_effort_;

  /// IMU states in the driver's layout: magnetometer xyz (0-2), gyro xyz
  /// (3-5), accel xyz (6-8), orientation xyzw (9-12).
  std::vector<double> imu_states_;
  bool has_imu_ = false;
  std::string imu_sensor_name_;
  std::array<double, 4> mount_quat_{1.0, 0.0, 0.0, 0.0};
  std::array<double, 3> earth_field_ut_{20.0, 0.0, -44.0};

  std::string scene_xml_, mesh_dir_, initial_pose_{"home"};
  bool dry_run_ = false;

  // Ground truth: what the real robot cannot know and the simulation can.
  // Published from a node of our own (a hardware component has none) at a
  // rate of its own -- the 400 Hz control loop is no rate for TF.
  std::shared_ptr<rclcpp::Node> node_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr vel_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr rtf_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr qpos_pub_;
  std_msgs::msg::Float64MultiArray qpos_msg_;
  double ground_truth_period_ = 0.01;
  double since_ground_truth_ = 0.0;
  double sim_time_at_window_ = 0.0;
  double wall_in_window_ = 0.0;
};

}  // namespace wojtek_mujoco_hardware_interface

#endif  // WOJTEK_MUJOCO_HARDWARE_INTERFACE__MUJOCO_HARDWARE_INTERFACE_HPP_
