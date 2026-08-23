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

#include "wojtek_mujoco_hardware_interface/mujoco_hardware_interface.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace wojtek_mujoco_hardware_interface
{
namespace
{

/// Fixed slots for the IMU state interfaces, matching the real driver's
/// layout so the two are easy to compare side by side.
const std::map<std::string, std::size_t> kImuSlots{
  {"magnetometer.x", 0}, {"magnetometer.y", 1}, {"magnetometer.z", 2},
  {"angular_velocity.x", 3}, {"angular_velocity.y", 4}, {"angular_velocity.z", 5},
  {"linear_acceleration.x", 6}, {"linear_acceleration.y", 7},
  {"linear_acceleration.z", 8},
  {"orientation.x", 9}, {"orientation.y", 10}, {"orientation.z", 11},
  {"orientation.w", 12},
};

std::string param(
  const std::unordered_map<std::string, std::string> & params,
  const std::string & key, const std::string & fallback = "")
{
  const auto it = params.find(key);
  return it == params.end() ? fallback : it->second;
}

bool isTrue(const std::string & value)
{
  std::string lowered = value;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(), ::tolower);
  return lowered == "true" || lowered == "1";
}

/// rpy (extrinsic xyz) -> quaternion wxyz, MuJoCo's convention.
std::array<double, 4> rpyToQuat(const std::array<double, 3> & rpy)
{
  const double cr = std::cos(rpy[0] / 2), sr = std::sin(rpy[0] / 2);
  const double cp = std::cos(rpy[1] / 2), sp = std::sin(rpy[1] / 2);
  const double cy = std::cos(rpy[2] / 2), sy = std::sin(rpy[2] / 2);
  return {
    cr * cp * cy + sr * sp * sy,
    sr * cp * cy - cr * sp * sy,
    cr * sp * cy + sr * cp * sy,
    cr * cp * sy - sr * sp * cy,
  };
}

std::vector<double> parseDoubles(const std::string & text)
{
  std::istringstream stream(text);
  std::vector<double> values;
  double value = 0.0;
  while (stream >> value) {
    values.push_back(value);
  }
  return values;
}

}  // namespace

hardware_interface::CallbackReturn MujocoHardwareInterface::on_init(
  const hardware_interface::HardwareInfo & info)
{
  if (
    hardware_interface::SystemInterface::on_init(info) !=
    hardware_interface::CallbackReturn::SUCCESS)
  {
    return hardware_interface::CallbackReturn::ERROR;
  }

  scene_xml_ = param(info_.hardware_parameters, "model_xml");
  mesh_dir_ = param(info_.hardware_parameters, "mesh_dir");
  if (scene_xml_.empty() || mesh_dir_.empty()) {
    RCLCPP_FATAL(
      logger_,
      "model_xml and mesh_dir are required (the launch resolves them from the "
      "package shares; the MJX XMLs ship away from the meshes they reference)");
    return hardware_interface::CallbackReturn::ERROR;
  }
  initial_pose_ = param(info_.hardware_parameters, "initial_pose", "home");
  dry_run_ = isTrue(param(info_.hardware_parameters, "dry_run", "false"));

  const auto joint_map_yaml = param(info_.hardware_parameters, "joint_map_yaml");
  if (joint_map_yaml.empty()) {
    RCLCPP_FATAL(logger_, "joint_map_yaml is required (wojtek_policy's joint_map.yaml)");
    return hardware_interface::CallbackReturn::ERROR;
  }
  try {
    joint_map_ = std::make_unique<JointMap>(joint_map_yaml);
  } catch (const std::exception & e) {
    RCLCPP_FATAL(logger_, "%s", e.what());
    return hardware_interface::CallbackReturn::ERROR;
  }

  // The physical unit is bolted on rotated (URDF imu_joint: rpy 0 pi 0), and
  // policy_node undoes exactly that rotation. Emulating the mount here is
  // what keeps imu_mount_rpy on the tested path instead of the sim quietly
  // shipping an unrotated sensor.
  const auto rpy = parseDoubles(
    param(info_.hardware_parameters, "imu_mount_rpy", "0 3.141592653589793 0"));
  if (rpy.size() != 3) {
    RCLCPP_FATAL(logger_, "imu_mount_rpy needs three numbers");
    return hardware_interface::CallbackReturn::ERROR;
  }
  mount_quat_ = rpyToQuat({rpy[0], rpy[1], rpy[2]});

  const auto field = parseDoubles(
    param(info_.hardware_parameters, "earth_field_ut", "20 0 -44"));
  if (field.size() == 3) {
    earth_field_ut_ = {field[0], field[1], field[2]};
  }

  const double gt_hz = std::stod(
    param(info_.hardware_parameters, "ground_truth_rate_hz", "100"));
  ground_truth_period_ = gt_hz > 0.0 ? 1.0 / gt_hz : 0.0;

  for (const auto & joint : info_.joints) {
    // Same contract as the MD80 driver: [position] is the plain impedance
    // servo, [position, effort] adds the feed-forward torque channel
    // (tau_ff policies; wojtek_sim.urdf.xacro tau_ff:=true).
    const auto & cmd_ifaces = joint.command_interfaces;
    const bool plain = cmd_ifaces.size() == 1 &&
      cmd_ifaces[0].name == hardware_interface::HW_IF_POSITION;
    const bool with_ff = cmd_ifaces.size() == 2 &&
      cmd_ifaces[0].name == hardware_interface::HW_IF_POSITION &&
      cmd_ifaces[1].name == hardware_interface::HW_IF_EFFORT;
    if (!plain && !with_ff) {
      RCLCPP_FATAL(
        logger_,
        "joint '%s' needs [position] or [position, effort] command interfaces",
        joint.name.c_str());
      return hardware_interface::CallbackReturn::ERROR;
    }
    joint_names_.push_back(joint.name);
    has_tau_ff_.push_back(with_ff);
  }
  position_.assign(joint_names_.size(), 0.0);
  velocity_.assign(joint_names_.size(), 0.0);
  effort_.assign(joint_names_.size(), 0.0);
  command_.assign(joint_names_.size(), 0.0);
  effort_command_.assign(joint_names_.size(), 0.0);

  if (info_.sensors.size() > 1) {
    RCLCPP_FATAL(logger_, "at most one sensor (the IMU) is supported");
    return hardware_interface::CallbackReturn::ERROR;
  }
  if (!info_.sensors.empty()) {
    has_imu_ = true;
    imu_sensor_name_ = info_.sensors[0].name;
    imu_states_.assign(kImuSlots.size(), 0.0);
    for (const auto & iface : info_.sensors[0].state_interfaces) {
      if (kImuSlots.count(iface.name) == 0) {
        RCLCPP_FATAL(
          logger_, "sensor '%s' exports unknown interface '%s'",
          imu_sensor_name_.c_str(), iface.name.c_str());
        return hardware_interface::CallbackReturn::ERROR;
      }
    }
  }
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn MujocoHardwareInterface::on_configure(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  try {
    plant_.load(scene_xml_, mesh_dir_);
    plant_.setDryRun(dry_run_);

    actuator_of_joint_.clear();
    const auto & actuators = plant_.actuatorNames();
    for (const auto & joint : joint_names_) {
      const auto it = std::find(actuators.begin(), actuators.end(), joint);
      if (it == actuators.end()) {
        RCLCPP_FATAL(
          logger_, "the model has no actuator for joint '%s'", joint.c_str());
        return hardware_interface::CallbackReturn::ERROR;
      }
      actuator_of_joint_.push_back(static_cast<int>(std::distance(actuators.begin(), it)));
    }

    // The servo contract travels with the policy (policy_meta.json pd block)
    // and reaches us through the xacro, exactly as it reaches the MD80s.
    // max_torque arrives as the DRIVE cap (pd max_torque + tau_ff.scale, see
    // launch_common), while the training sim clamps the PD servo and the
    // feed-forward head separately -- take the scale back off the servo and
    // keep it as the head's own hard bound.
    tau_ff_scale_.assign(joint_names_.size(), 0.0);
    for (std::size_t i = 0; i < joint_names_.size(); ++i) {
      const auto & params = info_.joints[i].parameters;
      const auto kp = params.find("ddq_kp");
      const auto kd = params.find("ddq_kd");
      const auto max_torque = params.find("max_torque");
      if (kp == params.end() || kd == params.end() || max_torque == params.end()) {
        RCLCPP_FATAL(
          logger_, "joint '%s' is missing ddq_kp/ddq_kd/max_torque",
          joint_names_[i].c_str());
        return hardware_interface::CallbackReturn::ERROR;
      }
      double scale = 0.0;
      if (has_tau_ff_[i]) {
        const auto it = params.find("tau_ff_scale");
        if (it == params.end()) {
          RCLCPP_FATAL(
            logger_, "joint '%s' has an effort channel but no tau_ff_scale",
            joint_names_[i].c_str());
          return hardware_interface::CallbackReturn::ERROR;
        }
        scale = std::stod(it->second);
      }
      tau_ff_scale_[i] = scale;
      plant_.applyServoSettings(
        joint_names_[i],
        {std::stod(kp->second), std::stod(kd->second),
          std::stod(max_torque->second) - scale});
    }
    plant_.reset(initial_pose_);
  } catch (const std::exception & e) {
    RCLCPP_FATAL(logger_, "%s", e.what());
    return hardware_interface::CallbackReturn::ERROR;
  }

  if (ground_truth_period_ > 0.0) {
    node_ = std::make_shared<rclcpp::Node>("wojtek_sim_ground_truth");
    tf_ = std::make_unique<tf2_ros::TransformBroadcaster>(node_);
    vel_pub_ = node_->create_publisher<geometry_msgs::msg::Twist>("odom_vel", 10);
    rtf_pub_ = node_->create_publisher<std_msgs::msg::Float32>("sim/rtf", 10);
    // Everything mirroring this physics state (the camera renderer) sets its
    // own model from here, so there is one source of truth for the pose.
    qpos_pub_ = node_->create_publisher<std_msgs::msg::Float64MultiArray>(
      "sim/qpos", rclcpp::SensorDataQoS());
    qpos_msg_.data.resize(plant_.qpos().size());
  }

  RCLCPP_INFO(
    logger_,
    "simulating %s at dt=%.4f s, initial pose '%s'%s",
    scene_xml_.c_str(), plant_.timestep(), initial_pose_.c_str(),
    dry_run_ ? " (dry run: no actuator force)" : "");
  return hardware_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface>
MujocoHardwareInterface::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> interfaces;
  for (std::size_t i = 0; i < joint_names_.size(); ++i) {
    interfaces.emplace_back(
      joint_names_[i], hardware_interface::HW_IF_POSITION, &position_[i]);
    interfaces.emplace_back(
      joint_names_[i], hardware_interface::HW_IF_VELOCITY, &velocity_[i]);
    interfaces.emplace_back(
      joint_names_[i], hardware_interface::HW_IF_EFFORT, &effort_[i]);
  }
  if (has_imu_) {
    for (const auto & [name, slot] : kImuSlots) {
      interfaces.emplace_back(imu_sensor_name_, name, &imu_states_[slot]);
    }
  }
  return interfaces;
}

std::vector<hardware_interface::CommandInterface>
MujocoHardwareInterface::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> interfaces;
  for (std::size_t i = 0; i < joint_names_.size(); ++i) {
    interfaces.emplace_back(
      joint_names_[i], hardware_interface::HW_IF_POSITION, &command_[i]);
    if (has_tau_ff_[i]) {
      interfaces.emplace_back(
        joint_names_[i], hardware_interface::HW_IF_EFFORT, &effort_command_[i]);
    }
  }
  return interfaces;
}

hardware_interface::CallbackReturn MujocoHardwareInterface::on_activate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  // Powering the drives up in whatever pose the robot is in: from here on
  // states are reported against this pose and commands are read in it, which
  // is what real_io_node's zero/offset logic is written against.
  plant_.latchActivationPose();
  std::fill(position_.begin(), position_.end(), 0.0);
  std::fill(velocity_.begin(), velocity_.end(), 0.0);
  std::fill(effort_.begin(), effort_.end(), 0.0);
  std::fill(command_.begin(), command_.end(), 0.0);
  std::fill(effort_command_.begin(), effort_command_.end(), 0.0);
  RCLCPP_INFO(logger_, "activated: current pose is the zero for joint states");
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn MujocoHardwareInterface::on_deactivate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::return_type MujocoHardwareInterface::read(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & period)
{
  const double seconds = period.seconds();
  plant_.advance(seconds);

  for (std::size_t i = 0; i < joint_names_.size(); ++i) {
    const int actuator = actuator_of_joint_[i];
    const auto & name = joint_names_[i];
    // Boot-relative, URDF convention: the offset cancels in a difference, so
    // only the sign carries over.
    const double s = joint_map_->sign(name);
    position_[i] = s * plant_.jointPosition(actuator);
    velocity_[i] = s * plant_.jointVelocity(actuator);
    effort_[i] = s * plant_.jointEffort(actuator);
  }

  if (has_imu_) {
    const auto sample = plant_.imu(mount_quat_, earth_field_ut_);
    for (int axis = 0; axis < 3; ++axis) {
      imu_states_[0 + axis] = sample.magnetometer[axis];
      imu_states_[3 + axis] = sample.angular_velocity[axis];
      imu_states_[6 + axis] = sample.linear_acceleration[axis];
    }
    for (int i = 0; i < 4; ++i) {
      imu_states_[9 + i] = sample.orientation[i];
    }
  }

  publishGroundTruth(period);
  return hardware_interface::return_type::OK;
}

hardware_interface::return_type MujocoHardwareInterface::write(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  for (std::size_t i = 0; i < joint_names_.size(); ++i) {
    const double command = command_[i];
    if (!std::isfinite(command)) {
      continue;  // NaN placeholders right after activation: hold position.
    }
    // Commands arrive boot-relative in URDF convention; undo the sign to get
    // back to the model's, where the activation pose is added.
    plant_.setCommand(
      actuator_of_joint_[i], command / joint_map_->sign(joint_names_[i]));
  }
  for (std::size_t i = 0; i < joint_names_.size(); ++i) {
    if (!has_tau_ff_[i]) {
      continue;
    }
    // Feed-forward torque on top of the servo. A non-finite command (a
    // controller activated before its first message) must never reach the
    // plant; the clamp is the head's hard authority bound, the same clip
    // the training env applies before qfrc_applied.
    double tau = effort_command_[i];
    tau = std::isfinite(tau) ?
      std::clamp(tau, -tau_ff_scale_[i], tau_ff_scale_[i]) : 0.0;
    plant_.setFeedForward(
      actuator_of_joint_[i], tau / joint_map_->sign(joint_names_[i]));
  }
  return hardware_interface::return_type::OK;
}

void MujocoHardwareInterface::publishGroundTruth(const rclcpp::Duration & period)
{
  if (!node_) {
    return;
  }
  // NOT read()'s time argument: controller_manager runs its cycles off the
  // steady clock, so that time is monotonic-since-boot while every other
  // publisher in the graph (the broadcasters, real_io_node, robot_state_
  // publisher) stamps in ROS time. Mixing the two silently breaks TF -- any
  // lookup crossing odom->base_link plus a joint transform lands in a
  // different epoch and comes back as ExtrapolationException, which reads
  // like "half the tree is missing".
  const auto stamp = node_->now();
  wall_in_window_ += period.seconds();
  since_ground_truth_ += period.seconds();
  if (since_ground_truth_ < ground_truth_period_) {
    return;
  }
  since_ground_truth_ = 0.0;

  const auto pose = plant_.basePose();
  geometry_msgs::msg::TransformStamped tf;
  tf.header.stamp = stamp;
  tf.header.frame_id = "odom";
  tf.child_frame_id = "base_link";
  tf.transform.translation.x = pose[0];
  tf.transform.translation.y = pose[1];
  tf.transform.translation.z = pose[2];
  tf.transform.rotation.w = pose[3];
  tf.transform.rotation.x = pose[4];
  tf.transform.rotation.y = pose[5];
  tf.transform.rotation.z = pose[6];
  tf_->sendTransform(tf);

  const auto twist = plant_.baseVelocity();
  geometry_msgs::msg::Twist vel;
  vel.linear.x = twist[0];
  vel.linear.y = twist[1];
  vel.linear.z = twist[2];
  vel.angular.x = twist[3];
  vel.angular.y = twist[4];
  vel.angular.z = twist[5];
  vel_pub_->publish(vel);

  const auto qpos = plant_.qpos();
  qpos_msg_.data.assign(qpos.begin(), qpos.end());
  qpos_pub_->publish(qpos_msg_);

  // Real-time factor over the window: how much simulated time we managed per
  // second of wall clock. Below 1.0 means physics is not keeping up.
  if (wall_in_window_ >= 1.0) {
    std_msgs::msg::Float32 rtf;
    rtf.data = static_cast<float>((plant_.simTime() - sim_time_at_window_) /
      wall_in_window_);
    rtf_pub_->publish(rtf);
    sim_time_at_window_ = plant_.simTime();
    wall_in_window_ = 0.0;
  }
}

}  // namespace wojtek_mujoco_hardware_interface

PLUGINLIB_EXPORT_CLASS(
  wojtek_mujoco_hardware_interface::MujocoHardwareInterface,
  hardware_interface::SystemInterface)
