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
#include <string>
#include <unordered_map>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace
{
double param_or(
  const std::unordered_map<std::string, std::string> & params, const std::string & key,
  double fallback)
{
  const auto it = params.find(key);
  return it == params.end() ? fallback : std::stod(it->second);
}

// Loads the ellipsoid calibration produced by ros/hw_tests/imu_i2c/
// mag_calib.py.  Keys: hard_iron_ut (3), soft_iron (row-major 9),
// field_norm_ut (scalar).  Missing keys keep the identity/learn-at-init
// defaults, so an absent or placeholder file still yields a working
// (merely uncalibrated) filter.
void load_mag_calib(const std::string & path, OrientationEskf::Params & p)
{
  const YAML::Node root = YAML::LoadFile(path);
  if (const auto hard = root["hard_iron_ut"]) {
    for (int i = 0; i < 3; ++i) {
      p.hard_iron(i) = hard[i].as<double>();
    }
  }
  if (const auto soft = root["soft_iron"]) {
    for (int i = 0; i < 9; ++i) {
      p.soft_iron(i / 3, i % 3) = soft[i].as<double>();
    }
  }
  if (const auto norm = root["field_norm_ut"]) {
    p.field_norm = norm.as<double>();
  }
}
}  // namespace

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

  // ESKF tuning: every knob is optional, defaults live in
  // OrientationEskf::Params (documented in orientation_eskf.hpp).
  OrientationEskf::Params eskf_params;
  const auto & hp = info_.hardware_parameters;
  eskf_params.gyro_noise = param_or(hp, "eskf_gyro_noise", eskf_params.gyro_noise);
  eskf_params.gyro_bias_walk = param_or(hp, "eskf_gyro_bias_walk", eskf_params.gyro_bias_walk);
  eskf_params.accel_noise = param_or(hp, "eskf_accel_noise", eskf_params.accel_noise);
  eskf_params.accel_norm_tol = param_or(hp, "eskf_accel_norm_tol", eskf_params.accel_norm_tol);
  eskf_params.mag_noise = param_or(hp, "eskf_mag_noise", eskf_params.mag_noise);
  eskf_params.mag_norm_tol = param_or(hp, "eskf_mag_norm_tol", eskf_params.mag_norm_tol);
  eskf_params.mag_mahalanobis_gate =
    param_or(hp, "eskf_mag_mahalanobis_gate", eskf_params.mag_mahalanobis_gate);

  if (const auto it = hp.find("mag_calib_file"); it != hp.end() && !it->second.empty()) {
    try {
      load_mag_calib(it->second, eskf_params);
      RCLCPP_INFO(logger_, "Loaded magnetometer calibration from %s", it->second.c_str());
    } catch (const std::exception & e) {
      RCLCPP_ERROR(
        logger_, "Failed to load mag calibration '%s': %s", it->second.c_str(), e.what());
      return hardware_interface::CallbackReturn::ERROR;
    }
  }
  eskf_ = std::make_unique<OrientationEskf>(eskf_params);

  hw_states_.resize(13, std::numeric_limits<double>::quiet_NaN());
  // orientation.{x,y,z,w}: all-zero "no fusion yet" marker until the ESKF
  // initializes, see the header.
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
  const rclcpp::Time & /*time*/, const rclcpp::Duration & period)
{
  gyro_dt_ += period.seconds();

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

  if (d.mag_fresh) {
    hw_states_[0] = d.mag_x;
    hw_states_[1] = d.mag_y;
    hw_states_[2] = d.mag_z;
    mag_pending_ = true;
  }
  if (d.accel_gyro_fresh) {
    hw_states_[3] = d.gyro_x;
    hw_states_[4] = d.gyro_y;
    hw_states_[5] = d.gyro_z;
    hw_states_[6] = d.accel_x;
    hw_states_[7] = d.accel_y;
    hw_states_[8] = d.accel_z;

    // Fuse on the gyro cadence; a mag sample that landed on an earlier
    // cycle (80 vs 104 Hz) was latched via mag_pending_.
    eskf_->step(
      {d.gyro_x, d.gyro_y, d.gyro_z}, gyro_dt_, true, {d.accel_x, d.accel_y, d.accel_z},
      mag_pending_, {hw_states_[0], hw_states_[1], hw_states_[2]});
    gyro_dt_ = 0.0;
    mag_pending_ = false;

    if (eskf_->initialized()) {
      const Eigen::Quaterniond q = eskf_->quat();
      hw_states_[9] = q.x();
      hw_states_[10] = q.y();
      hw_states_[11] = q.z();
      hw_states_[12] = q.w();
    }
  }

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
