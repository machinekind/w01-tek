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

#include "md80_hardware_interface/md80_hardware_interface.hpp"

#include <chrono>
#include <limits>
#include <thread>
#include <vector>

#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "rclcpp/rclcpp.hpp"

namespace md80_hardware_interface
{

// A single dropped USB transaction to the CANdle fails one addMd80(), which
// used to abort the whole bring-up. Measured on the real robot: ~5% of calls
// time out, hitting a different drive every run (observed on ids 11, 22, 30,
// 41, 42), so roughly every second launch died with a healthy robot. Retrying
// is safe -- Candle::addMd80 returns true for an id already on its update list.
constexpr int ADD_MD80_ATTEMPTS = 3;
constexpr auto ADD_MD80_RETRY_DELAY = std::chrono::milliseconds(50);
hardware_interface::CallbackReturn MD80HardwareInterface::on_init(
  const hardware_interface::HardwareInfo & info)
{
  if (hardware_interface::SystemInterface::on_init(info) != CallbackReturn::SUCCESS) {
    return CallbackReturn::ERROR;
  }

  const auto motors_size = info_.joints.size();
  RCLCPP_INFO_STREAM(rclcpp::get_logger(get_name()), "Requesting " << motors_size << " motors.");
  md80_info_.resize(motors_size);
  initial_positions_.resize(motors_size);

  return CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn MD80HardwareInterface::on_configure(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  try {
    add_candle_instances();
    try_to_initialize_motors();
    set_config_to_md80();
  } catch (const std::runtime_error & e) {
    RCLCPP_ERROR_STREAM(rclcpp::get_logger(get_name()), "Got error: " << e.what());
    return CallbackReturn::FAILURE;
  }

  set_modes();

  return CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface> MD80HardwareInterface::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> state_interfaces;
  for (size_t i = 0; i < info_.joints.size(); ++i) {
    state_interfaces.emplace_back(hardware_interface::StateInterface(
      info_.joints[i].name, hardware_interface::HW_IF_POSITION, &md80_info_[i].state.position));
    state_interfaces.emplace_back(hardware_interface::StateInterface(
      info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &md80_info_[i].state.velocity));
    state_interfaces.emplace_back(hardware_interface::StateInterface(
      info_.joints[i].name, hardware_interface::HW_IF_EFFORT, &md80_info_[i].state.effort));
  }

  return state_interfaces;
}

std::vector<hardware_interface::CommandInterface> MD80HardwareInterface::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> command_interfaces;
  for (size_t i = 0; i < info_.joints.size(); ++i) {
    if (info_.joints[i].command_interfaces.size() != 1) {
      RCLCPP_ERROR_STREAM(rclcpp::get_logger(get_name()), "Too many command interfaced defines!");
      return {};
    }
    const auto control_mode = info_.joints[i].command_interfaces[0].name;

    if (control_mode == "position") {
      command_interfaces.emplace_back(hardware_interface::CommandInterface(
        info_.joints[i].name, hardware_interface::HW_IF_POSITION, &md80_info_[i].command.position));

      md80_info_[i].control_mode = mab::Md80Mode_E::IMPEDANCE;
    } else if (control_mode == "velocity") {
      command_interfaces.emplace_back(hardware_interface::CommandInterface(
        info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &md80_info_[i].command.velocity));
      md80_info_[i].control_mode = mab::Md80Mode_E::VELOCITY_PID;
    } else if (control_mode == "effort") {
      command_interfaces.emplace_back(hardware_interface::CommandInterface(
        info_.joints[i].name, hardware_interface::HW_IF_EFFORT, &md80_info_[i].command.effort));
      md80_info_[i].control_mode = mab::Md80Mode_E::RAW_TORQUE;
    } else {
      RCLCPP_ERROR_STREAM(rclcpp::get_logger(get_name()), "Unknown control mode: " << control_mode);
      throw std::runtime_error("Unknown control mode");
    }
  }

  return command_interfaces;
}

hardware_interface::CallbackReturn MD80HardwareInterface::on_activate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  //  TODO: Add service to zero encoders
  // zero_encoders();
  std::size_t i = 0;
  read(rclcpp::Time{}, rclcpp::Duration(0, 0));
  for (auto candle : candle_instances) {
    for (auto & md : candle->md80s) {
      initial_positions_[i] = md80_info_[i].state.position;
      ++i;
    }
  }
  reset_command();
  log_current_joint_position();

  write(rclcpp::Time{}, rclcpp::Duration(0, 0));
  enable_motors();

  return CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn MD80HardwareInterface::on_deactivate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  disable_motors();
  return CallbackReturn::SUCCESS;
}

hardware_interface::return_type MD80HardwareInterface::read(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  std::size_t i = 0;
  for (auto candle : candle_instances) {
    for (auto & md : candle->md80s) {
      md80_info_[i].state.position = md.getPosition() - initial_positions_[i];
      md80_info_[i].state.velocity = md.getVelocity();
      md80_info_[i].state.effort = md.getTorque();

      // RCLCPP_INFO_STREAM(rclcpp::get_logger(get_name()),
      //                    "can id " << i << " status: " << md.getQuickStatus());
      ++i;
    }
  }
  return hardware_interface::return_type::OK;
}

hardware_interface::return_type MD80HardwareInterface::write(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  std::size_t i = 0;
  for (auto candle : candle_instances) {
    for (auto & md : candle->md80s) {
      const auto & control_mode = md80_info_[i].control_mode;
      if (control_mode == mab::Md80Mode_E::POSITION_PID) {
        md.setTargetPosition(md80_info_[i].command.position - initial_positions_[i]);
      } else if (control_mode == mab::Md80Mode_E::VELOCITY_PID) {
        md.setTargetVelocity(md80_info_[i].command.velocity);
      } else if (control_mode == mab::Md80Mode_E::IMPEDANCE) {
        // Commands must live in the same frame as the states read() reports
        // (activation-relative): shift back into the drive's raw frame.
        md.setTargetPosition(md80_info_[i].command.position + initial_positions_[i]);
      } else if (control_mode == mab::Md80Mode_E::RAW_TORQUE) {
        md.setTargetTorque(md80_info_[i].command.effort);
      }
      ++i;
    }
  }

  return hardware_interface::return_type::OK;
}

void MD80HardwareInterface::parse_urdf_joint_info(
  MD80Info & md80, const hardware_interface::ComponentInfo & info)
{
  const auto can_id = std::stoi(info.parameters.at("can_id"));

  const auto q_kp = std::stof(info.parameters.at("q_kp"));
  const auto q_ki = std::stof(info.parameters.at("q_ki"));
  const auto q_kd = std::stof(info.parameters.at("q_kd"));
  const auto q_windup = std::stof(info.parameters.at("q_windup"));

  const auto dq_kp = std::stof(info.parameters.at("dq_kp"));
  const auto dq_ki = std::stof(info.parameters.at("dq_ki"));
  const auto dq_kd = std::stof(info.parameters.at("dq_kd"));
  const auto dq_windup = std::stof(info.parameters.at("dq_windup"));

  const auto ddq_kp = std::stof(info.parameters.at("ddq_kp"));
  const auto ddq_kd = std::stof(info.parameters.at("ddq_kd"));

  const auto max_torque = std::stof(info.parameters.at("max_torque"));

  md80.can_id = can_id;
  md80.q_pid.kp = q_kp;
  md80.q_pid.ki = q_ki;
  md80.q_pid.kd = q_kd;
  md80.q_pid.windup = q_windup;

  md80.dq_pid.kp = dq_kp;
  md80.dq_pid.ki = dq_ki;
  md80.dq_pid.kd = dq_kd;
  md80.dq_pid.windup = dq_windup;

  md80.ddq_pid.kp = ddq_kp;
  md80.ddq_pid.kd = ddq_kd;

  md80.max_torque = max_torque;
}

std::shared_ptr<mab::Candle> MD80HardwareInterface::find_candle_by_motor_can_id(uint16_t can_id)
{
  for (auto & candle : candle_instances) {
    for (auto id : candle->md80s) {
      if (id.getId() == can_id) return candle;
    }
  }
  return nullptr;
}

void MD80HardwareInterface::add_candle_instances()
{
  // Bus + baud come from the URDF <hardware> params (bus: usb|spi, can_baud:
  // Mbps as 1|2|5|8). The drives' configured baudrate must match can_baud --
  // as of 2026-07-17 they are flashed to 8M for the CANdle HAT (SPI); the
  // legacy USB dongle path expects 1M unless the drives are flashed back.
  const auto & params = info_.hardware_parameters;
  const std::string bus_name = params.count("bus") ? params.at("bus") : "usb";

  mab::BusType_E bus;
  std::string device;  // empty -> candle's per-bus default (SPI: /dev/spidev0.0)
  if (bus_name == "spi") {
    bus = mab::BusType_E::SPI;
    if (params.count("spi_device")) device = params.at("spi_device");
  } else if (bus_name == "usb") {
    bus = mab::BusType_E::USB;
    device = params.at("usb_port");
  } else {
    throw std::runtime_error("Unsupported bus type '" + bus_name + "' (usb|spi)");
  }

  const std::string baud_str = params.count("can_baud") ? params.at("can_baud") : "1";
  if (baud_str != "1" && baud_str != "2" && baud_str != "5" && baud_str != "8") {
    throw std::runtime_error("Unsupported can_baud '" + baud_str + "' (1|2|5|8 Mbps)");
  }
  const auto baud = static_cast<mab::CANdleBaudrate_E>(std::stoi(baud_str));

  const std::string where =
    bus_name + (device.empty() ? std::string(" (default device)") : " " + device) +
    " @ " + baud_str + "M";
  try {
    candle_instances.emplace_back(std::make_shared<mab::Candle>(baud, true, bus, device));

    RCLCPP_INFO_STREAM(
      rclcpp::get_logger(get_name()),
      "Found CANdle with ID: " << candle_instances.back()->getDeviceId() << " at " << where);
  } catch (const char * eMsg) {
    throw std::runtime_error(std::string(eMsg) + " at " + where);
  }
}

void MD80HardwareInterface::try_to_initialize_motors()
{
  unsigned int found_devices = 0;
  for (const auto & joint_info : info_.joints) {
    int can_id = -1;

    try {
      can_id = std::stoi(joint_info.parameters.at("can_id"));
    } catch (const std::out_of_range & e) {
      throw std::runtime_error(
        "can_id param is not defined in joint " + joint_info.name +
        ". Use <param name=\"can_id\"></param>.");
    }

    RCLCPP_INFO_STREAM(rclcpp::get_logger(get_name()), "Check connection for can_id: " << can_id);

    bool added = false;
    for (int attempt = 1; attempt <= ADD_MD80_ATTEMPTS && !added; ++attempt) {
      for (auto & candle : candle_instances) {
        if (candle->addMd80(can_id, false)) {
          parse_urdf_joint_info(std::ref(md80_info_[found_devices]), joint_info);
          found_devices++;
          added = true;
          break;
        }
      }
      if (!added && attempt < ADD_MD80_ATTEMPTS) {
        RCLCPP_WARN_STREAM(
          rclcpp::get_logger(get_name()),
          "No answer from can_id " << can_id << " (attempt " << attempt << "/"
                                   << ADD_MD80_ATTEMPTS << ") -- retrying");
        std::this_thread::sleep_for(ADD_MD80_RETRY_DELAY);
      }
    }

    if (!added) {
      // Deliberately not "cannot find device": addMd80 also returns false when
      // the link to the CANdle times out, and the old wording sent everyone
      // hunting a drive that was fine. Name both possibilities. A third one
      // since the 8M migration: the drive's flashed baudrate does not match
      // the stack's can_baud param.
      throw std::runtime_error(
        "No answer from CAN id " + std::to_string(can_id) + " after " +
        std::to_string(ADD_MD80_ATTEMPTS) +
        " attempts: the drive is absent/unpowered, on a different CAN baudrate "
        "than can_baud, or the CANdle link timed out "
        "(look for 'Did not receive response' above).");
    }

    RCLCPP_INFO_STREAM(rclcpp::get_logger(get_name()), "Initialized motor at can_id: " << can_id);
  }
}

void MD80HardwareInterface::set_config_to_md80()
{
  std::size_t i = 0;
  for (auto candle : candle_instances) {
    for (auto & md : candle->md80s) {
      auto & md80_info = md80_info_[i];

      RCLCPP_INFO_STREAM(
        rclcpp::get_logger(get_name()), "Setting PIDs for motor at can_id: " << md80_info.can_id);

      md.setPositionControllerParams(
        md80_info.q_pid.kp, md80_info.q_pid.ki, md80_info.q_pid.kd, md80_info.q_pid.windup);
      md.setVelocityControllerParams(
        md80_info.dq_pid.kp, md80_info.dq_pid.ki, md80_info.dq_pid.kd, md80_info.dq_pid.windup);
      md.setImpedanceControllerParams(md80_info.ddq_pid.kp, md80_info.ddq_pid.kd);

      md.setMaxTorque(md80_info.max_torque);

      RCLCPP_INFO_STREAM(
        rclcpp::get_logger(get_name()), "PIDs is set motor at can_id: " << md80_info.can_id);

      ++i;
    }
  }
}

void MD80HardwareInterface::set_modes()
{
  for (auto & md80 : md80_info_) {
    auto candle = find_candle_by_motor_can_id(md80.can_id);

    RCLCPP_INFO_STREAM(
      rclcpp::get_logger(get_name()), "Initializing motor at can_id: " << md80.can_id);

    if (candle == nullptr) {
      throw std::runtime_error(
        std::string("\nCould not find instance for can_id: ") + std::to_string(md80.can_id));
    }

    candle->controlMd80Mode(md80.can_id, md80.control_mode);
    RCLCPP_INFO_STREAM(
      rclcpp::get_logger(get_name()), "Mode is set motor at can_id: " << md80.can_id);
  }
}

void MD80HardwareInterface::zero_encoders()
{
  for (auto & md80 : md80_info_) {
    auto candle = find_candle_by_motor_can_id(md80.can_id);
    candle->controlMd80SetEncoderZero(md80.can_id);
  }
}

void MD80HardwareInterface::enable_motors()
{
  for (auto & md80 : md80_info_) {
    auto candle = find_candle_by_motor_can_id(md80.can_id);
    candle->controlMd80Enable(md80.can_id, true);
  }

  for (auto & candle : candle_instances) {
    candle->begin();
  }
}

void MD80HardwareInterface::disable_motors()
{
  for (auto & md80 : md80_info_) {
    auto candle = find_candle_by_motor_can_id(md80.can_id);
    candle->controlMd80Enable(md80.can_id, false);
  }
}

void MD80HardwareInterface::reset_command()
{
  for (auto & md80_info : md80_info_) {
    md80_info.command.position = md80_info.state.position;
    md80_info.command.velocity = 0.0;
    md80_info.command.effort = 0.0;
  }
}

void MD80HardwareInterface::log_current_joint_position()
{
  for (auto & md80_info : md80_info_) {
    RCLCPP_INFO_STREAM(
      rclcpp::get_logger(get_name()),
      "can id " << md80_info.can_id << " position " << md80_info.state.position);
  }
}

}  // namespace md80_hardware_interface

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(
  md80_hardware_interface::MD80HardwareInterface, hardware_interface::SystemInterface)
