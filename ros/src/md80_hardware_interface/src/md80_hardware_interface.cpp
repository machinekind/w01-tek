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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
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
// First contact with each drive at configure: retry for first_contact_s
// seconds (URDF <hardware> param, default 5.0) every ADD_MD80_RETRY_DELAY.
// A drive that is still booting after a motor power-up, or a bus that is
// recovering from a dropout, answers within a few seconds; the old 3 x 50 ms
// gave up before that and the whole stack died, to be restarted by the
// on-robot watchdog minutes later (seen 2026-10-01: can_id 10 "absent" on
// three consecutive starts, present on the fourth).
constexpr auto ADD_MD80_RETRY_DELAY = std::chrono::milliseconds(100);
constexpr double ADD_MD80_FIRST_CONTACT_S_DEFAULT = 5.0;
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

  const auto & params = info_.hardware_parameters;
  if (params.count("dry_run")) {
    const auto & v = params.at("dry_run");
    dry_run_ = (v == "true" || v == "True" || v == "1");
  }

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

  const auto & params = info_.hardware_parameters;
  link_stall_cycles_ =
    params.count("link_stall_cycles") ? std::stoul(params.at("link_stall_cycles")) : 200;

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
    const auto & cmd_ifaces = info_.joints[i].command_interfaces;
    // [position] -> IMPEDANCE (PD on the drive), the default.
    // [position, effort] -> IMPEDANCE with a feed-forward torque channel:
    //   the drive runs tau = kp*e - kd*dq + tau_ff, tau_ff from the effort
    //   command interface (tau_ff policies; wojtek_ros2_control.urdf.xacro
    //   tau_ff:=true). effort starts at 0.0 (JointInfo default), so the
    //   channel is inert until a controller writes it.
    if (cmd_ifaces.size() == 2) {
      if (cmd_ifaces[0].name != "position" || cmd_ifaces[1].name != "effort") {
        RCLCPP_ERROR_STREAM(
          rclcpp::get_logger(get_name()),
          "Two command interfaces must be [position, effort], got ["
            << cmd_ifaces[0].name << ", " << cmd_ifaces[1].name << "]!");
        return {};
      }
      command_interfaces.emplace_back(hardware_interface::CommandInterface(
        info_.joints[i].name, hardware_interface::HW_IF_POSITION, &md80_info_[i].command.position));
      command_interfaces.emplace_back(hardware_interface::CommandInterface(
        info_.joints[i].name, hardware_interface::HW_IF_EFFORT, &md80_info_[i].command.effort));
      md80_info_[i].control_mode = mab::Md80Mode_E::IMPEDANCE;
      md80_info_[i].has_effort_ff = true;
      continue;
    }
    if (cmd_ifaces.size() != 1) {
      RCLCPP_ERROR_STREAM(rclcpp::get_logger(get_name()), "Too many command interfaced defines!");
      return {};
    }
    const auto control_mode = cmd_ifaces[0].name;

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
  // Commands live in the activation-relative frame (write() adds
  // initial_positions_ back). The read() above ran with initial_positions_
  // still zero, so state -- and a command seeded from it -- would be in the
  // drive's RAW frame; write() would then send raw + initial = 2x the raw
  // position and the robot would actively jump instead of holding. Re-read
  // now that the offsets are set: state ~= 0, so the seeded hold command is
  // "stay where you are".
  read(rclcpp::Time{}, rclcpp::Duration(0, 0));
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
  std::vector<double> raw;
  raw.reserve(3 * md80_info_.size());
  for (auto candle : candle_instances) {
    for (auto & md : candle->md80s) {
      const double pos = md.getPosition();
      const double vel = md.getVelocity();
      const double tau = md.getTorque();
      md80_info_[i].state.position = pos - initial_positions_[i];
      md80_info_[i].state.velocity = vel;
      md80_info_[i].state.effort = tau;
      raw.push_back(pos);
      raw.push_back(vel);
      raw.push_back(tau);
      ++i;
    }
  }

  // Silent-bus detector. The CANdle library only updates a drive's state
  // when that drive's response frame arrives; when the drives stop
  // answering (bus power sag, CAN fault, drives reset) it keeps handing back
  // the last values without any error -- measured 2026-09-26: an armed,
  // standing robot collapsed, the controller manager stayed up, and the
  // joint states were bit-identical for minutes. Live drives never are:
  // twelve torque readbacks carry ADC noise every frame. So a raw state
  // identical across ALL drives for link_stall_cycles consecutive cycles
  // (URDF <hardware> param, default 200 = 1 s at 200 Hz; 0 disables) is a
  // lost link. Returning ERROR makes the controller manager log "read cycle
  // resulted in an error" and drop this component, which is what the
  // on-robot CM watchdog (deploy/rpi/wojtek-cm-watchdog.sh) waits for: it
  // stops the service and starts it again once the motor power has been
  // cycled and the drives answer. Skipped in dry_run (disabled drives can sit
  // at exact zeros) and until the update loop has been started.
  if (update_loop_running_ && !dry_run_ && link_stall_cycles_ > 0) {
    if (raw == last_raw_) {
      if (++stall_cycles_ >= link_stall_cycles_) {
        RCLCPP_ERROR_STREAM(
          rclcpp::get_logger(get_name()),
          "Drive link lost: no fresh data from any of the " << md80_info_.size()
            << " drives for " << stall_cycles_ << " cycles");
        stall_cycles_ = 0;
        update_loop_running_ = false;
        return hardware_interface::return_type::ERROR;
      }
    } else {
      stall_cycles_ = 0;
    }
  }
  // Diagnostic snapshots (env MD80_SNAPSHOT=1): a few cycles after the
  // update loop starts, log every drive's raw position, the raw target the
  // stream carries, the torque readback and the drive's quick-status word.
  // Tells enabled-and-holding from enabled-but-idle (CAN watchdog tripped:
  // positions stream, torque stays 0) without any CLI on the loaded Pi.
  static const bool snapshot_on = std::getenv("MD80_SNAPSHOT") != nullptr;
  if (snapshot_on && update_loop_running_) {
    static std::size_t cyc = 0;
    ++cyc;
    if (cyc == 20 || cyc == 100 || cyc == 400 || cyc == 1000 || cyc % 2000 == 0) {
      std::size_t k = 0;
      for (auto candle : candle_instances) {
        for (auto & md : candle->md80s) {
          RCLCPP_INFO(
            rclcpp::get_logger(get_name()),
            "snapshot cyc %zu can_id %d raw_pos %.4f raw_target %.4f tau %.3f qstatus 0x%04x",
            cyc, md80_info_[k].can_id, md.getPosition(),
            md80_info_[k].command.position + initial_positions_[k], md.getTorque(),
            md.getQuickStatus());
          ++k;
        }
      }
    }
  }
  last_raw_ = std::move(raw);
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
        if (md80_info_[i].has_effort_ff) {
          // Feed-forward torque on top of the drive's PD law (frame-free:
          // a torque needs no activation offset). A non-finite command --
          // e.g. a controller activated before its first command -- must
          // never reach the drive; fall back to 0 (pure PD).
          const double tau = md80_info_[i].command.effort;
          md.setTargetTorque(std::isfinite(tau) ? tau : 0.0);
        }
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
    const auto & hw_params = info_.hardware_parameters;
    const double first_contact_s = hw_params.count("first_contact_s")
                                     ? std::stod(hw_params.at("first_contact_s"))
                                     : ADD_MD80_FIRST_CONTACT_S_DEFAULT;
    const int ADD_MD80_ATTEMPTS = std::max(
      1, static_cast<int>(first_contact_s * 1000.0 / ADD_MD80_RETRY_DELAY.count()));
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
        // One line per second, not per attempt: ten lines per silent second
        // would bury the journal during a slow motor power-up.
        if (attempt % 10 == 1) {
          RCLCPP_WARN_STREAM(
            rclcpp::get_logger(get_name()),
            "No answer from can_id " << can_id << " (attempt " << attempt << "/"
                                     << ADD_MD80_ATTEMPTS << ", waiting up to "
                                     << first_contact_s << " s) -- retrying");
        }
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
  const auto & params = info_.hardware_parameters;
  // Pre-load every IMPEDANCE drive with its hold target and gains through
  // registers BEFORE enabling it. The stream frame (FRAME_IMP_CONTROL) that
  // normally carries them only starts with begin(), after the LAST enable;
  // until it arrives an enabled drive runs its impedance loop on whatever
  // target and gains it holds -- after a power cycle its raw zero and the
  // flashed gains -- so the robot twitches toward the raw zero for the
  // 10-30 ms the twelve enables take (seen 2026-10-01 as the "jump" at
  // activation). Each drive is written and read back; a mismatch or a
  // silent drive is logged, never fatal: the stream still delivers the
  // right values a few ms later, exactly as before this change. Runs in
  // dry_run too (a disabled drive produces no torque whatever its target),
  // which makes dry_run the safe way to check that the registers take the
  // write. URDF <hardware> param preload_targets (default 1; 0 = off).
  const bool preload =
    params.count("preload_targets") ? std::stoi(params.at("preload_targets")) != 0 : true;
  if (preload) {
    std::size_t i = 0;
    for (auto & md80 : md80_info_) {
      if (md80.control_mode == mab::Md80Mode_E::IMPEDANCE) {
        auto candle = find_candle_by_motor_can_id(md80.can_id);
        // Same raw-frame shift as write(): command is activation-relative.
        const float target = static_cast<float>(md80.command.position + initial_positions_[i]);
        const float kp = md80.ddq_pid.kp;
        const float kd = md80.ddq_pid.kd;
        float target_rb = NAN, kp_rb = NAN, kd_rb = NAN;
        const bool written = candle->writeMd80Register(
          md80.can_id, mab::Md80Reg_E::targetPosition, target, mab::Md80Reg_E::motorImpPidKp, kp,
          mab::Md80Reg_E::motorImpPidKd, kd);
        const bool read_back =
          written && candle->readMd80Register(
                       md80.can_id, mab::Md80Reg_E::targetPosition, target_rb,
                       mab::Md80Reg_E::motorImpPidKp, kp_rb, mab::Md80Reg_E::motorImpPidKd, kd_rb);
        const bool match = read_back && std::fabs(target_rb - target) < 1e-3f &&
                           std::fabs(kp_rb - kp) < 1e-3f && std::fabs(kd_rb - kd) < 1e-4f;
        if (match) {
          RCLCPP_INFO_STREAM(
            rclcpp::get_logger(get_name()),
            "preloaded can_id " << md80.can_id << " before enable: target " << target
                                << " rad (raw), kp " << kp << ", kd " << kd);
        } else {
          RCLCPP_WARN_STREAM(
            rclcpp::get_logger(get_name()),
            "preload before enable NOT confirmed for can_id "
              << md80.can_id << ": write " << (written ? "ok" : "FAILED") << ", read-back "
              << (read_back ? "ok" : "FAILED") << ", drive reports target " << target_rb
              << " kp " << kp_rb << " kd " << kd_rb << " (wanted " << target << " / " << kp
              << " / " << kd << ") -- the drive may twitch toward its own target until "
              << "the stream starts");
        }
      }
      ++i;
    }
  }

  if (dry_run_) {
    RCLCPP_WARN(
      rclcpp::get_logger(get_name()),
      "DRY RUN: drives left DISABLED -- no torque will reach the motors; "
      "encoder states still stream");
  } else {
    // Optional pause between enables (URDF <hardware> param
    // enable_stagger_ms, default 0 = back to back). Measured 2026-09-26: the
    // drives' CAN watchdog (100 ms, mdtool setup info) starts counting at
    // enable, and the update loop that feeds it only begins after the LAST
    // enable -- with 150 ms here every drive but the last tripped its
    // watchdog and sat enabled-but-idle: positions streamed, zero torque,
    // no error flag. Any stagger adds to the first drive's wait, so leave
    // this at 0 unless the drives' watchdog is lengthened to match.
    const int stagger_ms =
      params.count("enable_stagger_ms") ? std::stoi(params.at("enable_stagger_ms")) : 0;
    for (auto & md80 : md80_info_) {
      auto candle = find_candle_by_motor_can_id(md80.can_id);
      candle->controlMd80Enable(md80.can_id, true);
      if (stagger_ms > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(stagger_ms));
      }
    }
  }

  for (auto & candle : candle_instances) {
    candle->begin();
  }
  stall_cycles_ = 0;
  last_raw_.clear();
  update_loop_running_ = true;
}

void MD80HardwareInterface::disable_motors()
{
  update_loop_running_ = false;
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
