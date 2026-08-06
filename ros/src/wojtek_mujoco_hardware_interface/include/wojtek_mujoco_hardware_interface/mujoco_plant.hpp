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

#ifndef WOJTEK_MUJOCO_HARDWARE_INTERFACE__MUJOCO_PLANT_HPP_
#define WOJTEK_MUJOCO_HARDWARE_INTERFACE__MUJOCO_PLANT_HPP_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "mujoco/mujoco.h"

namespace wojtek_mujoco_hardware_interface
{

/// Servo contract for one actuator, straight from the policy's pd block.
struct ServoSettings
{
  double kp;
  double kd;
  double max_torque;
};

/// One IMU reading, already expressed the way the physical sensor would
/// report it (see MujocoPlant::imu()).
struct ImuSample
{
  /// x, y, z, w -- the sensor frame's orientation in world.
  std::array<double, 4> orientation;
  std::array<double, 3> angular_velocity;
  std::array<double, 3> linear_acceleration;
  /// Microtesla, the unit the real driver's state interface carries.
  std::array<double, 3> magnetometer;
};

/// The simulated plant: a MuJoCo model stepped in lockstep with the
/// controller loop, presenting the interface the MD80s present.
///
/// Two things here are not "just simulation":
///
///  * States are reported relative to the pose the drives were in when the
///    controllers activated, and commands are interpreted in that same
///    frame. That is what the MD80 driver does, and it is why real_io_node's
///    zeroing and its boot-relative offset are exercised here instead of
///    being bypassed.
///  * Physics advances by the controller period with the remainder carried
///    over, not by whole steps per cycle. The model runs at 250 Hz (dt=4 ms)
///    under a 400 Hz loop, so rounding to whole steps would alternate 1 and
///    0 steps per cycle and inject a jitter the real robot does not have.
class MujocoPlant
{
public:
  MujocoPlant() = default;
  ~MujocoPlant();
  MujocoPlant(const MujocoPlant &) = delete;
  MujocoPlant & operator=(const MujocoPlant &) = delete;

  /// Load `scene_xml`, rewriting the robot file's meshdir to `mesh_dir`
  /// (the MJX XMLs are shipped away from the meshes they reference).
  /// Throws std::runtime_error with MuJoCo's message on failure.
  void load(const std::string & scene_xml, const std::string & mesh_dir);

  /// Actuator order as the model defines it -- the order every policy,
  /// joint-target message and pose in this stack uses.
  const std::vector<std::string> & actuatorNames() const {return actuator_names_;}

  /// Write the policy's servo contract into the actuators, so the simulated
  /// plant is the one the policy trained against rather than whatever the
  /// XML happens to carry.
  void applyServoSettings(const std::string & actuator, const ServoSettings & pd);

  /// The knee angle against the folding stop, taken from the model's own
  /// ctrlrange lower bound instead of a duplicated constant.
  double foldedKneeRad() const;

  /// "home" (the standing keyframe) or "folded" (lying flat, knees on the
  /// stop -- the real robot's boot pose). Folded is reached by settling from
  /// home under gravity so the four-bar closure stays consistent.
  void reset(const std::string & pose);

  /// Take the current joint positions as the zero the states are reported
  /// against (what powering the drives up in this pose would give).
  void latchActivationPose();

  /// Advance physics by `seconds`, carrying the sub-step remainder.
  /// Returns the number of steps actually taken.
  int advance(double seconds);

  double timestep() const {return model_->opt.timestep;}
  double simTime() const {return data_->time;}

  /// Joint position/velocity/effort in MuJoCo convention, relative to the
  /// latched activation pose (position only -- rates have no offset).
  double jointPosition(int actuator) const;
  double jointVelocity(int actuator) const;
  double jointEffort(int actuator) const;

  /// Command in MuJoCo convention, relative to the activation pose.
  void setCommand(int actuator, double q_relative);
  /// Bench mode: physics runs, no actuator force is applied.
  void setDryRun(bool dry_run) {dry_run_ = dry_run;}

  /// IMU as the physical sensor would report it: `mount_quat` is the sensor
  /// frame's orientation in base_link (the real unit is bolted on rotated),
  /// so the readings come back in the sensor frame and policy_node's
  /// imu_mount_rpy undoes exactly that rotation.
  ImuSample imu(const std::array<double, 4> & mount_quat_wxyz,
                const std::array<double, 3> & earth_field_ut) const;

  /// Ground truth the real robot has no way to know: base pose (position +
  /// quaternion wxyz) and spatial velocity.
  std::array<double, 7> basePose() const;
  std::array<double, 6> baseVelocity() const;
  /// Full generalized position, for anything that mirrors this state
  /// (the camera renderer does).
  std::vector<double> qpos() const;

private:
  int actuatorId(const std::string & name) const;

  mjModel * model_ = nullptr;
  mjData * data_ = nullptr;
  std::vector<std::string> actuator_names_;
  /// Joint qpos/dof addresses per actuator, resolved once.
  std::vector<int> qpos_adr_;
  std::vector<int> dof_adr_;
  std::vector<double> activation_qpos_;
  std::vector<int> sensor_adr_;  // orientation, gyro, accelerometer
  double leftover_ = 0.0;
  bool dry_run_ = false;
};

}  // namespace wojtek_mujoco_hardware_interface

#endif  // WOJTEK_MUJOCO_HARDWARE_INTERFACE__MUJOCO_PLANT_HPP_
