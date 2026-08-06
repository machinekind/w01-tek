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

#ifndef WOJTEK_MUJOCO_HARDWARE_INTERFACE__JOINT_MAP_HPP_
#define WOJTEK_MUJOCO_HARDWARE_INTERFACE__JOINT_MAP_HPP_

#include <string>
#include <unordered_map>

namespace wojtek_mujoco_hardware_interface
{

/// URDF <-> MuJoCo angle conversion, read from wojtek_policy's joint_map.yaml.
///
/// The same file wojtek_policy.joint_map reads in Python, on purpose: the
/// signs and offsets were fitted against the model (tools/fit_joint_map.py)
/// and a second copy of them in C++ would be a second thing to get wrong.
///
///     q_urdf = sign * q_mjc + offset
class JointMap
{
public:
  /// Throws std::runtime_error if the file is missing or malformed.
  explicit JointMap(const std::string & yaml_path);

  /// Throws std::out_of_range for a joint the map does not know.
  double sign(const std::string & joint) const;
  double offset(const std::string & joint) const;

  double toUrdf(const std::string & joint, double q_mjc) const;
  double toMjc(const std::string & joint, double q_urdf) const;
  /// Velocities and torques carry the sign but not the offset.
  double rateToUrdf(const std::string & joint, double dq_mjc) const;

private:
  std::unordered_map<std::string, double> sign_;
  std::unordered_map<std::string, double> offset_;
};

}  // namespace wojtek_mujoco_hardware_interface

#endif  // WOJTEK_MUJOCO_HARDWARE_INTERFACE__JOINT_MAP_HPP_
