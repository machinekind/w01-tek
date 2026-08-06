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

#include "wojtek_mujoco_hardware_interface/joint_map.hpp"

#include <stdexcept>
#include <string>

#include "yaml-cpp/yaml.h"

namespace wojtek_mujoco_hardware_interface
{

JointMap::JointMap(const std::string & yaml_path)
{
  YAML::Node root;
  try {
    root = YAML::LoadFile(yaml_path);
  } catch (const YAML::Exception & e) {
    throw std::runtime_error("joint_map: cannot read " + yaml_path + ": " + e.what());
  }
  const auto map = root["joint_map"];
  if (!map || !map.IsMap()) {
    throw std::runtime_error("joint_map: no 'joint_map' mapping in " + yaml_path);
  }
  for (const auto & entry : map) {
    const auto name = entry.first.as<std::string>();
    sign_[name] = entry.second["sign"].as<double>();
    offset_[name] = entry.second["offset"].as<double>();
  }
  if (sign_.empty()) {
    throw std::runtime_error("joint_map: empty mapping in " + yaml_path);
  }
}

double JointMap::sign(const std::string & joint) const
{
  return sign_.at(joint);
}

double JointMap::offset(const std::string & joint) const
{
  return offset_.at(joint);
}

double JointMap::toUrdf(const std::string & joint, double q_mjc) const
{
  return sign_.at(joint) * q_mjc + offset_.at(joint);
}

double JointMap::toMjc(const std::string & joint, double q_urdf) const
{
  return (q_urdf - offset_.at(joint)) / sign_.at(joint);
}

double JointMap::rateToUrdf(const std::string & joint, double dq_mjc) const
{
  return sign_.at(joint) * dq_mjc;
}

}  // namespace wojtek_mujoco_hardware_interface
