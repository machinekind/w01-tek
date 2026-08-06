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

#include <gtest/gtest.h>

#include <string>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "wojtek_mujoco_hardware_interface/joint_map.hpp"

using wojtek_mujoco_hardware_interface::JointMap;

namespace
{
std::string mapPath()
{
  return ament_index_cpp::get_package_share_directory("wojtek_policy") +
         "/config/joint_map.yaml";
}
}  // namespace

TEST(JointMap, reads_the_stacks_own_map)
{
  const JointMap map(mapPath());
  // Spot-check against the file: a knee is mirrored with a large offset.
  EXPECT_DOUBLE_EQ(map.sign("rear_left_third_joint"), 1.0);
  EXPECT_NEAR(map.offset("rear_left_third_joint"), -1.1169874336, 1e-9);
  EXPECT_DOUBLE_EQ(map.sign("rear_right_third_joint"), -1.0);
}

TEST(JointMap, round_trips_positions)
{
  const JointMap map(mapPath());
  for (const std::string joint : {
      "rear_left_first_joint", "rear_right_second_joint", "front_left_third_joint"})
  {
    for (const double q : {-1.5, -0.3, 0.0, 0.42, 3.1}) {
      EXPECT_NEAR(map.toMjc(joint, map.toUrdf(joint, q)), q, 1e-12) << joint;
    }
  }
}

TEST(JointMap, rates_carry_the_sign_but_not_the_offset)
{
  const JointMap map(mapPath());
  // A mirrored joint: the offset must not leak into a velocity.
  const std::string mirrored = "rear_right_third_joint";
  ASSERT_DOUBLE_EQ(map.sign(mirrored), -1.0);
  ASSERT_NE(map.offset(mirrored), 0.0);
  EXPECT_DOUBLE_EQ(map.rateToUrdf(mirrored, 2.0), -2.0);
  EXPECT_DOUBLE_EQ(map.rateToUrdf(mirrored, 0.0), 0.0);
}

TEST(JointMap, position_differences_are_offset_free)
{
  // Why read() can convert boot-relative positions with the sign alone: in a
  // difference of two absolute angles the offset cancels.
  const JointMap map(mapPath());
  const std::string joint = "front_left_third_joint";
  const double a = 0.5, b = 1.25;
  const double urdf_difference = map.toUrdf(joint, a) - map.toUrdf(joint, b);
  EXPECT_NEAR(urdf_difference, map.sign(joint) * (a - b), 1e-12);
}

TEST(JointMap, unknown_joints_and_files_are_loud)
{
  const JointMap map(mapPath());
  EXPECT_THROW(map.sign("no_such_joint"), std::out_of_range);
  EXPECT_THROW(JointMap("/nonexistent/joint_map.yaml"), std::runtime_error);
}
