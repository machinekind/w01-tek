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
//
// The plant's job is not "run MuJoCo" -- it is to behave like the MD80s do:
// states relative to the activation pose, commands read in that same frame,
// physics advanced by the controller period rather than by whole steps, and
// an IMU reported in the frame the physical sensor is bolted in. Those are
// what these tests pin down, on a two-joint stand-in model so nothing here
// depends on the robot's mesh set.

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>

#include "wojtek_mujoco_hardware_interface/mujoco_plant.hpp"

using wojtek_mujoco_hardware_interface::MujocoPlant;
using wojtek_mujoco_hardware_interface::ServoSettings;

namespace
{

constexpr double kTimestep = 0.004;   // the robot model's, so the 400 Hz
constexpr double kControlPeriod = 0.0025;  // loop lands between steps

const char * kRobot = R"(<mujoco model="stand_in">
  <compiler angle="radian" meshdir="../meshes"/>
  <option timestep="0.004"/>
  <worldbody>
    <body name="base" pos="0 0 0.5">
      <freejoint name="base_free"/>
      <geom type="box" size="0.1 0.1 0.05" mass="1"/>
      <site name="imu" pos="0 0 0"/>
      <body name="link" pos="0.1 0 0">
        <joint name="leg_third_joint" type="hinge" axis="0 1 0" range="0 3"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.2"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <general name="leg_third_joint" joint="leg_third_joint" ctrlrange="0.425 2.5"
             forcerange="-6 6" biastype="affine" gainprm="20" biasprm="0 -20 -1"/>
  </actuator>
  <sensor>
    <framequat objtype="site" objname="imu" name="orientation"/>
    <gyro site="imu" name="angular-velocity"/>
    <accelerometer site="imu" name="linear-acceleration"/>
  </sensor>
  <keyframe>
    <key name="home" qpos="0 0 0.5 1 0 0 0 1.0" ctrl="1.0"/>
  </keyframe>
</mujoco>
)";

const char * kScene = R"(<mujoco model="stand_in_scene">
  <include file="wojtek_mjx.xml"/>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.01"/>
  </worldbody>
</mujoco>
)";

/// Writes the stand-in model where load() expects a scene plus siblings.
std::string writeModel()
{
  const auto dir = std::filesystem::temp_directory_path() /
    ("plant_test_" + std::to_string(::getpid()));
  std::filesystem::create_directories(dir);
  std::ofstream(dir / "wojtek_mjx.xml") << kRobot;
  std::ofstream(dir / "scene_mjx.xml") << kScene;
  return (dir / "scene_mjx.xml").string();
}

class Plant : public ::testing::Test
{
protected:
  void SetUp() override
  {
    plant_.load(writeModel(), "/nonexistent/meshes");
  }
  MujocoPlant plant_;
};

}  // namespace

TEST_F(Plant, exposes_the_models_actuator_order)
{
  ASSERT_EQ(plant_.actuatorNames().size(), 1u);
  EXPECT_EQ(plant_.actuatorNames()[0], "leg_third_joint");
  EXPECT_DOUBLE_EQ(plant_.timestep(), kTimestep);
}

TEST_F(Plant, advances_by_the_control_period_without_drifting)
{
  // 400 Hz over a 250 Hz model: whole steps per cycle would alternate 1 and 0
  // and inject jitter the robot does not have, so the remainder is carried.
  plant_.latchActivationPose();
  constexpr int kCycles = 4000;  // 10 s of control at 2.5 ms
  int steps = 0;
  for (int i = 0; i < kCycles; ++i) {
    steps += plant_.advance(kControlPeriod);
  }
  const double expected = kCycles * kControlPeriod;
  EXPECT_NEAR(plant_.simTime(), expected, kTimestep);
  EXPECT_EQ(steps, static_cast<int>(expected / kTimestep));
}

TEST_F(Plant, a_stalled_caller_cannot_start_a_catch_up_spiral)
{
  plant_.latchActivationPose();
  const int steps = plant_.advance(30.0);  // a debugger breakpoint, say
  EXPECT_LE(steps, 200);
  // And the backlog is dropped rather than repaid over the next cycles.
  EXPECT_LE(plant_.advance(kControlPeriod), 1);
}

TEST_F(Plant, states_are_reported_against_the_activation_pose)
{
  // Powering the drives up in whatever pose the robot happens to be in: the
  // MD80s report zero there, and real_io_node's offset logic is written
  // against exactly that.
  plant_.latchActivationPose();
  EXPECT_NEAR(plant_.jointPosition(0), 0.0, 1e-12);

  // Move somewhere else, latch again: zero once more, from the new pose.
  plant_.setCommand(0, 0.4);
  for (int i = 0; i < 2000; ++i) {
    plant_.advance(kControlPeriod);
  }
  EXPECT_GT(std::abs(plant_.jointPosition(0)), 0.05);
  plant_.latchActivationPose();
  EXPECT_NEAR(plant_.jointPosition(0), 0.0, 1e-12);
}

TEST_F(Plant, a_zero_command_holds_the_activation_pose)
{
  plant_.latchActivationPose();
  plant_.setCommand(0, 0.0);
  for (int i = 0; i < 2000; ++i) {
    plant_.advance(kControlPeriod);
  }
  // Gravity pulls on the link; the servo has to hold it near where it was.
  EXPECT_NEAR(plant_.jointPosition(0), 0.0, 0.05);
}

TEST_F(Plant, servo_settings_from_the_policy_contract_reach_the_actuator)
{
  // A soft servo cannot hold what a stiff one holds: the gains are what the
  // policy trained against, so silently keeping the XML's would mean
  // simulating a different robot.
  const double soft = [this] {
      plant_.applyServoSettings("leg_third_joint", {2.0, 0.1, 0.5});
      plant_.reset("home");
      plant_.latchActivationPose();
      plant_.setCommand(0, 0.0);
      for (int i = 0; i < 2000; ++i) {plant_.advance(kControlPeriod);}
      return std::abs(plant_.jointPosition(0));
    }();

  MujocoPlant stiff;
  stiff.load(writeModel(), "/nonexistent/meshes");
  stiff.applyServoSettings("leg_third_joint", {200.0, 5.0, 20.0});
  stiff.latchActivationPose();
  stiff.setCommand(0, 0.0);
  for (int i = 0; i < 2000; ++i) {stiff.advance(kControlPeriod);}
  EXPECT_LT(std::abs(stiff.jointPosition(0)), soft);
}

TEST_F(Plant, dry_run_withholds_torque_but_keeps_physics_running)
{
  plant_.setDryRun(true);
  plant_.latchActivationPose();
  plant_.setCommand(0, 1.0);  // would be a big move with torque
  for (int i = 0; i < 800; ++i) {
    plant_.advance(kControlPeriod);
  }
  EXPECT_GT(plant_.simTime(), 0.0);          // time passes
  EXPECT_LT(plant_.basePose()[2], 0.5);      // and the robot drops
}

TEST_F(Plant, folded_knee_comes_from_the_models_own_ctrlrange)
{
  // Not a constant copied from wojtek_policy.poses: the folding stop is a
  // property of the model, and copies drift.
  EXPECT_DOUBLE_EQ(plant_.foldedKneeRad(), 0.425);
}

TEST_F(Plant, reset_rejects_a_pose_it_does_not_know)
{
  EXPECT_THROW(plant_.reset("crouched"), std::runtime_error);
  EXPECT_NO_THROW(plant_.reset("home"));
  EXPECT_NO_THROW(plant_.reset("folded"));
}

TEST_F(Plant, imu_is_reported_in_the_frame_the_sensor_is_mounted_in)
{
  // Let it come to rest on the floor first: in free fall the accelerometer
  // correctly reads nothing, which would say nothing about the mount.
  plant_.latchActivationPose();
  plant_.setCommand(0, 0.0);
  for (int i = 0; i < 1200; ++i) {
    plant_.advance(kControlPeriod);
  }
  ASSERT_NEAR(plant_.baseVelocity()[2], 0.0, 0.05) << "still falling";
  const std::array<double, 3> field{20.0, 0.0, -44.0};

  const auto upright = plant_.imu({1.0, 0.0, 0.0, 0.0}, field);   // no rotation
  const auto rotated = plant_.imu({0.0, 0.0, 1.0, 0.0}, field);   // pi about y

  // Standing still: specific force is gravity, magnitude unchanged by the
  // mount, but the axes the sensor reports it on are not.
  const auto magnitude = [](const std::array<double, 3> & v) {
      return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    };
  EXPECT_NEAR(magnitude(upright.linear_acceleration), 9.81, 0.2);
  EXPECT_NEAR(
    magnitude(rotated.linear_acceleration),
    magnitude(upright.linear_acceleration), 1e-9);
  EXPECT_NEAR(
    rotated.linear_acceleration[2], -upright.linear_acceleration[2], 1e-9);

  // The magnetometer is synthesized in the same frame, in microtesla.
  EXPECT_NEAR(magnitude(upright.magnetometer), magnitude(field), 1e-6);
  EXPECT_NEAR(rotated.magnetometer[2], -upright.magnetometer[2], 1e-9);

  // The reported orientation is the sensor frame's, not the base's: rotating
  // the mount composes into the quaternion. (Asserting "identity when level"
  // would be wrong -- the stand-in tips a little as it settles, and so does a
  // real robot.)
  const auto compose = [](const std::array<double, 4> & a_xyzw,
      const std::array<double, 4> & b_wxyz) {
      const double w1 = a_xyzw[3], x1 = a_xyzw[0], y1 = a_xyzw[1], z1 = a_xyzw[2];
      const double w2 = b_wxyz[0], x2 = b_wxyz[1], y2 = b_wxyz[2], z2 = b_wxyz[3];
      return std::array<double, 4>{
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
      };
    };
  const auto expected = compose(upright.orientation, {0.0, 0.0, 1.0, 0.0});
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(rotated.orientation[i], expected[i], 1e-9) << "component " << i;
  }
}
