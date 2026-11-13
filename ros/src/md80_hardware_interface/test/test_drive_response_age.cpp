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

#include <cstdint>
#include <vector>

#include "md80_hardware_interface/drive_response_age.hpp"

using md80_hardware_interface::DriveResponseAge;

namespace
{
constexpr int64_t kMs = 1'000'000;
// Double timestamps near this epoch value have about 256 ns resolution.
constexpr int64_t kT0 = 1'700'000'000'123'456'789;

std::vector<double> sample(double offset)
{
  return {0.1 + offset, 0.0, 0.01, 0.2 + offset, 0.0, 0.02, 0.3 + offset, 0.0, 0.03};
}

void set_drive(std::vector<double> & raw, std::size_t drive, double torque)
{
  raw[3 * drive + 2] = torque;
}

class DriveResponseAgeTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    age.resize(3);
    ages.assign(3, -1.0);
  }

  void step(int64_t now_ns, const std::vector<double> & raw)
  {
    age.update(now_ns, true, raw, ages);
  }

  DriveResponseAge age;
  std::vector<double> ages;
};
}  // namespace

TEST_F(DriveResponseAgeTest, FirstTimedCycleIsBaseline)
{
  EXPECT_FALSE(age.baseline_set());
  step(kT0, sample(0.0));
  EXPECT_TRUE(age.baseline_set());
  for (double a : ages) {
    EXPECT_EQ(a, 0.0);
  }
}

TEST_F(DriveResponseAgeTest, UnchangedSamplesAccumulateAge)
{
  const auto raw = sample(0.0);
  step(kT0, raw);
  step(kT0 + 10 * kMs, raw);
  for (double a : ages) {
    EXPECT_DOUBLE_EQ(a, 0.010);
  }
  step(kT0 + 250 * kMs, raw);
  for (double a : ages) {
    EXPECT_DOUBLE_EQ(a, 0.250);
  }
}

TEST_F(DriveResponseAgeTest, DrivesChangeIndependently)
{
  auto raw = sample(0.0);
  step(kT0, raw);

  set_drive(raw, 1, 0.021);
  step(kT0 + 10 * kMs, raw);
  EXPECT_DOUBLE_EQ(ages[0], 0.010);
  EXPECT_EQ(ages[1], 0.0);
  EXPECT_DOUBLE_EQ(ages[2], 0.010);

  set_drive(raw, 0, 0.011);
  set_drive(raw, 1, 0.022);
  step(kT0 + 20 * kMs, raw);
  EXPECT_EQ(ages[0], 0.0);
  EXPECT_EQ(ages[1], 0.0);
  EXPECT_DOUBLE_EQ(ages[2], 0.020);

  raw[3 * 2 + 0] += 1e-6;
  step(kT0 + 30 * kMs, raw);
  EXPECT_DOUBLE_EQ(ages[0], 0.010);
  EXPECT_DOUBLE_EQ(ages[1], 0.010);
  EXPECT_EQ(ages[2], 0.0);
  raw[3 * 0 + 1] = 0.5;
  step(kT0 + 40 * kMs, raw);
  EXPECT_EQ(ages[0], 0.0);
  EXPECT_DOUBLE_EQ(ages[2], 0.010);
}

TEST_F(DriveResponseAgeTest, UntimedActivationReadsLeaveAgesZeroAndNoBaseline)
{
  // Activation reads carry zero timestamps.
  const auto raw = sample(0.0);
  age.update(0, false, raw, ages);
  for (double a : ages) {
    EXPECT_EQ(a, 0.0);
  }
  EXPECT_FALSE(age.baseline_set());
  const auto shifted = sample(1.0);  // The second read includes the position offsets.
  age.update(0, false, shifted, ages);
  for (double a : ages) {
    EXPECT_EQ(a, 0.0);
  }
  EXPECT_FALSE(age.baseline_set());

  step(kT0, shifted);
  EXPECT_TRUE(age.baseline_set());
  for (double a : ages) {
    EXPECT_EQ(a, 0.0);
  }
  step(kT0 + 10 * kMs, shifted);
  for (double a : ages) {
    EXPECT_DOUBLE_EQ(a, 0.010);
  }
}

TEST_F(DriveResponseAgeTest, DeactivationResetsBaseline)
{
  const auto raw = sample(0.0);
  step(kT0, raw);
  step(kT0 + 500 * kMs, raw);
  EXPECT_DOUBLE_EQ(ages[0], 0.5);

  // Reads after deactivation carry no usable time.
  age.update(kT0 + 600 * kMs, false, raw, ages);
  EXPECT_FALSE(age.baseline_set());
  EXPECT_EQ(ages[0], 0.0);

  step(kT0 + 60'000 * kMs, raw);
  EXPECT_EQ(ages[0], 0.0);
  step(kT0 + 60'010 * kMs, raw);
  EXPECT_DOUBLE_EQ(ages[0], 0.010);

  age.reset();
  EXPECT_FALSE(age.baseline_set());
  step(kT0 + 70'000 * kMs, raw);
  EXPECT_EQ(ages[0], 0.0);
}

TEST_F(DriveResponseAgeTest, LargeTimestampsKeepNanosecondPrecision)
{
  const auto raw = sample(0.0);
  step(kT0, raw);
  step(kT0 + 1000, raw);
  EXPECT_DOUBLE_EQ(ages[0], 1e-6);
  step(kT0 + 1003, raw);
  EXPECT_DOUBLE_EQ(ages[0], 1.003e-6);
  step(kT0 + 10 * kMs + 7, raw);
  EXPECT_DOUBLE_EQ(ages[0], 0.010000007);
}

TEST_F(DriveResponseAgeTest, BackwardsClockStepClampsToZero)
{
  auto raw = sample(0.0);
  step(kT0, raw);
  step(kT0 + 100 * kMs, raw);
  EXPECT_DOUBLE_EQ(ages[0], 0.1);
  step(kT0 - 5'000 * kMs, raw);
  for (double a : ages) {
    EXPECT_EQ(a, 0.0);
  }
  step(kT0 + 200 * kMs, raw);
  EXPECT_DOUBLE_EQ(ages[0], 0.2);
}
