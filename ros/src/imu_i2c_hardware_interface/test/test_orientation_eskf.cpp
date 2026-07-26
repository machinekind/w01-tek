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

#include <Eigen/Dense>
#include <cmath>

#include "imu_i2c_hardware_interface/orientation_eskf.hpp"

namespace
{
constexpr double kGravity = 9.80665;
constexpr double kDt = 1.0 / 104.0;  // LSM6 ODR

// Warsaw-ish field: ~50 uT total, ~67 deg dip (down = -z in ENU).
// World frame is ENU with y = magnetic north.
const Eigen::Vector3d kFieldWorld(0.0, 50.0 * std::cos(67.0 * M_PI / 180.0),
                                  -50.0 * std::sin(67.0 * M_PI / 180.0));
const Eigen::Vector3d kGravityWorld(0.0, 0.0, kGravity);

// Ideal body-frame measurements for a given true orientation (world <- body).
Eigen::Vector3d accel_for(const Eigen::Quaterniond & q_true)
{
  return q_true.conjugate() * kGravityWorld;  // specific force ~ +g up
}
Eigen::Vector3d mag_for(const Eigen::Quaterniond & q_true)
{
  return q_true.conjugate() * kFieldWorld;
}

double angle_between(const Eigen::Quaterniond & a, const Eigen::Quaterniond & b)
{
  return a.angularDistance(b);
}
}  // namespace

TEST(OrientationEskf, InitializesFromAccelAndMag)
{
  OrientationEskf eskf{OrientationEskf::Params{}};
  EXPECT_FALSE(eskf.initialized());

  const Eigen::Quaterniond q_true =
    Eigen::Quaterniond(Eigen::AngleAxisd(0.3, Eigen::Vector3d(1, 2, 3).normalized()));
  eskf.step(Eigen::Vector3d::Zero(), kDt, true, accel_for(q_true), true, mag_for(q_true));

  ASSERT_TRUE(eskf.initialized());
  EXPECT_LT(angle_between(eskf.quat(), q_true), 1e-6);
}

TEST(OrientationEskf, StaticPoseStaysPut)
{
  OrientationEskf eskf{OrientationEskf::Params{}};
  const Eigen::Quaterniond q_true =
    Eigen::Quaterniond(Eigen::AngleAxisd(0.5, Eigen::Vector3d(0, 1, 1).normalized()));
  const Eigen::Vector3d accel = accel_for(q_true);
  const Eigen::Vector3d mag = mag_for(q_true);

  for (int i = 0; i < 1000; ++i) {
    eskf.step(Eigen::Vector3d::Zero(), kDt, true, accel, i % 2 == 0, mag);
  }
  EXPECT_LT(angle_between(eskf.quat(), q_true), 1e-3);
  EXPECT_LT(eskf.gyro_bias().norm(), 1e-3);
}

TEST(OrientationEskf, EstimatesGyroBias)
{
  OrientationEskf eskf{OrientationEskf::Params{}};
  const Eigen::Quaterniond q_true = Eigen::Quaterniond::Identity();
  const Eigen::Vector3d bias(0.01, -0.02, 0.015);  // rad/s, well above noise
  const Eigen::Vector3d accel = accel_for(q_true);
  const Eigen::Vector3d mag = mag_for(q_true);

  // Static robot, biased gyro: the filter must pin the pose to accel+mag
  // and push the discrepancy into the bias states.
  for (int i = 0; i < 30 * 104; ++i) {  // 30 s
    eskf.step(bias, kDt, true, accel, true, mag);
  }
  EXPECT_LT((eskf.gyro_bias() - bias).norm(), 2e-3);
  EXPECT_LT(angle_between(eskf.quat(), q_true), 0.02);
}

TEST(OrientationEskf, TracksYawRotation)
{
  OrientationEskf eskf{OrientationEskf::Params{}};
  Eigen::Quaterniond q_true = Eigen::Quaterniond::Identity();
  // Initialize static first.
  for (int i = 0; i < 200; ++i) {
    eskf.step(Eigen::Vector3d::Zero(), kDt, true, accel_for(q_true), true, mag_for(q_true));
  }

  // Rotate 90 deg about world z at 0.5 rad/s (body z == world z here).
  const double rate = 0.5;
  const int steps = static_cast<int>((M_PI / 2) / rate / kDt);
  for (int i = 0; i < steps; ++i) {
    q_true = q_true * Eigen::Quaterniond(Eigen::AngleAxisd(rate * kDt, Eigen::Vector3d::UnitZ()));
    eskf.step(
      Eigen::Vector3d(0, 0, rate), kDt, true, accel_for(q_true), true, mag_for(q_true));
  }
  EXPECT_LT(angle_between(eskf.quat(), q_true), 0.02);
}

TEST(OrientationEskf, GatesDisturbedMagSamples)
{
  OrientationEskf eskf{OrientationEskf::Params{}};
  const Eigen::Quaterniond q_true = Eigen::Quaterniond::Identity();
  const Eigen::Vector3d accel = accel_for(q_true);
  const Eigen::Vector3d mag = mag_for(q_true);

  for (int i = 0; i < 500; ++i) {
    eskf.step(Eigen::Vector3d::Zero(), kDt, true, accel, true, mag);
  }
  const Eigen::Quaterniond before = eskf.quat();

  // Motor-current style disturbance: field magnitude far off the norm.
  // The |B| hard gate must drop every sample; the estimate rides the gyro
  // and stays where it was.
  const Eigen::Vector3d disturbed = 3.0 * mag + Eigen::Vector3d(40.0, -25.0, 10.0);
  for (int i = 0; i < 500; ++i) {
    eskf.step(Eigen::Vector3d::Zero(), kDt, true, accel, true, disturbed);
  }
  EXPECT_LT(angle_between(eskf.quat(), before), 1e-3);
}

TEST(OrientationEskf, YawRecoversAfterMagOutage)
{
  OrientationEskf eskf{OrientationEskf::Params{}};
  Eigen::Quaterniond q_true = Eigen::Quaterniond::Identity();
  for (int i = 0; i < 500; ++i) {
    eskf.step(Eigen::Vector3d::Zero(), kDt, true, accel_for(q_true), true, mag_for(q_true));
  }

  // Yaw the robot 45 deg while the mag is gated (simulated by not feeding
  // it): gyro integration carries the estimate.
  const double rate = 0.5;
  const int steps = static_cast<int>((M_PI / 4) / rate / kDt);
  for (int i = 0; i < steps; ++i) {
    q_true = q_true * Eigen::Quaterniond(Eigen::AngleAxisd(rate * kDt, Eigen::Vector3d::UnitZ()));
    eskf.step(Eigen::Vector3d(0, 0, rate), kDt, true, accel_for(q_true), false, {});
  }
  EXPECT_LT(angle_between(eskf.quat(), q_true), 0.05);

  // Mag comes back: any residual yaw error is pulled out again.
  for (int i = 0; i < 2000; ++i) {
    eskf.step(Eigen::Vector3d::Zero(), kDt, true, accel_for(q_true), true, mag_for(q_true));
  }
  EXPECT_LT(angle_between(eskf.quat(), q_true), 0.01);
}

TEST(OrientationEskf, AppliesHardIronCalibration)
{
  OrientationEskf::Params params;
  params.hard_iron = Eigen::Vector3d(12.0, -7.5, 3.0);
  OrientationEskf eskf{params};

  const Eigen::Quaterniond q_true =
    Eigen::Quaterniond(Eigen::AngleAxisd(1.0, Eigen::Vector3d::UnitZ()));
  const Eigen::Vector3d mag_raw = mag_for(q_true) + params.hard_iron;

  for (int i = 0; i < 1000; ++i) {
    eskf.step(Eigen::Vector3d::Zero(), kDt, true, accel_for(q_true), true, mag_raw);
  }
  EXPECT_LT(angle_between(eskf.quat(), q_true), 1e-3);
}

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
