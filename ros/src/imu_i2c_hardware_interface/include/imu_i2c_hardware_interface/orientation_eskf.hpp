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

#ifndef IMU_I2C_HARDWARE_INTERFACE__ORIENTATION_ESKF_HPP_
#define IMU_I2C_HARDWARE_INTERFACE__ORIENTATION_ESKF_HPP_

#include <Eigen/Dense>

// Error-state Kalman filter fusing gyro + accel + magnetometer into an
// orientation quaternion (world <- IMU, world = ENU with y = magnetic
// north, z = up; yaw is therefore relative to MAGNETIC north -- no
// declination correction, which is irrelevant for a robot without global
// navigation).
//
// Design (agreed 2026-07-23, backlog pkt 17):
//  * nominal state: quaternion q + gyro bias b; error state
//    [delta_theta (3), delta_b (3)], P is 6x6.  Conventions follow Sola,
//    "Quaternion kinematics for the error-state Kalman filter" (local /
//    body-frame angle error).
//  * gyro drives the prediction only (its noise enters Q); the bias
//    estimate keeps pure-gyro stretches slow-drifting.
//  * accel corrects roll/pitch via the gravity direction; its R inflates
//    as |a| deviates from g (robot accelerating), with a hard gate.
//  * mag corrects YAW ONLY: the innovation is the horizontal-plane angle
//    of the (hard/soft-iron calibrated) field, so magnetic disturbance
//    from the drives cannot leak into roll/pitch, which is what
//    wojtek_policy actually consumes.  R adapts with the deviation of
//    |B| from the calibrated field norm and a Mahalanobis gate drops
//    grossly inconsistent samples entirely -- during motor transients the
//    filter then just "rides the gyro".
//
// Pure math, no ROS dependencies -- unit-tested in
// test/test_orientation_eskf.cpp.
class OrientationEskf
{
public:
  struct Params
  {
    // Per-sample white-noise stddev of the gyro [rad/s] and random-walk
    // stddev of its bias [rad/s per sqrt(s)].
    double gyro_noise = 2e-3;
    double gyro_bias_walk = 5e-4;
    // Stddev of the normalized gravity-direction measurement [rad].
    double accel_noise = 5e-2;
    // Relative |a|-vs-g deviation that doubles the accel R; 3x this hard
    // gates the sample.
    double accel_norm_tol = 0.1;
    // Stddev of the mag yaw measurement [rad].
    double mag_noise = 5e-2;
    // Relative |B|-vs-norm deviation that doubles the mag R; 3x this hard
    // gates the sample.
    double mag_norm_tol = 0.1;
    // Mahalanobis gate (squared, 1 dof) on the mag yaw innovation.
    double mag_mahalanobis_gate = 9.0;
    // Reserved: full 3D field-vector update instead of yaw-only.  Off
    // until the ellipsoid calibration has proven itself on the robot.
    bool mag_full_3d = false;

    // Ellipsoid calibration: m = soft_iron * (raw - hard_iron)  [uT].
    Eigen::Vector3d hard_iron = Eigen::Vector3d::Zero();
    Eigen::Matrix3d soft_iron = Eigen::Matrix3d::Identity();
    // Expected |B| after calibration [uT]; <= 0 means "learn from the
    // first accepted sample" so an uncalibrated setup still works.
    double field_norm = 0.0;
  };

  explicit OrientationEskf(const Params & params) : params_(params) { reset(); }

  void reset();

  bool initialized() const { return initialized_; }

  // Feed one IMU sample set.  accel/mag flags say which parts are fresh
  // (the LSM6 and LIS3 run at different ODRs); gyro is assumed fresh
  // whenever this is called with dt > 0.  Units: rad/s, m/s^2, uT (raw,
  // uncalibrated mag).
  void step(
    const Eigen::Vector3d & gyro, double dt, bool have_accel, const Eigen::Vector3d & accel,
    bool have_mag, const Eigen::Vector3d & mag_raw);

  // Orientation world <- IMU. Identity until initialized().
  Eigen::Quaterniond quat() const { return q_; }
  Eigen::Vector3d gyro_bias() const { return bias_; }

private:
  bool try_initialize(
    bool have_accel, const Eigen::Vector3d & accel, bool have_mag,
    const Eigen::Vector3d & mag_calibrated);
  void predict(const Eigen::Vector3d & gyro, double dt);
  void update_accel(const Eigen::Vector3d & accel);
  void update_mag_yaw(const Eigen::Vector3d & mag_calibrated);
  // Applies the 6x1 error-state correction and resets it into the
  // nominal state.
  void inject(const Eigen::Matrix<double, 6, 1> & dx);

  Params params_;
  bool initialized_ = false;
  Eigen::Quaterniond q_ = Eigen::Quaterniond::Identity();
  Eigen::Vector3d bias_ = Eigen::Vector3d::Zero();
  Eigen::Matrix<double, 6, 6> P_ = Eigen::Matrix<double, 6, 6>::Zero();
  double field_norm_ = 0.0;
};

#endif  // IMU_I2C_HARDWARE_INTERFACE__ORIENTATION_ESKF_HPP_
