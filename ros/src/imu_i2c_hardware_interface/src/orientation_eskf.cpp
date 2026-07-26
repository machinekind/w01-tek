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

#include "imu_i2c_hardware_interface/orientation_eskf.hpp"

#include <cmath>

namespace
{
constexpr double kGravity = 9.80665;
// Below this |a| or |B| the direction is numerically meaningless.
constexpr double kMinVectorNorm = 1e-6;

Eigen::Matrix3d skew(const Eigen::Vector3d & v)
{
  Eigen::Matrix3d s;
  s << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
  return s;
}

// Quaternion for a small rotation vector (exact axis-angle).
Eigen::Quaterniond quat_exp(const Eigen::Vector3d & theta)
{
  const double angle = theta.norm();
  if (angle < 1e-12) {
    return Eigen::Quaterniond(1.0, 0.5 * theta.x(), 0.5 * theta.y(), 0.5 * theta.z()).normalized();
  }
  const Eigen::Vector3d axis = theta / angle;
  return Eigen::Quaterniond(Eigen::AngleAxisd(angle, axis));
}
}  // namespace

void OrientationEskf::reset()
{
  initialized_ = false;
  q_ = Eigen::Quaterniond::Identity();
  bias_ = Eigen::Vector3d::Zero();
  P_.setZero();
  field_norm_ = params_.field_norm;
}

void OrientationEskf::step(
  const Eigen::Vector3d & gyro, double dt, bool have_accel, const Eigen::Vector3d & accel,
  bool have_mag, const Eigen::Vector3d & mag_raw)
{
  const Eigen::Vector3d mag = params_.soft_iron * (mag_raw - params_.hard_iron);

  if (!initialized_) {
    try_initialize(have_accel, accel, have_mag, mag);
    return;
  }
  if (dt <= 0.0) {
    return;
  }

  predict(gyro, dt);
  if (have_accel) {
    update_accel(accel);
  }
  if (have_mag) {
    update_mag_yaw(mag);
  }
}

bool OrientationEskf::try_initialize(
  bool have_accel, const Eigen::Vector3d & accel, bool have_mag,
  const Eigen::Vector3d & mag_calibrated)
{
  // Roll/pitch from gravity, yaw from the horizontal field component --
  // both measurements are needed once, together.
  if (!have_accel || !have_mag) {
    return false;
  }
  if (accel.norm() < kMinVectorNorm || mag_calibrated.norm() < kMinVectorNorm) {
    return false;
  }

  const Eigen::Vector3d up_b = accel.normalized();  // specific force ~ +g up
  Eigen::Vector3d north_b = mag_calibrated - mag_calibrated.dot(up_b) * up_b;
  if (north_b.norm() < kMinVectorNorm) {
    return false;  // field parallel to gravity, no heading information
  }
  north_b.normalize();
  const Eigen::Vector3d east_b = north_b.cross(up_b);

  // Rows of R (body -> world) are the world axes expressed in the body
  // frame: ENU = [east; north; up].
  Eigen::Matrix3d rot;
  rot.row(0) = east_b;
  rot.row(1) = north_b;
  rot.row(2) = up_b;
  q_ = Eigen::Quaterniond(rot).normalized();

  bias_.setZero();
  P_.setZero();
  // Generous initial uncertainty: the init sample is a single, unfiltered
  // measurement and the bias is entirely unknown.
  const double angle_var = std::pow(10.0 * M_PI / 180.0, 2);
  const double bias_var = std::pow(0.05, 2);
  P_.topLeftCorner<3, 3>() = angle_var * Eigen::Matrix3d::Identity();
  P_.bottomRightCorner<3, 3>() = bias_var * Eigen::Matrix3d::Identity();

  if (field_norm_ <= 0.0) {
    field_norm_ = mag_calibrated.norm();
  }
  initialized_ = true;
  return true;
}

void OrientationEskf::predict(const Eigen::Vector3d & gyro, double dt)
{
  const Eigen::Vector3d omega = gyro - bias_;
  q_ = (q_ * quat_exp(omega * dt)).normalized();

  // Error-state transition (Sola eq. 269): delta_theta rotates back by the
  // integrated rate and picks up the bias error; the bias is a random walk.
  Eigen::Matrix<double, 6, 6> f = Eigen::Matrix<double, 6, 6>::Identity();
  f.topLeftCorner<3, 3>() = Eigen::Matrix3d::Identity() - skew(omega * dt);
  f.topRightCorner<3, 3>() = -dt * Eigen::Matrix3d::Identity();

  Eigen::Matrix<double, 6, 6> qn = Eigen::Matrix<double, 6, 6>::Zero();
  qn.topLeftCorner<3, 3>() =
    std::pow(params_.gyro_noise * dt, 2) * Eigen::Matrix3d::Identity();
  qn.bottomRightCorner<3, 3>() =
    std::pow(params_.gyro_bias_walk, 2) * dt * Eigen::Matrix3d::Identity();

  P_ = f * P_ * f.transpose() + qn;
}

void OrientationEskf::update_accel(const Eigen::Vector3d & accel)
{
  const double norm = accel.norm();
  if (norm < kMinVectorNorm) {
    return;
  }
  // Trust the gravity direction less the more the robot accelerates; a
  // grossly non-gravity |a| is dropped outright.
  const double rel_dev = std::abs(norm - kGravity) / kGravity;
  if (rel_dev > 3.0 * params_.accel_norm_tol) {
    return;
  }
  const double r_scale = 1.0 + std::pow(rel_dev / params_.accel_norm_tol, 2);

  // Measurement: gravity direction in the body frame.  With the local
  // error convention, z = g_b_hat + skew(g_b_hat) * delta_theta + noise.
  const Eigen::Vector3d z = accel / norm;
  const Eigen::Vector3d g_pred = q_.conjugate() * Eigen::Vector3d::UnitZ();
  const Eigen::Vector3d innovation = z - g_pred;

  Eigen::Matrix<double, 3, 6> h = Eigen::Matrix<double, 3, 6>::Zero();
  h.topLeftCorner<3, 3>() = skew(g_pred);

  const Eigen::Matrix3d r =
    std::pow(params_.accel_noise, 2) * r_scale * Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d s = h * P_ * h.transpose() + r;
  const Eigen::Matrix<double, 6, 3> k = P_ * h.transpose() * s.inverse();

  inject(k * innovation);
  P_ = (Eigen::Matrix<double, 6, 6>::Identity() - k * h) * P_;
}

void OrientationEskf::update_mag_yaw(const Eigen::Vector3d & mag_calibrated)
{
  const double norm = mag_calibrated.norm();
  if (norm < kMinVectorNorm || field_norm_ <= 0.0) {
    return;
  }
  // |B| far from the calibrated norm means a nearby disturbance (drive
  // currents, ferrous object): inflate R, and past the hard gate skip the
  // sample -- the filter keeps riding the gyro.
  const double rel_dev = std::abs(norm - field_norm_) / field_norm_;
  if (rel_dev > 3.0 * params_.mag_norm_tol) {
    return;
  }
  const double r_scale = 1.0 + std::pow(rel_dev / params_.mag_norm_tol, 2);

  // Yaw-only innovation: the calibrated field rotated into the estimated
  // world frame should have its horizontal component along +y (magnetic
  // north).  Its angle east of north, psi = atan2(m_w.x, m_w.y), relates
  // to the error state as psi ~ phi_z = e_z^T * R * delta_theta
  // (world-frame error phi = R * delta_theta), deliberately ignoring the
  // dip-angle cross-coupling into roll/pitch -- that is exactly the leak
  // this projection avoids.
  const Eigen::Vector3d m_w = q_ * mag_calibrated;
  const double horizontal = std::hypot(m_w.x(), m_w.y());
  if (horizontal < 0.1 * field_norm_) {
    return;  // field (near) vertical in the estimate: no usable heading
  }
  const double psi = std::atan2(m_w.x(), m_w.y());

  Eigen::Matrix<double, 1, 6> h = Eigen::Matrix<double, 1, 6>::Zero();
  h.leftCols<3>() = Eigen::Vector3d::UnitZ().transpose() * q_.toRotationMatrix();

  const double r = std::pow(params_.mag_noise, 2) * r_scale;
  const double s = (h * P_ * h.transpose())(0, 0) + r;
  const double innovation = psi;
  if (innovation * innovation / s > params_.mag_mahalanobis_gate) {
    return;
  }
  const Eigen::Matrix<double, 6, 1> k = P_ * h.transpose() / s;

  inject(k * innovation);
  P_ = (Eigen::Matrix<double, 6, 6>::Identity() - k * h) * P_;
}

void OrientationEskf::inject(const Eigen::Matrix<double, 6, 1> & dx)
{
  q_ = (q_ * quat_exp(dx.head<3>())).normalized();
  bias_ += dx.tail<3>();
}
