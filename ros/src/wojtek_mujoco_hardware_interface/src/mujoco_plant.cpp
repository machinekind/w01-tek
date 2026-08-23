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

#include "wojtek_mujoco_hardware_interface/mujoco_plant.hpp"

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace wojtek_mujoco_hardware_interface
{
namespace
{

/// MuJoCo sensors the model exposes for the IMU site.
constexpr const char * kOrientationSensor = "orientation";
constexpr const char * kGyroSensor = "angular-velocity";
constexpr const char * kAccelSensor = "linear-acceleration";

std::string readFile(const std::string & path)
{
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("cannot open " + path);
  }
  std::stringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

/// The MJX scene includes the robot file, which points meshdir at a path
/// relative to itself; the XMLs ship in a package share directory away from
/// the meshes they reference. Stage the scene's whole directory with meshdir
/// rewritten, which is what the Python sim node has always done (the include
/// has to resolve, so siblings come along).
std::filesystem::path stageScene(
  const std::string & scene_xml, const std::string & mesh_dir)
{
  const std::filesystem::path scene(scene_xml);
  if (!std::filesystem::exists(scene)) {
    throw std::runtime_error("no such model: " + scene.string());
  }
  auto staged = std::filesystem::temp_directory_path() /
    ("wojtek_mjx_" + std::to_string(::getpid()));
  std::filesystem::create_directories(staged);
  for (const auto & entry : std::filesystem::directory_iterator(scene.parent_path())) {
    if (entry.path().extension() != ".xml") {
      continue;
    }
    const auto rewritten = std::regex_replace(
      readFile(entry.path().string()), std::regex("meshdir=\"[^\"]*\""),
      "meshdir=\"" + mesh_dir + "\"");
    std::ofstream(staged / entry.path().filename()) << rewritten;
  }
  return staged / scene.filename();
}

/// Rotate a world vector into the frame described by quat (wxyz).
std::array<double, 3> intoFrame(
  const std::array<double, 3> & v_world, const double * quat_wxyz)
{
  std::array<double, 4> conj{quat_wxyz[0], -quat_wxyz[1], -quat_wxyz[2], -quat_wxyz[3]};
  std::array<double, 3> out{};
  mju_rotVecQuat(out.data(), v_world.data(), conj.data());
  return out;
}

}  // namespace

MujocoPlant::~MujocoPlant()
{
  if (data_) {mj_deleteData(data_);}
  if (model_) {mj_deleteModel(model_);}
}

void MujocoPlant::load(const std::string & scene_xml, const std::string & mesh_dir)
{
  const auto staged = stageScene(scene_xml, mesh_dir);
  char error[1024] = "";
  model_ = mj_loadXML(staged.string().c_str(), nullptr, error, sizeof(error));
  if (!model_) {
    throw std::runtime_error("mj_loadXML failed: " + std::string(error));
  }
  data_ = mj_makeData(model_);

  actuator_names_.clear();
  qpos_adr_.clear();
  dof_adr_.clear();
  for (int i = 0; i < model_->nu; ++i) {
    const char * name = mj_id2name(model_, mjOBJ_ACTUATOR, i);
    if (!name) {
      throw std::runtime_error("model has an unnamed actuator");
    }
    actuator_names_.emplace_back(name);
    const int joint = model_->actuator_trnid[2 * i];
    qpos_adr_.push_back(model_->jnt_qposadr[joint]);
    dof_adr_.push_back(model_->jnt_dofadr[joint]);
  }
  activation_qpos_.assign(model_->nu, 0.0);

  for (const char * sensor : {kOrientationSensor, kGyroSensor, kAccelSensor}) {
    const int id = mj_name2id(model_, mjOBJ_SENSOR, sensor);
    if (id < 0) {
      throw std::runtime_error(
        "model has no '" + std::string(sensor) + "' sensor");
    }
    sensor_adr_.push_back(model_->sensor_adr[id]);
  }
  reset("home");
}

int MujocoPlant::actuatorId(const std::string & name) const
{
  const auto it = std::find(actuator_names_.begin(), actuator_names_.end(), name);
  if (it == actuator_names_.end()) {
    throw std::runtime_error("model has no actuator '" + name + "'");
  }
  return static_cast<int>(std::distance(actuator_names_.begin(), it));
}

void MujocoPlant::applyServoSettings(
  const std::string & actuator, const ServoSettings & pd)
{
  const int i = actuatorId(actuator);
  // Position servo: gain kp, bias (0, -kp, -kd) => tau = kp*(target-q) - kd*dq,
  // clamped to forcerange. Same actuator model the MD80s run in IMPEDANCE.
  model_->actuator_gainprm[i * mjNGAIN + 0] = pd.kp;
  model_->actuator_biasprm[i * mjNBIAS + 1] = -pd.kp;
  model_->actuator_biasprm[i * mjNBIAS + 2] = -pd.kd;
  model_->actuator_forcerange[2 * i + 0] = -pd.max_torque;
  model_->actuator_forcerange[2 * i + 1] = pd.max_torque;
}

double MujocoPlant::foldedKneeRad() const
{
  for (std::size_t i = 0; i < actuator_names_.size(); ++i) {
    if (actuator_names_[i].size() > 12 &&
      actuator_names_[i].compare(
        actuator_names_[i].size() - 12, 12, "_third_joint") == 0)
    {
      return model_->actuator_ctrlrange[2 * i + 0];
    }
  }
  throw std::runtime_error("model has no knee (third) actuator");
}

void MujocoPlant::reset(const std::string & pose)
{
  const int home = mj_name2id(model_, mjOBJ_KEY, "home");
  if (home < 0) {
    throw std::runtime_error("model has no 'home' keyframe");
  }
  mj_resetDataKeyframe(model_, data_, home);
  mj_forward(model_, data_);
  leftover_ = 0.0;
  if (pose == "home") {
    return;
  }
  if (pose != "folded") {
    throw std::runtime_error("initial_pose must be 'home' or 'folded', got " + pose);
  }
  // Settle into the boot pose instead of hand-writing a qpos: the four-bar
  // closure has to stay consistent, and gravity plus the folded targets get
  // there the same way the real robot does.
  const double knee = foldedKneeRad();
  for (int i = 0; i < model_->nu; ++i) {
    const auto & name = actuator_names_[i];
    const bool is_knee = name.size() > 12 &&
      name.compare(name.size() - 12, 12, "_third_joint") == 0;
    data_->ctrl[i] = is_knee ? knee : 0.0;
  }
  const int steps = static_cast<int>(2.0 / model_->opt.timestep);
  for (int i = 0; i < steps; ++i) {
    mj_step(model_, data_);
  }
  std::fill(data_->qvel, data_->qvel + model_->nv, 0.0);
  data_->time = 0.0;
  mj_forward(model_, data_);
}

void MujocoPlant::latchActivationPose()
{
  for (int i = 0; i < model_->nu; ++i) {
    activation_qpos_[i] = data_->qpos[qpos_adr_[i]];
    // Hold position: with no command written yet the servo would otherwise
    // drive towards whatever ctrl the reset left behind.
    data_->ctrl[i] = activation_qpos_[i];
  }
}

int MujocoPlant::advance(double seconds)
{
  leftover_ += seconds;
  int steps = 0;
  const double dt = model_->opt.timestep;
  // Bounded so a stalled caller (a debugger breakpoint, a long GC pause in
  // some other process) cannot turn into a death spiral of catch-up steps.
  constexpr int kMaxSteps = 200;
  while (leftover_ >= dt && steps < kMaxSteps) {
    mj_step(model_, data_);
    leftover_ -= dt;
    ++steps;
  }
  if (steps == kMaxSteps) {
    leftover_ = 0.0;
  }
  if (dry_run_) {
    std::fill(data_->ctrl, data_->ctrl + model_->nu, 0.0);
    std::fill(data_->qfrc_applied, data_->qfrc_applied + model_->nv, 0.0);
  }
  return steps;
}

double MujocoPlant::jointPosition(int actuator) const
{
  return data_->qpos[qpos_adr_[actuator]] - activation_qpos_[actuator];
}

double MujocoPlant::jointVelocity(int actuator) const
{
  return data_->qvel[dof_adr_[actuator]];
}

double MujocoPlant::jointEffort(int actuator) const
{
  return data_->actuator_force[actuator];
}

void MujocoPlant::setCommand(int actuator, double q_relative)
{
  data_->ctrl[actuator] = dry_run_ ? activation_qpos_[actuator]
    : activation_qpos_[actuator] + q_relative;
}

void MujocoPlant::setFeedForward(int actuator, double tau)
{
  data_->qfrc_applied[dof_adr_[actuator]] = dry_run_ ? 0.0 : tau;
}

ImuSample MujocoPlant::imu(
  const std::array<double, 4> & mount_quat_wxyz,
  const std::array<double, 3> & earth_field_ut) const
{
  ImuSample sample{};
  const double * site_quat = &data_->sensordata[sensor_adr_[0]];  // frame -> world
  const double * gyro = &data_->sensordata[sensor_adr_[1]];
  const double * accel = &data_->sensordata[sensor_adr_[2]];

  // The site is aligned with base_link, the physical sensor is not: compose
  // the mount rotation in so the reported quaternion is the sensor frame's
  // orientation in world, exactly what the real driver's filter reports.
  std::array<double, 4> sensor_quat{};
  mju_mulQuat(sensor_quat.data(), site_quat, mount_quat_wxyz.data());
  sample.orientation = {sensor_quat[1], sensor_quat[2], sensor_quat[3], sensor_quat[0]};

  // Rates and specific force arrive in the site (base) frame; the sensor
  // measures them in its own, so rotate by the mount. policy_node's
  // imu_mount_rpy is what undoes this.
  const std::array<double, 3> gyro_base{gyro[0], gyro[1], gyro[2]};
  const std::array<double, 3> accel_base{accel[0], accel[1], accel[2]};
  sample.angular_velocity = intoFrame(gyro_base, mount_quat_wxyz.data());
  sample.linear_acceleration = intoFrame(accel_base, mount_quat_wxyz.data());

  // Synthetic earth field: world -> base (the model knows the true attitude)
  // -> sensor. Uncalibrated hard/soft iron is deliberately absent; that is a
  // separate piece of work (the real magnetometer sees ~57 uT of hard iron
  // from the drives' magnets).
  const auto field_base = intoFrame(earth_field_ut, site_quat);
  sample.magnetometer = intoFrame(field_base, mount_quat_wxyz.data());
  return sample;
}

std::array<double, 7> MujocoPlant::basePose() const
{
  std::array<double, 7> pose{};
  std::copy(data_->qpos, data_->qpos + 7, pose.begin());
  return pose;
}

std::array<double, 6> MujocoPlant::baseVelocity() const
{
  std::array<double, 6> twist{};
  std::copy(data_->qvel, data_->qvel + 6, twist.begin());
  return twist;
}

std::vector<double> MujocoPlant::qpos() const
{
  return std::vector<double>(data_->qpos, data_->qpos + model_->nq);
}

}  // namespace wojtek_mujoco_hardware_interface
