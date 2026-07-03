# four_bar_bot RL locomotion — design

Date: 2026-07-03
Status: approved for planning

## Goal

Train a walking policy for the four_bar_bot quadruped in simulation. The policy tracks a commanded
velocity (forward, sideways, turn rate), the same interface as the existing `/cmd_vel` stack. The
policy must be deployable to the real robot later without retraining, so every simulation choice
matches the real hardware where the hardware is known.

Deployment itself is a separate later project. This spec covers simulation, training, and video
evaluation only.

## Context

- `quadruped_ros2_original/four_bar_bot_description/mujoco/` holds a working MuJoCo model of the
  robot. It has 12 torque actuators, passive linkage joints, and 4 equality constraints that close
  the four-bar loops. It has no standing pose and no base inertial.
- `3_jaxpot_robotics/` holds a working MJX + Brax PPO pipeline. `race_env.py` and `race_train.py`
  show how to train a robot from a custom XML without the playground registry.
- Training runs on a rented vast.ai machine with an RTX 4090. Nothing on that machine survives a
  recycle, so every run must be synced back.
- Nobody has measured the real robot. All masses in the repo are placeholders. The motor config
  caps torque at 1.0 N·m for bench safety; the AK80-9 hardware can deliver about 16 N·m.

## Decision

Build a custom MJX environment from the real robot model, following the `race_env.py` template.
Keep the four-bar loops in the physics. Handle the unknown physical parameters with wide domain
randomization.

Rejected: subclassing the playground Go1 environment (its code assumes Go1's sensors and joint
layout) and flattening the linkage into a serial leg (the dynamics would diverge from the real
robot). The flattened-leg variant remains the documented fallback if MJX proves too slow or
unstable with the equality constraints.

## Components

New top-level directory `4_four_bar_bot_rl/`, a Python project mirroring the jaxpot layout
(`pyproject.toml` with the same pins, including `jax[cuda12]==0.9.2`; `run.sh`; package
`fbb_rl/`). It imports nothing from ROS.

### 1. MJX-ready model

New files `four_bar_bot_mjx.xml` and `scene_mjx.xml` next to the existing model, derived from it.
The existing files stay untouched. Changes:

- Add a base `<inertial>`. Total robot mass becomes 10 kg. The value is a build-time parameter so
  a measured value can replace it later.
- Add a `home` keyframe with a standing pose. The pose is authored by hand and verified by letting
  the model settle under PD hold.
- Replace mesh collision geoms with primitives. Each foot gets a sphere. The base gets a box.
  Intermediate linkage geoms get no collider. The self-collision exclude list shrinks accordingly.
- Replace the 12 torque `motor` actuators with position actuators carrying explicit kp and kd.
  Force range is ±6 N·m per joint. The starting kp/kd are the smallest values that pass the
  standing check below. The final values are recorded in the run config, because deployment must
  copy them into the MD80 impedance mode.
- Keep the 4 `connect` equality constraints, joint ranges, damping, armature, and friction as they
  are.
- Simulation timestep 0.004 s. The policy acts every 0.02 s (50 Hz).

A `check_model_mjx.py` script (extending the existing `check_model.py`) asserts: loop closure
error stays under 2 mm, the robot stands under PD hold for 5 simulated seconds, and MJX steps the
model on the GPU at no worse than one fifth of the Go1 model's rate on the same machine. The
step-rate check runs first on the vast box; if it fails, we stop and reconsider the fallback
before building the rest.

### 2. Environment

`fbb_rl/env.py` defines `FourBarBotJoystick(mjx_env.MjxEnv)`.

- **Action**: 12 values, interpreted as position offsets around the home pose, scaled by a config
  factor.
- **Actor observation** (only signals the real robot has): gyro, gravity direction derived from
  the IMU orientation, 12 joint positions relative to home, 12 joint velocities, previous action,
  and the command. Sensor noise is added in training.
- **Critic observation**: the actor observation plus true base linear velocity and foot contact
  flags. The critic is discarded at deployment.
- **Reward**: the Go1 joystick terms, ported. Tracking rewards for commanded linear and angular
  velocity. Penalties for vertical velocity, body tilt, torque, action rate, and joint-limit
  proximity. A feet-air-time term encourages stepping. An episode ends early when the robot falls.
- **Commands**: resampled during the episode. Initial envelopes: forward ±0.6 m/s, sideways
  ±0.4 m/s, turn ±0.7 rad/s. These are config fields.
- **Episodes**: 20 s, reset to the home pose with small random offsets.

### 3. Domain randomization

Hand-written (the registry randomizers only cover registry robots), applied per environment
instance: ground friction, base mass ±30 %, link masses ±10 %, kp/kd scale ±20 %, motor strength
±20 %, one control step of action latency, and periodic random pushes to the base.

### 4. Training

`fbb_rl/train.py`, modeled on `race_train.py`. It owns its PPO config, seeded from the playground's
tuned Go1 values (150 M steps, 8192 parallel environments). W&B logging, orbax checkpoints, and a
`run.json` per run follow the jaxpot conventions. A CPU smoke mode runs the whole pipeline in
minutes to catch wiring errors before GPU time.

Workflow on the vast box: clone the repo, build the venv with uv, train, then sync `runs/` back to
this machine with scp. W&B holds the metrics as the off-box record.

### 5. Evaluation

`fbb_rl/eval.py`, following the jaxpot eval: load a checkpoint, roll out under a fixed command,
render an mp4, and print the achieved versus commanded velocity.

## Success criteria

- The robot stands for a full episode under zero command.
- It tracks 0.3 m/s forward for 20 s without falling.
- It turns in place on command.
- The video looks like walking, with a plausible gait.

## Out of scope

- Deployment to the real robot (a follow-up project: export the network, run it at 50 Hz in a ROS 2
  node, command the MD80 impedance mode with the training kp/kd, raise the 1.0 N·m config cap).
- Getting up from the ground and fall recovery.
- Rough terrain.

## Risks

- MJX may step the closed-loop model slowly. Mitigation: the step-rate gate in `check_model_mjx.py`
  runs before any environment work; the fallback is the flattened serial-leg model.
- The 10 kg mass guess may be far off. Mitigation: wide mass randomization; replace the guess when
  someone weighs the robot.
- Reward tuning may need several iterations. Mitigation: short smoke runs, W&B comparisons, and the
  Go1 terms as a proven starting point.

## Open questions

- Real total mass. Needs a scale.
- The torque cap for deployment. 6 N·m is the training assumption; the motor config must be raised
  to match before any real-robot run.
- Whether the front legs carry a belt reduction stage that changes the effective joint torque.
