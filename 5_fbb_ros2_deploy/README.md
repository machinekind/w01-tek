# 5_fbb_ros2_deploy — RL policy in ROS 2 (sim + RViz + real robot)

A clean ROS 2 (Humble) deployment of the `fbb_v3` locomotion policy trained in
`4_four_bar_bot_rl`: a MuJoCo simulator node running the exact training
physics, RViz visualization through the existing URDF, and the same policy
node wired to the real robot (MD80 motors + BMX160 IMU) through ros2_control.

`ros2_ws` is a plain colcon workspace — build it natively on any machine with
ROS 2 Humble. A Docker fallback (`run.sh docker-*`) exists for hosts without
ROS.

## Quick start (simulation + RViz, native ROS 2 Humble)

```bash
./run.sh deps       # one-time: apt packages + pip mujoco
./run.sh build      # colcon build (fbb_policy + four_bar_bot_description)
./run.sh sim        # MuJoCo sim + policy + RViz
./run.sh teleop     # in a second terminal: drive with the keyboard
```

(No ROS on this machine? `./run.sh docker-build && ./run.sh docker-sim`.)

The robot stands in RViz and walks when you send velocity commands
(`teleop_twist_keyboard` publishes `/cmd_vel`). Useful knobs:

```bash
ros2 service call /sim/reset std_srvs/srv/Trigger        # respawn at home pose
ros2 service call /fbb/reset std_srvs/srv/Trigger        # reset policy state
ros2 service call /fbb/enable std_srvs/srv/SetBool '{data: false}'  # pause policy
ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.4}}'
```

## How it fits together

```
                sim                                   real robot
  ┌────────────────────────────┐        ┌─────────────────────────────────┐
  │ mujoco_sim_node            │        │ ros2_control (MD80 IMPEDANCE,   │
  │  scene_mjx.xml @ 250 Hz    │        │  kp=20 kd=1 ±6Nm + BMX160 IMU)  │
  │  /joint_states /imu/data   │        │  joint_state & imu broadcasters │
  │  TF odom→base_link         │        │  forward_position_controller    │
  └─────▲──────────────┬───────┘        └────▲──────────────┬─────────────┘
        │ targets      │ sensors             │ commands     │ states (boot-rel)
        │              ▼                     │              ▼
  ┌─────┴──────────────────────┐        ┌────┴──────────────────────┐
  │ policy_node (50 Hz, numpy) │◄──────►│ real_io_node              │
  │  /cmd_vel → joint targets  │targets │  abs↔boot-relative shift, │
  └────────────────────────────┘ states │  arming gate, dry-run     │
                                        └───────────────────────────┘
```

* **`fbb_policy/policy.py`** — pure-numpy runtime of the exported policy
  (obs assembly, gait clock, MLP, action → position targets). No JAX at
  deploy time; validated against the Brax inference fn to 2e-6.
* **`config/policy.npz` + `policy_meta.json`** — exported by
  `4_four_bar_bot_rl/fbb_rl/export_policy.py` (`./run.sh export-policy` to
  refresh after retraining).
* **`config/joint_map.yaml`** — the URDF and the MuJoCo model use different
  joint-zero conventions (gear-correction offsets, mirrored axes). This
  per-joint `q_urdf = sign * q_mujoco + offset` table is fitted numerically
  by `tools/fit_joint_map.py` (`./run.sh fit-map`), validated to <0.1 µm of
  link-position agreement. `/joint_states` and `/fbb/joint_targets` always
  carry URDF-convention angles; only the policy internals use MuJoCo
  convention.
* **`mujoco_sim_node`** — steps the training scene (position servos kp=20
  kd=1, ±6 Nm, dt=4 ms) in real time and bridges joints/IMU/TF to ROS.
* **`policy_node`** — identical in sim and on the robot. Watchdog on stale
  sensors, soft-start blend, optional knee-singularity clamp, training-range
  clipping of `/cmd_vel`.

Verified end-to-end: the numpy policy through the full URDF↔MuJoCo
conversion pipeline tracks +0.40 m/s at +0.41 m/s measured over 10 s without
falling (see `tools/`, test `run.sh test`).

## Real robot

Two extra packages are needed (they live in `quadruped_ros2_original`);
`real-build` pulls the candle submodule and builds them into the same
workspace:

```bash
./run.sh real-build
./run.sh real        # = ros2 launch fbb_policy real.launch.py [args]
```

`real.launch.py` starts ros2_control with `urdf/fbb_real.urdf.xacro`:

* MD80 joints get the **`position` command interface → IMPEDANCE mode** with
  gains matching the trained actuator model (kp=20, kd=1, max_torque=6 N·m).
  For first tests cap the torque: `max_torque:=2.0`.
* The BMX160 IMU ros2_control sensor is enabled (it is commented out in the
  original description package) and published by `imu_sensor_broadcaster`
  in the `imu_link` frame; `policy_node` rotates gyro/gravity into
  `base_link` (`imu_mount_rpy = [0, π, 0]` from the URDF).

### Startup checklist (safety)

1. **Pose the robot in the home standing stance, then power the motors.**
   The MD80 driver zeroes encoders at activation; `real_io_node` assumes
   boot pose == home pose (`policy_meta.json: home_ctrl`) to convert between
   absolute angles and boot-relative hardware positions.
2. Launch with `dry_run:=true` first. Check in RViz that the displayed pose
   matches the physical robot as you move legs by hand (this validates joint
   directions and the boot-pose assumption).
3. `real_io_node` starts **DISARMED** — the policy runs but no commands reach
   the motors. Arm with
   `ros2 service call /fbb/arm std_srvs/srv/SetBool '{data: true}'`.
   Arming is refused if any joint is >0.15 rad from the boot pose.
4. Start with `max_torque:=2.0`, robot on a stand, and keep `/cmd_vel` at
   zero (the policy balances in place) before commanding velocities.

### Must-verify on hardware (cannot be checked in sim)

* `md80_hardware_interface` subtracts the boot offset from position *states*
  but writes IMPEDANCE position *commands* raw
  (`md80_hardware_interface.cpp: write()`). `real_io_node` assumes raw ==
  boot-relative (true when the drives zero at power-on). Verify with
  `dry_run` before the first powered test.
* Per-joint rotation direction of the motors vs the URDF (the original
  `quadruped_controller` carries its own direction table). Step 2 above
  catches any mismatch.
* BMX160 orientation-fusion quality. If it is poor, set the policy_node
  parameter `gravity_from_accel: true` to use the built-in
  gyro+accelerometer complementary filter instead.

## Layout

```
5_fbb_ros2_deploy/
├── run.sh                  deps / build / sim / teleop / real / test entry points
├── docker/                 optional fallback for hosts without ROS (+ ci_check.sh)
├── tools/fit_joint_map.py  regenerates config/joint_map.yaml
└── ros2_ws/src/fbb_policy/
    ├── fbb_policy/         policy.py, joint_map.py, 3 nodes
    ├── launch/             sim.launch.py, real.launch.py
    ├── config/             policy.npz, policy_meta.json, joint_map.yaml,
    │                       MJX scene XMLs, real_controllers.yaml
    ├── urdf/               fbb_real.urdf.xacro (+ ros2_control impedance)
    ├── rviz/fbb.rviz
    └── test/               pure-numpy unit tests (run.sh test)
```

The mechanical description (URDF, meshes) is reused from
`quadruped_ros2_original/four_bar_bot_description` — nothing there is
modified.
