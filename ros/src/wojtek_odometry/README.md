# wojtek_odometry

Leg-kinematics + IMU odometry: the Nav2-facing `odom` source. A stance
foot is pinned to the floor, so each stance leg measures the base
velocity through its Jacobian; the stance-set average is rotated by the
IMU orientation and integrated into a planar pose.

```bash
ros2 run wojtek_odometry leg_odometry_node
ros2 run wojtek_odometry odom_vs_ground_truth   # drift meter, sim only
```

| piece | source |
|---|---|
| chain constants | `robot_description` (latched), parsed at startup |
| four-bar closure | `wojtek_policy.poses.PASSIVE_FROM_KNEE` (no second copy) |
| foot point | MJX `foot_link` offset in `sixth_link` (0.21, 0, 0.0115) |
| contact detection | foot height: within `contact_z_delta` (3 mm) of the lowest IMU-levelled foot, knee torque above a floor. NOT torque thresholds -- measured on the walking policy, knee torque smears 0-6 N*m over both stance and swing |
| rolling correction | the 46 mm foot sphere rolls, so the pinned point is the contact, not the centre; effective radius `rolling_radius` = 0.031 m, fitted against sim ground truth |
| orientation | `/imu_sensor_broadcaster/imu` (ESKF on the robot, ground truth in sim) |

Measured against the simulator's ground truth (wojtek-stiff-height
policy, 2026-08-22): straight 4.5 m -> **1.1 %** drift; 1.4 turns in
place -> 0.16 m position wander, yaw 0.1 deg; 8 m mixed arc -> **2.7 %**
drift, yaw exact. Yaw is the IMU's, so on the real robot expect the
ESKF's yaw drift on top of these numbers.

Outputs `/wojtek/odom` (nav_msgs/Odometry, twist in base frame) and
`/wojtek/odom/debug` (`[4x knee |tau|, 4x stance flag]`, for threshold
tuning). TF `odom->base_link` only with `publish_tf:=true` — off by
default because the sim's ground truth and (on the robot)
launch_common's static identity own that edge today; replacing the
static transform is a deliberate, separate step.

## Testing in simulation

1. `ros2 launch wojtek_pc sim.launch.py` (hw:=mujoco), then the usual
   zero / stand_up / arm sequence.
2. `ros2 run wojtek_odometry leg_odometry_node`
3. `ros2 run wojtek_odometry odom_vs_ground_truth` — prints position/yaw
   error and drift % of distance once a second.
4. Drive from the web console (localhost:8080), a pad, or
   `teleop_twist_keyboard`.

Start the odometry node before driving: it zeroes its yaw at the first
IMU sample and the drift meter assumes both poses share their origin.

## Tests

```bash
cd ros/src/wojtek_odometry && python3 -m pytest test/ -q
```

Covers the FK geometry against the real URDF (stand height, left/right
and front/rear symmetry), Jacobian-vs-FK consistency, and that the knee
column actually carries the four-bar coupling.
