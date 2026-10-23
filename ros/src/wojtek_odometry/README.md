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

**The contact thresholds are the stiff gait's, and the robot's pinned
policy is not that gait.** Measured 2026-09-25 in the same simulator with
`wojtek-quiet-locomotion` (kp=20, the tau_ff head; the default a bringup
runs): on a 6 s straight walk the odometry reports **0.55** of the true
distance. The debug stream shows why: in 30 % of the samples no foot
passes as stance (knee |tau| median 1.2 N*m against the 0.8 floor, and
the soft servo unloads the knee more often), so the velocity reads zero
between steps. Lowering `contact_tau_floor` to 0.3 brings zero-stance
samples to 2 % but the ratio only to **0.64**: the quiet gait lifts its
feet little, so swing feet stay within `contact_z_delta` (3 mm) of the
lowest one and get averaged in as stance, diluting the estimate. The
thresholds need fitting per gait (ground truth in the sim, as before),
or a contact criterion that does not depend on the servo's stiffness.
Two more things the same session showed: a foot pushing against an
obstacle integrates as travel (15 s against a crate read as 1.5 m), and
every ~160 deg turn in place added ~0.1 m of position error on the stiff
gait while straight walking stayed under 2 %.

Outputs `/wojtek/odom` (nav_msgs/Odometry, twist in base frame) and
`/wojtek/odom/debug` (`[4x knee |tau|, 4x stance flag]`, for threshold
tuning). TF `odom->base_link` with `publish_tf:=true`: the bringup's
`leg_odom` switch (on by default on the robot, off in the sim, where the
ground truth owns the edge unless a run asks otherwise).

## Testing in simulation

Two ways to have the node up:

- `ros2 launch wojtek_pc sim.launch.py leg_odom:=true` -- the bringup runs
  it with the robot's parameters, it owns `odom->base_link`, and the
  plant's ground truth moves to `odom->base_link_gt`. This is the
  configuration everything downstream builds on. Point the drift meters
  at the truth: `--ros-args -p ground_truth_frame:=base_link_gt`.
- `ros2 launch wojtek_pc sim.launch.py` and `ros2 run wojtek_odometry
  leg_odometry_node` by hand -- TF stays the ground truth, the node only
  publishes `/wojtek/odom`, the meters' default `base_link` is right.

Then, either way: the usual zero / stand_up / arm sequence,
`ros2 run wojtek_odometry odom_vs_ground_truth` (position/yaw error and
drift % of distance once a second; `odom_trace` for the RViz trails), and
drive from the web console (localhost:8080), a pad, or
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
